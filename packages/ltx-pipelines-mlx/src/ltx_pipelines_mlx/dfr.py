"""DFR ("Diffusion Fidelity Rendering") base path — upstream ``ltx_pipelines/dfr_pipeline.py``.

Stage 1 (half resolution, distilled, ancestral on 2.5) runs on a canvas padded to whole keyframe
segments with one generated keyframe slot per segment boundary. Stage 2 (full resolution,
deterministic) runs with the detailing IC-LoRA attached at strength 0.5, conditioned on the
stage-1 latent as an IC-LoRA reference and on the spatially upsampled stage-1 slots. This port
covers ``spatial_upscalings`` 1 or 2 with ``temporal_upscalings`` 0, 1 or 2. Each temporal round
upsamples the latent x2 in time with the temporal upsampler, then re-denoises it in keyframe-seam
tiles (carried keyframes as anchors, fresh slots between them, frozen stage-1 audio, 4-step
ancestral Euler) with the distilled transformer without the detailing LoRA; the output plays at
``frame_rate * 2**temporal_upscalings``. The final keyframe bag (stage-2 slots, or the carry bag
after the rounds) is handed to the decoder as decoder keyframes (keyframe-aware decode on
``--video-decoder diffusion``; the conv decoder ignores them with a warning).

With ``spatial_upscalings=2`` the output is floored to multiples of 128 px, stage 1 runs at H/4 and
stage 2 plus the rounds at H/2, then a spatial epilogue details the full resolution: the carry
keyframes are decoded one plane at a time with the render's decoder, Lanczos-upsampled x2 and
re-encoded as strength-1.0 keyframes; the H/2 latent is spatially upsampled and re-denoised
(3-step deterministic Euler, detailing LoRA, H/2 latent as the IC-LoRA reference, frozen stage-1
audio) with every model call tiled 2x2 spatially and in ``2**T`` frame tiles cut on the last
round's seams. The re-encoded planes become the decoder keyframes.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub.errors import GatedRepoError
from PIL import Image

import ltx_pipelines_mlx.distilled as distilled_mod
from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots, extract_generated_keyframes
from ltx_core_mlx.loader import (
    LTXV_LORA_BLOCK_PREFIX,
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraStateDictWithStrength,
    SafetensorsStateDictLoader,
    StateDict,
    apply_loras,
)
from ltx_core_mlx.loader.block_streaming import BlockLoraSource
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import DecodeKeyframes
from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import apply_quantization
from ltx_pipelines_mlx.dfr_layout import TemporalTilePlan, pixel_to_latent_index, resolve_canvas
from ltx_pipelines_mlx.distilled import DistilledPipeline, Stage1Result
from ltx_pipelines_mlx.iclora_utils import (
    read_lora_reference_downscale_factor,
    reference_conditioning_from_latent,
)
from ltx_pipelines_mlx.scheduler import LTX_2_5_DISTILLED_SIGMAS, LTX_2_5_STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings, resolve_lora_path
from ltx_pipelines_mlx.utils.args import ImageConditioningInput
from ltx_pipelines_mlx.utils.progress import phase
from ltx_pipelines_mlx.utils.types import DEFAULT_AUTO_DURATION, AutoDuration

logger = logging.getLogger(__name__)

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern

#: Official LTX-2.5 detailing IC-LoRA (creative x2 spatial upsampler), ``reference_downscale_factor: 2``.
DEFAULT_DETAILING_LORA = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
#: Upstream ``_DETAILING_LORA_STRENGTH``; not a user knob.
DETAILING_LORA_STRENGTH = 0.5
#: Pack file stem of the temporal x2 latent upsampler (LTX-2.5 packs).
TEMPORAL_UPSAMPLER_STEM = "temporal_upscaler_x2_v1_0"
#: Pixel frames per latent frame of the video VAE (there is no pipeline-level constant).
_TEMPORAL_SCALE = 8
#: ``--spatial-upscalings 2`` runs stage 1 at H/4: the output must divide by 4 * 32 px (upstream
#: ``assert_resolution(divisor=128)``; floored with a warning here, per the repo dims policy).
_EPILOGUE_RESOLUTION_MULTIPLE = 128

#: Upstream ``_ANCHOR_KEYFRAME_STRENGTH``: carried keyframes pinned just short of clean.
ANCHOR_KEYFRAME_STRENGTH = 0.95
#: Upstream ``_TEMPORAL_ANCESTRAL_ETA``.
TEMPORAL_ANCESTRAL_ETA = 0.5
#: Upstream ``DISTILLED_SIGMAS[4:]``: the 4-step temporal-round schedule.
TEMPORAL_SIGMAS: list[float] = list(LTX_2_5_DISTILLED_SIGMAS[4:])
_MAX_CONDITIONING_FPS = 60.0
_SNAP_CONDITIONING_FPS_ABOVE = 30.0

#: Upstream ``_EPILOGUE_SPATIAL_OVERLAP``: 2x2 spatial tiles inside the epilogue overlap by this
#: many token-grid cells (clamped per-axis by :func:`clamp_tile_counts` on a small latent).
EPILOGUE_SPATIAL_OVERLAP = 12
#: Upstream ``_EPILOGUE_KEYFRAME_STRENGTH``: the epilogue's re-encoded keyframe planes are a hard
#: anchor, not a soft one like the temporal rounds' carried keyframes.
EPILOGUE_KEYFRAME_STRENGTH = 1.0
#: Seed offset for decoding each carry plane before Lanczos-upsampling it into the epilogue's
#: keyframe conditionings (mirrors ``KEYFRAME_PLANE_DECODE_SEED_OFFSET = seed + 4000 + i``).
KEYFRAME_PLANE_DECODE_SEED_OFFSET = 4000
#: Seed offset for the epilogue's own noise draw (``seed + 2000``), decorrelated from stage 1/2
#: and the temporal rounds' per-tile noise (``seed + 1000 * round + tile``).
EPILOGUE_NOISE_SEED_OFFSET = 2000


def conditioning_fps(playback_fps: float) -> float:
    """Transformer RoPE fps (upstream ``_conditioning_fps``): above 30 snaps to 60; playback fps is unchanged."""
    return _MAX_CONDITIONING_FPS if playback_fps > _SNAP_CONDITIONING_FPS_ABOVE else playback_fps


def lanczos_x2(frames: np.ndarray) -> np.ndarray:
    """Stretch each frame 2x with Lanczos resampling (upstream ``_lanczos_x2_fhwc``).

    Args:
        frames: ``(F, H, W, C)`` array in ``[0, 1]``, ``C`` in ``{1, 3}``.

    Returns:
        ``(F, 2H, 2W, C)`` float32 array in ``[0, 1]``.

    Raises:
        ValueError: ``frames`` is not 4-D, has no frames, or an unsupported channel count.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected (F, H, W, C), got shape {tuple(frames.shape)}")
    if frames.shape[0] < 1:
        raise ValueError("Need at least one frame to Lanczos-upsample")
    channels = frames.shape[-1]
    if channels not in (1, 3):
        raise ValueError(f"Lanczos x2 expects 1 or 3 channels, got {channels}")
    out: list[np.ndarray] = []
    for frame in frames:
        height, width, _channels = frame.shape
        array = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        image = Image.fromarray(array[..., 0], mode="L") if channels == 1 else Image.fromarray(array, mode="RGB")
        image = image.resize((width * 2, height * 2), resample=Image.Resampling.LANCZOS)
        resized = np.asarray(image, dtype=np.float32) / 255.0
        if resized.ndim == 2:
            resized = resized[..., None]
        out.append(resized)
    return np.stack(out, axis=0)


def _clamp_dim_tiling(cfg: DimensionTilingConfig, dim_size: int, axis: str) -> DimensionTilingConfig:
    """Clamp a single dimension's tile count and overlap to the latent's extent.

    Mirrors upstream ``_clamp_dim_tiling``. ``split_by_count`` requires ``overlap < tile_size``;
    with ``tile_size = (dim_size + overlap * (n - 1)) // n`` this reduces to
    ``overlap <= dim_size - n``. When the configured overlap exceeds this bound it is clamped;
    if the latent is too small to hold ``n`` tiles at all, tiling falls back to a single tile.
    """
    n = cfg.num_tiles
    if n <= 1:
        return cfg
    if dim_size < n:
        logger.warning("%s tiling: dim_size=%d < num_tiles=%d; falling back to 1 tile on this axis.", axis, dim_size, n)
        return DimensionTilingConfig(1, 0)
    max_overlap = dim_size - n
    if cfg.overlap <= max_overlap:
        return cfg
    logger.warning(
        "%s tiling: overlap=%d exceeds latent bound (%d); clamping to %d.", axis, cfg.overlap, max_overlap, max_overlap
    )
    return DimensionTilingConfig(n, max_overlap)


def clamp_tile_counts(tiling: TileCountConfig, latent_fhw: tuple[int, int, int]) -> TileCountConfig:
    """Clamp frame, height, and width tilings to the latent's extents (upstream ``_clamp_tile_to_latent``).

    Args:
        tiling: Requested tile counts.
        latent_fhw: ``(F, H, W)`` latent extents in token-grid units.

    Returns:
        A :class:`TileCountConfig` with each axis clamped by :func:`_clamp_dim_tiling`.
    """
    frames, height, width = latent_fhw
    return replace(
        tiling,
        frames=_clamp_dim_tiling(tiling.frames, frames, "Frame"),
        height=_clamp_dim_tiling(tiling.height, height, "Height"),
        width=_clamp_dim_tiling(tiling.width, width, "Width"),
    )


def floor_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` down to the nearest multiple of ``multiple``."""
    return (value // multiple) * multiple


def resample_audio_time(audio_latent: mx.array, src_start: float, src_end: float, out_frames: int) -> mx.array:
    """Linearly sample a ``(B, C, T, F)`` audio latent along T over ``[src_start, src_end)`` cells.

    Raises:
        ValueError: ``out_frames < 1``, empty latent, or empty window.
    """
    if out_frames < 1:
        raise ValueError(f"out_frames must be >= 1, got {out_frames}")
    full_t = audio_latent.shape[2]
    if full_t < 1:
        raise ValueError("Cannot resample an empty audio latent")
    span = src_end - src_start
    if span <= 0:
        raise ValueError(f"Audio window is empty: [{src_start}, {src_end})")
    positions = mx.clip(src_start + (span / out_frames) * mx.arange(out_frames, dtype=mx.float32), 0, full_t - 1)
    lo = mx.floor(positions).astype(mx.int32)
    hi = mx.minimum(lo + 1, full_t - 1)
    weight = (positions - lo.astype(mx.float32)).astype(audio_latent.dtype).reshape(1, 1, -1, 1)
    return audio_latent[:, :, lo] * (1 - weight) + audio_latent[:, :, hi] * weight


def audio_latent_for_tile(
    audio_latent: mx.array,
    *,
    pixel_start: int,
    local_frames: int,
    playback_fps: float,
    source_duration: float,
    cond_fps: float,
) -> mx.array:
    """Stage-1 audio for one temporal tile (upstream ``_audio_latent_for_tile``).

    The window is wall-clock ``[pixel_start, pixel_start + local_frames) / playback_fps`` as a fraction of
    ``source_duration`` (stage 1's ``canvas_frames / frame_rate``); the output length is the audio token
    count of ``local_frames`` at ``cond_fps``.

    Raises:
        ValueError: non-positive ``local_frames`` / ``playback_fps`` / ``source_duration`` or empty audio.
    """
    if local_frames <= 0:
        raise ValueError(f"local_frames must be >= 1, got {local_frames}")
    if playback_fps <= 0:
        raise ValueError(f"playback_fps must be > 0, got {playback_fps}")
    if source_duration <= 0:
        raise ValueError(f"source_duration must be > 0, got {source_duration}")
    full_t = audio_latent.shape[2]
    if full_t <= 0:
        raise ValueError("Cannot slice audio for a tile with an empty audio latent")
    src_start = pixel_start / playback_fps / source_duration * full_t
    src_end = (pixel_start + local_frames) / playback_fps / source_duration * full_t
    return resample_audio_time(
        audio_latent, src_start, src_end, compute_audio_token_count(local_frames, frame_rate=cond_fps)
    )


def slot_initials_from_video(video_latent: mx.array, positions: Sequence[int]) -> mx.array:
    """Nearest latent frames ``round(position / 8)`` (clamped) as ``(B, C, K, H, W)`` slot seeds."""
    last = video_latent.shape[2] - 1
    frames = [min(max(round(int(p) / _TEMPORAL_SCALE), 0), last) for p in positions]
    return mx.concatenate([video_latent[:, :, i : i + 1] for i in frames], axis=2)


def dedupe_slots(positions: Sequence[int], latents: mx.array) -> tuple[list[int], mx.array]:
    """Sorted unique slot positions; lead-in duplicates keep the earlier tile's latent."""
    first: dict[int, int] = {}
    for index, position in enumerate(positions):
        first.setdefault(int(position), index)
    ordered = sorted(first)
    return ordered, mx.concatenate([latents[:, :, first[p] : first[p] + 1] for p in ordered], axis=2)


def merge_carry_forward_keyframes(
    anchor_positions: Sequence[int],
    anchor_latents: mx.array | None,
    slot_positions: Sequence[int],
    slot_latents: mx.array | None,
) -> tuple[list[int], mx.array]:
    """Next round's keyframe bag: carried anchors plus this round's slots, sorted by position.

    Mirrors upstream ``_merge_carry_forward_keyframes`` (a slot at an anchor's position replaces it).

    Raises:
        RuntimeError: missing latents for non-empty positions, or an empty bag.
        ValueError: latent count != position count.
    """
    by_position: dict[int, mx.array] = {}
    for positions, latents, label in (
        (anchor_positions, anchor_latents, "anchor"),
        (slot_positions, slot_latents, "slot"),
    ):
        if not positions:
            continue
        if latents is None:
            raise RuntimeError(f"Missing {label} keyframe latents for carry-forward merge")
        if latents.shape[2] != len(positions):
            raise ValueError(f"{label} latents K={latents.shape[2]} != {len(positions)} positions")
        for index, position in enumerate(positions):
            by_position[int(position)] = latents[:, :, index : index + 1]
    if not by_position:
        raise RuntimeError("Carry-forward keyframe bag is empty")
    ordered = sorted(by_position)
    return ordered, mx.concatenate([by_position[p] for p in ordered], axis=2)


def rebase_image_conditionings(
    images: Sequence[ImageConditioningInput], *, pixel_scale: int, pixel_start: int = 0, pixel_end: int | None = None
) -> list[ImageConditioningInput]:
    """Map ``ImageConditioningInput.frame_idx`` onto a temporally upsampled grid (upstream ``_rebase_image_conditionings``).

    After ``r`` rounds a moment sits at ``frame_idx * 2**r``; with ``pixel_end`` only images inside
    ``[pixel_start, pixel_end]`` are kept and their index becomes tile-local.
    """
    rebased = []
    for image in images:
        scaled = image.frame_idx * pixel_scale
        if pixel_end is not None and not (pixel_start <= scaled <= pixel_end):
            continue
        rebased.append(image._replace(frame_idx=scaled - pixel_start))
    return rebased


class DFRPipeline(DistilledPipeline):
    """DFR base path on top of the distilled two-stage pipeline (see the module docstring).

    Args:
        model_dir: Path to model weights or HuggingFace repo ID (an LTX-2.5 pack: the
            keyframe slots require ``use_keyframes_abs_pos_embedding``).
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        low_ram_streaming: Stream transformer blocks from disk.
        tile_count: Optional modality tiling configuration.
        detailing_lora: Local ``.safetensors`` path or HF repo id of the detailing IC-LoRA.
        temporal_upscalings: Number of temporal-round passes (0, 1, or 2). 0 disables temporal
            rounds (current base-path behaviour).
        temporal_upsampler_path: Explicit path to the temporal x2 latent upsampler weights.
            ``None`` resolves it from the pack (see :meth:`_resolve_temporal_upsampler_path`).
        spatial_upscalings: 1 (stage 2 at full resolution) or 2 (stage 1 at H/4, stage 2 and the
            temporal rounds at H/2, then the tiled full-resolution spatial epilogue).
    """

    #: Class default so a ``__new__``-built instance (tests) decodes at the base frame rate.
    temporal_upscalings: int = 0
    #: Class default so a ``__new__``-built instance (tests) runs without the spatial epilogue.
    spatial_upscalings: int = 1

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        tile_count=None,
        detailing_lora: str = DEFAULT_DETAILING_LORA,
        temporal_upscalings: int = 0,
        temporal_upsampler_path: str | None = None,
        spatial_upscalings: int = 1,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            tile_count=tile_count,
        )
        if temporal_upscalings not in (0, 1, 2):
            raise ValueError(f"temporal_upscalings must be 0, 1 or 2, got {temporal_upscalings}")
        if spatial_upscalings not in (1, 2):
            raise ValueError(f"spatial_upscalings must be 1 or 2, got {spatial_upscalings}")
        self.detailing_lora = detailing_lora
        self._detailing_lora_path: str | None = None
        self._detailing_downscale: int | None = None
        self._detailing_source: BlockLoraSource | None = None
        self.canvas_frames: int = 0
        self.generated_keyframe_positions: list[int] = []
        self.temporal_upscalings = temporal_upscalings
        self.temporal_upsampler_path = temporal_upsampler_path
        self.spatial_upscalings = spatial_upscalings

    # ---- detailing LoRA -----------------------------------------------------------
    def _resolve_detailing_lora(self) -> str:
        """Resolve (download if needed) the detailing LoRA once and read its downscale factor.

        Returns:
            Local path to the detailing LoRA ``.safetensors`` file.
        """
        if self._detailing_lora_path is None:
            try:
                self._detailing_lora_path = resolve_lora_path(self.detailing_lora)
            except GatedRepoError as exc:
                raise PermissionError(
                    f"The detailing IC-LoRA '{self.detailing_lora}' is a gated HuggingFace repo: accept its "
                    f"licence once at https://huggingface.co/{self.detailing_lora} with the account you are "
                    "logged in as (huggingface-cli login), then rerun. No model was loaded."
                ) from exc
            self._detailing_downscale = read_lora_reference_downscale_factor(self._detailing_lora_path)
        return self._detailing_lora_path

    def _attach_detailing_lora(self) -> None:
        """Attach the detailing LoRA at ``DETAILING_LORA_STRENGTH`` to the resident distilled DiT.

        Streaming (``--low-ram``): append a :class:`BlockLoraSource` (fused at each block bind,
        exactly like ``ICLoraPipeline._fuse_loras``). Otherwise fuse in place and re-quantize:
        stage 1 is finished and the model is reused clean only by the temporal rounds, which
        reload it (see :meth:`_detach_detailing_lora`).
        """
        assert self.dit is not None
        path = self._resolve_detailing_lora()
        with phase("Attaching the detailing IC-LoRA (strength 0.5)", verbose=self.verbose):
            if self.low_ram_streaming:
                sources: list = list(object.__getattribute__(self.dit, "_lora_sources"))
                source = BlockLoraSource(
                    path,
                    block_prefix=LTXV_LORA_BLOCK_PREFIX,
                    strength=DETAILING_LORA_STRENGTH,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
                sources.append(source)
                object.__setattr__(self.dit, "_lora_sources", sources)
                self._detailing_source = source
                logger.info("Attached detailing LoRA streamer: %s (strength=%s)", path, DETAILING_LORA_STRENGTH)
                return

            import mlx.utils

            model_sd = StateDict(sd=dict(mlx.utils.tree_flatten(self.dit.parameters())), size=0, dtype=set())
            lora_sd = SafetensorsStateDictLoader().load(path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            fused = apply_loras(
                model_sd=model_sd,
                lora_sd_and_strengths=[LoraStateDictWithStrength(state_dict=lora_sd, strength=DETAILING_LORA_STRENGTH)],
            )
            apply_quantization(self.dit, fused.sd)
            self.dit.load_weights(list(fused.sd.items()))
            # The fuse happens mid-pipeline with the stage-1 latents resident, so the lazy
            # dequantize -> fuse -> requantize graph must not be deferred to the first stage-2
            # forward: materialize the fused weights here, then drop the pre-fuse state dicts.
            _materialize(self.dit.parameters())
            del model_sd, lora_sd, fused
            aggressive_cleanup()
            logger.info("Fused detailing LoRA: %s (strength=%s)", path, DETAILING_LORA_STRENGTH)

    def _detach_detailing_lora(self) -> None:
        """Give the temporal rounds the distilled transformer without the detailing IC-LoRA (upstream ``self.stage``).

        Streaming: drop the detailing :class:`BlockLoraSource` (user LoRAs stay). Otherwise the LoRA is fused
        into the weights, so free the fused DiT and reload a clean one; pending user LoRAs are re-fused by
        :meth:`_load_transformer_with_optional_streaming`.
        """
        if self.low_ram_streaming:
            sources = [s for s in object.__getattribute__(self.dit, "_lora_sources") if s is not self._detailing_source]
            object.__setattr__(self.dit, "_lora_sources", sources)
            self._detailing_source = None
            return
        self.dit = None
        aggressive_cleanup()
        transformer_path = self.model_dir / "transformer.safetensors"
        if not transformer_path.exists():
            transformer_path = self._resolve_safetensors(self.model_dir, "transformer-distilled")
        self.dit = self._load_transformer_with_optional_streaming(transformer_path)

    # ---- temporal upsampler -----------------------------------------------------------
    def _resolve_temporal_upsampler_path(self) -> Path:
        """The temporal x2 upsampler: ``temporal_upsampler_path`` when set, else the pack's file.

        Raises:
            FileNotFoundError: neither exists (a random temporal upsampler would silently wreck the video).
        """
        if self.temporal_upsampler_path is not None:
            path = Path(self.temporal_upsampler_path)
            if not path.exists():
                raise FileNotFoundError(f"--temporal-upsampler-path {path} does not exist")
            return path
        path = self._resolve_safetensors(self.model_dir, TEMPORAL_UPSAMPLER_STEM)
        if not path.exists():
            raise FileNotFoundError(
                f"Temporal upsampler weights not found in {self.model_dir} (looked for "
                f"{TEMPORAL_UPSAMPLER_STEM}*.safetensors); --temporal-upscalings needs it. Pass "
                "--temporal-upsampler-path or download it into the pack."
            )
        return path

    def _load_temporal_upsampler(self) -> LatentUpsampler:
        """Build and load the temporal x2 latent upsampler.

        Raises:
            ValueError: the resolved weights build a spatial (or otherwise non-temporal)
                upsampler — either because the ``<stem>_config.json`` alongside the weights
                is missing (``_build_upsampler`` then falls back to a default *spatial*
                ``LatentUpsampler()``) or because it names a spatial variant. A spatial
                upsampler silently wrecks the video exactly like the untrained-module case
                :meth:`TI2VidTwoStagesPipeline._resolve_upsampler_path` guards against.
        """
        with phase("Loading the temporal upsampler", verbose=self.verbose):
            path = self._resolve_temporal_upsampler_path()
            upsampler = self._build_upsampler(path)
            if not upsampler.temporal_upsample:
                raise ValueError(
                    f"{path} does not build a temporal upsampler (LatentUpsampler.temporal_upsample is False). "
                    "--temporal-upscalings needs a temporal x2 upsampler (pack file "
                    f"'{TEMPORAL_UPSAMPLER_STEM}.safetensors') with its matching "
                    f"'{TEMPORAL_UPSAMPLER_STEM}_config.json' alongside it; a spatial upsampler "
                    "(or weights loaded without their config) will silently produce garbage output."
                )
            return upsampler

    # ---- generation -----------------------------------------------------------------
    def generate_two_stage(  # type: ignore[override]
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        image: str | None = None,
        images=None,
        prompt_relay=None,
        generated_keyframes: int | Sequence[int] = 0,
        enable_teacache: bool = False,
        **_unused_kwargs,
    ) -> tuple[mx.array, mx.array]:
        """DFR base path; returns ``(video_latent, audio_latent)`` trimmed to ``num_frames``.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames, or an :class:`AutoDuration` request.
            frame_rate: Video frame rate.
            seed: Random seed.
            stage1_steps: Stage 1 steps (default: full stage-1 sigma table).
            stage2_steps: Stage 2 steps (default: full stage-2 sigma table).
            image: Optional reference image for I2V conditioning.
            images: Optional multi-anchor I2V conditioning inputs.
            prompt_relay: Optional Prompt Relay segment specs.
            generated_keyframes: Must stay falsy — DFR places its own slots.
            enable_teacache: Must stay ``False`` — not available on the DFR path.
            **_unused_kwargs: Accepted (and ignored) for signature compatibility with
                :meth:`DistilledPipeline.generate_two_stage`.

        Returns:
            Tuple of (video_latent, audio_latent), trimmed back to the requested duration.

        Raises:
            ValueError: on a pack without the keyframe embedding, when ``generated_keyframes``
                is passed (DFR places its own slots from the canvas) or when TeaCache is requested.
            FileNotFoundError: when the detailing LoRA cannot be resolved (raised up front,
                before any prompt encoding).
            RuntimeError: when ``_stage1`` did not run the canvas hook, leaving the requested
                duration unknown.
        """
        if generated_keyframes:
            raise ValueError("DFR places its keyframe slots from the canvas; --num-generated-keyframes does not apply")
        if enable_teacache:
            raise ValueError("TeaCache is not available on the DFR path (distilled flow)")
        self._require_generated_keyframes_support([1])  # DFR always uses slots: refuse 2.3 packs up front
        self._require_num_frames_source(num_frames)
        if self.temporal_upscalings:
            if prompt_relay is not None:
                raise ValueError("Prompt Relay (--segment) is not supported with DFR temporal rounds")
            if self._tile_count is not None:
                raise ValueError("DFR temporal rounds do not support modality tiling (--tile-*)")
            self._resolve_temporal_upsampler_path()  # fail before any Gemma load
        if self.spatial_upscalings == 2:
            if prompt_relay is not None:
                raise ValueError("Prompt Relay (--segment) is not supported with the DFR spatial epilogue")
            if self._tile_count is not None:
                raise ValueError(
                    "The DFR spatial epilogue tiles its own model calls; modality tiling (--tile-*) is refused"
                )
            height, width = self._floor_epilogue_resolution(height, width)
        # Resolve (and download) the detailing LoRA before any text encoding: it is required
        # by stage 2, so a bad path must fail here rather than after a full stage-1 render.
        self._resolve_detailing_lora()

        # Stage 1 on the padded canvas. AutoDuration is resolved inside _stage1; the canvas is
        # derived from the resolved value through the `canvas_for` hook below.
        requested = 0

        def canvas_for(resolved_frames: int) -> tuple[int, list[int]]:
            """Pad the resolved clip length to whole keyframe segments (``_stage1`` hook)."""
            nonlocal requested
            canvas_frames, segment, positions = resolve_canvas(resolved_frames)
            requested = resolved_frames
            if self.verbose:
                print(
                    f"[dfr] canvas {canvas_frames} frames (requested {resolved_frames}), segment {segment}, "
                    f"{len(positions)} keyframe slots at {positions}",
                    file=sys.stderr,
                    flush=True,
                )
            self.canvas_frames = canvas_frames
            self.generated_keyframe_positions = positions
            return canvas_frames, positions

        # With the epilogue, stages 1/2 run the base two-stage flow at H/2 (stage 1 at H/4).
        stage_div = 2 if self.spatial_upscalings == 2 else 1
        stage1, canvas_frames, _stage_height, _stage_width = self._stage1(
            prompt,
            height // stage_div,
            width // stage_div,
            num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            image=image,
            images=images,
            prompt_relay=prompt_relay,
            generated_keyframes=0,
            enable_teacache=False,
            canvas_for=canvas_for,
            video_fps=conditioning_fps(frame_rate),
        )
        if not requested:
            raise RuntimeError(
                "_stage1 returned without calling the canvas_for hook, so the requested duration "
                "is unknown and the canvas padding cannot be trimmed back off."
            )

        # Upsample the stage-1 latent and the slots (one call each, as upstream).
        video_half = self.video_patchifier.unpatchify(stage1.video_tokens, stage1.latent_dims)
        video_upscaled = self._upsample_latent(video_half)
        assert stage1.generated_keyframes is not None
        slots_upscaled = self._upsample_latent(stage1.generated_keyframes)

        self._attach_detailing_lora()
        assert self._detailing_downscale is not None

        cond_fps_base = conditioning_fps(frame_rate)
        extra = [
            VideoGeneratedKeyframeSlots(
                pixel_frame_indices=self.generated_keyframe_positions,
                frame_rate=cond_fps_base,
                initial_keyframes=slots_upscaled,
            ),
            reference_conditioning_from_latent(
                video_half,
                frame_rate=cond_fps_base,
                downscale_factor=self._detailing_downscale,
                strength=1.0,
            ),
        ]
        video_latent, _stage2_audio = self._stage2(
            stage1,
            video_upscaled,
            num_frames=canvas_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage2_steps=stage2_steps,
            extra_conditionings=extra,
            video_fps=cond_fps_base,
        )
        # Stage 2 slots are kept in self.generated_keyframes (overwritten by _stage2's
        # extraction hook) for _decode_and_save_video to hand to the decoder as keyframes;
        # the temporal rounds replace them with their carry bag.
        num_frames = canvas_frames
        epilogue_seams: list[int] = []
        if self.temporal_upscalings:
            video_latent, num_frames, epilogue_seams = self._run_temporal_rounds(
                stage1, video_latent, canvas_frames=canvas_frames, frame_rate=frame_rate, seed=seed
            )
        if self.spatial_upscalings == 2:
            if self.temporal_upscalings:
                # The rounds ran without the detailing LoRA; the epilogue is a detailing pass.
                aggressive_cleanup()
                self._attach_detailing_lora()
            video_latent = self._run_spatial_epilogue(
                stage1,
                video_latent,
                canvas_frames=num_frames,
                frame_rate=frame_rate,
                seed=seed,
                epilogue_seams=epilogue_seams,
                stage2_steps=stage2_steps,
            )

        # Trim the canvas padding: video to the requested latent frames, audio (stage 1's, as
        # upstream ships it) to the requested duration in audio tokens, both on the final
        # (temporally upsampled) grid.
        scale = 2**self.temporal_upscalings
        target = (requested - 1) * scale + 1
        if target > num_frames:
            raise RuntimeError(f"Target {target} frames exceeds the generated canvas {num_frames}")
        keep_latent = (target - 1) // _TEMPORAL_SCALE + 1
        video_latent = video_latent[:, :, :keep_latent]
        audio_latent = self.audio_patchifier.unpatchify(stage1.audio_tokens)
        audio_latent = audio_latent[:, :, : compute_audio_token_count(target, frame_rate=frame_rate * scale)]
        return video_latent, audio_latent

    # ---- temporal rounds ------------------------------------------------------------
    def _denoise_temporal_tile(
        self,
        stage1: Stage1Result,
        tile_video: mx.array,
        *,
        cond_fps: float,
        anchors: list[tuple[int, mx.array]],
        slots_local: list[int],
        images: list[ImageConditioningInput],
        audio_tokens: mx.array,
        noise_seed: int,
        init_seed: int,
    ) -> tuple[mx.array, mx.array | None]:
        """Re-denoise one temporal tile (upstream round body): anchors, fresh slots, frozen audio, ancestral Euler.

        Args:
            stage1: The stage-1 result (text embeddings).
            tile_video: ``(1, 128, F, H, W)`` slice of the temporally upsampled latent.
            cond_fps: Transformer fps (:func:`conditioning_fps`).
            anchors: ``(local pixel index, (1, 128, 1, H, W) latent)`` carried keyframes.
            slots_local: Local pixel positions of this tile's new slots.
            images: Tile-local ``ImageConditioningInput`` anchors.
            audio_tokens: ``(1, T, 128)`` frozen audio tokens for this tile.
            noise_seed: Ancestral loop seed (``seed + 1000 * round + tile``).
            init_seed: Seed of the initial partial re-noise.

        Returns:
            ``(tile latent (1, 128, F, H, W), slot latents (1, 128, K, H, W) or None)``.
        """
        F, H, W = tile_video.shape[2], tile_video.shape[3], tile_video.shape[4]
        tokens, _ = self.video_patchifier.patchify(tile_video)
        conditionings: list = []
        if images:
            conditionings = combined_image_conditionings(
                images,
                enc_h=H * 32,
                enc_w=W * 32,
                spatial_dims=(F, H, W),
                video_encoder=self.vae_encoder,
                frame_rate=cond_fps,
            )
        # Every seam in the window is a hard keyframe, including the one at local frame 0.
        for local_index, latent in anchors:
            kf_tokens, _ = self.video_patchifier.patchify(latent)
            conditionings.append(
                VideoConditionByKeyframeIndex(
                    frame_idx=local_index,
                    keyframe_latent=kf_tokens,
                    spatial_dims=(F, H, W),
                    frame_rate=cond_fps,
                    strength=ANCHOR_KEYFRAME_STRENGTH,
                )
            )
        if slots_local:
            conditionings.append(
                VideoGeneratedKeyframeSlots(
                    pixel_frame_indices=slots_local,
                    frame_rate=cond_fps,
                    initial_keyframes=slot_initials_from_video(tile_video, slots_local),
                )
            )
        video_state = distilled_mod.create_noised_state(
            base_shape=tokens.shape,
            conditionings=conditionings,
            spatial_dims=(F, H, W),
            positions=compute_video_positions(F, H, W, frame_rate=cond_fps),
            seed=init_seed,
            sigma=TEMPORAL_SIGMAS[0],
            initial_latent=tokens,
        )
        # Frozen audio, as upstream ``ModalitySpec(frozen=True, noise_scale=0.0)``: an all-zero denoise mask keeps
        # the latent clean, and ``frozen`` makes the loop feed sigma 0 to the audio prompt AdaLN and the A->V gate.
        audio_state = distilled_mod.create_noised_state(
            base_shape=audio_tokens.shape,
            conditionings=[],
            spatial_dims=(F, H, W),  # unused
            positions=compute_audio_positions(audio_tokens.shape[1]),
            seed=init_seed + 1,
            sigma=0.0,
            initial_latent=audio_tokens,
            frozen=True,
        )
        output = distilled_mod.euler_ancestral_denoising_loop(
            transformer=distilled_mod.X0Model(self.dit),
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=stage1.video_embeds,
            audio_text_embeds=stage1.audio_embeds,
            sigmas=TEMPORAL_SIGMAS,
            stepper=EulerAncestralDiffusionStep(eta=TEMPORAL_ANCESTRAL_ETA),
            noise_seed=noise_seed,
        )
        slots = extract_generated_keyframes(
            output.video_latent, video_state.generated_keyframe_layout, self.video_patchifier, (H, W)
        )
        latent = self.video_patchifier.unpatchify(output.video_latent[:, : F * H * W, :], (F, H, W))
        _materialize(latent, *([] if slots is None else [slots]))
        return latent, slots

    def _run_temporal_rounds(
        self, stage1: Stage1Result, video_latent: mx.array, *, canvas_frames: int, frame_rate: float, seed: int
    ) -> tuple[mx.array, int, list[int]]:
        """Upstream ``DFRPipeline.__call__`` temporal rounds 1..T on the stage-2 canvas.

        Args:
            stage1: The stage-1 result (text embeddings, frozen audio tokens, I2V inputs).
            video_latent: Stage-2 latent ``(1, 128, F, H, W)`` over the whole canvas.
            canvas_frames: Pixel frames of the stage-2 canvas.
            frame_rate: Stage-1/2 playback frame rate.
            seed: Pipeline seed.

        Returns:
            ``(video latent after the last round, its pixel frame count, the last round's seams)``;
            the seams (pixel frames, upstream ``last_window_seams``) cut the spatial epilogue's
            temporal tiles. The carry bag is left in ``self.generated_keyframes`` /
            ``self.generated_keyframe_positions`` for the decode (or the epilogue).

        Raises:
            RuntimeError: missing carry keyframes, a tile without slots, or a stitched length mismatch.
        """
        # stage1.x0_model wraps the detailing-fused DiT: drop it (in place, so the caller's alias goes too)
        # or it stays resident beside the clean DiT the detach reloads. The rounds build their own X0Model.
        stage1.x0_model = None
        self._detach_detailing_lora()
        temporal_upsampler = self._load_temporal_upsampler()
        # low_memory stage 2 frees the VAE encoder; the rounds need its latent stats (denorm/renorm
        # around the temporal upsampler) and re-encode tile-local I2V images.
        self.image_conditioner.load()
        audio_full = self.audio_patchifier.unpatchify(stage1.audio_tokens)
        source_duration = canvas_frames / frame_rate
        carry_positions = list(self.generated_keyframe_positions)
        carry_keyframes = self.generated_keyframes
        num_frames, current_fps = canvas_frames, frame_rate
        seams: list[int] = []
        for round_idx in range(1, self.temporal_upscalings + 1):
            if carry_keyframes is None or not carry_positions:
                raise RuntimeError(f"Temporal round {round_idx}: missing carry-forward keyframes")
            video_latent = self._upsample_latent(video_latent, upsampler=temporal_upsampler)
            num_frames = 2 * (num_frames - 1) + 1
            current_fps *= 2
            cond_fps = conditioning_fps(current_fps)
            # Carried keyframes are single-frame latents, so only their positions scale with the round.
            seams = [2 * p for p in carry_positions]
            seam_index = {p: i for i, p in enumerate(seams)}
            plan = TemporalTilePlan(seams, num_frames, 2**round_idx)
            pieces: list[mx.array] = []
            slot_positions: list[int] = []
            slot_latents: list[mx.array] = []
            with phase(
                f"Temporal round {round_idx}/{self.temporal_upscalings} ({len(plan)} tiles)", verbose=self.verbose
            ):
                for tile_index, (interval, pixel_start, pixel_end, anchor_global, slot_global) in enumerate(plan):
                    local_frames = (interval.end - interval.start - 1) * _TEMPORAL_SCALE + 1
                    tile_video = video_latent[:, :, interval.start : interval.end]
                    missing = [p for p in anchor_global if p not in seam_index]
                    if missing:
                        raise RuntimeError(f"Anchor seams {missing} missing from the carry-forward bag")
                    anchors = [
                        (p - pixel_start, carry_keyframes[:, :, seam_index[p] : seam_index[p] + 1])
                        for p in anchor_global
                    ]
                    tile_audio = audio_latent_for_tile(
                        audio_full,
                        pixel_start=pixel_start,
                        local_frames=local_frames,
                        playback_fps=current_fps,
                        source_duration=source_duration,
                        cond_fps=cond_fps,
                    )
                    audio_tokens, _ = self.audio_patchifier.patchify(tile_audio)
                    latent, slots = self._denoise_temporal_tile(
                        stage1,
                        tile_video,
                        cond_fps=cond_fps,
                        anchors=anchors,
                        slots_local=[p - pixel_start for p in slot_global],
                        images=rebase_image_conditionings(
                            stage1.resolved_images,
                            pixel_scale=2**round_idx,
                            pixel_start=pixel_start,
                            pixel_end=pixel_end,
                        ),
                        audio_tokens=audio_tokens,
                        # Tiles are positionally identical: a shared ancestral seed would inject
                        # byte-identical noise into every one of them (upstream).
                        noise_seed=seed + 1000 * round_idx + tile_index,
                        # Upstream shares one GaussianNoiser for the initial re-noise; MLX noise is not
                        # torch-comparable anyway, so a per-tile seed (+500) is an MLX-side choice.
                        init_seed=seed + 1000 * round_idx + tile_index + 500,
                    )
                    pieces.append(latent[:, :, interval.left_ramp :])
                    if slot_global:
                        if slots is None:
                            raise RuntimeError(f"Temporal round {round_idx}: tile {tile_index} produced no slots")
                        slot_positions.extend(slot_global)
                        slot_latents.append(slots)
                    aggressive_cleanup()
            video_latent = mx.concatenate(pieces, axis=2)
            expected = (num_frames - 1) // _TEMPORAL_SCALE + 1
            if video_latent.shape[2] != expected:
                raise RuntimeError(f"Stitched latent T={video_latent.shape[2]} != expected {expected}")
            new_positions: list[int] = []
            new_latents = None
            if slot_positions:
                # Lead-in segments repeat the previous tile's slots; the earlier tile's version wins.
                new_positions, new_latents = dedupe_slots(slot_positions, mx.concatenate(slot_latents, axis=2))
            carry_positions, carry_keyframes = merge_carry_forward_keyframes(
                seams, carry_keyframes, new_positions, new_latents
            )
            _materialize(video_latent, carry_keyframes)
        self.generated_keyframes = carry_keyframes
        self.generated_keyframe_positions = carry_positions
        return video_latent, num_frames, seams

    # ---- spatial epilogue -----------------------------------------------------------
    @staticmethod
    def _floor_epilogue_resolution(height: int, width: int) -> tuple[int, int]:
        """Floor the output to multiples of 128 px for ``spatial_upscalings=2`` and warn when it changes.

        Upstream refuses such dims (``assert_resolution(divisor=128)``); the repo policy floors with a
        warning instead (``snap_output_dimensions``).
        """
        multiple = _EPILOGUE_RESOLUTION_MULTIPLE
        floored_h = max(multiple, floor_to_multiple(height, multiple))
        floored_w = max(multiple, floor_to_multiple(width, multiple))
        if (floored_h, floored_w) != (height, width):
            print(
                f"[dfr] --spatial-upscalings 2 snaps dims to multiples of {multiple}; output will be "
                f"{floored_w}x{floored_h} (requested {width}x{height}).",
                file=sys.stderr,
                flush=True,
            )
        return floored_h, floored_w

    def _decode_lanczos_carry_keyframes(self, keyframes: mx.array, seed: int) -> list[np.ndarray]:
        """Decode each carry plane as its own 1-frame clip, then Lanczos x2 it (upstream ``_decode_lanczos_carry_keyframes``).

        The render's decoder (``--video-decoder``) is loaded for the planes only and freed afterwards.

        Args:
            keyframes: ``(1, C, K, H, W)`` carry keyframe latents.
            seed: Pipeline seed; plane ``i`` decodes with ``seed + 4000 + i``.

        Returns:
            ``K`` arrays ``(1, 2 * H * 32, 2 * W * 32, 3)`` float32 in ``[0, 1]``.
        """
        if keyframes.ndim != 5:
            raise ValueError(f"Expected carry keyframes (B, C, K, H, W), got {tuple(keyframes.shape)}")
        block = self.video_decoder_block
        block.video_decoder = self.video_decoder
        block.diffvae_tile = self.diffvae_tile
        planes: list[np.ndarray] = []
        with phase(f"Decoding {keyframes.shape[2]} carry keyframes for the epilogue", verbose=self.verbose):
            for index in range(keyframes.shape[2]):
                rgb = block.decode_single_frame(
                    keyframes[:, :, index : index + 1], seed=seed + KEYFRAME_PLANE_DECODE_SEED_OFFSET + index
                )
                planes.append(lanczos_x2(np.array(rgb.astype(mx.float32))))
                del rgb
            block.free()
        return planes

    def _run_spatial_epilogue(
        self,
        stage1: Stage1Result,
        guide_latent: mx.array,
        *,
        canvas_frames: int,
        frame_rate: float,
        seed: int,
        epilogue_seams: list[int],
        stage2_steps: int | None = None,
    ) -> mx.array:
        """Full-resolution spatial detailing (upstream ``spatial_upscalings == 2`` epilogue).

        One deterministic Euler loop over the whole canvas; each model call is tiled 2x2 spatially
        (overlap 12) and in ``2**T`` frame tiles cut on the last round's seams, and the tile
        predictions are blended. Conditioned on the user images (full resolution, frame indices on
        the final grid), the carry keyframes re-decoded, Lanczos x2 upsampled and re-encoded
        (strength 1.0), and the H/2 latent as the detailing IC-LoRA reference; the stage-1 audio is
        frozen over the whole canvas (its output is discarded).

        Args:
            stage1: The stage-1 result (text embeddings, stage-1 audio, I2V inputs).
            guide_latent: ``(1, 128, F, H/2 cells, W/2 cells)`` latent after stage 2 / the rounds.
            canvas_frames: Pixel frames of ``guide_latent``'s canvas.
            frame_rate: Stage-1/2 playback frame rate (the canvas plays at ``frame_rate * 2**T``).
            seed: Pipeline seed.
            epilogue_seams: The last temporal round's seams in pixel frames (``[]`` without rounds).
            stage2_steps: Truncates the stage-2 sigma table, as for stage 2 (upstream shares it).

        Returns:
            The full-resolution latent ``(1, 128, F, 2H, 2W)``. ``self.generated_keyframes`` becomes the
            re-encoded planes (positions unchanged) for the final keyframe-aware decode.

        Raises:
            RuntimeError: missing carry keyframes.
            ValueError: the re-encoded plane count differs from the carry positions.
        """
        carry_positions = list(self.generated_keyframe_positions)
        carry_keyframes = self.generated_keyframes
        if carry_keyframes is None or not carry_positions:
            raise RuntimeError("Spatial epilogue: missing carry-forward keyframes")
        # The epilogue builds its own tiled X0Model; drop stage 1's so no second wrapper stays alive.
        stage1.x0_model = None
        scale = 2**self.temporal_upscalings
        playback_fps = frame_rate * scale
        cond_fps = conditioning_fps(playback_fps)

        pixel_planes = self._decode_lanczos_carry_keyframes(carry_keyframes, seed)
        # low_memory stage 2 frees the VAE encoder and the spatial upsampler; the epilogue needs both.
        self.image_conditioner.load()
        if self.upsampler is None:
            self._load_upsampler()
        video_latent = self._upsample_latent(guide_latent)
        F, H, W = video_latent.shape[2], video_latent.shape[3], video_latent.shape[4]

        conditionings: list = []
        images = rebase_image_conditionings(stage1.resolved_images, pixel_scale=scale)
        if images:
            conditionings = combined_image_conditionings(
                images,
                enc_h=H * 32,
                enc_w=W * 32,
                spatial_dims=(F, H, W),
                video_encoder=self.vae_encoder,
                frame_rate=cond_fps,
            )
        encoded: list[mx.array] = []
        for rgb in pixel_planes:
            # (1, H, W, 3) in [0, 1] -> (1, 3, 1, H, W) in [-1, 1] (``to_vae_range``), bf16 as upstream.
            sample = mx.array(rgb * 2.0 - 1.0).transpose(3, 0, 1, 2)[None].astype(mx.bfloat16)
            encoded.append(self.vae_encoder.encode(sample))
        encoded_kfs = mx.concatenate(encoded, axis=2)
        # Materialise everything the encoder produced so the low_memory free below takes effect at once.
        image_tokens = [
            tokens
            for c in conditionings
            for tokens in (getattr(c, "clean_latent", None), getattr(c, "keyframe_latent", None))
            if tokens is not None
        ]
        _materialize(encoded_kfs, *image_tokens)
        del pixel_planes, encoded, image_tokens
        if encoded_kfs.shape[2] != len(carry_positions):
            # Upstream ``_keyframe_conditionings_from_latents``; MLX slicing would silently clamp instead.
            raise ValueError(f"Expected {len(carry_positions)} keyframe latents, got K={encoded_kfs.shape[2]}")
        for index, position in enumerate(carry_positions):
            kf_tokens, _ = self.video_patchifier.patchify(encoded_kfs[:, :, index : index + 1])
            conditionings.append(
                VideoConditionByKeyframeIndex(
                    frame_idx=position,
                    keyframe_latent=kf_tokens,
                    spatial_dims=(F, H, W),
                    frame_rate=cond_fps,
                    strength=EPILOGUE_KEYFRAME_STRENGTH,
                )
            )
        assert self._detailing_downscale is not None
        conditionings.append(
            reference_conditioning_from_latent(
                guide_latent, frame_rate=cond_fps, downscale_factor=self._detailing_downscale, strength=1.0
            )
        )
        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        seams_latent = [pixel_to_latent_index(p) for p in epilogue_seams]
        time_tiles = max(1, scale)
        temporal_overlap = seams_latent[0] + 1 if time_tiles > 1 and seams_latent else 0
        tiling = clamp_tile_counts(
            TileCountConfig(
                frames=DimensionTilingConfig(time_tiles, temporal_overlap),
                height=DimensionTilingConfig(2, EPILOGUE_SPATIAL_OVERLAP),
                width=DimensionTilingConfig(2, EPILOGUE_SPATIAL_OVERLAP),
            ),
            (F, H, W),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W), seams=seams_latent)
        model = distilled_mod.X0Model(TiledLTXModel(self.dit, tiler, normalize_positions=True))

        tokens, _ = self.video_patchifier.patchify(video_latent)
        sigmas = LTX_2_5_STAGE_2_DISTILLED_SIGMAS
        sigmas = sigmas[: stage2_steps + 1] if stage2_steps else sigmas
        video_state = distilled_mod.create_noised_state(
            base_shape=tokens.shape,
            conditionings=conditionings,
            spatial_dims=(F, H, W),
            positions=compute_video_positions(F, H, W, frame_rate=cond_fps),
            seed=seed + EPILOGUE_NOISE_SEED_OFFSET,
            sigma=sigmas[0],
            initial_latent=tokens,
        )
        audio_tile = audio_latent_for_tile(
            self.audio_patchifier.unpatchify(stage1.audio_tokens),
            pixel_start=0,
            local_frames=canvas_frames,
            playback_fps=playback_fps,
            source_duration=self.canvas_frames / frame_rate,
            cond_fps=cond_fps,
        )
        audio_tokens, _ = self.audio_patchifier.patchify(audio_tile)
        # Frozen audio (upstream ``ModalitySpec(frozen=True, noise_scale=0.0)``), as in the temporal rounds.
        audio_state = distilled_mod.create_noised_state(
            base_shape=audio_tokens.shape,
            conditionings=[],
            spatial_dims=(F, H, W),  # unused
            positions=compute_audio_positions(audio_tokens.shape[1]),
            seed=seed + EPILOGUE_NOISE_SEED_OFFSET + 1,
            sigma=0.0,
            initial_latent=audio_tokens,
            frozen=True,
        )
        with phase(
            f"Spatial epilogue ({len(sigmas) - 1} steps over {len(tiler.tiles)} tiles, "
            f"frames={tiling.frames} height={tiling.height} width={tiling.width}, seams={seams_latent})",
            verbose=self.verbose,
        ):
            self._pre_denoise_flush(video_state, audio_state)
            output = distilled_mod.denoise_loop(
                model=model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=stage1.video_embeds,
                audio_text_embeds=stage1.audio_embeds,
                sigmas=sigmas,
                on_step=self._stepwise_hook(F, H, W, stage=3),
            )
        latent = self.video_patchifier.unpatchify(output.video_latent[:, : F * H * W, :], (F, H, W))
        _materialize(latent)
        self.generated_keyframes = encoded_kfs
        aggressive_cleanup()
        return latent

    def _decode_and_save_video(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        output_path: str,
        *,
        frame_rate: float,
        seed: int = 0,
        keyframes: DecodeKeyframes | None = None,
    ) -> str:
        """Decode with the final keyframe bag as decoder keyframes (upstream's final ``decode_keyframes_from_slots`` handoff).

        The keyframes are the carry bag after the temporal rounds, else the stage-2 slots. The conv decoder
        ignores them with a warning; ``--video-decoder diffusion`` runs the keyframe-aware decode. The output
        is written at ``frame_rate * 2**temporal_upscalings`` (each round doubles the frame count).
        """
        if keyframes is None:
            num_frames = (video_latent.shape[2] - 1) * _TEMPORAL_SCALE + 1
            keyframes = decode_keyframes_from_slots(
                self.generated_keyframes, self.generated_keyframe_positions, num_frames, verbose=self.verbose
            )
        return super()._decode_and_save_video(
            video_latent,
            audio_latent,
            output_path,
            frame_rate=frame_rate * 2**self.temporal_upscalings,
            seed=seed,
            keyframes=keyframes,
        )


def decode_keyframes_from_slots(
    slots: mx.array | None, positions: Sequence[int], num_frames: int, *, verbose: bool = False
) -> DecodeKeyframes | None:
    """Build the decoder's keyframe input from the stage-2 slot latents (upstream ``helpers.decode_keyframes_from_slots``).

    Slots at pixel frames outside ``[0, num_frames)`` (the canvas padding) are dropped; ``None`` when
    nothing is left.

    Args:
        slots: ``(B, C, K, H, W)`` generated keyframe slot latents (normalised), or ``None``.
        positions: Canvas pixel-frame index of each slot.
        num_frames: Pixel frames of the trimmed video latent.
        verbose: Print the dropped slots to stdout.
    """
    if slots is None:
        return None
    if slots.shape[2] != len(positions):
        raise ValueError(f"slot count {slots.shape[2]} != len(positions) {len(positions)}")
    keep = [i for i, p in enumerate(positions) if 0 <= p < num_frames]
    dropped = len(positions) - len(keep)
    if verbose and dropped:
        print(
            f"[dfr] dropping {dropped} keyframe slot(s) beyond the trimmed {num_frames} frames",
            file=sys.stderr,
            flush=True,
        )
    if not keep:
        return None
    return DecodeKeyframes(
        latents=slots[:, :, keep],
        pixel_frame_indices=tuple(int(positions[i]) for i in keep),
        clip_start_frame=0,
    )


__all__ = [
    "ANCHOR_KEYFRAME_STRENGTH",
    "DEFAULT_DETAILING_LORA",
    "DETAILING_LORA_STRENGTH",
    "EPILOGUE_KEYFRAME_STRENGTH",
    "EPILOGUE_NOISE_SEED_OFFSET",
    "EPILOGUE_SPATIAL_OVERLAP",
    "KEYFRAME_PLANE_DECODE_SEED_OFFSET",
    "TEMPORAL_ANCESTRAL_ETA",
    "TEMPORAL_SIGMAS",
    "DFRPipeline",
    "audio_latent_for_tile",
    "clamp_tile_counts",
    "conditioning_fps",
    "decode_keyframes_from_slots",
    "dedupe_slots",
    "floor_to_multiple",
    "lanczos_x2",
    "merge_carry_forward_keyframes",
    "rebase_image_conditionings",
    "resample_audio_time",
    "slot_initials_from_video",
]
