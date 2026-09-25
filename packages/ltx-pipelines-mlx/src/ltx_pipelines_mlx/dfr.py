"""DFR ("Diffusion Fidelity Rendering") base path — upstream ``ltx_pipelines/dfr_pipeline.py``.

Stage 1 (half resolution, distilled, ancestral on 2.5) runs on a canvas padded to whole keyframe
segments with one generated keyframe slot per segment boundary. Stage 2 (full resolution,
deterministic) runs with the detailing IC-LoRA attached at strength 0.5, conditioned on the
stage-1 latent as an IC-LoRA reference and on the spatially upsampled stage-1 slots. This port
covers ``spatial_upscalings=1`` with ``temporal_upscalings`` 0, 1 or 2. Each temporal round
upsamples the latent x2 in time with the temporal upsampler, then re-denoises it in keyframe-seam
tiles (carried keyframes as anchors, fresh slots between them, frozen stage-1 audio, 4-step
ancestral Euler) with the distilled transformer without the detailing LoRA; the output plays at
``frame_rate * 2**temporal_upscalings``. The final keyframe bag (stage-2 slots, or the carry bag
after the rounds) is handed to the decoder as decoder keyframes (keyframe-aware decode on
``--video-decoder diffusion``; the conv decoder ignores them with a warning). The spatial
epilogue is a follow-up.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
from huggingface_hub.errors import GatedRepoError

import ltx_pipelines_mlx.distilled as distilled_mod
from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
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
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import apply_quantization
from ltx_pipelines_mlx.dfr_layout import TemporalTilePlan, resolve_canvas
from ltx_pipelines_mlx.distilled import DistilledPipeline, Stage1Result
from ltx_pipelines_mlx.iclora_utils import (
    read_lora_reference_downscale_factor,
    reference_conditioning_from_latent,
)
from ltx_pipelines_mlx.scheduler import LTX_2_5_DISTILLED_SIGMAS
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

#: Upstream ``_ANCHOR_KEYFRAME_STRENGTH``: carried keyframes pinned just short of clean.
ANCHOR_KEYFRAME_STRENGTH = 0.95
#: Upstream ``_TEMPORAL_ANCESTRAL_ETA``.
TEMPORAL_ANCESTRAL_ETA = 0.5
#: Upstream ``DISTILLED_SIGMAS[4:]``: the 4-step temporal-round schedule.
TEMPORAL_SIGMAS: list[float] = list(LTX_2_5_DISTILLED_SIGMAS[4:])
_MAX_CONDITIONING_FPS = 60.0
_SNAP_CONDITIONING_FPS_ABOVE = 30.0


def conditioning_fps(playback_fps: float) -> float:
    """Transformer RoPE fps (upstream ``_conditioning_fps``): above 30 snaps to 60; playback fps is unchanged."""
    return _MAX_CONDITIONING_FPS if playback_fps > _SNAP_CONDITIONING_FPS_ABOVE else playback_fps


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
    """

    #: Class default so a ``__new__``-built instance (tests) decodes at the base frame rate.
    temporal_upscalings: int = 0

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
        self.detailing_lora = detailing_lora
        self._detailing_lora_path: str | None = None
        self._detailing_downscale: int | None = None
        self._detailing_source: BlockLoraSource | None = None
        self.canvas_frames: int = 0
        self.generated_keyframe_positions: list[int] = []
        self.temporal_upscalings = temporal_upscalings
        self.temporal_upsampler_path = temporal_upsampler_path

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

        stage1, canvas_frames, height, width = self._stage1(
            prompt,
            height,
            width,
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
        if self.temporal_upscalings:
            video_latent, num_frames = self._run_temporal_rounds(
                stage1, video_latent, canvas_frames=canvas_frames, frame_rate=frame_rate, seed=seed
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
        # Frozen audio: setting denoise_mask=0 below keeps the audio latent clean (never denoised)
        # across the ancestral loop. Known divergence: upstream additionally builds this stream as
        # ``ModalitySpec(frozen=True, ...)``, which forces the audio ``sigma`` fed to the model to
        # 0 as well; that sigma drives the audio prompt AdaLN and the audio->video cross-attention
        # gate. Here, because the mask is uniformly 0, ``euler_ancestral_denoising_loop`` treats the
        # audio state as "uniform" and skips computing per-token timesteps for it, so the model
        # falls back to the loop's *global* step sigma for the audio AdaLN / A->V gate instead of a
        # frozen 0 — the audio latent itself still stays clean (denoise_mask blending is unaffected),
        # but the model isn't told the audio is frozen. Shared with a2v and lipdub; tracked for a
        # follow-up fix.
        audio_state = distilled_mod.create_noised_state(
            base_shape=audio_tokens.shape,
            conditionings=[],
            spatial_dims=(F, H, W),  # unused
            positions=compute_audio_positions(audio_tokens.shape[1]),
            seed=init_seed + 1,
            sigma=0.0,
            initial_latent=audio_tokens,
        )
        audio_state = dataclasses.replace(audio_state, denoise_mask=mx.zeros_like(audio_state.denoise_mask))
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
    ) -> tuple[mx.array, int]:
        """Upstream ``DFRPipeline.__call__`` temporal rounds 1..T on the stage-2 canvas.

        Args:
            stage1: The stage-1 result (text embeddings, frozen audio tokens, I2V inputs).
            video_latent: Stage-2 latent ``(1, 128, F, H, W)`` over the whole canvas.
            canvas_frames: Pixel frames of the stage-2 canvas.
            frame_rate: Stage-1/2 playback frame rate.
            seed: Pipeline seed.

        Returns:
            ``(video latent after the last round, its pixel frame count)``; the carry bag is left in
            ``self.generated_keyframes`` / ``self.generated_keyframe_positions`` for the decode.

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
        return video_latent, num_frames

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
    "TEMPORAL_ANCESTRAL_ETA",
    "TEMPORAL_SIGMAS",
    "DFRPipeline",
    "audio_latent_for_tile",
    "conditioning_fps",
    "decode_keyframes_from_slots",
    "dedupe_slots",
    "merge_carry_forward_keyframes",
    "rebase_image_conditionings",
    "resample_audio_time",
    "slot_initials_from_video",
]
