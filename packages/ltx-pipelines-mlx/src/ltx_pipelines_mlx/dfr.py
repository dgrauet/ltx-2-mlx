"""DFR ("Diffusion Fidelity Rendering") base path — upstream ``ltx_pipelines/dfr_pipeline.py``.

Stage 1 (half resolution, distilled, ancestral on 2.5) runs on a canvas padded to whole keyframe
segments with one generated keyframe slot per segment boundary. Stage 2 (full resolution,
deterministic) runs with the detailing IC-LoRA attached at strength 0.5, conditioned on the
stage-1 latent as an IC-LoRA reference and on the spatially upsampled stage-1 slots. This port
covers ``spatial_upscalings=1`` / ``temporal_upscalings=0``. The stage-2 keyframe slot latents
are handed to the decoder as decoder keyframes (keyframe-aware decode on
``--video-decoder diffusion``; the conv decoder ignores them with a warning). The temporal
rounds and the spatial epilogue are follow-ups.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

import mlx.core as mx
from huggingface_hub.errors import GatedRepoError

from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core_mlx.loader import (
    LTXV_LORA_BLOCK_PREFIX,
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraStateDictWithStrength,
    SafetensorsStateDictLoader,
    StateDict,
    apply_loras,
)
from ltx_core_mlx.loader.block_streaming import BlockLoraSource
from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import DecodeKeyframes
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_token_count
from ltx_core_mlx.utils.weights import apply_quantization
from ltx_pipelines_mlx.dfr_layout import resolve_canvas
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.iclora_utils import (
    read_lora_reference_downscale_factor,
    reference_conditioning_from_latent,
)
from ltx_pipelines_mlx.scheduler import LTX_2_5_DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils._orchestration import resolve_lora_path
from ltx_pipelines_mlx.utils.progress import phase
from ltx_pipelines_mlx.utils.types import DEFAULT_AUTO_DURATION, AutoDuration

logger = logging.getLogger(__name__)

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern

#: Official LTX-2.5 detailing IC-LoRA (creative x2 spatial upsampler), ``reference_downscale_factor: 2``.
DEFAULT_DETAILING_LORA = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
#: Upstream ``_DETAILING_LORA_STRENGTH``; not a user knob.
DETAILING_LORA_STRENGTH = 0.5
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
    images: Sequence, *, pixel_scale: int, pixel_start: int = 0, pixel_end: int | None = None
) -> list:
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
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        tile_count=None,
        detailing_lora: str = DEFAULT_DETAILING_LORA,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            tile_count=tile_count,
        )
        self.detailing_lora = detailing_lora
        self._detailing_lora_path: str | None = None
        self._detailing_downscale: int | None = None
        self.canvas_frames: int = 0
        self.generated_keyframe_positions: list[int] = []

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
        stage 1 is finished and the model is never reused clean.
        """
        assert self.dit is not None
        path = self._resolve_detailing_lora()
        with phase("Attaching the detailing IC-LoRA (strength 0.5)", verbose=self.verbose):
            if self.low_ram_streaming:
                sources: list = list(object.__getattribute__(self.dit, "_lora_sources"))
                sources.append(
                    BlockLoraSource(
                        path,
                        block_prefix=LTXV_LORA_BLOCK_PREFIX,
                        strength=DETAILING_LORA_STRENGTH,
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                )
                object.__setattr__(self.dit, "_lora_sources", sources)
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

        extra = [
            VideoGeneratedKeyframeSlots(
                pixel_frame_indices=self.generated_keyframe_positions,
                frame_rate=frame_rate,
                initial_keyframes=slots_upscaled,
            ),
            reference_conditioning_from_latent(
                video_half,
                frame_rate=frame_rate,
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
        )
        # Stage 2 slots are kept in self.generated_keyframes (overwritten by _stage2's
        # extraction hook) for _decode_and_save_video to hand to the decoder as keyframes.

        # Trim the canvas padding: video to the requested latent frames, audio (stage 1's, as
        # upstream ships it) to the requested duration in audio tokens.
        keep_latent = (requested - 1) // _TEMPORAL_SCALE + 1
        video_latent = video_latent[:, :, :keep_latent]
        audio_latent = self.audio_patchifier.unpatchify(stage1.audio_tokens)
        audio_latent = audio_latent[:, :, : compute_audio_token_count(requested, frame_rate=frame_rate)]
        return video_latent, audio_latent

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
        """Decode with the stage-2 keyframe slots as decoder keyframes (upstream's final ``decode_keyframes_from_slots`` handoff).

        The conv decoder ignores them with a warning; ``--video-decoder diffusion`` runs the keyframe-aware decode.
        """
        if keyframes is None:
            num_frames = (video_latent.shape[2] - 1) * _TEMPORAL_SCALE + 1
            keyframes = decode_keyframes_from_slots(
                self.generated_keyframes, self.generated_keyframe_positions, num_frames, verbose=self.verbose
            )
        return super()._decode_and_save_video(
            video_latent, audio_latent, output_path, frame_rate=frame_rate, seed=seed, keyframes=keyframes
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
