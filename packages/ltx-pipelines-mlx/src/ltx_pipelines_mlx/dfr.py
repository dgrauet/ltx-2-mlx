"""DFR ("Diffusion Fidelity Rendering") base path — upstream ``ltx_pipelines/dfr_pipeline.py``.

Stage 1 (half resolution, distilled, ancestral on 2.5) runs on a canvas padded to whole keyframe
segments with one generated keyframe slot per segment boundary. Stage 2 (full resolution,
deterministic) runs with the detailing IC-LoRA attached at strength 0.5, conditioned on the
stage-1 latent as an IC-LoRA reference and on the spatially upsampled stage-1 slots. This port
covers ``spatial_upscalings=1`` / ``temporal_upscalings=0`` and decodes without keyframes; the
keyframe-aware decode, the temporal rounds and the spatial epilogue are follow-ups.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import mlx.core as mx

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
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_token_count
from ltx_core_mlx.utils.weights import apply_quantization
from ltx_pipelines_mlx.dfr_layout import resolve_canvas
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.iclora_utils import (
    read_lora_reference_downscale_factor,
    reference_conditioning_from_latent,
)
from ltx_pipelines_mlx.utils._orchestration import resolve_lora_path
from ltx_pipelines_mlx.utils.progress import phase
from ltx_pipelines_mlx.utils.types import DEFAULT_AUTO_DURATION, AutoDuration

logger = logging.getLogger(__name__)

#: Official LTX-2.5 detailing IC-LoRA (creative x2 spatial upsampler), ``reference_downscale_factor: 2``.
DEFAULT_DETAILING_LORA = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
#: Upstream ``_DETAILING_LORA_STRENGTH``; not a user knob.
DETAILING_LORA_STRENGTH = 0.5
#: Pixel frames per latent frame of the video VAE (there is no pipeline-level constant).
_TEMPORAL_SCALE = 8


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
            self._detailing_lora_path = resolve_lora_path(self.detailing_lora)
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
        # extraction hook) for the keyframe-aware decode (follow-up 4b).

        # Trim the canvas padding: video to the requested latent frames, audio (stage 1's, as
        # upstream ships it) to the requested duration in audio tokens.
        keep_latent = (requested - 1) // _TEMPORAL_SCALE + 1
        video_latent = video_latent[:, :, :keep_latent]
        audio_latent = self.audio_patchifier.unpatchify(stage1.audio_tokens)
        audio_latent = audio_latent[:, :, : compute_audio_token_count(requested, frame_rate=frame_rate)]
        return video_latent, audio_latent


__all__ = ["DEFAULT_DETAILING_LORA", "DETAILING_LORA_STRENGTH", "DFRPipeline"]
