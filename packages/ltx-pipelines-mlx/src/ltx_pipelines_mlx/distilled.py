"""Distilled two-stage video generation pipeline.

Mirrors upstream ``ltx_pipelines.distilled.DistilledPipeline`` 1:1:

  Stage 1: Distilled DiT at **half resolution** (8 steps, no CFG).
  Stage 2: Spatial 2x upscaler + distilled DiT refine at **full resolution**
           (3 steps, no CFG).

Same distilled checkpoint is used in both stages — no LoRA fusion between
stages (the model is already distilled). Use this pipeline when you want
the speed of the distilled model at higher target resolutions, where
running distilled directly at full res can produce out-of-distribution
artefacts.

For the simpler distilled-at-target one-stage path, see
:class:`BasePipeline`.

For dev model + CFG quality, see :class:`TI2VidTwoStagesPipeline` /
:class:`TI2VidTwoStagesHQPipeline`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx

from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core_mlx.components.patchifiers import (
    compute_video_latent_shape,
    snap_output_dimensions,
)
from ltx_core_mlx.conditioning.types.keyframe_slots import extract_generated_keyframes
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)
from ltx_pipelines_mlx.utils.helpers import generated_keyframe_conditionings

from .scheduler import (
    DISTILLED_SIGMAS,
    LTX_2_5_DISTILLED_SIGMAS,
    LTX_2_5_STAGE_2_DISTILLED_SIGMAS,
    STAGE_2_SIGMAS,
    shorten_schedule,
)
from .ti2vid_two_stages import TI2VidTwoStagesPipeline
from .utils.helpers import create_noised_state
from .utils.progress import phase
from .utils.samplers import denoise_loop, euler_ancestral_denoising_loop
from .utils.types import DEFAULT_AUTO_DURATION, AutoDuration

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern

# Ancestral-sampler defaults for LTX-2.5 distilled packs (upstream
# distilled.py). eta=1.0 / s_noise=1.0 reproduce upstream's default
# EulerAncestralDiffusionStep configuration verbatim.
ANCESTRAL_ETA = 1.0
ANCESTRAL_S_NOISE = 1.0
# The loop's noise generator is seeded from the pipeline seed plus this
# offset. Without it the loop's first draw would be bit-identical to the
# initial latent noise: GaussianNoiser and the loop's plain-noise draw both
# pull mx.random.normal at the same shape/dtype from a freshly seeded
# generator, so reusing the raw seed would correlate the two draws.
ANCESTRAL_NOISE_SEED_OFFSET = 10000


@dataclass
class Stage1Result:
    """What stage 1 hands to stage 2 (see :meth:`DistilledPipeline._stage1`).

    Attributes:
        video_tokens: Generated video tokens at half resolution, `(1, F*H_half*W_half, 128)`,
            with any appended slot/conditioning tokens already stripped.
        audio_tokens: Generated audio tokens at half resolution.
        generated_keyframes: Extracted generated-keyframe slot content,
            `(1, 128, K, H_half, W_half)`, or ``None`` when no slots were requested.
        latent_dims: Stage 1 latent dimensions `(F, H_half, W_half)`.
        resolved_images: Resolved multi-anchor I2V conditioning inputs.
        video_embeds: Video text embeddings from prompt encoding.
        audio_embeds: Audio text embeddings from prompt encoding.
        relay_mask: Prompt Relay cross-attention mask builder.
        x0_model: The stage 1 X0 model, reused for stage 2 when tiling is off.
    """

    video_tokens: mx.array
    audio_tokens: mx.array
    generated_keyframes: mx.array | None
    latent_dims: tuple[int, int, int]
    resolved_images: list
    video_embeds: mx.array
    audio_embeds: mx.array
    relay_mask: Callable
    x0_model: X0Model


class DistilledPipeline(TI2VidTwoStagesPipeline):
    """Distilled two-stage T2V/I2V pipeline (half-res → upscale → full-res refine).

    Reuses :class:`TI2VidTwoStagesPipeline`'s upsampler loading and helpers but
    overrides ``generate_two_stage`` to:

    - Skip negative-prompt encoding (no CFG).
    - Load the distilled transformer directly (no dev model, no LoRA fusion).
    - Run simple ``denoise_loop`` with ``DISTILLED_SIGMAS`` for stage 1.
    - Run the same distilled transformer for stage 2 with ``STAGE_2_SIGMAS``.

    On an LTX-2.5 pack (detected once at construction via
    :func:`~ltx_pipelines_mlx.utils.generation.is_ltx25_pack`) both stages run
    on the ``LTX_2_5_*`` sigma tables, stage 1 switches to the ancestral (SDE)
    Euler loop (stage 2 stays deterministic, as upstream), and stage 2 resolves
    the ``spatial_upscaler_x2_v1_0`` upscaler.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID. Must
            contain the distilled checkpoint (e.g. ``dgrauet/ltx-2.3-mlx-q8``
            ships ``transformer-distilled.safetensors``).
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        low_ram_streaming: Stream transformer blocks from disk.
        tile_count: Optional modality tiling configuration.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        tile_count=None,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            tile_count=tile_count,
        )

    def load(self) -> None:
        """Load distilled DiT + VAE encoder + upsampler (skip decoders).

        Skips reloading the text encoder: ``generate_two_stage`` encodes
        the prompt and frees Gemma BEFORE calling :meth:`load`. Loading
        Gemma again here would just thrash the Metal heap (7.5 GB
        load/mmap + free) right before DiT is loaded — a documented
        cause of macOS GPU watchdog crashes under sustained system
        contention.
        """
        if self._loaded:
            return

        if self.dit is None:
            transformer_path = self.model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = self._resolve_safetensors(self.model_dir, "transformer-distilled")
            self.dit = self._load_transformer_with_optional_streaming(transformer_path)

        self._load_vae_encoder()

        if self.upsampler is None:
            self._load_upsampler()

        self._loaded = True

    def _run_denoise_loop(
        self,
        *,
        model,
        video_state,
        audio_state,
        video_text_embeds: mx.array,
        audio_text_embeds: mx.array,
        sigmas: list[float],
        video_cross_attention_mask: mx.array | None,
        on_step,
        seed: int,
        ancestral: bool,
    ):
        """Dispatch one stage onto the deterministic or ancestral (SDE) loop.

        LTX-2.5 distilled checkpoints are trained for the ancestral (SDE) Euler
        sampler; 2.3 checkpoints keep the deterministic loop they were shipped
        with. Upstream makes the same choice through ``DiffusionStage``'s
        ``stepper`` / ``loop`` overrides, and scopes them to stage 1 only
        (``_stage_1_sampler_kwargs``) — quoting upstream ``distilled.py``:

            Stage 1 samples with the ancestral (SDE) Euler sampler or the
            deterministic one according to ``self.use_ancestral_sampler``.
            Stage 2 is always deterministic -- its 3-step refinement schedule
            is too short to remove freshly injected noise.

        Hence ``ancestral`` is passed per stage rather than read off
        ``self._is_25``: only stage 1 of a 2.5 pack sets it.

        The two loops differ only in the model keyword (``model=`` vs upstream's
        ``transformer=``) and in the ancestral extras (``stepper`` /
        ``noise_seed``): positions and attention masks are resolved from the
        :class:`LatentState` by both, so the call structure is otherwise the
        one already used at the 2.3 call sites.
        """
        if not ancestral:
            return denoise_loop(
                model=model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_text_embeds,
                audio_text_embeds=audio_text_embeds,
                sigmas=sigmas,
                video_cross_attention_mask=video_cross_attention_mask,
                on_step=on_step,
            )

        return euler_ancestral_denoising_loop(
            transformer=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            sigmas=sigmas,
            stepper=EulerAncestralDiffusionStep(eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE),
            noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            video_cross_attention_mask=video_cross_attention_mask,
            on_step=on_step,
        )

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
        """Generate video using the distilled two-stage pipeline.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames, or an :class:`AutoDuration` request to
                predict it from the prompt (requires a DurationHead-equipped
                checkpoint).
            generated_keyframes: ``0`` (off), an ``int`` for that many evenly spaced interior
                generated keyframe slots, or explicit pixel-frame indices. Stage 1 only;
                requires a pack with ``use_keyframes_abs_pos_embedding`` (LTX 2.5).
            seed: Random seed.
            stage1_steps: Stage 1 steps (default: full DISTILLED_SIGMAS = 8).
            stage2_steps: Stage 2 steps (default: full STAGE_2_SIGMAS = 3).
            image: Optional reference image for I2V conditioning.
            enable_teacache: Accepted for signature compatibility with
                :meth:`TI2VidTwoStagesPipeline.generate_two_stage`. Ignored on
                LTX-2.3 packs (the 8-step distilled flow has never used
                TeaCache); rejected on LTX-2.5 packs.
            **_unused_kwargs: Accepted (and ignored) for signature compatibility
                with :meth:`TI2VidTwoStagesPipeline.generate_two_stage`. CFG / STG
                flags don't apply to the distilled flow.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.

        Raises:
            ValueError: when ``enable_teacache`` is requested on an LTX-2.5 pack.
        """
        stage1, num_frames, height, width = self._stage1(
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
            generated_keyframes=generated_keyframes,
            enable_teacache=enable_teacache,
        )
        video_half = self.video_patchifier.unpatchify(stage1.video_tokens, stage1.latent_dims)
        video_upscaled = self._upsample_latent(video_half)
        return self._stage2(
            stage1,
            video_upscaled,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage2_steps=stage2_steps,
            extra_conditionings=[],
        )

    def _stage1(
        self,
        prompt: str,
        height: int,
        width: int,
        num_frames: int | AutoDuration,
        *,
        frame_rate: float,
        seed: int,
        stage1_steps: int | None,
        image: str | None,
        images,
        prompt_relay,
        generated_keyframes: int | Sequence[int],
        enable_teacache: bool,
        canvas_for: Callable[[int], tuple[int, list[int]]] | None = None,
    ) -> tuple[Stage1Result, int, int, int]:
        """Encode the prompt and denoise stage 1 at half resolution.

        Args:
            prompt: Text prompt.
            height: Requested video height (before dimension snapping).
            width: Requested video width (before dimension snapping).
            num_frames: Number of frames, or an :class:`AutoDuration` request.
            frame_rate: Video frame rate.
            seed: Random seed.
            stage1_steps: Stage 1 steps (default: full stage-1 sigma table).
            image: Optional reference image for I2V conditioning.
            images: Optional multi-anchor I2V conditioning inputs.
            prompt_relay: Optional Prompt Relay segment specs.
            generated_keyframes: ``0`` (off), an ``int``, or explicit pixel-frame indices.
            enable_teacache: Whether TeaCache was requested (rejected on 2.5 packs).
            canvas_for: Optional hook called with the resolved ``num_frames``, returning
                ``(canvas_frames, slot_pixel_indices)``. Used by :class:`DFRPipeline` to pad
                the clip to whole keyframe segments and place one slot per boundary; the
                returned canvas length replaces ``num_frames`` for the rest of stage 1 and is
                what this method returns. ``None`` (every other caller) leaves both untouched.

        Returns:
            Tuple of (stage1 result, resolved num_frames, resolved height, resolved width).

        Raises:
            ValueError: when ``enable_teacache`` is requested on an LTX-2.5 pack.
        """
        self._require_num_frames_source(num_frames)
        self._require_generated_keyframes_support(generated_keyframes)
        if enable_teacache and self._is_25:
            raise ValueError(
                "TeaCache is not calibrated for LTX-2.5 packs: the polynomial "
                "coefficients and threshold were fitted on the LTX-2.3 dev "
                "model with the deterministic Euler sampler, and the 2.5 "
                "distilled path runs an ancestral (SDE) sampler whose per-step "
                "residual deltas differ. Re-run without --enable-teacache."
            )

        # --- Prompt Relay setup (temporal prompt gating on video cross-attn) ---
        encode_prompt, relay_token_ranges = self._prompt_relay_setup(prompt, prompt_relay)

        # --- Text encoding (positive only — no CFG) ---
        self._load_text_encoder()
        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(encode_prompt)
            _materialize(video_embeds, audio_embeds)
        num_frames = self._resolve_num_frames(
            num_frames, video_encoding=video_embeds, audio_encoding=audio_embeds, frame_rate=frame_rate
        )
        if canvas_for is not None:
            num_frames, generated_keyframes = canvas_for(num_frames)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        # Per-stage Prompt Relay mask builder. Ranges were computed pre-encode;
        # the mask is rebuilt each stage because tokens-per-frame (H*W) differs.
        num_text_tokens = video_embeds.shape[1]
        relay_mask = self._prompt_relay_mask_builder(prompt_relay, relay_token_ranges, num_text_tokens)

        # --- Load distilled DiT + VAE encoder + upsampler ---
        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # --- Stage 1: half resolution ---
        # Snap to the two-stage grid (multiples of 64) and report if it changed.
        height, width = snap_output_dimensions(height, width, two_stage=True)
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # I2V conditioning at half resolution. ``images`` is the upstream-iso
        # multi-anchor list; ``image`` is the legacy single-image shorthand.
        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        enc_h_half = H_half * 32
        enc_w_half = W_half * 32
        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings_1: list = []
        if resolved_images:
            conditionings_1 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_half,
                enc_w=enc_w_half,
                spatial_dims=(F, H_half, W_half),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
            )
        conditionings_1 = [
            *conditionings_1,
            *generated_keyframe_conditionings(generated_keyframes, num_frames, frame_rate=frame_rate),
        ]

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings_1,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H_half, W_half),  # unused
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        stage1_table = LTX_2_5_DISTILLED_SIGMAS if self._is_25 else DISTILLED_SIGMAS
        sigmas_1 = shorten_schedule(stage1_table, stage1_steps, keep="head")

        stage1_dit = self.dit
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_1 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_half, W_half))
            stage1_dit = TiledLTXModel(self.dit, tiler_1)

        x0_model = X0Model(stage1_dit)

        self._pre_denoise_flush(video_state, audio_state)
        output_1 = self._run_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
            video_cross_attention_mask=relay_mask(F, H_half, W_half, video_state.latent.shape[1]),
            on_step=self._stepwise_hook(F, H_half, W_half, stage=1),
            seed=seed,
            ancestral=self._is_25,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Upscale (same denorm/upsample/renorm as TI2VidTwoStagesPipeline) ---
        # Strip appended keyframe tokens (multi-anchor with frame_idx>0).
        self.generated_keyframes = extract_generated_keyframes(
            output_1.video_latent, video_state.generated_keyframe_layout, self.video_patchifier, (H_half, W_half)
        )
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]

        return (
            Stage1Result(
                video_tokens=gen_tokens_1,
                audio_tokens=output_1.audio_latent,
                generated_keyframes=self.generated_keyframes,
                latent_dims=(F, H_half, W_half),
                resolved_images=resolved_images,
                video_embeds=video_embeds,
                audio_embeds=audio_embeds,
                relay_mask=relay_mask,
                x0_model=x0_model,
            ),
            num_frames,
            height,
            width,
        )

    def _upsample_latent(self, video_half: mx.array) -> mx.array:
        """Denormalize, spatially upsample 2x, and renormalize a stage-1 latent.

        Args:
            video_half: Stage 1 video latent, `(1, 128, F, H_half, W_half)`.

        Returns:
            Upscaled video latent, `(1, 128, F, 2*H_half, 2*W_half)`, materialized.
        """
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(video_upscaled)
        return video_upscaled

    def _stage2(
        self,
        stage1: Stage1Result,
        video_upscaled: mx.array,
        *,
        num_frames: int,
        frame_rate: float,
        seed: int,
        stage2_steps: int | None,
        extra_conditionings: list,
    ) -> tuple[mx.array, mx.array]:
        """Denoise stage 2 at full resolution and unpatchify the result.

        Args:
            stage1: Result of :meth:`_stage1`.
            video_upscaled: Upscaled video latent from :meth:`_upsample_latent`.
            num_frames: Resolved number of frames.
            frame_rate: Video frame rate.
            seed: Random seed.
            stage2_steps: Stage 2 steps (default: full stage-2 sigma table).
            extra_conditionings: Additional conditioning items appended after the
                re-encoded I2V anchors (e.g. generated-keyframe slots in later tasks).

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        F, H_half, W_half = stage1.latent_dims
        resolved_images = stage1.resolved_images
        video_embeds = stage1.video_embeds
        audio_embeds = stage1.audio_embeds
        relay_mask = stage1.relay_mask
        x0_model = stage1.x0_model

        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution (re-encode at upscaled dims)
        conditionings_2: list = []
        if resolved_images:
            from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings

            enc_h_full = H_full * 32
            enc_w_full = W_full * 32
            conditionings_2 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_full,
                enc_w=enc_w_full,
                spatial_dims=(F, H_full, W_full),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
            )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: full resolution refine (no LoRA swap — already distilled) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)
        stage2_table = LTX_2_5_STAGE_2_DISTILLED_SIGMAS if self._is_25 else STAGE_2_SIGMAS
        sigmas_2 = shorten_schedule(stage2_table, stage2_steps, keep="tail")
        # Upstream renoises the upscaled stage-1 latent at ``stage_2_sigmas[0]``
        # (``ModalitySpec(noise_scale=stage_2_sigmas[0].item())``); our
        # ``create_noised_state(sigma=...)`` below is that same mechanism.
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=[*conditionings_2, *extra_conditionings],
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )

        audio_tokens_1 = stage1.audio_tokens
        audio_state_2 = create_noised_state(
            base_shape=audio_tokens_1.shape,
            conditionings=[],
            spatial_dims=(F, H_full, W_full),  # unused
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=audio_tokens_1,
        )

        stage2_x0_model = x0_model
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_2 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_full, W_full))
            stage2_x0_model = X0Model(TiledLTXModel(self.dit, tiler_2))

        self._pre_denoise_flush(video_state_2, audio_state_2)
        output_2 = self._run_denoise_loop(
            model=stage2_x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
            video_cross_attention_mask=relay_mask(F, H_full, W_full, video_state_2.latent.shape[1]),
            on_step=self._stepwise_hook(F, H_full, W_full, stage=2),
            seed=seed,
            # Deterministic on every pack, 2.5 included (see _run_denoise_loop).
            ancestral=False,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Stage-2 slot content, extracted before the conditioning tokens are cut. Only
        # assigned when stage 2 actually carried slots (``extra_conditionings`` from
        # :class:`DFRPipeline`): the parent's stage 2 has none, and its
        # ``self.generated_keyframes`` from stage 1 must not be clobbered with ``None``.
        if video_state_2.generated_keyframe_layout is not None:
            self.generated_keyframes = extract_generated_keyframes(
                output_2.video_latent,
                video_state_2.generated_keyframe_layout,
                self.video_patchifier,
                (H_full, W_full),
            )

        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent


__all__ = ["DistilledPipeline"]
