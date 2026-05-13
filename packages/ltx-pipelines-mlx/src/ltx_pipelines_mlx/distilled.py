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

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)

from .scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from .ti2vid_two_stages import TI2VidTwoStagesPipeline
from .utils.helpers import create_noised_state
from .utils.samplers import denoise_loop

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern


class DistilledPipeline(TI2VidTwoStagesPipeline):
    """Distilled two-stage T2V/I2V pipeline (half-res → upscale → full-res refine).

    Reuses :class:`TI2VidTwoStagesPipeline`'s upsampler loading and helpers but
    overrides ``generate_two_stage`` to:

    - Skip negative-prompt encoding (no CFG).
    - Load the distilled transformer directly (no dev model, no LoRA fusion).
    - Run simple ``denoise_loop`` with ``DISTILLED_SIGMAS`` for stage 1.
    - Run the same distilled transformer for stage 2 with ``STAGE_2_SIGMAS``.

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
                transformer_path = self.model_dir / "transformer-distilled.safetensors"
            self.dit = self._load_transformer_with_optional_streaming(transformer_path)

            pending_loras = getattr(self, "_pending_loras", None)
            if pending_loras:
                # Post-load LoRA fusion for T2V (generate_two_stage calls self.load()).
                # Mirrors the identical block in generate_from_image — see that method
                # for the full rationale (pre-load fusion causes >30 GB peak OOM;
                # block-by-block materialisation pre-compiles Metal kernels before
                # the first denoise command buffer).
                from ltx_core_mlx.loader import (
                    LoraStateDictWithStrength as _LoraSDWS,
                    SafetensorsStateDictLoader as _SftLoader,
                    StateDict as _StateDict,
                    apply_loras as _apply_loras_fn,
                )
                from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP as _LORA_MAP
                from ltx_core_mlx.utils.weights import apply_quantization as _apply_q_fn
                import mlx.utils as _mu_lora

                _model_weights = dict(_mu_lora.tree_flatten(self.dit.parameters()))
                _model_sd = _StateDict(sd=_model_weights, size=0, dtype=set())
                _loader = _SftLoader()
                _lora_sds = []
                for _lpath, _lstr in pending_loras:
                    _lora_sd = _loader.load(_lpath, sd_ops=_LORA_MAP)
                    _lora_sds.append(_LoraSDWS(state_dict=_lora_sd, strength=_lstr))
                    print(f"  Fusing LoRA into loaded DiT (T2V): {_lpath.split('/')[-1]} (strength={_lstr:.2f})")
                _fused_sd = _apply_loras_fn(model_sd=_model_sd, lora_sd_and_strengths=_lora_sds)
                _apply_q_fn(self.dit, _fused_sd.sd)
                self.dit.load_weights(list(_fused_sd.sd.items()))
                aggressive_cleanup()
                import mlx.utils as _mu_pre
                if hasattr(self.dit, 'transformer_blocks'):
                    for _blk in self.dit.transformer_blocks:
                        _blk_ps = [v for _, v in _mu_pre.tree_flatten(_blk.parameters())]
                        _materialize(*_blk_ps)
                        del _blk_ps
                        aggressive_cleanup()
                    _non_blk = [v for k, v in _mu_pre.tree_flatten(self.dit.parameters())
                                if not k.startswith('transformer_blocks.')]
                    if _non_blk:
                        _materialize(*_non_blk)
                        aggressive_cleanup()
                    del _non_blk

        self._load_vae_encoder()

        if self.upsampler is None:
            self._load_upsampler()

        self._loaded = True

    def generate_two_stage(  # type: ignore[override]
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        image: str | None = None,
        images=None,
        **_unused_kwargs,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using the distilled two-stage pipeline.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Stage 1 steps (default: full DISTILLED_SIGMAS = 8).
            stage2_steps: Stage 2 steps (default: full STAGE_2_SIGMAS = 3).
            image: Optional reference image for I2V conditioning.
            **_unused_kwargs: Accepted (and ignored) for signature compatibility
                with :meth:`TI2VidTwoStagesPipeline.generate_two_stage`. CFG / STG /
                TeaCache flags don't apply to the distilled flow.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        # --- Text encoding (positive only — no CFG) ---
        self._load_text_encoder()
        video_embeds, audio_embeds = self._encode_text(prompt)
        _materialize(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        # --- Load distilled DiT + VAE encoder + upsampler ---
        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # --- Stage 1: half resolution ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
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
            )

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

        # Flush the initial noised state before the denoise loop. Without this,
        # the Metal command buffer for the first DiT block includes the full
        # state-init graph on top of the block's computation — enough to trip the
        # macOS GPU watchdog (MTLCommandBufferErrorInternal code 14) on M2 Max 64 GB.
        _materialize(video_state.latent, video_state.clean_latent, audio_state.latent)

        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS

        stage1_dit = self.dit
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_1 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_half, W_half))
            stage1_dit = TiledLTXModel(self.dit, tiler_1)

        x0_model = X0Model(stage1_dit)

        output_1 = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Upscale (same denorm/upsample/renorm as TI2VidTwoStagesPipeline) ---
        # Strip appended keyframe tokens (multi-anchor with frame_idx>0).
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution (re-encode at upscaled dims)
        conditionings_2: list = []
        if resolved_images:
            enc_h_full = H_full * 32
            enc_w_full = W_full * 32
            conditionings_2 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_full,
                enc_w=enc_w_full,
                spatial_dims=(F, H_full, W_full),
                video_encoder=self.vae_encoder,
            )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: full resolution refine (no LoRA swap — already distilled) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full)

        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=conditionings_2,
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )

        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = create_noised_state(
            base_shape=audio_tokens_1.shape,
            conditionings=[],
            spatial_dims=(F, H_full, W_full),  # unused
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=audio_tokens_1,
        )

        # Same watchdog guard as stage 1 — flush upscaled video tokens + state
        # before stage 2 denoise loop.
        _materialize(video_state_2.latent, video_state_2.clean_latent, audio_state_2.latent)

        stage2_x0_model = x0_model
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_2 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_full, W_full))
            stage2_x0_model = X0Model(TiledLTXModel(self.dit, tiler_2))

        output_2 = denoise_loop(
            model=stage2_x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
        )
        if self.low_memory:
            aggressive_cleanup()

        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent

    def generate_from_image(
        self,
        prompt: str,
        image: str | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
        images=None,
        **_unused_kwargs,
    ) -> tuple[mx.array, mx.array]:
        """Single-stage I2V at full target resolution using the distilled model.

        Mirrors the removed ``ImageToVideoPipeline.generate_from_image``.
        Runs at the full target resolution rather than half-res like
        :meth:`generate_two_stage`, which keeps per-token I2V conditioning
        inside the distilled model's training distribution and avoids the
        tiling artefacts produced by conditioning at half-resolution spatial
        dimensions (H=7, W=10 at 640×480 quick quality).

        Args:
            prompt: Text prompt.
            height: Target video height.
            width: Target video width.
            num_frames: Number of frames.
            seed: Random seed.
            num_steps: Denoising steps (default: full DISTILLED_SIGMAS = 8).
            image: Optional reference image path for I2V conditioning.
            images: Optional list of :class:`ImageConditioningInput` (multi-anchor).
            **_unused_kwargs: Absorbs extra kwargs for call-site compatibility.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        # --- Text encoding (positive only — no CFG) ---
        self._load_text_encoder()
        video_embeds, audio_embeds = self._encode_text(prompt)
        _materialize(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        # VAE encoder is always needed to condition on the input image.
        self._load_vae_encoder()
        assert self.vae_encoder is not None

        # --- Load distilled DiT (upsampler not needed here) ---
        if self.dit is None:
            transformer_path = self.model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = self.model_dir / "transformer-distilled.safetensors"
            self.dit = self._load_transformer_with_optional_streaming(transformer_path)

            pending_loras = getattr(self, "_pending_loras", None)
            if pending_loras:
                # Post-load fusion — mirrors ICLoraPipeline._fuse_loras().
                #
                # generate_from_image bypasses self.load() so _pending_loras would
                # otherwise be silently ignored. Pre-load fusion (load safetensors →
                # fuse → quantize → load weights) requires Metal to hold raw lazy
                # arrays + float32 dequantize intermediates + fused output simultaneously
                # (~triple the transformer size), causing MetalAllocator::malloc failures.
                #
                # Post-load fusion fuses into already-evaluated, already-resident weights.
                # Only 2-block worth of float32 dequantize intermediates exist at any
                # time because _DIT_EVAL_EVERY=min(2, orig) below drives incremental
                # evaluation during the denoise loop.
                from ltx_core_mlx.loader import (
                    LoraStateDictWithStrength as _LoraSDWS,
                    SafetensorsStateDictLoader as _SftLoader,
                    StateDict as _StateDict,
                    apply_loras as _apply_loras_fn,
                )
                from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP as _LORA_MAP
                from ltx_core_mlx.utils.weights import apply_quantization as _apply_q_fn
                import mlx.utils as _mu_lora

                _model_weights = dict(_mu_lora.tree_flatten(self.dit.parameters()))
                _model_sd = _StateDict(sd=_model_weights, size=0, dtype=set())
                _loader = _SftLoader()
                _lora_sds = []
                for _lpath, _lstr in pending_loras:
                    _lora_sd = _loader.load(_lpath, sd_ops=_LORA_MAP)
                    _lora_sds.append(_LoraSDWS(state_dict=_lora_sd, strength=_lstr))
                    print(f"  Fusing LoRA into loaded DiT: {_lpath.split('/')[-1]} (strength={_lstr:.2f})")
                _fused_sd = _apply_loras_fn(model_sd=_model_sd, lora_sd_and_strengths=_lora_sds)
                _apply_q_fn(self.dit, _fused_sd.sd)
                self.dit.load_weights(list(_fused_sd.sd.items()))
                aggressive_cleanup()
                # Pre-materialise fused weights one transformer block at a time.
                # Evaluating all 48 blocks at once holds O(48×8 layers) float32
                # dequantize intermediates simultaneously (>30 GB peak) →
                # MetalAllocator::malloc failure. Per-block with cleanup: ~2 GB
                # peak — safe on 64 GB. Also pre-compiles Metal dequantize/add/
                # requantize kernels so they don't land in the first denoise
                # command buffer (cold-compile + forward-pass → watchdog).
                import mlx.utils as _mu_pre
                if hasattr(self.dit, 'transformer_blocks'):
                    for _blk in self.dit.transformer_blocks:
                        _blk_ps = [v for _, v in _mu_pre.tree_flatten(_blk.parameters())]
                        _materialize(*_blk_ps)
                        del _blk_ps
                        aggressive_cleanup()
                    _non_blk = [v for k, v in _mu_pre.tree_flatten(self.dit.parameters())
                                if not k.startswith('transformer_blocks.')]
                    if _non_blk:
                        _materialize(*_non_blk)
                        aggressive_cleanup()
                    del _non_blk
        assert self.dit is not None

        # Pre-materialise DiT weights in their own command buffer (non-LoRA path).
        # LoRA path does block-by-block materialisation above.
        if not getattr(self, "_pending_loras", None):
            import mlx.utils as _mu
            _materialize(*[v for _, v in _mu.tree_flatten(self.dit.parameters())])

        # --- Full-resolution latent shape ---
        F, H, W = compute_video_latent_shape(num_frames, height, width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        # --- I2V conditioning at full resolution ---
        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        enc_h = H * 32
        enc_w = W * 32
        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings: list = []
        if resolved_images:
            conditionings = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h,
                enc_w=enc_w,
                spatial_dims=(F, H, W),
                video_encoder=self.vae_encoder,
            )
            # Flush VAE encoding into its own Metal command buffer before
            # building the noise state. Without this flush the lazy VAE
            # encode graph accumulates with create_noised_state into one
            # oversized buffer that trips the Metal watchdog on cold start.
            for _cond in conditionings:
                if hasattr(_cond, 'clean_latent') and _cond.clean_latent is not None:
                    _materialize(_cond.clean_latent)

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings,
            spatial_dims=(F, H, W),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H, W),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        # Flush initial state before denoise loop (Metal watchdog guard — same
        # rationale as generate_two_stage stage 1 flush above).
        _materialize(video_state.latent, video_state.clean_latent, audio_state.latent)

        sigmas = DISTILLED_SIGMAS[: num_steps + 1] if num_steps else DISTILLED_SIGMAS

        dit = self.dit
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler = VideoModalityTiler(self._tile_count, latent_shape=(F, H, W))
            dit = TiledLTXModel(self.dit, tiler)

        x0_model = X0Model(dit)

        # Tighten the DiT eval cadence for the full-res pass. The default
        # LTX2_DIT_EVAL_EVERY=8 is calibrated for ~1120 tokens (half-res).
        # At 4800 tokens (full-res 640×480) each 8-block group takes
        # ~4–16× longer (O(N^1.5) scaling) → exceeds the 10-second Metal
        # watchdog. Using 2 on cold-start Metal keeps each buffer well under
        # the watchdog even when Metal kernel compilation overhead applies.
        import ltx_core_mlx.model.transformer.model as _dit_mod
        _orig_eval_every = _dit_mod._DIT_EVAL_EVERY
        _dit_mod._DIT_EVAL_EVERY = min(2, _orig_eval_every) if _orig_eval_every > 0 else _orig_eval_every
        try:
            output = denoise_loop(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas,
            )
        finally:
            _dit_mod._DIT_EVAL_EVERY = _orig_eval_every
        if self.low_memory:
            aggressive_cleanup()

        # Strip any appended keyframe tokens (multi-anchor frame_idx>0 conditioning).
        gen_tokens = output.video_latent[:, : F * H * W, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent


__all__ = ["DistilledPipeline"]
