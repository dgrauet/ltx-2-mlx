"""HQ two-stage pipeline — res_2s second-order sampler for Stage 1.

Same architecture as TwoStagePipeline but uses the res_2s second-order sampler
instead of Euler for Stage 1 denoising, producing higher quality at fewer steps.
Supports guidance (CFG/STG) with the res_2s sampler.

Ported from ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
    create_initial_state,
    noise_latent_state,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.ti2vid_two_stages import DEFAULT_CFG_SCALE, TwoStagePipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop, res2s_denoise_loop


class TwoStageHQPipeline(TwoStagePipeline):
    """HQ two-stage generation with res_2s second-order sampler.

    Inherits from TwoStagePipeline and overrides Stage 1 to use the res_2s
    sampler for higher quality at fewer steps. Stage 2 is identical.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Dev transformer filename.
        distilled_lora: Distilled LoRA filename for Stage 2.
        distilled_lora_strength: LoRA fusion strength.
    """

    def generate_two_stage(
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int = 15,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 0.0,
        image: str | None = None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using HQ two-stage pipeline with res_2s sampler.

        Same as TwoStagePipeline.generate_two_stage but uses res_2s sampler
        for Stage 1 instead of Euler.
        """
        # --- Text encoding ---
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        # --- Load DiT + VAE encoder + upsampler ---
        if self.dit is None:
            self.dit = self._load_dev_transformer()

        self._load_vae_encoder()
        if self.upsampler is None:
            self._load_upsampler()

        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # --- Stage 1: Half resolution with res_2s sampler + guidance ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions_1)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # I2V conditioning at half resolution
        enc_h_half = H_half * 32
        enc_w_half = W_half * 32
        conditionings_1: list[VideoConditionByLatentIndex] = []
        if image is not None:
            img_tensor = prepare_image_for_encoding(image, enc_h_half, enc_w_half)
            img_tensor = img_tensor[:, :, None, :, :]
            ref_latent = self.vae_encoder.encode(img_tensor)
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            conditionings_1.append(
                VideoConditionByLatentIndex(
                    frame_indices=[0],
                    clean_latent=ref_tokens,
                    strength=1.0,
                )
            )

        if conditionings_1:
            video_state = apply_conditioning(video_state, conditionings_1, (F, H_half, W_half))
            video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)
            audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # Stage 1 sigma schedule (dynamic for dev model)
        num_tokens = F * H_half * W_half
        sigmas_1 = ltx2_schedule(stage1_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        # Build guider params (HQ defaults: no STG, lower rescale)
        if video_guider_params is None:
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                rescale_scale=0.45,
                modality_scale=3.0,
                stg_blocks=[],
            )
        if audio_guider_params is None:
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=stg_scale,
                rescale_scale=1.0,
                modality_scale=3.0,
                stg_blocks=[],
            )

        video_factory = create_multimodal_guider_factory(video_guider_params, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_guider_params, negative_context=neg_audio_embeds)

        # Stage 1: res_2s with guidance
        output_1 = res2s_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Fuse distilled LoRA for Stage 2 ---
        self._fuse_distilled_lora(self.dit)

        # --- Upscale with denormalize/renormalize ---
        video_half = self.video_patchifier.unpatchify(output_1.video_latent, (F, H_half, W_half))

        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution for Stage 2
        conditionings_2: list[VideoConditionByLatentIndex] = []
        if image is not None:
            enc_h_full = H_full * 32
            enc_w_full = W_full * 32
            img_tensor = prepare_image_for_encoding(image, enc_h_full, enc_w_full)
            img_tensor = img_tensor[:, :, None, :, :]
            ref_latent = self.vae_encoder.encode(img_tensor)
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            conditionings_2.append(
                VideoConditionByLatentIndex(
                    frame_indices=[0],
                    clean_latent=ref_tokens,
                    strength=1.0,
                )
            )

        if self.low_memory:
            self.vae_encoder = None
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: Refine at full resolution (no CFG) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens * (1.0 - start_sigma)

        video_positions_2 = compute_video_positions(F, H_full, W_full)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens,
            denoise_mask=mx.ones((1, video_tokens.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        if conditionings_2:
            video_state_2 = apply_conditioning(video_state_2, conditionings_2, (F, H_full, W_full))

        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = LatentState(
            latent=audio_tokens_1,
            clean_latent=audio_tokens_1,
            denoise_mask=mx.ones((1, audio_tokens_1.shape[1], 1), dtype=audio_tokens_1.dtype),
            positions=audio_positions,
        )
        audio_state_2 = noise_latent_state(audio_state_2, sigma=start_sigma, seed=seed + 2)

        # Stage 2: simple denoising (no CFG)
        output_2 = denoise_loop(
            model=x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output_2.video_latent, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent
