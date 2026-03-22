"""IC-LoRA pipeline — reference video conditioning with two-stage generation.

Ported from ltx-pipelines/src/ltx_pipelines/ic_lora.py

Generates video conditioned on a reference video (e.g., depth maps, poses,
edges) using IC-LoRA style reference latent conditioning.
"""

from __future__ import annotations

import mlx.core as mx
from PIL import Image

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.attention_strength_wrapper import (
    ConditioningItemAttentionStrengthWrapper,
)
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
    create_initial_state,
    noise_latent_state,
)
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.loader import (
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraStateDictWithStrength,
    SafetensorsStateDictLoader,
    StateDict,
    apply_loras,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import load_split_safetensors
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop


class ICLoraPipeline(TextToVideoPipeline):
    """Two-stage video generation pipeline with IC-LoRA reference conditioning.

    Conditions the generated video on a reference video (e.g., depth, pose, edges)
    via VideoConditionByReferenceLatent. Stage 1 generates at half resolution,
    then Stage 2 upscales and refines.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        low_memory: Aggressive memory management.
        reference_downscale_factor: Target/reference resolution ratio.
            Use 2 if IC-LoRA was trained with half-resolution references.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        reference_downscale_factor: int = 1,
    ):
        super().__init__(model_dir, gemma_model_id=gemma_model_id, low_memory=low_memory)
        self.vae_encoder: VideoEncoder | None = None
        self.upsampler: LatentUpsampler | None = None
        self.reference_downscale_factor = reference_downscale_factor

    def load(self) -> None:
        """Load all components including VAE encoder and upsampler."""
        super().load()

        if self.vae_encoder is None:
            self.vae_encoder = VideoEncoder()
            enc_weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
            enc_weights = {
                k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
                for k, v in enc_weights.items()
            }
            self.vae_encoder.load_weights(list(enc_weights.items()))
            aggressive_cleanup()

        if self.upsampler is None:
            import json

            name = "spatial_upscaler_x2_v1_1"
            config_path = self.model_dir / f"{name}_config.json"
            weights_path = self.model_dir / f"{name}.safetensors"
            if config_path.exists():
                config = json.loads(config_path.read_text()).get("config", {})
                self.upsampler = LatentUpsampler.from_config(config)
            else:
                self.upsampler = LatentUpsampler()
            if weights_path.exists():
                weights = load_split_safetensors(weights_path, prefix=f"{name}.")
                self.upsampler.load_weights(list(weights.items()))
            aggressive_cleanup()

    def load_lora(
        self,
        lora_path: str,
        strength: float = 1.0,
    ) -> None:
        """Load and fuse LoRA weights into the transformer.

        Must be called after load(). Dequantizes affected weights, fuses the
        LoRA delta, and re-quantizes.

        Args:
            lora_path: Path to a safetensors file with LoRA weights (ComfyUI format).
            strength: LoRA strength multiplier. Default 1.0.
        """
        self.load()
        assert self.dit is not None

        # Flatten nested parameter dicts/lists to dot-separated keys
        def _flatten(prefix: str, d: dict | list) -> dict[str, mx.array]:
            result: dict[str, mx.array] = {}
            items = enumerate(d) if isinstance(d, list) else d.items()
            for k, v in items:
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, dict | list):
                    result.update(_flatten(key, v))
                else:
                    result[key] = v
            return result

        model_weights = _flatten("", dict(self.dit.parameters()))
        model_sd = StateDict(sd=model_weights, size=0, dtype=set())

        # Load LoRA with ComfyUI key remapping
        loader = SafetensorsStateDictLoader()
        lora_sd = loader.load(lora_path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)

        # Fuse
        fused_sd = apply_loras(
            model_sd=model_sd,
            lora_sd_and_strengths=[LoraStateDictWithStrength(state_dict=lora_sd, strength=strength)],
        )

        # Load fused weights back into model
        from ltx_core_mlx.utils.weights import apply_quantization

        apply_quantization(self.dit, fused_sd.sd)
        self.dit.load_weights(list(fused_sd.sd.items()))
        aggressive_cleanup()

    def generate(
        self,
        prompt: str,
        reference_video_latent: mx.array,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        images: list[tuple[Image.Image | str, int, float]] | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """Generate video with IC-LoRA reference conditioning.

        Args:
            prompt: Text prompt.
            reference_video_latent: Encoded reference video latent from VAE encoder,
                shape (1, C, F, H, W).
            height: Output video height.
            width: Output video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1.
            stage2_steps: Denoising steps for stage 2.
            images: Optional list of (image, frame_index, strength) for I2V conditioning.
            conditioning_attention_strength: Attention strength for reference conditioning.
                0.0 = ignore, 1.0 = full conditioning. Default 1.0.
            skip_stage_2: Skip upscale + refine, output at half resolution.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None

        # Text encoding
        video_embeds, audio_embeds = self._encode_text(prompt)
        if self.low_memory:
            aggressive_cleanup()

        # --- Stage 1: Half-resolution generation ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions_1)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Apply image conditioning (I2V) if provided
        conditionings = []
        if images:
            for img, frame_idx, strength in images:
                img_tensor = prepare_image_for_encoding(img, half_h, half_w)
                img_tensor = img_tensor[:, :, None, :, :]
                ref_latent = self.vae_encoder.encode(img_tensor)
                ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
                conditionings.append(
                    VideoConditionByLatentIndex(
                        frame_indices=[frame_idx],
                        clean_latent=ref_tokens,
                        strength=strength,
                    )
                )
        if conditionings:
            video_state = apply_conditioning(video_state, conditionings, (F, H_half, W_half))

        # Apply IC-LoRA reference video conditioning
        scale = self.reference_downscale_factor

        # Derive reference latent dims from the actual encoded shape (1, C, F', H', W')
        ref_F = reference_video_latent.shape[2]
        ref_H = reference_video_latent.shape[3]
        ref_W = reference_video_latent.shape[4]
        ref_positions = compute_video_positions(ref_F, ref_H, ref_W)

        # Patchify reference to tokens
        ref_tokens = reference_video_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)

        ref_cond = VideoConditionByReferenceLatent(
            reference_latent=ref_tokens,
            reference_positions=ref_positions,
            downscale_factor=scale,
            strength=1.0,
        )

        # Wrap with attention strength if needed
        if conditioning_attention_strength < 1.0:
            ref_cond = ConditioningItemAttentionStrengthWrapper(
                conditioning=ref_cond,
                attention_mask=conditioning_attention_strength,
            )

        video_state = ref_cond.apply(video_state, (F, H_half, W_half))

        # Denoise stage 1
        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

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

        # Extract only generation tokens (exclude appended reference tokens)
        gen_tokens = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens, (F, H_half, W_half))

        if skip_stage_2:
            audio_latent = self.audio_patchifier.unpatchify(output_1.audio_latent)
            return video_half, audio_latent

        # --- Stage 2: Upscale + refine ---
        assert self.upsampler is not None
        video_upscaled = self.upsampler(video_half)
        if self.low_memory:
            aggressive_cleanup()

        _, H_full, W_full = compute_video_latent_shape(num_frames, height, width)
        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens_up.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens_up * (1.0 - start_sigma)

        video_positions_2 = compute_video_positions(F, H_full, W_full)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens_up,
            denoise_mask=mx.ones((1, video_tokens_up.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        # Apply I2V conditioning at full resolution for stage 2
        if images:
            conditionings_2 = []
            for img, frame_idx, strength in images:
                img_tensor = prepare_image_for_encoding(img, height, width)
                img_tensor = img_tensor[:, :, None, :, :]
                ref_latent = self.vae_encoder.encode(img_tensor)
                ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
                conditionings_2.append(
                    VideoConditionByLatentIndex(
                        frame_indices=[frame_idx],
                        clean_latent=ref_tokens,
                        strength=strength,
                    )
                )
            video_state_2 = apply_conditioning(video_state_2, conditionings_2, (F, H_full, W_full))

        # Audio refined in stage 2
        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = LatentState(
            latent=audio_tokens_1,
            clean_latent=audio_tokens_1,
            denoise_mask=mx.ones((1, audio_tokens_1.shape[1], 1), dtype=audio_tokens_1.dtype),
            positions=audio_positions,
        )
        audio_state_2 = noise_latent_state(audio_state_2, sigma=start_sigma, seed=seed + 2)

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

    def encode_reference_video(
        self,
        video_frames: mx.array,
    ) -> mx.array:
        """Encode a reference video to latent space.

        Args:
            video_frames: Video tensor of shape (1, 3, F, H, W) in [0, 1].

        Returns:
            Encoded latent of shape (1, C, F', H', W').
        """
        self.load()
        assert self.vae_encoder is not None
        return self.vae_encoder.encode(video_frames)
