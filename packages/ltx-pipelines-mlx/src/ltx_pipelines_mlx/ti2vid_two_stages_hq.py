"""HQ two-stage pipeline — generate with res_2s sampler, upscale, then refine.

Ported from ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py

Uses the res_2s second-order sampler in stage 1 for higher quality at
fewer steps, and distilled schedule for stage 2 refinement.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from PIL import Image

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
    create_initial_state,
    noise_latent_state,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import load_split_safetensors
from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop, res2s_denoise_loop


class TwoStageHQPipeline(TextToVideoPipeline):
    """HQ two-stage generation with res_2s sampler.

    Stage 1: Generate at half spatial resolution using the res_2s second-order
    sampler for higher quality.
    Stage 2: Upscale latents 2x with neural upsampler, then refine with
    distilled schedule.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        low_memory: Aggressive memory management.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
    ):
        super().__init__(model_dir, gemma_model_id=gemma_model_id, low_memory=low_memory)
        self.vae_encoder: VideoEncoder | None = None
        self.upsampler: LatentUpsampler | None = None

    def load(self) -> None:
        """Load DiT + VAE encoder + upsampler (skip decoders for memory).

        Decoders (VAE decoder, audio, vocoder) are loaded on-demand in
        ``generate_and_save()`` after the transformer is freed. This keeps
        peak memory under 32 GB on Apple Silicon.
        """
        if self._loaded:
            return

        model_dir = self.model_dir

        # Text encoder + connector (loaded first, freed before DiT)
        if self.text_encoder is None:
            from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

            self.text_encoder = GemmaLanguageModel()
            self.text_encoder.load(self._gemma_model_id)
            aggressive_cleanup()

        if self.feature_extractor is None:
            from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2

            self.feature_extractor = GemmaFeaturesExtractorV2()
            connector_weights = load_split_safetensors(model_dir / "connector.safetensors", prefix="connector.")
            self.feature_extractor.connector.load_weights(list(connector_weights.items()))
            aggressive_cleanup()

        # DiT
        if self.dit is None:
            from ltx_core_mlx.model.transformer.model import LTXModel
            from ltx_core_mlx.utils.weights import apply_quantization

            if self.low_memory:
                self.text_encoder = None
                aggressive_cleanup()

            self.dit = LTXModel()
            transformer_path = model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = model_dir / "transformer-distilled.safetensors"
            transformer_weights = load_split_safetensors(transformer_path, prefix="transformer.")
            apply_quantization(self.dit, transformer_weights)
            self.dit.load_weights(list(transformer_weights.items()))
            aggressive_cleanup()

        # VAE encoder (needed for I2V conditioning + denorm/renorm)
        if self.vae_encoder is None:
            self.vae_encoder = VideoEncoder()
            enc_weights = load_split_safetensors(model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
            enc_weights = {
                k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
                for k, v in enc_weights.items()
            }
            self.vae_encoder.load_weights(list(enc_weights.items()))
            aggressive_cleanup()

        # Upsampler
        if self.upsampler is None:
            import json

            name = "spatial_upscaler_x2_v1_1"
            config_path = model_dir / f"{name}_config.json"
            weights_path = model_dir / f"{name}.safetensors"
            if config_path.exists():
                config = json.loads(config_path.read_text()).get("config", {})
                self.upsampler = LatentUpsampler.from_config(config)
            else:
                self.upsampler = LatentUpsampler()
            if weights_path.exists():
                weights = load_split_safetensors(weights_path, prefix=f"{name}.")
                self.upsampler.load_weights(list(weights.items()))
            aggressive_cleanup()

        self._loaded = True

    def _load_decoders(self) -> None:
        """Load decoders on-demand (VAE decoder, audio decoder, vocoder)."""
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder

        model_dir = self.model_dir

        if self.vae_decoder is None:
            self.vae_decoder = VideoDecoder()
            vae_weights = load_split_safetensors(model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
            self.vae_decoder.load_weights(list(vae_weights.items()))
            aggressive_cleanup()

        if self.audio_decoder is None:
            from ltx_core_mlx.utils.weights import remap_audio_vae_keys

            self.audio_decoder = AudioVAEDecoder()
            audio_weights = load_split_safetensors(model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder.")
            all_audio = load_split_safetensors(model_dir / "audio_vae.safetensors", prefix="audio_vae.")
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    audio_weights[k] = v
            audio_weights = remap_audio_vae_keys(audio_weights)
            self.audio_decoder.load_weights(list(audio_weights.items()))
            aggressive_cleanup()

        if self.vocoder is None:
            self.vocoder = VocoderWithBWE()
            vocoder_weights = load_split_safetensors(model_dir / "vocoder.safetensors", prefix="vocoder.")
            self.vocoder.load_weights(list(vocoder_weights.items()))
            aggressive_cleanup()

    def generate_hq(
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int = 20,
        stage2_steps: int | None = None,
        image: Image.Image | str | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using HQ two-stage pipeline with res_2s sampler.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Steps for stage 1 (res_2s sampler, default 20).
            stage2_steps: Steps for stage 2 refinement.
            image: Optional reference image for I2V conditioning.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        # Encode text first, then load remaining components (DiT, VAE encoder, etc.).
        # In low_memory mode, _encode_text_and_load frees Gemma before loading DiT.
        video_embeds, audio_embeds = self._encode_text_and_load(prompt)
        assert self.dit is not None
        assert self.upsampler is not None
        assert self.vae_encoder is not None

        # Stage 1: Half-resolution generation with res_2s sampler
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        num_tokens = F * H_half * W_half
        video_shape = (1, num_tokens, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions_1)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Apply I2V conditioning if image provided
        if image is not None:
            assert self.vae_encoder is not None
            img_tensor = prepare_image_for_encoding(image, half_h, half_w)
            img_tensor = img_tensor[:, :, None, :, :]
            ref_latent = self.vae_encoder.encode(img_tensor)
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens)
            video_state = apply_conditioning(video_state, [condition], (F, H_half, W_half))
            if self.low_memory:
                aggressive_cleanup()

        # Dynamic sigma schedule for res_2s
        sigmas_1 = ltx2_schedule(steps=stage1_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        output_1 = res2s_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Unpatchify and upscale with denormalize/renormalize wrapping.
        # The neural upsampler operates in un-normalized latent space.
        video_half = self.video_patchifier.unpatchify(output_1.video_latent, (F, H_half, W_half))

        assert self.vae_encoder is not None
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)  # back to (B,C,F,H,W)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)  # back to (B,C,F,H,W)
        if self.low_memory:
            aggressive_cleanup()

        # Stage 2: Refine at full resolution with distilled schedule.
        # Derive dims from actual upscaled shape, not target height/width,
        # to avoid RoPE shape mismatch (H_half*2 may differ from H at target res).
        H_full = H_half * 2
        W_full = W_half * 2
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

        # I2V conditioning at full resolution for stage 2
        if image is not None:
            assert self.vae_encoder is not None
            img_tensor = prepare_image_for_encoding(image, height, width)
            img_tensor = img_tensor[:, :, None, :, :]
            ref_latent = self.vae_encoder.encode(img_tensor)
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens)
            video_state_2 = apply_conditioning(video_state_2, [condition], (F, H_full, W_full))

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

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        image: Image.Image | str | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int = 20,
        stage2_steps: int | None = None,
    ) -> str:
        """Generate HQ two-stage video+audio and save to file.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            image: Optional reference image for I2V conditioning.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Steps for stage 1 (res_2s sampler, default 20).
            stage2_steps: Steps for stage 2 refinement.

        Returns:
            Path to the output video file.
        """
        video_latent, audio_latent = self.generate_hq(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            image=image,
        )

        # Free transformer + encoder to make room for decoders
        if self.low_memory:
            self.dit = None
            self.text_encoder = None
            self.feature_extractor = None
            self.vae_encoder = None
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        # Load decoders on-demand (not loaded during generate to save memory)
        self._load_decoders()

        # Decode audio first (smaller)
        assert self.audio_decoder is not None
        assert self.vocoder is not None
        mel = self.audio_decoder.decode(audio_latent)
        waveform = self.vocoder(mel)
        if self.low_memory:
            aggressive_cleanup()

        # Save audio to temp file
        import tempfile

        audio_path = tempfile.mktemp(suffix=".wav")
        self._save_waveform(waveform, audio_path, sample_rate=48000)

        # Decode video and stream to ffmpeg
        assert self.vae_decoder is not None
        self.vae_decoder.decode_and_stream(video_latent, output_path, fps=24.0, audio_path=audio_path)

        # Cleanup temp audio
        Path(audio_path).unlink(missing_ok=True)
        aggressive_cleanup()

        return output_path
