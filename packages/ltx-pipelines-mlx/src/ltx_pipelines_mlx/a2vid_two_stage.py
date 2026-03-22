"""Audio-to-Video two-stage pipeline.

Stage 1: Generate video at half resolution conditioned on encoded audio (audio frozen).
Stage 2: Upscale video 2x and refine both modalities with distilled schedule.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, create_initial_state, noise_latent_state
from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder, encode_audio
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop


class AudioToVideoPipeline(TextToVideoPipeline):
    """Audio-to-Video two-stage generation pipeline.

    Stage 1: Generate video at half spatial resolution with audio frozen.
    Stage 2: Upscale latents 2x, then refine both video and audio.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        low_memory: Aggressive memory management.
    """

    def __init__(self, model_dir: str, low_memory: bool = True):
        super().__init__(model_dir, low_memory=low_memory)
        self.audio_encoder: AudioVAEEncoder | None = None
        self.audio_processor: AudioProcessor | None = None
        self.upsampler: LatentUpsampler | None = None

    def load(self) -> None:
        """Load all components including audio encoder and upsampler."""
        super().load()

        if self.audio_encoder is None:
            self.audio_encoder = AudioVAEEncoder()
            encoder_weights = load_split_safetensors(
                self.model_dir / "audio_vae.safetensors",
                prefix="audio_vae.encoder.",
            )
            # Also load per_channel_statistics (shared, not under encoder. prefix)
            all_audio = load_split_safetensors(
                self.model_dir / "audio_vae.safetensors",
                prefix="audio_vae.",
            )
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    encoder_weights[k] = v
            encoder_weights = remap_audio_vae_keys(encoder_weights)
            self.audio_encoder.load_weights(list(encoder_weights.items()))
            aggressive_cleanup()

            self.audio_processor = AudioProcessor()

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

    def generate(
        self,
        prompt: str,
        audio_path: str | Path,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        fps: float = 24.0,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video from audio + text prompt using two-stage pipeline.

        Args:
            prompt: Text description of the desired video.
            audio_path: Path to audio file.
            height: Output video height in pixels.
            width: Output video width in pixels.
            num_frames: Number of output video frames.
            fps: Frame rate.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1 (default: full schedule).
            stage2_steps: Denoising steps for stage 2 (default: full schedule).
            audio_start_time: Start time in seconds for audio.
            audio_max_duration: Max audio duration (defaults to video duration).

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        self.load()
        assert self.dit is not None
        assert self.audio_encoder is not None
        assert self.audio_processor is not None
        assert self.upsampler is not None

        if audio_max_duration is None:
            audio_max_duration = num_frames / fps

        # --- Load and encode audio ---
        audio_data = load_audio(
            audio_path,
            target_sample_rate=16000,
            start_time=audio_start_time,
            max_duration=audio_max_duration,
        )
        if audio_data is None:
            raise ValueError(f"No audio found in {audio_path}")

        audio_latent = encode_audio(
            audio_data.waveform,
            audio_data.sample_rate,
            self.audio_encoder,
            self.audio_processor,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Patchify audio latent to tokens
        audio_T = compute_audio_token_count(num_frames, fps)
        # Trim to expected length
        audio_latent = audio_latent[:, :, :audio_T, :]
        audio_tokens, _ = self.audio_patchifier.patchify(audio_latent)  # (1, audio_T, 128)

        # --- Encode text ---
        video_embeds, audio_embeds = self._encode_text(prompt)
        if self.low_memory:
            aggressive_cleanup()

        # --- Stage 1: Half-res video generation, audio frozen ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, fps)
        audio_positions = compute_audio_positions(audio_T)

        video_state_1 = create_initial_state(video_shape, seed, positions=video_positions_1)

        # Audio frozen in stage 1 (denoise_mask=0 = preserve)
        audio_state_1 = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=mx.zeros((1, audio_tokens.shape[1], 1), dtype=mx.bfloat16),
            positions=audio_positions,
        )

        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        from ltx_core_mlx.model.transformer.model import X0Model

        x0_model = X0Model(self.dit)

        output_1 = denoise_loop(
            model=x0_model,
            video_state=video_state_1,
            audio_state=audio_state_1,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Upscale video latents 2x ---
        video_half = self.video_patchifier.unpatchify(output_1.video_latent, (F, H_half, W_half))
        video_upscaled = self.upsampler(video_half)
        if self.low_memory:
            aggressive_cleanup()

        # --- Stage 2: Refine at full resolution ---
        _, H_full, W_full = compute_video_latent_shape(num_frames, height, width)
        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        # Add noise to upscaled video
        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens_up.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens_up * (1.0 - start_sigma)

        video_positions_2 = compute_video_positions(F, H_full, W_full, fps)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens_up,
            denoise_mask=mx.ones((1, video_tokens_up.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        # Audio gets noised and refined in stage 2
        audio_state_2 = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=mx.ones((1, audio_tokens.shape[1], 1), dtype=audio_tokens.dtype),
            positions=audio_positions,
        )
        audio_state_2 = noise_latent_state(audio_state_2, sigma=start_sigma, seed=seed + 3)

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

        # Unpatchify outputs
        video_latent = self.video_patchifier.unpatchify(output_2.video_latent, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        audio_path: str | Path | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        fps: float = 24.0,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
    ) -> str:
        """Generate video from audio and save to file.

        Uses the original input audio for the output (not VAE-decoded audio)
        for maximum fidelity.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            audio_path: Path to input audio file.
            height: Video height.
            width: Video width.
            num_frames: Number of frames.
            fps: Frame rate.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1.
            stage2_steps: Denoising steps for stage 2.
            audio_start_time: Start time in seconds for audio.
            audio_max_duration: Max audio duration.

        Returns:
            Path to the output video file.

        Raises:
            ValueError: If audio_path is not provided or has no audio.
        """
        if audio_path is None:
            raise ValueError("audio_path is required for AudioToVideoPipeline")

        video_latent, _audio_latent = self.generate(
            prompt=prompt,
            audio_path=audio_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
        )

        # Extract original audio segment to temp WAV for muxing
        import tempfile

        audio_data = load_audio(
            audio_path,
            target_sample_rate=48000,
            start_time=audio_start_time,
            max_duration=audio_max_duration if audio_max_duration else num_frames / fps,
        )
        if audio_data is not None:
            temp_audio = tempfile.mktemp(suffix=".wav")
            self._save_waveform(audio_data.waveform, temp_audio, sample_rate=48000)
        else:
            temp_audio = None

        # Decode video and stream to ffmpeg with original audio
        assert self.vae_decoder is not None
        self.vae_decoder.decode_and_stream(
            video_latent,
            output_path,
            fps=fps,
            audio_path=temp_audio,
        )

        # Cleanup temp audio
        if temp_audio is not None:
            Path(temp_audio).unlink(missing_ok=True)
        aggressive_cleanup()

        return output_path
