"""Retake pipeline — regenerate a time segment of an existing video.

Ported from ltx-pipelines retake functionality.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, TemporalRegionMask, noise_latent_state
from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder, encode_audio
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.image import load_video_frames
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop


class RetakePipeline(TextToVideoPipeline):
    """Retake pipeline: regenerate a time segment while preserving the rest.

    Args:
        model_dir: Path to model weights.
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
        self.audio_encoder: AudioVAEEncoder | None = None
        self.audio_processor: AudioProcessor | None = None

    def _load_encoders(self) -> None:
        """Load VAE encoder and audio encoder for video-from-file workflows."""
        if self.vae_encoder is None:
            self.vae_encoder = VideoEncoder()
            enc_weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
            enc_weights = {
                k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
                for k, v in enc_weights.items()
            }
            self.vae_encoder.load_weights(list(enc_weights.items()))
            aggressive_cleanup()

        if self.audio_encoder is None:
            self.audio_encoder = AudioVAEEncoder()
            encoder_weights = load_split_safetensors(
                self.model_dir / "audio_vae.safetensors", prefix="audio_vae.encoder."
            )
            # Also load per_channel_statistics (lives under audio_vae., not audio_vae.encoder.)
            all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    encoder_weights[k] = v
            encoder_weights = remap_audio_vae_keys(encoder_weights)
            self.audio_encoder.load_weights(list(encoder_weights.items()))
            self.audio_processor = AudioProcessor()
            aggressive_cleanup()

    def retake_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video file.

        Convenience wrapper that loads the video, encodes it to latents,
        and calls :meth:`retake`.

        Args:
            prompt: Text prompt for the regenerated segment.
            video_path: Path to the source video file.
            start_frame: First frame to regenerate (inclusive, in latent space).
            end_frame: Last frame to regenerate (exclusive, in latent space).
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        self.load()
        self._load_encoders()
        assert self.vae_encoder is not None

        video_path = str(video_path)
        info = probe_video_info(video_path)

        # Load video frames via ffmpeg -> (B, 3, F, H, W)
        video_tensor = load_video_frames(video_path, info.height, info.width, info.num_frames)
        video_latent = self.vae_encoder.encode(video_tensor)
        if self.low_memory:
            del video_tensor
            aggressive_cleanup()

        # Encode audio if present
        audio_latent: mx.array | None = None
        if info.has_audio:
            assert self.audio_encoder is not None
            assert self.audio_processor is not None
            audio_data = load_audio(
                video_path,
                target_sample_rate=16000,
                max_duration=info.num_frames / info.fps,
            )
            if audio_data is not None:
                audio_latent = encode_audio(
                    audio_data.waveform,
                    audio_data.sample_rate,
                    self.audio_encoder,
                    self.audio_processor,
                )
                if self.low_memory:
                    aggressive_cleanup()

        if audio_latent is None:
            # Create silence audio latent
            audio_T = compute_audio_token_count(info.num_frames)
            audio_latent = mx.zeros((1, 8, audio_T, 16), dtype=mx.bfloat16)

        # Free encoders to save memory
        if self.low_memory:
            self.vae_encoder = None
            self.audio_encoder = None
            self.audio_processor = None
            aggressive_cleanup()

        return self.retake(
            prompt=prompt,
            source_video_latent=video_latent,
            source_audio_latent=audio_latent,
            start_frame=start_frame,
            end_frame=end_frame,
            height=info.height,
            width=info.width,
            num_frames=info.num_frames,
            seed=seed,
            num_steps=num_steps,
        )

    def retake(
        self,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        start_frame: int,
        end_frame: int,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video.

        Args:
            prompt: Text prompt for the regenerated segment.
            source_video_latent: (B, C, F, H, W) source video latent.
            source_audio_latent: (B, 8, T, 16) source audio latent.
            start_frame: First frame to regenerate (inclusive, in latent space).
            end_frame: Last frame to regenerate (exclusive, in latent space).
            height: Video height.
            width: Video width.
            num_frames: Total number of frames.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        self.load()
        assert self.dit is not None

        F, H, W = compute_video_latent_shape(num_frames, height, width)
        tokens_per_frame = H * W

        # Patchify source
        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)

        # Compute positions for RoPE
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        # Create video state with temporal mask
        region = TemporalRegionMask(start_frame, end_frame)
        denoise_mask = region.create_mask(F, tokens_per_frame)

        video_state = LatentState(
            latent=source_tokens,
            clean_latent=source_tokens,
            denoise_mask=denoise_mask,
            positions=video_positions,
        )
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)

        # Audio state: apply same temporal mask to audio
        audio_tokens_per_video_frame = audio_T / F
        audio_start = round(start_frame * audio_tokens_per_video_frame)
        audio_end = round(end_frame * audio_tokens_per_video_frame)

        audio_mask = mx.zeros((1, audio_T, 1), dtype=mx.bfloat16)
        audio_mask = audio_mask.at[:, audio_start:audio_end, :].add(
            mx.ones((1, audio_end - audio_start, 1), dtype=mx.bfloat16)
        )

        audio_state = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=audio_mask,
            positions=audio_positions,
        )
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # Encode text
        video_embeds, audio_embeds = self._encode_text(prompt)
        if self.low_memory:
            aggressive_cleanup()

        # Denoise
        sigmas = DISTILLED_SIGMAS[: num_steps + 1] if num_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
