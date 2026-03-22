"""Extend pipeline — add frames before or after an existing video.

Ported from ltx-pipelines extend functionality.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
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


class ExtendPipeline(TextToVideoPipeline):
    """Extend pipeline: add frames before or after an existing video.

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

    def extend_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        extend_frames: int,
        direction: str = "after",
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Extend a video file by adding frames.

        Convenience wrapper that loads the video, encodes it to latents,
        and calls :meth:`extend`.

        Args:
            prompt: Text prompt for new frames.
            video_path: Path to the source video file.
            extend_frames: Number of latent frames to add.
            direction: "before" or "after".
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).
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

        return self.extend(
            prompt=prompt,
            source_video_latent=video_latent,
            source_audio_latent=audio_latent,
            extend_frames=extend_frames,
            direction=direction,
            height=info.height,
            width=info.width,
            seed=seed,
            num_steps=num_steps,
        )

    def extend(
        self,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        extend_frames: int,
        direction: str = "after",
        height: int = 480,
        width: int = 704,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Extend a video by adding frames.

        Args:
            prompt: Text prompt for new frames.
            source_video_latent: (B, C, F, H, W) source video latent.
            source_audio_latent: (B, 8, T, 16) source audio latent.
            extend_frames: Number of latent frames to add.
            direction: "before" or "after".
            height: Video height.
            width: Video width.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).
        """
        self.load()
        assert self.dit is not None

        B = source_video_latent.shape[0]
        F_source = source_video_latent.shape[2]
        _, H, W = compute_video_latent_shape(1, height, width)
        F_total = F_source + extend_frames
        tokens_per_frame = H * W

        # Patchify source
        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)

        # Create new tokens for extension
        new_shape = (B, extend_frames * tokens_per_frame, 128)
        mx.random.seed(seed)
        new_noise = mx.random.normal(new_shape).astype(mx.bfloat16)

        # Combine source (preserved) and new (to denoise)
        if direction == "after":
            combined_latent = mx.concatenate([source_tokens, new_noise], axis=1)
            denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, source_tokens.shape[1], 1)),
                    mx.ones((B, new_noise.shape[1], 1)),
                ],
                axis=1,
            )
        else:  # before
            combined_latent = mx.concatenate([new_noise, source_tokens], axis=1)
            denoise_mask = mx.concatenate(
                [
                    mx.ones((B, new_noise.shape[1], 1)),
                    mx.zeros((B, source_tokens.shape[1], 1)),
                ],
                axis=1,
            )

        # Compute positions for RoPE
        video_positions = compute_video_positions(F_total, H, W)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)
        extend_audio_T = round(audio_T * extend_frames / F_source)
        audio_total_T = audio_T + extend_audio_T
        audio_positions = compute_audio_positions(audio_total_T)

        video_state = LatentState(
            latent=combined_latent,
            clean_latent=combined_latent * (1.0 - denoise_mask),  # clean only preserved parts
            denoise_mask=denoise_mask,
            positions=video_positions,
        )

        # Audio: preserve source audio, extend with noise
        mx.random.seed(seed + 1)
        new_audio_noise = mx.random.normal((B, extend_audio_T, 128)).astype(mx.bfloat16)

        if direction == "after":
            audio_combined = mx.concatenate([audio_tokens, new_audio_noise], axis=1)
            audio_denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, audio_T, 1), dtype=mx.bfloat16),
                    mx.ones((B, extend_audio_T, 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
        else:  # before
            audio_combined = mx.concatenate([new_audio_noise, audio_tokens], axis=1)
            audio_denoise_mask = mx.concatenate(
                [
                    mx.ones((B, extend_audio_T, 1), dtype=mx.bfloat16),
                    mx.zeros((B, audio_T, 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )

        audio_state = LatentState(
            latent=audio_combined,
            clean_latent=audio_combined * (1.0 - audio_denoise_mask),
            denoise_mask=audio_denoise_mask,
            positions=audio_positions,
        )

        # Text encoding
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

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F_total, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
