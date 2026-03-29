"""Extend pipeline — add frames before or after an existing video.

No direct Lightricks reference (custom MLX pipeline). Uses the dev model
with CFG guidance for consistency with retake and two-stage pipelines.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, noise_latent_state
from ltx_core_mlx.model.audio_vae import encode_audio
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.image import load_video_frames
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.scheduler import ltx2_schedule
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop

# Reference defaults (LTX_2_3_PARAMS)
DEFAULT_CFG_SCALE = 3.0
DEFAULT_STG_SCALE = 0.0  # 32GB safe


class ExtendPipeline(TextToVideoPipeline):
    """Extend pipeline: add frames before or after an existing video.

    Uses the dev model with CFG guidance. Single-stage.

    Args:
        model_dir: Path to model weights.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Dev transformer filename.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        dev_transformer: str = "transformer-dev.safetensors",
    ):
        super().__init__(model_dir, gemma_model_id=gemma_model_id, low_memory=low_memory)
        self._dev_transformer = dev_transformer

    def extend_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        extend_frames: int,
        direction: str = "after",
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
    ) -> tuple[mx.array, mx.array]:
        """Extend a video file by adding frames.

        Args:
            prompt: Text prompt for new frames.
            video_path: Path to the source video file.
            extend_frames: Number of latent frames to add.
            direction: "before" or "after".
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 0.0).

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).
        """
        # --- Encode source video + audio ---
        self._load_vae_encoder()
        self._load_audio_encoder()
        assert self.vae_encoder is not None

        video_path = str(video_path)
        info = probe_video_info(video_path)

        # Round down to VAE-compatible frame count (1 + 8k) that fits in the source
        k = max(1, (info.num_frames - 1) // 8)
        vae_compatible_frames = 1 + k * 8

        video_tensor = load_video_frames(video_path, info.height, info.width, vae_compatible_frames)
        video_latent = self.vae_encoder.encode(video_tensor)
        mx.synchronize()
        if self.low_memory:
            del video_tensor
            aggressive_cleanup()

        audio_latent: mx.array | None = None
        if info.has_audio:
            assert self.audio_encoder is not None
            assert self.audio_processor is not None
            audio_data = load_audio(
                video_path,
                target_sample_rate=16000,
                max_duration=vae_compatible_frames / info.fps,
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
            audio_T = compute_audio_token_count(vae_compatible_frames)
            audio_latent = mx.zeros((1, 8, audio_T, 16), dtype=mx.bfloat16)

        # Free encoders
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
            num_frames=vae_compatible_frames,
            fps=info.fps,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
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
        num_frames: int = 97,
        fps: float = 24.0,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
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
            num_frames: Total number of pixel frames in source.
            fps: Frame rate.
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 0.0).

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).
        """
        # --- Text encoding (positive + negative for CFG) ---
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        # --- Load dev transformer ---
        if self.dit is None:
            self.dit = self._load_dev_transformer()
        assert self.dit is not None

        B = source_video_latent.shape[0]
        F_source = source_video_latent.shape[2]
        _, H, W = compute_video_latent_shape(1, height, width)
        F_total = F_source + extend_frames
        tokens_per_frame = H * W

        # Patchify source
        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)

        # Compute total pixel frames for audio token count
        total_pixel_frames = num_frames + extend_frames * 8  # each latent frame = 8 pixel frames
        extend_audio_T = compute_audio_token_count(total_pixel_frames, fps) - audio_T
        if extend_audio_T < 0:
            extend_audio_T = 0
        audio_total_T = audio_T + extend_audio_T

        # Create video state with noise for new frames, preserve source
        new_shape = (B, extend_frames * tokens_per_frame, 128)

        if direction == "after":
            denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, source_tokens.shape[1], 1), dtype=mx.bfloat16),
                    mx.ones((B, new_shape[1], 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            # Clean latent: source for preserved, zeros for new
            clean_video = mx.concatenate([source_tokens, mx.zeros(new_shape, dtype=mx.bfloat16)], axis=1)
        else:  # before
            denoise_mask = mx.concatenate(
                [
                    mx.ones((B, new_shape[1], 1), dtype=mx.bfloat16),
                    mx.zeros((B, source_tokens.shape[1], 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_video = mx.concatenate([mx.zeros(new_shape, dtype=mx.bfloat16), source_tokens], axis=1)

        video_positions = compute_video_positions(F_total, H, W)
        audio_positions = compute_audio_positions(audio_total_T)

        video_state = LatentState(
            latent=clean_video,
            clean_latent=clean_video,
            denoise_mask=denoise_mask,
            positions=video_positions,
        )
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)

        # Audio: preserve source, extend with noise
        if direction == "after":
            audio_denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, audio_T, 1), dtype=mx.bfloat16),
                    mx.ones((B, extend_audio_T, 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_audio = mx.concatenate([audio_tokens, mx.zeros((B, extend_audio_T, 128), dtype=mx.bfloat16)], axis=1)
        else:
            audio_denoise_mask = mx.concatenate(
                [
                    mx.ones((B, extend_audio_T, 1), dtype=mx.bfloat16),
                    mx.zeros((B, audio_T, 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_audio = mx.concatenate([mx.zeros((B, extend_audio_T, 128), dtype=mx.bfloat16), audio_tokens], axis=1)

        audio_state = LatentState(
            latent=clean_audio,
            clean_latent=clean_audio,
            denoise_mask=audio_denoise_mask,
            positions=audio_positions,
        )
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # --- Guided denoising ---
        num_tokens = F_total * H * W
        sigmas = ltx2_schedule(num_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        video_gp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        audio_gp = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )

        video_factory = create_multimodal_guider_factory(video_gp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_gp, negative_context=neg_audio_embeds)

        output = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas,
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F_total, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
