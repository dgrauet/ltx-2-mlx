"""Retake pipeline — regenerate a time segment of an existing video.

Matches the reference architecture: single-stage with dev model + CFG guidance.
Default: 30 steps, cfg=3.0, full multi-modal guidance.

Ported from ltx-pipelines/src/ltx_pipelines/retake.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, TemporalRegionMask, noise_latent_state
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
DEFAULT_STG_SCALE = 0.0  # 32GB safe; ref uses 1.0 but OOM on long videos


class RetakePipeline(TextToVideoPipeline):
    """Retake pipeline: regenerate a time segment while preserving the rest.

    Uses the dev (non-distilled) model with CFG guidance for quality output.
    Single-stage pipeline (no upsampler).

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

    def retake_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
        regenerate_audio: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video file.

        Args:
            prompt: Text prompt for the regenerated segment.
            video_path: Path to the source video file.
            start_frame: First latent frame to regenerate (inclusive).
            end_frame: Last latent frame to regenerate (exclusive).
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 0.0).
            regenerate_audio: If True, regenerate audio in the retake region.
                If False, preserve original audio entirely.

        Returns:
            Tuple of (video_latent, audio_latent).
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

        # Free encoders before text encoding
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
            num_frames=vae_compatible_frames,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            regenerate_audio=regenerate_audio,
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
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
        regenerate_audio: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video.

        Args:
            prompt: Text prompt for the regenerated segment.
            source_video_latent: (B, C, F, H, W) source video latent.
            source_audio_latent: (B, 8, T, 16) source audio latent.
            start_frame: First latent frame to regenerate (inclusive).
            end_frame: Last latent frame to regenerate (exclusive).
            height: Video height.
            width: Video width.
            num_frames: Total number of pixel frames.
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 0.0).
            regenerate_audio: If True, regenerate audio in the retake region.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        # --- Text encoding (positive + negative for CFG) ---
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        # --- Load dev transformer ---
        if self.dit is None:
            self.dit = self._load_dev_transformer()
        assert self.dit is not None

        F, H, W = compute_video_latent_shape(num_frames, height, width)
        tokens_per_frame = H * W

        # Patchify source
        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)

        # Compute positions
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        # Create video state with temporal mask (1 = regenerate, 0 = preserve)
        region = TemporalRegionMask(start_frame, end_frame)
        denoise_mask = region.create_mask(F, tokens_per_frame)

        video_state = LatentState(
            latent=source_tokens,
            clean_latent=source_tokens,
            denoise_mask=denoise_mask,
            positions=video_positions,
        )
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)

        # Audio state
        if regenerate_audio:
            # Apply same temporal mask to audio
            audio_tokens_per_video_frame = audio_T / F
            audio_start = round(start_frame * audio_tokens_per_video_frame)
            audio_end = round(end_frame * audio_tokens_per_video_frame)

            audio_mask = mx.zeros((1, audio_T, 1), dtype=mx.bfloat16)
            audio_mask = audio_mask.at[:, audio_start:audio_end, :].add(
                mx.ones((1, audio_end - audio_start, 1), dtype=mx.bfloat16)
            )
        else:
            # Preserve all audio (mask=0)
            audio_mask = mx.zeros((1, audio_T, 1), dtype=mx.bfloat16)

        audio_state = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=audio_mask,
            positions=audio_positions,
        )
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # --- Guided denoising ---
        num_tokens = F * H * W
        sigmas = ltx2_schedule(num_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        # Build guidance (ref LTX_2_3_PARAMS)
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

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
