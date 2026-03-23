"""Two-stage pipeline — generate at half res, neural upscale, then refine.

Ported from ltx-pipelines two-stage generation.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, create_initial_state, noise_latent_state
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import load_split_safetensors
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop


class TwoStagePipeline(TextToVideoPipeline):
    """Two-stage generation: half-res generate -> neural upscale -> refine.

    Stage 1: Generate at half spatial resolution.
    Stage 2: Upscale latents 2x with neural upsampler, then refine.

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
        self.upsampler: LatentUpsampler | None = None

    def _load_upsampler(self, name: str = "spatial_upscaler_x2_v1_1") -> None:
        """Load upsampler from config and weights.

        Args:
            name: Base name of the upsampler files (without extension).
        """
        import json

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

    def load(self) -> None:
        """Load all components including upsampler."""
        super().load()

        if self.upsampler is None:
            self._load_upsampler()

    def generate_two_stage(
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using two-stage pipeline.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1.
            stage2_steps: Denoising steps for stage 2.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        self.load()
        assert self.dit is not None
        assert self.upsampler is not None

        # Stage 1: Generate at half resolution
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_embeds, audio_embeds = self._encode_text(prompt)
        if self.low_memory:
            aggressive_cleanup()

        # Compute positions for RoPE (stage 1 — half resolution)
        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions_1)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

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

        # Unpatchify and upscale
        video_half = self.video_patchifier.unpatchify(output_1.video_latent, (F, H_half, W_half))
        video_upscaled = self.upsampler(video_half)
        if self.low_memory:
            aggressive_cleanup()

        # Stage 2: Refine at full resolution
        _, H_full, W_full = compute_video_latent_shape(num_frames, height, width)
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)

        # Start from upscaled + small noise
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens * (1.0 - start_sigma)

        # Compute positions for RoPE (stage 2 — full resolution)
        video_positions_2 = compute_video_positions(F, H_full, W_full)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens,
            denoise_mask=mx.ones((1, video_tokens.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        # Audio gets noised and refined in stage 2 (reference behavior)
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
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
    ) -> str:
        """Generate two-stage video+audio and save to file.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1.
            stage2_steps: Denoising steps for stage 2.

        Returns:
            Path to the output video file.
        """
        video_latent, audio_latent = self.generate_two_stage(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
        )

        # Free transformer + text encoder to make room for VAE decode
        if self.low_memory:
            self.dit = None
            self.text_encoder = None
            self.feature_extractor = None
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

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
