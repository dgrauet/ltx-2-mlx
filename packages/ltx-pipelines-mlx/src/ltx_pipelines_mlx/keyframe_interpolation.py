"""Keyframe interpolation pipeline — interpolate between reference frames.

Ported from ltx-pipelines keyframe interpolation.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import create_initial_state
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop


class KeyframeInterpolationPipeline(TextToVideoPipeline):
    """Keyframe interpolation pipeline.

    Generates video by interpolating between reference keyframe images.
    Each keyframe is applied as a separate conditioning item (matching
    the reference), so their tokens don't cross-attend to each other.

    Args:
        model_dir: Path to model weights.
        low_memory: Aggressive memory management.
    """

    def interpolate(
        self,
        prompt: str,
        keyframe_latents: list[mx.array],
        keyframe_indices: list[int],
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        fps: float = 24.0,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video interpolating between keyframes.

        Args:
            prompt: Text prompt.
            keyframe_latents: List of encoded keyframe latents, each (1, C, 1, H', W').
            keyframe_indices: Pixel frame indices for each keyframe (0-based).
                E.g. for a 97-frame video, use [0, 96] for first and last.
            height: Video height.
            width: Video width.
            num_frames: Total number of pixel frames.
            fps: Frame rate (used for position computation).
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        self.load()
        assert self.dit is not None

        F, H, W = compute_video_latent_shape(num_frames, height, width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        # Compute positions for RoPE
        video_positions = compute_video_positions(F, H, W, fps=fps)
        audio_positions = compute_audio_positions(audio_T)

        # Create initial states
        video_state = create_initial_state(video_shape, seed, positions=video_positions)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Apply keyframes one by one (matching reference: sequential application
        # so each keyframe builds its own attention mask block)
        for kf_lat, kf_idx in zip(keyframe_latents, keyframe_indices):
            # Patchify keyframe latent to tokens: (1, C, 1, H', W') -> (1, H'*W', C)
            tokens = kf_lat.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)

            kf_condition = VideoConditionByKeyframeIndex(
                frame_idx=kf_idx,
                keyframe_latent=tokens,
                spatial_dims=(F, H, W),
                fps=fps,
            )
            video_state = kf_condition.apply(video_state, (F, H, W))

        # Text encoding
        video_embeds, audio_embeds = self._encode_text(prompt)
        if self.low_memory:
            aggressive_cleanup()

        # Denoise — attention mask is resolved from video_state automatically
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

        # Remove appended keyframe tokens before unpatchifying
        gen_tokens = output.video_latent[:, : F * H * W, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
