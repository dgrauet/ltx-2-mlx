"""Extend a clip on the distilled two-stage path -- no dev model, no CFG.

Upstream has no ``extend``; it continues a video with its chunk system (``ltx_pipelines.chunks``,
used by ``DistilledPipeline.stream_chunks``): each new window is generated with the previous
window's last ``next_video_carry_frames`` (25 pixel frames, 4 latent frames) of video latent and
the matching audio latent pinned at index 0 (``VideoConditionByLatentIndex`` /
``AudioConditionByLatentIndex``, strength 1.0), in both distilled stages. Here the previous
window is the source clip:

  Stage 1: the source's last 4 latent frames (encoded at half resolution) and its last audio
           tokens pinned at the start of a window of ``4 + extend_frames`` latent frames; the
           rest is generated at half resolution (8 steps, ancestral on LTX-2.5).
  Stage 2: spatial 2x upsample, the full-resolution source tail pinned, 3-step refine.

The window's new latent frames are appended to the source's full-resolution latent (the pinned
frames are the source's own last frames, so the join is the same latent the source ends on), and
the whole clip is decoded once. Denoising cost follows the window, not the source length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import snap_output_dimensions
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, VideoConditionByLatentIndex
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.image import load_video_frames
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_token_count
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.retake import encode_source_audio_latent, encode_video_tensor
from ltx_pipelines_mlx.utils.args import ImageConditioningInput, resolve_frame_indices
from ltx_pipelines_mlx.utils.helpers import create_noised_state

#: Pixel frames of the source carried into the new window (upstream ``ChunkConfig.next_video_carry_frames``).
EXTEND_CARRY_PIXEL_FRAMES = 25


def carry_latent_frames(carry_pixel_frames: int = EXTEND_CARRY_PIXEL_FRAMES) -> int:
    """Latent frames that hold ``carry_pixel_frames`` (upstream ``(carry - 1) // 8 + 1``)."""
    return (carry_pixel_frames - 1) // 8 + 1


def fit_audio_latent(audio_latent: mx.array, audio_tokens: int) -> mx.array:
    """Right-pad (zeros) or trim an audio latent ``(1, 8, T, 16)`` to exactly ``audio_tokens`` tokens.

    The source's audio stream can end before its video (trimmed or edited clips), so the encoded track can be shorter
    than the clip. The carry takes its last tokens and the window's audio is appended after the whole source audio,
    so both need the audio to span the same length as the video.

    Args:
        audio_latent: Encoded source audio.
        audio_tokens: The clip's audio token count (``compute_audio_token_count``).

    Returns:
        The latent with ``audio_tokens`` tokens on the time axis.
    """
    have = audio_latent.shape[2]
    if have > audio_tokens:
        return audio_latent[:, :, :audio_tokens, :]
    if have < audio_tokens:
        pad = mx.zeros((*audio_latent.shape[:2], audio_tokens - have, audio_latent.shape[3]), dtype=audio_latent.dtype)
        return mx.concatenate([audio_latent, pad], axis=2)
    return audio_latent


def carry_audio_tokens(frame_rate: float, carry_pixel_frames: int = EXTEND_CARRY_PIXEL_FRAMES) -> int:
    """Audio latent tokens that cover the carried video (upstream ``AudioLatentShape.from_duration``, at least 1)."""
    return max(1, compute_audio_token_count(carry_pixel_frames, frame_rate=frame_rate))


@dataclass(frozen=True)
class _Carry:
    """The source's tail, patchified, pinned at the start of the new window."""

    video_half: mx.array
    video_full: mx.array
    audio: mx.array
    latent_frames: int


def _pin_prefix(tokens: mx.array, tokens_per_frame: int) -> VideoConditionByLatentIndex:
    """Pin ``tokens`` over the first frames of a state (upstream ``*ConditionByLatentIndex(latent_idx=0)``)."""
    return VideoConditionByLatentIndex(
        frame_indices=list(range(tokens.shape[1] // tokens_per_frame)), clean_latent=tokens, strength=1.0
    )


def _as_guide(image: ImageConditioningInput) -> ImageConditioningInput:
    """The same anchor as a guide.

    Every anchor guides: the window starts with the pinned carry, so the new frames are counted
    from a point that is not on the latent grid of the window. A replace binding is only taken on
    frame 0, the binding ``auto`` (and ``ImageConditioningInput.at``) gives that frame.

    Args:
        image: An anchor on the new frames, ``frame_idx`` possibly negative (counted from the end).

    Returns:
        The anchor bound as a guide, same path, frame, strength and CRF.

    Raises:
        ValueError: on a replace binding off frame 0 (upstream would honour it on its own frame grid;
            here it would silently become a guide).
    """
    if image.replaces and image.frame_idx != 0:
        raise ValueError(
            f"--image {image.path}: extend --distilled takes every anchor as a guide; "
            f"a replace binding is not supported (got frame {image.frame_idx}), use guide"
        )
    return ImageConditioningInput.guide(image.path, image.frame_idx, image.strength, image.crf)


class ExtendDistilledPipeline(DistilledPipeline):
    """Append frames to a clip with the distilled transformer (``extend --distilled``).

    Everything but the pinned tail is :class:`~ltx_pipelines_mlx.distilled.DistilledPipeline`:
    weights, sigma tables, the ancestral sampler on LTX-2.5, ``--low-ram`` streaming and the
    stage-2 options (``LTX2_SOL_TAU``, ``LTX2_COMPUTE_DTYPE``).
    """

    #: The source tail for the render in progress; ``None`` outside :meth:`extend_from_video`,
    #: where the hooks fall back to the parent's behaviour.
    _carry: _Carry | None = None
    #: Source frame rate, recorded for the CLI decode helper (as ``RetakePipeline``).
    source_frame_rate: float | None = None

    def _stage1_video_conditionings(self, spatial_dims: tuple[int, int, int]) -> list:
        """Stage 1: the source's last latent frames, at half resolution, pinned at frame 0."""
        if self._carry is None:
            return super()._stage1_video_conditionings(spatial_dims)
        _, h, w = spatial_dims
        return [_pin_prefix(self._carry.video_half, h * w)]

    def _stage1_audio_state(
        self,
        audio_shape: tuple[int, int, int],
        audio_positions: mx.array,
        spatial_dims: tuple[int, int, int],
        seed: int,
    ) -> LatentState:
        """Stage 1's audio: noise, with the source's last audio tokens pinned at the start."""
        if self._carry is None:
            return super()._stage1_audio_state(audio_shape, audio_positions, spatial_dims, seed)
        return create_noised_state(
            base_shape=audio_shape,
            conditionings=[_pin_prefix(self._carry.audio, 1)],
            spatial_dims=(audio_shape[1], 1, 1),
            positions=audio_positions,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

    def _stage2_audio_state(
        self,
        audio_tokens: mx.array,
        audio_positions: mx.array,
        spatial_dims: tuple[int, int, int],
        seed: int,
        sigma: float,
    ) -> LatentState:
        """Stage 2's audio: stage 1's re-noised at ``sigma``, the source's tail pinned again."""
        if self._carry is None:
            return super()._stage2_audio_state(audio_tokens, audio_positions, spatial_dims, seed, sigma)
        return create_noised_state(
            base_shape=audio_tokens.shape,
            conditionings=[_pin_prefix(self._carry.audio, 1)],
            spatial_dims=(audio_tokens.shape[1], 1, 1),
            positions=audio_positions,
            seed=seed,
            sigma=sigma,
            initial_latent=audio_tokens,
        )

    def extend_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        extend_frames: int,
        *,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        images: list[ImageConditioningInput] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Append ``extend_frames`` latent frames (``8 * extend_frames`` pixel frames) to a clip.

        Args:
            prompt: Text prompt for the new window (the carried frames are its first 25).
            video_path: Source clip. Its size must be on the two-stage grid (multiples of 64).
            extend_frames: Latent frames to add.
            seed: Random seed.
            stage1_steps: Stage 1 steps (default: the full distilled table, 8).
            stage2_steps: Stage 2 steps (default: the full stage-2 table, 3).
            images: Optional image anchors for the new frames, as ``generate --distilled --image``
                takes them (upstream passes images to each chunk). ``frame_idx`` counts the new
                frames only: ``0`` is the first appended frame, ``-1`` (``last``) the last one. They
                are placed after the 25 carried frames, so every anchor is a guide
                (``VideoConditionByKeyframeIndex``), re-encoded at full resolution for stage 2,
                and its tokens are dropped before decoding.

        Returns:
            ``(video_latent, audio_latent)`` of the whole extended clip, at the source resolution.

        Raises:
            ValueError: on ``extend_frames < 1``, an image anchor outside the new frames, a source
                off the two-stage grid, or a source shorter than the carried frames -- before any
                model is loaded.
        """
        if extend_frames < 1:
            raise ValueError(f"extend_frames must be >= 1, got {extend_frames}")
        # Anchors are given on the new frames; in the window they come after the carried ones.
        # Every anchor guides (the window starts with the pinned carry), whatever its binding.
        window_images = [
            image.with_frame_idx(image.frame_idx + EXTEND_CARRY_PIXEL_FRAMES)
            for image in resolve_frame_indices([_as_guide(image) for image in images or []], 8 * extend_frames)
        ]
        video_path = str(video_path)
        info = probe_video_info(video_path)
        height, width, frame_rate = info.height, info.width, info.fps
        if snap_output_dimensions(height, width, two_stage=True) != (height, width):
            raise ValueError(
                f"extend --distilled renders at the source size, which must be a multiple of 64 on both sides "
                f"(two-stage grid); got {width}x{height}."
            )
        num_frames = 1 + 8 * ((info.num_frames - 1) // 8)
        carry = carry_latent_frames()
        if num_frames < EXTEND_CARRY_PIXEL_FRAMES:
            raise ValueError(
                f"The source has {info.num_frames} frames; extend --distilled carries its last "
                f"{EXTEND_CARRY_PIXEL_FRAMES}, so it needs at least that many."
            )
        carry_audio = carry_audio_tokens(frame_rate)
        self.source_frame_rate = frame_rate

        # --- Source: full resolution (the output's first part) and half resolution (the stage-1 carry) ---
        def _encode_both(encoder) -> tuple[mx.array, mx.array]:
            # One encoder load for both; each pixel tensor is dropped once its latent is materialized.
            full = encode_video_tensor(encoder, load_video_frames(video_path, height, width, num_frames))
            half = encode_video_tensor(encoder, load_video_frames(video_path, height // 2, width // 2, num_frames))
            return full, half

        full_latent, half_latent = self.image_conditioner(_encode_both, free_after=self.low_memory)
        audio_latent = encode_source_audio_latent(
            self, video_path, num_frames=num_frames, frame_rate=frame_rate, has_audio=info.has_audio
        )
        if self.low_memory:
            aggressive_cleanup()

        audio_latent = fit_audio_latent(audio_latent, compute_audio_token_count(num_frames, frame_rate=frame_rate))
        audio_tokens, _ = self.audio_patchifier.patchify(audio_latent)
        self._carry = _Carry(
            video_half=self.video_patchifier.patchify(half_latent[:, :, -carry:])[0],
            video_full=self.video_patchifier.patchify(full_latent[:, :, -carry:])[0],
            audio=audio_tokens[:, -carry_audio:, :],
            latent_frames=carry,
        )
        del half_latent
        window_frames = 1 + 8 * (carry + extend_frames - 1)
        try:
            stage1, window_frames, _, _ = self._stage1(
                prompt,
                height,
                width,
                window_frames,
                frame_rate=frame_rate,
                seed=seed,
                stage1_steps=stage1_steps,
                image=None,
                images=window_images or None,
                prompt_relay=None,
                generated_keyframes=0,
                enable_teacache=False,
            )
            video_half = self.video_patchifier.unpatchify(stage1.video_tokens, stage1.latent_dims)
            video_upscaled = self._upsample_latent(video_half)
            _, h_full, w_full = video_upscaled.shape[2:]
            window_video, window_audio = self._stage2(
                stage1,
                video_upscaled,
                num_frames=window_frames,
                frame_rate=frame_rate,
                seed=seed,
                stage2_steps=stage2_steps,
                extra_conditionings=[_pin_prefix(self._carry.video_full, h_full * w_full)],
            )
        finally:
            self._carry = None

        # The window's first ``carry`` latent frames are the source's last ones (pinned); append the rest.
        video = mx.concatenate([full_latent, window_video[:, :, carry:].astype(full_latent.dtype)], axis=2)
        audio = mx.concatenate([audio_latent, window_audio[:, :, carry_audio:].astype(audio_latent.dtype)], axis=2)
        return video, audio


__all__ = [
    "EXTEND_CARRY_PIXEL_FRAMES",
    "ExtendDistilledPipeline",
    "carry_audio_tokens",
    "carry_latent_frames",
    "fit_audio_latent",
]
