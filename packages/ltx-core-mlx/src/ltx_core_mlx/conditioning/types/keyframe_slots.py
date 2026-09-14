"""Generated keyframe slots: extra single-pixel-frame tokens denoised with the target.

Port of ``ltx_core.conditioning.types.keyframe_slots``. Unlike every other conditioning item,
this one supplies no content by default. It appends empty, fully-denoised token slots whose
RoPE position marks a single interior pixel frame, letting the model generate additional frames
at those positions and condition the surrounding video on them. That relaxes the effective
temporal compression ratio there, at the cost of one latent frame's worth of tokens per keyframe
(which buys 1 pixel frame instead of the usual 8).

Optional ``initial_keyframes`` seeds the appended ``latent`` tokens (not ``clean_latent``) so
the noiser can lerp from that content at ``denoise_mask=1``.

Requires a checkpoint trained for it -- one whose transformer config sets
``use_keyframes_abs_pos_embedding``. On any other checkpoint the learned marker does not exist,
so the slots would be denoised as unmarked tokens and the extra compute wasted; pipelines
validate this up front rather than silently degrading.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import extend_keyframes_mask, update_attention_mask
from ltx_core_mlx.conditioning.types.keyframe_cond import _compute_keyframe_positions
from ltx_core_mlx.conditioning.types.latent_cond import GeneratedKeyframeLayout, LatentState
from ltx_core_mlx.utils.positions import VIDEO_TEMPORAL_SCALE


class VideoGeneratedKeyframeSlots:
    """Appends fully-denoised keyframe slots at the given target pixel-frame indices.

    Each slot occupies one latent frame's worth of tokens at the target's spatial resolution,
    with ``denoise_mask = 1`` so the noiser fills it from the slot ``latent`` (zeros, or
    ``initial_keyframes`` when provided) and the denoising loop generates its content. The slot
    is marked in ``keyframes_mask`` so it receives the model's learned keyframe absolute-position
    embedding.

    Deliberately separate from :class:`VideoConditionByKeyframeIndex`, which appends *given*
    keyframe content for image guidance and must not be marked.

    All slots are appended by a single item so the resulting token range is contiguous and can
    be recorded as one :class:`GeneratedKeyframeLayout`.

    Args:
        pixel_frame_indices: Target pixel-frame index for each slot. Non-empty, strictly
            increasing, within the target's frame range.
        frame_rate: Frame rate used to express slot positions in seconds.
        initial_keyframes: Optional ``(B, C, K, H, W)`` latent content written into the
            appended ``latent`` tokens before noising. ``K`` must equal ``len(pixel_frame_indices)``.
    """

    def __init__(
        self,
        pixel_frame_indices: Sequence[int],
        frame_rate: float,
        initial_keyframes: mx.array | None = None,
    ) -> None:
        indices = tuple(int(index) for index in pixel_frame_indices)
        if not indices:
            raise ValueError("pixel_frame_indices must be non-empty")
        if any(index < 0 for index in indices):
            raise ValueError(f"pixel_frame_indices must be non-negative, got {indices}")
        if any(b <= a for a, b in itertools.pairwise(indices)):
            raise ValueError(f"pixel_frame_indices must be strictly increasing, got {indices}")
        if initial_keyframes is not None:
            if initial_keyframes.ndim != 5:
                raise ValueError(f"initial_keyframes must have shape (B, C, K, H, W), got {initial_keyframes.shape}")
            if initial_keyframes.shape[2] != len(indices):
                raise ValueError(
                    f"initial_keyframes K={initial_keyframes.shape[2]} must match {len(indices)} pixel_frame_indices"
                )
        self.pixel_frame_indices = indices
        self.frame_rate = float(frame_rate)
        self.initial_keyframes = initial_keyframes

    def apply(self, state: LatentState, spatial_dims: tuple[int, int, int]) -> LatentState:
        """Append the slots to ``state`` (item API shared with the other conditioning types)."""
        from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier

        num_latent_frames, height, width = spatial_dims
        num_pixel_frames = (num_latent_frames - 1) * VIDEO_TEMPORAL_SCALE + 1
        if self.pixel_frame_indices[-1] >= num_pixel_frames:
            raise ValueError(
                f"Generated keyframe at pixel frame {self.pixel_frame_indices[-1]} is outside the target's "
                f"{num_pixel_frames} frames"
            )
        if state.generated_keyframe_layout is not None:
            raise ValueError("Generated keyframe slots were already applied to this state; use a single item")

        tokens_per_keyframe = height * width
        num_new_tokens = tokens_per_keyframe * len(self.pixel_frame_indices)
        batch_size, num_existing, channels = state.latent.shape
        dtype = state.latent.dtype

        positions = mx.concatenate(
            [_compute_keyframe_positions(index, height, width, self.frame_rate) for index in self.pixel_frame_indices],
            axis=1,
        )
        if positions.shape[0] != batch_size:
            positions = mx.broadcast_to(positions, (batch_size, num_new_tokens, positions.shape[-1]))

        if self.initial_keyframes is None:
            slot_tokens = mx.zeros((batch_size, num_new_tokens, channels), dtype=dtype)
        else:
            initials = self.initial_keyframes.astype(dtype)
            if initials.shape[0] != batch_size:
                raise ValueError(
                    f"initial_keyframes batch {initials.shape[0]} does not match latent batch {batch_size}"
                )
            if (initials.shape[3], initials.shape[4]) != (height, width):
                raise ValueError(
                    f"initial_keyframes spatial size {(initials.shape[3], initials.shape[4])} does not match "
                    f"target latent spatial size {(height, width)}"
                )
            patchifier = VideoLatentPatchifier()
            slot_tokens = mx.concatenate(
                [patchifier.patchify(initials[:, :, k : k + 1])[0] for k in range(initials.shape[2])], axis=1
            )

        # denoise_mask 1 => the noiser lerps the slot latent toward noise (clean_latent is ignored).
        denoise_mask = mx.ones((batch_size, num_new_tokens, 1), dtype=state.denoise_mask.dtype)
        keyframes_mask = extend_keyframes_mask(state, num_new_tokens, marked=True)
        attention_mask = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=num_latent_frames * tokens_per_keyframe,
            num_new_tokens=num_new_tokens,
            batch_size=batch_size,
        )
        if state.positions is None:
            raise ValueError("Generated keyframe slots need a state with positions")

        return LatentState(
            latent=mx.concatenate([state.latent, slot_tokens], axis=1),
            clean_latent=mx.concatenate([state.clean_latent, mx.zeros_like(slot_tokens)], axis=1),
            denoise_mask=mx.concatenate([state.denoise_mask, denoise_mask], axis=1),
            positions=mx.concatenate([state.positions, positions.astype(state.positions.dtype)], axis=1),
            attention_mask=attention_mask,
            keyframes_mask=keyframes_mask,
            generated_keyframe_layout=GeneratedKeyframeLayout(
                pixel_frame_indices=self.pixel_frame_indices,
                tokens_per_keyframe=tokens_per_keyframe,
                first_token=num_existing,
            ),
            generated_keyframes=state.generated_keyframes,
        )


def extract_generated_keyframes(
    tokens: mx.array,
    layout: GeneratedKeyframeLayout | None,
    patchifier,
    spatial_hw: tuple[int, int],
) -> mx.array | None:
    """Extract generated keyframe slot content as a ``(B, C, K, H, W)`` latent.

    Mirror of upstream ``VideoLatentTools.extract_generated_keyframes``. Each slot is
    unpatchified on its own as a one-frame clip; the result must be decoded frame by frame,
    never as a K-frame video (a causal decode would blend slots that were never adjacent).

    Args:
        tokens: ``(B, T, C)`` denoised token sequence *before* conditioning tokens are cut.
        layout: The layout recorded by :class:`VideoGeneratedKeyframeSlots`, or ``None``.
        patchifier: Video patchifier with ``unpatchify(tokens, (F, H, W))``.
        spatial_hw: ``(H, W)`` latent spatial size of the target.

    Returns:
        ``(B, C, K, H, W)`` latent, or ``None`` when the state carries no slots.
    """
    if layout is None:
        return None
    height, width = spatial_hw
    available = tokens.shape[1]
    if layout.first_token + layout.num_tokens > available:
        raise ValueError(
            f"Generated keyframe layout spans tokens [{layout.first_token}, {layout.first_token + layout.num_tokens}) "
            f"but the sequence has only {available} tokens. The layout was recorded against a different token sequence."
        )
    if layout.tokens_per_keyframe != height * width:
        raise ValueError(
            f"Generated keyframe layout has {layout.tokens_per_keyframe} tokens per keyframe, but this target uses "
            f"{height * width}. The layout belongs to a different resolution."
        )
    slot_tokens = tokens[:, layout.token_slice]
    frames = [
        patchifier.unpatchify(
            slot_tokens[:, k * layout.tokens_per_keyframe : (k + 1) * layout.tokens_per_keyframe], (1, height, width)
        )
        for k in range(layout.num_keyframes)
    ]
    return mx.concatenate(frames, axis=2)
