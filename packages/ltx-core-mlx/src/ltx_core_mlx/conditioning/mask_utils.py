"""Utilities for building 2D self-attention masks for conditioning items.

Ported from ltx-core/src/ltx_core/conditioning/mask_utils.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState


def resolve_cross_mask(
    attention_mask: float | int | mx.array,
    num_new_tokens: int,
    batch_size: int,
) -> mx.array:
    """Convert an attention_mask (scalar or array) to a (B, M) cross_mask.

    Args:
        attention_mask: Scalar, 1D (M,), or 2D (B, M) array.
        num_new_tokens: Number of new conditioning tokens M.
        batch_size: Batch size B.

    Returns:
        Cross-mask of shape (B, M).
    """
    if isinstance(attention_mask, int | float):
        return mx.full((batch_size, num_new_tokens), attention_mask)

    mask = attention_mask
    if mask.ndim == 0:
        return mx.full((batch_size, num_new_tokens), float(mask.item()))

    if mask.ndim == 1:
        if mask.shape[0] != num_new_tokens:
            raise ValueError(
                f"1-D attention_mask length must equal num_new_tokens ({num_new_tokens}), got shape {tuple(mask.shape)}"
            )
        return mx.broadcast_to(mask[None, :], (batch_size, num_new_tokens))

    if mask.ndim == 2:
        _b, m = mask.shape
        if m != num_new_tokens:
            raise ValueError(
                f"2-D attention_mask dim-1 must equal num_new_tokens ({num_new_tokens}), got shape {tuple(mask.shape)}"
            )
        return mx.broadcast_to(mask, (batch_size, num_new_tokens))

    raise ValueError(f"attention_mask must be 0-D, 1-D, or 2-D, got {mask.ndim}-D")


def build_attention_mask(
    existing_mask: mx.array | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    num_existing_tokens: int,
    cross_mask: mx.array,
) -> mx.array:
    """Build or expand a (B, N+M, N+M) self-attention mask.

    Block structure:
                 noisy      prev_ref    new_ref
               (N_noisy)   (N-N_noisy)    (M)
             +───────────+───────────+───────────+
    noisy    |  existing |  existing |   cross   |
             +───────────+───────────+───────────+
    prev_ref |  existing |  existing |     0     |
             +───────────+───────────+───────────+
    new_ref  |   cross   |     0     |     1     |
             +───────────+───────────+───────────+

    Args:
        existing_mask: Current mask (B, N, N) or None.
        num_noisy_tokens: Original noisy token count.
        num_new_tokens: New conditioning tokens M.
        num_existing_tokens: Current total tokens N.
        cross_mask: Per-token attention weight (B, M), values in [0, 1].

    Returns:
        Attention mask (B, N+M, N+M).
    """
    batch_size = cross_mask.shape[0]
    total = num_existing_tokens + num_new_tokens
    mask = mx.zeros((batch_size, total, total))

    # Top-left: preserve existing or fill with 1s
    if existing_mask is not None:
        mask = mask.at[:, :num_existing_tokens, :num_existing_tokens].add(existing_mask)
    else:
        mask = mask.at[:, :num_existing_tokens, :num_existing_tokens].add(
            mx.ones((batch_size, num_existing_tokens, num_existing_tokens))
        )

    # Bottom-right: new ref tokens self-attend
    mask = mask.at[:, num_existing_tokens:, num_existing_tokens:].add(
        mx.ones((batch_size, num_new_tokens, num_new_tokens))
    )

    # Noisy -> new_ref: cross_mask[:, j] for each column j
    mask = mask.at[:, :num_noisy_tokens, num_existing_tokens:].add(
        mx.broadcast_to(cross_mask[:, None, :], (batch_size, num_noisy_tokens, num_new_tokens))
    )

    # New_ref -> noisy: cross_mask[:, i] for each row i
    mask = mask.at[:, num_existing_tokens:, :num_noisy_tokens].add(
        mx.broadcast_to(cross_mask[:, :, None], (batch_size, num_new_tokens, num_noisy_tokens))
    )

    return mask


def update_attention_mask(
    latent_state: LatentState,
    attention_mask: float | mx.array | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    batch_size: int,
) -> mx.array | None:
    """Build or update the self-attention mask for newly appended conditioning tokens.

    If attention_mask is None and no existing mask is present, returns None.
    If attention_mask is None but an existing mask is present, the mask is
    expanded with full attention (1s) for the new tokens.

    Args:
        latent_state: Current latent state.
        attention_mask: Per-token attention weight (scalar, 1D, 2D, or None).
        num_noisy_tokens: Original noisy token count.
        num_new_tokens: New conditioning tokens being appended.
        batch_size: Batch size.

    Returns:
        Updated mask (B, N+M, N+M) or None.
    """
    if attention_mask is None:
        if latent_state.attention_mask is None:
            return None
        cross_mask = mx.ones((batch_size, num_new_tokens))
        return build_attention_mask(
            existing_mask=latent_state.attention_mask,
            num_noisy_tokens=num_noisy_tokens,
            num_new_tokens=num_new_tokens,
            num_existing_tokens=latent_state.latent.shape[1],
            cross_mask=cross_mask,
        )

    cross_mask = resolve_cross_mask(attention_mask, num_new_tokens, batch_size)
    return build_attention_mask(
        existing_mask=latent_state.attention_mask,
        num_noisy_tokens=num_noisy_tokens,
        num_new_tokens=num_new_tokens,
        num_existing_tokens=latent_state.latent.shape[1],
        cross_mask=cross_mask,
    )


def first_frame_keyframes_mask(denoise_mask: mx.array, tokens_per_frame: int) -> mx.array:
    """Mark the target's first latent frame, which encodes a single pixel frame.

    The video encoder is causal: the first temporal latent frame covers 1 pixel frame while
    every later one covers ``temporal_scale_factor``. That makes it the same token class as a
    generated keyframe slot, and the reference marks it unconditionally, independently of
    whether any slots exist. Plain data about the latent; models without the keyframe
    embedding ignore it.

    Args:
        denoise_mask: ``(B, N, 1)`` mask whose layout and dtype the marker shares.
        tokens_per_frame: Token count of one latent frame at the target's spatial size.

    Returns:
        ``(B, N, 1)`` marker: 1 on the first ``tokens_per_frame`` tokens, 0 elsewhere.
    """
    batch, num_tokens, _ = denoise_mask.shape
    head = mx.ones((batch, min(tokens_per_frame, num_tokens), 1), dtype=denoise_mask.dtype)
    tail = mx.zeros((batch, max(num_tokens - tokens_per_frame, 0), 1), dtype=denoise_mask.dtype)
    return mx.concatenate([head, tail], axis=1)


def extend_keyframes_mask(latent_state: LatentState, num_new_tokens: int, *, marked: bool) -> mx.array | None:
    """Extend :attr:`LatentState.keyframes_mask` to cover newly appended tokens.

    Every conditioning item that appends tokens must call this, otherwise the per-token marker
    goes out of sync with the token sequence.

    Args:
        latent_state: State being extended.
        num_new_tokens: Number of tokens the item appends.
        marked: Whether the new tokens encode a single standalone pixel frame in the
            *generated* stream. True only for generated keyframe slots; given-content tokens
            (image guidance, reference latents) are never marked, matching the reference.

    Returns:
        The extended mask, or ``None`` when there was no mask and the new tokens are unmarked.
    """
    existing = latent_state.keyframes_mask
    if existing is None and not marked:
        return None
    batch = latent_state.latent.shape[0]
    dtype = existing.dtype if existing is not None else latent_state.denoise_mask.dtype
    if existing is None:
        existing = mx.zeros((batch, latent_state.latent.shape[1], 1), dtype=dtype)
    fill = mx.ones if marked else mx.zeros
    return mx.concatenate([existing, fill((batch, num_new_tokens, 1), dtype=dtype)], axis=1)
