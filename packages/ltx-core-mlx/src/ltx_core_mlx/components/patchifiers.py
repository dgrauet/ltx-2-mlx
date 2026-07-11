"""Patchification utilities for video and audio latents.

Ported from ltx-core/src/ltx_core/components/patchifiers.py
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

# VAE spatial compression: pixel dims must be multiples of this to encode cleanly.
SPATIAL_COMPRESSION = 32


class VideoLatentPatchifier:
    """Converts between spatial video latents and flat patch sequences.

    Video VAE produces latents of shape (B, C, F, H, W).
    The patchifier reshapes them to (B, F*H*W, C) for transformer input,
    and back again for decoding.

    Args:
        patch_size_t: Temporal patch size (default 1).
        patch_size_h: Height patch size (default 1).
        patch_size_w: Width patch size (default 1).
    """

    def __init__(self, patch_size_t: int = 1, patch_size_h: int = 1, patch_size_w: int = 1):
        self.patch_size_t = patch_size_t
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w

    def patchify(self, latent: mx.array) -> tuple[mx.array, tuple[int, int, int]]:
        """Convert (B, C, F, H, W) -> (B, N, C) and return spatial dims.

        Args:
            latent: Video latent of shape (B, C, F, H, W).

        Returns:
            Tuple of (tokens, (F, H, W)) where tokens is (B, F*H*W, C).
        """
        B, C, F, H, W = latent.shape
        # (B, C, F, H, W) -> (B, F, H, W, C) -> (B, F*H*W, C)
        tokens = latent.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)
        return tokens, (F, H, W)

    def unpatchify(self, tokens: mx.array, spatial_dims: tuple[int, int, int]) -> mx.array:
        """Convert (B, N, C) -> (B, C, F, H, W).

        Args:
            tokens: Flat tokens of shape (B, F*H*W, C).
            spatial_dims: Tuple of (F, H, W).

        Returns:
            Video latent of shape (B, C, F, H, W).
        """
        F, H, W = spatial_dims
        B, _N, C = tokens.shape
        # (B, F*H*W, C) -> (B, F, H, W, C) -> (B, C, F, H, W)
        return tokens.reshape(B, F, H, W, C).transpose(0, 4, 1, 2, 3)


class AudioPatchifier:
    """Converts between audio latents and flat patch sequences.

    Audio latents have shape (B, 8, T, 16) from the audio VAE.
    Flattened to (B, T, 128) for transformer.
    """

    def patchify(self, latent: mx.array) -> tuple[mx.array, int]:
        """Convert (B, 8, T, 16) -> (B, T, 128).

        Args:
            latent: Audio latent of shape (B, 8, T, 16).

        Returns:
            Tuple of (tokens, T) where tokens is (B, T, 128).
        """
        B, C1, T, C2 = latent.shape
        # (B, 8, T, 16) -> (B, T, 8, 16) -> (B, T, 128)
        tokens = latent.transpose(0, 2, 1, 3).reshape(B, T, C1 * C2)
        return tokens, T

    def unpatchify(self, tokens: mx.array, _time_dim: int | None = None) -> mx.array:
        """Convert (B, T, 128) -> (B, 8, T, 16).

        Args:
            tokens: Flat tokens of shape (B, T, 128).
            _time_dim: Unused (kept for API symmetry).

        Returns:
            Audio latent of shape (B, 8, T, 16).
        """
        B, T, _C = tokens.shape
        # (B, T, 128) -> (B, T, 8, 16) -> (B, 8, T, 16)
        return tokens.reshape(B, T, 8, 16).transpose(0, 2, 1, 3)


def compute_video_latent_shape(
    num_frames: int,
    height: int,
    width: int,
    temporal_compression: int = 8,
    spatial_compression: int = 32,
) -> tuple[int, int, int]:
    """Compute the latent spatial dimensions after VAE encoding.

    Args:
        num_frames: Number of video frames.
        height: Video height in pixels.
        width: Video width in pixels.
        temporal_compression: VAE temporal compression factor.
        spatial_compression: VAE spatial compression factor.

    Returns:
        Tuple of (F', H', W') latent dimensions.
    """
    F = (num_frames + temporal_compression - 1) // temporal_compression
    H = height // spatial_compression
    W = width // spatial_compression
    return F, H, W


def snap_output_dimensions(
    height: int,
    width: int,
    *,
    two_stage: bool,
    warn: bool = True,
) -> tuple[int, int]:
    """Snap output dimensions down to the grid the VAE + pipeline topology require.

    The video VAE encodes in blocks of ``SPATIAL_COMPRESSION`` (32) px, so every
    pipeline already floors its latent grid ``//32`` (see
    :func:`compute_video_latent_shape`). Two-stage pipelines additionally generate
    Stage 1 at half resolution (``height // 2``) and upscale ``2x``, so the *full*
    dimension must be a multiple of ``64`` for the half-res grid to itself be a
    multiple of 32. Single-stage pipelines only need multiples of ``32``.

    This snap is not a new behavior — non-conforming dims are already floored
    silently downstream (and the CLI default ``--height 480`` is not a multiple of
    64, so a bare two-stage run already snaps 480 -> 448). This helper centralizes
    the arithmetic so every pipeline reports the same output size the same way,
    rather than each silently rounding. It is a pure floor: it never raises and
    never rounds up (staying within the requested/memory budget), and it is
    idempotent, so it composes safely through nested/chained flows.

    Args:
        height: Requested output height in pixels.
        width: Requested output width in pixels.
        two_stage: True for half-res Stage-1 pipelines (modulus 64); False for
            single-stage / one-stage pipelines (modulus 32).
        warn: When True, log an informational warning (once) if either dim was
            snapped, naming the resulting output size.

    Returns:
        ``(snapped_height, snapped_width)`` — each floored to the required
        multiple and clamped to at least one tile.
    """
    modulus = 2 * SPATIAL_COMPRESSION if two_stage else SPATIAL_COMPRESSION
    snapped_height = max(modulus, (height // modulus) * modulus)
    snapped_width = max(modulus, (width // modulus) * modulus)

    if warn and (snapped_height != height or snapped_width != width):
        stage = "two-stage half-res" if two_stage else "single-stage"
        hint = " Use --single-stage for exact dims." if two_stage else ""
        logger.warning(
            "%s snaps dims to multiples of %d; output will be %dx%d (requested %dx%d).%s",
            stage,
            modulus,
            snapped_width,
            snapped_height,
            width,
            height,
            hint,
        )

    return snapped_height, snapped_width
