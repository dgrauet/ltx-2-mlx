"""Video modality tiling for the LTX-2 DiT.

MLX-native port of upstream ``ltx_core.modality_tiling``. Splits the
flat patchified video token sequence into spatial/temporal tiles so
each tile can be denoised independently, then blends the tile outputs
back into the full token space with trapezoidal weights at overlaps
(or, on a seam split, rectangular weights that drop the overlap).

Combined with ``--low-ram`` (block streaming), this lets long / high-
resolution video generations fit into memory by trading wall-clock for
peak working set.

API differences vs upstream
---------------------------

Upstream operates on a ``Modality`` dataclass that bundles latent +
sigma + timesteps + positions + context + masks. Our pipeline passes
those as separate args to :meth:`LTXModel.__call__`, so
:class:`TiledLTXModel` wraps them into a :class:`Modality` at the
boundary. The helper is constructed from the latent ``(F, H, W)`` shape
instead of upstream's ``VideoLatentTools``.

Upstream positions are per-token ``[start, end)`` intervals (shape
``(B, num_axes, T, 2)``); ours are the interval midpoints (shape
``(B, T, num_axes)``). The conditioning keep test and the position
normalisation are nonetheless exact: each tile's generated extent is
rebuilt from the generated tokens' exact pixel intervals (temporal
``[max(0, 8 f0 - 7), 8 (f1 - 1) + 1) / fps`` with the causal first
frame, spatial ``[32 h0, 32 h1)``), and a conditioning token is kept
iff its midpoint lies in that **closed** extent on every axis (or its
time is negative). For every conditioning lattice the pipelines append
(single-pixel-frame keyframes / slots, 32-px cells, x2 reference cells,
8-frame reference latents) this equals upstream's
``start < tile_end and end > tile_start``.

Conditioning token bookkeeping
------------------------------

When a pipeline appends conditioning tokens to the end of the latent
(keyframe / reference video), the helper keeps each conditioning token
in every tile whose generated extent overlaps it. Cond-token
contributions from multiple tiles are weighted by
``1 / num_tiles_that_kept_this_token`` so they sum to one in the final
output.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.transformer.modality import Modality
from ltx_core_mlx.model.video_vae.tiling import (
    DimensionTilingConfig,
    SplitOperation,
    Tile,
    TileCountConfig,
    create_tiles,
    identity_mapping_operation,
    split_at_seams,
    split_by_count,
)
from ltx_core_mlx.utils.positions import VIDEO_SPATIAL_SCALE, VIDEO_TEMPORAL_SCALE

logger = logging.getLogger(__name__)

__all__ = ["TiledLTXModel", "TilingContext", "VideoModalityTiler", "seam_split"]


@dataclass(frozen=True)
class TilingContext:
    """Opaque context produced by :meth:`VideoModalityTiler.tile_modality`.

    Carries the token-level keep indices and per-conditioning-token blend
    weights needed by :meth:`VideoModalityTiler.blend`.

    Attributes:
        keep_indices: ``(num_kept,)`` int32 — indices of the tokens the tile
            processes: its generated tokens (row-major) then its kept
            conditioning tokens (ascending).
        num_total_tokens: Total number of tokens in the full (untiled) sequence.
        cond_blend_weights: ``(num_kept_cond,)`` weight per kept
            conditioning token, equal to ``1 / num_tiles_that_keep_it``.
            ``None`` when no conditioning tokens are appended.
    """

    keep_indices: mx.array
    num_total_tokens: int
    cond_blend_weights: mx.array | None


def seam_split(
    seams: Sequence[int],
    latent_frames: int,
    frames: DimensionTilingConfig,
) -> SplitOperation | None:
    """A temporal split cut on ``seams``, or ``None`` when they cannot carry one.

    Mirrors upstream ``ltx_core.modality_tiling.seam_split``. ``seams`` are interior
    latent-frame indices supplied by the caller. A boundary there needs no blending: the
    overlap is denoised for context and dropped (:func:`split_at_seams`). Leftover segments go
    to the leading tiles. Missing or non-interior seams fall back to the requested overlap
    split, which blends.

    Args:
        seams: Candidate seam cells (latent frames).
        latent_frames: Number of latent frames ``F``.
        frames: Temporal tiling config (tile count + context overlap).

    Returns:
        The seam split, or ``None`` with one frame tile or no interior seam.
    """
    if frames.num_tiles < 2:
        return None
    interior = sorted({cell for cell in seams if 0 < cell < latent_frames - 1})
    if not interior:
        if seams:
            logger.info(
                "Temporal tiling: seams %s are not interior to %d latent frames; keeping blended tiles",
                list(seams),
                latent_frames,
            )
        return None
    boundaries = [0, *interior, latent_frames - 1]
    logger.info("Temporal tiling: %d tiles cut on seams %s", frames.num_tiles, boundaries)
    return split_at_seams(boundaries, frames.num_tiles, overlap=frames.overlap)


class VideoModalityTiler:
    """Tile / blend video DiT tokens by spatial+temporal region.

    Stateless helper. Construct once with a :class:`TileCountConfig`
    and the latent ``(F, H, W)`` shape; iterate over :attr:`tiles`,
    call :meth:`tile_modality` to slice out each tile's sub-modality, run the
    DiT on it, then accumulate the result via :meth:`blend`.

    Args:
        tiling: ``TileCountConfig`` describing tile counts + overlap
            per dimension.
        latent_shape: ``(F, H, W)`` of the patchified token grid.
        seams: Interior latent-frame indices a temporal split may land on instead
            of blended overlaps (:func:`seam_split`); missing or non-interior seams
            keep the blended split. A regular keyframe belongs in that list only at
            strength 0; generated keyframe slots never do.

    Notes:
        ``F``/``H``/``W`` are token-grid units, not pixel units —
        they are the values returned by
        :func:`ltx_core_mlx.components.patchifiers.compute_video_latent_shape`.
    """

    def __init__(
        self,
        tiling: TileCountConfig,
        latent_shape: tuple[int, int, int],
        seams: Sequence[int] = (),
    ) -> None:
        self._latent_shape = latent_shape
        F, H, W = latent_shape
        self._num_generated_tokens = F * H * W
        frames = split_by_count(tiling.frames.num_tiles, tiling.frames.overlap)
        frames_mapper = identity_mapping_operation
        seam_op = seam_split(seams, F, tiling.frames)
        if seam_op is not None:
            frames, frames_mapper = seam_op, partial(identity_mapping_operation, rectangular=True)
        self._tiles: list[Tile] = create_tiles(
            (F, H, W),
            splitters=[
                frames,
                split_by_count(tiling.height.num_tiles, tiling.height.overlap),
                split_by_count(tiling.width.num_tiles, tiling.width.overlap),
            ],
            mappers=[frames_mapper, identity_mapping_operation, identity_mapping_operation],
        )
        # Exact generated extent of each tile in pixel units, (num_tiles, 3) [time, h, w]:
        # the union of its generated tokens' intervals (temporal causal fix on frame 0).
        starts, ends = [], []
        for tile in self._tiles:
            f, h, w = tile.in_coords
            starts.append(
                [
                    max(0, VIDEO_TEMPORAL_SCALE * f.start - 7),
                    VIDEO_SPATIAL_SCALE * h.start,
                    VIDEO_SPATIAL_SCALE * w.start,
                ]
            )
            ends.append(
                [VIDEO_TEMPORAL_SCALE * (f.stop - 1) + 1, VIDEO_SPATIAL_SCALE * h.stop, VIDEO_SPATIAL_SCALE * w.stop]
            )
        self._extent_starts_px = np.asarray(starts, dtype=np.float32)
        self._extent_ends_px = np.asarray(ends, dtype=np.float32)

    @property
    def tiles(self) -> list[Tile]:
        """All tiles for the configured layout (call :meth:`tile_modality` per tile)."""
        return self._tiles

    @property
    def num_generated_tokens(self) -> int:
        """Number of generated (non-conditioning) tokens in the full sequence."""
        return self._num_generated_tokens

    def _tile_index(self, tile: Tile) -> int:
        tile_idx = next((i for i, t in enumerate(self._tiles) if t.in_coords == tile.in_coords), None)
        if tile_idx is None:
            raise RuntimeError(
                f"Tile with in_coords={tile.in_coords} is not in this helper's tile set; "
                f"pass a tile obtained from `tiler.tiles`."
            )
        return tile_idx

    def _tile_generated_token_count(self, tile: Tile) -> int:
        f, h, w = tile.in_coords
        return (f.stop - f.start) * (h.stop - h.start) * (w.stop - w.start)

    def _generated_token_indices(self, tile: Tile) -> mx.array:
        """Flat indices of the tile's generated tokens in the full sequence."""
        _, H, W = self._latent_shape
        f, h, w = tile.in_coords
        f_idx = mx.arange(f.start, f.stop)
        h_idx = mx.arange(h.start, h.stop)
        w_idx = mx.arange(w.start, w.stop)
        return (f_idx[:, None, None] * H * W + h_idx[None, :, None] * W + w_idx[None, None, :]).reshape(-1)

    def _tile_extents(self, positions: mx.array) -> tuple[mx.array, mx.array]:
        """Per-tile generated extents in position units, each ``(num_tiles, B, 3)``.

        Pixel extents are converted to seconds on the temporal axis with the frame
        duration recovered from the first generated token, whose midpoint
        ``compute_video_positions`` puts at ``0.5 / fps`` (so ``1 / fps = 2 * t0``).
        """
        frame_duration = 2.0 * positions[:, 0, 0]  # (B,)
        if not bool(mx.all(frame_duration > 0).item()):
            raise ValueError(
                "VideoModalityTiler expects pixel-space video positions whose first generated token sits at "
                f"0.5 / fps (compute_video_positions); got t0={np.asarray(positions[:, 0, 0]).tolist()}"
            )
        ones = mx.ones_like(frame_duration)
        scale = mx.stack([frame_duration, ones, ones], axis=-1)  # (B, 3)
        starts = mx.array(self._extent_starts_px)[:, None, :] * scale[None]
        ends = mx.array(self._extent_ends_px)[:, None, :] * scale[None]
        return starts, ends

    def _all_tiles_cond_keep(self, positions: mx.array) -> np.ndarray:
        """Vectorised ``(num_tiles, num_cond)`` bool: which tiles keep each conditioning token.

        A conditioning token is kept by a tile when its midpoint lies in the tile's closed
        generated extent on all three axes (upstream: its ``[start, end)`` interval overlaps
        the tile), or when it has a negative time coordinate (reference token).
        """
        starts, ends = self._tile_extents(positions)  # (num_tiles, B, 3)
        cond = positions[:, self._num_generated_tokens :, :]  # (B, num_cond, 3)
        inside = (cond[None] >= starts[:, :, None, :]) & (cond[None] <= ends[:, :, None, :])
        keep = inside.all(axis=-1) | (cond[None, :, :, 0] < 0)  # (num_tiles, B, num_cond)
        return np.asarray(keep.any(axis=1))

    def tile_modality(
        self,
        modality: Modality,
        tile: Tile,
        normalize_positions: bool = True,
    ) -> tuple[Modality, TilingContext]:
        """Slice ``modality`` to the tokens covered by ``tile``.

        Mirrors upstream ``VideoModalityTilingHelper.tile_modality``
        signature. Returns a new :class:`Modality` for the tile + an
        opaque :class:`TilingContext` to pass back to :meth:`blend`.

        Args:
            modality: input modality. ``modality.latent``, ``timesteps``,
                and ``positions`` are sliced to the tile; ``sigma``,
                ``context``, ``context_mask``, and ``enabled`` are
                forwarded unchanged. ``attention_mask``, when present,
                is reduced to the kept tokens x kept tokens submatrix.
            tile: which tile to extract (one of :attr:`tiles`).
            normalize_positions: when True, shift all positions so the
                tile's generated tokens' intervals start at zero on every
                axis (upstream semantics; midpoints keep their half-cell).

        Returns:
            ``(tiled_modality, context)``. ``context`` carries the
            keep indices and per-cond-token blend weights needed by
            :meth:`blend`.
        """
        latent = modality.latent
        positions = modality.positions
        attention_mask = modality.attention_mask

        num_total = latent.shape[1]
        gen_idx = self._generated_token_indices(tile)
        tile_idx = self._tile_index(tile)

        cond_blend_weights: mx.array | None = None
        if num_total > self._num_generated_tokens:
            keep_per_tile_cond = self._all_tiles_cond_keep(positions)  # (num_tiles, num_cond)
            my_cond_keep = keep_per_tile_cond[tile_idx]
            cond_idx = mx.array(self._num_generated_tokens + np.flatnonzero(my_cond_keep), dtype=gen_idx.dtype)
            keep_idx = mx.concatenate([gen_idx, cond_idx])
            total_keepers = keep_per_tile_cond.sum(axis=0).astype(np.float32)  # (num_cond,)
            cond_blend_weights = mx.array(1.0 / total_keepers[my_cond_keep])
        else:
            keep_idx = gen_idx

        tiled_latent = latent[:, keep_idx, :]
        tiled_positions = positions[:, keep_idx, :]
        tiled_timesteps = modality.timesteps[:, keep_idx]
        if normalize_positions:
            starts, _ = self._tile_extents(positions)
            tiled_positions = tiled_positions - starts[tile_idx][:, None, :]

        tiled_attention_mask: mx.array | None = None
        if attention_mask is not None:
            tiled_attention_mask = attention_mask[:, keep_idx, :][:, :, keep_idx]
        tiled_keyframes_mask: mx.array | None = None
        if modality.keyframes_mask is not None:
            tiled_keyframes_mask = modality.keyframes_mask[:, keep_idx, :]

        tiled = Modality(
            latent=tiled_latent,
            sigma=modality.sigma,
            timesteps=tiled_timesteps,
            positions=tiled_positions,
            context=modality.context,
            enabled=modality.enabled,
            context_mask=modality.context_mask,
            attention_mask=tiled_attention_mask,
            keyframes_mask=tiled_keyframes_mask,
        )
        return tiled, TilingContext(
            keep_indices=keep_idx, num_total_tokens=num_total, cond_blend_weights=cond_blend_weights
        )

    def blend(
        self,
        tile_output: mx.array,
        tile: Tile,
        ctx: TilingContext,
        output: mx.array | None = None,
    ) -> mx.array:
        """Blend-weight the tile result and accumulate into the full token buffer.

        The tile's generated-token output is multiplied by the tile's
        per-token blend mask before being added to the output buffer at
        the matching positions. Conditioning-token output is multiplied by
        ``ctx.cond_blend_weights`` (so summed contributions from all tiles
        equal 1).

        Args:
            tile_output: ``(B, num_tile_tokens, D)`` from the model.
                The first ``num_tile_gen`` rows are generated tokens
                (in the tile's order), the remainder are kept cond
                tokens (in their original order in the full sequence).
            tile: the :class:`Tile` used in :meth:`tile_modality`.
            ctx: the :class:`TilingContext` from :meth:`tile_modality`.
            output: optional pre-allocated ``(B, num_total, D)`` buffer
                to accumulate into. ``None`` allocates a fresh
                zero-filled buffer.

        Returns:
            The output buffer with the tile's contribution added.
        """
        B, _, D = tile_output.shape
        num_total = ctx.num_total_tokens
        if output is None:
            output = mx.zeros((B, num_total, D), dtype=tile_output.dtype)
        elif output.shape != (B, num_total, D):
            raise ValueError(f"output shape mismatch: expected {(B, num_total, D)}, got {output.shape}")

        num_tile_gen = self._tile_generated_token_count(tile)
        gen_idx = self._generated_token_indices(tile)
        blend_mask = tile.blend_mask.reshape(-1).astype(tile_output.dtype)

        gen_part = tile_output[:, :num_tile_gen, :] * blend_mask[None, :, None]
        output[:, gen_idx, :] = output[:, gen_idx, :] + gen_part

        if ctx.cond_blend_weights is not None and ctx.cond_blend_weights.size > 0:
            cond_idx_full = ctx.keep_indices[num_tile_gen:]
            weights = ctx.cond_blend_weights.astype(tile_output.dtype)
            cond_part = tile_output[:, num_tile_gen:, :] * weights[None, :, None]
            output[:, cond_idx_full, :] = output[:, cond_idx_full, :] + cond_part

        return output


class TiledLTXModel:
    """Drop-in LTXModel wrapper that tiles the video forward across spatial/temporal regions.

    Iterates over :attr:`VideoModalityTiler.tiles`; for each tile it
    slices the video-relevant args (latent, positions, attention_mask,
    optional per-token timesteps) and calls the wrapped model with the
    tiled video + the full audio. Outputs are accumulated:

    - Video velocity / x0: blended via :meth:`VideoModalityTiler.blend`
      with trapezoidal weights at overlaps.
    - Audio velocity / x0: averaged across tiles (the audio path is
      replicated in each tile call, so per-tile outputs differ only in
      the joint audio↔video cross-attention contribution).

    Composes with :class:`~ltx_core_mlx.loader.block_streaming.StreamingLTXModel`:
    wrap the dev/distilled LTXModel in TiledLTXModel, then optionally
    in StreamingLTXModel (or vice versa — order doesn't matter, both
    intercept ``__call__`` and forward to ``self.inner``).

    Args:
        inner: An ``LTXModel`` (or another wrapper around it) — anything
            whose ``__call__`` signature matches LTXModel's.
        tiler: A pre-built :class:`VideoModalityTiler`.
        normalize_positions: Forwarded to :meth:`VideoModalityTiler.tile_modality`:
            shift each tile's positions so its generated intervals start at zero.
            Default False (the ``--tile-*`` behaviour).
    """

    def __init__(self, inner, tiler: VideoModalityTiler, normalize_positions: bool = False) -> None:
        self._inner = inner
        self._tiler = tiler
        self._normalize_positions = normalize_positions

    def __call__(self, *args, **kwargs):
        if args:
            raise TypeError("TiledLTXModel expects keyword arguments only")

        # Adapt: build a video Modality from per-arg kwargs. Our
        # pipelines pass (latent, positions, mask, ...) separately;
        # the tiler API takes Modality (isomorphic with upstream).
        # This boundary-layer adapter wraps then unwraps.
        video_modality = self._build_video_modality(kwargs)

        video_out: mx.array | None = None
        audio_outs: list[mx.array] = []

        for tile in self._tiler.tiles:
            tiled_modality, ctx = self._tiler.tile_modality(
                video_modality, tile, normalize_positions=self._normalize_positions
            )

            tile_kwargs = dict(kwargs)
            tile_kwargs["video_latent"] = tiled_modality.latent
            tile_kwargs["video_positions"] = tiled_modality.positions
            tile_kwargs["video_attention_mask"] = tiled_modality.attention_mask
            tile_kwargs["video_keyframes_mask"] = tiled_modality.keyframes_mask
            if "video_timesteps" in kwargs and kwargs["video_timesteps"] is not None:
                tile_kwargs["video_timesteps"] = tiled_modality.timesteps

            tile_video_out, tile_audio_out = self._inner(**tile_kwargs)

            video_out = self._tiler.blend(tile_video_out, tile, ctx, output=video_out)
            audio_outs.append(tile_audio_out)

        if len(audio_outs) == 1:
            audio_out = audio_outs[0]
        else:
            audio_out = mx.mean(mx.stack(audio_outs, axis=0), axis=0)

        return video_out, audio_out

    @staticmethod
    def _build_video_modality(kwargs: dict) -> Modality:
        """Adapter: assemble a video Modality from LTXModel.__call__ kwargs.

        Fills missing per-token timesteps from the scalar ``timestep``
        when not supplied. Defaults context_mask to None.
        """
        latent = kwargs["video_latent"]
        positions = kwargs.get("video_positions")
        attention_mask = kwargs.get("video_attention_mask")
        keyframes_mask = kwargs.get("video_keyframes_mask")
        timesteps = kwargs.get("video_timesteps")
        sigma = kwargs.get("timestep")
        context = kwargs.get("video_text_embeds")

        if positions is None:
            raise ValueError("TiledLTXModel requires video_positions to be provided.")
        if sigma is None:
            raise ValueError("TiledLTXModel requires timestep to be provided.")

        if timesteps is None:
            # Broadcast scalar sigma to per-token timesteps so the
            # tiler can slice them with the keep_mask just like the
            # latent.
            timesteps = mx.broadcast_to(sigma[:, None], (latent.shape[0], latent.shape[1]))

        return Modality(
            latent=latent,
            sigma=sigma,
            timesteps=timesteps,
            positions=positions,
            context=context if context is not None else mx.zeros((latent.shape[0], 0, 0), dtype=latent.dtype),
            enabled=True,
            context_mask=None,
            attention_mask=attention_mask,
            keyframes_mask=keyframes_mask,
        )

    def __getattr__(self, name: str):
        # Proxy other attribute reads (e.g. ``self.config``) to the inner model.
        if name in {"_inner", "_tiler", "_normalize_positions"}:
            raise AttributeError(name)
        try:
            inner = object.__getattribute__(self, "_inner")
        except AttributeError as e:
            raise AttributeError(name) from e
        return getattr(inner, name)
