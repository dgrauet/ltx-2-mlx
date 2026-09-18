"""Tile schedule for the diffusion video decoder (upstream ``diffusion_tiling.py`` + ``tiling.py``).

Tiles live on the stage-4 input grid (the output of det stages 1-3). Every function here is a
verbatim transcription of the upstream formulas quoted in the design spec; keep them free of MLX
model code so they stay testable without weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ltx_core_mlx.model.video_vae.diffusion_decoder.config import DiffusionDecoderConfig, Kernel


@dataclass(frozen=True)
class Interval:
    """Half-open ``[start, end)`` span on one axis with fade-in / fade-out ramp lengths.

    Attributes:
        start: First index (inclusive).
        end: Last index (exclusive).
        left_ramp: Fade-in length at the start of the span.
        right_ramp: Fade-out length at the end of the span.
    """

    start: int
    end: int
    left_ramp: int = 0
    right_ramp: int = 0

    @property
    def length(self) -> int:
        """Length of the interval: ``end - start``."""
        return self.end - self.start


def _grow_last_tile_to_min(intervals: list[Interval], min_tile_size: int) -> list[Interval]:
    """Extend a too-short last interval backwards to ``min_tile_size`` (upstream ``tl:140-154``)."""
    if len(intervals) < 2 or intervals[-1].length >= min_tile_size:
        return intervals
    prev, last = intervals[-2], intervals[-1]
    new_start = last.end - min_tile_size
    new_overlap = prev.end - new_start
    return [*intervals[:-2], replace(prev, right_ramp=new_overlap), Interval(new_start, last.end, new_overlap, 0)]


def _validate_intervals(intervals: list[Interval], length: int, min_tile_size: int | None) -> None:
    """Coverage, minimum length, ramp bounds and consistent overlaps (upstream ``tl:157-171``)."""
    if intervals[0].start != 0 or intervals[-1].end != length:
        raise ValueError(f"intervals {intervals} do not cover [0, {length})")
    for iv in intervals:
        if iv.length <= 0 or iv.left_ramp > iv.length or iv.right_ramp > iv.length:
            raise ValueError(f"invalid interval {iv}")
        if min_tile_size is not None and len(intervals) > 1 and iv.length < min_tile_size:
            raise ValueError(f"interval {iv} shorter than min tile size {min_tile_size}")
    for prev, cur in zip(intervals, intervals[1:], strict=False):
        overlap = prev.end - cur.start
        if overlap != prev.right_ramp or overlap != cur.left_ramp:
            raise ValueError(f"inconsistent overlap between {prev} and {cur}")


def split_by_size(length: int, size: int, overlap: int, min_tile_size: int | None = None) -> list[Interval]:
    """Split ``[0, length)`` into overlapping intervals of ``size`` (upstream ``tl:174-223``).

    Args:
        length: Axis length in stage-4 cells.
        size: Tile size in cells (``> 0``).
        overlap: Overlap between consecutive tiles in cells (``0 <= overlap < size``).
        min_tile_size: When set, an axis shorter than it is not split and a short last
            interval is grown backwards to this length.

    Returns:
        Intervals covering the axis; a single ``[0, length)`` interval when no split applies.

    Raises:
        ValueError: Bad arguments, or a grown last tile that breaks overlap consistency.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if min_tile_size is not None and min_tile_size < 1:
        raise ValueError("min_tile_size must be >= 1")
    if min_tile_size is not None and length < min_tile_size:
        return [Interval(0, length)]
    if length <= size:
        return [Interval(0, length)]
    n = (length + size - 2 * overlap - 1) // (size - overlap)
    intervals: list[Interval] = []
    for i in range(n):
        start = i * (size - overlap)
        if i == 0:
            intervals.append(Interval(0, size, 0, overlap))
        elif i < n - 1:
            intervals.append(Interval(start, start + size, overlap, overlap))
        else:
            intervals.append(Interval(start, length, overlap, 0))
    if min_tile_size is not None:
        intervals = _grow_last_tile_to_min(intervals, min_tile_size)
    _validate_intervals(intervals, length, min_tile_size)
    return intervals


def propagate_temporal(iv: Interval, stride: int) -> Interval:
    """Map a temporal interval through one causal pixel-shuffle hop (upstream ``dt:872-900``).

    With ``stride == 2`` the duplicated leading frame is dropped: ``end -= 1`` always, and
    ``start -= 1`` for every interval that does not start at 0, so non-origin tiles line up
    with the origin tile's frame indexing.
    """
    start, end = iv.start * stride, iv.end * stride
    if stride == 2:
        end -= 1
        if iv.start != 0:
            start -= 1
    return Interval(start, end, iv.left_ramp * stride, iv.right_ramp * stride)


def propagate_spatial(iv: Interval, stride: int) -> Interval:
    """Scale a spatial interval and its ramps by ``stride`` (non-causal hop)."""
    return Interval(iv.start * stride, iv.end * stride, iv.left_ramp * stride, iv.right_ramp * stride)


def round_up(value: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is ``>= value``."""
    return -(-value // multiple) * multiple


def padded_latent_fhw(cfg: DiffusionDecoderConfig, fhw: tuple[int, int, int]) -> tuple[int, int, int]:
    """Latent ``(F, H, W)`` after :meth:`NADiffusionDecoder.pad_to_floor` (each axis at least the floor)."""
    f_min, h_min, w_min = cfg.min_latent_shape()
    return (max(fhw[0], f_min), max(fhw[1], h_min), max(fhw[2], w_min))


@dataclass(frozen=True)
class DiffusionTileGeometry:
    """Everything the tile schedule needs from the decoder config (upstream ``dt:711-821``).

    Attributes:
        strides: Upsample strides of the four hops.
        pixel_scale: Pixels per stage-4 cell ``(t, h, w)`` = ``(st3, sh3 * patch, sw3 * patch)``.
        latent_scale: Pixels per latent cell ``(8, 32, 32)`` in production.
        patch_size: Stage-5 spatial patch.
        min_tile_s4: Minimum tile per axis in stage-4 cells ``max(k4, ceil(k5 / up3))``.
        halo4: Stage-4 receptive halo ``depth4 * (k4 // 2)``.
        halo5: Stage-5 halo in stage-4 cells ``ceil(depth5 * (k5 // 2) / up3)``.
        overlap_frames: Recommended temporal overlap in pixel frames.
        overlap_px: Recommended spatial overlap in pixels (shared by H and W).
        min_tile_frames: Smallest AUTO tile in frames.
        min_tile_px: Smallest AUTO tile in pixels.
        step_frames: AUTO candidate grid step in frames.
        step_px: AUTO candidate grid step in pixels.
        ghost_frames_s4: Ghost frames appended to the stage-4 feature by the trailing pad.
        stage4_channels: Channels of the stage-4 feature (memory model).
        stage5_channels: Channels of the diffusion stage (memory model).
    """

    strides: tuple[Kernel, Kernel, Kernel, Kernel]
    pixel_scale: Kernel
    latent_scale: Kernel
    patch_size: int
    min_tile_s4: Kernel
    halo4: Kernel
    halo5: Kernel
    overlap_frames: int
    overlap_px: int
    min_tile_frames: int
    min_tile_px: int
    step_frames: int
    step_px: int
    ghost_frames_s4: int
    stage4_channels: int
    stage5_channels: int

    @classmethod
    def from_config(cls, cfg: DiffusionDecoderConfig) -> DiffusionTileGeometry:
        """Derive the geometry from a decoder config (production values in the module docstring)."""
        strides = tuple(s for s, _ in cfg.upsamples)
        k4, k5, up3, p = cfg.stage_kernels[3], cfg.stage5_kernel, strides[3], cfg.patch_size
        pixel_scale = (up3[0], up3[1] * p, up3[2] * p)
        full = cfg.cumulative_strides()[4]
        latent_scale = (full[0], full[1] * p, full[2] * p)
        min_tile = tuple(max(k4[a], math.ceil(k5[a] / up3[a])) for a in range(3))
        halo4 = tuple(cfg.stage_depths[3] * (k4[a] // 2) for a in range(3))
        halo5 = tuple(math.ceil(cfg.diff_depth * (k5[a] // 2) / up3[a]) for a in range(3))
        dom = tuple(max(halo4[a], halo5[a]) for a in range(3))
        overlap_frames = round_up(dom[0] * pixel_scale[0], latent_scale[0])
        overlap_px = round_up(max(dom[1], dom[2]) * pixel_scale[1], latent_scale[1])
        step_frames = math.lcm(pixel_scale[0], latent_scale[0])
        step_px = math.lcm(pixel_scale[1], latent_scale[1])
        min_tile_frames = round_up(
            max(
                2 * pixel_scale[0],
                2 * overlap_frames,
                round_up(min_tile[0] * pixel_scale[0], pixel_scale[0]),
                2 * latent_scale[0],
            ),
            step_frames,
        )
        min_tile_px = round_up(
            max(
                2 * pixel_scale[1],
                2 * overlap_px,
                round_up(min_tile[1] * pixel_scale[1], pixel_scale[1]),
                2 * latent_scale[1],
            ),
            step_px,
        )
        return cls(
            strides=strides,  # type: ignore[arg-type]
            pixel_scale=pixel_scale,
            latent_scale=latent_scale,
            patch_size=p,
            min_tile_s4=min_tile,  # type: ignore[arg-type]
            halo4=halo4,  # type: ignore[arg-type]
            halo5=halo5,  # type: ignore[arg-type]
            overlap_frames=overlap_frames,
            overlap_px=overlap_px,
            min_tile_frames=min_tile_frames,
            min_tile_px=min_tile_px,
            step_frames=step_frames,
            step_px=step_px,
            ghost_frames_s4=cfg.ghost_pad_frames() * cfg.cumulative_strides()[3][0],
            stage4_channels=cfg.stage_channels[3],
            stage5_channels=cfg.stage_channels[4],
        )

    def stage4_content_thw(self, f: int, h: int, w: int) -> Kernel:
        """Stage-4 input grid of a (padded, not ghost-padded) latent (upstream ``dt:711-725``)."""
        t = f
        for st, sh, sw in self.strides[:3]:
            t, h, w = t * st, h * sh, w * sw
            if st == 2:
                t -= 1
        return (t, h, w)

    def intervals_for_axis(self, axis: int, length: int, tile_px: int, overlap_px: int) -> list[Interval]:
        """Intervals (stage-4 cells) for one axis; ``tile_px == 0`` leaves the axis untiled (``tl:860-897``)."""
        if tile_px == 0:
            return [Interval(0, length)]
        factor = self.pixel_scale[axis]
        size, overlap = tile_px // factor, overlap_px // factor
        tile = max(2, overlap + 1, size)
        return split_by_size(length, tile, overlap, self.min_tile_s4[axis])


@dataclass(frozen=True)
class DiffusionTileConfig:
    """Tile sizes and overlaps in pixel units (frames / pixels); ``0`` size = axis untiled."""

    tile_frames: int
    overlap_frames: int
    tile_px_h: int
    overlap_px_h: int
    tile_px_w: int
    overlap_px_w: int

    def axis(self, axis: int) -> tuple[int, int]:
        """``(tile, overlap)`` of axis 0 (t), 1 (h) or 2 (w)."""
        return [
            (self.tile_frames, self.overlap_frames),
            (self.tile_px_h, self.overlap_px_h),
            (self.tile_px_w, self.overlap_px_w),
        ][axis]

    def validate(self, geometry: DiffusionTileGeometry, *, allow_small_overlap: bool = False) -> DiffusionTileConfig:
        """Check grid alignment, minimum size and (unless ``allow_small_overlap``) the recommended overlap."""
        names = ("frames", "height", "width")
        recommended = (geometry.overlap_frames, geometry.overlap_px, geometry.overlap_px)
        for a in range(3):
            tile, overlap = self.axis(a)
            factor = geometry.pixel_scale[a]
            if tile == 0:
                continue
            if tile % factor or overlap % factor:
                raise ValueError(
                    f"diffusion decoder tile {names[a]}: size {tile} and overlap {overlap} must be multiples of {factor}"
                )
            if tile < 2 * factor:
                raise ValueError(f"diffusion decoder tile {names[a]}: size {tile} must be at least {2 * factor}")
            if overlap >= tile:
                raise ValueError(
                    f"diffusion decoder tile {names[a]}: overlap {overlap} must be smaller than the tile size {tile}"
                )
            if not allow_small_overlap and overlap < recommended[a]:
                raise ValueError(
                    f"diffusion decoder tile {names[a]}: overlap {overlap} is below the recommended {recommended[a]}"
                )
        return self

    @classmethod
    def from_pixels(cls, geometry: DiffusionTileGeometry, frames: int, height: int, width: int) -> DiffusionTileConfig:
        """Build a config with the recommended overlaps from explicit tile sizes (``0`` = axis untiled)."""
        ov_t = geometry.overlap_frames if frames else 0
        ov_h = geometry.overlap_px if height else 0
        ov_w = geometry.overlap_px if width else 0
        return cls(frames, ov_t, height, ov_h, width, ov_w).validate(geometry)
