"""Tile schedule for the diffusion video decoder (upstream ``diffusion_tiling.py`` + ``tiling.py``).

Tiles live on the stage-4 input grid (the output of det stages 1-3). Every function here is a
verbatim transcription of the upstream formulas quoted in the design spec; keep them free of MLX
model code so they stay testable without weights.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


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
