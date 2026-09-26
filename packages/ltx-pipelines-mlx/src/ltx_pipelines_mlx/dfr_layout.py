"""DFR canvas layout (upstream ``ltx_pipelines/dfr_layout.py``).

The DFR base stage pads the clip to a whole number of keyframe segments and puts one generated
keyframe slot at every segment boundary. ``TemporalTilePlan`` (below) partitions a temporal-round
canvas into keyframe-seam tiles for the temporal-upsample rounds.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import NamedTuple, overload

from ltx_core_mlx.model.video_vae.tiling import split_at_seams as core_split_at_seams

#: Candidate keyframe segment lengths in pixel frames (upstream ``SEGMENT_CANDIDATES``).
SEGMENT_CANDIDATES: tuple[int, ...] = (24, 32)
#: Pixel frames per latent frame of the video VAE.
_TEMPORAL_SCALE = 8


def padding_to_segment(content_frames: int, segment: int) -> int:
    """Frames to add so ``content_frames`` becomes a multiple of ``segment``."""
    return (-content_frames) % segment


def choose_segment_length(content_frames: int) -> int:
    """The candidate segment that needs the least padding; the larger one on ties."""
    return min(SEGMENT_CANDIDATES, key=lambda segment: (padding_to_segment(content_frames, segment), -segment))


def resolve_canvas(num_frames: int, *, temporal_scale: int = _TEMPORAL_SCALE) -> tuple[int, int, list[int]]:
    """Pad ``num_frames`` to whole keyframe segments and place one slot per segment boundary.

    Args:
        num_frames: Requested clip length on the VAE grid (``(num_frames - 1) % temporal_scale == 0``).
        temporal_scale: Pixel frames per latent frame.

    Returns:
        ``(canvas_frames, segment, positions)``: the padded length (also on the grid, since both
        candidates are multiples of 8), the chosen segment, and the slot pixel-frame indices
        ``[segment, 2 * segment, ...]`` up to the padded content length.

    Raises:
        ValueError: ``num_frames`` is below 9 or off the ``1 + 8k`` grid.
    """
    if num_frames < 1 + temporal_scale or (num_frames - 1) % temporal_scale:
        raise ValueError(f"num_frames must be 1 + {temporal_scale}k with k >= 1, got {num_frames}")
    content = num_frames - 1
    segment = choose_segment_length(content)
    content_padded = content + padding_to_segment(content, segment)
    positions = [segment * index for index in range(1, content_padded // segment + 1)]
    return content_padded + 1, segment, positions


def pixel_to_latent_index(pixel_frame: int, temporal_scale: int = _TEMPORAL_SCALE) -> int:
    """Map an x8-border pixel frame to its latent index (upstream ``pixel_to_latent_index``).

    Raises:
        ValueError: negative, or not on the ``x temporal_scale`` latent border (frame 0 excepted).
    """
    if pixel_frame < 0:
        raise ValueError(f"pixel_frame must be >= 0, got {pixel_frame}")
    if pixel_frame != 0 and pixel_frame % temporal_scale != 0:
        raise ValueError(f"pixel_frame {pixel_frame} is not on the x{temporal_scale} latent border")
    return pixel_frame // temporal_scale


class TemporalInterval(NamedTuple):
    """One tile's latent-frame interval ``[start, end)``; ``left_ramp`` leading cells are context only."""

    start: int
    end: int
    left_ramp: int


def split_at_seams(boundaries: Sequence[int], num_tiles: int, overlap: int, dim_size: int) -> list[TemporalInterval]:
    """Split a dimension on known boundary cells (upstream ``ltx_core.tiling.split_at_seams``).

    ``boundaries`` are the ``K + 1`` segment edges (grid cells), starting at 0 and ending at the last
    cell. Segments are dealt so leftovers go to the leading tiles; ``num_tiles`` above ``K`` is clamped.
    Every tile but the first starts ``overlap`` cells before the cell it resumes at; that lead-in is its
    ``left_ramp`` (context only, dropped when stitching), so the earlier tile keeps the boundary cell.
    Delegates to the core :func:`ltx_core_mlx.model.video_vae.tiling.split_at_seams`.

    Raises:
        ValueError: invalid ``num_tiles`` / ``overlap`` / boundaries, or boundaries not ending at ``dim_size - 1``.
    """
    intervals = core_split_at_seams(boundaries, num_tiles, overlap)(dim_size)
    return [
        TemporalInterval(start=start, end=end, left_ramp=left_ramp)
        for start, end, left_ramp in zip(intervals.starts, intervals.ends, intervals.left_ramps, strict=True)
    ]


def split_canvas_at_seams(seams: Sequence[int], num_tiles: int, overlap: int, dim_size: int) -> list[TemporalInterval]:
    """Split a DFR canvas on keyframe boundary cells (upstream ``split_canvas_at_seams``)."""
    return split_at_seams(seams, num_tiles, overlap, dim_size)


class TemporalTile(NamedTuple):
    """One temporal-upsample window: latent interval plus pixel-frame anchors and slots."""

    interval: TemporalInterval
    pixel_start: int
    pixel_end: int
    anchors: tuple[int, ...]
    slots: tuple[int, ...]


class TemporalTilePlan:
    """Keyframe-seam tiles for one temporal-upsample round (upstream ``TemporalTilePlan``).

    Overlap is one canvas segment in latent cells plus the shared seam cell. Anchors are the seams
    inside ``[pixel_start, pixel_end]``; slots are the midpoints between consecutive marks
    ``[pixel_start, seams in (pixel_start, pixel_end]]``.
    """

    tiles: tuple[TemporalTile, ...]

    def __init__(
        self, seam_positions: Sequence[int], num_frames: int, num_tiles: int, temporal_scale: int = _TEMPORAL_SCALE
    ) -> None:
        """Partition the canvas and attach per-window anchors and slots.

        Args:
            seam_positions: Pixel-frame seams on this round's grid (multiples of ``temporal_scale``).
            num_frames: Pixel frames of this round's canvas.
            num_tiles: Requested tile count (``2**round``), clamped to the segment count.
            temporal_scale: Pixel frames per latent frame.
        """
        seams = [0, *(pixel_to_latent_index(position, temporal_scale) for position in seam_positions)]
        latent_len = (num_frames - 1) // temporal_scale + 1
        overlap = (seams[1] - seams[0]) + 1 if len(seams) > 1 else 0
        tiles: list[TemporalTile] = []
        for interval in split_canvas_at_seams(seams, num_tiles, overlap, latent_len):
            pixel_start = interval.start * temporal_scale
            pixel_end = (interval.end - 1) * temporal_scale
            anchors = tuple(p for p in seam_positions if pixel_start <= p <= pixel_end)
            marks = [pixel_start, *[p for p in seam_positions if pixel_start < p <= pixel_end]]
            slots = tuple((left + right) // 2 for left, right in itertools.pairwise(marks))
            tiles.append(TemporalTile(interval, pixel_start, pixel_end, anchors, slots))
        self.tiles = tuple(tiles)

    def __iter__(self) -> Iterator[TemporalTile]:
        """Iterate over the tiles in time order."""
        return iter(self.tiles)

    def __len__(self) -> int:
        """Number of tiles."""
        return len(self.tiles)

    @overload
    def __getitem__(self, index: int) -> TemporalTile: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TemporalTile, ...]: ...

    def __getitem__(self, index: int | slice) -> TemporalTile | tuple[TemporalTile, ...]:
        """Tile(s) by index."""
        return self.tiles[index]
