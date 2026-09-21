"""DFR canvas layout (upstream ``ltx_pipelines/dfr_layout.py``, base path only).

The DFR base stage pads the clip to a whole number of keyframe segments and puts one generated
keyframe slot at every segment boundary. Temporal-round helpers (``TemporalTilePlan``) are not
ported until the temporal rounds are (sub-project 4c).
"""

from __future__ import annotations

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
    """Latent frame holding ``pixel_frame`` (upstream ``pixel_frame // temporal_scale``)."""
    return pixel_frame // temporal_scale
