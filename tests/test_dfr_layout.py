"""DFR canvas layout — transcription of upstream ``dfr_layout.py`` pinned on concrete lengths."""

import pytest

from ltx_pipelines_mlx.dfr_layout import (
    SEGMENT_CANDIDATES,
    choose_segment_length,
    padding_to_segment,
    pixel_to_latent_index,
    resolve_canvas,
)


def test_constants():
    assert SEGMENT_CANDIDATES == (24, 32)


def test_padding_to_segment():
    assert padding_to_segment(48, 24) == 0
    assert padding_to_segment(48, 32) == 16
    assert padding_to_segment(136, 24) == 8
    assert padding_to_segment(136, 32) == 24


def test_choose_segment_prefers_least_padding_then_larger():
    assert choose_segment_length(48) == 24  # 24 pads 0, 32 pads 16
    assert choose_segment_length(96) == 32  # both pad 0 -> larger wins
    assert choose_segment_length(136) == 24  # 24 pads 8, 32 pads 24


@pytest.mark.parametrize(
    "num_frames,canvas,segment,positions",
    [
        (9, 25, 24, [24]),
        (25, 25, 24, [24]),
        (49, 49, 24, [24, 48]),
        (97, 97, 32, [32, 64, 96]),
        (121, 121, 24, [24, 48, 72, 96, 120]),
        (137, 145, 24, [24, 48, 72, 96, 120, 144]),
        (193, 193, 32, [32, 64, 96, 128, 160, 192]),
    ],
)
def test_resolve_canvas(num_frames, canvas, segment, positions):
    assert resolve_canvas(num_frames) == (canvas, segment, positions)
    assert (canvas - 1) % 8 == 0
    assert all(p < canvas for p in positions)


def test_resolve_canvas_rejects_off_grid_or_too_short():
    with pytest.raises(ValueError):
        resolve_canvas(10)
    with pytest.raises(ValueError):
        resolve_canvas(1)


def test_pixel_to_latent_index():
    assert pixel_to_latent_index(0) == 0
    assert pixel_to_latent_index(24) == 3
    assert pixel_to_latent_index(31) == 3
    assert pixel_to_latent_index(32) == 4
