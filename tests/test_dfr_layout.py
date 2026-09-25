"""DFR canvas layout — transcription of upstream ``dfr_layout.py`` pinned on concrete lengths."""

import pytest

from ltx_pipelines_mlx.dfr_layout import (
    SEGMENT_CANDIDATES,
    TemporalInterval,
    TemporalTilePlan,
    choose_segment_length,
    padding_to_segment,
    pixel_to_latent_index,
    resolve_canvas,
    split_at_seams,
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
    assert pixel_to_latent_index(32) == 4
    with pytest.raises(ValueError, match="x8"):
        pixel_to_latent_index(31)
    with pytest.raises(ValueError, match=">= 0"):
        pixel_to_latent_index(-8)


def test_split_at_seams_leftover_goes_to_leading_tiles():
    # 5 segments on 2 tiles -> counts [3, 2]; second tile resumes after cell 18, 7 cells of lead-in
    assert split_at_seams([0, 6, 12, 18, 24, 30], 2, 7, 31) == [
        TemporalInterval(0, 19, 0),
        TemporalInterval(12, 31, 7),
    ]


def test_split_at_seams_validates():
    with pytest.raises(ValueError, match="num_tiles"):
        split_at_seams([0, 6], 0, 0, 7)
    with pytest.raises(ValueError, match="start at 0"):
        split_at_seams([1, 6], 1, 0, 7)
    with pytest.raises(ValueError, match="strictly increasing"):
        split_at_seams([0, 6, 6], 1, 0, 7)
    with pytest.raises(ValueError, match="last cell"):
        split_at_seams([0, 6], 1, 0, 9)


def test_plan_matches_upstream_121_frame_example_round_1():
    # canvas 121 @ 24 fps, slots [24..120]; round 1 doubles: seams x2, N = 241, 2 tiles
    plan = TemporalTilePlan([48, 96, 144, 192, 240], 241, 2)
    assert len(plan) == 2
    t0, t1 = plan
    assert t0.interval == TemporalInterval(0, 19, 0)
    assert (t0.pixel_start, t0.pixel_end) == (0, 144)
    assert t0.anchors == (48, 96, 144) and t0.slots == (24, 72, 120)
    assert t1.interval == TemporalInterval(12, 31, 7)
    assert (t1.pixel_start, t1.pixel_end) == (96, 240)
    assert t1.anchors == (96, 144, 192, 240) and t1.slots == (120, 168, 216)


def test_plan_round_2_has_4_tiles_covering_the_canvas():
    carry = sorted([48, 96, 144, 192, 240, 24, 72, 120, 168, 216])  # 10 positions after round 1
    seams = [2 * p for p in carry]
    plan = TemporalTilePlan(seams, 2 * (241 - 1) + 1, 4)
    assert len(plan) == 4
    assert plan[0].interval.start == 0 and plan[-1].interval.end == (481 - 1) // 8 + 1
    kept = sum(t.interval.end - t.interval.start - t.interval.left_ramp for t in plan)
    assert kept == (481 - 1) // 8 + 1  # stitched length


def test_plan_clamps_tiles_to_segments():
    plan = TemporalTilePlan([48], 49, 4)  # one segment only
    assert len(plan) == 1
    assert plan[0].interval == TemporalInterval(0, 7, 0)
    assert plan[0].anchors == (48,) and plan[0].slots == (24,)
