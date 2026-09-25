"""DFR temporal-round helpers — pinned on upstream dfr_pipeline.py semantics."""

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.utils.positions import compute_audio_token_count
from ltx_pipelines_mlx.dfr import (
    ANCHOR_KEYFRAME_STRENGTH,
    TEMPORAL_ANCESTRAL_ETA,
    TEMPORAL_SIGMAS,
    audio_latent_for_tile,
    conditioning_fps,
    dedupe_slots,
    merge_carry_forward_keyframes,
    rebase_image_conditionings,
    resample_audio_time,
    slot_initials_from_video,
)
from ltx_pipelines_mlx.utils.args import ImageConditioningInput


def test_constants():
    assert ANCHOR_KEYFRAME_STRENGTH == 0.95 and TEMPORAL_ANCESTRAL_ETA == 0.5
    assert TEMPORAL_SIGMAS == [0.975, 0.909375, 0.725, 0.421875, 0.0]


@pytest.mark.parametrize(("fps", "cond"), [(24.0, 24.0), (30.0, 30.0), (48.0, 60.0), (50.0, 60.0), (96.0, 60.0)])
def test_conditioning_fps(fps, cond):
    assert conditioning_fps(fps) == cond


def test_resample_audio_time_is_linear_and_clamped():
    a = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)  # values 0..3 along T
    out = resample_audio_time(a, 0.0, 4.0, 8)  # step 0.5
    assert np.allclose(np.array(out).reshape(-1), [0, 0.5, 1, 1.5, 2, 2.5, 3, 3])
    with pytest.raises(ValueError, match="empty"):
        resample_audio_time(a, 2.0, 2.0, 4)
    with pytest.raises(ValueError, match="out_frames"):
        resample_audio_time(a, 0.0, 1.0, 0)


def test_audio_latent_for_tile_window_and_token_count():
    full = mx.arange(100, dtype=mx.float32).reshape(1, 1, 100, 1)  # stage-1 audio, 4 s canvas
    out = audio_latent_for_tile(
        full, pixel_start=96, local_frames=145, playback_fps=48.0, source_duration=121 / 24.0, cond_fps=60.0
    )
    assert out.shape[2] == compute_audio_token_count(145, frame_rate=60.0)
    start = 96 / 48.0 / (121 / 24.0) * 100
    assert abs(float(out[0, 0, 0, 0]) - start) < 1e-3


def test_slot_initials_pick_the_nearest_latent_frame():
    v = mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1, 1)
    out = slot_initials_from_video(v, [0, 12, 20, 100])  # round(p/8) -> 0, 2 (1.5 rounds to 2), 2 (2.5 -> 2), clamp 4
    assert np.array(out).reshape(-1).tolist() == [0.0, 2.0, 2.0, 4.0]


def test_dedupe_slots_keeps_the_earlier_tile():
    lat = mx.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1, 1)
    pos, out = dedupe_slots([24, 72, 72, 120], lat)
    assert pos == [24, 72, 120] and np.array(out).reshape(-1).tolist() == [1.0, 2.0, 4.0]


def test_merge_carry_forward_sorts_and_lets_slots_override():
    a = mx.array([10.0, 20.0]).reshape(1, 1, 2, 1, 1)
    s = mx.array([15.0, 25.0]).reshape(1, 1, 2, 1, 1)
    pos, lat = merge_carry_forward_keyframes([48, 96], a, [24, 72], s)
    assert pos == [24, 48, 72, 96] and np.array(lat).reshape(-1).tolist() == [15.0, 10.0, 25.0, 20.0]
    with pytest.raises(ValueError, match="positions"):
        merge_carry_forward_keyframes([48], a, [], None)
    with pytest.raises(RuntimeError, match="empty"):
        merge_carry_forward_keyframes([], None, [], None)


def test_rebase_image_conditionings_scales_filters_and_rebases():
    imgs = [ImageConditioningInput("a.png", 0, 1.0), ImageConditioningInput("b.png", 60, 0.8)]
    assert rebase_image_conditionings(imgs, pixel_scale=2, pixel_start=0, pixel_end=144) == [
        ImageConditioningInput("a.png", 0, 1.0),
        ImageConditioningInput("b.png", 120, 0.8),
    ]
    assert rebase_image_conditionings(imgs, pixel_scale=2, pixel_start=96, pixel_end=240) == [
        ImageConditioningInput("b.png", 24, 0.8)
    ]
