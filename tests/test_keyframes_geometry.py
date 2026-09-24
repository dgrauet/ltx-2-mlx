"""Keyframe (dual-stream) geometry of the diffusion decoder: times, slot tables, tile selection, plane upsample."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder.config import LTX_2_5_DIFFUSION_DECODER
from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import (
    KEYFRAME_CONTEXT_SLOTS,
    DecodeKeyframes,
    KeyframeStream,
    keyframe_clip_times,
    keyframe_stage_times,
    keyframe_video_slots,
    planes_for_tile,
    remaining_time_strides,
    upsample_keyframe_planes,
    video_keyframe_slots,
)
from ltx_core_mlx.model.video_vae.diffusion_decoder.layers import LinearPixelShuffleUpsample
from tests.diffvae_tiny import TINY


def test_constants_and_production_strides():
    assert KEYFRAME_CONTEXT_SLOTS == 2
    assert remaining_time_strides(LTX_2_5_DIFFUSION_DECODER) == (8, 8, 4, 2, 1)
    assert remaining_time_strides(TINY) == (8, 8, 4, 2, 1)


def test_stage_times_center_the_chunk_and_pin_frame_zero():
    t = np.array(keyframe_stage_times((0, 8, 13), 8))
    assert t.dtype == np.float32
    assert t.tolist() == [0.0, (8 + 3.5) / 8, (13 + 3.5) / 8]
    assert np.array(keyframe_stage_times((0, 5, 9), 1)).tolist() == [0.0, 5.0, 9.0]
    with pytest.raises(ValueError, match="remaining_time_stride"):
        keyframe_stage_times((1,), 0)


def test_clip_times_subtract_the_origin_instead_of_rebasing():
    # t_s is not linear through a fake clip start: t_s(48) - t_s(56) != t_s(48 - 56 rebased)
    got = np.array(keyframe_clip_times((48, 64), 8, 56, extra_origin=0.0)).tolist()
    ts = lambda f, r: 0.0 if f == 0 else (f + (r - 1) / 2) / r  # noqa: E731
    assert got == pytest.approx([ts(48, 8) - ts(56, 8), ts(64, 8) - ts(56, 8)])
    assert np.array(keyframe_clip_times((24,), 2, 0, extra_origin=3.0)).tolist() == pytest.approx([ts(24, 2) - 3.0])


def test_planes_for_tile_keeps_inside_planes_plus_one_neighbour_each_side():
    idx = (96, 24, 72, 48, 120)  # unsorted on purpose
    assert planes_for_tile(idx, 40, 80) == [
        True,
        True,
        True,
        True,
        False,
    ]  # 48, 72 inside; 24 latest before; 96 earliest after
    assert planes_for_tile(idx, 50, 70) == [False, False, True, True, False]  # nothing inside -> 48 before, 72 after
    assert planes_for_tile(idx, 0, 10) == [False, True, False, False, False]  # only the earliest plane after
    assert planes_for_tile(idx, 130, 140) == [False, False, False, False, True]  # only the latest plane before
    # global indices with a Dist-style clip start: local [0, 20) selects global [56, 76)
    assert planes_for_tile((48, 60, 90), 0, 19, clip_start_frame=56) == [True, True, True]


def test_slot_tables_rank_by_distance_then_index_with_minus_one_padding():
    times = np.array([2.0, 6.0, 6.0], dtype=np.float32)
    valid = np.array([True, True, True])
    v = video_keyframe_slots(times, valid, video_length=5)
    assert v.dtype == np.int32 and v.shape == (5, 2)
    # frame 4: |4-2|=2, |4-6|=2 twice -> tie broken by plane index: 0 then 1
    assert v.tolist() == [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    k = keyframe_video_slots(times, valid, video_length=5)
    assert k.tolist() == [[2, 1], [4, 3], [4, 3]]  # |t'-6| minimal at 4 then 3
    # invalid planes never appear and their rows are -1; fewer candidates than slots pad with -1
    v2 = video_keyframe_slots(times, np.array([True, False, False]), 3)
    assert v2.tolist() == [[0, -1], [0, -1], [0, -1]]
    assert keyframe_video_slots(times, np.array([True, False, False]), 1).tolist() == [[0, -1], [-1, -1], [-1, -1]]


def test_decode_keyframes_validation():
    kf = DecodeKeyframes(mx.zeros((1, 8, 2, 3, 3)), (5, 12))
    kf.validate(num_frames=17)
    assert kf.num_planes == 2
    with pytest.raises(ValueError, match="at least one plane"):
        DecodeKeyframes(mx.zeros((1, 8, 0, 3, 3)), ()).validate()
    with pytest.raises(ValueError, match="plane count"):
        DecodeKeyframes(mx.zeros((1, 8, 2, 3, 3)), (5,)).validate()
    with pytest.raises(ValueError, match="non-negative"):
        DecodeKeyframes(mx.zeros((1, 8, 1, 3, 3)), (-1,)).validate()
    with pytest.raises(ValueError, match="B, C, P, H, W"):
        DecodeKeyframes(mx.zeros((8, 2, 3, 3)), (5, 12)).validate()
    with pytest.raises(ValueError, match="clip_start_frame"):
        DecodeKeyframes(mx.zeros((1, 8, 1, 3, 3)), (5,), clip_start_frame=-1).validate()
    span = kf.for_frame_span(0, 8)  # plane 5 inside, 12 = nearest after
    assert span.pixel_frame_indices == (5, 12) and span.clip_start_frame == 0
    only_first = DecodeKeyframes(mx.zeros((1, 8, 3, 3, 3)), (5, 12, 20)).for_frame_span(0, 8)
    assert only_first.pixel_frame_indices == (5, 12) and only_first.latents.shape == (1, 8, 2, 3, 3)
    assert kf.crop_spatial(slice(0, 2), slice(1, 3)).latents.shape == (1, 8, 2, 2, 2)


def test_keyframe_stream_helpers():
    s = KeyframeStream(mx.ones((1, 3, 2, 2, 4)), mx.array([1.0, 2.0, 3.0]), mx.array([True, False, True]))
    assert mx.array_equal(s.masked().x[0, 1], mx.zeros((2, 2, 4)))
    assert mx.array_equal(s.masked().x[0, 0], mx.ones((2, 2, 4)))
    sub = s.select_planes([True, False, True])
    assert sub.num_planes == 2 and np.array(sub.times).tolist() == [1.0, 3.0]
    with pytest.raises(ValueError, match="keep"):
        s.select_planes([True])
    assert s.crop_spatial(slice(0, 1), slice(1, 2)).x.shape == (1, 3, 1, 1, 4)


def test_upsample_keyframe_planes_keeps_the_plane_count_and_matches_per_plane_clips():
    up = LinearPixelShuffleUpsample(8, (2, 2, 2), 2)
    x = mx.random.normal((1, 3, 4, 5, 8))
    y = upsample_keyframe_planes(up, x)
    assert y.shape == (1, 3, 8, 10, 4)
    for p in range(3):
        single = up(x[:, p : p + 1], drop_leading_frame=True)  # T=1 -> 2 -> drop -> 1
        assert single.shape == (1, 1, 8, 10, 4)
        assert mx.allclose(y[:, p], single[:, 0], atol=1e-6, rtol=1e-6)
