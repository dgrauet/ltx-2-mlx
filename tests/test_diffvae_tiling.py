"""Tile schedule math of the diffusion video decoder, checked against a plain re-transcription of upstream."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder.config import LTX_2_5_DIFFUSION_DECODER
from ltx_core_mlx.model.video_vae.diffusion_decoder.tiling import (
    MIN_MODEL_BYTES,
    RESERVE_BYTES,
    DiffusionTileConfig,
    DiffusionTileGeometry,
    Interval,
    auto_tile_config,
    build_tile_schedule,
    describe_tiling,
    estimate_untiled_bytes,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
    output_fhw,
    padded_latent_fhw,
    propagate_spatial,
    propagate_temporal,
    round_up,
    split_by_size,
)
from tests.diffvae_tiny import TINY


def _upstream_split(length: int, size: int, overlap: int, min_tile: int | None) -> list[tuple[int, int, int, int]]:
    """Independent transcription of upstream tl:174-223 (split + grow-last-tile), tuples (start, end, l, r)."""
    if min_tile is not None and length < min_tile:
        return [(0, length, 0, 0)]
    if length <= size:
        return [(0, length, 0, 0)]
    n = (length + size - 2 * overlap - 1) // (size - overlap)
    out = []
    for i in range(n):
        if i == 0:
            out.append((0, size, 0, overlap))
        elif i < n - 1:
            out.append((i * (size - overlap), i * (size - overlap) + size, overlap, overlap))
        else:
            out.append(((n - 1) * (size - overlap), length, overlap, 0))
    if min_tile is not None and len(out) >= 2 and out[-1][1] - out[-1][0] < min_tile:
        s, e, _l, _r = out[-1]
        ps, pe, pl, _pr = out[-2]
        new_start = e - min_tile
        new_overlap = pe - new_start
        out[-2] = (ps, pe, pl, new_overlap)
        out[-1] = (new_start, e, new_overlap, 0)
    return out


@pytest.mark.parametrize("size,overlap,min_tile", [(8, 4, 3), (8, 4, None), (40, 20, 6), (9, 3, 6), (5, 0, None)])
def test_split_by_size_matches_upstream_transcription(size, overlap, min_tile):
    for length in range(1, 200):
        ref = _upstream_split(length, size, overlap, min_tile)
        try:
            got = split_by_size(length, size, overlap, min_tile)
        except ValueError:
            # Only the grown-last-tile validation may reject; the reference must then be inconsistent.
            assert min_tile is not None and len(ref) >= 2 and ref[-1][2] >= ref[-2][1] - ref[-2][0]
            continue
        assert [(iv.start, iv.end, iv.left_ramp, iv.right_ramp) for iv in got] == ref, (length, size, overlap)


def test_split_by_size_examples():
    assert split_by_size(17, 8, 4, 3) == [
        Interval(0, 8, 0, 4),
        Interval(4, 12, 4, 4),
        Interval(8, 16, 4, 4),
        Interval(12, 17, 4, 0),
    ]
    assert split_by_size(49, 40, 20, 6) == [Interval(0, 40, 0, 20), Interval(20, 49, 20, 0)]
    assert split_by_size(136, 40, 20, 6)[-1] == Interval(100, 136, 20, 0)
    assert len(split_by_size(240, 40, 20, 6)) == 11
    assert split_by_size(5, 8, 4, 3) == [Interval(0, 5)]  # length <= size -> one interval
    assert split_by_size(2, 8, 4, 3) == [Interval(0, 2)]  # below min tile -> one interval


def test_split_by_size_grows_last_tile():
    # length 17, size 8, overlap 2 -> n = (17+8-4-1)//6 = 3 -> [0,8),[6,14),[12,17): last len 5 < min 6 -> grown
    got = split_by_size(17, 8, 2, 6)
    assert got[-1] == Interval(11, 17, 3, 0) and got[-2] == Interval(6, 14, 2, 3)


def test_split_by_size_rejects_bad_arguments():
    with pytest.raises(ValueError):
        split_by_size(10, 0, 0)
    with pytest.raises(ValueError):
        split_by_size(10, 4, 4)
    with pytest.raises(ValueError):
        split_by_size(10, 4, 1, 0)


def test_temporal_propagation_is_causal():
    assert propagate_temporal(Interval(0, 8, 0, 4), 2) == Interval(0, 15, 0, 8)
    assert propagate_temporal(Interval(4, 12, 4, 4), 2) == Interval(7, 23, 8, 8)
    assert propagate_temporal(Interval(12, 17, 4, 0), 2) == Interval(23, 33, 8, 0)
    assert propagate_temporal(Interval(3, 5, 1, 1), 1) == Interval(3, 5, 1, 1)


def test_spatial_propagation_scales_everything():
    assert propagate_spatial(Interval(4, 12, 4, 4), 2) == Interval(8, 24, 8, 8)
    assert propagate_spatial(propagate_spatial(Interval(4, 12, 4, 4), 2), 4) == Interval(32, 96, 32, 32)


def test_production_geometry_matches_upstream_numbers():
    g = DiffusionTileGeometry.from_config(LTX_2_5_DIFFUSION_DECODER)
    assert g.pixel_scale == (2, 8, 8) and g.latent_scale == (8, 32, 32)
    assert g.min_tile_s4 == (6, 6, 6) and g.halo4 == (2, 4, 4) and g.halo5 == (20, 20, 20)
    assert (g.overlap_frames, g.overlap_px) == (40, 160)
    assert (g.min_tile_frames, g.min_tile_px) == (80, 320)
    assert (g.step_frames, g.step_px) == (8, 32)
    assert g.ghost_frames_s4 == 8 and g.stage4_channels == 512 and g.stage5_channels == 256
    assert g.stage4_content_thw(13, 34, 60) == (49, 136, 240)  # 1088x1920x97


def test_tiny_geometry():
    g = DiffusionTileGeometry.from_config(TINY)
    assert g.min_tile_s4 == (3, 3, 3) and g.halo4 == (1, 1, 1) and g.halo5 == (1, 1, 1)
    assert (g.overlap_frames, g.overlap_px) == (8, 32)
    assert (g.min_tile_frames, g.min_tile_px) == (16, 64)
    assert g.ghost_frames_s4 == 8
    assert g.stage4_content_thw(5, 3, 3) == (17, 12, 12)


def test_round_up_and_padded_fhw():
    assert round_up(41, 8) == 48 and round_up(40, 8) == 40 and round_up(0, 8) == 0
    assert padded_latent_fhw(LTX_2_5_DIFFUSION_DECODER, (2, 2, 3)) == (3, 7, 7)
    assert padded_latent_fhw(LTX_2_5_DIFFUSION_DECODER, (13, 34, 60)) == (13, 34, 60)


def test_tile_config_validation():
    g = DiffusionTileGeometry.from_config(LTX_2_5_DIFFUSION_DECODER)
    ok = DiffusionTileConfig(80, 40, 320, 160, 320, 160)
    assert ok.validate(g) is ok
    with pytest.raises(ValueError, match="overlap"):
        DiffusionTileConfig(80, 32, 320, 160, 320, 160).validate(g)  # below recommended 40
    DiffusionTileConfig(80, 32, 320, 160, 320, 160).validate(g, allow_small_overlap=True)
    with pytest.raises(ValueError, match="multiple"):
        DiffusionTileConfig(81, 40, 320, 160, 320, 160).validate(g)
    with pytest.raises(ValueError, match="multiple"):
        DiffusionTileConfig(80, 40, 324, 160, 320, 160).validate(g)
    with pytest.raises(ValueError, match="at least"):
        DiffusionTileConfig(80, 40, 8, 0, 320, 160).validate(g, allow_small_overlap=True)
    with pytest.raises(ValueError, match="smaller"):
        DiffusionTileConfig(40, 40, 320, 160, 320, 160).validate(g)
    # A zero size disables that axis; its overlap is then ignored.
    assert DiffusionTileConfig(0, 0, 320, 160, 0, 0).validate(g) is not None


def test_from_pixels_uses_recommended_overlaps():
    g = DiffusionTileGeometry.from_config(LTX_2_5_DIFFUSION_DECODER)
    assert DiffusionTileConfig.from_pixels(g, 80, 320, 640) == DiffusionTileConfig(80, 40, 320, 160, 640, 160)
    assert DiffusionTileConfig.from_pixels(g, 0, 320, 0) == DiffusionTileConfig(0, 0, 320, 160, 0, 0)
    with pytest.raises(ValueError):
        DiffusionTileConfig.from_pixels(g, 80, 300, 320)


def test_intervals_for_axis_converts_pixels_to_cells():
    g = DiffusionTileGeometry.from_config(TINY)
    assert g.intervals_for_axis(0, 17, 16, 8) == split_by_size(17, 8, 4, 3)
    assert g.intervals_for_axis(1, 12, 64, 32) == split_by_size(12, 8, 4, 3)
    assert g.intervals_for_axis(1, 12, 0, 0) == [Interval(0, 12)]


TINY_G = DiffusionTileGeometry.from_config(TINY)
TINY_CFG = DiffusionTileConfig(16, 8, 64, 32, 64, 32)


def test_untiled_schedule_is_one_tile_covering_everything():
    tiles = build_tile_schedule(TINY_G, (5, 3, 3), None)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.in_t, t.in_h, t.in_w) == (slice(0, 17), slice(0, 12), slice(0, 12))
    assert (t.out_t, t.out_h, t.out_w) == (slice(0, 33), slice(0, 96), slice(0, 96))
    assert t.is_origin and t.pad_trailing and t.index == 0
    assert mx.array_equal(t.mask_t, mx.ones(33)) and t.mask_t.dtype == mx.float32
    assert output_fhw(TINY_G, (5, 3, 3)) == (33, 96, 96)


def test_tiny_schedule_geometry():
    tiles = build_tile_schedule(TINY_G, (5, 3, 3), TINY_CFG)
    assert len(tiles) == 16 and [t.index for t in tiles] == list(range(16))
    # temporal slowest: tiles 0-3 share in_t [0,8)
    assert all(t.in_t == slice(0, 8) for t in tiles[:4]) and tiles[4].in_t == slice(4, 12)
    assert [t.out_t for t in tiles[::4]] == [slice(0, 15), slice(7, 23), slice(15, 31), slice(23, 33)]
    assert [t.is_origin for t in tiles[::4]] == [True, False, False, False]
    assert [t.pad_trailing for t in tiles[::4]] == [False, False, False, True]
    assert tiles[0].out_h == slice(0, 64) and tiles[1].out_w == slice(32, 96) and tiles[1].in_w == slice(4, 12)
    assert tiles[0].mask_t.shape == (15,) and tiles[4].mask_t.shape == (16,) and tiles[12].mask_t.shape == (10,)
    # ramps: first tile fades out over 8 frames with k/(r+1); middle tiles fade in with the complement
    assert mx.allclose(tiles[0].mask_t[7:], mx.array([(9 - k) / 9 for k in range(1, 9)])).item()
    assert mx.allclose(tiles[4].mask_t[:8], mx.array([k / 9 for k in range(1, 9)])).item()
    assert masks_are_complementary(tiles, output_fhw(TINY_G, (5, 3, 3)))


def test_production_schedule_1088x1920x97():
    g = DiffusionTileGeometry.from_config(LTX_2_5_DIFFUSION_DECODER)
    tiles = build_tile_schedule(g, (13, 34, 60), DiffusionTileConfig(80, 40, 320, 160, 320, 160))
    assert len(tiles) == 2 * 6 * 11
    assert tiles[0].in_t == slice(0, 40) and tiles[-1].in_t == slice(20, 49)
    assert tiles[0].out_t == slice(0, 79) and tiles[-1].out_t == slice(39, 97)
    assert tiles[-1].in_h == slice(100, 136) and tiles[-1].out_h == slice(800, 1088)
    assert tiles[-1].in_w == slice(200, 240) and tiles[-1].out_w == slice(1600, 1920)
    assert masks_are_complementary(tiles, output_fhw(g, (13, 34, 60)))
    assert output_fhw(g, (13, 34, 60)) == (97, 1088, 1920)


def test_schedule_rejects_small_overlap_unless_allowed():
    with pytest.raises(ValueError):
        build_tile_schedule(TINY_G, (5, 3, 3), DiffusionTileConfig(16, 0, 0, 0, 0, 0))
    tiles = build_tile_schedule(TINY_G, (5, 3, 3), DiffusionTileConfig(16, 0, 0, 0, 0, 0), allow_small_overlap=True)
    assert len(tiles) == 3 and tiles[1].out_t == slice(15, 31)


def test_masks_not_complementary_for_asymmetric_ramps():
    tiles = build_tile_schedule(TINY_G, (5, 3, 3), TINY_CFG)
    bad = [dataclasses.replace(tiles[0], mask_t=mx.ones(15)), *tiles[1:]]
    assert not masks_are_complementary(bad, output_fhw(TINY_G, (5, 3, 3)))


def test_group_by_temporal_slice():
    tiles = build_tile_schedule(TINY_G, (5, 3, 3), TINY_CFG)
    groups = group_tiles_by_temporal_slice(tiles)
    assert [len(g) for g in groups] == [4, 4, 4, 4]
    assert [g[0].out_t.start for g in groups] == [0, 7, 15, 23]


def test_describe_tiling():
    s = describe_tiling(TINY_G, (5, 3, 3), TINY_CFG)
    assert "tiles=4x2x2" in s and "frames=16/8" in s and "px=64x64/32" in s and "redundancy=x" in s


# TINY on latent (5, 3, 3): F_px=33, H_px=W_px=96 -> stage-5 tokens 33*24*24 = 19008, c5 = 4.
_S4_BYTES = (17 + 8) * 12 * 12 * 8 * 2
_UNTILED = 19008 * 4 * 2 * 5 + 33 * 96 * 96 * 6


def _budget(usable: int) -> int:
    return usable + MIN_MODEL_BYTES + RESERVE_BYTES + _S4_BYTES


def test_estimate_untiled_bytes():
    assert estimate_untiled_bytes(TINY_G, (5, 3, 3)) == _UNTILED


def test_auto_returns_none_when_the_whole_decode_fits():
    assert auto_tile_config(TINY_G, (5, 3, 3), budget_bytes=_budget(_UNTILED), weight_bytes=0) is None
    assert auto_tile_config(TINY_G, (5, 3, 3), budget_bytes=_budget(_UNTILED - 1), weight_bytes=0) is not None


def test_auto_picks_the_least_redundant_feasible_tile():
    # usable 2,000,000: tile_t=16 -> acc 2*16*96*96*6 = 1,769,472; max tokens (2e6-acc)//40 = 5763
    # 16x64x64 -> 16*16*16 = 4096 ok; 16x64x96 -> 6144 too big; tile_t=24 -> acc 2,654,208 > usable.
    cfg = auto_tile_config(TINY_G, (5, 3, 3), budget_bytes=_budget(2_000_000), weight_bytes=0)
    assert cfg == DiffusionTileConfig(16, 8, 64, 32, 64, 32)


def test_auto_prefers_whole_axes_when_memory_allows():
    # usable = _UNTILED - 1 = 2,585,087. tile_t=16: acc 1,769,472, max tokens (usable-acc)//40 = 20,390;
    # 16x96x96 -> 16*24*24 = 9216 tokens fits, and 96 px = 12 cells covers the whole axis (n_h = n_w = 1):
    # redundancy 4*1*1*16*96*96/(33*96*96) = 1.94, versus 16x64x64 at 4*2*2*16*64*64/(33*96*96) = 3.45.
    cfg = auto_tile_config(TINY_G, (5, 3, 3), budget_bytes=_budget(_UNTILED - 1), weight_bytes=0)
    assert cfg == DiffusionTileConfig(16, 8, 96, 32, 96, 32)


def test_auto_weights_floor_and_failure():
    # weights below 1 GiB are charged as 1 GiB (so the same budget gives the same answer)
    a = auto_tile_config(TINY_G, (5, 3, 3), budget_bytes=_budget(2_000_000), weight_bytes=10)
    b = auto_tile_config(TINY_G, (5, 3, 3), budget_bytes=_budget(2_000_000), weight_bytes=0)
    assert a == b
    with pytest.raises(ValueError, match="LTX2_VAE_DECODE_BUDGET_GB"):
        auto_tile_config(TINY_G, (5, 3, 3), budget_bytes=_budget(100), weight_bytes=0)
