"""Tile schedule math of the diffusion video decoder, checked against a plain re-transcription of upstream."""

from __future__ import annotations

import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder.config import LTX_2_5_DIFFUSION_DECODER
from ltx_core_mlx.model.video_vae.diffusion_decoder.tiling import (
    DiffusionTileConfig,
    DiffusionTileGeometry,
    Interval,
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
