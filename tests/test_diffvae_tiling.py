"""Tile schedule math of the diffusion video decoder, checked against a plain re-transcription of upstream."""

from __future__ import annotations

import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder.tiling import (
    Interval,
    propagate_spatial,
    propagate_temporal,
    split_by_size,
)


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
