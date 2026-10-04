"""HLG OETF / diffuse-white mapping and BT.2020 limited 10-bit packing (upstream ``color/hlg.py``, ``color/yuv.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from ltx_core_mlx.color.hlg import hlg_oetf, linear_to_hlg_signal
from ltx_core_mlx.color.primaries import Primaries
from ltx_core_mlx.color.yuv import rgb_to_yuv420p10_bt2020_limited

colour = pytest.importorskip("colour")


def test_hlg_oetf_matches_colour_science():
    x = np.array([0.0, 1 / 12, 0.1, 0.26496256, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(hlg_oetf(x), colour.models.oetf_BT2100_HLG(x), atol=1e-6)


def test_diffuse_white_maps_to_075():
    white = np.ones((1, 1, 3), dtype=np.float32)
    assert abs(float(linear_to_hlg_signal(white, Primaries.REC709)[0, 0, 0]) - 0.75) < 1e-5


def test_highlights_roll_off_below_one():
    out = linear_to_hlg_signal(np.full((1, 1, 3), 50.0, dtype=np.float32))
    assert 0.99 < float(out.max()) <= 1.0


def test_yuv_codes_stay_in_limited_range():
    rng = np.random.default_rng(0)
    rgb = np.concatenate([np.zeros((2, 4, 3)), np.ones((2, 4, 3)), rng.random((4, 4, 3))]).astype(np.float32)
    y, u, v = rgb_to_yuv420p10_bt2020_limited(rgb)
    assert y.dtype == np.uint16 and y.shape == (8, 4) and u.shape == (4, 2) and v.shape == (4, 2)
    assert y.min() >= 64 and y.max() <= 940 and u.min() >= 64 and u.max() <= 960
    assert int(y[0, 0]) == 64 and int(y[2, 0]) == 940  # black and white rows
    assert int(u[0, 0]) == 512 and int(v[0, 0]) == 512  # neutral chroma


def test_yuv_matches_colour_science_matrix():
    rgb = np.random.default_rng(1).random((2, 2, 3)).astype(np.float32)
    y, _, _ = rgb_to_yuv420p10_bt2020_limited(rgb)
    ycbcr = colour.RGB_to_YCbCr(
        rgb, K=colour.WEIGHTS_YCBCR["ITU-R BT.2020"], in_bits=10, out_bits=10, out_legal=True, out_int=True
    )
    np.testing.assert_allclose(y, ycbcr[..., 0], atol=1)
