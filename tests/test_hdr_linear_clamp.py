"""``to_hdr_linear`` clamps sub-black values in AP1 before the primaries matrix (upstream LTX-2 v1.6.1).

ACEScct codes under the toe intercept (~0.0729) decompress to negative AP1 values. Clamped only after
the matrix, one channel's undershoot leaked into the others on the way to Rec.709 (HLG master,
``srgb_linear`` EXRs).
"""

from __future__ import annotations

import numpy as np

from ltx_core_mlx.color.primaries import Primaries
from ltx_core_mlx.hdr import _acescct_decompress, to_hdr_linear

#: Two pixels with a channel under the toe intercept, one fully in range.
WORKING = np.array([[0.0, 0.5, 0.5], [0.02, 0.0, 0.6], [0.5, 0.5, 0.5]], dtype=np.float32)


def test_sub_black_codes_are_clamped_before_the_matrix() -> None:
    ap1 = _acescct_decompress(WORKING)
    assert (ap1[:2] < 0).any()  # the fixture does undershoot

    expected = np.maximum(Primaries.AP1.to(Primaries.REC709, np.maximum(ap1, 0.0)), 0.0)
    leaked = np.maximum(Primaries.AP1.to(Primaries.REC709, ap1), 0.0)
    assert np.abs(expected[:2] - leaked[:2]).max() > 1e-4  # the two orders give different pixels here

    out = to_hdr_linear(WORKING, out_primaries=Primaries.REC709)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)


def test_in_range_pixels_do_not_move() -> None:
    in_range = WORKING[2:]
    unclamped = np.maximum(Primaries.AP1.to(Primaries.REC709, _acescct_decompress(in_range)), 0.0)
    np.testing.assert_array_equal(to_hdr_linear(in_range, out_primaries=Primaries.REC709), unclamped)


def test_ap1_output_is_the_clamped_decompression() -> None:
    np.testing.assert_array_equal(
        to_hdr_linear(WORKING, out_primaries=Primaries.AP1), np.maximum(_acescct_decompress(WORKING), 0.0)
    )
