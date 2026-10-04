"""ACEScct / sRGB / primaries, pinned against colour-science (upstream's source)."""

from __future__ import annotations

import numpy as np
import pytest

from ltx_core_mlx.color.primaries import Primaries
from ltx_core_mlx.hdr import srgb_eotf_to_linear, to_acescct_working_space, to_hdr_linear

colour = pytest.importorskip("colour")


def test_matrices_match_colour_science_bradford():
    cs = colour.RGB_COLOURSPACES
    ref = colour.matrix_RGB_to_RGB(cs["ACEScg"], cs["ITU-R BT.709"], chromatic_adaptation_transform="Bradford")
    rgb = np.random.default_rng(0).random((5, 3)).astype(np.float32)
    np.testing.assert_allclose(Primaries.AP1.to(Primaries.REC709, rgb), rgb @ ref.T, atol=1e-6)
    ref2020 = colour.matrix_RGB_to_RGB(
        cs["ITU-R BT.709"], cs["ITU-R BT.2020"], chromatic_adaptation_transform="Bradford"
    )
    np.testing.assert_allclose(Primaries.REC709.matrix_to_rec2020, ref2020, atol=1e-12)


def test_acescct_matches_colour_science():
    lin = np.array([0.0, 0.001, 0.0078125, 0.18, 1.0, 16.0], dtype=np.float32)
    ours = to_acescct_working_space(np.stack([lin] * 3, axis=-1), source_primaries=Primaries.AP1)[..., 0]
    ref = np.clip(colour.models.log_encoding_ACEScct(lin), 0.0, 1.0)
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_acescct_round_trip():
    rgb = np.random.default_rng(1).random((4, 4, 3)).astype(np.float32) * 4
    back = to_hdr_linear(to_acescct_working_space(rgb), out_primaries=Primaries.REC709)
    np.testing.assert_allclose(back, rgb, rtol=2e-3, atol=2e-4)


def test_acescct_handles_zero_and_large():
    rgb = np.array([[[0.0, 0.0, 0.0], [1e6, 1e6, 1e6], [-1.0, 0.5, 2.0]]], dtype=np.float32)
    out = to_acescct_working_space(rgb)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_srgb_eotf_matches_colour_science():
    x = np.linspace(0, 1, 11, dtype=np.float32)
    np.testing.assert_allclose(srgb_eotf_to_linear(x), colour.models.eotf_sRGB(x), atol=1e-6)


def test_chromaticities():
    assert Primaries.AP1.exr_chromaticities == (0.713, 0.293, 0.165, 0.83, 0.128, 0.044, 0.32168, 0.33767)
    assert Primaries.REC709.exr_chromaticities == (0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329)
