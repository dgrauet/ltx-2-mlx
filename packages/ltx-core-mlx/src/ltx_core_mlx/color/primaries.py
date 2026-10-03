"""Linear primaries: gamut conversion and EXR chromaticity tags (upstream ``ltx_core/color/primaries.py``).

Upstream builds the matrices with colour-science (Bradford CAT); they are hard-coded here from
colour-science 0.4.7 so the runtime does not depend on it (``tests/test_color_acescct.py`` pins them).
"""

from __future__ import annotations

import enum

import numpy as np

_ACESCG_TO_REC709 = np.array(
    [
        [1.705050992657984, -0.6217921206570046, -0.0832588720009786],
        [-0.1302564175070437, 1.1408047365754013, -0.010548319068358037],
        [-0.02400335680461801, -0.12896897606497054, 1.1529723328695887],
    ]
)
_REC709_TO_ACESCG = np.array(
    [
        [0.6130974024011875, 0.33952314618410556, 0.04737945141470665],
        [0.07019372246958185, 0.9163538790573443, 0.01345239847307413],
        [0.02061559288222692, 0.10956977293813541, 0.8698146341796376],
    ]
)
_REC709_TO_REC2020 = np.array(
    [
        [0.6274038959346991, 0.329283038377884, 0.0433130656874172],
        [0.06909728935823205, 0.9195403950754583, 0.011362315566309178],
        [0.01639143887515023, 0.08801330787722578, 0.8955952532476243],
    ]
)
_ACESCG_TO_REC2020 = np.array(
    [
        [1.025824747666011, -0.02005319083821476, -0.00577155682779561],
        [-0.00223436951997613, 1.004586501888479, -0.0023521323685036307],
        [-0.005013351468089274, -0.025290071810785176, 1.0303034232788748],
    ]
)


def apply_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 linear map to channel-last ``(..., 3)`` RGB, in float32."""
    return (rgb.astype(np.float32) @ matrix.T.astype(np.float32)).astype(np.float32)


class Primaries(enum.Enum):
    """Linear-light primaries the HDR path authors in."""

    REC709 = "rec709"
    AP1 = "ap1"

    @property
    def exr_chromaticities(self) -> tuple[float, ...]:
        """EXR ``chromaticities``: R xy, G xy, B xy, white xy."""
        if self is Primaries.AP1:
            return (0.713, 0.293, 0.165, 0.83, 0.128, 0.044, 0.32168, 0.33767)
        return (0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329)

    @property
    def matrix_to_rec2020(self) -> np.ndarray:
        """3x3 linear map from this basis to Rec.2020 (HLG master)."""
        return _ACESCG_TO_REC2020 if self is Primaries.AP1 else _REC709_TO_REC2020

    def to(self, target: Primaries, rgb: np.ndarray) -> np.ndarray:
        """Convert channel-last linear RGB from this basis to ``target`` (identity when equal)."""
        if self is target:
            return rgb.astype(np.float32)
        matrix = _REC709_TO_ACESCG if self is Primaries.REC709 else _ACESCG_TO_REC709
        return apply_matrix(rgb, matrix)
