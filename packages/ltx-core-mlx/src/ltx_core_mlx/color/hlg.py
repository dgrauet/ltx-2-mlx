"""HLG (BT.2100 / ARIB STD-B67) signal for the HDR master (upstream ``ltx_core/color/hlg.py``).

Pipeline: scene-linear RGB -> Rec.2020 -> diffuse white mapped to ``white_signal`` with an exponential
highlight roll-off -> HLG OETF. Encoding is done by ``ltx_pipelines_mlx.utils.hdr_media.HlgFfmpegWriter``.
"""

from __future__ import annotations

import numpy as np

from ltx_core_mlx.color.primaries import Primaries, apply_matrix

HLG_A = 0.17883277
HLG_B = 0.28466892
HLG_C = 0.55991073
#: ``oetf_inverse_BT2100_HLG(0.75)`` (colour-science): the scene-linear level that diffuse white maps to.
_WHITE_X = {0.75: 0.26496256042100724}


def hlg_oetf(x: np.ndarray) -> np.ndarray:
    """HLG OETF on scene-linear ``x`` (colour ``oetf_BT2100_HLG``), clamped to ``[0, 1]``."""
    x = x.astype(np.float32)
    low = np.sqrt(np.maximum(3.0 * x, 0.0))
    high = HLG_A * np.log(np.maximum(12.0 * x - HLG_B, 1e-12)) + HLG_C
    return np.clip(np.where(x <= 1.0 / 12.0, low, high), 0.0, 1.0).astype(np.float32)


def hlg_inverse_oetf(v: float) -> float:
    """Inverse HLG OETF for a scalar signal value (only used to resolve ``white_signal``)."""
    if v <= 0.5:
        return v * v / 3.0
    return (np.exp((v - HLG_C) / HLG_A) + HLG_B) / 12.0


def linear_to_hlg_signal(
    rgb: np.ndarray, primaries: Primaries = Primaries.REC709, white_signal: float = 0.75
) -> np.ndarray:
    """Channel-last scene-linear RGB in ``primaries`` -> Rec.2020 HLG signal ``[0, 1]`` (upstream ``to_hlg_signal_``)."""
    white_x = _WHITE_X.get(white_signal, hlg_inverse_oetf(white_signal))
    roll_k = white_x / (1.0 - white_x)
    lin = np.nan_to_num(np.maximum(apply_matrix(rgb, primaries.matrix_to_rec2020), 0.0), nan=0.0, neginf=0.0)
    x = np.where(lin <= 1.0, lin * white_x, 1.0 - (1.0 - white_x) * np.exp(-roll_k * (lin - 1.0)))
    return hlg_oetf(x)
