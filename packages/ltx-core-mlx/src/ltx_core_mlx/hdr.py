"""HDR colour utilities: the ACEScct working space and the sRGB EOTF (upstream v1.4 ``ltx_core/hdr.py``).

The SDR-to-HDR IC-LoRA works in ACEScct: SDR plates are linearised (sRGB EOTF), moved to AP1 and
log-encoded on load; the decoded ACEScct output is expanded back to scene-linear for HLG / EXR.
"""

from __future__ import annotations

import numpy as np

from ltx_core_mlx.color.primaries import Primaries

# ---- ACEScct working space (upstream v1.4 ``ltx_core/hdr.py``) -------------------------------
_ACESCCT_A_LIN = 10.5402377416545
_ACESCCT_B_LIN = 0.0729055341958355
_ACESCCT_X_BRK = 0.0078125
_ACESCCT_Y_BRK = 0.155251141552511
_ACESCCT_LOG_M = 17.52
_ACESCCT_LOG_B = 9.72
_SRGB_A = 0.055
_SRGB_LINEAR_THRESHOLD = 0.04045
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_GAMMA = 2.4


def _acescct_compress(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x.astype(np.float32), 0.0)
    log_part = (np.log2(np.maximum(x, 1e-12)) + _ACESCCT_LOG_B) / _ACESCCT_LOG_M
    lin_part = _ACESCCT_A_LIN * x + _ACESCCT_B_LIN
    return np.clip(np.where(x > _ACESCCT_X_BRK, log_part, lin_part), 0.0, 1.0).astype(np.float32)


def _acescct_decompress(ct: np.ndarray) -> np.ndarray:
    ct = np.clip(ct.astype(np.float32), 0.0, 1.0)
    from_log = np.power(2.0, ct * _ACESCCT_LOG_M - _ACESCCT_LOG_B)
    from_lin = (ct - _ACESCCT_B_LIN) / _ACESCCT_A_LIN
    return np.where(ct > _ACESCCT_Y_BRK, from_log, from_lin).astype(np.float32)


def to_acescct_working_space(rgb: np.ndarray, source_primaries: Primaries = Primaries.REC709) -> np.ndarray:
    """Scene-linear channel-last RGB in ``source_primaries`` -> ACEScct ``[0, 1]``."""
    return _acescct_compress(np.maximum(source_primaries.to(Primaries.AP1, rgb), 0.0))


def to_hdr_linear(working: np.ndarray, out_primaries: Primaries = Primaries.REC709) -> np.ndarray:
    """ACEScct ``[0, 1]`` -> scene-linear channel-last RGB in ``out_primaries`` (clamped at 0)."""
    return np.maximum(Primaries.AP1.to(out_primaries, _acescct_decompress(working)), 0.0)


def srgb_eotf_to_linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB-encoded ``[0, 1]`` -> display-linear Rec.709 (IEC 61966-2-1 EOTF)."""
    x = np.clip(srgb.astype(np.float32), 0.0, 1.0)
    return np.where(
        x <= _SRGB_LINEAR_THRESHOLD, x / _SRGB_LINEAR_SLOPE, np.power((x + _SRGB_A) / (1.0 + _SRGB_A), _SRGB_GAMMA)
    ).astype(np.float32)
