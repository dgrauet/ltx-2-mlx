"""BT.2020 non-constant-luminance RGB -> 10-bit limited-range YUV 4:2:0 (upstream ``color/yuv.py``)."""

from __future__ import annotations

import numpy as np

#: ``inv(colour.matrix_YCbCr(WEIGHTS_YCBCR["ITU-R BT.2020"], is_legal=False, is_int=False))``.
_RGB_TO_YCBCR_BT2020 = np.array(
    [
        [0.2627, 0.678, 0.0593],
        [-0.1396300627192516, -0.3603699372807484, 0.5],
        [0.5, -0.459785704597857, -0.04021429540214294],
    ],
    dtype=np.float32,
)


def rgb_to_yuv420p10_bt2020_limited(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(H, W, 3)`` signal in ``[0, 1]`` -> planar uint16 Y ``(H, W)``, U/V ``(H/2, W/2)``.

    Chroma is 2x2 box-averaged (upstream ``avg_pool2d``); limited range: Y = (219 E' + 16) * 4,
    Cb/Cr = (224 E' + 128) * 4.

    Raises:
        ValueError: odd H or W (4:2:0 needs even dimensions).
    """
    h, w, _ = rgb.shape
    if h % 2 or w % 2:
        raise ValueError(f"4:2:0 needs even dimensions, got {w}x{h}")
    yuv = rgb.astype(np.float32) @ _RGB_TO_YCBCR_BT2020.T
    y = yuv[..., 0] * 876.0 + 64.0
    uv = yuv[..., 1:].reshape(h // 2, 2, w // 2, 2, 2).mean(axis=(1, 3)) * 896.0 + 512.0
    to_u16 = lambda a: np.clip(np.rint(a), 0, 1023).astype(np.uint16)  # noqa: E731
    return to_u16(y), to_u16(uv[..., 0]), to_u16(uv[..., 1])
