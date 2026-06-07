"""Regression tests for the mlx 0.31.2 strided-scatter audio bug (issue #34).

mlx 0.31.2 shipped a Metal scatter-kernel regression (ml-explore/mlx#3266,
fixed upstream by #3483 but unreleased as of 2026-06) where
``mx.array.at[<strided slice>].add(x)`` mis-indexes the *source*: element
``x[k]`` is read with the destination's linearized stride instead of the
source's, corrupting the zero-insert used by the BigVGAN vocoder
(``UpSample1d``) and the BWE resampler (``HannSincResampler``). The corrupted
zero-insert feeds every SnakeBeta activation, collapsing the waveform to
noise (~-50 dB instead of ~-13 dB).

These tests exercise the two real call sites against a NumPy reference and a
pure-MLX canary so the regression is caught at the framework level (Metal
only) and at the audio-path level.
"""

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.model.audio_vae.bwe import HannSincResampler
from ltx_core_mlx.model.audio_vae.vocoder import UpSample1d


def _strided_scatter_wrong_elements() -> int:
    """Probe the mlx ``at[strided].add()`` scatter on the active backend."""
    B, T, C = 2, 64, 8
    x = (mx.arange(B * T * C, dtype=mx.float32).reshape(B, T, C) % 7 - 3) / 3.0
    y = mx.zeros((B, T * 2, C)).at[:, ::2, :].add(x)
    mx.eval(y)
    expected = np.zeros((B, T * 2, C), dtype=np.float32)
    expected[:, ::2, :] = np.array(x)
    return int((np.array(y) != expected).sum())


def test_mlx_strided_scatter_add_canary():
    """Framework guard for the mlx 0.31.2 Metal scatter regression (issue #34).

    ``at[<strided slice>].add()`` mis-indexes the source on Metal in mlx
    0.31.2 (ml-explore/mlx#3477, fixed by #3483 but unreleased). ltx-2-mlx no
    longer relies on this op in the audio path (see the UpSample1d /
    HannSincResampler tests), so a broken backend does NOT break our output —
    this canary therefore SKIPS with a diagnostic rather than failing the suite
    on an affected MLX build. On a healthy backend it asserts correctness.
    """
    wrong = _strided_scatter_wrong_elements()
    if wrong:
        pytest.skip(
            f"active mlx backend has the known strided-scatter bug "
            f"({wrong} wrong elements; issue #34 / ml-explore/mlx#3477). "
            "ltx-2-mlx works around it; upgrade mlx when a fix ships."
        )
    assert wrong == 0


def _upsample1d_reference(x: np.ndarray, kernel_size: int = 12) -> np.ndarray:
    """NumPy mirror of ``UpSample1d.__call__`` (filter = ones, scale = 2.0)."""
    B, T, C = x.shape
    x_up = np.zeros((B, T * 2, C), dtype=np.float32)
    x_up[:, ::2, :] = x  # correct zero-insert

    xc = np.transpose(x_up, (0, 2, 1)).reshape(B * C, T * 2, 1)
    pad = kernel_size // 2
    left = np.repeat(xc[:, :1, :], pad, axis=1)
    right = np.repeat(xc[:, -1:, :], pad - 1, axis=1)
    xc = np.concatenate([left, xc, right], axis=1)

    # valid conv1d with an all-ones kernel == sliding window sum
    win = np.lib.stride_tricks.sliding_window_view(xc[:, :, 0], kernel_size, axis=1)
    out = win.sum(axis=-1)  # (B*C, T_out)
    T_out = out.shape[1]
    return np.transpose(out.reshape(B, C, T_out), (0, 2, 1)) * 2.0


def test_upsample1d_zero_insert_matches_reference():
    """``UpSample1d`` (vocoder.py call site) must match a NumPy reference."""
    B, T, C = 2, 37, 8
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((B, T, C)).astype(np.float32)

    up = UpSample1d()
    out = up(mx.array(x_np))
    mx.eval(out)

    ref = _upsample1d_reference(x_np)
    assert np.allclose(np.array(out), ref, atol=1e-4, rtol=1e-4)


def _hann_sinc_reference(x: np.ndarray, resampler: HannSincResampler) -> np.ndarray:
    """NumPy mirror of ``HannSincResampler.__call__`` (correct zero-insert)."""
    B, T = x.shape
    ratio = resampler.upsample_factor
    pad = resampler._pad

    first = np.repeat(x[:, :1], pad, axis=1)
    last = np.repeat(x[:, -1:], pad, axis=1)
    x_padded = np.concatenate([first, x, last], axis=1)
    T_padded = x_padded.shape[1]

    zi_len = (T_padded - 1) * ratio + 1
    upsampled = np.zeros((B, zi_len), dtype=np.float32)
    upsampled[:, ::ratio] = x_padded  # correct zero-insert

    kernel = np.array(resampler.kernel)[:, 0].astype(np.float32)  # (K,)
    K = kernel.shape[0]
    padded = np.pad(upsampled, [(0, 0), (K - 1, K - 1)])
    # forward conv1d (cross-correlation, matching mx.conv1d)
    win = np.lib.stride_tricks.sliding_window_view(padded, K, axis=1)
    result = (win * kernel[None, None, :]).sum(axis=-1)  # (B, zi_len + K - 1)
    result = result * ratio
    result = result[:, resampler._pad_left : -resampler._pad_right]
    return result[:, : T * ratio]


def test_hann_sinc_resampler_matches_reference():
    """``HannSincResampler`` (bwe.py call site) must match a NumPy reference.

    Uses a non-constant (sinusoidal) input so the scatter mis-indexing
    corrupts the result — a DC input is robust to the bug because every
    source element is identical.
    """
    resampler = HannSincResampler(upsample_factor=3)
    B, T = 2, 200
    t = np.arange(T, dtype=np.float32)
    x_np = np.stack([np.sin(2 * np.pi * 0.03 * t), np.cos(2 * np.pi * 0.05 * t)]).astype(np.float32)
    assert x_np.shape == (B, T)

    out = resampler(mx.array(x_np))
    mx.eval(out)

    ref = _hann_sinc_reference(x_np, resampler)
    assert np.allclose(np.array(out), ref, atol=1e-4, rtol=1e-4)
