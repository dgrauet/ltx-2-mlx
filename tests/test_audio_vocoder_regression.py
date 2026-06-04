"""Non-regression tests for the mlx 0.31.2 Metal strided-scatter bug.

mlx 0.31.2 shipped a Metal scatter kernel (ml-explore/mlx#3266) that
mis-indexes the source of strided ``.at[].add()`` updates
(``y[2k] <- x[2k]`` instead of ``y[2k] <- x[k]``; CPU is unaffected,
small shapes are unaffected). This silently collapsed the BigVGAN
vocoder output to ~-50 dB noise via the zero-insert in
``UpSample1d`` / ``HannSincResampler``. Fixed upstream by
ml-explore/mlx#3483 (unreleased as of 2026-06). See issue #34.

Run with: pytest tests/test_audio_vocoder_regression.py -v
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from tests.conftest import MODEL_DIR

skip_no_weights = pytest.mark.skipif(MODEL_DIR is None, reason="q8 weights not found")


def test_mlx_strided_scatter_add_canary():
    """Framework canary: strided ``.at[].add()`` must match NumPy semantics.

    Broken on the Metal (GPU) stream in mlx 0.31.2 — 872/2048 wrong
    elements for this shape. Passes on 0.31.1 and on mlx main. If this
    test fails, the installed mlx release mis-executes strided
    scatter-adds and audio output cannot be trusted.
    """
    B, T, C = 2, 64, 8
    x = (mx.arange(B * T * C, dtype=mx.float32).reshape(B, T, C) % 7 - 3) / 3.0
    y = mx.zeros((B, T * 2, C)).at[:, ::2, :].add(x)
    mx.eval(y)

    expected = np.zeros((B, T * 2, C), dtype=np.float32)
    expected[:, ::2, :] = np.array(x)
    wrong = int((np.array(y) != expected).sum())
    assert wrong == 0, (
        f"strided .at[].add() mis-indexed {wrong} elements — known mlx 0.31.2 "
        "Metal bug (ml-explore/mlx#3477); upgrade/downgrade mlx. The vocoder "
        "itself is guarded by slice assignment, see issue #34."
    )


def test_strided_zero_insert_assignment_matches_add_semantics():
    """The workaround (slice assignment into zeros) equals the intended add.

    Guards the patched zero-insert pattern in ``UpSample1d`` and
    ``HannSincResampler`` against semantic drift, independent of weights.
    """
    for B, T, C, stride in [(2, 64, 8, 2), (1, 269, 24, 2), (16, 887, 1, 3)]:
        x = mx.random.normal((B, T, C))
        out_len = T * stride if stride == 2 else (T - 1) * stride + 1
        y = mx.zeros((B, out_len, C))
        y[:, ::stride, :] = x[:, : (out_len + stride - 1) // stride, :]
        mx.eval(y)

        yn = np.array(y)
        expected = np.zeros((B, out_len, C), dtype=np.float32)
        expected[:, ::stride, :] = np.array(x)[:, : (out_len + stride - 1) // stride, :]
        np.testing.assert_array_equal(yn, expected)


@skip_no_weights
def test_vocoder_rms_floor():
    """BigVGAN vocoder output must not collapse to noise.

    On a fixed random mel, a healthy vocoder produces ≈ -14 dB RMS;
    the mlx 0.31.2 strided-scatter bug produced ≈ -51 dB. The -30 dB
    floor separates the two regimes with wide margin. This is the test
    that would have caught the silent-audio regression immediately.
    """
    from ltx_pipelines_mlx.utils.blocks import AudioDecoder

    _, vocoder = AudioDecoder(MODEL_DIR).load()

    mx.random.seed(456)
    mel = (mx.random.normal((1, 2, 269, 64)) * 0.5).astype(mx.bfloat16)
    wav = vocoder(mel)
    mx.eval(wav)

    w = wav.astype(mx.float32)
    rms_db = 20 * math.log10(max(float(mx.sqrt(mx.mean(w * w))), 1e-12))
    assert rms_db > -30.0, (
        f"vocoder output collapsed to noise: RMS = {rms_db:.1f} dB "
        "(healthy ≈ -14 dB; mlx 0.31.2 strided-scatter bug ≈ -51 dB — see issue #34)"
    )
