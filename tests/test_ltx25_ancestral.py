"""EulerAncestralDiffusionStep — asserté contre les formules upstream.

Les valeurs de référence sont recalculées dans le test avec l'arithmétique
Python float (indépendante de MLX), reproduisant upstream
diffusion_steps.py:43-107 ligne à ligne.
"""

import mlx.core as mx
import pytest

from ltx_core_mlx.components.diffusion_steps import (
    EulerAncestralDiffusionStep,
    EulerDiffusionStep,
)


def _reference_step(x, x0, sigma, sigma_next, eta, s_noise, noise):
    """Transcription float-Python des formules upstream (rectified flow)."""
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    r = sigma_down / sigma
    out = [r * xi + (1.0 - r) * x0i for xi, x0i in zip(x, x0)]
    if eta > 0:
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        coeff = max(sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2, 0.0) ** 0.5
        out = [(alpha_next / alpha_down) * o + n * s_noise * coeff for o, n in zip(out, noise)]
    return out


def test_matches_upstream_formulas_elementwise():
    x = [0.8, -1.2, 0.3, 2.0]
    x0 = [0.1, -0.4, 0.05, 1.1]
    noise = [0.5, -0.5, 1.0, -1.0]
    sigmas = mx.array([0.9, 0.6, 0.0])
    step = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)
    got = step.step(
        sample=mx.array(x, dtype=mx.float32),
        denoised_sample=mx.array(x0, dtype=mx.float32),
        sigmas=sigmas,
        step_index=0,
        noise=mx.array(noise, dtype=mx.float32),
    )
    want = _reference_step(x, x0, 0.9, 0.6, 1.0, 1.0, noise)
    assert mx.allclose(got, mx.array(want), atol=1e-6, rtol=1e-6).item()


def test_eta_zero_reduces_to_euler():
    # Algébriquement identique au pas d'Euler (formes interpolation vs
    # vélocité) — égal à tolérance FP, pas bit-exact.
    x = mx.random.normal((64,)).astype(mx.float32)
    x0 = mx.random.normal((64,)).astype(mx.float32)
    sigmas = mx.array([0.9, 0.6, 0.0])
    anc = EulerAncestralDiffusionStep(eta=0.0).step(sample=x, denoised_sample=x0, sigmas=sigmas, step_index=0)
    eul = EulerDiffusionStep().step(sample=x, denoised_sample=x0, sigmas=sigmas, step_index=0)
    assert mx.allclose(anc, eul, atol=1e-5, rtol=1e-5).item()


def test_terminal_sigma_returns_denoised():
    x = mx.random.normal((8,)).astype(mx.bfloat16)
    x0 = mx.random.normal((8,)).astype(mx.float32)
    sigmas = mx.array([0.42, 0.0])
    got = EulerAncestralDiffusionStep(eta=1.0).step(
        sample=x,
        denoised_sample=x0,
        sigmas=sigmas,
        step_index=0,
        noise=mx.zeros((8,), dtype=mx.float32),
    )
    assert got.dtype == x.dtype  # cast au dtype de sample
    assert mx.allclose(got, x0.astype(x.dtype)).item()


def test_eta_positive_requires_noise():
    with pytest.raises(ValueError, match="noise"):
        EulerAncestralDiffusionStep(eta=1.0).step(
            sample=mx.zeros((4,)),
            denoised_sample=mx.zeros((4,)),
            sigmas=mx.array([0.9, 0.6]),
            step_index=0,
        )


def test_s_noise_zero_still_rescales():
    # draw_noise est gated sur eta seul : à s_noise=0 le rescale
    # alpha_next/alpha_down s'applique quand même (docstring upstream).
    x = [1.0, -1.0]
    x0 = [0.2, 0.4]
    sigmas = mx.array([0.9, 0.6, 0.0])
    got = EulerAncestralDiffusionStep(eta=1.0, s_noise=0.0).step(
        sample=mx.array(x, dtype=mx.float32),
        denoised_sample=mx.array(x0, dtype=mx.float32),
        sigmas=sigmas,
        step_index=0,
        noise=mx.zeros((2,), dtype=mx.float32),
    )
    want = _reference_step(x, x0, 0.9, 0.6, 1.0, 0.0, [0.0, 0.0])
    assert mx.allclose(got, mx.array(want), atol=1e-6, rtol=1e-6).item()
