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
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx.utils.samplers import denoise_loop, euler_ancestral_denoising_loop


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


# ---------------------------------------------------------------------------
# euler_ancestral_denoising_loop
# ---------------------------------------------------------------------------

VIDEO_TOKENS = 8
AUDIO_TOKENS = 4
CHANNELS = 4


class _StubX0Model:
    """Deterministic X0Model stub: x0 = 0.5 * latent, for both modalities."""

    def __call__(self, *, video_latent, audio_latent, **_kwargs):
        return 0.5 * video_latent, 0.5 * audio_latent


def _stub_transformer_and_states(
    denoise_mask_prefix_zero: int = 0,
    clean_value: float = 3.0,
):
    """A minimal video+audio LatentState pair and a deterministic X0Model stub.

    ``denoise_mask_prefix_zero`` zeros out that many leading video tokens in
    the denoise mask (preserved/conditioning tokens); the rest stay at 1.0
    (fully denoised). ``clean_value`` is the clean_latent fill value for the
    preserved tokens, chosen far from the noisy/init latent so a leaked
    renoise is trivially detectable.
    """
    video_latent = mx.random.normal((1, VIDEO_TOKENS, CHANNELS)).astype(mx.bfloat16)
    audio_latent = mx.random.normal((1, AUDIO_TOKENS, CHANNELS)).astype(mx.bfloat16)

    video_mask = mx.ones((1, VIDEO_TOKENS, 1), dtype=mx.bfloat16)
    if denoise_mask_prefix_zero:
        prefix = mx.zeros((1, denoise_mask_prefix_zero, 1), dtype=mx.bfloat16)
        rest = mx.ones((1, VIDEO_TOKENS - denoise_mask_prefix_zero, 1), dtype=mx.bfloat16)
        video_mask = mx.concatenate([prefix, rest], axis=1)

    video_clean = mx.full((1, VIDEO_TOKENS, CHANNELS), clean_value, dtype=mx.bfloat16)
    audio_clean = mx.zeros((1, AUDIO_TOKENS, CHANNELS), dtype=mx.bfloat16)

    video_state = LatentState(latent=video_latent, clean_latent=video_clean, denoise_mask=video_mask)
    audio_state = LatentState(
        latent=audio_latent, clean_latent=audio_clean, denoise_mask=mx.ones((1, AUDIO_TOKENS, 1), dtype=mx.bfloat16)
    )
    return video_state, audio_state, _StubX0Model()


def _common_loop_kwargs(video_state, audio_state, transformer):
    return dict(
        video_state=video_state,
        audio_state=audio_state,
        transformer=transformer,
        video_text_embeds=mx.zeros((1, 2, CHANNELS)),
        audio_text_embeds=mx.zeros((1, 2, CHANNELS)),
        show_progress=False,
    )


def test_loop_eta_zero_matches_denoise_loop():
    # eta=0 => no noise is drawn; the trajectory must coincide with
    # denoise_loop on the same stub (allclose — FP evaluation order may differ).
    video_state, audio_state, transformer = _stub_transformer_and_states()
    sigmas = [1.0, 0.6, 0.3, 0.0]

    ancestral_out = euler_ancestral_denoising_loop(
        sigmas,
        stepper=EulerAncestralDiffusionStep(eta=0.0),
        noise_seed=0,
        **_common_loop_kwargs(video_state, audio_state, transformer),
    )
    euler_out = denoise_loop(
        model=transformer,
        video_state=video_state,
        audio_state=audio_state,
        video_text_embeds=mx.zeros((1, 2, CHANNELS)),
        audio_text_embeds=mx.zeros((1, 2, CHANNELS)),
        sigmas=sigmas,
        show_progress=False,
    )

    assert mx.allclose(
        ancestral_out.video_latent.astype(mx.float32), euler_out.video_latent.astype(mx.float32), atol=1e-2, rtol=1e-2
    ).item()
    assert mx.allclose(
        ancestral_out.audio_latent.astype(mx.float32), euler_out.audio_latent.astype(mx.float32), atol=1e-2, rtol=1e-2
    ).item()


def test_loop_determinism_and_seed_sensitivity():
    # Same seed -> bit-identical outputs; different seed -> different outputs.
    sigmas = [1.0, 0.6, 0.3, 0.0]
    stepper = EulerAncestralDiffusionStep(eta=1.0)

    video_state, audio_state, transformer = _stub_transformer_and_states()
    out_a = euler_ancestral_denoising_loop(
        sigmas, stepper=stepper, noise_seed=42, **_common_loop_kwargs(video_state, audio_state, transformer)
    )
    out_b = euler_ancestral_denoising_loop(
        sigmas, stepper=stepper, noise_seed=42, **_common_loop_kwargs(video_state, audio_state, transformer)
    )
    assert mx.array_equal(out_a.video_latent, out_b.video_latent).item()
    assert mx.array_equal(out_a.audio_latent, out_b.audio_latent).item()

    out_c = euler_ancestral_denoising_loop(
        sigmas, stepper=stepper, noise_seed=7, **_common_loop_kwargs(video_state, audio_state, transformer)
    )
    assert not mx.array_equal(out_a.video_latent, out_c.video_latent).item()


def test_preserved_tokens_stay_clean_after_renoise():
    # denoise_mask=0 on a block of tokens: after the loop (eta=1), those
    # tokens must equal clean_latent exactly — the mask is re-applied AFTER
    # the noise injection. This is the exact point that breaks I2V if forgotten.
    #
    # The primary schedule deliberately does NOT end at sigma=0: with a
    # terminal sigma, the loop's last iteration takes the short-circuit
    # branch (latent = already-mask-blended x0), which passes even if the
    # post-noise apply_denoise_mask block is deleted — making the assertion
    # vacuous. A non-terminal schedule forces the final state through the
    # renoise + re-mask path, so the assertion can only pass through that
    # block (verified by neutering it locally: the assertion then fails).
    video_state, audio_state, transformer = _stub_transformer_and_states(denoise_mask_prefix_zero=3)
    expected = video_state.clean_latent[:, :3, :]

    non_terminal_sigmas = [1.0, 0.6, 0.3]
    out = euler_ancestral_denoising_loop(
        non_terminal_sigmas,
        stepper=EulerAncestralDiffusionStep(eta=1.0),
        noise_seed=123,
        **_common_loop_kwargs(video_state, audio_state, transformer),
    )
    preserved = out.video_latent[:, :3, :]
    assert mx.array_equal(preserved, expected).item()

    # Terminal schedule covers the short-circuit path separately.
    terminal_sigmas = [1.0, 0.6, 0.3, 0.0]
    out_terminal = euler_ancestral_denoising_loop(
        terminal_sigmas,
        stepper=EulerAncestralDiffusionStep(eta=1.0),
        noise_seed=123,
        **_common_loop_kwargs(video_state, audio_state, transformer),
    )
    preserved_terminal = out_terminal.video_latent[:, :3, :]
    assert mx.array_equal(preserved_terminal, expected).item()


def test_25_sigma_tables():
    from ltx_pipelines_mlx.scheduler import (
        DISTILLED_SIGMAS,
        LTX_2_5_DISTILLED_SIGMAS,
        LTX_2_5_STAGE_2_DISTILLED_SIGMAS,
        STAGE_2_SIGMAS,
    )

    assert LTX_2_5_DISTILLED_SIGMAS == [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    assert LTX_2_5_STAGE_2_DISTILLED_SIGMAS == [0.909375, 0.725, 0.421875, 0.0]
    # Le 0.0 terminal est porteur (bug de troncature témoin) et l'invariant
    # de sous-schedule est le même qu'en 2.3.
    assert LTX_2_5_DISTILLED_SIGMAS[-1] == 0.0
    assert LTX_2_5_DISTILLED_SIGMAS[-4:] == LTX_2_5_STAGE_2_DISTILLED_SIGMAS
    # Les tables 2.3 ne bougent pas d'un octet.
    assert DISTILLED_SIGMAS[-4:] == STAGE_2_SIGMAS
