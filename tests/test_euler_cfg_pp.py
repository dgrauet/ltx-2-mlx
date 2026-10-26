"""CFG++ step and ``euler_cfg_pp_denoising_loop`` (upstream ``ltx_pipelines.utils.samplers``).

The step is checked against a plain-Python transcription of upstream
``EulerCfgPpDiffusionStep.step`` / ``_get_ancestral_step``; the loop against
its contract: two forwards per non-terminal step (positive target, negative
direction), one on the terminal step, conditioning re-pinned after the noise.
"""

import math

import mlx.core as mx
import pytest

from ltx_core_mlx.components.diffusion_steps import EulerCfgPpDiffusionStep
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx.utils.samplers import denoise_loop, euler_cfg_pp_denoising_loop

NV, NA, NT, C = 8, 4, 3, 4
SIGMAS = [1.0, 0.7, 0.4, 0.0]


def _reference_step(x, x0, uncond, sigma_s, sigma_t, eta, s_noise, noise):
    """Upstream ``EulerCfgPpDiffusionStep.step`` in Python floats."""
    eps = 2.0**-23  # torch.finfo(torch.float32).eps
    alpha_s = max(1.0 - sigma_s, eps)
    alpha_t = max(1.0 - sigma_t, eps)
    rs, rt = sigma_s / alpha_s, sigma_t / alpha_t
    if eta:
        variance = rt**2 * max(rs**2 - rt**2, 0.0) / rs**2
        up = min(eta * math.sqrt(variance), rt)
        down = math.sqrt(max(rt**2 - up**2, 0.0))
    else:
        up, down = 0.0, rt
    down = alpha_t * down
    out = []
    for xi, x0i, ui, ni in zip(x, x0, uncond, noise):
        d = (xi - alpha_s * ui) / sigma_s
        v = alpha_t * x0i + down * d
        if eta > 0 and s_noise > 0:
            v += alpha_t * ni * s_noise * up
        out.append(v)
    return out


@pytest.mark.parametrize(("sigma_s", "sigma_t", "eta", "s_noise"), [(0.9, 0.6, 1.0, 1.0), (1.0, 0.7, 0.5, 0.8)])
def test_step_matches_upstream_formula(sigma_s, sigma_t, eta, s_noise):
    x, x0, uncond, noise = [0.8, -1.2, 0.3, 2.0], [0.1, -0.4, 0.05, 1.1], [0.3, -0.1, 0.2, 0.7], [0.5, -0.5, 1.0, -1.0]
    got = EulerCfgPpDiffusionStep(eta=eta, s_noise=s_noise).step(
        sample=mx.array(x, dtype=mx.float32),
        denoised_sample=mx.array(x0, dtype=mx.float32),
        sigmas=mx.array([sigma_s, sigma_t, 0.0], dtype=mx.float32),
        step_index=0,
        uncond_denoised=mx.array(uncond, dtype=mx.float32),
        noise=mx.array(noise, dtype=mx.float32),
    )
    want = _reference_step(x, x0, uncond, sigma_s, sigma_t, eta, s_noise, noise)
    assert mx.allclose(got, mx.array(want), atol=1e-5, rtol=1e-5).item()


class _Recorder:
    """X0 stub: depends on the latent and the text context, records each call."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, *, video_latent, audio_latent, video_text_embeds, audio_text_embeds, **kw):
        self.calls.append(dict(kw, video_text_embeds=video_text_embeds, audio_text_embeds=audio_text_embeds))
        v = mx.tanh(video_latent.astype(mx.float32)) * 0.7 + mx.mean(video_text_embeds)
        a = mx.tanh(audio_latent.astype(mx.float32)) * 0.6 + mx.mean(audio_text_embeds)
        return v, a


def _states(preserved: int = 0) -> tuple[LatentState, LatentState]:
    mx.random.seed(3)
    video_mask = mx.concatenate([mx.zeros((1, preserved, 1)), mx.ones((1, NV - preserved, 1))], axis=1)
    video = LatentState(
        mx.random.normal((1, NV, C)).astype(mx.bfloat16),
        mx.full((1, NV, C), 3.0, dtype=mx.bfloat16),
        video_mask.astype(mx.bfloat16),
    )
    audio = LatentState(
        mx.random.normal((1, NA, C)).astype(mx.bfloat16),
        mx.zeros((1, NA, C), dtype=mx.bfloat16),
        mx.ones((1, NA, 1), dtype=mx.bfloat16),
    )
    return video, audio


POSITIVE = mx.ones((1, NT, C))
NEGATIVE = -mx.ones((1, NT, C))


def _run(model, video, audio, sigmas=SIGMAS, eta=1.0, s_noise=1.0, seed=5, negative=NEGATIVE, **extra):
    return euler_cfg_pp_denoising_loop(
        sigmas,
        video,
        audio,
        EulerCfgPpDiffusionStep(eta=eta, s_noise=s_noise),
        model,
        POSITIVE,
        POSITIVE,
        negative,
        negative,
        noise_seed=seed,
        show_progress=False,
        **extra,
    )


def test_two_forwards_per_step_and_one_on_the_terminal_step():
    model = _Recorder()
    relay = mx.ones((1, NV, NT))
    _run(model, *_states(), video_cross_attention_mask=relay)
    assert len(model.calls) == 2 * (len(SIGMAS) - 2) + 1
    positive, negative = model.calls[0], model.calls[1]
    assert positive["video_text_embeds"] is POSITIVE and negative["video_text_embeds"] is NEGATIVE
    assert negative["audio_text_embeds"] is NEGATIVE
    # The negative pass never gets the Prompt Relay mask.
    assert positive["video_cross_attention_mask"] is relay and "video_cross_attention_mask" not in negative
    # Both passes share everything else (sigma, positions, per-token timesteps).
    shared = set(positive) - {"video_text_embeds", "audio_text_embeds", "video_cross_attention_mask"}
    assert shared == set(negative) - {"video_text_embeds", "audio_text_embeds"}
    assert model.calls[-1]["video_text_embeds"] is POSITIVE


def test_same_context_and_no_noise_is_the_euler_trajectory():
    # With uncond == cond and eta = 0 the CFG++ update is the DDIM step, which on
    # the rectified-flow schedule equals the Euler step of denoise_loop.
    video, audio = _states()
    cfg_pp = _run(_Recorder(), video, audio, eta=0.0, negative=POSITIVE)
    euler = denoise_loop(
        model=_Recorder(),
        video_state=video,
        audio_state=audio,
        video_text_embeds=POSITIVE,
        audio_text_embeds=POSITIVE,
        sigmas=SIGMAS,
        show_progress=False,
    )
    for got, want in ((cfg_pp.video_latent, euler.video_latent), (cfg_pp.audio_latent, euler.audio_latent)):
        assert mx.allclose(got.astype(mx.float32), want.astype(mx.float32), atol=2e-2, rtol=2e-2).item()


def test_negative_prompt_changes_the_result():
    video, audio = _states()
    same = _run(_Recorder(), video, audio, negative=POSITIVE)
    other = _run(_Recorder(), video, audio)
    assert not mx.array_equal(same.video_latent, other.video_latent).item()


def test_seeded_noise():
    video, audio = _states()
    a = _run(_Recorder(), video, audio, seed=5)
    b = _run(_Recorder(), video, audio, seed=5)
    c = _run(_Recorder(), video, audio, seed=6)
    assert mx.array_equal(a.video_latent, b.video_latent).item()
    assert mx.array_equal(a.audio_latent, b.audio_latent).item()
    assert not mx.array_equal(a.video_latent, c.video_latent).item()


def test_preserved_tokens_stay_clean_after_the_noise():
    # A schedule without a terminal sigma ends on a renoised step, so the
    # preserved tokens can only stay clean if the mask is re-applied after the noise.
    video, audio = _states(preserved=3)
    out = _run(_Recorder(), video, audio, sigmas=[1.0, 0.7, 0.4])
    assert mx.array_equal(out.video_latent[:, :3], video.clean_latent[:, :3]).item()
