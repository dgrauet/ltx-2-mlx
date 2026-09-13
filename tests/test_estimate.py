"""Up-front denoising cost estimate (#94): state the work before step 1, refine to a time after it.

Pure tests -- no model. The work description is derived from the same inputs
the samplers already hold at loop entry (latent shapes, sigmas, guider
schedule), so it can never disagree with what the loop then executes.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import mlx.core as mx
import pytest

from ltx_core_mlx.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_pipelines_mlx.utils.estimate import RETAKE_COST_NOTE, DenoiseWork, StepEstimator, describe_work


def _state(num_tokens: int, *, preserved: int = 0) -> SimpleNamespace:
    mask = mx.ones((1, num_tokens, 1))
    if preserved:
        mask = mx.concatenate([mx.zeros((1, preserved, 1)), mx.ones((1, num_tokens - preserved, 1))], axis=1)
    return SimpleNamespace(latent=mx.zeros((1, num_tokens, 128)), denoise_mask=mask)


SIGMAS_4 = [1.0, 0.75, 0.5, 0.25, 0.0]  # 4 steps


def test_single_pass_work_counts_steps_and_tokens():
    work = describe_work(video_state=_state(1650), audio_state=_state(200), sigmas=SIGMAS_4)
    assert work == DenoiseWork(steps=4, forwards=4, video_tokens=1650, audio_tokens=200, passes_per_step=(1, 1, 1, 1))
    assert work.note is None


def test_cfg_doubles_forwards_and_stg_modality_add_passes():
    cfg = MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0))
    assert (
        describe_work(video_state=_state(8), audio_state=_state(2), sigmas=SIGMAS_4, video_guider_factory=cfg).forwards
        == 8
    )

    full = MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0, modality_scale=3.0))
    assert (
        describe_work(video_state=_state(8), audio_state=_state(2), sigmas=SIGMAS_4, video_guider_factory=full).forwards
        == 16
    )


def test_skip_step_removes_whole_steps():
    # skip_step=1 -> every other step is skipped (step % 2 != 0)
    f = MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0, skip_step=1))
    work = describe_work(video_state=_state(8), audio_state=_state(2), sigmas=SIGMAS_4, video_guider_factory=f)
    assert work.passes_per_step == (2, 0, 2, 0)
    assert work.forwards == 4


def test_sigma_scheduled_guidance_is_counted_per_step():
    # CFG only while sigma > 0.6, then plain -- upstream-style per-sigma schedule.
    f = MultiModalGuiderFactory.from_dict(
        {float("inf"): MultiModalGuiderParams(cfg_scale=3.0), 0.6: MultiModalGuiderParams(cfg_scale=1.0)}
    )
    work = describe_work(video_state=_state(8), audio_state=_state(2), sigmas=SIGMAS_4, video_guider_factory=f)
    assert work.passes_per_step == (2, 2, 1, 1)


def test_res2s_doubles_evaluations_and_counts_final_predict():
    work = describe_work(
        video_state=_state(8), audio_state=_state(2), sigmas=SIGMAS_4, evals_per_step=2, extra_forwards=1
    )
    assert work.forwards == 4 * 2 + 1


def test_non_uniform_mask_adds_retake_note():
    work = describe_work(video_state=_state(1650, preserved=1000), audio_state=_state(200), sigmas=SIGMAS_4)
    assert work.note == RETAKE_COST_NOTE
    assert work.video_tokens == 1650  # all tokens are computed, not just the regenerated window


def test_describe_line_states_the_work():
    cfg = MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0))
    work = describe_work(
        video_state=_state(1650), audio_state=_state(200), sigmas=[1.0] + [0.5] * 29 + [0.0], video_guider_factory=cfg
    )
    assert work.describe() == "30 steps x 2 passes over 1650 video + 200 audio tokens = 60 forwards"

    varying = describe_work(
        video_state=_state(8),
        audio_state=_state(2),
        sigmas=SIGMAS_4,
        video_guider_factory=MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0, skip_step=1)),
    )
    assert varying.describe() == "4 steps (2 skipped) x 2 passes over 8 video + 2 audio tokens = 4 forwards"


def test_estimator_announces_work_then_projects_time_after_first_step():
    out = io.StringIO()
    now = [100.0]
    cfg = MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0))
    work = describe_work(
        video_state=_state(1650), audio_state=_state(200), sigmas=[1.0] + [0.5] * 29 + [0.0], video_guider_factory=cfg
    )
    est = StepEstimator(work, show=True, out=out, clock=lambda: now[0], label="stage 1")

    est.announce()
    assert (
        out.getvalue() == "[estimate] stage 1: 30 steps x 2 passes over 1650 video + 200 audio tokens = 60 forwards\n"
    )

    now[0] += 87.2  # first step: 2 forwards -> 43.6 s/forward; 58 forwards remain -> 2528.8 s
    est.step_done(0)
    lines = out.getvalue().splitlines()
    assert lines[1] == "[estimate] stage 1: ~42 min 9 s remaining (43.6 s/forward) (first step includes warm-up)"

    now[0] += 40.0  # second step on its own: 20 s/forward; 56 forwards remain -> 1120 s
    est.step_done(1)
    lines = out.getvalue().splitlines()
    assert lines[2] == "[estimate] stage 1: ~18 min 40 s remaining (20.0 s/forward) (refined)"

    now[0] += 40.0
    est.step_done(2)
    assert len(out.getvalue().splitlines()) == 3, "after the refinement nothing more is printed"


def test_estimator_includes_note_and_is_silent_when_hidden():
    out = io.StringIO()
    work = describe_work(video_state=_state(16, preserved=8), audio_state=_state(2), sigmas=SIGMAS_4)
    StepEstimator(work, show=True, out=out, clock=lambda: 0.0, label="retake").announce()
    assert out.getvalue().endswith(f" ({RETAKE_COST_NOTE})\n")

    hidden = io.StringIO()
    est = StepEstimator(work, show=False, out=hidden, clock=lambda: 0.0, label="retake")
    est.announce()
    est.step_done(0)
    assert hidden.getvalue() == ""


def test_estimator_skips_a_zero_pass_first_step():
    """With skip_step the first computed step may not be index 0; time from the first step that did work."""
    out = io.StringIO()
    now = [0.0]
    f = MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0, skip_step=1))
    work = describe_work(video_state=_state(8), audio_state=_state(2), sigmas=SIGMAS_4, video_guider_factory=f)
    est = StepEstimator(work, show=True, out=out, clock=lambda: now[0], label="s")
    est.announce()
    now[0] += 10.0
    est.step_done(0)  # 2 forwards in 10 s -> 5 s/forward; 2 forwards remain
    assert (
        out.getvalue().splitlines()[1] == "[estimate] s: ~10 s remaining (5.0 s/forward) (first step includes warm-up)"
    )


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (5.0, "5 s"),
        (59.6, "1 min 0 s"),
        (90.0, "1 min 30 s"),
        (2528.8, "42 min 9 s"),
        (3600.0, "1 h 0 min"),
        (5400.0, "1 h 30 min"),
    ],
)
def test_duration_formatting(seconds, text):
    from ltx_pipelines_mlx.utils.estimate import format_duration

    assert format_duration(seconds) == text


# ---------------------------------------------------------------------------
# Wiring: every sampler loop announces its work and projects once
# ---------------------------------------------------------------------------

from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep  # noqa: E402
from ltx_core_mlx.conditioning.types.latent_cond import LatentState  # noqa: E402
from ltx_pipelines_mlx.utils.samplers import (  # noqa: E402
    denoise_loop,
    euler_ancestral_denoising_loop,
    guided_denoise_loop,
    res2s_denoise_loop,
)


class _StubX0Model:
    def __call__(self, *, video_latent, audio_latent, **kwargs):
        return mx.zeros_like(video_latent), mx.zeros_like(audio_latent)


def _latent_state(num_tokens: int) -> LatentState:
    return LatentState(
        latent=mx.ones((1, num_tokens, 16), dtype=mx.bfloat16),
        clean_latent=mx.zeros((1, num_tokens, 16), dtype=mx.bfloat16),
        denoise_mask=mx.ones((1, num_tokens, 1), dtype=mx.bfloat16),
    )


def _estimate_lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().err.splitlines() if line.startswith("[estimate]")]


_COMMON = dict(video_text_embeds=mx.zeros((1, 4, 16)), audio_text_embeds=mx.zeros((1, 4, 16)))


def test_denoise_loop_announces_and_projects(capsys):
    denoise_loop(
        model=_StubX0Model(), video_state=_latent_state(12), audio_state=_latent_state(3), sigmas=SIGMAS_4, **_COMMON
    )
    lines = _estimate_lines(capsys)
    assert lines[0] == "[estimate] denoising: 4 steps x 1 passes over 12 video + 3 audio tokens = 4 forwards"
    assert lines[1].startswith("[estimate] denoising: ~") and lines[1].endswith("(first step includes warm-up)")
    assert lines[2].startswith("[estimate] denoising: ~") and lines[2].endswith("(refined)")
    assert len(lines) == 3


def test_denoise_loop_is_silent_without_progress(capsys):
    denoise_loop(
        model=_StubX0Model(),
        video_state=_latent_state(12),
        audio_state=_latent_state(3),
        sigmas=SIGMAS_4,
        show_progress=False,
        **_COMMON,
    )
    assert _estimate_lines(capsys) == []


def test_ancestral_loop_announces(capsys):
    euler_ancestral_denoising_loop(
        SIGMAS_4,
        _latent_state(12),
        _latent_state(3),
        EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0),
        _StubX0Model(),
        noise_seed=0,
        **_COMMON,
    )
    lines = _estimate_lines(capsys)
    assert (
        lines[0] == "[estimate] denoising (ancestral): 4 steps x 1 passes over 12 video + 3 audio tokens = 4 forwards"
    )
    assert len(lines) == 3


def test_res2s_loop_counts_two_evals_per_step_plus_final(capsys):
    cfg = MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0), negative_context=mx.zeros((1, 4, 16)))
    res2s_denoise_loop(
        _StubX0Model(),
        _latent_state(12),
        _latent_state(3),
        sigmas=SIGMAS_4,
        video_guider_factory=cfg,
        bongmath=False,
        **_COMMON,
    )
    lines = _estimate_lines(capsys)
    # 4 steps x (2 passes x 2 evals) + 1 final predict = 17
    assert (
        lines[0]
        == "[estimate] denoising (res2s guided): 4 steps x 4 passes over 12 video + 3 audio tokens = 17 forwards"
    )
    assert len(lines) == 3


def test_guided_loop_counts_cfg_passes_and_retake_note(capsys):
    cfg = MultiModalGuiderFactory.constant(MultiModalGuiderParams(cfg_scale=3.0), negative_context=mx.zeros((1, 4, 16)))
    video = _latent_state(12)
    video.denoise_mask = mx.concatenate([mx.zeros((1, 6, 1)), mx.ones((1, 6, 1))], axis=1).astype(mx.bfloat16)
    guided_denoise_loop(_StubX0Model(), video, _latent_state(3), video_guider_factory=cfg, sigmas=SIGMAS_4, **_COMMON)
    lines = _estimate_lines(capsys)
    assert lines[0] == (
        f"[estimate] denoising (guided): 4 steps x 2 passes over 12 video + 3 audio tokens = 8 forwards ({RETAKE_COST_NOTE})"
    )
    assert len(lines) == 3


@pytest.mark.parametrize("command", ["retake", "extend"])
def test_retake_and_extend_help_carry_the_cost_note(command):
    from ltx_pipelines_mlx.cli import _build_parser

    sub = next(a for a in _build_parser()._subparsers._group_actions[0].choices.items() if a[0] == command)[1]
    assert "TOTAL clip length" in sub.format_help()
