"""Up-front denoising cost estimate (#94).

tqdm cannot say anything until its first iteration completes, and at tens of
seconds per step that is exactly the window in which a user decides the run
is hung. Everything needed to state the *work* is known at loop entry:

- token counts, from the latent shapes;
- step count, from the sigma schedule;
- forwards per step, from the guider schedule (``cond`` always, ``uncond``
  under CFG, ``ptb`` under STG, ``mod`` under modality isolation, none on a
  ``skip_step`` step), times the sampler's evaluations per step (res_2s: 2).

Seconds-per-forward is the one unknown (chip, quantisation, ``--low-ram``,
memory pressure), so it is measured on the first computed step and projected
over the remaining forwards, once. Both lines go to ``stderr`` next to the
``[phase]`` markers and are gated by the same ``show_progress`` flag as the
tqdm bar.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TextIO

import mlx.core as mx
from tqdm import tqdm

if TYPE_CHECKING:
    from ltx_core_mlx.components.guiders import MultiModalGuiderFactory

#: Appended to the estimate when part of the sequence is preserved (retake /
#: extend): every token is still computed and attended over on every pass.
RETAKE_COST_NOTE = "cost follows total clip length, not the regenerated window"

#: One projection after the first computed step, one refinement after the second.
_MAX_PROJECTIONS = 2


@dataclass(frozen=True)
class DenoiseWork:
    """The work one denoising stage will do, derived from its inputs."""

    steps: int
    forwards: int
    video_tokens: int
    audio_tokens: int
    passes_per_step: tuple[int, ...]
    note: str | None = None

    def describe(self) -> str:
        """``"30 steps x 2 passes over 1650 video + 200 audio tokens = 60 forwards"``.

        Skipped steps (``skip_step``) are called out rather than averaged in;
        a sigma-scheduled pass count shows as a range.
        """
        computed = [p for p in self.passes_per_step if p > 0]
        skipped = len(self.passes_per_step) - len(computed)
        lo, hi = (min(computed), max(computed)) if computed else (0, 0)
        passes = f"{lo}" if lo == hi else f"{lo}-{hi}"
        steps = f"{self.steps} steps" + (f" ({skipped} skipped)" if skipped else "")
        return f"{steps} x {passes} passes over {self.video_tokens} video + {self.audio_tokens} audio tokens = {self.forwards} forwards"


def _passes_for_sigma(factory: MultiModalGuiderFactory | None, sigma: float, step_idx: int) -> int:
    if factory is None:
        return 1
    guider = factory.build_from_sigma(sigma)
    if guider.should_skip_step(step_idx):
        return 0
    return (
        1
        + int(guider.do_unconditional_generation())
        + int(guider.do_perturbed_generation())
        + int(guider.do_isolated_modality_generation())
    )


def describe_work(
    *,
    video_state,
    audio_state,
    sigmas: list[float],
    video_guider_factory: MultiModalGuiderFactory | None = None,
    evals_per_step: int = 1,
    extra_forwards: int = 0,
) -> DenoiseWork:
    """Count the forwards a denoising loop over ``sigmas`` will run.

    Args:
        video_state: Latent state; ``latent`` is ``(B, N, C)`` and
            ``denoise_mask`` marks preserved tokens with ``0``.
        audio_state: Same for audio.
        sigmas: Sigma schedule including the terminal value; one step per
            consecutive pair, as the samplers iterate it.
        video_guider_factory: Guider schedule deciding the passes per step.
            ``None`` means a single conditioned pass.
        evals_per_step: Model evaluations per step (res_2s: 2).
        extra_forwards: Forwards outside the step loop (res_2s' final
            predict at ``sigma == 0``).
    """
    step_sigmas = sigmas[:-1]
    passes = tuple(
        _passes_for_sigma(video_guider_factory, float(s), i) * evals_per_step for i, s in enumerate(step_sigmas)
    )
    uniform = bool(mx.all(video_state.denoise_mask == 1.0).item())
    return DenoiseWork(
        steps=len(step_sigmas),
        forwards=sum(passes) + extra_forwards,
        video_tokens=int(video_state.latent.shape[1]),
        audio_tokens=int(audio_state.latent.shape[1]),
        passes_per_step=passes,
        note=None if uniform else RETAKE_COST_NOTE,
    )


def format_duration(seconds: float) -> str:
    """``5 s`` / ``1 min 30 s`` / ``1 h 30 min``."""
    total = round(seconds)
    if total < 60:
        return f"{total} s"
    if total < 3600:
        return f"{total // 60} min {total % 60} s"
    return f"{total // 3600} h {(total % 3600) // 60} min"


class StepEstimator:
    """Print the work up front, then one time projection after the first computed step."""

    def __init__(
        self,
        work: DenoiseWork,
        *,
        show: bool,
        label: str,
        out: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._work = work
        self._show = show
        self._label = label
        self._out = out if out is not None else sys.stderr
        self._clock = clock
        self._t0: float | None = None
        self._last_t: float | None = None
        self._projections = 0

    @property
    def wants_sync(self) -> bool:
        """True until the projection is printed: the caller should ``mx.eval`` the step's outputs before ``step_done``.

        The loops dispatch each step with ``mx.async_eval``; without one
        blocking eval on the timed step the measured duration would be the
        graph dispatch, not the GPU work.
        """
        return self._show and self._projections < _MAX_PROJECTIONS and self._t0 is not None

    def announce(self) -> None:
        """Print the work line and start the clock."""
        if not self._show:
            return
        line = f"[estimate] {self._label}: {self._work.describe()}"
        if self._work.note:
            line += f" ({self._work.note})"
        self._write(line)
        self._t0 = self._last_t = self._clock()

    def step_done(self, step_idx: int) -> None:
        """Call after each step. Projects the remaining time after the first computed step, refines it
        once after the second, then stays quiet.

        The first step carries one-off costs (kernel compilation, cache warm-up,
        weight paging) and can run several times slower than steady state, so
        the first projection is an upper bound; the second is timed on its own
        step only and is the one to trust.
        """
        if not self.wants_sync:
            return
        passes = self._work.passes_per_step
        done = passes[step_idx] if step_idx < len(passes) else 0
        if done <= 0:
            return
        now = self._clock()
        assert self._last_t is not None
        per_forward = (now - self._last_t) / done
        self._last_t = now
        remaining = (self._work.forwards - sum(passes[: step_idx + 1])) * per_forward
        self._projections += 1
        tag = " (refined)" if self._projections > 1 else " (first step includes warm-up)"
        self._write(
            f"[estimate] {self._label}: ~{format_duration(remaining)} remaining ({per_forward:.1f} s/forward){tag}"
        )

    def _write(self, line: str) -> None:
        # tqdm.write clears any active bar, prints the line, and redraws the
        # bar below it, so the estimate never lands mid-bar.
        tqdm.write(line, file=self._out)
        self._out.flush()
