"""LTX-2 sigma schedules.

`ltx2_schedule` is a thin wrapper over `mlx_arsenal.diffusion.dynamic_shift_schedule`
that preserves LTX's original keyword name (``steps``) and default ``num_tokens``.
The predefined LTX-specific tables (DISTILLED_SIGMAS, STAGE_2_SIGMAS) and the
LTX-only helpers (get_sigma_schedule, sigma_to_timestep) stay local.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_arsenal.diffusion import dynamic_shift_schedule

_MAX_SHIFT_ANCHOR = 4096


def ltx2_schedule(
    steps: int,
    num_tokens: int = _MAX_SHIFT_ANCHOR,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """LTX-2 token-count-adaptive flow-matching sigma schedule."""
    return dynamic_shift_schedule(
        steps,
        num_tokens=num_tokens,
        base_shift=base_shift,
        max_shift=max_shift,
        stretch=stretch,
        terminal=terminal,
    )


__all__ = [
    "DISTILLED_SIGMAS",
    "LTX_2_5_DISTILLED_SIGMAS",
    "LTX_2_5_STAGE_2_DISTILLED_SIGMAS",
    "STAGE_2_SIGMAS",
    "get_sigma_schedule",
    "ltx2_schedule",
    "sigma_to_timestep",
    "stage2_sigmas",
]

# Predefined sigma schedule for 8-step distilled model.
# 9 values = 8 steps (iterate consecutive pairs: sigmas[i], sigmas[i+1]).
DISTILLED_SIGMAS: list[float] = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# Sigma schedule for stage 2 refinement (two-stage pipeline).
# 4 values = 3 steps.
STAGE_2_SIGMAS: list[float] = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# Predefined sigma schedule for LTX-2.5 8-step distilled model.
# Upstream reuses the name DISTILLED_SIGMAS; we diverge on surface (LTX_2_5_ prefix)
# to keep both model tables in scope without ambiguity.
# 9 values = 8 steps (iterate consecutive pairs: sigmas[i], sigmas[i+1]).
LTX_2_5_DISTILLED_SIGMAS: list[float] = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# Sigma schedule for LTX-2.5 stage 2 refinement (two-stage pipeline).
# 4 values = 3 steps.
LTX_2_5_STAGE_2_DISTILLED_SIGMAS: list[float] = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]


def stage2_sigmas(stage2_steps: int | None = None) -> list[float]:
    """Stage-2 refinement sigma schedule that always terminates at 0.0.

    ``STAGE_2_SIGMAS`` is a fixed 4-value table (3 steps) whose terminal value
    is 0.0. Naively prefix-slicing it for ``stage2_steps < 3`` (e.g.
    ``STAGE_2_SIGMAS[:stage2_steps + 1]``) drops that terminal 0.0, so
    ``denoise_loop`` stops at a non-zero sigma and hands the VAE a latent that
    is still partially noised — which the decoder renders as heavy colour
    speckle. Always append the terminal 0.0 so the final Euler step fully
    denoises, for any step count.

    Args:
        stage2_steps: Number of stage-2 steps. ``None``/0 → the full default
            3-step schedule. Values ``>= 3`` are clamped to the 3 steps the
            table provides (no phantom zero-length final step).

    Note:
        Deliberately scoped to ``STAGE_2_SIGMAS``. The LTX-2.5 distilled path has
        its own value-identical ``LTX_2_5_STAGE_2_DISTILLED_SIGMAS`` and appears
        to carry the same truncation, but upstream's tests assert that slice AND
        the table's object identity on purpose, and the speckle this fixes was
        only ever reproduced on the dev two-stage path. Left alone rather than
        "fixed" on inference.

    Returns:
        Sigma list of length ``num_steps + 1``, always ending at 0.0.
    """
    if not stage2_steps:
        return list(STAGE_2_SIGMAS)
    head = list(STAGE_2_SIGMAS[:stage2_steps])
    if head and head[-1] == 0.0:
        # Only stage2_steps >= 4 lands here: STAGE_2_SIGMAS[:3] stops at 0.421875, so
        # stage2_steps == 3 takes the append path below and reconstructs the full table.
        # Same result either way; this branch just avoids appending a duplicate 0.0.
        return head
    return head + [0.0]


def get_sigma_schedule(
    schedule_name: str = "distilled",
    num_steps: int | None = None,
) -> list[float]:
    """Get a sigma schedule by name.

    Args:
        schedule_name: "distilled" or "stage_2".
        num_steps: Optional number of steps (truncates schedule).

    Returns:
        List of sigma values.
    """
    if schedule_name == "distilled":
        sigmas = DISTILLED_SIGMAS
    elif schedule_name == "stage_2":
        sigmas = STAGE_2_SIGMAS
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    if num_steps is not None:
        sigmas = sigmas[:num_steps]
    return sigmas


def sigma_to_timestep(sigma: float) -> mx.array:
    """Convert sigma to timestep array.

    Args:
        sigma: Noise level.

    Returns:
        Timestep as (1,) array.
    """
    return mx.array([sigma], dtype=mx.bfloat16)
