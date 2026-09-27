"""LTX-2 sigma schedules.

`ltx2_schedule` is a thin wrapper over `mlx_arsenal.diffusion.dynamic_shift_schedule`
that preserves LTX's original keyword name (``steps``) and default ``num_tokens``.
The predefined LTX-specific tables (DISTILLED_SIGMAS, STAGE_2_SIGMAS) and the
LTX-only helpers (get_sigma_schedule, sigma_to_timestep) stay local.
"""

from __future__ import annotations

from typing import Literal

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
    "shorten_schedule",
    "sigma_to_timestep",
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


def shorten_schedule(
    table: list[float],
    steps: int | None,
    *,
    keep: Literal["start", "tail"],
) -> list[float]:
    """Return a ``steps``-step version of ``table`` that still ends at ``table[-1]``.

    A plain slice (``table[: steps + 1]``) drops the terminal sigma, so the
    denoising loop stops part way down the schedule and returns a latent that
    still carries noise (``DISTILLED_SIGMAS[:4]`` ends at 0.98125). Every
    shortened schedule built here keeps the table's last value (0.0).

    Both modes keep the *tail* of the table, where the large sigma moves are:
    the distilled table is front-loaded (five of its eight steps sit between
    1.0 and 0.975), so keeping its head instead would spend the budget on
    near-zero moves and then jump from ~0.98 to 0.0 in one step. Upstream has
    no step-count parameter on these tables, so there is no reference to
    mirror; the tail is the convention ``ic_lora --refine-steps`` already uses.

    Args:
        table: A full sigma table, ending at 0.0.
        steps: Number of denoising steps wanted. ``None`` or 0, or a value
            at or above the table's own step count, returns ``table`` unchanged.
        keep: ``"start"`` pins ``table[0]`` and then takes the last ``steps``
            sigmas (``steps=3`` on ``DISTILLED_SIGMAS`` gives
            ``[1.0, 0.725, 0.421875, 0.0]``); use it for a stage that starts
            from pure noise (``create_noised_state(sigma=1.0)``). ``"tail"``
            keeps the last ``steps + 1`` sigmas, so the stage starts lower and
            re-noises less; use it for a refinement stage, where the stage's
            start sigma is ``sigmas[0]``.

    Returns:
        A list of ``steps + 1`` sigmas (or ``table`` itself).
    """
    if steps is not None and steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    if keep not in ("start", "tail"):
        raise ValueError(f"keep must be 'start' or 'tail', got {keep!r}")
    if not steps or steps >= len(table) - 1:
        return table
    if keep == "start":
        return [table[0], *table[-steps:]]
    return table[len(table) - 1 - steps :]


def sigma_to_timestep(sigma: float) -> mx.array:
    """Convert sigma to timestep array.

    Args:
        sigma: Noise level.

    Returns:
        Timestep as (1,) array.
    """
    return mx.array([sigma], dtype=mx.bfloat16)
