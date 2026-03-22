"""Sigma schedules for diffusion denoising.

Ported from ltx-core/src/ltx_core/components/schedulers.py
"""

from __future__ import annotations

import mlx.core as mx

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


# --- Dynamic schedulers ---

_BASE_SHIFT_ANCHOR = 1024
_MAX_SHIFT_ANCHOR = 4096


def ltx2_schedule(
    steps: int,
    num_tokens: int = _MAX_SHIFT_ANCHOR,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
) -> list[float]:
    """Generate a dynamic sigma schedule with token-count-dependent shifting.

    Ported from ltx-core LTX2Scheduler. Used for non-distilled (full) models
    with CFG guidance.

    Args:
        steps: Number of denoising steps.
        num_tokens: Number of latent tokens (affects sigma shift).
        max_shift: Maximum shift parameter.
        base_shift: Base shift parameter.
        stretch: Whether to stretch sigmas to match terminal value.
        terminal: Terminal sigma value for stretching.

    Returns:
        List of steps+1 sigma values (includes terminal 0.0 if stretch).
    """
    import math

    import numpy as np

    sigmas = np.linspace(1.0, 0.0, steps + 1)

    # Compute shift based on token count
    mm = (max_shift - base_shift) / (_MAX_SHIFT_ANCHOR - _BASE_SHIFT_ANCHOR)
    b = base_shift - mm * _BASE_SHIFT_ANCHOR
    sigma_shift = num_tokens * mm + b

    sigmas = np.where(
        sigmas != 0,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1.0 / sigmas - 1.0)),
        0.0,
    )

    if stretch:
        non_zero = sigmas != 0
        non_zero_sigmas = sigmas[non_zero]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero] = stretched

    return sigmas.tolist()
