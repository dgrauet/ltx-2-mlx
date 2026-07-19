"""macOS GPU-watchdog failure recognition (#75).

On macOS 26.x + MLX 0.31.x the Metal watchdog kills sustained GPU work while
the display is active (upstream regression, ml-explore/mlx#3267). The raw
error gives users no clue about the known mitigations, so the CLI turns it
into actionable guidance.

Deliberate decision (#75): this module only *explains*; it never sets
``AGX_RELAX_CDM_CTXSTORE_TIMEOUT`` (or any driver env var) on the user's
behalf — the knob trades UI responsiveness for run stability and that
trade-off belongs to the user.
"""

from __future__ import annotations

_MARKERS = (
    "Impacting Interactivity",
    "kIOGPUCommandBufferCallbackErrorImpactingInteractivity",
)

_HINT = """\
This is the macOS GPU watchdog killing sustained GPU work while the display
is active — a known macOS 26.x + MLX 0.31.x regression, tracked upstream at
https://github.com/ml-explore/mlx/issues/3267 (runs on the same machines were
green on MLX 0.30).

Known mitigations (pick one, ltx-2-mlx never sets these for you):
  * relaunch with AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 — relaxes the watchdog
    timeout for THIS process only; the GPU may feel less responsive for the
    UI while the run is active, and on some machines it is not sufficient;
  * run with the display off (close the lid / let it sleep) — the only
    workaround reported 100% reliable upstream.

See docs/PIPELINES.md (environment variables) for details."""


def watchdog_hint(exc: BaseException) -> str | None:
    """Return actionable guidance when ``exc`` is the macOS GPU-watchdog kill.

    Matches the Metal ``Impacting Interactivity`` command-buffer failure by
    message content; returns ``None`` for every other error so callers can
    re-raise unrelated failures unchanged.
    """
    message = str(exc)
    if not any(marker in message for marker in _MARKERS):
        return None
    return _HINT
