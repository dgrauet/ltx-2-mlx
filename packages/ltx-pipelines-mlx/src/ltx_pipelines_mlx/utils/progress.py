"""CLI phase markers for long-running pipeline stages.

Writes brief ``[phase] ...`` / ``[phase] done in X.Ys`` lines to ``stderr``
so the user sees progress through silent stages (Gemma encode, DiT load,
VAE decode) without polluting ``stdout`` for callers that pipe it.

Gated by a single ``verbose`` flag plumbed from CLI ``--quiet``. Optional
telemetry and Metal-capture hooks remain dormant unless explicitly configured.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

_ProfileSink = Callable[..., None]
_PhaseCapture = tuple[str, str, Callable[[], None], Callable[[], None]]
_profile_sink: _ProfileSink | None = None
_phase_capture: _PhaseCapture | None = None


def _set_profile_sink(sink: _ProfileSink | None) -> _ProfileSink | None:
    """Install a process-local phase event sink and return the previous sink."""
    global _profile_sink
    previous = _profile_sink
    _profile_sink = sink
    return previous


def _set_phase_capture(capture: _PhaseCapture | None) -> _PhaseCapture | None:
    """Install a process-local phase capture and return the previous capture."""
    global _phase_capture
    previous = _phase_capture
    _phase_capture = capture
    return previous


@contextmanager
def phase(label: str, *, verbose: bool = True) -> Iterator[None]:
    """Print human progress and emit optional telemetry/capture events."""
    sink = _profile_sink
    capture = _phase_capture
    selected_capture = capture if capture is not None and capture[0] == label else None
    if not verbose and sink is None and selected_capture is None:
        yield
        return

    if verbose:
        print(f"[{label}] ...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    if sink is not None:
        sink("phase_start", phase=label)

    body_error: BaseException | None = None
    capture_started = False
    capture_t0 = 0.0
    stop_error: BaseException | None = None
    try:
        if selected_capture is not None:
            _, capture_path, start_capture, _ = selected_capture
            try:
                start_capture()
            except BaseException as exc:
                if sink is not None:
                    sink(
                        "metal_capture_error",
                        phase=label,
                        capture_path=capture_path,
                        operation="start",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                raise
            capture_started = True
            capture_t0 = time.perf_counter()
            if sink is not None:
                sink("metal_capture_start", phase=label, capture_path=capture_path)

        yield
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        if capture_started and selected_capture is not None:
            _, capture_path, _, stop_capture = selected_capture
            capture_elapsed = time.perf_counter() - capture_t0
            try:
                stop_capture()
            except BaseException as exc:
                stop_error = exc
                if sink is not None:
                    sink(
                        "metal_capture_error",
                        phase=label,
                        capture_path=capture_path,
                        operation="stop",
                        capture_elapsed_seconds=capture_elapsed,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
            else:
                if sink is not None:
                    sink(
                        "metal_capture_end",
                        phase=label,
                        capture_path=capture_path,
                        capture_elapsed_seconds=capture_elapsed,
                        phase_failed=body_error is not None,
                    )

        error = body_error if body_error is not None else stop_error
        dt = time.perf_counter() - t0
        if sink is not None:
            fields: dict[str, Any] = {"phase": label, "phase_elapsed_seconds": dt}
            if error is None:
                sink("phase_end", **fields)
            else:
                sink(
                    "phase_error",
                    **fields,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
        if verbose:
            print(f"[{label}] done in {dt:.1f}s", file=sys.stderr, flush=True)
        if body_error is None and stop_error is not None:
            raise stop_error
