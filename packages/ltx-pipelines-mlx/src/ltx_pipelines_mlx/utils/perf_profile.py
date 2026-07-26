"""Opt-in, crash-resilient JSONL telemetry for CLI generation runs."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from ltx_pipelines_mlx.utils.progress import _set_phase_capture, _set_profile_sink

_GB = 1024**3


def _mlx_memory_snapshot() -> dict[str, float | str | None]:
    """Read MLX allocator counters without making telemetry fatal to inference."""
    try:
        import mlx.core as mx

        return {
            "mlx_active_gb": mx.get_active_memory() / _GB,
            "mlx_peak_gb": mx.get_peak_memory() / _GB,
            "mlx_cache_gb": mx.get_cache_memory() / _GB,
        }
    except Exception as exc:  # pragma: no cover - defensive for unsupported MLX runtimes
        return {
            "mlx_active_gb": None,
            "mlx_peak_gb": None,
            "mlx_cache_gb": None,
            "mlx_memory_error": f"{type(exc).__name__}: {exc}",
        }


class _JsonlProfiler:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.run_id = str(uuid.uuid4())
        self.started = time.perf_counter()
        self._stream: TextIO | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def emit(self, event: str, **fields: Any) -> None:
        if self._stream is None:
            raise RuntimeError("Profiler stream is not open")
        record = {
            "schema_version": 1,
            "event": event,
            "run_id": self.run_id,
            "timestamp_unix_seconds": time.time(),
            "elapsed_seconds": time.perf_counter() - self.started,
            **_mlx_memory_snapshot(),
            **fields,
        }
        self._stream.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        self._stream.flush()


def _metal_capture_api() -> Any:
    """Return MLX's Metal capture API, imported only for an enabled capture."""
    import mlx.core as mx

    return mx.metal


@contextmanager
def capture_phase(path: str | Path, *, phase_label: str) -> Iterator[None]:
    """Capture exactly one named pipeline phase with MLX's Metal debugger hook."""
    capture_path = Path(path).expanduser().resolve()
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    metal_api: Any = None

    def start_capture() -> None:
        nonlocal metal_api
        metal_api = _metal_capture_api()
        try:
            metal_api.start_capture(str(capture_path))
        except RuntimeError as exc:
            raise RuntimeError(
                "MLX Metal capture failed to start. On macOS 14+, MTL_CAPTURE_ENABLED=1 must be set "
                "when launching the process. See "
                "https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically. "
                f"MLX reported: {exc}"
            ) from exc

    def stop_capture() -> None:
        if metal_api is None:
            raise RuntimeError("Metal capture was not started")
        metal_api.stop_capture()

    previous_capture = _set_phase_capture((phase_label, str(capture_path), start_capture, stop_capture))
    try:
        yield
    finally:
        _set_phase_capture(previous_capture)


@contextmanager
def profile_run(path: str | Path, *, metadata: Mapping[str, object]) -> Iterator[None]:
    """Append a fully flushed JSONL event stream for one generation run."""
    profiler = _JsonlProfiler(path)
    profiler.open()
    previous_sink = _set_profile_sink(profiler.emit)
    try:
        profiler.emit(
            "run_start",
            status="running",
            process_id=os.getpid(),
            python_version=sys.version.split()[0],
            metadata=dict(metadata),
        )
        try:
            yield
        except BaseException as exc:
            profiler.emit(
                "run_error",
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        else:
            profiler.emit("run_end", status="success")
    finally:
        _set_profile_sink(previous_sink)
        profiler.close()
