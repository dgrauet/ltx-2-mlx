"""Opt-in, crash-resilient JSONL telemetry for CLI generation runs."""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, TextIO

from ltx_pipelines_mlx.utils.progress import _set_phase_capture, _set_profile_sink

_GB = 1024**3


def _process_memory_snapshot() -> dict[str, float | None]:
    """Read current RSS and the Darwin physical-footprint high-water mark."""
    rss_gb: float | None = None
    footprint_gb: float | None = None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes = usage.ru_maxrss if platform.system() == "Darwin" else usage.ru_maxrss * 1024
        rss_gb = rss_bytes / _GB
    except Exception:  # pragma: no cover - platform defensive path
        pass

    if platform.system() == "Darwin":
        try:
            import ctypes

            class RUsageInfoV4(ctypes.Structure):
                _fields_ = [
                    ("ri_uuid", ctypes.c_uint8 * 16),
                    ("ri_user_time", ctypes.c_uint64),
                    ("ri_system_time", ctypes.c_uint64),
                    ("ri_pkg_idle_wkups", ctypes.c_uint64),
                    ("ri_interrupt_wkups", ctypes.c_uint64),
                    ("ri_pageins", ctypes.c_uint64),
                    ("ri_wired_size", ctypes.c_uint64),
                    ("ri_resident_size", ctypes.c_uint64),
                    ("ri_phys_footprint", ctypes.c_uint64),
                    ("ri_proc_start_abstime", ctypes.c_uint64),
                    ("ri_proc_exit_abstime", ctypes.c_uint64),
                    ("ri_child_user_time", ctypes.c_uint64),
                    ("ri_child_system_time", ctypes.c_uint64),
                    ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
                    ("ri_child_interrupt_wkups", ctypes.c_uint64),
                    ("ri_child_pageins", ctypes.c_uint64),
                    ("ri_child_elapsed_abstime", ctypes.c_uint64),
                    ("ri_diskio_bytesread", ctypes.c_uint64),
                    ("ri_diskio_byteswritten", ctypes.c_uint64),
                    ("ri_cpu_time_qos_default", ctypes.c_uint64),
                    ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
                    ("ri_cpu_time_qos_background", ctypes.c_uint64),
                    ("ri_cpu_time_qos_utility", ctypes.c_uint64),
                    ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
                    ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
                    ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
                    ("ri_billed_system_time", ctypes.c_uint64),
                    ("ri_serviced_system_time", ctypes.c_uint64),
                    ("ri_logical_writes", ctypes.c_uint64),
                    ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
                    ("ri_instructions", ctypes.c_uint64),
                    ("ri_cycles", ctypes.c_uint64),
                    ("ri_billed_energy", ctypes.c_uint64),
                    ("ri_serviced_energy", ctypes.c_uint64),
                    ("ri_interval_max_phys_footprint", ctypes.c_uint64),
                    ("ri_runnable_time", ctypes.c_uint64),
                ]

            library = ctypes.CDLL("/usr/lib/libproc.dylib")
            proc_pid_rusage = library.proc_pid_rusage
            proc_pid_rusage.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(RUsageInfoV4),
            ]
            proc_pid_rusage.restype = ctypes.c_int
            info = RUsageInfoV4()
            if proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
                footprint_gb = info.ri_lifetime_max_phys_footprint / _GB
                rss_gb = info.ri_resident_size / _GB
        except Exception:  # pragma: no cover - platform defensive path
            pass
    return {
        "process_rss_gb": rss_gb,
        "process_lifetime_max_phys_footprint_gb": footprint_gb,
    }


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


@lru_cache(maxsize=1)
def runtime_identity() -> dict[str, object]:
    """Return package, MLX, device, and source revision identity."""
    import importlib.metadata

    versions: dict[str, str | None] = {}
    for distribution in ("ltx-pipelines-mlx", "ltx-core-mlx", "mlx", "mlx-metal"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None

    commit = os.environ.get("LTX2_RUNTIME_COMMIT")
    dirty: bool | None = None
    source_root: Path | None = None
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            source_root = parent
            break
    if source_root is not None and commit is None:
        try:
            resolved = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            commit = resolved.stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            dirty = bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    if commit is None:
        commit = f"package:{versions['ltx-pipelines-mlx'] or 'unknown'}"

    device: dict[str, object] = {}
    try:
        import mlx.core as mx

        device = dict(mx.device_info())
    except Exception:  # pragma: no cover - unsupported MLX runtime
        pass
    return {
        "runtime_commit": commit,
        "runtime_dirty": dirty,
        "runtime_version": versions["ltx-pipelines-mlx"],
        "core_version": versions["ltx-core-mlx"],
        "mlx_version": versions["mlx"],
        "mlx_metal_version": versions["mlx-metal"],
        "device_name": device.get("device_name"),
        "device_architecture": device.get("architecture"),
        "device_memory_bytes": device.get("memory_size"),
        "device_recommended_working_set_bytes": device.get("max_recommended_working_set_size"),
    }


class _JsonlProfiler:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.run_id = str(uuid.uuid4())
        self.started = time.perf_counter()
        self._stream: TextIO | None = None
        self._observed_peak_mlx_gb = 0.0
        self._observed_peak_phys_footprint_gb = 0.0

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
        memory = {
            **_mlx_memory_snapshot(),
            **_process_memory_snapshot(),
        }
        mlx_peak = memory.get("mlx_peak_gb")
        if isinstance(mlx_peak, (float, int)):
            self._observed_peak_mlx_gb = max(
                self._observed_peak_mlx_gb,
                float(mlx_peak),
            )
        footprint_peak = memory.get("process_lifetime_max_phys_footprint_gb")
        if isinstance(footprint_peak, (float, int)):
            self._observed_peak_phys_footprint_gb = max(
                self._observed_peak_phys_footprint_gb,
                float(footprint_peak),
            )
        if event in {"run_end", "run_error"}:
            fields.setdefault(
                "observed_peak_mlx_gb",
                self._observed_peak_mlx_gb,
            )
            fields.setdefault(
                "observed_peak_phys_footprint_gb",
                self._observed_peak_phys_footprint_gb or None,
            )
        record = {
            "schema_version": 2,
            "event": event,
            "run_id": self.run_id,
            "timestamp_unix_seconds": time.time(),
            "elapsed_seconds": time.perf_counter() - self.started,
            **memory,
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
