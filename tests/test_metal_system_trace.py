"""Command construction coverage for opt-in Metal System Trace capture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_metal_system_trace.py"
_SPEC = importlib.util.spec_from_file_location("capture_metal_system_trace", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_record_command_is_scoped_to_target_and_explicit_environment(tmp_path) -> None:
    output = tmp_path / "stage2.trace"
    command = _MODULE._build_record_command(
        output,
        ["ltx-2-mlx", "generate", "--distilled"],
        time_limit="5m",
        environment=["LTX2_DIT_EVAL_EVERY=8"],
    )

    assert command[:5] == [
        "xcrun",
        "xctrace",
        "record",
        "--template",
        "Metal System Trace",
    ]
    assert command[command.index("--time-limit") : command.index("--time-limit") + 2] == [
        "--time-limit",
        "5m",
    ]
    assert command[command.index("--env") : command.index("--env") + 2] == [
        "--env",
        "LTX2_DIT_EVAL_EVERY=8",
    ]
    assert command[-5:] == [
        "--launch",
        "--",
        "ltx-2-mlx",
        "generate",
        "--distilled",
    ]
