"""Focused tests for phase-scoped MLX Metal capture."""

from __future__ import annotations

import json
import sys

import pytest

from ltx_pipelines_mlx import cli
from ltx_pipelines_mlx.utils import perf_profile
from ltx_pipelines_mlx.utils.perf_profile import capture_phase, profile_run
from ltx_pipelines_mlx.utils.progress import phase

_MEMORY = {"mlx_active_gb": 1.0, "mlx_peak_gb": 2.0, "mlx_cache_gb": 0.25}
_STAGE1 = "Stage 1 half-resolution denoise"


class _FakeMetal:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def start_capture(self, path: str) -> None:
        self.calls.append(("start", path))

    def stop_capture(self) -> None:
        self.calls.append("stop")


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_capture_wraps_only_selected_phase(tmp_path, monkeypatch) -> None:
    calls = []
    capture_path = tmp_path / "stage1.gputrace"
    monkeypatch.setattr(perf_profile, "_metal_capture_api", lambda: _FakeMetal(calls))

    with capture_phase(capture_path, phase_label=_STAGE1):
        with phase("Encoding prompt", verbose=False):
            calls.append("other")
        with phase(_STAGE1, verbose=False):
            calls.append("body")

    assert calls == ["other", ("start", str(capture_path.resolve())), "body", "stop"]


def test_capture_stops_on_phase_error_and_records_jsonl(tmp_path, monkeypatch) -> None:
    calls = []
    capture_path = tmp_path / "stage1.gputrace"
    profile_path = tmp_path / "profile.jsonl"
    monkeypatch.setattr(perf_profile, "_metal_capture_api", lambda: _FakeMetal(calls))
    monkeypatch.setattr(perf_profile, "_mlx_memory_snapshot", lambda: _MEMORY)

    with (
        pytest.raises(ValueError, match="denoise failed"),
        profile_run(profile_path, metadata={"command": "generate", "metal_capture_phase": "stage1"}),
        capture_phase(capture_path, phase_label=_STAGE1),
        phase(_STAGE1, verbose=False),
    ):
        calls.append("body")
        raise ValueError("denoise failed")

    assert calls == [("start", str(capture_path.resolve())), "body", "stop"]
    events = _events(profile_path)
    assert [event["event"] for event in events] == [
        "run_start",
        "phase_start",
        "metal_capture_start",
        "metal_capture_end",
        "phase_error",
        "run_error",
    ]
    assert events[3]["phase_failed"] is True
    assert events[3]["capture_path"] == str(capture_path.resolve())


@pytest.mark.parametrize(
    "capture_args",
    [
        ["--metal-capture", "trace.gputrace"],
        ["--metal-capture-phase", "decode"],
        ["--metal-capture", "trace.gputrace", "--metal-capture-phase", "stage1"],
    ],
)
def test_cli_rejects_invalid_capture_combinations(capture_args, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(cli, "_cmd_generate", lambda args: called.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ltx-2-mlx",
            "generate",
            "--prompt",
            "test",
            "--output",
            "out.mp4",
            "--frame-rate",
            "24",
            "--one-stage",
            *capture_args,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert called == []


def test_cli_preflights_capture_environment_before_generation(tmp_path, monkeypatch, capsys) -> None:
    called = []
    monkeypatch.delenv("MTL_CAPTURE_ENABLED", raising=False)
    monkeypatch.setattr(cli, "_cmd_generate", lambda args: called.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ltx-2-mlx",
            "generate",
            "--prompt",
            "test",
            "--output",
            "out.mp4",
            "--frame-rate",
            "24",
            "--one-stage",
            "--metal-capture",
            str(tmp_path / "decode.gputrace"),
            "--metal-capture-phase",
            "decode",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert called == []
    error = capsys.readouterr().err
    assert "MTL_CAPTURE_ENABLED=1 ltx-2-mlx generate" in error
    assert "capturing-a-metal-workload-programmatically" in error
