"""Tests for opt-in JSONL generation telemetry."""

from __future__ import annotations

import json
import sys

import pytest

from ltx_pipelines_mlx import cli
from ltx_pipelines_mlx.utils import perf_profile
from ltx_pipelines_mlx.utils.perf_profile import profile_run
from ltx_pipelines_mlx.utils.progress import phase

_MEMORY = {"mlx_active_gb": 1.25, "mlx_peak_gb": 2.5, "mlx_cache_gb": 0.5}


def _events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_quiet_phase_events_are_flushed_during_run(tmp_path, monkeypatch) -> None:
    output = tmp_path / "profile.jsonl"
    monkeypatch.setattr(perf_profile, "_mlx_memory_snapshot", lambda: _MEMORY)
    monkeypatch.setattr(
        perf_profile,
        "_process_memory_snapshot",
        lambda: {
            "process_rss_gb": 3.0,
            "process_lifetime_max_phys_footprint_gb": 4.0,
        },
    )

    with profile_run(output, metadata={"command": "generate"}):
        with phase("Encoding prompt", verbose=False):
            pass
        in_flight = _events(output)
        assert [event["event"] for event in in_flight] == ["run_start", "phase_start", "phase_end"]

    events = _events(output)
    assert [event["event"] for event in events] == ["run_start", "phase_start", "phase_end", "run_end"]
    assert all(event["mlx_active_gb"] == 1.25 for event in events)
    assert events[-1]["observed_peak_mlx_gb"] == 2.5
    assert events[-1]["observed_peak_phys_footprint_gb"] == 4.0
    assert events[-1]["schema_version"] == 2
    assert all(event["elapsed_seconds"] >= 0 for event in events)
    assert events[2]["phase_elapsed_seconds"] >= 0


def test_phase_and_run_errors_are_recorded(tmp_path, monkeypatch) -> None:
    output = tmp_path / "profile.jsonl"
    monkeypatch.setattr(perf_profile, "_mlx_memory_snapshot", lambda: _MEMORY)

    with (
        pytest.raises(ValueError, match="oops"),
        profile_run(output, metadata={"command": "generate"}),
        phase("Transformer", verbose=False),
    ):
        raise ValueError("oops")

    events = _events(output)
    assert [event["event"] for event in events] == ["run_start", "phase_start", "phase_error", "run_error"]
    assert events[2]["error_type"] == "ValueError"
    assert events[3]["error_message"] == "oops"
    assert not any(event["event"] == "run_end" for event in events)


def test_generate_profile_flag_wraps_command_without_generation(tmp_path, monkeypatch) -> None:
    output = tmp_path / "cli-profile.jsonl"
    called = []
    monkeypatch.setattr(perf_profile, "_mlx_memory_snapshot", lambda: _MEMORY)
    monkeypatch.setattr(cli, "_cmd_generate", lambda args: called.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ltx-2-mlx",
            "generate",
            "--prompt",
            "test prompt",
            "--output",
            "out.mp4",
            "--frame-rate",
            "24",
            "--one-stage",
            "--seed",
            "7",
            "--quiet",
            "--profile-json",
            str(output),
        ],
    )

    cli.main()

    assert len(called) == 1
    events = _events(output)
    assert [event["event"] for event in events] == ["run_start", "run_end"]
    assert events[0]["metadata"]["mode"] == "one_stage"
    assert events[0]["metadata"]["seed"] == 7
    assert events[0]["metadata"]["prompt_characters"] == 11
