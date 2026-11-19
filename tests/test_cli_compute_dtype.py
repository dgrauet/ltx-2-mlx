"""--compute-dtype sets LTX2_COMPUTE_DTYPE before the command runs (and leaves it alone when omitted)."""

from __future__ import annotations

import os
import sys

import pytest

from ltx_pipelines_mlx import cli


def _run(monkeypatch, argv: list[str]) -> str | None:
    seen = {}

    def fake(args):
        seen["env"] = os.environ.get("LTX2_COMPUTE_DTYPE")

    monkeypatch.setattr(cli, "_cmd_generate", fake)
    monkeypatch.setattr(sys, "argv", ["ltx-2-mlx", *argv])
    cli.main()
    return seen["env"]


BASE = ["generate", "--distilled", "-p", "x", "-o", "o.mp4", "-f", "9", "--frame-rate", "24", "--seed", "1"]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_flag_sets_env(monkeypatch, dtype):
    monkeypatch.setenv("LTX2_COMPUTE_DTYPE", "float16")
    assert _run(monkeypatch, [*BASE, "--compute-dtype", dtype]) == dtype


def test_omitted_flag_leaves_env(monkeypatch):
    monkeypatch.delenv("LTX2_COMPUTE_DTYPE", raising=False)
    assert _run(monkeypatch, BASE) is None


def test_rejects_unknown_dtype(monkeypatch):
    with pytest.raises(SystemExit):
        _run(monkeypatch, [*BASE, "--compute-dtype", "int8"])
