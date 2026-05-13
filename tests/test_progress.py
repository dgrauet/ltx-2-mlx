"""Smoke tests for the CLI phase marker helper."""

from __future__ import annotations

from ltx_pipelines_mlx.utils.progress import phase


def test_phase_verbose_prints_enter_and_exit(capsys) -> None:
    with phase("test", verbose=True):
        pass

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[test] ..." in captured.err
    assert "[test] done in" in captured.err


def test_phase_quiet_prints_nothing(capsys) -> None:
    with phase("test", verbose=False):
        pass

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_phase_prints_exit_even_on_exception(capsys) -> None:
    try:
        with phase("boom", verbose=True):
            raise ValueError("oops")
    except ValueError:
        pass

    captured = capsys.readouterr()
    assert "[boom] ..." in captured.err
    assert "[boom] done in" in captured.err
