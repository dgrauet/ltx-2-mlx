"""#75: macOS GPU-watchdog kills must die with an actionable hint.

On macOS 26.x + MLX 0.31.x the Metal watchdog kills sustained GPU work while
the display is active (upstream ml-explore/mlx#3267): the run dies ~20 s in
with a cryptic ``Impacting Interactivity`` runtime error and no clue about
the known mitigation. The CLI must recognize that specific failure and print
the ``AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`` / display-off guidance — WITHOUT
ever setting driver env vars on the user's behalf (decision on #75).
"""

from __future__ import annotations

import pytest

from ltx_pipelines_mlx.utils.watchdog import watchdog_hint

WATCHDOG_MSG = (
    "[METAL] Command buffer execution failed: Impacting Interactivity "
    "(0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"
)


class TestWatchdogHint:
    def test_matches_full_metal_message(self):
        hint = watchdog_hint(RuntimeError(WATCHDOG_MSG))
        assert hint is not None
        assert "AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1" in hint
        assert "3267" in hint, "must point at the upstream MLX issue"
        assert "display" in hint.lower(), "display-off is the only complete workaround"

    def test_matches_error_code_alone(self):
        exc = RuntimeError("kIOGPUCommandBufferCallbackErrorImpactingInteractivity")
        assert watchdog_hint(exc) is not None

    def test_ignores_unrelated_errors(self):
        assert watchdog_hint(RuntimeError("out of memory")) is None
        assert watchdog_hint(ValueError("Impacting nothing")) is None

    def test_never_mutates_environment(self, monkeypatch):
        """The hint is advice only — no driver env var may be set (decision #75)."""
        import os

        monkeypatch.delenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", raising=False)
        watchdog_hint(RuntimeError(WATCHDOG_MSG))
        assert "AGX_RELAX_CDM_CTXSTORE_TIMEOUT" not in os.environ


class TestCliWiring:
    def test_watchdog_error_exits_1_with_hint(self, monkeypatch, capsys):
        """A watchdog kill in any subcommand prints the hint and exits 1."""
        import ltx_pipelines_mlx.cli as cli

        def _boom(args):
            raise RuntimeError(WATCHDOG_MSG)

        monkeypatch.setattr(cli, "_cmd_info", _boom)
        monkeypatch.setattr("sys.argv", ["ltx-2-mlx", "info"])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "Impacting Interactivity" in err, "original error must stay visible"
        assert "AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1" in err

    def test_unrelated_errors_propagate_unchanged(self, monkeypatch):
        """Non-watchdog failures must not be swallowed or rewrapped."""
        import ltx_pipelines_mlx.cli as cli

        def _boom(args):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(cli, "_cmd_info", _boom)
        monkeypatch.setattr("sys.argv", ["ltx-2-mlx", "info"])

        with pytest.raises(RuntimeError, match="out of memory"):
            cli.main()
