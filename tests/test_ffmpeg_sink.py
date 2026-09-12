"""The ffmpeg sink behind ``VideoDecoder.decode_and_stream`` neither deadlocks nor fails silently (#92).

ffmpeg's stderr used to be piped and never read: past the OS pipe buffer
(64 KB on macOS) ffmpeg blocks on stderr, stops draining stdin, and
``proc.wait()`` never returns. Its exit status was also never checked.
These tests drive ``_ffmpeg_sink`` with shell stand-ins for ffmpeg.
"""

from __future__ import annotations

import stat
import threading
from pathlib import Path

import pytest

from ltx_core_mlx.model.video_vae.video_vae import _ffmpeg_sink

# Well past the 64 KB pipe buffer that a never-read stderr would fill.
_VERBOSE_STDERR_BYTES = 256 * 1024


def _fake_ffmpeg(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake_ffmpeg.sh"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _run_with_deadline(fn, seconds: float = 10.0) -> None:
    """Run ``fn`` on a daemon thread; fail (instead of hanging the suite) if it does not return in time."""
    errors: list[BaseException] = []

    def target() -> None:
        try:
            fn()
        except BaseException as e:  # surfaced on the test thread below
            errors.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"ffmpeg sink still blocked after {seconds}s: stderr deadlock"
    if errors:
        raise errors[0]


def test_ffmpeg_sink_does_not_deadlock_on_verbose_stderr(tmp_path) -> None:
    ffmpeg = _fake_ffmpeg(
        tmp_path,
        f"head -c {_VERBOSE_STDERR_BYTES} /dev/zero | tr '\\0' 'w' >&2\ncat > /dev/null\nexit 0\n",
    )

    def body() -> None:
        with _ffmpeg_sink([ffmpeg]) as proc:
            proc.stdin.write(b"x" * 1000)

    _run_with_deadline(body)


def test_ffmpeg_sink_raises_with_stderr_on_nonzero_exit(tmp_path) -> None:
    ffmpeg = _fake_ffmpeg(tmp_path, "cat > /dev/null\necho 'Unknown encoder libx264' >&2\nexit 3\n")

    with (
        pytest.raises(RuntimeError, match=r"(?s)exit(ed)?.*3.*Unknown encoder libx264"),
        _ffmpeg_sink([ffmpeg]) as proc,
    ):
        proc.stdin.write(b"x" * 1000)


def test_ffmpeg_sink_body_exception_wins_over_exit_status(tmp_path) -> None:
    """An error raised while streaming is the root cause; ffmpeg's resulting non-zero exit must not mask it."""
    ffmpeg = _fake_ffmpeg(tmp_path, "cat > /dev/null\nexit 3\n")

    with pytest.raises(ValueError, match="upstream failure"), _ffmpeg_sink([ffmpeg]):
        raise ValueError("upstream failure")


def test_ffmpeg_sink_reaps_process_on_success(tmp_path) -> None:
    ffmpeg = _fake_ffmpeg(tmp_path, "cat > /dev/null\nexit 0\n")

    with _ffmpeg_sink([ffmpeg]) as proc:
        proc.stdin.write(b"x" * 1000)

    assert proc.returncode == 0
    assert proc.stdin.closed
    # stderr is redirected to a file, never a pipe we own and never read.
    assert proc.stderr is None
