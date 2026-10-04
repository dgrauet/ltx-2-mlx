"""Skip helpers for the HDR export tools (OpenEXR extra, an ffmpeg built with libx265)."""

from __future__ import annotations

import functools
import subprocess

import pytest

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg


@functools.cache
def ffmpeg_has_libx265() -> bool:
    """True when the ffmpeg on PATH lists a ``libx265`` encoder."""
    try:
        out = subprocess.run(
            [find_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
        ).stdout
    except (RuntimeError, OSError):
        return False
    return any(len(f) > 1 and f[1] == "libx265" for f in (line.split() for line in out.splitlines()))


def openexr_available() -> bool:
    """True when the optional ``OpenEXR`` package imports."""
    try:
        import OpenEXR  # noqa: F401  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    return True


needs_libx265 = pytest.mark.skipif(not ffmpeg_has_libx265(), reason="ffmpeg has no libx265 encoder")
needs_openexr = pytest.mark.skipif(not openexr_available(), reason="OpenEXR not installed (the hdr extra)")
