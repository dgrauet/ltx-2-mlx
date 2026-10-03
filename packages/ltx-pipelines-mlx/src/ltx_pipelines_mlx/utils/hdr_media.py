"""HDR media I/O for the SDR->HDR IC-LoRA (upstream ``ltx_pipelines/utils/media_io``).

ffmpeg subprocess instead of PyAV, OpenEXR (optional ``hdr`` extra) instead of OpenImageIO.
"""

from __future__ import annotations

import contextlib
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from ltx_core_mlx.color.hlg import linear_to_hlg_signal
from ltx_core_mlx.color.primaries import Primaries
from ltx_core_mlx.color.yuv import rgb_to_yuv420p10_bt2020_limited
from ltx_core_mlx.utils.ffmpeg import find_ffmpeg


def _x265_params(width: int, height: int) -> str:
    """Upstream ``_x265_encode_params`` (the tiny-clip workaround included)."""
    base = "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:repeat-headers=1:info=0"
    if width <= 32 and height <= 32:
        return f"{base}:frame-threads=1:bframes=0:lookahead=0"
    return f"{base}:frame-threads=4"


class HlgFfmpegWriter:
    """Stream scene-linear Rec.709 frames into a BT.2020/HLG 10-bit HEVC mp4 (upstream ``encode_linear_hdr_frames_to_hlg_mp4``).

    Frames are converted to HLG signal and packed to planar yuv420p10le here; ffmpeg only encodes.
    """

    def __init__(
        self, output_path: str, width: int, height: int, fps: float, *, crf: int = 12, preset: str = "ultrafast"
    ) -> None:
        if width % 2 or height % 2:
            raise ValueError(f"HLG 4:2:0 output needs even dimensions, got {width}x{height}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.width, self.height = width, height
        frac = Fraction(fps).limit_denominator(1000)
        fps_str = f"{frac.numerator}/{frac.denominator}"
        self._cmd = [
            find_ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p10le",
            "-s",
            f"{width}x{height}",
            "-r",
            fps_str,
            "-i",
            "-",
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p10le",
            "-tag:v",
            "hvc1",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "arib-std-b67",
            "-colorspace",
            "bt2020nc",
            "-color_range",
            "tv",
            "-x265-params",
            _x265_params(width, height),
            "-movflags",
            "+faststart",
            output_path,
        ]
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> HlgFfmpegWriter:
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write(self, frames: np.ndarray) -> None:
        """Encode ``(F, H, W, 3)`` scene-linear Rec.709 float frames."""
        assert self._proc is not None and self._proc.stdin is not None
        if frames.shape[1:] != (self.height, self.width, 3):
            raise ValueError(f"expected (F, {self.height}, {self.width}, 3), got {frames.shape}")
        try:
            for frame in frames:
                y, u, v = rgb_to_yuv420p10_bt2020_limited(linear_to_hlg_signal(frame, Primaries.REC709))
                self._proc.stdin.write(y.tobytes() + u.tobytes() + v.tobytes())
        except BrokenPipeError as broken_pipe_err:
            # ffmpeg died mid-stream; reap it and raise with its stderr
            assert self._proc.stdin is not None
            with contextlib.suppress(BrokenPipeError, OSError):
                self._proc.stdin.close()
            err = self._proc.stderr.read().decode() if self._proc.stderr else ""
            self._proc.wait()
            raise RuntimeError(f"ffmpeg HLG encode failed: {err}") from broken_pipe_err

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            with contextlib.suppress(BrokenPipeError, OSError):
                self._proc.stdin.close()
            err = self._proc.stderr.read().decode() if self._proc.stderr else ""
            code = self._proc.wait()
        finally:
            # Always unlink on any failure
            if exc_type is not None or code != 0:
                Path(self.output_path).unlink(missing_ok=True)
            # Raise RuntimeError only if no exception is in flight
            if exc_type is None and code != 0:
                raise RuntimeError(f"ffmpeg HLG encode failed ({code}): {err}")
