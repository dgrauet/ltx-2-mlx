"""HDR media I/O for the SDR->HDR IC-LoRA (upstream ``ltx_pipelines/utils/media_io``).

ffmpeg subprocess instead of PyAV, OpenEXR (optional ``hdr`` extra) instead of OpenImageIO.
"""

from __future__ import annotations

import contextlib
import enum
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from ltx_core_mlx.color.hlg import linear_to_hlg_signal
from ltx_core_mlx.color.primaries import Primaries
from ltx_core_mlx.color.yuv import rgb_to_yuv420p10_bt2020_limited
from ltx_core_mlx.hdr import srgb_eotf_to_linear, to_acescct_working_space, to_hdr_linear
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
        # None = ffmpeg never reaped (e.g. KeyboardInterrupt while draining stderr): a failure
        code: int | None = None
        err = ""
        try:
            with contextlib.suppress(BrokenPipeError, OSError):
                self._proc.stdin.close()
            err = self._proc.stderr.read().decode() if self._proc.stderr else ""
            code = self._proc.wait()
        finally:
            # Always unlink on any failure
            if exc_type is not None or code != 0:
                Path(self.output_path).unlink(missing_ok=True)
            # Raise RuntimeError only if no exception is in flight (code None = the drain itself raised)
            if exc_type is None and code is not None and code != 0:
                raise RuntimeError(f"ffmpeg HLG encode failed ({code}): {err}")


class EXRColorSpace(enum.Enum):
    """Colour space of an EXR plate (upstream ``media_io.color_config.EXRColorSpace``)."""

    SRGB_LINEAR = "srgb_linear"
    ACESCG = "acescg"
    ACESCCT = "acescct"

    @property
    def is_log_working(self) -> bool:
        """Whether the plate already holds ACEScct log codes."""
        return self is EXRColorSpace.ACESCCT

    @property
    def source_primaries(self) -> Primaries:
        """Primaries of the plate."""
        return Primaries.REC709 if self is EXRColorSpace.SRGB_LINEAR else Primaries.AP1

    @property
    def exr_tag(self) -> str:
        """Value of the EXR ``colorSpace`` attribute."""
        return {"srgb_linear": "sRGB", "acescg": "ACEScg", "acescct": "ACEScct"}[self.value]


@dataclass(frozen=True)
class VideoInput:
    """MP4/MOV source; ``gamma_encoded`` = display-referred sRGB (EOTF applied on load)."""

    path: Path
    gamma_encoded: bool


@dataclass(frozen=True)
class EXRVideoInput:
    """EXR-frame folder with a declared colour space and playback fps."""

    dir: Path
    color_space: EXRColorSpace
    frame_rate: float


def align_resolution(width: int, height: int, divisor: int = 32) -> tuple[int, int, int, int]:
    """Reflect-pad alignment (upstream ``align_resolution``): round up, crop back after decode.

    Returns:
        ``(gen_w, gen_h, crop_w, crop_h)``.
    """
    gen_w = ((width + divisor - 1) // divisor) * divisor
    gen_h = ((height + divisor - 1) // divisor) * divisor
    return gen_w, gen_h, width, height


def resize_and_reflect_pad(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """``(H, W, 3)`` -> ``(height, width, 3)`` (upstream ``resize_and_reflect_pad``).

    The bilinear down-scale (sources larger than the target) uses PIL per-channel float
    bilinear, close to but not bit-identical with torch ``interpolate``; the pipeline always
    pads up, so that branch is kept for API parity only.
    """
    src_h, src_w = frame.shape[:2]
    if not (height >= src_h and width >= src_w):
        from PIL import Image

        scale = min(height / src_h, width / src_w)
        new_h, new_w = round(src_h * scale), round(src_w * scale)
        frame = np.stack(
            [
                np.asarray(Image.fromarray(frame[..., c]).resize((new_w, new_h), Image.Resampling.BILINEAR))
                for c in range(3)
            ],
            axis=-1,
        ).astype(np.float32)
        src_h, src_w = new_h, new_w
    pad_h, pad_w = height - src_h, width - src_w
    if pad_h or pad_w:
        mode = "reflect" if pad_h < src_h and pad_w < src_w else "edge"
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return frame.astype(np.float32)


def _decode_rgb24_frames(path: Path, frame_cap: int) -> Iterator[np.ndarray]:
    """Yield ``(H, W, 3)`` uint8 frames from a video via ffmpeg (no scaling, no auto-rotation).

    ``-noautorotate`` keeps the coded ``W x H`` that ffprobe reports (upstream PyAV does not
    rotate either); without it a rotated phone MOV would come out ``H x W`` with the same byte
    count and be silently scrambled by the reshape.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero status.
    """
    import tempfile

    from ltx_core_mlx.utils.ffmpeg import probe_video_info

    info = probe_video_info(str(path))
    w, h = info.width, info.height
    # stderr to a temp file: a PIPE nobody drains could fill up and stall the decode
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(_rgb24_decode_cmd(path, frame_cap), stdout=subprocess.PIPE, stderr=err_file)
        assert proc.stdout is not None
        size = w * h * 3
        completed = False
        try:
            while (buf := proc.stdout.read(size)) and len(buf) == size:
                yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            completed = True
        finally:
            proc.stdout.close()
            code = proc.wait()
        # Only on a full read: an early-closed generator kills ffmpeg's pipe on purpose
        if completed and code != 0:
            err_file.seek(0)
            raise RuntimeError(f"ffmpeg decode of '{path}' failed ({code}): {err_file.read().decode(errors='replace')}")


def _rgb24_decode_cmd(path: Path, frame_cap: int) -> list[str]:
    """ffmpeg command decoding ``path`` to raw rgb24 on stdout, in coded orientation."""
    return [
        find_ffmpeg(),
        "-loglevel",
        "error",
        "-noautorotate",
        "-i",
        str(path),
        "-frames:v",
        str(frame_cap),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def load_video_as_hdr_conditioning(
    path: str | Path, height: int, width: int, frame_cap: int, *, gamma_encoded: bool
) -> Iterator[np.ndarray]:
    """SDR MP4/MOV -> ACEScct VAE-range frames ``(height, width, 3)`` (upstream ``load_video_as_hdr_conditioning``).

    Order matches upstream: ``/255`` -> optional sRGB EOTF -> Rec.709->AP1 -> ACEScct -> reflect pad -> ``*2-1``.
    """
    for frame in _decode_rgb24_frames(Path(path), frame_cap):
        ldr = frame.astype(np.float32) / 255.0
        linear = srgb_eotf_to_linear(ldr) if gamma_encoded else ldr
        working = to_acescct_working_space(linear, Primaries.REC709)
        yield resize_and_reflect_pad(working, height, width) * 2.0 - 1.0


def _openexr():
    try:
        import OpenEXR  # ty: ignore[unresolved-import]
    except ImportError as err:
        raise ImportError("EXR I/O needs the optional extra: uv sync --extra hdr (OpenEXR>=3.3)") from err
    return OpenEXR


def read_exr(path: str | Path) -> np.ndarray:
    """Read an EXR frame as float32 ``(H, W, 3)`` RGB (alpha dropped, mono broadcast).

    Raises:
        ValueError: If the channels are neither RGB/RGBA nor a recognisable mono channel.
    """
    oe = _openexr()
    with oe.File(str(path)) as f:
        channels = f.channels()
        if "RGB" in channels:
            rgb = channels["RGB"].pixels
        elif "RGBA" in channels:
            rgb = channels["RGBA"].pixels[..., :3]
        else:
            names = list(channels)
            if not names:
                raise RuntimeError(f"EXR '{path}' has no channels")
            if "Y" in channels:
                mono = channels["Y"].pixels
            elif len(names) == 1 and channels[names[0]].pixels.ndim == 2:
                mono = channels[names[0]].pixels
            else:
                raise ValueError(
                    f"EXR '{path}' has channels {sorted(names)}; expected RGB, RGBA or a single mono channel (Y)"
                )
            rgb = np.repeat(mono[..., None], 3, axis=-1)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def save_exr_frame(rgb: np.ndarray, path: str | Path, primaries: Primaries, color_space_tag: str) -> None:
    """Write ``(H, W, 3)`` as half-float ZIP EXR with ``chromaticities`` + ``colorSpace`` tags."""
    oe = _openexr()
    header = {
        "compression": oe.ZIP_COMPRESSION,
        "type": oe.scanlineimage,
        "chromaticities": primaries.exr_chromaticities,
        "colorSpace": color_space_tag,
    }
    with oe.File(header, {"RGB": np.ascontiguousarray(rgb, dtype=np.float16)}) as f:
        f.write(str(path))


def load_exr_as_hdr_conditioning(
    directory: str | Path, height: int, width: int, frame_cap: int, *, color_space: EXRColorSpace
) -> Iterator[np.ndarray]:
    """EXR folder -> ACEScct VAE-range frames (upstream ``load_exr_as_hdr_conditioning``)."""
    files = sorted(Path(directory).glob("*.exr"))[:frame_cap]
    if not files:
        raise RuntimeError(f"No EXR frames found in {directory}")
    for fp in files:
        frame = resize_and_reflect_pad(read_exr(fp), height, width)
        working = (
            np.clip(frame, 0.0, 1.0)
            if color_space.is_log_working
            else to_acescct_working_space(frame, color_space.source_primaries)
        )
        yield working * 2.0 - 1.0


def encode_hdr_outputs(chunks: Iterable[np.ndarray], output_path: str, fps: float, color_space: EXRColorSpace) -> Path:
    """ACEScct ``[0, 1]`` ``(F, H, W, 3)`` chunks -> EXR sequence + HLG master (upstream ``_encode_hdr_video_outputs``).

    The HLG master is always Rec.709 scene-linear -> BT.2020/HLG; the EXR frames are log codes for
    ``ACESCCT``, else scene-linear in the colour space's primaries.

    Returns:
        The EXR directory.
    """
    out = Path(output_path)
    exr_dir = out.parent / f"{out.stem}_{color_space.value}_exr"
    exr_dir.mkdir(parents=True, exist_ok=True)
    writer: HlgFfmpegWriter | None = None
    index = 0
    with contextlib.ExitStack() as stack:
        for chunk in chunks:
            hlg_linear = to_hdr_linear(chunk, Primaries.REC709)
            if color_space.is_log_working:
                exr = chunk
            elif color_space.source_primaries is Primaries.REC709:
                exr = hlg_linear
            else:
                exr = to_hdr_linear(chunk, color_space.source_primaries)
            for frame in exr:
                save_exr_frame(
                    frame, exr_dir / f"frame_{index:05d}.exr", color_space.source_primaries, color_space.exr_tag
                )
                index += 1
            if writer is None:
                writer = stack.enter_context(HlgFfmpegWriter(output_path, chunk.shape[2], chunk.shape[1], fps))
            writer.write(hlg_linear)
        if index == 0:
            raise ValueError("No HDR frames to encode.")
    return exr_dir
