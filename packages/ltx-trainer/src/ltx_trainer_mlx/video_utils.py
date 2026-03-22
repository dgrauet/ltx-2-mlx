"""Video I/O utilities using ffmpeg subprocess.

Ported from ltx-trainer (Lightricks). Replaces PyAV with ffmpeg subprocess
calls, matching the patterns in ``ltx_core_mlx/utils/ffmpeg.py`` and
``ltx_core_mlx/utils/video.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg, probe_video_info


def get_video_frame_count(video_path: str | Path) -> int:
    """Get the number of frames in a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Number of frames in the video.
    """
    info = probe_video_info(str(video_path))
    return info.num_frames


def read_video(video_path: str | Path, max_frames: int | None = None) -> tuple[mx.array, float]:
    """Load frames from a video file using ffmpeg.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to read. If ``None``, reads all frames.

    Returns:
        Tuple of (video, fps) where video is an ``mx.array`` with shape
        ``(F, C, H, W)`` in ``[0, 1]`` range and fps is the frame rate.
    """
    video_path = str(video_path)
    info = probe_video_info(video_path)
    fps = info.fps
    height, width = info.height, info.width

    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-i",
        video_path,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
    ]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.append("pipe:1")

    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decoding failed: {result.stderr.decode()}")

    raw = result.stdout
    frame_bytes = height * width * 3
    num_frames = len(raw) // frame_bytes
    if num_frames == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    # (F, H, W, 3) uint8 -> float32 [0, 1]
    arr = np.frombuffer(raw[: num_frames * frame_bytes], dtype=np.uint8)
    arr = arr.reshape(num_frames, height, width, 3).astype(np.float32) / 255.0

    # (F, H, W, 3) -> (F, 3, H, W)
    video = mx.array(arr).transpose(0, 3, 1, 2)
    return video, fps


def save_video(
    video_array: mx.array,
    output_path: Path | str,
    fps: float = 24.0,
    audio: mx.array | None = None,
    audio_sample_rate: int | None = None,
) -> None:
    """Save a video array to a file using ffmpeg, optionally with audio.

    Args:
        video_array: Video as ``mx.array`` of shape ``(C, F, H, W)`` or
            ``(F, C, H, W)`` in ``[0, 1]`` or ``[0, 255]`` range.
        output_path: Path to save the video.
        fps: Frames per second for the output video.
        audio: Optional audio as ``mx.array`` of shape ``(C, samples)`` or
            ``(samples, C)`` in ``[-1, 1]`` range.
        audio_sample_rate: Sample rate for the audio (required if audio
            is provided).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    video_np = _prepare_video_array(video_array)
    num_frames, height, width, _ = video_np.shape

    ffmpeg = find_ffmpeg()

    if audio is not None and audio_sample_rate is not None:
        # Mux video + audio
        _save_video_with_audio(ffmpeg, video_np, output_path, fps, audio, audio_sample_rate)
    else:
        # Video only
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.communicate(input=video_np.tobytes())
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg encoding failed")


def _prepare_video_array(video_array: mx.array) -> np.ndarray:
    """Convert video array to ``(F, H, W, C)`` uint8 numpy array."""
    arr = np.array(video_array, copy=False)

    # Handle [C, F, H, W] vs [F, C, H, W] format
    if arr.shape[0] == 3 and arr.shape[1] > 3:
        arr = arr.transpose(1, 0, 2, 3)  # [C, F, H, W] -> [F, C, H, W]

    # Normalise to [0, 255] uint8
    if arr.dtype in (np.float32, np.float64, np.float16) and arr.max() <= 1.0:
        arr = arr * 255.0

    # [F, C, H, W] -> [F, H, W, C]
    arr = arr.transpose(0, 2, 3, 1)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_video_with_audio(
    ffmpeg: str,
    video_np: np.ndarray,
    output_path: Path,
    fps: float,
    audio: mx.array,
    sample_rate: int,
) -> None:
    """Save video with audio using ffmpeg with dual pipe inputs."""
    import tempfile

    num_frames, height, width, _ = video_np.shape

    # Prepare audio as WAV-compatible PCM data
    audio_np = np.array(audio, copy=False).astype(np.float32)

    # Normalise to [samples, 2] stereo
    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np], axis=1)
    elif audio_np.shape[0] == 2 and audio_np.shape[1] != 2:
        audio_np = audio_np.T  # [2, samples] -> [samples, 2]
    if audio_np.shape[1] == 1:
        audio_np = np.concatenate([audio_np, audio_np], axis=1)

    # Convert to int16
    audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)

    # Write audio to temp file (ffmpeg needs seekable audio input for muxing)
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp.write(audio_int16.tobytes())
        audio_tmp_path = tmp.name

    try:
        cmd = [
            ffmpeg,
            "-y",
            # Video input from pipe
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            # Audio input from file
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-i",
            audio_tmp_path,
            # Output
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.communicate(input=video_np.tobytes())
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg encoding with audio failed")
    finally:
        Path(audio_tmp_path).unlink(missing_ok=True)
