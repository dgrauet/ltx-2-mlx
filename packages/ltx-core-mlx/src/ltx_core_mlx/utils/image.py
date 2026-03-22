"""Image and video preparation utilities for VAE encoding."""

from __future__ import annotations

import math
import subprocess

import mlx.core as mx
import numpy as np
from PIL import Image

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg


def prepare_image_for_encoding(
    image: Image.Image | str,
    height: int,
    width: int,
) -> mx.array:
    """Load and prepare an image for VAE encoding.

    Resizes to (height, width), normalizes to [-1, 1], returns as (1, 3, H, W).

    Args:
        image: PIL Image or path to image file.
        height: Target height.
        width: Target width.

    Returns:
        mx.array of shape (1, 3, H, W) in [-1, 1] range, bfloat16.
    """
    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGB")
    # Aspect-ratio-preserving resize + center crop (matches reference)
    src_w, src_h = image.size
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # Center crop to target size
    crop_left = (new_w - width) // 2
    crop_top = (new_h - height) // 2
    image = image.crop((crop_left, crop_top, crop_left + width, crop_top + height))

    # HWC uint8 -> float32 -> [-1, 1]
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0

    # HWC -> CHW -> BCHW
    tensor = mx.array(arr).transpose(2, 0, 1)[None, ...]
    return tensor.astype(mx.bfloat16)


def load_video_frames(
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> mx.array:
    """Load video frames via ffmpeg as a tensor for VAE encoding.

    Args:
        video_path: Path to the video file.
        height: Frame height in pixels.
        width: Frame width in pixels.
        num_frames: Number of frames to read.

    Returns:
        Video tensor of shape (1, 3, F, H, W) in [-1, 1] range, bfloat16.

    Raises:
        RuntimeError: If ffmpeg fails to read the video.
    """
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-i",
        video_path,
        "-vframes",
        str(num_frames),
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to read video: {result.stderr.decode()}")

    raw = result.stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    # Normalize to [-1, 1]
    frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
    # FHWC -> BCFHW: (F, H, W, 3) -> (3, F, H, W) -> (1, 3, F, H, W)
    tensor = mx.array(frames).transpose(3, 0, 1, 2)[None, ...]
    return tensor.astype(mx.bfloat16)
