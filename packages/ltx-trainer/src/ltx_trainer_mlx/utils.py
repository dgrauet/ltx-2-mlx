"""Image I/O utilities for training.

Ported from ltx-trainer (Lightricks). Uses PIL + numpy for image handling,
matching the patterns in ``ltx_core_mlx/utils/image.py``.
"""

from __future__ import annotations

import io
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import ExifTags, Image, ImageCms, ImageOps
from PIL.Image import Image as PilImage


def open_image_as_srgb(image_path: str | Path | io.BytesIO) -> PilImage:
    """Open an image file, apply rotation and convert to sRGB colour space.

    Applies EXIF-based rotation and respects embedded ICC profiles when
    converting to sRGB.

    Args:
        image_path: Path to the image file or a ``BytesIO`` buffer.

    Returns:
        PIL Image in sRGB colour space.
    """
    exif_colorspace_srgb = 1

    with Image.open(image_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw)

    input_icc_profile = img.info.get("icc_profile")

    srgb_profile = ImageCms.createProfile(colorSpace="sRGB")
    if input_icc_profile is not None:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc_profile))
        srgb_img = ImageCms.profileToProfile(img, input_profile, srgb_profile, outputMode="RGB")
    else:
        exif_data = img.getexif()
        if exif_data is not None:
            color_space_value = exif_data.get(ExifTags.Base.ColorSpace.value)
            if color_space_value is not None and color_space_value != exif_colorspace_srgb:
                raise ValueError(
                    "Image has colorspace tag in EXIF but it isn't set to sRGB,"
                    " conversion is not supported."
                    f" EXIF ColorSpace tag value is {color_space_value}",
                )

        srgb_img = img.convert("RGB")

        srgb_profile_data = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
        srgb_img.info["icc_profile"] = srgb_profile_data

    return srgb_img


def save_image(image_array: mx.array, output_path: Path | str) -> None:
    """Save an image array to a file.

    Args:
        image_array: Image as ``mx.array`` of shape ``(C, H, W)`` or
            ``(C, 1, H, W)`` in range ``[0, 1]`` or ``[0, 255]``.
            ``C`` must be 3 (RGB).
        output_path: Path to save the image (any PIL-supported format).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy for PIL
    arr = np.array(image_array, copy=False)

    # Handle [C, 1, H, W] format (single frame from video tensor)
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        else:
            raise ValueError(f"Expected single-frame tensor with shape [C, 1, H, W], got shape {arr.shape}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C, H, W], got {arr.ndim}D tensor")

    if arr.shape[0] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {arr.shape[0]} channels")

    # Normalise to [0, 255] uint8
    if arr.dtype in (np.float32, np.float64, np.float16) and arr.max() <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # [C, H, W] -> [H, W, C]
    arr = arr.transpose(1, 2, 0)

    Image.fromarray(arr).save(output_path)
