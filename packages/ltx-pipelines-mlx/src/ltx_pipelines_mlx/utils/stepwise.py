"""Stepwise previews: decode intermediate latents during the denoising loop.

A generation is opaque until the final VAE decode, which for a long clip can be
many minutes away. This module decodes a *single latent frame* of the x0
prediction every N steps and writes it as a PNG, appending each one to a growing
animated WebP that can be opened mid-run to watch the frame resolve out of noise.

Cost is roughly ``full_decode / F_lat`` per preview: the decoder is shape-generic
on the temporal axis, so decoding one latent frame runs the full up-block stack
over 1/F_lat of the volume. The tradeoff is that the VAE decoder must stay
resident during denoising, which the pipelines otherwise avoid.

The loops in :mod:`ltx_pipelines_mlx.utils.samplers` never see this class — they
take a plain ``on_step`` callable, which :meth:`StepwisePreview.bind` produces
with the latent geometry and decoder already closed over.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from PIL import Image

if TYPE_CHECKING:
    from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier
    from ltx_pipelines_mlx.utils.blocks import VideoDecoder

logger = logging.getLogger(__name__)

# Playback speed of the progress animation, milliseconds per step.
ANIMATION_FRAME_DURATION_MS = 250

# Lossy, but still full 24-bit colour (unlike GIF's 256-entry palette). Lossless
# WebP re-encode of a growing frame list gets slow at 1080p, and the animation is
# rewritten from scratch on every preview. The per-step PNGs are the lossless
# record — judge chroma from those, use the animation to watch progression.
ANIMATION_QUALITY = 90

# Above this many frames, halve the list by dropping every other frame. Bounds
# both host memory (~6 MB/frame at 1080p) and the O(n^2) re-encode cost.
MAX_ANIMATION_FRAMES = 60


@dataclass(frozen=True)
class StepwiseConfig:
    """User-facing knobs for stepwise previews."""

    output_dir: Path
    interval: int = 1
    frame: int | None = None  # latent frame index; None = middle, negative = from end
    seed: int = 0


def resolve_frame_index(num_latent_frames: int, requested: int | None) -> int:
    """Resolve a requested latent frame index, clamped into range.

    Defaults to the middle frame: frame 0 is uninformative for image-conditioned
    generation, since it is the clean conditioning image and never changes.
    """
    if requested is None:
        return num_latent_frames // 2
    if requested < 0:
        requested += num_latent_frames
    return max(0, min(num_latent_frames - 1, requested))


def to_pil_image(pixels: mx.array) -> Image.Image:
    """Convert a decoded ``(B, 3, 1, H, W)`` tensor in ``[-1, 1]`` to a PIL image.

    Mirrors the frame conversion in ``video_vae.VideoDecoder.decode_and_stream``.
    """
    frame = mx.clip(pixels[:, :, 0], -1.0, 1.0)
    frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
    frame_hwc = frame[0].transpose(1, 2, 0)  # (H, W, 3)
    mx.eval(frame_hwc)  # required: memoryview races GPU writes without this sync
    height, width = frame_hwc.shape[0], frame_hwc.shape[1]
    return Image.frombytes("RGB", (width, height), bytes(memoryview(frame_hwc)))


def write_animation(frames: list[Image.Image], path: Path) -> None:
    """Rewrite the progress animation atomically.

    Writes to a temp file in the same directory and ``os.replace``s it into
    place. Being openable mid-run is the whole point of the animation, so a
    reader must never catch a half-written file.
    """
    tmp = path.parent / f"{path.name}.tmp"
    frames[0].save(
        tmp,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=ANIMATION_FRAME_DURATION_MS,
        loop=0,
        quality=ANIMATION_QUALITY,
        method=4,
    )
    os.replace(tmp, path)


class StepwisePreview:
    """Writes per-step preview PNGs plus a growing animation, per denoising stage.

    One instance is shared across a whole generation; :meth:`bind` is called once
    per denoising loop with that loop's latent geometry, and returns the
    ``on_step`` callable the sampler invokes.

    A preview failure never propagates: the first one disables the handler for
    the rest of the run so a broken preview can neither abort a long generation
    nor slow every remaining step.
    """

    def __init__(self, config: StepwiseConfig, *, verbose: bool = True) -> None:
        self.config = config
        self.verbose = verbose
        self._disabled = False
        self._announced = False

    def bind(
        self,
        *,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        decoder_block: VideoDecoder,
        patchifier: VideoLatentPatchifier,
        stage: int | None = None,
    ) -> Callable[[int, int, mx.array, float], None] | None:
        """Return an ``on_step(step_idx, num_steps, video_x0, sigma)`` callable.

        Args:
            latent_frames: Latent grid depth (F) for this loop.
            latent_height: Latent grid height (H) for this loop.
            latent_width: Latent grid width (W) for this loop. Two-stage
                pipelines run their loops at different resolutions, hence
                per-bind geometry.
            decoder_block: The pipeline's VAE decoder block (lazily loaded).
            patchifier: Converts the flat token sequence back to a spatial latent.
            stage: Stage number for multi-stage pipelines, used to keep each
                stage's files (and animation) separate. ``None`` for single-loop
                pipelines, which omit the tag entirely.
        """
        if self._disabled:
            return None

        frame_index = resolve_frame_index(latent_frames, self.config.frame)
        tag = "" if stage is None else f"_s{stage}"
        prefix = f"seed_{self.config.seed}{tag}"
        animation_path = self.config.output_dir / f"{prefix}_progress.webp"
        frames: list[Image.Image] = []

        def on_step(step_idx: int, num_steps: int, video_x0: mx.array, sigma: float) -> None:
            if self._disabled:
                return
            # Always preview the last step so the animation ends on the result.
            if step_idx % self.config.interval != 0 and step_idx != num_steps - 1:
                return
            try:
                image = self._decode_frame(
                    video_x0,
                    latent_frames=latent_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    frame_index=frame_index,
                    decoder_block=decoder_block,
                    patchifier=patchifier,
                )
                png_path = self.config.output_dir / f"{prefix}_step{step_idx + 1:03d}of{num_steps:03d}.png"
                image.save(png_path, format="PNG")

                frames.append(image)
                if len(frames) > MAX_ANIMATION_FRAMES:
                    del frames[1::2]
                write_animation(frames, animation_path)

                self._announce(animation_path)
            except Exception as exc:
                self._disable(exc)

        return on_step

    def _decode_frame(
        self,
        video_x0: mx.array,
        *,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        frame_index: int,
        decoder_block: VideoDecoder,
        patchifier: VideoLatentPatchifier,
    ) -> Image.Image:
        """Decode one latent frame of an x0 prediction into an image."""
        # Strip appended keyframe tokens (multi-anchor conditioning appends
        # tokens past the F*H*W grid, which unpatchify cannot reshape).
        grid = (latent_frames, latent_height, latent_width)
        tokens = video_x0[:, : latent_frames * latent_height * latent_width, :]
        latent = patchifier.unpatchify(tokens, grid)
        # res2s works in float32; the decoder expects the model dtype.
        frame_latent = latent[:, :, frame_index : frame_index + 1].astype(mx.bfloat16)
        del tokens, latent

        # decode() denormalizes internally, so the loop's latent goes in as-is.
        pixels = decoder_block.load().decode(frame_latent)
        image = to_pil_image(pixels)
        del frame_latent, pixels
        return image

    def _announce(self, animation_path: Path) -> None:
        if self._announced:
            return
        self._announced = True
        if self.verbose:
            print(f"[stepwise] previews -> {animation_path}", file=sys.stderr, flush=True)

    def _disable(self, exc: Exception) -> None:
        self._disabled = True
        logger.warning("stepwise preview failed, disabling for the rest of this run: %s", exc, exc_info=True)
        print(f"[stepwise] preview failed ({exc}); previews disabled for this run", file=sys.stderr, flush=True)
