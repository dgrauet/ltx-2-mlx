"""Stepwise previews: decode intermediate latents during the denoising loop.

A generation is opaque until the final VAE decode, which for a long clip can be
many minutes away. This module decodes a short *contiguous window* of latent
frames from the x0 prediction every N steps and writes it as a self-contained
animated WebP — a couple of seconds of real motion at the current denoise
quality, so temporal problems (flicker, incoherent movement, a subject that
never resolves) are visible early instead of after the full clip finishes.

Why a window rather than a single frame: the VAE upsamples time 8x, so
``F_lat`` latent frames decode to ``8 * F_lat - 7`` pixel frames. A single
latent frame yields exactly one picture while paying for the whole up-block
stack — the worst point on the cost curve. Eight latent frames cost roughly
26x a single frame but yield 57, which at 25 fps is 2.3 seconds of motion.

Each step's chunk is written once and never rewritten. Callers that want the
whole progression play the chunks back to back, which keeps write cost linear
in the step count instead of quadratic.

The loops in :mod:`ltx_pipelines_mlx.utils.samplers` never see this class — they
take a plain ``on_step`` callable, which :meth:`StepwisePreview.bind` produces
with the latent geometry and decoder already closed over.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from PIL import Image

if TYPE_CHECKING:
    from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier
    from ltx_pipelines_mlx.utils.blocks import VideoDecoder
    from ltx_pipelines_mlx.utils.samplers import OnStepFn

logger = logging.getLogger(__name__)

# Latent frames decoded per preview. 8 -> 57 pixel frames -> ~2.3 s at 25 fps,
# which is long enough to actually watch. Cost is linear in this beyond the
# first frame, and independent of the clip's total length.
DEFAULT_PREVIEW_FRAMES = 8

# Lossy, but still full 24-bit colour (unlike GIF's 256-entry palette). Previews
# are for judging motion and composition, not final chroma.
ANIMATION_QUALITY = 90


@dataclass(frozen=True)
class StepwiseConfig:
    """User-facing knobs for stepwise previews."""

    output_dir: Path
    interval: int = 1
    frame: int | None = None  # centre of the window; None = middle, negative = from end
    frames: int = DEFAULT_PREVIEW_FRAMES
    frame_rate: float = 24.0
    seed: int = 0


def resolve_window(num_latent_frames: int, centre: int | None, count: int) -> tuple[int, int]:
    """Resolve a contiguous latent-frame window, clamped inside the clip.

    Centres on the middle by default: frame 0 is uninformative for
    image-conditioned generation, since it is the clean conditioning image and
    never changes.

    Returns:
        ``(start, count)`` — count is reduced when the clip is shorter than the
        requested window.
    """
    count = max(1, min(count, num_latent_frames))
    if centre is None:
        middle = num_latent_frames // 2
    elif centre < 0:
        middle = centre + num_latent_frames
    else:
        middle = centre
    start = middle - count // 2
    start = max(0, min(start, num_latent_frames - count))
    return start, count


def to_pil_images(pixels: mx.array) -> list[Image.Image]:
    """Convert a decoded ``(B, 3, T, H, W)`` tensor in ``[-1, 1]`` to PIL images.

    Mirrors the frame conversion in ``video_vae.VideoDecoder.decode_and_stream``.
    """
    images: list[Image.Image] = []
    for index in range(pixels.shape[2]):
        frame = mx.clip(pixels[:, :, index], -1.0, 1.0)
        frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
        frame_hwc = frame[0].transpose(1, 2, 0)  # (H, W, 3)
        mx.eval(frame_hwc)  # required: memoryview races GPU writes without this sync
        height, width = frame_hwc.shape[0], frame_hwc.shape[1]
        images.append(Image.frombytes("RGB", (width, height), bytes(memoryview(frame_hwc))))
        del frame, frame_hwc
    return images


def write_animation(frames: list[Image.Image], path: Path, *, frame_rate: float) -> None:
    """Write one preview chunk atomically, played back at the clip's own frame rate.

    Writes to a temp file in the same directory and ``os.replace``s it into
    place. A reader polling the directory during a run must never open a
    half-written file.
    """
    duration_ms = max(10, round(1000.0 / max(1.0, frame_rate)))
    tmp = path.parent / f"{path.name}.tmp"
    frames[0].save(
        tmp,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=ANIMATION_QUALITY,
        method=4,
    )
    os.replace(tmp, path)


class StepwisePreview:
    """Writes one self-contained motion clip per previewed denoising step.

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
    ) -> OnStepFn | None:
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
                stage's chunks separate. ``None`` for single-loop pipelines,
                which omit the tag entirely.
        """
        if self._disabled:
            return None

        start, count = resolve_window(latent_frames, self.config.frame, self.config.frames)
        tag = "" if stage is None else f"_s{stage}"
        prefix = f"seed_{self.config.seed}{tag}"

        def on_step(step_idx: int, num_steps: int, video_x0: mx.array, sigma: float) -> None:
            if self._disabled:
                return
            # Always preview the last step so the sequence ends on the result.
            if step_idx % self.config.interval != 0 and step_idx != num_steps - 1:
                return
            try:
                images = self._decode_window(
                    video_x0,
                    latent_frames=latent_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    start=start,
                    count=count,
                    decoder_block=decoder_block,
                    patchifier=patchifier,
                )
                path = self.config.output_dir / f"{prefix}_step{step_idx + 1:03d}of{num_steps:03d}.webp"
                write_animation(images, path, frame_rate=self.config.frame_rate)
                del images
                self._announce(path.parent, count)
            except Exception as exc:
                self._disable(exc)

        return on_step

    def _decode_window(
        self,
        video_x0: mx.array,
        *,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        start: int,
        count: int,
        decoder_block: VideoDecoder,
        patchifier: VideoLatentPatchifier,
    ) -> list[Image.Image]:
        """Decode a contiguous latent-frame window of an x0 prediction."""
        # Strip appended keyframe tokens (multi-anchor conditioning appends
        # tokens past the F*H*W grid, which unpatchify cannot reshape).
        grid = (latent_frames, latent_height, latent_width)
        tokens = video_x0[:, : latent_frames * latent_height * latent_width, :]
        latent = patchifier.unpatchify(tokens, grid)
        # res2s works in float32; the decoder expects the model dtype.
        window = latent[:, :, start : start + count].astype(mx.bfloat16)
        del tokens, latent

        # decode() denormalizes internally, so the loop's latent goes in as-is.
        # The temporal axis is shape-generic: count frames in, 8*count-7 out.
        pixels = decoder_block.load().decode(window)
        images = to_pil_images(pixels)
        del window, pixels
        return images

    def _announce(self, directory: Path, count: int) -> None:
        if self._announced:
            return
        self._announced = True
        if self.verbose:
            print(
                f"[stepwise] previews ({8 * count - 7} frames/step) -> {directory}",
                file=sys.stderr,
                flush=True,
            )

    def _disable(self, exc: Exception) -> None:
        self._disabled = True
        logger.warning("stepwise preview failed, disabling for the rest of this run: %s", exc, exc_info=True)
        print(f"[stepwise] preview failed ({exc}); previews disabled for this run", file=sys.stderr, flush=True)
