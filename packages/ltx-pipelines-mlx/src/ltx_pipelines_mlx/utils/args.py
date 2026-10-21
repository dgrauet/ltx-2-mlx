"""CLI argument helpers — mirrors upstream ``ltx_pipelines.utils.args``.

Currently exposes the multi-image I2V conditioning parser used by the
text-to-video pipelines (``ImageAction``) and re-exports
``ImageConditioningInput``, which lives in ``utils.types`` as upstream.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

# Re-export for legacy import paths. Single source of truth lives in
# ``ltx_pipelines_mlx.utils.media_io`` (mirrors upstream's
# ``ltx_pipelines.utils.media_io.DEFAULT_IMAGE_CRF``).
from ltx_pipelines_mlx.utils.media_io import DEFAULT_IMAGE_CRF
from ltx_pipelines_mlx.utils.types import (
    ImageBinding,
    ImageConditioningInput,
    ImageGuide,
    ImageReplace,
    replacement_latent_index,
)


def _append_to_dest(namespace: argparse.Namespace, dest: str, item: object) -> None:
    """Append ``item`` to the namespace list at ``dest`` (creating it if absent).

    Shared by the repeatable list actions (:class:`ImageAction`,
    :class:`SegmentAction`) so their append-to-dest handling can't drift.
    """
    existing = getattr(namespace, dest, None) or []
    existing.append(item)
    setattr(namespace, dest, existing)


def resolve_frame_indices(images: list[ImageConditioningInput], num_frames: int) -> list[ImageConditioningInput]:
    """Turn negative ``frame_idx`` values into pixel-frame positions counted from the end.

    ``-1`` becomes ``num_frames - 1`` (the last pixel frame), ``-2`` the one
    before it, and so on. Non-negative indices pass through unchanged. The
    binding is kept: a negative index parsed without a binding (``auto``) is a
    guide, as upstream ``ImageConditioningInput.at`` makes every non-zero index,
    so ``-num_frames`` resolves to a guide at frame 0. A ``replace`` is checked
    against the latent grid once resolved (``0`` or ``1 + 8 * k``).

    Args:
        images: Conditioning inputs as parsed from ``--image``.
        num_frames: Concrete pixel-frame count of the clip being generated.

    Returns:
        The inputs with every ``frame_idx`` in ``[0, num_frames)``.

    Raises:
        ValueError: If an index lies outside the clip, before its first frame or
            past its last (upstream ``assert_image_frames_in_clip`` checks the
            same range), or if a resolved ``replace`` does not start a latent.
    """
    resolved = []
    for image in images:
        frame_idx = num_frames + image.frame_idx if image.frame_idx < 0 else image.frame_idx
        if frame_idx < 0:
            raise ValueError(
                f"--image {image.path}: frame index {image.frame_idx} is before the first of {num_frames} frames"
            )
        if frame_idx >= num_frames:
            raise ValueError(
                f"--image {image.path}: frame index {image.frame_idx} is past the last of {num_frames} frames"
            )
        try:
            resolved.append(image.with_frame_idx(frame_idx))
        except ValueError as e:
            raise ValueError(f"--image {image.path}: {e}") from e
    return resolved


_IMAGE_BINDING_KEYWORDS = frozenset({"auto", "replace", "guide"})


def image_conditioning_input(
    path: str, frame_idx: int, strength: float, binding_mode: str = "auto", crf: int | None = None
) -> ImageConditioningInput:
    """Build an :class:`ImageConditioningInput` from a ``BINDING`` keyword (upstream ``ImageAction``'s ``match``).

    Args:
        path: Image path.
        frame_idx: Pixel frame index (negative counts back from the end, resolved later).
        strength: Conditioning strength.
        binding_mode: ``auto`` (replace on frame 0, guide elsewhere), ``replace`` or ``guide``.
        crf: H.264 CRF of the re-compression, ``None`` to resolve it from the checkpoint.

    Returns:
        The conditioning input with its binding.

    Raises:
        ValueError: on an unknown keyword, or a ``replace`` off the latent grid (frame 0 or ``1 + 8k``).
    """
    match binding_mode:
        case "auto":
            return ImageConditioningInput.at(path, frame_idx, strength, crf)
        case "replace":
            return ImageConditioningInput.replace(path, frame_idx, strength, crf)
        case "guide":
            return ImageConditioningInput.guide(path, frame_idx, strength, crf)
    raise ValueError(f"BINDING must be auto, replace, or guide, got {binding_mode!r}")


class ImageAction(argparse.Action):
    """Variadic argparse action accepting ``PATH [FRAME_IDX STRENGTH [BINDING] [CRF]]``.

    Repeatable. Each invocation appends one
    :class:`ImageConditioningInput` to the namespace list. Mirrors upstream
    ``ltx_pipelines.utils.args.ImageAction`` (v1.5.0): ``BINDING`` is
    ``auto`` (the default when omitted: frame 0 replaces the first latent,
    later frames guide as keyframes), ``replace`` (pins the still on a causal
    latent start, pixel frame 0 or ``1 + 8 * k``) or ``guide`` (a keyframe at
    any pixel frame). With one trailing token, a binding keyword selects
    ``BINDING`` and an integer selects ``CRF``. An omitted CRF stays ``None``,
    which the pipeline resolves from the checkpoint it runs
    (``ImageConditioner.resolve_crf``), as upstream.

    Local extensions: ``--image PATH`` alone is accepted and defaults to
    ``frame_idx=0, strength=1.0`` (the prior single-arg API). ``FRAME_IDX``
    may also be ``last`` (or ``end``), or a negative number counting back from
    the end: ``--image end.png last 1.0`` anchors the final frame even when
    ``--auto-duration`` picks the length.
    """

    def __call__(  # type: ignore[override]
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        if not isinstance(values, list):
            values = [values]
        if len(values) not in (1, 3, 4, 5):
            parser.error(
                f"{option_string} requires 1 or 3 to 5 arguments "
                f"(PATH [FRAME_IDX STRENGTH [BINDING] [CRF]]), got {len(values)}: {values}"
            )

        path = values[0]
        if len(values) == 1:
            _append_to_dest(namespace, self.dest, ImageConditioningInput.at(path, 0, 1.0))
            return
        binding_mode = "auto"
        crf: int | None = None
        if len(values) == 5 and values[3] not in _IMAGE_BINDING_KEYWORDS:
            parser.error(f"{option_string} BINDING must be auto, replace, or guide, got {values[3]!r}")
        try:
            frame_idx = -1 if str(values[1]).lower() in ("last", "end") else int(values[1])
            strength = float(values[2])
            if len(values) == 4:
                if values[3] in _IMAGE_BINDING_KEYWORDS:
                    binding_mode = values[3]
                else:
                    crf = int(values[3])
            elif len(values) == 5:
                binding_mode = values[3]
                crf = int(values[4])
        except (ValueError, TypeError) as e:
            parser.error(f"{option_string}: could not parse FRAME_IDX/STRENGTH/CRF from {values[1:]}: {e}")
        try:
            item = image_conditioning_input(path, frame_idx, strength, binding_mode, crf)
        except ValueError as e:
            parser.error(f"{option_string} {path}: {e}")
        _append_to_dest(namespace, self.dest, item)


class SegmentInput(NamedTuple):
    """One Prompt Relay segment: a local prompt + optional latent-frame length.

    Args:
        text: Local prompt gated to this segment's time window.
        length: Segment length in *latent* frames, or ``None`` to auto-distribute
            evenly across the timeline.
    """

    text: str
    length: int | None = None


class SegmentAction(argparse.Action):
    """Variadic argparse action accepting ``TEXT [LEN_FRAMES]``. Repeatable.

    Each invocation appends one :class:`SegmentInput` (in timeline order) to the
    namespace list. ``--segment "a red car"`` auto-distributes; ``--segment "a red
    car" 4`` pins that segment to 4 latent frames.
    """

    def __call__(  # type: ignore[override]
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        if not isinstance(values, list):
            values = [values]
        if len(values) not in (1, 2):
            parser.error(f"{option_string}: expected TEXT [LEN_FRAMES], got {len(values)}: {values}")

        text = values[0]
        length: int | None = None
        if len(values) == 2:
            try:
                length = int(values[1])
            except (ValueError, TypeError) as e:
                parser.error(f"{option_string}: could not parse LEN_FRAMES from {values[1]!r}: {e}")
            if length <= 0:
                parser.error(f"{option_string}: LEN_FRAMES must be positive, got {length}")

        _append_to_dest(namespace, self.dest, SegmentInput(text=text, length=length))


__all__ = [
    "DEFAULT_IMAGE_CRF",
    "ImageAction",
    "ImageBinding",
    "ImageConditioningInput",
    "ImageGuide",
    "ImageReplace",
    "SegmentAction",
    "SegmentInput",
    "image_conditioning_input",
    "replacement_latent_index",
    "resolve_frame_indices",
]
