"""CLI argument helpers — mirrors upstream ``ltx_pipelines.utils.args``.

Currently exposes the multi-image I2V conditioning parser used by the
text-to-video pipelines (``ImageAction``) and re-exports
``ImageConditioningInput``, which lives in ``utils.types`` as upstream.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    """Place ``last`` anchors and check every index against the clip length.

    Mirrors upstream ``assert_image_frames_in_clip`` (v1.6.2): a guide may sit anywhere,
    including before frame 0 (``-1`` is one frame before the clip) or past the last frame,
    because its index only sets the keyframe's position in time; a ``replace`` must lie
    inside the clip. On top of that, the local ``last`` marker
    (:attr:`ImageConditioningInput.from_end`) becomes ``num_frames - 1``.

    Until 0.18 a negative index counted back from the end (``-1`` was the last frame).
    Use ``last`` for that.

    Args:
        images: Conditioning inputs as parsed from ``--image``.
        num_frames: Concrete pixel-frame count of the clip being generated.

    Returns:
        The inputs with every ``last`` anchor placed (``from_end`` cleared).

    Raises:
        ValueError: If a ``replace`` lies outside ``[0, num_frames)``, or if a ``last``
            anchor asks for a ``replace`` on a frame that does not start a latent.
    """
    resolved = []
    for image in images:
        if image.from_end:
            try:
                resolved.append(replace(image.with_frame_idx(num_frames - 1), from_end=False))
            except ValueError as e:
                raise ValueError(f"--image {image.path}: {e}") from e
        elif image.replaces and image.frame_idx >= num_frames:
            raise ValueError(
                f"--image {image.path}: replace frame index {image.frame_idx} is outside the clip "
                f"[0, {num_frames}); only a guide may sit outside it"
            )
        else:
            resolved.append(image)
    return resolved


_IMAGE_BINDING_KEYWORDS = frozenset({"auto", "replace", "guide"})


def image_conditioning_input(
    path: str, frame_idx: int | None, strength: float, binding_mode: str = "auto", crf: int | None = None
) -> ImageConditioningInput:
    """Build an :class:`ImageConditioningInput` from a ``BINDING`` keyword (upstream ``ImageAction``'s ``match``).

    Args:
        path: Image path.
        frame_idx: Pixel frame index; a guide may sit before the clip (negative) or after it.
            ``None`` anchors the last frame, whatever the clip length (``last`` on the CLI).
        strength: Conditioning strength.
        binding_mode: ``auto`` (replace on frame 0, guide elsewhere), ``replace`` or ``guide``.
        crf: H.264 CRF of the re-compression, ``None`` to resolve it from the checkpoint.

    Returns:
        The conditioning input with its binding.

    Raises:
        ValueError: on an unknown keyword, or a ``replace`` off the latent grid (frame 0 or ``1 + 8k``).
    """
    if binding_mode not in _IMAGE_BINDING_KEYWORDS:
        raise ValueError(f"BINDING must be auto, replace, or guide, got {binding_mode!r}")
    if frame_idx is None:
        return ImageConditioningInput.last(path, strength, crf, replaces=binding_mode == "replace")
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
    may also be ``last`` (or ``end``): ``--image end.png last 1.0`` anchors the
    final frame even when ``--auto-duration`` picks the length. A negative
    number is upstream's (v1.6.2): a guide that many frames before the clip.
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
            frame_idx = None if str(values[1]).lower() in ("last", "end") else int(values[1])
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
