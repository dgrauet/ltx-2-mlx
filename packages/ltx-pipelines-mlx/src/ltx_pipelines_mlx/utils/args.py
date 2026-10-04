"""CLI argument helpers — mirrors upstream ``ltx_pipelines.utils.args``.

Currently exposes the multi-image I2V conditioning input used by the
text-to-video pipelines: ``ImageConditioningInput`` + ``ImageAction``.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

# Re-export for legacy import paths. Single source of truth lives in
# ``ltx_pipelines_mlx.utils.media_io`` (mirrors upstream's
# ``ltx_pipelines.utils.media_io.DEFAULT_IMAGE_CRF``).
from ltx_pipelines_mlx.utils.media_io import DEFAULT_IMAGE_CRF


def _append_to_dest(namespace: argparse.Namespace, dest: str, item: object) -> None:
    """Append ``item`` to the namespace list at ``dest`` (creating it if absent).

    Shared by the repeatable list actions (:class:`ImageAction`,
    :class:`SegmentAction`) so their append-to-dest handling can't drift.
    """
    existing = getattr(namespace, dest, None) or []
    existing.append(item)
    setattr(namespace, dest, existing)


class ImageConditioningInput(NamedTuple):
    """One image conditioning entry for multi-anchor I2V.

    Mirrors upstream ``ltx_pipelines.utils.args.ImageConditioningInput``
    verbatim.

    Args:
        path: Path to the image file.
        frame_idx: Target latent frame index. ``0`` replaces the first
            latent frame (``VideoConditionByLatentIndex``); any other
            value appends a keyframe at that pixel-frame position
            (``VideoConditionByKeyframeIndex``). A negative value counts
            back from the end (``-1`` is the last pixel frame) and is
            resolved by :func:`resolve_frame_indices` once the frame count
            is known, which is what makes an end anchor usable with an
            auto-predicted duration.
        strength: Conditioning strength in ``[0, 1]``. ``1.0`` = fully
            preserved.
        crf: H.264 CRF the image is re-compressed at before encoding.
            ``None`` (the default) means "use what matches the model": the
            pipeline fills it in from the checkpoint it runs
            (``ImageConditioner.resolve_crf``; 33 before LTX-2.4, 18 from
            2.4 on). Pass it explicitly to override, including ``0`` to skip
            re-compression entirely.
    """

    path: str
    frame_idx: int
    strength: float
    crf: int | None = None


def resolve_frame_indices(images: list[ImageConditioningInput], num_frames: int) -> list[ImageConditioningInput]:
    """Turn negative ``frame_idx`` values into pixel-frame positions counted from the end.

    ``-1`` becomes ``num_frames - 1`` (the last pixel frame), ``-2`` the one
    before it, and so on. Non-negative indices pass through unchanged.
    ``-num_frames`` resolves to ``0``, so it gets the hard first-latent replace
    (``VideoConditionByLatentIndex``) rather than a soft keyframe.

    Args:
        images: Conditioning inputs as parsed from ``--image``.
        num_frames: Concrete pixel-frame count of the clip being generated.

    Returns:
        The inputs with every ``frame_idx`` in ``[0, num_frames)``.

    Raises:
        ValueError: If an index lies outside the clip, before its first frame or
            past its last (upstream ``assert_image_frames_in_clip`` checks the
            same range).
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
        resolved.append(image._replace(frame_idx=frame_idx))
    return resolved


class ImageAction(argparse.Action):
    """Variadic argparse action accepting ``PATH [FRAME_IDX [STRENGTH [CRF]]]``.

    Repeatable. Each invocation appends one
    :class:`ImageConditioningInput` to the namespace list.

    Backward-compat: ``--image PATH`` alone is accepted and defaults to
    ``frame_idx=0, strength=1.0`` — matches the prior single-arg API. The
    strict upstream form is ``--image PATH FRAME_IDX STRENGTH [CRF]`` (3 or
    4 args). An omitted CRF stays ``None``, which the pipeline resolves from
    the checkpoint it runs (``ImageConditioner.resolve_crf``), as upstream.

    ``FRAME_IDX`` may also be ``last`` (or ``end``), or a negative number
    counting back from the end: ``--image end.png last 1.0`` anchors the
    final frame even when ``--auto-duration`` picks the length.
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
        if len(values) not in (1, 3, 4):
            parser.error(
                f"{option_string}: expected 1, 3, or 4 values "
                "(PATH [FRAME_IDX STRENGTH [CRF]]), got "
                f"{len(values)}: {values}"
            )

        path = values[0]
        crf: int | None = None
        if len(values) == 1:
            frame_idx, strength = 0, 1.0
        else:
            try:
                frame_idx = -1 if str(values[1]).lower() in ("last", "end") else int(values[1])
                strength = float(values[2])
                if len(values) == 4:
                    crf = int(values[3])
            except (ValueError, TypeError) as e:
                parser.error(f"{option_string}: could not parse FRAME_IDX/STRENGTH/CRF from {values[1:]}: {e}")

        item = ImageConditioningInput(
            path=path,
            frame_idx=frame_idx,
            strength=strength,
            crf=crf,
        )

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
    "ImageConditioningInput",
    "SegmentAction",
    "SegmentInput",
]
