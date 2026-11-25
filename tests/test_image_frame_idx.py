"""``--image`` FRAME_IDX: the ``last`` marker, and guides outside the clip (upstream v1.6.2)."""

import argparse

import mlx.core as mx
import pytest

from ltx_pipelines_mlx.dfr_layout import resolve_canvas
from ltx_pipelines_mlx.distilled import resolve_stage1_frames
from ltx_pipelines_mlx.utils.args import ImageAction, ImageConditioningInput, resolve_frame_indices
from ltx_pipelines_mlx.utils.types import ImageGuide


def _parse(*values: str) -> list[ImageConditioningInput]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action=ImageAction, nargs="+", dest="images", default=None)
    return parser.parse_args(["--image", *values]).images


def test_last_and_end_parse_to_the_from_end_marker() -> None:
    for word in ("last", "END"):
        image = _parse("a.png", word, "0.5")[0]
        assert image.from_end
        assert isinstance(image.binding, ImageGuide)
        assert image.strength == 0.5


def test_numeric_indices_parse_unchanged_and_are_not_markers() -> None:
    assert _parse("a.png", "-3", "1.0")[0].frame_idx == -3
    assert _parse("a.png", "24", "1.0")[0].frame_idx == 24
    assert _parse("a.png")[0].frame_idx == 0
    assert not any(_parse("a.png", idx, "1.0")[0].from_end for idx in ("-1", "0", "24"))


def test_last_resolves_to_the_final_frame() -> None:
    resolved = resolve_frame_indices([ImageConditioningInput.last("b.png", 0.5, 18)], 97)
    assert resolved == [ImageConditioningInput.guide("b.png", 96, 0.5, 18)]
    assert not resolved[0].from_end


def test_minus_one_is_one_frame_before_the_clip_not_the_last_frame() -> None:
    """Upstream v1.6.2. Until 0.18 the port read ``-1`` as the last frame."""
    resolved = resolve_frame_indices(_parse("look.png", "-1", "1.0"), 97)
    assert resolved[0].binding == ImageGuide(-1)


@pytest.mark.parametrize("frame_idx", [-26, -1, 25, 40])
def test_guides_outside_the_clip_pass_through(frame_idx: int) -> None:
    images = [ImageConditioningInput.at("a.png", frame_idx, 1.0)]
    assert resolve_frame_indices(images, 25) == images


def test_in_clip_indices_pass_through() -> None:
    images = [ImageConditioningInput.at("a.png", 0, 1.0), ImageConditioningInput.at("d.png", 48, 0.5)]
    assert resolve_frame_indices(images, 97) == images


def test_a_replace_past_the_clip_is_refused() -> None:
    with pytest.raises(ValueError, match=r"a\.png: replace frame index 25 is outside the clip \[0, 25\)"):
        resolve_frame_indices([ImageConditioningInput.replace("a.png", 25, 1.0)], 25)
    assert resolve_frame_indices([ImageConditioningInput.replace("a.png", 17, 1.0)], 25)[0].frame_idx == 17


@pytest.mark.parametrize(("requested", "canvas"), [(137, 145), (41, 49)])
def test_last_lands_on_the_requested_end_when_dfr_pads_the_canvas(requested: int, canvas: int) -> None:
    """``--dfr`` pads the clip to whole keyframe segments and trims the output back afterwards."""
    seen = []

    def canvas_for(resolved_frames: int) -> tuple[int, list[int]]:
        seen.append(resolved_frames)
        canvas_frames, _segment, positions = resolve_canvas(resolved_frames)
        return canvas_frames, positions

    images = [ImageConditioningInput.at("start.png", 0, 1.0), ImageConditioningInput.last("end.png", 1.0)]
    num_frames, slots, resolved = resolve_stage1_frames(requested, None, images, 0, canvas_for)

    assert seen == [requested]
    assert num_frames == canvas
    assert slots == resolve_canvas(requested)[2]
    assert [image.frame_idx for image in resolved] == [0, requested - 1]


def test_stage1_frames_without_a_canvas_hook() -> None:
    num_frames, slots, resolved = resolve_stage1_frames(97, "a.png", None, 2, None)
    assert (num_frames, slots) == (97, 2)
    assert resolved == [ImageConditioningInput.at("a.png", 0, 1.0)]


@pytest.mark.parametrize("frame_idx", [-1, 97])
def test_out_of_clip_guide_keeps_its_index_as_a_time_offset(frame_idx: int) -> None:
    """The keyframe token sits before time 0 for ``-1`` and past the clip for ``num_frames``; nothing clamps it."""
    from ltx_core_mlx.conditioning.types.keyframe_cond import _compute_keyframe_positions

    times = _compute_keyframe_positions(frame_idx, 2, 2, 24.0)[0, :, 0]
    expected = (frame_idx + 0.5) / 24.0
    assert float(mx.abs(times - expected).max().item()) < 1e-6
