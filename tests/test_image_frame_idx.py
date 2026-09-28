"""``--image`` FRAME_IDX given as ``last`` or a negative number, resolved against the clip length."""

import argparse

import pytest

from ltx_pipelines_mlx.utils.args import ImageAction, ImageConditioningInput, resolve_frame_indices


def _parse(*values: str) -> list[ImageConditioningInput]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action=ImageAction, nargs="+", dest="images", default=None)
    return parser.parse_args(["--image", *values]).images


def test_last_and_end_parse_to_minus_one() -> None:
    assert _parse("a.png", "last", "1.0")[0].frame_idx == -1
    assert _parse("a.png", "END", "0.5")[0].frame_idx == -1


def test_numeric_indices_parse_unchanged() -> None:
    assert _parse("a.png", "-3", "1.0")[0].frame_idx == -3
    assert _parse("a.png", "24", "1.0")[0].frame_idx == 24
    assert _parse("a.png")[0].frame_idx == 0


def test_resolve_counts_back_from_the_end() -> None:
    images = [
        ImageConditioningInput("a.png", 0, 1.0),
        ImageConditioningInput("b.png", -1, 1.0),
        ImageConditioningInput("c.png", -9, 0.5),
        ImageConditioningInput("d.png", 48, 1.0),
    ]
    resolved = resolve_frame_indices(images, 97)
    assert [image.frame_idx for image in resolved] == [0, 96, 88, 48]
    assert resolved[2].path == "c.png"
    assert resolved[2].strength == 0.5


def test_resolve_rejects_an_index_before_the_first_frame() -> None:
    with pytest.raises(ValueError, match="before the first of 25 frames"):
        resolve_frame_indices([ImageConditioningInput("a.png", -26, 1.0)], 25)
