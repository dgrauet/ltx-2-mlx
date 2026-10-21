"""``--image`` replace / guide bindings (upstream v1.5.0 ``ImageReplace`` / ``ImageGuide``)."""

from __future__ import annotations

import argparse

import mlx.core as mx
import pytest

from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, VideoConditionByLatentIndex
from ltx_pipelines_mlx.dfr import _refuse_replace_under_temporal_rounds, rebase_image_conditionings
from ltx_pipelines_mlx.extend_distilled import _as_guide
from ltx_pipelines_mlx.utils import _orchestration as orch
from ltx_pipelines_mlx.utils import media_io
from ltx_pipelines_mlx.utils.args import (
    ImageAction,
    ImageConditioningInput,
    ImageGuide,
    ImageReplace,
    resolve_frame_indices,
)
from ltx_pipelines_mlx.utils.types import replacement_latent_index


def _parse(*values: str) -> ImageConditioningInput:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action=ImageAction, nargs="+", dest="images", default=None)
    return parser.parse_args(["--image", *values]).images[0]


# --- types -------------------------------------------------------------------------------


def test_at_replaces_frame_zero_and_guides_later_frames() -> None:
    assert ImageConditioningInput.at("a.png", 0, 1.0).binding == ImageReplace(0)
    assert ImageConditioningInput.at("a.png", 9, 1.0).binding == ImageGuide(9)
    assert ImageConditioningInput.at("a.png", -1, 1.0).binding == ImageGuide(-1)


@pytest.mark.parametrize("frame_idx", [0, 1, 9, 17, 97])
def test_replace_accepts_latent_starts(frame_idx: int) -> None:
    assert ImageConditioningInput.replace("a.png", frame_idx, 1.0).frame_idx == frame_idx


@pytest.mark.parametrize("frame_idx", [2, 8, 10, 96])
def test_replace_rejects_frames_inside_a_latent(frame_idx: int) -> None:
    with pytest.raises(ValueError, match=rf"ImageReplace frame_idx must be 0 or 1 \+ 8 \* k, got {frame_idx}"):
        ImageReplace(frame_idx)


def test_binding_must_be_a_binding() -> None:
    with pytest.raises(TypeError, match="binding must be ImageReplace or ImageGuide, got int"):
        ImageConditioningInput("a.png", 1.0, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(("frame_idx", "latent_idx"), [(0, 0), (1, 1), (9, 2), (17, 3), (89, 12)])
def test_replacement_latent_index(frame_idx: int, latent_idx: int) -> None:
    """Latent 0 covers pixel frame 0 alone, so pixel frame ``1 + 8k`` is latent ``k + 1``."""
    assert replacement_latent_index(frame_idx) == latent_idx


def test_map_frame_index_keeps_the_binding() -> None:
    guide = ImageConditioningInput.guide("a.png", 24, 0.5, 18)
    assert guide.map_frame_index(lambda f: f - 24) == ImageConditioningInput.guide("a.png", 0, 0.5, 18)
    assert guide.map_frame_index(lambda f: None) is None
    replace = ImageConditioningInput.replace("a.png", 9, 1.0)
    assert replace.map_frame_index(lambda f: f + 8).binding == ImageReplace(17)
    with pytest.raises(ValueError, match="ImageReplace frame_idx"):
        replace.map_frame_index(lambda f: f * 2)


# --- parser ------------------------------------------------------------------------------


def test_three_tokens_are_auto() -> None:
    assert _parse("a.png", "0", "1.0") == ImageConditioningInput.replace("a.png", 0, 1.0)
    assert _parse("a.png", "48", "0.7") == ImageConditioningInput.guide("a.png", 48, 0.7)


def test_fourth_token_is_a_binding_keyword_or_a_crf() -> None:
    assert _parse("a.png", "0", "1.0", "guide") == ImageConditioningInput.guide("a.png", 0, 1.0)
    assert _parse("a.png", "9", "1.0", "replace") == ImageConditioningInput.replace("a.png", 9, 1.0)
    assert _parse("a.png", "48", "1.0", "auto") == ImageConditioningInput.guide("a.png", 48, 1.0)
    assert _parse("a.png", "48", "1.0", "0") == ImageConditioningInput.guide("a.png", 48, 1.0, 0)


def test_five_tokens_are_binding_then_crf() -> None:
    assert _parse("a.png", "17", "1.0", "replace", "18") == ImageConditioningInput.replace("a.png", 17, 1.0, 18)


def test_five_tokens_refuse_crf_before_binding(capsys) -> None:
    with pytest.raises(SystemExit):
        _parse("a.png", "0", "1.0", "18", "guide")
    assert "BINDING must be auto, replace, or guide, got '18'" in capsys.readouterr().err


def test_replace_inside_a_latent_fails_at_parse_time(capsys) -> None:
    with pytest.raises(SystemExit):
        _parse("a.png", "8", "1.0", "replace")
    assert "ImageReplace frame_idx must be 0 or 1 + 8 * k, got 8" in capsys.readouterr().err


def test_two_tokens_are_refused(capsys) -> None:
    with pytest.raises(SystemExit):
        _parse("a.png", "0")
    assert "requires 1 or 3 to 5 arguments" in capsys.readouterr().err


def test_path_alone_is_the_legacy_first_frame_replace() -> None:
    assert _parse("a.png") == ImageConditioningInput.replace("a.png", 0, 1.0)


def test_last_takes_a_binding() -> None:
    assert _parse("a.png", "last", "1.0", "guide").binding == ImageGuide(-1)
    assert _parse("a.png", "last", "1.0").binding == ImageGuide(-1)


# --- negative indices --------------------------------------------------------------------


def test_negative_replace_is_checked_once_resolved() -> None:
    """``num_frames - n`` starts a latent iff ``n % 8 == 0`` (``num_frames`` is ``8m + 1``)."""
    assert resolve_frame_indices([_parse("a.png", "-8", "1.0", "replace")], 97)[0].binding == ImageReplace(89)
    with pytest.raises(ValueError, match=r"--image a\.png: ImageReplace frame_idx must be 0 or 1 \+ 8 \* k, got 96"):
        resolve_frame_indices([_parse("a.png", "last", "1.0", "replace")], 97)


def test_auto_minus_num_frames_is_a_guide_at_frame_zero() -> None:
    """Upstream ``.at`` decides from the literal index, so ``-num_frames`` guides frame 0."""
    assert resolve_frame_indices([_parse("a.png", "-25", "1.0")], 25)[0].binding == ImageGuide(0)


# --- lowering ----------------------------------------------------------------------------


class _Encoder:
    def encode(self, x):
        return mx.ones((1, 128, 1, x.shape[3] // 32, x.shape[4] // 32))


def _lower(images: list[ImageConditioningInput]) -> list:
    return orch.combined_image_conditionings(
        images, enc_h=64, enc_w=96, spatial_dims=(4, 2, 3), video_encoder=_Encoder(), frame_rate=24.0
    )


def test_lowering_dispatches_on_the_binding(monkeypatch) -> None:
    monkeypatch.setattr(media_io, "load_image_and_preprocess", lambda path, h, w, crf=None: mx.zeros((1, 3, h, w)))
    conds = _lower(
        [
            ImageConditioningInput.replace("a.png", 0, 1.0, 0),
            ImageConditioningInput.replace("b.png", 9, 0.8, 0),
            ImageConditioningInput.guide("c.png", 0, 1.0, 0),
            ImageConditioningInput.guide("d.png", 20, 1.0, 0),
        ]
    )
    assert [type(c) for c in conds] == [
        VideoConditionByLatentIndex,
        VideoConditionByLatentIndex,
        VideoConditionByKeyframeIndex,
        VideoConditionByKeyframeIndex,
    ]
    assert conds[0].frame_indices == [0]
    assert conds[1].frame_indices == [2]
    assert conds[2].frame_idx == 0
    assert conds[3].frame_idx == 20


def test_replace_at_nine_pins_the_third_latent_frame(monkeypatch) -> None:
    monkeypatch.setattr(media_io, "load_image_and_preprocess", lambda path, h, w, crf=None: mx.zeros((1, 3, h, w)))
    (cond,) = _lower([ImageConditioningInput.replace("a.png", 9, 1.0, 0)])
    tokens_per_frame = 2 * 3
    state = LatentState(
        latent=mx.zeros((1, 4 * tokens_per_frame, 128)),
        clean_latent=mx.zeros((1, 4 * tokens_per_frame, 128)),
        denoise_mask=mx.ones((1, 4 * tokens_per_frame, 1)),
    )
    out = cond.apply(state, (4, 2, 3))
    mask = out.denoise_mask[0, :, 0].tolist()
    assert mask == [1.0] * 12 + [0.0] * 6 + [1.0] * 6
    assert out.latent.shape == state.latent.shape


# --- DFR / extend ------------------------------------------------------------------------


def test_dfr_rebase_keeps_a_guide_on_tile_local_frame_zero() -> None:
    """Upstream v1.5.0 rebases through ``map_frame_index``; v1.4.2 turned this guide into a replace."""
    rebased = rebase_image_conditionings(
        [ImageConditioningInput.guide("a.png", 24, 1.0)], pixel_scale=2, pixel_start=48, pixel_end=96
    )
    assert rebased == [ImageConditioningInput.guide("a.png", 0, 1.0)]


def test_dfr_temporal_rounds_refuse_a_replace_off_frame_zero() -> None:
    _refuse_replace_under_temporal_rounds([ImageConditioningInput.at("a.png", 0, 1.0)])
    _refuse_replace_under_temporal_rounds([ImageConditioningInput.guide("a.png", 9, 1.0)])
    with pytest.raises(ValueError, match="replace binding off frame 0 is not supported with DFR temporal rounds"):
        _refuse_replace_under_temporal_rounds([ImageConditioningInput.replace("a.png", 9, 1.0)])


def test_extend_anchors_are_guides() -> None:
    assert _as_guide(ImageConditioningInput.replace("a.png", 0, 0.7, 18)) == ImageConditioningInput.guide(
        "a.png", 0, 0.7, 18
    )


def test_extend_refuses_a_replace_off_frame_zero() -> None:
    """The window's latent grid starts on the carry, so a replace would silently become a guide."""
    with pytest.raises(ValueError, match="extend --distilled takes every anchor as a guide"):
        _as_guide(ImageConditioningInput.replace("a.png", 9, 1.0))
    with pytest.raises(ValueError, match="got frame -8"):
        _as_guide(_parse("a.png", "-8", "1.0", "replace"))


def test_dfr_guard_resolves_negative_replace_indices() -> None:
    """``-num_frames`` is frame 0; with the length still to be predicted a negative index waits."""
    _refuse_replace_under_temporal_rounds([_parse("a.png", "-97", "1.0", "replace")], 97)
    _refuse_replace_under_temporal_rounds([_parse("a.png", "-96", "1.0", "replace")], None)
    with pytest.raises(ValueError, match="replace binding off frame 0"):
        _refuse_replace_under_temporal_rounds([_parse("a.png", "-96", "1.0", "replace")], 97)


def test_refine_audio_is_refused_by_pipelines_without_it() -> None:
    """Upstream defines ``refine_audio`` on ``TI2VidTwoStagesPipeline`` only."""
    from ltx_pipelines_mlx.distilled import DistilledPipeline
    from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

    for cls in (TI2VidTwoStagesHQPipeline, DistilledPipeline):
        pipe = cls.__new__(cls)
        with pytest.raises(ValueError, match=f"refine_audio is not supported by {cls.__name__}"):
            pipe.generate_and_save(prompt="p", output_path="out.mp4", frame_rate=24.0, refine_audio=True)
