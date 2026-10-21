"""The image and upsample latents are evaluated while their encoder is still loaded (follow-up to #201 / #208).

Left lazy, an ``encoder.encode(...)`` result only runs when the first denoising step reads it, with the encoder still
referenced next to the DiT. ``mx.synchronize()`` does not help: it waits for queued work and evaluates nothing.
"""

from __future__ import annotations

import inspect

import mlx.core as mx

from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_pipelines_mlx import a2vid_two_stage
from ltx_pipelines_mlx.utils import _orchestration as orch
from ltx_pipelines_mlx.utils import media_io
from ltx_pipelines_mlx.utils.args import ImageConditioningInput


class _Encoder:
    """Records every ``encode`` and whether its output was evaluated before the next call."""

    def __init__(self, events: list[str]):
        self.events = events

    def encode(self, x):
        self.events.append("encode")
        return mx.ones((1, 128, 1, x.shape[3] // 32, x.shape[4] // 32))


def test_combined_image_conditionings_evaluates_each_latent_while_the_encoder_is_loaded(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(media_io, "load_image_and_preprocess", lambda path, h, w, crf=None: mx.zeros((1, 3, h, w)))
    real_eval = orch.mx.eval

    def spy_eval(*arrays):
        events.append("eval")
        return real_eval(*arrays)

    monkeypatch.setattr(orch.mx, "eval", spy_eval)
    images = [
        ImageConditioningInput.at("a.png", 0, 1.0, 0),
        ImageConditioningInput.at("b.png", 16, 0.7, 0),
    ]

    conds = orch.combined_image_conditionings(
        images, enc_h=64, enc_w=96, spatial_dims=(3, 2, 3), video_encoder=_Encoder(events), frame_rate=24.0
    )

    assert events == ["encode", "eval", "encode", "eval"]
    assert isinstance(conds[0], VideoConditionByLatentIndex)
    assert isinstance(conds[1], VideoConditionByKeyframeIndex)


def test_a2v_two_stage_has_no_synchronize_standing_in_for_an_eval():
    source = inspect.getsource(a2vid_two_stage)
    code = [line for line in source.splitlines() if not line.strip().startswith("#")]
    assert not any("mx.synchronize()" in line for line in code)
    assert "mx.eval(v_up_renorm)" in source
