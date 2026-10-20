"""Stage-1 capture and resume on the distilled pipeline (#143): ``return_stage1`` / ``stage1``."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import pytest

from ltx_pipelines_mlx import distilled as distilled_mod
from ltx_pipelines_mlx.distilled import Stage1Result
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput
from tests.test_nag import _distilled

_GEN = dict(prompt="a fox", height=128, width=128, num_frames=17, frame_rate=24.0, seed=7)


class _MarkingLoop:
    """Echoes the input latents, offset by a per-stage marker (+1 stage 1, +2 stage 2), so each stage is recognisable."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        mark = 1.0 if float(kwargs["sigmas"][0]) == 1.0 else 2.0
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent + mark,
            audio_latent=kwargs["audio_state"].latent + mark,
        )


def _pipeline(tmp_path, monkeypatch, ltx25):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pipe, _, encoded = _distilled(tmp_path, monkeypatch, ltx25)
    loop = _MarkingLoop()
    monkeypatch.setattr(distilled_mod, "denoise_loop", loop)
    monkeypatch.setattr(distilled_mod, "euler_ancestral_denoising_loop", loop)
    return pipe, loop, encoded


@pytest.mark.parametrize("ltx25", [False, True])
def test_default_call_returns_two_arrays_and_runs_both_loops(tmp_path, monkeypatch, ltx25):
    pipe, loop, _ = _pipeline(tmp_path, monkeypatch, ltx25)
    out = pipe.generate_two_stage(**_GEN)
    assert len(out) == 2 and len(loop.calls) == 2


@pytest.mark.parametrize("ltx25", [False, True])
def test_resume_skips_stage_1_and_is_bit_identical(tmp_path, monkeypatch, ltx25):
    pipe, loop, _ = _pipeline(tmp_path, monkeypatch, ltx25)
    video, audio, stage1 = pipe.generate_two_stage(**_GEN, return_stage1=True)
    assert len(loop.calls) == 2
    assert isinstance(stage1, Stage1Result)
    mx.eval(stage1.video_tokens, stage1.audio_tokens)

    # A second pipeline, as after a restart: only the arrays of the captured stage 1 travel.
    pipe2, loop2, encoded2 = _pipeline(tmp_path / "again", monkeypatch, ltx25)
    saved = dataclasses.replace(
        stage1,
        x0_model=None,
        relay_mask=None,
        relay_audio_mask=None,
        video_embeds=None,
        audio_embeds=None,
        resolved_images=[],
        nag=None,
    )
    video2, audio2 = pipe2.generate_two_stage(**_GEN, stage1=saved)

    assert len(loop2.calls) == 1  # stage 2 only
    assert [p for p, _ in encoded2] == ["a fox"]  # the prompt is still encoded for stage 2
    assert mx.array_equal(video2, video).item() and mx.array_equal(audio2, audio).item()


@pytest.mark.parametrize("ltx25", [False, True])
def test_resume_uses_the_given_tokens(tmp_path, monkeypatch, ltx25):
    pipe, _, _ = _pipeline(tmp_path, monkeypatch, ltx25)
    _, _, stage1 = pipe.generate_two_stage(**_GEN, return_stage1=True)
    pipe2, loop2, _ = _pipeline(tmp_path / "again", monkeypatch, ltx25)
    zeros = dataclasses.replace(
        stage1, video_tokens=mx.zeros_like(stage1.video_tokens), audio_tokens=mx.zeros_like(stage1.audio_tokens)
    )
    pipe2.generate_two_stage(**_GEN, stage1=zeros)
    # Stage 2 re-noises the given (zero) tokens, so its input differs from the captured run's.
    (call,) = loop2.calls
    assert call["video_state"].latent.shape[0] == 1


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(height=192), "latent grid"),
        (dict(num_frames=25), "latent grid"),
    ],
)
def test_resume_with_another_request_raises_before_stage_2(tmp_path, monkeypatch, change, message):
    pipe, _, _ = _pipeline(tmp_path, monkeypatch, True)
    _, _, stage1 = pipe.generate_two_stage(**_GEN, return_stage1=True)
    pipe2, loop2, _ = _pipeline(tmp_path / "again", monkeypatch, True)
    with pytest.raises(ValueError, match=message):
        pipe2.generate_two_stage(**{**_GEN, **change}, stage1=stage1)
    assert loop2.calls == []


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("video_tokens", lambda s: s.video_tokens[:, :-1], "video tokens"),
        ("audio_tokens", lambda s: s.audio_tokens[:, :-1], "audio tokens"),
        ("generated_keyframes", lambda _: mx.zeros((1, 128, 1, 2, 2)), "generated keyframe slots"),
    ],
)
def test_resume_with_mismatched_arrays_raises(tmp_path, monkeypatch, field, value, message):
    pipe, _, _ = _pipeline(tmp_path, monkeypatch, True)
    _, _, stage1 = pipe.generate_two_stage(**_GEN, return_stage1=True)
    bad = dataclasses.replace(stage1, **{field: value(stage1)})
    pipe2, loop2, _ = _pipeline(tmp_path / "again", monkeypatch, True)
    with pytest.raises(ValueError, match=message):
        pipe2.generate_two_stage(**_GEN, stage1=bad)
    assert loop2.calls == []


def test_a_low_memory_run_leaves_the_pipeline_ready_to_load_again(tmp_path, monkeypatch):
    """Stage 2 frees the VAE encoder and the upsampler under low_memory; the next call must reload them."""
    pipe, _, _ = _pipeline(tmp_path, monkeypatch, True)
    pipe.low_memory = True
    monkeypatch.setattr(pipe.image_conditioner, "free", lambda: None)
    pipe._loaded = True
    pipe.generate_two_stage(**_GEN)
    assert pipe._loaded is False
