"""A2VidPipelineTwoStage stage 1: TeaCache wiring and the LTX-2.5 guard.

``--enable-teacache`` is added to every generation subcommand by ``_add_generation_args``,
but the a2v pipeline used to accept it and never hand a controller to
``guided_denoise_loop``. These drive ``generate_and_save`` over a synthetic pack, with
the heavy stages stubbed, and stop inside the stage-1 loop.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_arsenal.diffusion import TeaCacheController

from ltx_pipelines_mlx import a2vid_two_stage as a2v_mod
from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage


class _StopAtStage1Error(Exception):
    """Raised by the stubbed stage-1 loop once it has recorded its kwargs."""


def _make_a2v(tmp_path, *, ltx25: bool) -> A2VidPipelineTwoStage:
    transformer: dict = {"num_layers": 48}
    if ltx25:
        transformer["ff_bias"] = False
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))
    return A2VidPipelineTwoStage(model_dir=str(tmp_path))


@pytest.fixture
def stubbed_a2v(tmp_path, monkeypatch):
    """A 2.3-pack a2v pipeline that reaches the stage-1 loop without loading weights."""
    pipe = _make_a2v(tmp_path, ltx25=False)
    calls: list[dict] = []

    def fake_loop(**kwargs):
        calls.append(kwargs)
        raise _StopAtStage1Error

    monkeypatch.setattr(pipe, "_load_audio_encoder", lambda: None)
    pipe.audio_encoder = object()
    pipe.audio_processor = object()
    monkeypatch.setattr(
        a2v_mod, "load_audio", lambda *a, **k: SimpleNamespace(waveform=mx.zeros((1, 16000)), sample_rate=16000)
    )
    monkeypatch.setattr(a2v_mod, "encode_audio", lambda *a, **k: mx.zeros((1, 8, 64, 16), dtype=mx.bfloat16))
    embeds = (
        mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
        mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
    )
    monkeypatch.setattr(pipe, "_encode_text_with_negative", lambda *a, **k: embeds)
    monkeypatch.setattr(pipe, "_load_dev_transformer", lambda: object())
    monkeypatch.setattr(a2v_mod, "guided_denoise_loop", fake_loop)

    def run(**kwargs):
        with pytest.raises(_StopAtStage1Error):
            pipe.generate_and_save(
                prompt="a singer",
                output_path=str(tmp_path / "out.mp4"),
                audio_path="song.wav",
                height=128,
                width=128,
                num_frames=9,
                frame_rate=24.0,
                stage1_steps=4,
                **kwargs,
            )
        assert len(calls) == 1
        return calls[0]

    return pipe, run


def test_teacache_controller_reaches_the_stage1_loop(stubbed_a2v, monkeypatch):
    pipe, run = stubbed_a2v
    built: list[TeaCacheController | None] = []
    original = pipe._make_stage1_teacache

    def spy(*args):
        built.append(original(*args))
        return built[-1]

    monkeypatch.setattr(pipe, "_make_stage1_teacache", spy)

    call = run(enable_teacache=True, teacache_thresh=0.7)

    assert isinstance(built[0], TeaCacheController)
    assert call["teacache"] is built[0]


def test_teacache_off_by_default(stubbed_a2v):
    _, run = stubbed_a2v

    assert run()["teacache"] is None


def test_teacache_rejected_on_25_pack_before_any_load(tmp_path, monkeypatch):
    pipe = _make_a2v(tmp_path, ltx25=True)

    def must_not_load():
        raise AssertionError("the guard must fire before the audio encoder loads")

    monkeypatch.setattr(pipe, "_load_audio_encoder", must_not_load)
    with pytest.raises(ValueError, match="TeaCache"):
        pipe.generate_and_save(
            prompt="a singer",
            output_path=str(tmp_path / "out.mp4"),
            audio_path="song.wav",
            num_frames=9,
            frame_rate=24.0,
            enable_teacache=True,
        )


def test_cli_a2v_forwards_the_teacache_flags(monkeypatch, tmp_path):
    """``a2v`` must accept both flags and hand them to ``generate_and_save``."""
    from ltx_pipelines_mlx import cli

    received: dict = {}

    class _FakePipe:
        def __init__(self, **kwargs) -> None:
            pass

        def generate_and_save(self, **kwargs) -> str:
            received.update(kwargs)
            return kwargs["output_path"]

    monkeypatch.setattr(a2v_mod, "A2VidPipelineTwoStage", _FakePipe)
    monkeypatch.setattr(cli, "_print_result", lambda *a, **k: None)
    args = cli._build_parser().parse_args(
        [
            "a2v",
            "-p",
            "a singer",
            "--audio",
            "song.wav",
            "--frame-rate",
            "24",
            "-o",
            str(tmp_path / "out.mp4"),
            "--enable-teacache",
            "--teacache-thresh",
            "0.7",
            "--quiet",
        ]
    )
    cli._cmd_a2v(args)

    assert received["enable_teacache"] is True
    assert received["teacache_thresh"] == 0.7
