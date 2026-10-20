"""a2v --dfr: the distilled A2V audio hooks on the DFR base path (no weights)."""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from ltx_pipelines_mlx import a2vid_dfr as a2vdfr_mod
from ltx_pipelines_mlx.a2vid_dfr import A2VidDFRPipeline, fit_audio_tokens
from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline
from ltx_pipelines_mlx.dfr import DEFAULT_DETAILING_LORA, DFRPipeline


def _pack(tmp_path) -> str:
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    return str(tmp_path)


def test_method_resolution_mixes_the_two_pipelines():
    assert A2VidDFRPipeline.generate_two_stage is DFRPipeline.generate_two_stage
    assert A2VidDFRPipeline._stage1_audio_state is A2VidDistilledPipeline._stage1_audio_state
    assert A2VidDFRPipeline._stage2_audio_state is A2VidDistilledPipeline._stage2_audio_state


def test_fit_audio_tokens_pads_trims_and_keeps():
    tokens = mx.arange(1, 13, dtype=mx.float32).reshape(1, 6, 2)
    padded = fit_audio_tokens(tokens, 9)
    assert padded.shape == (1, 9, 2)
    assert mx.array_equal(padded[:, :6], tokens).item() and mx.all(padded[:, 6:] == 0).item()
    assert mx.array_equal(fit_audio_tokens(tokens, 4), tokens[:, :4]).item()
    assert fit_audio_tokens(tokens, 6) is tokens


@pytest.mark.parametrize("canvas_tokens", [51, 60])
def test_frozen_audio_spans_the_canvas_and_keeps_the_source(tmp_path, canvas_tokens):
    pipe = A2VidDFRPipeline(model_dir=_pack(tmp_path))
    source = mx.random.normal((1, 51, 128), key=mx.random.key(1)).astype(mx.bfloat16)
    pipe._source_audio_tokens = source
    state = pipe._stage1_audio_state((1, canvas_tokens, 128), mx.zeros((1, canvas_tokens, 1)), (4, 2, 2), seed=3)
    assert state.frozen and state.latent.shape == (1, canvas_tokens, 128)
    assert mx.array_equal(state.latent[:, :51], source).item()
    assert mx.all(state.latent[:, 51:] == 0).item()
    assert not mx.any(state.denoise_mask).item()
    assert pipe._source_audio_tokens is source  # restored after the call
    state2 = pipe._stage2_audio_state(state.latent, mx.zeros((1, canvas_tokens, 1)), (4, 4, 4), seed=3, sigma=0.9)
    assert state2.frozen and mx.array_equal(state2.latent, state.latent).item()


def _a2v_args(tmp_path, *extra: str):
    from ltx_pipelines_mlx import cli

    return cli._build_parser().parse_args(
        ["a2v", "-p", "a singer", "--audio", "song.wav", "--frame-rate", "24", "-o", str(tmp_path / "out.mp4"), "-q"]
        + list(extra)
    )


def test_cli_a2v_dfr_builds_the_dfr_pipeline(monkeypatch, tmp_path):
    from ltx_pipelines_mlx import cli

    received: dict = {}

    class _FakePipe:
        def __init__(self, **kwargs) -> None:
            received["init"] = kwargs

        def generate_and_save(self, **kwargs) -> str:
            received.update(kwargs)
            return kwargs["output_path"]

    monkeypatch.setattr(a2vdfr_mod, "A2VidDFRPipeline", _FakePipe)
    monkeypatch.setattr(cli, "_print_result", lambda *a, **k: None)
    cli._cmd_a2v(_a2v_args(tmp_path, "--dfr", "--detailing-lora", "lora.safetensors", "--low-ram", "-i", "photo.png"))
    assert received["init"]["detailing_lora"] == "lora.safetensors"
    assert received["init"]["low_ram_streaming"] is True
    assert received["audio_path"] == "song.wav"
    assert not {"cfg_scale", "stg_scale", "negative_prompt", "enable_teacache"} & received.keys()


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--dfr", "--distilled"], "not both"),
        (["--detailing-lora", "x.safetensors"], "only applies with a2v --dfr"),
        (["--distilled", "--detailing-lora", "x.safetensors"], "only applies with a2v --dfr"),
        (["--dfr", "--cfg-scale", "3"], "no CFG"),
        (["--dfr", "--stg-scale", "1"], "no CFG"),
        (["--dfr", "--negative-prompt", "bad"], "no CFG"),
        (["--dfr", "--enable-teacache"], "TeaCache"),
    ],
)
def test_cli_a2v_dfr_rejects_flags_before_building_anything(monkeypatch, tmp_path, flags, message):
    from ltx_pipelines_mlx import a2vid_distilled, cli

    def must_not_build(**kwargs):
        raise AssertionError("the flags must be rejected before the pipeline is built")

    monkeypatch.setattr(a2vdfr_mod, "A2VidDFRPipeline", must_not_build)
    monkeypatch.setattr(a2vid_distilled, "A2VidDistilledPipeline", must_not_build)
    with pytest.raises(SystemExit, match=message):
        cli._cmd_a2v(_a2v_args(tmp_path, *flags))


def test_default_detailing_lora_is_dfrs():
    from ltx_pipelines_mlx import cli

    args = cli._build_parser().parse_args(["a2v", "-p", "x", "--audio", "a.wav", "--frame-rate", "24", "-o", "o.mp4"])
    assert args.detailing_lora == DEFAULT_DETAILING_LORA and args.dfr is False


def test_refuses_a_23_pack_before_encoding_the_audio(tmp_path, monkeypatch):
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48}}))
    pipe = A2VidDFRPipeline(model_dir=str(tmp_path), detailing_lora=str(tmp_path / "x.safetensors"))

    def must_not_run(*args, **kwargs):
        raise AssertionError("the source audio must not be encoded")

    monkeypatch.setattr(A2VidDistilledPipeline, "generate_and_save", must_not_run)
    with pytest.raises(ValueError, match="use_keyframes_abs_pos_embedding"):
        pipe.generate_and_save(
            prompt="a singer", output_path=str(tmp_path / "o.mp4"), audio_path="song.wav", frame_rate=24.0
        )
