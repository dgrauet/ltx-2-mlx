"""User-supplied ``negative_prompt`` on the CFG pipelines (no weights needed).

Covers the encode helper semantics (``None`` -> ``DEFAULT_NEGATIVE_PROMPT``,
any string -> verbatim), the forwarding from every public CFG entry point down
to ``_encode_text_with_negative``, the refusal on the CFG-less distilled / DFR
paths, and the ``--negative-prompt`` CLI flag.
"""

from __future__ import annotations

import inspect
import json

import mlx.core as mx
import pytest

from ltx_pipelines_mlx import ti2vid_one_stage as one_stage_mod
from ltx_pipelines_mlx import ti2vid_two_stages as two_stage_mod
from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines_mlx.cli import _build_parser, _cmd_generate
from ltx_pipelines_mlx.dfr import DFRPipeline
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines_mlx.retake import RetakePipeline
from ltx_pipelines_mlx.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines_mlx.utils.constants import DEFAULT_NEGATIVE_PROMPT


class _CapturedError(Exception):
    """Raised by the encode stub once it has recorded its arguments."""


def _pack(tmp_path, *, ltx25: bool = False) -> str:
    """Minimal synthetic pack: enough for the pipeline constructors."""
    transformer: dict = {"num_layers": 48}
    if ltx25:
        transformer["ff_bias"] = False
        transformer["use_keyframes_abs_pos_embedding"] = True
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))
    return str(tmp_path)


def _capture_encode(pipe, monkeypatch) -> dict:
    """Replace ``_encode_text_with_negative`` with a stub that records and aborts."""
    seen: dict = {}

    def stub(prompt, negative_prompt=None):
        seen["prompt"] = prompt
        seen["negative_prompt"] = negative_prompt
        raise _CapturedError

    monkeypatch.setattr(pipe, "_encode_text_with_negative", stub)
    return seen


# --- _encode_text_with_negative ---------------------------------------------------------


@pytest.fixture
def encode_pipe(tmp_path, monkeypatch):
    pipe = TI2VidTwoStagesPipeline(model_dir=_pack(tmp_path))
    pipe.verbose = False
    encoded: list[str] = []

    def fake_encode(text):
        encoded.append(text)
        return mx.zeros((1, 4, 8)), mx.zeros((1, 4, 4))

    monkeypatch.setattr(pipe, "_load_text_encoder", lambda: None)
    monkeypatch.setattr(pipe, "_encode_text", fake_encode)
    pipe.encoded = encoded
    return pipe


def test_encode_none_uses_default_negative(encode_pipe):
    out = encode_pipe._encode_text_with_negative("a fox")
    assert encode_pipe.encoded == ["a fox", DEFAULT_NEGATIVE_PROMPT]
    assert len(out) == 4


def test_encode_custom_negative_is_verbatim(encode_pipe):
    encode_pipe._encode_text_with_negative("a fox", "blurry, text overlay")
    assert encode_pipe.encoded == ["a fox", "blurry, text overlay"]


def test_encode_empty_negative_is_not_replaced_by_default(encode_pipe):
    encode_pipe._encode_text_with_negative("a fox", "")
    assert encode_pipe.encoded == ["a fox", ""]


# --- public signatures ------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        TI2VidOneStagePipeline.generate_one_stage_dev,
        TI2VidOneStagePipeline.generate_and_save,
        TI2VidTwoStagesPipeline.generate_two_stage,
        TI2VidTwoStagesPipeline.generate_and_save,
        TI2VidTwoStagesHQPipeline.generate_two_stage,
        A2VidPipelineTwoStage.generate_and_save,
        RetakePipeline.retake,
        RetakePipeline.extend,
        RetakePipeline.retake_from_video,
        RetakePipeline.extend_from_video,
        KeyframeInterpolationPipeline.interpolate,
        KeyframeInterpolationPipeline.generate_and_save,
    ],
    ids=lambda m: m.__qualname__,
)
def test_cfg_entry_points_expose_negative_prompt(method):
    param = inspect.signature(method).parameters["negative_prompt"]
    assert param.default is None
    assert param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)


# --- forwarding to the encoder ----------------------------------------------------------


@pytest.mark.parametrize("negative", [None, "", "watermark"])
@pytest.mark.parametrize(
    ("cls", "method"),
    [
        (TI2VidTwoStagesPipeline, "generate_two_stage"),
        (TI2VidTwoStagesHQPipeline, "generate_two_stage"),
        (TI2VidOneStagePipeline, "generate_one_stage_dev"),
    ],
)
def test_t2v_generate_forwards_negative(tmp_path, monkeypatch, cls, method, negative):
    pipe = cls(model_dir=_pack(tmp_path))
    seen = _capture_encode(pipe, monkeypatch)
    with pytest.raises(_CapturedError):
        getattr(pipe, method)(prompt="a fox", num_frames=9, frame_rate=24.0, negative_prompt=negative)
    assert seen == {"prompt": "a fox", "negative_prompt": negative}


@pytest.mark.parametrize("cls", [TI2VidTwoStagesPipeline, TI2VidTwoStagesHQPipeline, TI2VidOneStagePipeline])
def test_generate_and_save_forwards_negative(tmp_path, monkeypatch, cls):
    pipe = cls(model_dir=_pack(tmp_path))
    seen = _capture_encode(pipe, monkeypatch)
    with pytest.raises(_CapturedError):
        pipe.generate_and_save(prompt="a fox", output_path="x.mp4", num_frames=9, frame_rate=24.0, negative_prompt="n")
    assert seen["negative_prompt"] == "n"


def test_two_stage_generate_and_save_omits_negative_when_unset(tmp_path, monkeypatch):
    """Subclasses whose generate_two_stage lacks the kwarg must keep working."""
    pipe = TI2VidTwoStagesPipeline(model_dir=_pack(tmp_path))
    seen = {}

    def gen(**kwargs):
        seen.update(kwargs)
        raise _CapturedError

    monkeypatch.setattr(pipe, "generate_two_stage", gen)
    with pytest.raises(_CapturedError):
        pipe.generate_and_save(prompt="a fox", output_path="x.mp4", num_frames=9, frame_rate=24.0)
    assert "negative_prompt" not in seen


def test_prompt_relay_keeps_a_single_global_negative(tmp_path, monkeypatch):
    from ltx_core_mlx.conditioning.prompt_relay import PromptRelayInput

    pipe = TI2VidTwoStagesPipeline(model_dir=_pack(tmp_path))
    seen = _capture_encode(pipe, monkeypatch)
    # The real setup tokenizes with Gemma; stub it to return the combined positive.
    monkeypatch.setattr(pipe, "_prompt_relay_setup", lambda prompt, relay: ("a cat. a cat sits. a cat jumps", [(0, 1)]))
    relay = PromptRelayInput(local_prompts=["a cat sits", "a cat jumps"])
    with pytest.raises(_CapturedError):
        pipe.generate_two_stage(
            prompt="a cat", num_frames=9, frame_rate=24.0, prompt_relay=relay, negative_prompt="blurry"
        )
    assert seen["negative_prompt"] == "blurry"
    assert seen["prompt"] == "a cat. a cat sits. a cat jumps"  # combined positive, untouched negative


@pytest.mark.parametrize("method", ["retake", "extend"])
def test_retake_extend_forward_negative(tmp_path, monkeypatch, method):
    pipe = RetakePipeline(model_dir=_pack(tmp_path))
    seen = _capture_encode(pipe, monkeypatch)
    video = mx.zeros((1, 128, 3, 2, 2))
    audio = mx.zeros((1, 8, 10, 16))
    extra = {"start_frame": 1, "end_frame": 2} if method == "retake" else {"extend_frames": 1}
    with pytest.raises(_CapturedError):
        getattr(pipe, method)(
            prompt="p",
            source_video_latent=video,
            source_audio_latent=audio,
            frame_rate=24.0,
            negative_prompt="neg",
            **extra,
        )
    assert seen["negative_prompt"] == "neg"


@pytest.mark.parametrize(("method", "inner"), [("retake_from_video", "retake"), ("extend_from_video", "extend")])
def test_from_video_helpers_forward_negative(tmp_path, monkeypatch, method, inner):
    from ltx_pipelines_mlx.retake import _SourceMeta

    pipe = RetakePipeline(model_dir=_pack(tmp_path))
    meta = _SourceMeta(**{f: 1 for f in _SourceMeta.__dataclass_fields__})
    monkeypatch.setattr(pipe, "_encode_source_video", lambda path: (None, None, meta))
    seen = {}

    def inner_stub(**kwargs):
        seen.update(kwargs)
        return "v", "a"

    monkeypatch.setattr(pipe, inner, inner_stub)
    extra = {"start_frame": 1, "end_frame": 2} if inner == "retake" else {"extend_frames": 1}
    getattr(pipe, method)(prompt="p", video_path="v.mp4", negative_prompt="neg", **extra)
    assert seen["negative_prompt"] == "neg"


def test_a2v_forwards_negative(tmp_path, monkeypatch):
    from ltx_pipelines_mlx import a2vid_two_stage as a2v_mod

    pipe = A2VidPipelineTwoStage(model_dir=_pack(tmp_path))
    seen = _capture_encode(pipe, monkeypatch)
    monkeypatch.setattr(pipe, "_load_audio_encoder", lambda: None)
    pipe.audio_encoder = object()
    pipe.audio_processor = object()

    class _Audio:
        waveform = None
        sample_rate = 16000

    monkeypatch.setattr(a2v_mod, "load_audio", lambda *a, **k: _Audio())
    monkeypatch.setattr(a2v_mod, "encode_audio", lambda *a, **k: mx.zeros((1, 8, 64, 16)))
    with pytest.raises(_CapturedError):
        pipe.generate_and_save(
            prompt="p", output_path="x.mp4", audio_path="a.wav", num_frames=9, frame_rate=24.0, negative_prompt="neg"
        )
    assert seen["negative_prompt"] == "neg"


def test_keyframe_interpolate_forwards_negative(tmp_path, monkeypatch):
    pipe = KeyframeInterpolationPipeline(model_dir=_pack(tmp_path))
    seen = _capture_encode(pipe, monkeypatch)
    pipe.image_conditioner = lambda fn, free_after: ([], [])
    with pytest.raises(_CapturedError):
        pipe.interpolate(
            prompt="p",
            keyframe_images=["a.png", "b.png"],
            keyframe_indices=[0, 8],
            num_frames=9,
            frame_rate=24.0,
            cfg_scale=3.0,
            negative_prompt="neg",
        )
    assert seen["negative_prompt"] == "neg"


def test_keyframe_generate_and_save_forwards_negative(tmp_path, monkeypatch):
    pipe = KeyframeInterpolationPipeline(model_dir=_pack(tmp_path))
    seen = {}

    def interp(**kwargs):
        seen.update(kwargs)
        raise _CapturedError

    monkeypatch.setattr(pipe, "interpolate", interp)
    with pytest.raises(_CapturedError):
        pipe.generate_and_save(
            prompt="p",
            output_path="x.mp4",
            keyframe_images=["a.png"],
            keyframe_indices=[0],
            frame_rate=24.0,
            negative_prompt="neg",
        )
    assert seen["negative_prompt"] == "neg"


def test_keyframe_rejects_negative_prompt_and_embeds_together(tmp_path):
    pipe = KeyframeInterpolationPipeline(model_dir=_pack(tmp_path))
    with pytest.raises(ValueError, match="not both"):
        pipe.interpolate(
            prompt="p",
            keyframe_images=["a.png"],
            keyframe_indices=[0],
            frame_rate=24.0,
            negative_prompt="neg",
            negative_prompt_embeds=(mx.zeros((1, 1, 1)), mx.zeros((1, 1, 1))),
        )


# --- CFG-less pipelines refuse it -------------------------------------------------------


@pytest.mark.parametrize("negative", ["", "watermark"])
def test_distilled_refuses_negative_prompt(tmp_path, monkeypatch, negative):
    pipe = DistilledPipeline(model_dir=_pack(tmp_path))
    _capture_encode(pipe, monkeypatch)
    with pytest.raises(ValueError, match="negative_prompt requires a CFG pipeline"):
        pipe.generate_and_save(prompt="p", output_path="x.mp4", num_frames=9, frame_rate=24.0, negative_prompt=negative)
    with pytest.raises(ValueError, match="negative_prompt requires a CFG pipeline"):
        pipe.generate_two_stage(prompt="p", num_frames=9, frame_rate=24.0, negative_prompt=negative)


def test_dfr_refuses_negative_prompt(tmp_path):
    pipe = DFRPipeline(model_dir=_pack(tmp_path, ltx25=True))
    with pytest.raises(ValueError, match="negative_prompt requires a CFG pipeline"):
        pipe.generate_two_stage(prompt="p", num_frames=9, frame_rate=24.0, negative_prompt="n")


# --- CLI --------------------------------------------------------------------------------


@pytest.mark.parametrize("subcommand", ["generate", "a2v", "keyframe", "retake", "extend"])
def test_cli_cfg_subcommands_have_negative_prompt_flag(subcommand):
    from tests.test_docs_flags import parser_flags

    assert "--negative-prompt" in parser_flags(subcommand)


@pytest.mark.parametrize("subcommand", ["ic-lora", "hdr-ic-lora", "lipdub"])
def test_cli_distilled_sampler_subcommands_lack_negative_prompt_flag(subcommand):
    from tests.test_docs_flags import parser_flags

    assert "--negative-prompt" not in parser_flags(subcommand)


def _argv(tmp_path, *extra):
    return ["generate", "-p", "x", "-o", "o.mp4", "--frame-rate", "24", "-f", "9", "--model", _pack(tmp_path), *extra]


@pytest.mark.parametrize("mode", ["--distilled", "--dfr"])
def test_cli_distilled_modes_reject_negative_prompt(tmp_path, mode):
    with pytest.raises(SystemExit):
        _cmd_generate(_build_parser().parse_args(_argv(tmp_path, mode, "--negative-prompt", "n")))


@pytest.mark.parametrize(
    ("mode", "mod", "attr"),
    [
        ("--two-stage", two_stage_mod, "TI2VidTwoStagesPipeline"),
        ("--one-stage", one_stage_mod, "TI2VidOneStagePipeline"),
    ],
)
@pytest.mark.parametrize("negative", [None, "", "blurry"])
def test_cli_generate_forwards_negative_prompt(tmp_path, monkeypatch, mode, mod, attr, negative):
    seen = {}

    class _FakePipe:
        def __init__(self, *a, **k):
            pass

        def generate_and_save(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(mod, attr, _FakePipe)
    extra = [] if negative is None else ["--negative-prompt", negative]
    _cmd_generate(_build_parser().parse_args(_argv(tmp_path, mode, "-q", *extra)))
    if negative is None:
        assert "negative_prompt" not in seen
    else:
        assert seen["negative_prompt"] == negative
