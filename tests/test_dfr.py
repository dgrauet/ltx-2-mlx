"""DFRPipeline core path: canvas, slots, detailing LoRA attach, reference conditioning, trimming."""

from __future__ import annotations

import json
from typing import ClassVar

import mlx.core as mx
import pytest

import ltx_pipelines_mlx.dfr as dfr_mod
import ltx_pipelines_mlx.distilled as distilled_mod
from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_pipelines_mlx.cli import _build_parser, _cmd_generate
from ltx_pipelines_mlx.dfr import DEFAULT_DETAILING_LORA, DETAILING_LORA_STRENGTH, DFRPipeline
from ltx_pipelines_mlx.scheduler import LTX_2_5_STAGE_2_DISTILLED_SIGMAS
from tests.test_ltx25_distilled import _fake_upsampler, _FakeVaeEncoder, _LoopSpy


def _write_25_pack(tmp_path):
    cfg = {"transformer": {"num_layers": 48, "ff_bias": False, "use_keyframes_abs_pos_embedding": True}}
    (tmp_path / "embedded_config.json").write_text(json.dumps(cfg))
    (tmp_path / "detail.safetensors").write_bytes(b"")
    (tmp_path / "vae_decoder_av.safetensors").write_bytes(b"")
    return tmp_path


def _make(tmp_path, monkeypatch, *, low_ram=False):
    _write_25_pack(tmp_path)
    pipe = DFRPipeline(
        str(tmp_path),
        low_memory=False,
        low_ram_streaming=low_ram,
        detailing_lora=str(tmp_path / "detail.safetensors"),
    )
    pipe._load_text_encoder = lambda: None  # type: ignore[method-assign]
    pipe._encode_text = lambda prompt: (  # type: ignore[method-assign]
        mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
    )
    pipe.load = lambda: None  # type: ignore[method-assign]
    pipe.dit = object()  # type: ignore[assignment]
    pipe.vae_encoder = _FakeVaeEncoder()  # type: ignore[assignment]
    pipe.upsampler = _fake_upsampler  # type: ignore[assignment]
    # Pre-resolved: generate_two_stage resolves the LoRA up front, which would otherwise
    # read the empty stub file and reset the downscale factor to the default 1.
    pipe._detailing_lora_path = str(tmp_path / "detail.safetensors")
    pipe._detailing_downscale = 2
    attached: list = []
    monkeypatch.setattr(pipe, "_attach_detailing_lora", lambda: attached.append(("attach", pipe.dit)))
    monkeypatch.setattr(distilled_mod, "X0Model", lambda dit: dit)
    euler, ancestral = _LoopSpy(), _LoopSpy()
    monkeypatch.setattr(distilled_mod, "denoise_loop", euler)
    monkeypatch.setattr(distilled_mod, "euler_ancestral_denoising_loop", ancestral)
    noised: list[dict] = []
    real = distilled_mod.create_noised_state

    def spy(**kwargs):
        noised.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(distilled_mod, "create_noised_state", spy)
    return pipe, euler, ancestral, noised, attached


def _run(pipe, **overrides):
    kwargs = dict(prompt="a fox", height=128, width=128, num_frames=49, frame_rate=24.0, seed=7)
    kwargs.update(overrides)
    return pipe.generate_two_stage(**kwargs)


def test_constants():
    assert DEFAULT_DETAILING_LORA == "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
    assert DETAILING_LORA_STRENGTH == 0.5


def test_stage1_runs_on_the_canvas_with_segment_slots(tmp_path, monkeypatch):
    pipe, euler, ancestral, noised, _ = _make(tmp_path, monkeypatch)
    _run(pipe, num_frames=49)  # 49 -> canvas 49, segment 24, slots [24, 48]
    assert pipe.canvas_frames == 49 and pipe.generated_keyframe_positions == [24, 48]
    slots = [c for c in noised[0]["conditionings"] if isinstance(c, VideoGeneratedKeyframeSlots)]
    assert len(slots) == 1 and slots[0].pixel_frame_indices == (24, 48) and slots[0].initial_keyframes is None
    assert len(ancestral.calls) == 1


def test_stage2_gets_upsampled_slots_reference_and_detailing_lora(tmp_path, monkeypatch):
    pipe, euler, ancestral, noised, attached = _make(tmp_path, monkeypatch)
    _run(pipe, num_frames=49)
    conds = noised[2]["conditionings"]  # stage-2 video state
    slots = [c for c in conds if isinstance(c, VideoGeneratedKeyframeSlots)]
    refs = [c for c in conds if isinstance(c, VideoConditionByReferenceLatent)]
    assert len(slots) == 1 and slots[0].pixel_frame_indices == (24, 48)
    assert slots[0].initial_keyframes is not None and slots[0].initial_keyframes.shape == (1, 128, 2, 4, 4)
    assert len(refs) == 1 and refs[0].downscale_factor == 2 and refs[0].strength == 1.0
    # half-res stage-1 latent (7, 2, 2) as reference tokens
    assert refs[0].reference_latent.shape == (1, 7 * 2 * 2, 128)
    assert attached == [("attach", pipe.dit)]  # attached once, after stage 1 (dit still the stage-1 object)
    assert noised[2]["sigma"] == LTX_2_5_STAGE_2_DISTILLED_SIGMAS[0]
    assert len(euler.calls) == 1


def test_outputs_are_trimmed_to_the_requested_frames(tmp_path, monkeypatch):
    pipe, *_ = _make(tmp_path, monkeypatch)
    video, audio = _run(pipe, num_frames=137)  # canvas 145 -> 19 latent frames; keep 18
    assert pipe.canvas_frames == 145
    assert video.shape[2] == (137 - 1) // 8 + 1
    from ltx_core_mlx.utils.positions import compute_audio_token_count

    assert audio.shape[2] == compute_audio_token_count(137, 24.0)  # audio latent is (B, 8, T, 16)


def test_audio_is_stage1_audio(tmp_path, monkeypatch):
    pipe, euler, ancestral, _, _ = _make(tmp_path, monkeypatch)
    _, audio = _run(pipe, num_frames=49)
    stage1_audio = ancestral.calls[0]["audio_state"].latent
    assert audio.shape[2] == stage1_audio.shape[1]  # (B, 8, T, 16) vs stage-1 tokens (B, T, 128)


def test_stage2_slots_are_kept_for_keyframe_decode(tmp_path, monkeypatch):
    pipe, *_ = _make(tmp_path, monkeypatch)
    _run(pipe, num_frames=49)
    assert pipe.generated_keyframes is not None and pipe.generated_keyframes.shape == (1, 128, 2, 4, 4)


def test_refuses_23_pack_before_any_load(tmp_path):
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48}}))
    pipe = DFRPipeline(str(tmp_path), low_memory=False, detailing_lora=str(tmp_path / "x.safetensors"))
    pipe._load_text_encoder = lambda: (_ for _ in ()).throw(AssertionError("must not load"))  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="use_keyframes_abs_pos_embedding"):
        _run(pipe)


def test_refuses_generated_keyframes_and_teacache_kwargs(tmp_path, monkeypatch):
    pipe, *_ = _make(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="canvas"):
        _run(pipe, generated_keyframes=3)
    with pytest.raises(ValueError, match="TeaCache"):
        _run(pipe, enable_teacache=True)


def test_refuses_an_unresolvable_detailing_lora_before_any_load(tmp_path, monkeypatch):
    """A bad --detailing-lora must fail before stage 1, not after a full half-res render.

    ``resolve_lora_path`` treats a non-existent local path as an HF repo id, so the stub
    stands in for its failure instead of reaching the network.
    """
    pipe, *_ = _make(tmp_path, monkeypatch)
    pipe._detailing_lora_path = None  # force a real resolve
    pipe._load_text_encoder = lambda: (_ for _ in ()).throw(AssertionError("must not load"))  # type: ignore[method-assign]
    monkeypatch.setattr(
        dfr_mod,
        "resolve_lora_path",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(f"no such LoRA: {path}")),
    )
    with pytest.raises(FileNotFoundError, match="no such LoRA"):
        _run(pipe)


def test_attach_detailing_lora_fuses_in_place_when_not_streaming(tmp_path, monkeypatch):
    _write_25_pack(tmp_path)
    pipe = DFRPipeline(str(tmp_path), low_memory=False, detailing_lora=str(tmp_path / "detail.safetensors"))

    class _Dit:
        def __init__(self):
            self.loaded: list = []

        def parameters(self):
            return {"w": mx.zeros((2, 2))}

        def load_weights(self, items):
            self.loaded.append(items)

    pipe.dit = _Dit()  # type: ignore[assignment]
    seen: dict = {}

    class _Loader:
        def load(self, path, sd_ops=None):
            seen["loaded_path"] = path
            return {"lora": 1}

    class _Fused:
        sd: ClassVar[dict] = {"w": mx.ones((2, 2))}

    monkeypatch.setattr(dfr_mod, "SafetensorsStateDictLoader", _Loader)
    monkeypatch.setattr(dfr_mod, "apply_loras", lambda **kw: seen.update(apply_loras=kw) or _Fused())
    quantized: list = []
    monkeypatch.setattr(dfr_mod, "apply_quantization", lambda dit, sd: quantized.append((dit, sd)))

    pipe._attach_detailing_lora()

    assert seen["loaded_path"] == str(tmp_path / "detail.safetensors")
    (with_strength,) = seen["apply_loras"]["lora_sd_and_strengths"]
    assert with_strength.strength == DETAILING_LORA_STRENGTH == 0.5
    assert quantized and quantized[0][0] is pipe.dit
    assert pipe.dit.loaded == [list(_Fused.sd.items())]


def test_attach_detailing_lora_streaming_appends_a_block_source(tmp_path, monkeypatch):
    _write_25_pack(tmp_path)
    pipe = DFRPipeline(
        str(tmp_path),
        low_memory=False,
        low_ram_streaming=True,
        detailing_lora=str(tmp_path / "detail.safetensors"),
    )

    class _Streamer:
        _lora_sources: ClassVar[list] = []

    pipe.dit = _Streamer()  # type: ignore[assignment]
    made = []
    monkeypatch.setattr(dfr_mod, "BlockLoraSource", lambda path, **kw: made.append((path, kw)) or ("src", path))
    pipe._attach_detailing_lora()
    assert made[0][0] == str(tmp_path / "detail.safetensors") and made[0][1]["strength"] == 0.5
    assert object.__getattribute__(pipe.dit, "_lora_sources") == [("src", str(tmp_path / "detail.safetensors"))]


def _argv(tmp_path, *extra):
    return [
        "generate",
        "-p",
        "x",
        "-o",
        "o.mp4",
        "--frame-rate",
        "24",
        "-f",
        "49",
        "--model",
        str(_write_25_pack(tmp_path)),
        *extra,
    ]


def test_cli_dfr_flag_parses_with_defaults(tmp_path):
    args = _build_parser().parse_args(_argv(tmp_path, "--dfr"))
    assert args.dfr is True and args.detailing_lora == DEFAULT_DETAILING_LORA
    args = _build_parser().parse_args(_argv(tmp_path, "--dfr", "--detailing-lora", "/x/y.safetensors"))
    assert args.detailing_lora == "/x/y.safetensors"


def test_cli_dfr_reaches_the_pipeline(monkeypatch, tmp_path):
    seen = {}

    class _FakePipe:
        def __init__(self, *a, **k):
            seen["init"] = k

        def generate_and_save(self, **kwargs):
            seen["kwargs"] = kwargs
            seen["video_decoder"] = getattr(self, "video_decoder", None)

    monkeypatch.setattr(dfr_mod, "DFRPipeline", _FakePipe)
    _cmd_generate(
        _build_parser().parse_args(
            _argv(
                tmp_path, "--dfr", "--detailing-lora", "/x/y.safetensors", "--low-ram", "--video-decoder", "diffusion"
            )
        )
    )
    assert seen["init"]["detailing_lora"] == "/x/y.safetensors" and seen["init"]["low_ram_streaming"] is True
    assert seen["kwargs"]["num_frames"] == 49 and "generated_keyframes" not in seen["kwargs"]
    assert seen["video_decoder"] == "diffusion"


@pytest.mark.parametrize(
    "bad",
    [
        ["--num-generated-keyframes", "2"],
        ["--enable-teacache"],
        ["--cfg-scale", "3"],
        ["--stg-scale", "1"],
        ["--distilled"],
    ],
)
def test_cli_dfr_rejects_incompatible_flags(tmp_path, bad):
    with pytest.raises(SystemExit):
        _cmd_generate(_build_parser().parse_args(_argv(tmp_path, "--dfr", *bad)))


def test_cli_detailing_lora_requires_dfr(tmp_path):
    with pytest.raises(SystemExit):
        _cmd_generate(_build_parser().parse_args(_argv(tmp_path, "--distilled", "--detailing-lora", "/x.safetensors")))
