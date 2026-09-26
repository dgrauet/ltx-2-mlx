"""DFRPipeline core path: canvas, slots, detailing LoRA attach, reference conditioning, trimming."""

from __future__ import annotations

import json
import re
from typing import ClassVar

import mlx.core as mx
import pytest

import ltx_pipelines_mlx.dfr as dfr_mod
import ltx_pipelines_mlx.distilled as distilled_mod
from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import DecodeKeyframes
from ltx_pipelines_mlx.cli import _build_parser, _cmd_generate
from ltx_pipelines_mlx.dfr import (
    DEFAULT_DETAILING_LORA,
    DETAILING_LORA_STRENGTH,
    DFRPipeline,
    decode_keyframes_from_slots,
)
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


def test_stage_positions_use_the_snapped_conditioning_fps(tmp_path, monkeypatch):
    """Above 30 fps, stage 1/2 video-side RoPE positions and conditionings snap to 60 fps
    (upstream ``_conditioning_fps``), while the stage-1 audio token count keeps the real
    playback fps."""
    from ltx_core_mlx.utils.positions import compute_audio_token_count, compute_video_positions

    pipe, euler, ancestral, noised, _ = _make(tmp_path, monkeypatch)
    _run(pipe, num_frames=49, frame_rate=48.0)

    stage1_spatial_dims = noised[0]["spatial_dims"]
    expected_pos_1 = compute_video_positions(*stage1_spatial_dims, frame_rate=60.0)
    assert mx.array_equal(noised[0]["positions"], expected_pos_1)

    stage2_spatial_dims = noised[2]["spatial_dims"]
    expected_pos_2 = compute_video_positions(*stage2_spatial_dims, frame_rate=60.0)
    assert mx.array_equal(noised[2]["positions"], expected_pos_2)

    audio_T = compute_audio_token_count(49, frame_rate=48.0)
    assert noised[1]["base_shape"][1] == audio_T

    conds = noised[2]["conditionings"]
    slots = [c for c in conds if isinstance(c, VideoGeneratedKeyframeSlots)]
    assert slots[0].frame_rate == 60.0
    refs = [c for c in conds if isinstance(c, VideoConditionByReferenceLatent)]
    expected_ref_positions = compute_video_positions(*stage1_spatial_dims, frame_rate=60.0)
    assert mx.array_equal(refs[0].reference_positions, expected_ref_positions)


def test_stage_positions_unchanged_at_24fps(tmp_path, monkeypatch):
    """At 24 fps (<= the 30 fps snap threshold), the conditioning fps equals the playback
    fps, so nothing changes numerically."""
    from ltx_core_mlx.utils.positions import compute_video_positions

    pipe, euler, ancestral, noised, _ = _make(tmp_path, monkeypatch)
    _run(pipe, num_frames=49, frame_rate=24.0)

    stage1_spatial_dims = noised[0]["spatial_dims"]
    expected_pos_1 = compute_video_positions(*stage1_spatial_dims, frame_rate=24.0)
    assert mx.array_equal(noised[0]["positions"], expected_pos_1)


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
    materialized: list = []
    monkeypatch.setattr(dfr_mod, "_materialize", lambda *a: materialized.append(a))

    pipe._attach_detailing_lora()

    assert seen["loaded_path"] == str(tmp_path / "detail.safetensors")
    (with_strength,) = seen["apply_loras"]["lora_sd_and_strengths"]
    assert with_strength.strength == DETAILING_LORA_STRENGTH == 0.5
    assert quantized and quantized[0][0] is pipe.dit
    assert pipe.dit.loaded == [list(_Fused.sd.items())]
    # The fused weights must be materialized in place, before the pre-fuse state dicts are
    # dropped — the lazy dequantize->fuse->requantize graph must not defer to stage 2.
    assert len(materialized) == 1


def test_gated_detailing_lora_explains_the_licence_before_any_load(tmp_path, monkeypatch):
    """A gated HF repo (licence not accepted) must fail with the licence URL, before stage 1."""
    from huggingface_hub.errors import GatedRepoError

    pipe, *_ = _make(tmp_path, monkeypatch)
    pipe._detailing_lora_path = None
    pipe.detailing_lora = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
    pipe._load_text_encoder = lambda: (_ for _ in ()).throw(AssertionError("must not load"))  # type: ignore[method-assign]
    gated = GatedRepoError.__new__(GatedRepoError)  # the real ctor needs an HTTP response; only the type matters
    Exception.__init__(gated, "403 Client Error: gated repo")
    monkeypatch.setattr(dfr_mod, "resolve_lora_path", lambda path: (_ for _ in ()).throw(gated))
    with pytest.raises(
        PermissionError, match=re.escape("huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler")
    ):
        _run(pipe)


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
    # `--video-decoder diffusion` goes through the CLI's `_validate_diffvae_tiling` preflight,
    # which sizes the decode budget from host memory (mx.device_info()["memory_size"] // 2) unless
    # overridden here. Pin it well above any CI runner's RAM so this test doesn't depend on the
    # host's memory size.
    monkeypatch.setenv("LTX2_VAE_DECODE_BUDGET_GB", "64")
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


def test_cli_temporal_flags_require_dfr(tmp_path):
    with pytest.raises(SystemExit, match="--temporal-upscalings"):
        _cmd_generate(_build_parser().parse_args(_argv(tmp_path, "--distilled", "--temporal-upscalings", "1")))
    with pytest.raises(SystemExit, match="--temporal-upsampler-path"):
        _cmd_generate(_build_parser().parse_args(_argv(tmp_path, "--distilled", "--temporal-upsampler-path", "t")))


def test_cli_temporal_upscalings_choices(tmp_path):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(_argv(tmp_path, "--dfr", "--temporal-upscalings", "3"))


def test_cli_temporal_flags_reach_the_pipeline(monkeypatch, tmp_path):
    captured = {}

    class _FakePipe:
        def __init__(self, **kw):
            captured.update(kw)

        def generate_and_save(self, **kw):
            return "o.mp4"

    monkeypatch.setattr(dfr_mod, "DFRPipeline", _FakePipe)
    upsampler_path = str(tmp_path / "t.safetensors")
    _cmd_generate(
        _build_parser().parse_args(
            _argv(tmp_path, "--dfr", "--temporal-upscalings", "2", "--temporal-upsampler-path", upsampler_path)
        )
    )
    assert captured["temporal_upscalings"] == 2
    assert captured["temporal_upsampler_path"] == upsampler_path


def test_cli_rounds_refuse_segments(tmp_path):
    with pytest.raises(SystemExit, match="--segment"):
        _cmd_generate(
            _build_parser().parse_args(_argv(tmp_path, "--dfr", "--temporal-upscalings", "1", "--segment", "a"))
        )


def test_cli_rounds_refuse_tiling(tmp_path):
    with pytest.raises(SystemExit, match="--tile"):
        _cmd_generate(
            _build_parser().parse_args(_argv(tmp_path, "--dfr", "--temporal-upscalings", "1", "--tile-spatial", "2"))
        )


def test_cli_spatial_upscalings_flag_parses_with_default(tmp_path):
    args = _build_parser().parse_args(_argv(tmp_path, "--dfr"))
    assert args.spatial_upscalings == 1
    args = _build_parser().parse_args(_argv(tmp_path, "--dfr", "--spatial-upscalings", "2"))
    assert args.spatial_upscalings == 2


def test_cli_spatial_upscalings_choices(tmp_path):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(_argv(tmp_path, "--dfr", "--spatial-upscalings", "3"))


def test_cli_spatial_upscalings_requires_dfr(tmp_path):
    with pytest.raises(SystemExit, match="--spatial-upscalings"):
        _cmd_generate(_build_parser().parse_args(_argv(tmp_path, "--distilled", "--spatial-upscalings", "2")))


def test_cli_spatial_upscalings_reaches_the_pipeline(monkeypatch, tmp_path):
    captured = {}

    class _FakePipe:
        def __init__(self, **kw):
            captured.update(kw)

        def generate_and_save(self, **kw):
            return "o.mp4"

    monkeypatch.setattr(dfr_mod, "DFRPipeline", _FakePipe)
    _cmd_generate(_build_parser().parse_args(_argv(tmp_path, "--dfr", "--spatial-upscalings", "2")))
    assert captured["spatial_upscalings"] == 2


def test_cli_spatial_upscalings_2_refuses_segments_before_pipeline_load(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise AssertionError("DFRPipeline must not be constructed when the CLI refuses --segment up front")

    monkeypatch.setattr(dfr_mod, "DFRPipeline", _boom)
    with pytest.raises(SystemExit, match="--segment"):
        _cmd_generate(
            _build_parser().parse_args(_argv(tmp_path, "--dfr", "--spatial-upscalings", "2", "--segment", "a"))
        )


def test_cli_spatial_upscalings_2_refuses_tiling_before_pipeline_load(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise AssertionError("DFRPipeline must not be constructed when the CLI refuses --tile-* up front")

    monkeypatch.setattr(dfr_mod, "DFRPipeline", _boom)
    with pytest.raises(SystemExit, match="--tile"):
        _cmd_generate(
            _build_parser().parse_args(_argv(tmp_path, "--dfr", "--spatial-upscalings", "2", "--tile-spatial", "2"))
        )


def test_decode_keyframes_from_slots_filters_the_canvas_padding(capsys):
    slots = mx.random.normal((1, 128, 3, 4, 4))
    kf = decode_keyframes_from_slots(slots, [24, 48, 72], num_frames=49, verbose=True)
    assert (
        isinstance(kf, DecodeKeyframes) and kf.pixel_frame_indices == (24, 48) and kf.latents.shape == (1, 128, 2, 4, 4)
    )
    assert mx.array_equal(kf.latents[:, :, 1], slots[:, :, 1]) and kf.clip_start_frame == 0
    assert "dropping 1 keyframe slot" in capsys.readouterr().err
    assert decode_keyframes_from_slots(slots, [72, 96, 120], num_frames=49) is None
    assert decode_keyframes_from_slots(None, [], num_frames=49) is None
    with pytest.raises(ValueError, match="slot count"):
        decode_keyframes_from_slots(slots, [24], num_frames=49)


def test_dfr_decode_passes_the_trimmed_slots_as_keyframes(tmp_path, monkeypatch):
    pipe, *_ = _make(tmp_path, monkeypatch)
    _run(pipe, num_frames=49)  # resolve_canvas(49): 48 = 2 x 24 -> canvas 49, slots at [24, 48], both < 49
    seen = {}
    monkeypatch.setattr(
        distilled_mod.DistilledPipeline,
        "_decode_and_save_video",
        lambda self, v, a, out, *, frame_rate, seed=0, keyframes=None: seen.update(keyframes=keyframes) or out,
    )
    video = mx.zeros((1, 128, 7, 8, 8))  # 49 pixel frames
    pipe._decode_and_save_video(video, mx.zeros((1, 8, 4, 16)), str(tmp_path / "o.mp4"), frame_rate=24.0, seed=1)
    kf = seen["keyframes"]
    assert isinstance(kf, DecodeKeyframes)
    assert all(0 <= i < 49 for i in kf.pixel_frame_indices)
    assert kf.num_planes == len(kf.pixel_frame_indices) == sum(1 for p in pipe.generated_keyframe_positions if p < 49)
    # an explicit keyframes= wins over the pipeline's own slots
    explicit = DecodeKeyframes(mx.zeros((1, 128, 1, 8, 8)), (3,))
    pipe._decode_and_save_video(
        video, mx.zeros((1, 8, 4, 16)), str(tmp_path / "o.mp4"), frame_rate=24.0, keyframes=explicit
    )
    assert seen["keyframes"] is explicit
