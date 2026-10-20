"""ExtendDistilledPipeline (``extend --distilled``): the source tail pinned at the start of a new distilled window.

Upstream continues a video chunk by chunk, pinning the previous window's last 25 frames of video and audio latent at
index 0 in both distilled stages. These tests pin (1) the new stage-1 hook's default (no conditioning, so every other
distilled path is unchanged) and the fall-back of the overridden hooks, (2) the pinned tokens in the stage-1 / stage-2
states and their survival through both distilled loops, (3) the ``extend_from_video`` flow (which tails are carried,
the window size, how the window is appended, the checks that run before any model work), and (4) the CLI.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions
from ltx_pipelines_mlx import extend_distilled as ext_mod
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.extend_distilled import (
    EXTEND_CARRY_PIXEL_FRAMES,
    ExtendDistilledPipeline,
    _Carry,
    carry_audio_tokens,
    carry_latent_frames,
    fit_audio_latent,
)
from ltx_pipelines_mlx.utils.helpers import create_noised_state

H, W = 2, 3  # latent tokens per frame side (tiny)


def _write_pack(tmp_path) -> str:
    cfg = {"transformer": {"num_layers": 48, "ff_bias": False}}  # an LTX-2.5 pack: ancestral sampler
    (tmp_path / "embedded_config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def _tokens(n: int, seed: int) -> mx.array:
    return mx.random.normal((1, n, 128), key=mx.random.key(seed)).astype(mx.bfloat16)


def _carry(audio_tokens: int = 4) -> _Carry:
    return _Carry(
        video_half=_tokens(4 * H * W, 1),
        video_full=_tokens(4 * 4 * H * W, 2),
        audio=_tokens(audio_tokens, 3),
        latent_frames=4,
    )


def test_carry_sizes_follow_upstream_chunk_defaults():
    """25 carried pixel frames = 4 latent frames; audio = round(25 / fps * 25) tokens."""
    assert EXTEND_CARRY_PIXEL_FRAMES == 25
    assert carry_latent_frames() == 4
    assert carry_audio_tokens(24.0) == 26
    assert carry_audio_tokens(30.0) == 21


# --- the hooks --------------------------------------------------------------------------------------


def test_default_stage1_video_hook_adds_nothing(tmp_path):
    """Refactor guard: generate --distilled / --dfr / a2v --distilled get no extra stage-1 conditioning."""
    pipe = DistilledPipeline(model_dir=_write_pack(tmp_path))
    assert pipe._stage1_video_conditionings((5, H, W)) == []


def test_hooks_without_a_carry_fall_back_to_the_parent(tmp_path):
    ext = ExtendDistilledPipeline(model_dir=_write_pack(tmp_path))
    parent = DistilledPipeline(model_dir=str(tmp_path))
    positions = compute_audio_positions(13)
    stage1_out = _tokens(13, 5)

    assert ext._stage1_video_conditionings((5, H, W)) == []
    for a, b in (
        (
            ext._stage1_audio_state((1, 13, 128), positions, (5, H, W), 43),
            parent._stage1_audio_state((1, 13, 128), positions, (5, H, W), 43),
        ),
        (
            ext._stage2_audio_state(stage1_out, positions, (5, H, W), 44, 0.9),
            parent._stage2_audio_state(stage1_out, positions, (5, H, W), 44, 0.9),
        ),
    ):
        for field in ("latent", "clean_latent", "denoise_mask"):
            assert mx.array_equal(getattr(a, field), getattr(b, field)).item(), field


def _assert_prefix_pinned(state, tokens: mx.array) -> None:
    n = tokens.shape[1]
    assert mx.array_equal(state.latent[:, :n], tokens).item()
    assert mx.array_equal(state.clean_latent[:, :n], tokens).item()
    assert not mx.any(state.denoise_mask[:, :n]).item()
    assert mx.all(state.denoise_mask[:, n:] == 1).item()
    assert not state.frozen


def test_stage1_video_hook_pins_the_half_resolution_tail(tmp_path):
    pipe = ExtendDistilledPipeline(model_dir=_write_pack(tmp_path))
    pipe._carry = _carry()
    F = 7
    state = create_noised_state(
        base_shape=(1, F * H * W, 128),
        conditionings=pipe._stage1_video_conditionings((F, H, W)),
        spatial_dims=(F, H, W),
        positions=compute_video_positions(F, H, W, frame_rate=24.0),
        seed=42,
        sigma=1.0,
        legacy_scalar_blend=True,
    )
    _assert_prefix_pinned(state, pipe._carry.video_half)


@pytest.mark.parametrize("stage", [1, 2])
def test_audio_hooks_pin_the_audio_tail(tmp_path, stage):
    pipe = ExtendDistilledPipeline(model_dir=_write_pack(tmp_path))
    pipe._carry = _carry(audio_tokens=5)
    positions = compute_audio_positions(13)
    if stage == 1:
        state = pipe._stage1_audio_state((1, 13, 128), positions, (5, H, W), 43)
    else:
        state = pipe._stage2_audio_state(_tokens(13, 9), positions, (5, H, W), 44, 0.9)
    _assert_prefix_pinned(state, pipe._carry.audio)
    assert state.positions.shape == (1, 13, 1)


@pytest.mark.parametrize("ancestral", [False, True], ids=["euler", "ancestral"])
def test_pinned_tail_comes_out_of_both_loops_bit_identical(tmp_path, ancestral):
    """The carried video and audio tokens leave the denoising loop exactly as they went in; the rest is generated."""
    pipe = ExtendDistilledPipeline(model_dir=_write_pack(tmp_path))
    pipe._carry = _carry(audio_tokens=5)
    F = 6
    video_state = create_noised_state(
        base_shape=(1, F * H * W, 128),
        conditionings=pipe._stage1_video_conditionings((F, H, W)),
        spatial_dims=(F, H, W),
        positions=compute_video_positions(F, H, W, frame_rate=24.0),
        seed=42,
        sigma=1.0,
        legacy_scalar_blend=True,
    )
    audio_state = pipe._stage1_audio_state((1, 13, 128), compute_audio_positions(13), (F, H, W), 43)
    calls: list[dict] = []

    def model(**kwargs):
        calls.append(kwargs)
        return (
            mx.full(kwargs["video_latent"].shape, 7.0, dtype=mx.bfloat16),
            mx.full(kwargs["audio_latent"].shape, -7.0, dtype=mx.bfloat16),
        )

    out = pipe._run_denoise_loop(
        model=model,
        video_state=video_state,
        audio_state=audio_state,
        video_text_embeds=mx.zeros((1, 8, 4096)),
        audio_text_embeds=mx.zeros((1, 8, 2048)),
        sigmas=[1.0, 0.7, 0.4, 0.0],
        video_cross_attention_mask=None,
        on_step=None,
        seed=42,
        ancestral=ancestral,
        noise_seed_offset=10000,
    )

    n_video = pipe._carry.video_half.shape[1]
    assert mx.array_equal(out.video_latent[:, :n_video], pipe._carry.video_half).item()
    assert not mx.array_equal(out.video_latent[:, n_video:], video_state.latent[:, n_video:]).item()
    assert mx.array_equal(out.audio_latent[:, :5], pipe._carry.audio).item()
    # The pinned tokens are conditioned at sigma 0 on every step, the rest at the step sigma.
    for kwargs in calls:
        assert not mx.any(kwargs["video_timesteps"][:, :n_video]).item()
        assert not mx.any(kwargs["audio_timesteps"][:, :5]).item()


# --- extend_from_video ------------------------------------------------------------------------------------


class _StopError(Exception):
    """Raised by a stub to end a run at a known point."""


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """A pipeline whose encoders and stages are stubs; records what each stage got."""
    pipe = ExtendDistilledPipeline(model_dir=_write_pack(tmp_path), low_memory=True)
    record: dict = {"encoded": []}
    info = SimpleNamespace(num_frames=49, height=128, width=192, fps=24.0, has_audio=True)
    monkeypatch.setattr(ext_mod, "probe_video_info", lambda path: info)
    monkeypatch.setattr(ext_mod, "load_video_frames", lambda path, h, w, f: (h, w, f))

    def fake_encode(encoder, pixels):
        h, w, f = pixels
        record["encoded"].append(pixels)
        lat_f = (f - 1) // 8 + 1
        # Latent frame i holds the value i (+100 at half resolution), so the tail is recognisable.
        frames = mx.arange(lat_f).reshape(1, 1, lat_f, 1, 1).astype(mx.bfloat16)
        return mx.broadcast_to(frames + (100 if h == 64 else 0), (1, 128, lat_f, h // 32, w // 32))

    monkeypatch.setattr(ext_mod, "encode_video_tensor", fake_encode)
    monkeypatch.setattr(pipe.image_conditioner, "load", lambda: object())
    source_audio = mx.broadcast_to(mx.arange(51).reshape(1, 1, 51, 1).astype(mx.bfloat16), (1, 8, 51, 16))
    monkeypatch.setattr(ext_mod, "encode_source_audio_latent", lambda *a, **k: source_audio)

    def fake_stage1(prompt, height, width, num_frames, **kwargs):
        record["stage1"] = dict(prompt=prompt, height=height, width=width, num_frames=num_frames, **kwargs)
        F = (num_frames - 1) // 8 + 1
        record["stage1_video_cond"] = pipe._stage1_video_conditionings((F, height // 64, width // 64))
        record["carry_during_stage1"] = pipe._carry
        tokens = mx.zeros((1, F * (height // 64) * (width // 64), 128))
        return (
            SimpleNamespace(video_tokens=tokens, latent_dims=(F, height // 64, width // 64)),
            num_frames,
            height,
            width,
        )

    def fake_upsample(video_half):
        return mx.repeat(mx.repeat(video_half, 2, axis=3), 2, axis=4)

    def fake_stage2(stage1, video_upscaled, *, num_frames, frame_rate, seed, stage2_steps, extra_conditionings):
        record["stage2"] = dict(num_frames=num_frames, seed=seed, extra=extra_conditionings)
        F = video_upscaled.shape[2]
        video = mx.full((1, 128, F, video_upscaled.shape[3], video_upscaled.shape[4]), -1.0)
        audio = mx.full((1, 8, round(num_frames / frame_rate * 25), 16), -1.0)
        return video, audio

    monkeypatch.setattr(pipe, "_stage1", fake_stage1)
    monkeypatch.setattr(pipe, "_upsample_latent", fake_upsample)
    monkeypatch.setattr(pipe, "_stage2", fake_stage2)
    return pipe, record, source_audio


def test_extend_from_video_pins_the_tails_and_appends_the_window(stubbed):
    pipe, record, source_audio = stubbed

    video, audio = pipe.extend_from_video("she smiles", "src.mp4", extend_frames=3, seed=7, stage1_steps=4)

    # Both resolutions of the 49-frame source are encoded once each.
    assert record["encoded"] == [(128, 192, 49), (64, 96, 49)]
    # Window: 4 carried + 3 new latent frames = 49 pixel frames, at the source size.
    s1 = record["stage1"]
    assert (s1["num_frames"], s1["height"], s1["width"], s1["seed"], s1["stage1_steps"]) == (49, 128, 192, 7, 4)
    assert s1["image"] is None and s1["images"] is None and s1["generated_keyframes"] == 0
    # Stage 1 pins the half-resolution tail (source latent frames 3..6 -> values 103..106).
    carry = record["carry_during_stage1"]
    assert carry.video_half.shape == (1, 4 * 2 * 3, 128)
    assert mx.array_equal(carry.video_half[0, :, 0], mx.repeat(mx.arange(103, 107), 6).astype(mx.bfloat16)).item()
    assert record["stage1_video_cond"][0].frame_indices == [0, 1, 2, 3]
    # Audio tail: the last 26 tokens (25 frames at 24 fps).
    assert mx.array_equal(carry.audio[0, :, 0], mx.arange(25, 51).astype(mx.bfloat16)).item()
    # Stage 2 pins the full-resolution tail.
    (cond,) = record["stage2"]["extra"]
    assert cond.frame_indices == [0, 1, 2, 3]
    assert mx.array_equal(cond.clean_latent[0, :, 0], mx.repeat(mx.arange(3, 7), 4 * 6).astype(mx.bfloat16)).item()
    # Output: the source's 7 latent frames, then the window's 3 new ones; audio likewise.
    assert video.shape == (1, 128, 10, 4, 6)
    assert mx.array_equal(video[:, :, :7, 0, 0][0, 0], mx.arange(7).astype(mx.bfloat16)).item()
    assert mx.all(video[:, :, 7:] == -1).item()
    assert audio.shape == (1, 8, 51 + 51 - 26, 16)
    assert mx.array_equal(audio[:, :, :51], source_audio).item()
    assert mx.all(audio[:, :, 51:] == -1).item()
    assert pipe._carry is None
    assert pipe.source_frame_rate == 24.0


def test_the_carry_is_cleared_when_a_stage_fails(stubbed, monkeypatch):
    pipe, _, _ = stubbed

    def boom(*a, **k):
        raise _StopError

    monkeypatch.setattr(pipe, "_stage2", boom)
    with pytest.raises(_StopError):
        pipe.extend_from_video("x", "src.mp4", extend_frames=2)
    assert pipe._carry is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(extend_frames=0), "extend_frames must be >= 1"),
        (dict(height=120), "multiple of 64"),
        (dict(num_frames=17), "needs at least that many"),
    ],
)
def test_bad_requests_fail_before_any_encode(stubbed, change, message):
    pipe, record, _ = stubbed
    info_change = {k: v for k, v in change.items() if k != "extend_frames"}
    if info_change:
        info = ext_mod.probe_video_info("src.mp4")
        for k, v in info_change.items():
            setattr(info, k, v)
    with pytest.raises(ValueError, match=message):
        pipe.extend_from_video("x", "src.mp4", extend_frames=change.get("extend_frames", 2))
    assert record["encoded"] == []


# --- CLI ------------------------------------------------------------------------------------------------


def _extend_args(tmp_path, *extra: str):
    from ltx_pipelines_mlx import cli

    return cli._build_parser().parse_args(
        ["extend", "-p", "continue", "-v", "src.mp4", "--extend-frames", "12", "-o", str(tmp_path / "o.mp4"), "-q"]
        + list(extra)
    )


def test_cli_extend_distilled_builds_the_distilled_pipeline(monkeypatch, tmp_path):
    from ltx_pipelines_mlx import cli

    received: dict = {}

    class _FakePipe:
        def __init__(self, **kwargs) -> None:
            received["init"] = kwargs

        def extend_from_video(self, **kwargs):
            received.update(kwargs)
            return None, None

    monkeypatch.setattr(ext_mod, "ExtendDistilledPipeline", _FakePipe)
    monkeypatch.setattr(cli, "_decode_and_save", lambda *a, **k: received.setdefault("decoded", True))
    monkeypatch.setattr(cli, "_print_result", lambda *a, **k: None)
    cli._cmd_extend(_extend_args(tmp_path, "--distilled", "--low-ram", "--seed", "9"))

    assert received["init"]["low_ram_streaming"] is True
    assert (received["extend_frames"], received["seed"], received["video_path"]) == (12, 9, "src.mp4")
    assert received["decoded"] is True


def test_cli_extend_default_still_builds_the_dev_pipeline(monkeypatch, tmp_path):
    from ltx_pipelines_mlx import cli
    from ltx_pipelines_mlx import retake as retake_mod

    def must_not_build(**kwargs):
        raise AssertionError("the default extend must not build the distilled pipeline")

    built: dict = {}

    class _FakeRetake:
        def __init__(self, **kwargs) -> None:
            built["init"] = kwargs

        def extend_from_video(self, **kwargs):
            built.update(kwargs)
            return None, None

    monkeypatch.setattr(ext_mod, "ExtendDistilledPipeline", must_not_build)
    monkeypatch.setattr(retake_mod, "RetakePipeline", _FakeRetake)
    monkeypatch.setattr(cli, "_decode_and_save", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_print_result", lambda *a, **k: None)
    cli._cmd_extend(_extend_args(tmp_path, "--cfg-scale", "2.5"))

    assert "distilled" not in built["init"]
    assert built["cfg_scale"] == 2.5


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (("--cfg-scale", "3"), "--cfg-scale"),
        (("--stg-scale", "1"), "--stg-scale"),
        (("--negative-prompt", "blurry"), "--negative-prompt"),
        (("--negative-prompt", ""), "--negative-prompt"),
        (("--steps", "8"), "--steps"),
        (("--direction", "before"), "--direction before"),
    ],
)
def test_cli_extend_distilled_rejects_flags_before_building_anything(monkeypatch, tmp_path, flags, message):
    from ltx_pipelines_mlx import cli

    def must_not_build(**kwargs):
        raise AssertionError("the flags must be rejected before the pipeline is built")

    monkeypatch.setattr(ext_mod, "ExtendDistilledPipeline", must_not_build)
    with pytest.raises(SystemExit, match=message):
        cli._cmd_extend(_extend_args(tmp_path, "--distilled", *flags))


def test_stage1_applies_the_video_hook(tmp_path, monkeypatch):
    """DistilledPipeline._stage1 builds its video state with the hook's conditionings appended last."""
    from ltx_pipelines_mlx import distilled as distilled_mod

    pipe = ExtendDistilledPipeline(model_dir=_write_pack(tmp_path), low_memory=False)
    pipe._carry = _carry()
    seen: dict = {}
    monkeypatch.setattr(pipe, "_load_text_encoder", lambda: None)
    monkeypatch.setattr(pipe, "_encode_text", lambda p: (mx.zeros((1, 4, 4096)), mx.zeros((1, 4, 2048))))

    def fake_load() -> None:
        pipe.dit, pipe.upsampler = object(), object()
        pipe.image_conditioner._encoder = object()

    monkeypatch.setattr(pipe, "load", fake_load)
    monkeypatch.setattr(type(pipe), "vae_encoder", property(lambda self: object()), raising=False)

    def capture(**kwargs):
        seen.update(kwargs)
        raise _StopError

    monkeypatch.setattr(distilled_mod, "create_noised_state", capture)
    with pytest.raises(_StopError):
        pipe._stage1(
            "x",
            128,
            192,
            49,
            frame_rate=24.0,
            seed=1,
            stage1_steps=None,
            image=None,
            images=None,
            prompt_relay=None,
            generated_keyframes=0,
            enable_teacache=False,
        )
    (cond,) = seen["conditionings"]
    assert cond.frame_indices == [0, 1, 2, 3]
    assert cond.clean_latent is pipe._carry.video_half


# --- image anchors on the new frames --------------------------------------------------------------------------


def test_image_anchors_count_the_new_frames_and_move_past_the_carry(stubbed):
    """--image FRAME counts appended frames (0 = first new frame, -1 = last); the window puts them after the carry."""
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    pipe, record, _ = stubbed
    images = [
        ImageConditioningInput.at("a.png", 0, 0.7),
        ImageConditioningInput.at("b.png", -1, 0.7),
        ImageConditioningInput.at("c.png", 10, 1.0, 0),
    ]
    pipe.extend_from_video("x", "src.mp4", extend_frames=3, images=images)

    # 3 new latent frames = 24 new pixel frames, window = 25 carried + 24 = 49 frames.
    passed = record["stage1"]["images"]
    assert [(i.path, i.frame_idx, i.strength, i.crf) for i in passed] == [
        ("a.png", 25, 0.7, None),
        ("b.png", 48, 0.7, None),
        ("c.png", 35, 1.0, 0),
    ]
    assert record["stage1"]["num_frames"] == 49


def test_without_images_stage1_gets_none(stubbed):
    """No anchors: _stage1 gets images=None, exactly the call the pipeline made before anchors existed."""
    pipe, record, _ = stubbed
    pipe.extend_from_video("x", "src.mp4", extend_frames=3, images=[])
    assert record["stage1"]["images"] is None


@pytest.mark.parametrize("frame", [24, -25])
def test_image_anchor_outside_the_new_frames_fails_before_any_encode(stubbed, frame):
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    pipe, record, _ = stubbed
    with pytest.raises(ValueError, match="frame index"):
        pipe.extend_from_video("x", "src.mp4", extend_frames=3, images=[ImageConditioningInput.at("a.png", frame, 0.7)])
    assert record["encoded"] == []


def test_cli_extend_distilled_forwards_image_anchors(monkeypatch, tmp_path):
    from ltx_pipelines_mlx import cli

    received: dict = {}

    class _FakePipe:
        def __init__(self, **kwargs) -> None:
            pass

        def extend_from_video(self, **kwargs):
            received.update(kwargs)
            return None, None

    monkeypatch.setattr(ext_mod, "ExtendDistilledPipeline", _FakePipe)
    monkeypatch.setattr(cli, "_decode_and_save", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_print_result", lambda *a, **k: None)
    cli._cmd_extend(
        _extend_args(tmp_path, "--distilled", "--image", "face.png", "last", "0.7", "--image", "f2.png", "40", "1.0")
    )

    assert [(i.path, i.frame_idx, i.strength) for i in received["images"]] == [
        ("face.png", -1, 0.7),
        ("f2.png", 40, 1.0),
    ]


def test_cli_dev_extend_rejects_image_anchors(monkeypatch, tmp_path):
    from ltx_pipelines_mlx import cli
    from ltx_pipelines_mlx import retake as retake_mod

    def must_not_build(**kwargs):
        raise AssertionError("--image without --distilled must be rejected before anything is built")

    monkeypatch.setattr(retake_mod, "RetakePipeline", must_not_build)
    with pytest.raises(SystemExit, match="--image needs --distilled"):
        cli._cmd_extend(_extend_args(tmp_path, "--image", "face.png", "last", "0.7"))


# --- source audio that does not span the video -------------------------------------------------------------


def _ramp(tokens: int) -> mx.array:
    return mx.broadcast_to(mx.arange(1, tokens + 1).reshape(1, 1, tokens, 1).astype(mx.bfloat16), (1, 8, tokens, 16))


def test_fit_audio_latent_pads_trims_and_keeps_exact():
    audio = _ramp(40)
    padded = fit_audio_latent(audio, 51)
    assert padded.shape == (1, 8, 51, 16) and padded.dtype == audio.dtype
    assert mx.array_equal(padded[:, :, :40], audio).item() and mx.all(padded[:, :, 40:] == 0).item()
    assert mx.array_equal(fit_audio_latent(_ramp(60), 51), _ramp(51)).item()
    assert fit_audio_latent(audio, 40) is audio


@pytest.mark.parametrize("have", [40, 10, 60])
def test_a_source_audio_that_does_not_span_the_video_is_fitted_before_the_carry_and_the_append(
    stubbed, monkeypatch, have
):
    pipe, record, _ = stubbed
    monkeypatch.setattr(ext_mod, "encode_source_audio_latent", lambda *a, **k: _ramp(have))
    expected = fit_audio_latent(_ramp(have), 51)

    _, audio = pipe.extend_from_video("she smiles", "src.mp4", extend_frames=3, seed=7, stage1_steps=4)

    # The carry is the last 26 tokens of the video-length track, not of the (shorter or longer) encoded one.
    assert mx.array_equal(record["carry_during_stage1"].audio[0, :, 0], expected[0, 0, -26:, 0]).item()
    # The appended window audio starts right after the video-length source audio.
    assert audio.shape == (1, 8, 51 + 51 - 26, 16)
    assert mx.array_equal(audio[:, :, :51], expected).item()
    assert mx.all(audio[:, :, 51:] == -1).item()
