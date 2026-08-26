"""is_ltx25_pack detection + ancestral sampler constants.

The real-pack tests are gated behind LTX25_Q8_DIR / MODEL_DIR (see
tests/conftest.py) since they require locally converted weight packs.
The synthetic tests exercise the same code path (LTXModelConfig.from_checkpoint_dir)
against tmp_path directories shaped like a 2.5 config, a 2.3 config, and no
config at all, so the contract is pinned even on machines without the real
packs.
"""

import json

import pytest

from ltx_pipelines_mlx.distilled import (
    ANCESTRAL_ETA,
    ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_S_NOISE,
)
from ltx_pipelines_mlx.utils.generation import is_ltx25_pack
from tests.conftest import LTX25_Q8_DIR, MODEL_DIR

skip_no_25_weights = pytest.mark.skipif(LTX25_Q8_DIR is None, reason="ltx-2.5-mlx-q8 pack not found")
skip_no_23_weights = pytest.mark.skipif(MODEL_DIR is None, reason="q8 weights not found")


def test_ancestral_constants_exact_values():
    assert ANCESTRAL_ETA == 1.0
    assert ANCESTRAL_S_NOISE == 1.0
    assert ANCESTRAL_NOISE_SEED_OFFSET == 10000


@pytest.mark.slow
@skip_no_25_weights
def test_is_ltx25_pack_true_on_real_25_pack():
    assert is_ltx25_pack(LTX25_Q8_DIR) is True


@pytest.mark.slow
@skip_no_23_weights
def test_is_ltx25_pack_false_on_real_23_pack():
    assert is_ltx25_pack(MODEL_DIR) is False


def test_is_ltx25_pack_true_on_synthetic_25_config(tmp_path):
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    assert is_ltx25_pack(tmp_path) is True


def test_is_ltx25_pack_false_on_synthetic_23_config(tmp_path):
    (tmp_path / "embedded_config.json").write_text(
        json.dumps({"transformer": {"num_layers": 48, "av_ca_timestep_scale_multiplier": 1000.0}})
    )
    assert is_ltx25_pack(tmp_path) is False


def test_is_ltx25_pack_false_when_no_config_present(tmp_path):
    # LTXModelConfig.from_checkpoint_dir finds neither embedded_config.json
    # nor config.json: it warns on stderr and returns the hardcoded
    # defaults, where ff_bias=True (2.3-shaped) -> is_ltx25_pack is False.
    # Must not raise.
    assert is_ltx25_pack(tmp_path) is False


# --------------------------------------------------------------------------
# Task 2: 2.5 routing inside DistilledPipeline.generate_two_stage
# --------------------------------------------------------------------------
#
# The routing tests drive the real ``generate_two_stage`` at toy resolution
# (128x128x9 -> 2x2x2 latent tokens) with the heavyweight collaborators
# stubbed: text encoder, DiT, VAE encoder, upsampler and both denoising
# loops. Everything else (dimension snapping, patchify/unpatchify, position
# computation, ``create_noised_state``) runs for real, so the assertions pin
# the actual call structure rather than a mock of it.

import mlx.core as mx  # noqa: E402

from ltx_pipelines_mlx import distilled as distilled_mod  # noqa: E402
from ltx_pipelines_mlx.distilled import DistilledPipeline  # noqa: E402
from ltx_pipelines_mlx.scheduler import (  # noqa: E402
    DISTILLED_SIGMAS,
    LTX_2_5_DISTILLED_SIGMAS,
    LTX_2_5_STAGE_2_DISTILLED_SIGMAS,
    STAGE_2_SIGMAS,
)
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput  # noqa: E402


def _write_pack_config(tmp_path, *, ltx25: bool):
    """Write a minimal transformer config marking the dir as a 2.5 / 2.3 pack."""
    transformer: dict = {"num_layers": 48}
    if ltx25:
        transformer["ff_bias"] = False
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))
    return tmp_path


class _LoopSpy:
    """Stand-in for a denoising loop: records kwargs, echoes the input latents."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent,
            audio_latent=kwargs["audio_state"].latent,
        )


class _FakeVaeEncoder:
    """Identity latent (de)normalization — the upscale path only needs shapes."""

    def denormalize_latent(self, x):
        return x

    def normalize_latent(self, x):
        return x


def _fake_upsampler(x):
    """2x nearest-neighbour spatial upscale on a (B, C, F, H, W) latent."""
    return mx.repeat(mx.repeat(x, 2, axis=3), 2, axis=4)


def _make_stubbed_pipeline(tmp_path, monkeypatch, *, ltx25: bool):
    """Build a DistilledPipeline over a synthetic pack with stubbed collaborators."""
    _write_pack_config(tmp_path, ltx25=ltx25)
    pipe = DistilledPipeline(str(tmp_path), low_memory=False)

    pipe._load_text_encoder = lambda: None  # type: ignore[method-assign]
    pipe._encode_text = lambda prompt: (  # type: ignore[method-assign]
        mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
    )
    pipe.load = lambda: None  # type: ignore[method-assign]
    pipe.dit = object()  # type: ignore[assignment]
    pipe.vae_encoder = _FakeVaeEncoder()  # type: ignore[assignment]
    pipe.upsampler = _fake_upsampler  # type: ignore[assignment]

    monkeypatch.setattr(distilled_mod, "X0Model", lambda dit: dit)

    euler = _LoopSpy()
    ancestral = _LoopSpy()
    monkeypatch.setattr(distilled_mod, "denoise_loop", euler)
    monkeypatch.setattr(distilled_mod, "euler_ancestral_denoising_loop", ancestral)

    noised_calls: list[dict] = []
    real_create_noised_state = distilled_mod.create_noised_state

    def spy_create_noised_state(**kwargs):
        noised_calls.append(kwargs)
        return real_create_noised_state(**kwargs)

    monkeypatch.setattr(distilled_mod, "create_noised_state", spy_create_noised_state)

    return pipe, euler, ancestral, noised_calls


def _run(pipe, **overrides):
    kwargs = dict(prompt="a fox", height=128, width=128, num_frames=9, frame_rate=24.0, seed=7)
    kwargs.update(overrides)
    return pipe.generate_two_stage(**kwargs)


def test_25_pack_routes_stage1_through_ancestral_loop(tmp_path, monkeypatch):
    pipe, euler, ancestral, _ = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=True)

    _run(pipe)

    assert len(ancestral.calls) == 1
    stage_1 = ancestral.calls[0]
    assert stage_1["sigmas"] is LTX_2_5_DISTILLED_SIGMAS, (
        "2.3/2.5 tables are value-identical; identity proves the 2.5 selection ran"
    )
    assert stage_1["noise_seed"] == 7 + ANCESTRAL_NOISE_SEED_OFFSET
    assert stage_1["stepper"].eta == ANCESTRAL_ETA
    assert stage_1["stepper"].s_noise == ANCESTRAL_S_NOISE
    assert len(euler.calls) == 1


def test_25_pack_stage2_stays_deterministic(tmp_path, monkeypatch):
    """Upstream: "Stage 2 is always deterministic -- its 3-step refinement
    schedule is too short to remove freshly injected noise." The ancestral
    override is scoped to stage 1 (``_stage_1_sampler_kwargs``), so a 2.5 pack
    must run its stage 2 on the plain Euler loop with the 2.5 stage-2 table.
    """
    pipe, euler, ancestral, _ = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=True)

    _run(pipe)

    assert len(euler.calls) == 1
    stage_2 = euler.calls[0]
    assert stage_2["sigmas"] is LTX_2_5_STAGE_2_DISTILLED_SIGMAS, (
        "identity, not equality: the 2.3 table is value-identical"
    )
    # The load-bearing assertion: no ancestral machinery reaches stage 2.
    assert "stepper" not in stage_2
    assert "noise_seed" not in stage_2
    assert all(call["sigmas"] is not LTX_2_5_STAGE_2_DISTILLED_SIGMAS for call in ancestral.calls)


def test_25_pack_stage2_renoises_at_first_stage2_sigma(tmp_path, monkeypatch):
    pipe, _, _, noised_calls = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=True)

    _run(pipe)

    # 4 calls: stage-1 video/audio (sigma 1.0), stage-2 video/audio.
    assert [c["sigma"] for c in noised_calls[:2]] == [1.0, 1.0]
    assert [c["sigma"] for c in noised_calls[2:]] == [
        LTX_2_5_STAGE_2_DISTILLED_SIGMAS[0],
        LTX_2_5_STAGE_2_DISTILLED_SIGMAS[0],
    ]


def test_23_pack_keeps_the_deterministic_euler_loop(tmp_path, monkeypatch):
    pipe, euler, ancestral, noised_calls = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=False)

    _run(pipe)

    assert ancestral.calls == []
    assert len(euler.calls) == 2
    assert euler.calls[0]["sigmas"] == DISTILLED_SIGMAS
    assert euler.calls[1]["sigmas"] == STAGE_2_SIGMAS
    for call in euler.calls:
        assert "stepper" not in call
        assert "noise_seed" not in call
    assert [c["sigma"] for c in noised_calls[2:]] == [STAGE_2_SIGMAS[0], STAGE_2_SIGMAS[0]]


def test_stage_step_truncation_applies_to_the_25_tables(tmp_path, monkeypatch):
    pipe, euler, ancestral, _ = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=True)

    _run(pipe, stage1_steps=3, stage2_steps=2)

    assert ancestral.calls[0]["sigmas"] == LTX_2_5_DISTILLED_SIGMAS[:4]
    assert euler.calls[0]["sigmas"] == LTX_2_5_STAGE_2_DISTILLED_SIGMAS[:3]


def test_teacache_rejected_on_25_pack(tmp_path, monkeypatch):
    pipe, euler, ancestral, _ = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=True)

    with pytest.raises(ValueError, match=r"TeaCache is not calibrated for LTX-2\.5"):
        _run(pipe, enable_teacache=True)

    # Guard fires before any denoising work.
    assert euler.calls == []
    assert ancestral.calls == []


def test_teacache_flag_still_ignored_on_23_pack(tmp_path, monkeypatch):
    pipe, euler, _, _ = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=False)

    _run(pipe, enable_teacache=True)

    assert len(euler.calls) == 2


# --- upsampler resolution ---------------------------------------------------


def _upsampler_pipeline(tmp_path, *, ltx25: bool, files: list[str]):
    _write_pack_config(tmp_path, ltx25=ltx25)
    for name in files:
        (tmp_path / name).touch()
    return DistilledPipeline(str(tmp_path), low_memory=False)


def test_upsampler_resolves_v1_0_on_25_pack(tmp_path):
    pipe = _upsampler_pipeline(tmp_path, ltx25=True, files=["spatial_upscaler_x2_v1_0.safetensors"])
    assert pipe._resolve_upsampler_path().name == "spatial_upscaler_x2_v1_0.safetensors"


def test_upsampler_falls_back_to_v1_1(tmp_path):
    pipe = _upsampler_pipeline(tmp_path, ltx25=True, files=["spatial_upscaler_x2_v1_1.safetensors"])
    assert pipe._resolve_upsampler_path().name == "spatial_upscaler_x2_v1_1.safetensors"


def test_upsampler_keeps_v1_1_on_23_pack(tmp_path):
    pipe = _upsampler_pipeline(tmp_path, ltx25=False, files=["spatial_upscaler_x2_v1_1.safetensors"])
    assert pipe._resolve_upsampler_path().name == "spatial_upscaler_x2_v1_1.safetensors"


def test_upsampler_missing_raises_file_not_found(tmp_path):
    pipe = _upsampler_pipeline(tmp_path, ltx25=True, files=[])
    with pytest.raises(FileNotFoundError, match="Spatial upsampler weights not found"):
        pipe._resolve_upsampler_path()


# --------------------------------------------------------------------------
# Task 3: auto-duration wiring inside DistilledPipeline.generate_two_stage
# --------------------------------------------------------------------------

from ltx_pipelines_mlx.utils.types import DEFAULT_AUTO_DURATION, AutoDuration  # noqa: E402


class _FakeDurationPredictor:
    """Records its call args and always predicts a fixed frame count."""

    def __init__(self, frames: int):
        self._frames = frames
        self.call_args: dict | None = None

    def __call__(self, video_encoding, audio_encoding, *, frame_rate, min_seconds=1.0, max_seconds=20.0):
        self.call_args = {
            "video_encoding": video_encoding,
            "audio_encoding": audio_encoding,
            "frame_rate": frame_rate,
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
        }
        return self._frames


def test_auto_duration_resolves_after_encode_on_25(tmp_path, monkeypatch):
    """A 2.5 pack with a predictor resolves AutoDuration from the (stubbed) encodings.

    18 -> latent F = (18 + 7) // 8 = 3 on the 2.5 pack's stub predictor
    (prompt encoding stub returns zeros regardless, so any int works here;
    17 is used to also pin the 8k+1-grid convention documented on DurationPredictor).
    """
    pipe, _, ancestral, noised_calls = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=True)
    fake_predictor = _FakeDurationPredictor(17)
    pipe._duration_predictor = fake_predictor  # type: ignore[assignment]

    _run(pipe, num_frames=DEFAULT_AUTO_DURATION)

    # Predictor was called with the (stubbed) positive encodings, not negatives —
    # DistilledPipeline has no CFG / negative prompt.
    assert fake_predictor.call_args is not None
    assert fake_predictor.call_args["frame_rate"] == 24.0
    assert fake_predictor.call_args["min_seconds"] == DEFAULT_AUTO_DURATION.min_seconds
    assert fake_predictor.call_args["max_seconds"] == DEFAULT_AUTO_DURATION.max_seconds

    expected_f = (17 + 7) // 8
    assert expected_f == 3
    # Stage 1 video/audio noised-state calls carry the resolved latent F.
    assert noised_calls[0]["spatial_dims"][0] == expected_f
    assert noised_calls[1]["spatial_dims"][0] == expected_f
    assert len(ancestral.calls) == 1


def test_auto_duration_raises_early_on_23(tmp_path, monkeypatch):
    """A pack without a DurationHead (predictor None) must fail before any encode work."""
    pipe, euler, ancestral, _ = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=False)
    assert pipe._duration_predictor is None

    def _unreachable_encode(prompt):
        raise AssertionError("_encode_text must not be reached when the guard should fire first")

    pipe._encode_text = _unreachable_encode  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Pass num_frames explicitly"):
        _run(pipe, num_frames=AutoDuration())

    assert euler.calls == []
    assert ancestral.calls == []
