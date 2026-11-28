"""ic-lora's stage 2 runs on the reloaded clean transformer, not on stage 1's (with the IC-LoRA fused)."""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from ltx_pipelines_mlx import ic_lora as ic_mod
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput


class _Enc:
    def denormalize_latent(self, x):
        return x

    def normalize_latent(self, x):
        return x


def _pipe(tmp_path, monkeypatch):
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    pipe = ic_mod.ICLoraPipeline(model_dir=str(tmp_path), low_memory=False)
    calls: list = []

    def loop(**kw):
        calls.append(kw["model"])
        return DenoiseOutput(video_latent=kw["video_state"].latent, audio_latent=kw["audio_state"].latent)

    def load():
        pipe.dit = "stage-1 transformer (IC-LoRA fused)"
        pipe.upsampler = lambda x: mx.repeat(mx.repeat(x, 2, axis=3), 2, axis=4)

    monkeypatch.setattr(ic_mod, "denoise_loop", loop)
    monkeypatch.setattr(ic_mod, "X0Model", lambda dit: ("x0", dit))
    monkeypatch.setattr(type(pipe), "vae_encoder", property(lambda self: _Enc()))
    pipe._load_text_encoder = lambda: None
    pipe._encode_text = lambda prompt: (mx.zeros((1, 8, 4096)), mx.zeros((1, 8, 2048)))
    pipe.load = load
    pipe._fuse_loras = lambda: None
    pipe._create_conditionings = lambda **kw: []
    pipe._pre_denoise_flush = lambda *a: None
    return pipe, calls


def test_stage_2_uses_the_reloaded_transformer(tmp_path, monkeypatch):
    pipe, calls = _pipe(tmp_path, monkeypatch)

    def reload():
        pipe.dit = "clean transformer"

    pipe._reload_clean_transformer = reload
    pipe.generate("a fox", [], height=128, width=128, num_frames=17, frame_rate=24.0, seed=1)
    assert calls == [("x0", "stage-1 transformer (IC-LoRA fused)"), ("x0", "clean transformer")]


@pytest.mark.parametrize("kw", [dict(upsample_only=True, refine_steps=2)])
def test_control_aware_refine_keeps_the_stage_1_model(tmp_path, monkeypatch, kw):
    pipe, calls = _pipe(tmp_path, monkeypatch)

    def reload():
        raise AssertionError("the control-aware refine keeps the IC-LoRA: no reload")

    pipe._reload_clean_transformer = reload
    monkeypatch.setattr(ic_mod, "append_ic_lora_reference_video_conditionings", lambda conds, *a, **k: None)
    pipe.generate("a fox", [], height=128, width=128, num_frames=17, frame_rate=24.0, seed=1, **kw)
    assert calls == [("x0", "stage-1 transformer (IC-LoRA fused)")] * 2


def test_upsampler_resolves_the_25_file(tmp_path):
    """A 2.5 pack ships only ``spatial_upscaler_x2_v1_0``: ic-lora loaded ``v1_1`` and, finding none,
    silently upscaled with an untrained module (stage 2 decoded to a mosaic grid)."""
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    (tmp_path / "spatial_upscaler_x2_v1_0.safetensors").touch()
    pipe = ic_mod.ICLoraPipeline(model_dir=str(tmp_path), low_memory=False)
    assert pipe._resolve_upsampler_path().name == "spatial_upscaler_x2_v1_0.safetensors"


def _pipe_without_upscaler(tmp_path, monkeypatch):
    """``_pipe`` on a pack with no upscaler file, where ``load()`` (as the real one) leaves the upsampler out."""
    pipe, calls = _pipe(tmp_path, monkeypatch)
    pipe.load = lambda: setattr(pipe, "dit", "stage-1 transformer (IC-LoRA fused)")
    monkeypatch.setattr(ic_mod, "append_ic_lora_reference_video_conditionings", lambda conds, *a, **k: None)
    return pipe, calls


@pytest.mark.parametrize("kw", [{"single_stage": True}, {"skip_stage_2": True}])
def test_no_upscaler_needed_without_stage_2(tmp_path, monkeypatch, kw):
    """single_stage and skip_stage_2 never upscale: they must run on a pack that has no upscaler file."""
    pipe, calls = _pipe_without_upscaler(tmp_path, monkeypatch)
    pipe.generate("a fox", [], height=128, width=128, num_frames=17, frame_rate=24.0, seed=1, **kw)
    assert len(calls) == 1
    assert pipe.upsampler is None


def test_missing_upscaler_fails_before_stage_1(tmp_path, monkeypatch):
    """A two-stage run on a pack without the upscaler fails loud, and before spending stage 1."""
    pipe, calls = _pipe_without_upscaler(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="Spatial upsampler weights not found"):
        pipe.generate("a fox", [], height=128, width=128, num_frames=17, frame_rate=24.0, seed=1)
    assert calls == []


def test_load_leaves_the_upsampler_out(tmp_path, monkeypatch):
    """``load()`` itself must not need the upscaler file."""
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    pipe = ic_mod.ICLoraPipeline(model_dir=str(tmp_path), low_memory=False)
    monkeypatch.setattr(pipe, "_load_transformer_with_optional_streaming", lambda path: "dit")
    monkeypatch.setattr(pipe, "_load_vae_encoder", lambda: None)
    pipe.load()
    assert pipe.upsampler is None


@pytest.mark.parametrize(
    ("is_25", "repo", "name"),
    [
        (True, "dgrauet/ltx-2.5-mlx-q8", "spatial_upscaler_x2_v1_0.safetensors"),
        (False, "dgrauet/ltx-2.3-mlx-q8", "spatial_upscaler_x2_v1_1.safetensors"),
    ],
)
def test_missing_upscaler_message_names_the_pack_repo(tmp_path, is_25, repo, name):
    """The download hint must point at the repo that has the expected file."""
    config = {"transformer": {"num_layers": 48, "ff_bias": False}} if is_25 else {}
    (tmp_path / "embedded_config.json").write_text(json.dumps(config))
    pipe = ic_mod.ICLoraPipeline(model_dir=str(tmp_path), low_memory=False)
    assert pipe._is_25 is is_25
    with pytest.raises(FileNotFoundError, match=f"hf download {repo} {name}"):
        pipe._resolve_upsampler_path()
