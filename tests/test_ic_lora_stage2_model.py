"""ic-lora's stage 2: which transformer it runs on and which tokens it carries.

Stage 2 runs on the reloaded clean transformer unless the IC-LoRA stays on for it (the
control-aware refine, ``stage_2_ic_lora``); tokens appended to it (image guides, the
re-encoded reference) are stripped before the unpatchify.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from ltx_pipelines_mlx import ic_lora as ic_mod
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput
from ltx_pipelines_mlx.utils.types import ImageConditioningInput


class _Enc:
    def denormalize_latent(self, x):
        return x

    def normalize_latent(self, x):
        return x

    def encode(self, video: mx.array) -> mx.array:
        b, _c, f_pix, h_pix, w_pix = video.shape
        return mx.zeros((b, 128, (f_pix - 1) // 8 + 1, h_pix // 32, w_pix // 32), dtype=mx.bfloat16)


def _pipe(tmp_path, monkeypatch):
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    pipe = ic_mod.ICLoraPipeline(model_dir=str(tmp_path), low_memory=False)
    calls: list = []

    def loop(**kw):
        calls.append(kw)
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


def _still(tmp_path, name="still.png"):
    path = tmp_path / name
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(path)
    return str(path)


def test_stage_2_uses_the_reloaded_transformer(tmp_path, monkeypatch):
    pipe, calls = _pipe(tmp_path, monkeypatch)

    def reload():
        pipe.dit = "clean transformer"

    pipe._reload_clean_transformer = reload
    pipe.generate("a fox", [], height=128, width=128, num_frames=17, frame_rate=24.0, seed=1)
    assert [c["model"] for c in calls] == [("x0", "stage-1 transformer (IC-LoRA fused)"), ("x0", "clean transformer")]


@pytest.mark.parametrize("kw", [dict(upsample_only=True, refine_steps=2)])
def test_control_aware_refine_keeps_the_stage_1_model(tmp_path, monkeypatch, kw):
    pipe, calls = _pipe(tmp_path, monkeypatch)

    def reload():
        raise AssertionError("the control-aware refine keeps the IC-LoRA: no reload")

    pipe._reload_clean_transformer = reload
    monkeypatch.setattr(ic_mod, "append_ic_lora_reference_video_conditionings", lambda conds, *a, **k: None)
    pipe.generate("a fox", [], height=128, width=128, num_frames=17, frame_rate=24.0, seed=1, **kw)
    assert [c["model"] for c in calls] == [("x0", "stage-1 transformer (IC-LoRA fused)")] * 2


def test_stage_2_strips_image_guide_tokens_before_unpatchify(tmp_path, monkeypatch):
    """A guide appends keyframe tokens to stage 2; the output keeps the generated grid only."""
    pipe, calls = _pipe(tmp_path, monkeypatch)
    pipe._reload_clean_transformer = lambda: None
    guide = ImageConditioningInput.guide(_still(tmp_path), 8, 1.0, crf=0)

    video, _ = pipe.generate("a fox", [], height=128, width=128, num_frames=17, frame_rate=24.0, images=[guide])

    # 128 px -> 4x4 latent cells, F=3: 48 generated tokens plus one 16-token guide frame.
    assert calls[1]["video_state"].latent.shape == (1, 64, 128)
    assert video.shape == (1, 128, 3, 4, 4)
