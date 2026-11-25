"""IC-LoRA ``--sampler euler_ancestral_cfg_pp``: stepper equations, pipeline dispatch and CLI wiring.

The loop itself (negative pass at CFG 1, seeded noise, re-pinned conditioning tokens) is covered with
``euler_cfg_pp_denoising_loop`` in ``tests/test_euler_cfg_pp.py``."""

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.components.diffusion_steps import EulerCfgPpDiffusionStep
from ltx_pipelines_mlx.cli import _build_parser
from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
from ltx_pipelines_mlx.utils.constants import DEFAULT_NEGATIVE_PROMPT


@pytest.mark.parametrize("sigma,next_sigma", [(1.0, 0.99375), (0.975, 0.909375), (0.725, 0.421875), (0.421875, 0.0)])
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_cfgpp_matches_independent_comfy_flow_equations(sigma, next_sigma, eta):
    # Comfy sample_euler_ancestral_cfg_pp: rescaled DDIM variance, not the
    # ordinary rectified-flow Euler ancestral downstep ratio.
    x, positive, negative, noise = 0.8, 0.2, -0.3, 0.7
    sigma, next_sigma = np.float32(sigma), np.float32(next_sigma)
    a_s, a_t = max(np.float32(1) - sigma, np.float32(1e-7)), max(np.float32(1) - next_sigma, np.float32(1e-7))
    s, t = sigma / a_s, next_sigma / a_t
    up = min(t, eta * np.sqrt(t * t * max(s * s - t * t, np.float32(0)) / (s * s)))
    down = np.sqrt(max(t * t - up * up, np.float32(0)))
    expected = a_t * positive + a_t * down * (x - a_s * negative) / sigma + a_t * noise * up
    actual = EulerCfgPpDiffusionStep(eta=eta).step(
        mx.array([x]),
        mx.array([positive]),
        mx.array([float(sigma), float(next_sigma)]),
        0,
        mx.array([negative]),
        noise=mx.array([noise]),
    )
    assert actual.item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"sampler": "other"}, "Unsupported"),
        ({"sampler": "euler_ancestral_cfg_pp"}, "single_stage"),
        ({"negative_prompt": "bad"}, "negative_prompt"),
    ],
)
def test_pipeline_rejects_unsupported_combinations_before_loading(options, message):
    pipe = object.__new__(ICLoraPipeline)
    with pytest.raises(ValueError, match=message):
        pipe.generate("scene", [], frame_rate=24, **options)


def test_cli_explicit_sampler_and_unchanged_default():
    args = [
        "ic-lora",
        "--prompt",
        "scene",
        "--output",
        "unused.mp4",
        "--lora",
        "control.safetensors",
        "1",
        "--video-conditioning",
        "motion.mp4",
        "1",
        "--frame-rate",
        "24",
    ]
    default = _build_parser().parse_args(args)
    assert (default.sampler, default.negative_prompt) == ("euler", None)
    parsed = _build_parser().parse_args(
        args + ["--single-stage", "--sampler", "euler_ancestral_cfg_pp", "--negative-prompt", "blur"]
    )
    assert parsed.sampler == "euler_ancestral_cfg_pp"
    assert parsed.negative_prompt == "blur"


def _routed_pipeline(tmp_path, monkeypatch):
    """An ``ICLoraPipeline`` whose model parts are stubbed; records prompts and the loop each stage-1 run takes."""
    from ltx_pipelines_mlx import ic_lora as ic_mod
    from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

    pipe = ICLoraPipeline(model_dir=str(tmp_path), low_memory=False)
    prompts, calls = [], []

    def encode(prompt):
        prompts.append(prompt)
        value = 2.0 if prompt == "positive" else -3.0
        return mx.full((1, 8, 4), value), mx.full((1, 8, 4), value)

    def euler(**kw):
        calls.append(("euler", kw))
        return DenoiseOutput(kw["video_state"].latent, kw["audio_state"].latent)

    def cfg_pp(sigmas, video_state, audio_state, stepper, transformer, v_pos, a_pos, v_neg, a_neg, **kw):
        calls.append(("cfg_pp", dict(kw, stepper=stepper, positive=(v_pos, a_pos), negative=(v_neg, a_neg))))
        return DenoiseOutput(video_state.latent, audio_state.latent)

    monkeypatch.setattr(type(pipe), "vae_encoder", property(lambda self: object()))
    monkeypatch.setattr(ic_mod, "X0Model", lambda dit: dit)
    monkeypatch.setattr(ic_mod, "denoise_loop", euler)
    monkeypatch.setattr(ic_mod, "euler_cfg_pp_denoising_loop", cfg_pp)
    pipe._load_text_encoder = lambda: None
    pipe._encode_text = encode
    pipe.load = lambda: None
    pipe.dit = object()
    pipe._fuse_loras = lambda: None
    pipe._create_conditionings = lambda **kw: []
    pipe._pre_denoise_flush = lambda *args: None
    return pipe, prompts, calls


_RUN = dict(height=64, width=64, num_frames=9, frame_rate=24, single_stage=True, seed=123)


def test_cfgpp_runs_the_upstream_loop_on_the_encoded_negative(tmp_path, monkeypatch):
    pipe, prompts, calls = _routed_pipeline(tmp_path, monkeypatch)
    pipe.generate("positive", [], sampler="euler_ancestral_cfg_pp", negative_prompt="negative", **_RUN)
    assert prompts == ["positive", "negative"]
    assert [name for name, _ in calls] == ["cfg_pp"]
    kw = calls[0][1]
    assert isinstance(kw["stepper"], EulerCfgPpDiffusionStep)
    assert kw["noise_seed"] == 123
    assert [e.mean().item() for e in kw["positive"]] == [2.0, 2.0]
    assert [e.mean().item() for e in kw["negative"]] == [-3.0, -3.0]


def test_cfgpp_without_a_negative_prompt_uses_the_upstream_default(tmp_path, monkeypatch):
    pipe, prompts, _ = _routed_pipeline(tmp_path, monkeypatch)
    pipe.generate("positive", [], sampler="euler_ancestral_cfg_pp", **_RUN)
    assert prompts == ["positive", DEFAULT_NEGATIVE_PROMPT]


def test_default_sampler_stays_on_the_euler_loop_with_one_prompt(tmp_path, monkeypatch):
    pipe, prompts, calls = _routed_pipeline(tmp_path, monkeypatch)
    pipe.generate("positive", [], **_RUN)
    assert prompts == ["positive"]
    assert [name for name, _ in calls] == ["euler"]


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--sampler", "euler_ancestral_cfg_pp"], "needs --single-stage"),
        (["--single-stage", "--negative-prompt", "blur"], "needs --sampler"),
    ],
)
def test_cli_refuses_unsupported_combinations_before_building_the_pipeline(monkeypatch, extra, message):
    from ltx_pipelines_mlx.cli import _cmd_ic_lora

    monkeypatch.setattr(ICLoraPipeline, "__init__", lambda self, **kw: pytest.fail("pipeline was built"))
    args = _build_parser().parse_args(
        ["ic-lora", "-p", "scene", "-o", "unused.mp4", "--lora", "c.safetensors", "1"]
        + ["--video-conditioning", "motion.mp4", "1", "--frame-rate", "24", *extra]
    )
    with pytest.raises(SystemExit, match=message):
        _cmd_ic_lora(args)


@pytest.mark.parametrize("sampler,negative", [("euler", None), ("euler_ancestral_cfg_pp", "blur")])
def test_real_cli_export_wrapper_forwards_sampler_to_inference(tmp_path, monkeypatch, sampler, negative):
    from ltx_pipelines_mlx.cli import _cmd_ic_lora

    lora = tmp_path / "control.safetensors"
    mx.save_safetensors(str(lora), {"test": mx.ones((1,))})
    received = []

    def generate(self, **kwargs):
        received.append(kwargs)
        return mx.zeros((1,)), mx.zeros((1,))

    monkeypatch.setattr(ICLoraPipeline, "generate", generate)
    monkeypatch.setattr(ICLoraPipeline, "_load_decoders", lambda self: None)
    monkeypatch.setattr(ICLoraPipeline, "_decode_and_save_video", lambda self, v, a, path, **kw: path)
    args = _build_parser().parse_args(
        [
            "ic-lora",
            "--quiet",
            "--model",
            str(tmp_path),
            "--prompt",
            "scene",
            "--output",
            str(tmp_path / "out.mp4"),
            "--lora",
            str(lora),
            "1",
            "--video-conditioning",
            "motion.mp4",
            "1",
            "--frame-rate",
            "24",
            "--single-stage",
            "--sampler",
            sampler,
            *([] if negative is None else ["--negative-prompt", negative]),
        ]
    )
    _cmd_ic_lora(args)
    assert len(received) == 1
    assert received[0]["sampler"] == sampler
    assert received[0]["negative_prompt"] == negative
    assert received[0]["single_stage"] is True
