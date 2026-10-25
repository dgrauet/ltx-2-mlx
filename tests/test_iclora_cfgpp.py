"""CFG++ must use negative predictions and preserve visual control tokens."""

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.components.diffusion_steps import EulerCfgPpDiffusionStep
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx.cli import _build_parser
from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop


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


def state(masked=False, frozen=False):
    mask = mx.array([[[0.0], [1.0], [0.0]]]) if masked else mx.ones((1, 3, 1))
    if frozen:
        mask = mx.zeros_like(mask)
    clean = mx.full((1, 3, 4), 7.0)
    return LatentState(latent=mask * 0.8 + (1 - mask) * clean, clean_latent=clean, denoise_mask=mask, frozen=frozen)


class Model:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        # Output depends on the latent so stochastic and unconditional effects
        # remain visible after the last step, instead of a constant-x0 false pass.
        v = 0.4 * kw["video_latent"] + kw["video_text_embeds"].mean()
        a = None if kw["audio_latent"] is None else 0.4 * kw["audio_latent"] + kw["audio_text_embeds"].mean()
        return v, a


def run(model=None, seed=42, negative=-0.3, video=None, audio=True, **kwargs):
    return denoise_loop(
        model=model or Model(),
        video_state=video or state(),
        audio_state=state() if audio is True else (None if audio is False else audio),
        video_text_embeds=mx.array([0.2]),
        audio_text_embeds=mx.array([0.2]) if audio is not False else None,
        uncond_video_text_embeds=mx.array([negative]),
        uncond_audio_text_embeds=mx.array([negative]) if audio is not False else None,
        cfg_pp=EulerCfgPpDiffusionStep(),
        noise_seed=seed,
        sigmas=[1.0, 0.725, 0.421875, 0.0],
        show_progress=False,
        **kwargs,
    )


def test_unconditional_pass_is_used_even_at_cfg_one():
    m = Model()
    a = run(model=m)
    b = run(negative=-2.0)
    assert [round(c["video_text_embeds"].item(), 1) for c in m.calls] == [0.2, -0.3, 0.2, -0.3, 0.2]
    assert not mx.allclose(a.video_latent, b.video_latent).item()
    assert not mx.allclose(a.audio_latent, b.audio_latent).item()


def test_seed_reproduces_both_streams_and_changes_output():
    a, b, c = run(seed=42), run(seed=42), run(seed=43)
    for name in ("video_latent", "audio_latent"):
        assert mx.array_equal(getattr(a, name), getattr(b, name)).item()
        assert not mx.allclose(getattr(a, name), getattr(c, name)).item()
    assert not mx.array_equal(a.video_latent, a.audio_latent).item()


def test_control_tokens_and_frozen_audio_stay_clean_at_every_forward():
    m = Model()
    v, a = state(masked=True), state(frozen=True)
    out = run(model=m, video=v, audio=a)
    for call in m.calls:
        assert mx.array_equal(call["video_latent"][:, [0, 2]], v.clean_latent[:, [0, 2]]).item()
        assert mx.array_equal(call["audio_latent"], a.clean_latent).item()
        assert mx.array_equal(call["video_timesteps"][:, [0, 2]], mx.zeros((1, 2))).item()
        assert call["audio_sigma"].item() == 0.0
    assert mx.array_equal(out.video_latent[:, [0, 2]], v.clean_latent[:, [0, 2]]).item()
    assert mx.array_equal(out.audio_latent, a.clean_latent).item()


def test_video_only_and_negative_pass_without_positive_relay_mask():
    m = Model()
    out = run(model=m, audio=False, video_cross_attention_mask=mx.zeros((1, 1)))
    assert out.audio_latent is None
    assert "video_cross_attention_mask" in m.calls[0]
    assert "video_cross_attention_mask" not in m.calls[1]


def test_missing_negative_fails_before_model():
    m = Model()
    with pytest.raises(ValueError, match="negative text embeddings"):
        denoise_loop(m, state(), None, mx.zeros((1,)), None, cfg_pp=EulerCfgPpDiffusionStep())
    assert not m.calls


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
    assert _build_parser().parse_args(args).sampler == "euler"
    parsed = _build_parser().parse_args(
        args + ["--single-stage", "--sampler", "euler_ancestral_cfg_pp", "--negative-prompt", "blur"]
    )
    assert parsed.sampler == "euler_ancestral_cfg_pp"
    assert parsed.negative_prompt == "blur"


def test_pipeline_encodes_negative_and_routes_cfgpp_to_generation(tmp_path, monkeypatch):
    from ltx_pipelines_mlx import ic_lora as ic_mod
    from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

    pipe = ICLoraPipeline(model_dir=str(tmp_path), low_memory=False)
    prompts, calls = [], []

    def encode(prompt):
        prompts.append(prompt)
        value = 2.0 if prompt == "positive" else -3.0
        return mx.full((1, 8, 4), value), mx.full((1, 8, 4), value)

    def loop(**kw):
        calls.append(kw)
        return DenoiseOutput(kw["video_state"].latent, kw["audio_state"].latent)

    monkeypatch.setattr(type(pipe), "vae_encoder", property(lambda self: object()))
    monkeypatch.setattr(ic_mod, "X0Model", lambda dit: dit)
    monkeypatch.setattr(ic_mod, "denoise_loop", loop)
    pipe._load_text_encoder = lambda: None
    pipe._encode_text = encode
    pipe.load = lambda: None
    pipe.dit = object()
    pipe._fuse_loras = lambda: None
    pipe._create_conditionings = lambda **kw: []
    pipe._pre_denoise_flush = lambda *args: None
    pipe.generate(
        "positive",
        [],
        height=64,
        width=64,
        num_frames=9,
        frame_rate=24,
        single_stage=True,
        sampler="euler_ancestral_cfg_pp",
        negative_prompt="negative",
        seed=123,
    )
    assert prompts == ["positive", "negative"]
    assert len(calls) == 1
    assert isinstance(calls[0]["cfg_pp"], EulerCfgPpDiffusionStep)
    assert calls[0]["noise_seed"] == 123
    assert calls[0]["uncond_video_text_embeds"].mean().item() == -3.0
    assert calls[0]["uncond_audio_text_embeds"].mean().item() == -3.0


def test_work_estimate_counts_the_extra_negative_predictions(monkeypatch):
    from ltx_pipelines_mlx.utils import samplers

    original = samplers.StepEstimator
    counts = []

    def estimator(work, **kw):
        counts.append(work)
        return original(work, **kw)

    monkeypatch.setattr(samplers, "StepEstimator", estimator)
    run()
    assert counts[0].passes_per_step == (2, 2, 1)
    assert counts[0].forwards == 5


@pytest.mark.parametrize("sampler,negative", [("euler", ""), ("euler_ancestral_cfg_pp", "blur")])
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
            "--negative-prompt",
            negative,
        ]
    )
    _cmd_ic_lora(args)
    assert len(received) == 1
    assert received[0]["sampler"] == sampler
    assert received[0]["negative_prompt"] == negative
    assert received[0]["single_stage"] is True
