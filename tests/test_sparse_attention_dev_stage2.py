"""Sol sparse attention (LTX2_SOL_TAU) in the dev two-stage stage 2: --two-stage, --two-stages-hq (res_2s), a2v."""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from ltx_core_mlx.model.transformer.sparse_attention import SparseAttentionState
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput, res2s_evaluation_sigmas

NVIDIA_TAUS = (1.0, 1.25, 1.5)


class _RecordingDit:
    def __init__(self):
        self.sparse_attention: SparseAttentionState | None = None
        self.history: list[SparseAttentionState | None] = []

    def set_sparse_attention(self, state):
        self.sparse_attention = state
        self.history.append(state)


def _stubbed(cls, module, tmp_path, monkeypatch, taus: str | None, loops: tuple[str, ...]):
    if taus is None:
        monkeypatch.delenv("LTX2_SOL_TAU", raising=False)
    else:
        monkeypatch.setenv("LTX2_SOL_TAU", taus)
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    pipe = cls(str(tmp_path), low_memory=False)
    pipe._encode_text_with_negative = lambda *a, **k: (  # type: ignore[method-assign]
        mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
        mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
    )
    pipe._load_vae_encoder = lambda: None  # type: ignore[method-assign]
    pipe._fuse_distilled_lora = lambda dit: None  # type: ignore[method-assign]
    pipe.dit = dit = _RecordingDit()  # type: ignore[assignment]

    class _Vae:
        def denormalize_latent(self, x):
            return x

        def normalize_latent(self, x):
            return x

    pipe.vae_encoder = _Vae()  # type: ignore[assignment]
    pipe.upsampler = lambda x: mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)  # type: ignore[assignment]
    monkeypatch.setattr(module, "X0Model", lambda m: m)
    seen: list[SparseAttentionState | None] = []

    def loop(*args, **kwargs):
        seen.append(dit.sparse_attention)
        return DenoiseOutput(video_latent=kwargs["video_state"].latent, audio_latent=kwargs["audio_state"].latent)

    for name in loops:
        monkeypatch.setattr(module, name, loop)
    return pipe, dit, seen


def _two_stage(tmp_path, monkeypatch, taus):
    from ltx_pipelines_mlx import ti2vid_two_stages as mod

    return _stubbed(
        mod.TI2VidTwoStagesPipeline, mod, tmp_path, monkeypatch, taus, ("guided_denoise_loop", "denoise_loop")
    )


def _hq(tmp_path, monkeypatch, taus):
    from ltx_pipelines_mlx import ti2vid_two_stages_hq as mod

    return _stubbed(mod.TI2VidTwoStagesHQPipeline, mod, tmp_path, monkeypatch, taus, ("res2s_denoise_loop",))


def _generate(pipe):
    return pipe.generate_two_stage(
        prompt="a fox", height=128, width=128, num_frames=9, frame_rate=24.0, seed=7, stage1_steps=2
    )


def test_two_stage_turns_it_on_for_stage_2_only(tmp_path, monkeypatch):
    from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS

    pipe, dit, seen = _two_stage(tmp_path, monkeypatch, "1.0,1.25,1.5")
    _generate(pipe)
    stage_1, stage_2 = seen
    assert stage_1 is None
    assert stage_2.taus == NVIDIA_TAUS and stage_2.sigmas == tuple(STAGE_2_SIGMAS[:-1])
    assert dit.history == [stage_2, None]


def test_hq_routes_both_res2s_evaluations_of_a_step(tmp_path, monkeypatch):
    from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS

    pipe, dit, seen = _hq(tmp_path, monkeypatch, "1.0,1.25,1.5")
    _generate(pipe)
    stage_1, stage_2 = seen
    assert stage_1 is None
    evaluations = res2s_evaluation_sigmas(list(STAGE_2_SIGMAS))
    for sigma, step in evaluations:
        assert stage_2.tau_for_sigma(sigma) == NVIDIA_TAUS[min(step, 2)]
    assert dit.history == [stage_2, None]


@pytest.mark.parametrize("make", [_two_stage, _hq])
def test_default_never_touches_the_dit(make, tmp_path, monkeypatch):
    pipe, dit, seen = make(tmp_path, monkeypatch, None)
    _generate(pipe)
    assert seen == [None, None] and dit.history == []


def test_res2s_evaluation_sigmas_match_the_loop():
    """The helper lists exactly the sigmas res2s_denoise_loop evaluates the model at."""
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState
    from ltx_pipelines_mlx.utils.samplers import res2s_denoise_loop

    sigmas = [0.909375, 0.725, 0.421875, 0.0]
    seen: list[float] = []

    class _Model:
        def __call__(self, video_latent, audio_latent, sigma, **kwargs):  # X0Model signature
            seen.append(float(sigma.reshape(-1)[0].item()))
            return video_latent, audio_latent

    def state(n):
        latent = mx.zeros((1, n, 8))
        return LatentState(latent=latent, clean_latent=latent, denoise_mask=mx.ones((1, n, 1)))

    res2s_denoise_loop(
        _Model(),
        state(4),
        state(2),
        video_text_embeds=mx.zeros((1, 2, 8)),
        audio_text_embeds=mx.zeros((1, 2, 8)),
        sigmas=sigmas,
        bongmath=False,
        show_progress=False,
    )
    evaluations = res2s_evaluation_sigmas(sigmas)
    assert len(seen) == len(evaluations)
    # The loop passes bfloat16-rounded sigmas; SparseAttentionState matches within 1e-2 for that reason.
    assert all(abs(a - b) < 1e-2 for a, (b, _) in zip(seen, evaluations, strict=True))
    state = SparseAttentionState(
        taus=tuple(NVIDIA_TAUS[min(step, 2)] for _, step in evaluations), sigmas=tuple(s for s, _ in evaluations)
    )
    assert [state.tau_for_sigma(a) for a in seen] == [NVIDIA_TAUS[min(step, 2)] for _, step in evaluations]
