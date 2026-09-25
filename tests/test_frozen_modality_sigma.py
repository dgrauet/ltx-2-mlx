"""Frozen modality streams condition the model on sigma 0 (upstream-iso).

Upstream ``modality_from_latent_state`` forces ``Modality.sigma = 0`` for a
``LatentState`` with ``frozen=True``. That sigma drives the modality's prompt
AdaLN and the *other* modality's cross-attention gate
(``transformer_args.py``: ``prompt_timestep`` from ``modality.sigma``, gate
from ``cross_modality_sigma``). The MLX port exposes it as the optional
``video_sigma`` / ``audio_sigma`` kwargs of ``LTXModel.__call__`` (default: the
global ``timestep``), which the sampler loops pass only for frozen states.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core_mlx.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_core_mlx.conditioning.types.attention_strength_wrapper import ConditioningItemAttentionStrengthWrapper
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, VideoConditionByLatentIndex
from ltx_core_mlx.conditioning.types.reference_audio_cond import AudioConditionByReferenceLatent
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding
from ltx_core_mlx.utils.positions import compute_video_positions
from ltx_pipelines_mlx.utils.samplers import (
    denoise_loop,
    euler_ancestral_denoising_loop,
    guided_denoise_loop,
    res2s_denoise_loop,
)

F, H, W, C = 3, 2, 2, 16
N = F * H * W
TPF = H * W

# ---------------------------------------------------------------------------
# (a) LatentState.frozen and its carry-through
# ---------------------------------------------------------------------------


def _video_state(frozen: bool = False) -> LatentState:
    return LatentState(
        latent=mx.zeros((1, N, C)),
        clean_latent=mx.zeros((1, N, C)),
        denoise_mask=mx.ones((1, N, 1)),
        positions=compute_video_positions(F, H, W, frame_rate=24.0),
        frozen=frozen,
    )


def test_latent_state_frozen_defaults_false_and_survives_replace():
    state = LatentState(latent=mx.zeros((1, 2, 4)), clean_latent=mx.zeros((1, 2, 4)), denoise_mask=mx.ones((1, 2, 1)))
    assert state.frozen is False
    frozen = dataclasses.replace(state, frozen=True)
    assert dataclasses.replace(frozen, latent=mx.ones((1, 2, 4))).frozen is True


def _video_item(name: str):
    if name == "latent_index":
        return VideoConditionByLatentIndex(frame_indices=[0], clean_latent=mx.zeros((1, TPF, C)))
    if name == "keyframe":
        return VideoConditionByKeyframeIndex(
            frame_idx=9, keyframe_latent=mx.zeros((1, TPF, C)), spatial_dims=(F, H, W), frame_rate=24.0
        )
    if name == "reference":
        return VideoConditionByReferenceLatent(
            reference_latent=mx.zeros((1, TPF, C)),
            reference_positions=compute_video_positions(1, H, W, frame_rate=24.0),
            downscale_factor=1,
        )
    if name == "slots":
        return VideoGeneratedKeyframeSlots(pixel_frame_indices=[5], frame_rate=24.0)
    if name == "attention_wrapper":
        return ConditioningItemAttentionStrengthWrapper(_video_item("reference"), attention_mask=0.5)
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["latent_index", "keyframe", "reference", "slots", "attention_wrapper"])
@pytest.mark.parametrize("frozen", [False, True])
def test_video_conditionings_carry_frozen(name, frozen):
    after = _video_item(name).apply(_video_state(frozen=frozen), (F, H, W))
    assert after.frozen is frozen


@pytest.mark.parametrize("frozen", [False, True])
def test_audio_reference_conditioning_carries_frozen(frozen):
    na = 5
    state = LatentState(
        latent=mx.zeros((1, na, C)),
        clean_latent=mx.zeros((1, na, C)),
        denoise_mask=mx.ones((1, na, 1)),
        positions=mx.zeros((1, na, 1)),
        frozen=frozen,
    )
    item = AudioConditionByReferenceLatent(patchified=mx.zeros((1, 3, C)), positions=mx.zeros((1, 3, 1)))
    assert item.apply(state, num_noisy_tokens=na).frozen is frozen


# ---------------------------------------------------------------------------
# (b) LTXModel: per-modality sigma drives prompt AdaLN + the cross gates
# ---------------------------------------------------------------------------


def _tiny_config(num_layers: int = 2) -> LTXModelConfig:
    # av_ca multiplier != timestep multiplier so the gate scaling is observable.
    return LTXModelConfig(
        num_layers=num_layers,
        video_dim=32,
        audio_dim=16,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=4,
        av_cross_head_dim=4,
        video_patch_channels=8,
        audio_patch_channels=8,
        ff_mult=2.0,
        timestep_embedding_dim=32,
        av_ca_timestep_scale_multiplier=1000,
    )


def _tiny_model() -> tuple[LTXModel, LTXModelConfig]:
    cfg = _tiny_config()
    mx.random.seed(11)
    model = LTXModel(cfg)
    # Random-init AdaLN linears can be tiny; give the scalar-driven modules a
    # real signal so the sigma change is visible in the outputs.
    for name in (
        "prompt_adaln_single",
        "audio_prompt_adaln_single",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_v2a_gate_adaln_single",
    ):
        module = getattr(model, name)
        module.update(_randomize(module.parameters()))
    mx.eval(model.parameters())
    return model, cfg


def _randomize(tree):
    if isinstance(tree, mx.array):
        return mx.random.normal(tree.shape).astype(tree.dtype)
    if isinstance(tree, dict):
        return {k: _randomize(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_randomize(v) for v in tree]
    return tree


def _inputs(cfg: LTXModelConfig) -> dict:
    mx.random.seed(3)
    B, Nv, Na, Nt = 1, 8, 4, 4
    return dict(
        video_latent=mx.random.normal((B, Nv, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((B, Na, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([0.5]),
        video_text_embeds=mx.random.normal((B, Nt, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((B, Nt, cfg.audio_dim)).astype(mx.bfloat16),
    )


@pytest.fixture
def adaln_inputs(monkeypatch):
    """Record the embedding each AdaLayerNormSingle instance receives, by module id."""
    seen: dict[int, list[mx.array]] = {}
    original = AdaLayerNormSingle.__call__

    def spy(self, t_emb, *args, **kwargs):
        seen.setdefault(id(self), []).append(t_emb)
        return original(self, t_emb, *args, **kwargs)

    monkeypatch.setattr(AdaLayerNormSingle, "__call__", spy)
    return seen


def _embeds(cfg: LTXModelConfig, sigma: float) -> tuple[mx.array, mx.array]:
    """Expected (prompt, gate) embeddings for ``sigma``, scaled as upstream."""
    s = mx.array([sigma]).astype(mx.bfloat16)
    prompt = get_timestep_embedding(s * cfg.timestep_scale_multiplier, cfg.timestep_embedding_dim)
    factor = cfg.av_ca_timestep_scale_multiplier / cfg.timestep_scale_multiplier
    gate = get_timestep_embedding(s * cfg.timestep_scale_multiplier * factor, cfg.timestep_embedding_dim)
    return prompt, gate


def _only(seen: dict, module) -> mx.array:
    calls = seen[id(module)]
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize("kwarg", ["video_sigma", "audio_sigma"])
def test_default_sigma_is_the_global_timestep(kwarg):
    model, cfg = _tiny_model()
    common = _inputs(cfg)
    base_v, base_a = model(**common)
    same_v, same_a = model(**common, **{kwarg: common["timestep"]})
    mx.eval(base_v, base_a, same_v, same_a)
    assert mx.array_equal(base_v, same_v).item()
    assert mx.array_equal(base_a, same_a).item()


def test_audio_sigma_routes_to_audio_prompt_and_a2v_gate(adaln_inputs):
    model, cfg = _tiny_model()
    common = _inputs(cfg)
    model(**common, audio_sigma=mx.array([0.25]))

    glob_prompt, glob_gate = _embeds(cfg, 0.5)
    audio_prompt, audio_gate = _embeds(cfg, 0.25)
    assert mx.array_equal(_only(adaln_inputs, model.audio_prompt_adaln_single), audio_prompt).item()
    assert mx.array_equal(_only(adaln_inputs, model.av_ca_a2v_gate_adaln_single), audio_gate).item()
    assert mx.array_equal(_only(adaln_inputs, model.prompt_adaln_single), glob_prompt).item()
    assert mx.array_equal(_only(adaln_inputs, model.av_ca_v2a_gate_adaln_single), glob_gate).item()
    # The 9-param AdaLNs keep the global timestep (per-token path untouched).
    assert mx.array_equal(_only(adaln_inputs, model.audio_adaln_single), glob_prompt).item()


def test_video_sigma_routes_to_video_prompt_and_v2a_gate(adaln_inputs):
    model, cfg = _tiny_model()
    common = _inputs(cfg)
    model(**common, video_sigma=mx.array([0.25]))

    glob_prompt, glob_gate = _embeds(cfg, 0.5)
    video_prompt, video_gate = _embeds(cfg, 0.25)
    assert mx.array_equal(_only(adaln_inputs, model.prompt_adaln_single), video_prompt).item()
    assert mx.array_equal(_only(adaln_inputs, model.av_ca_v2a_gate_adaln_single), video_gate).item()
    assert mx.array_equal(_only(adaln_inputs, model.audio_prompt_adaln_single), glob_prompt).item()
    assert mx.array_equal(_only(adaln_inputs, model.av_ca_a2v_gate_adaln_single), glob_gate).item()
    assert mx.array_equal(_only(adaln_inputs, model.adaln_single), glob_prompt).item()


def test_zero_audio_sigma_changes_video_output():
    model, cfg = _tiny_model()
    common = _inputs(cfg)
    base_v, _ = model(**common)
    frozen_v, _ = model(**common, audio_sigma=mx.zeros((1,)))
    mx.eval(base_v, frozen_v)
    assert not mx.array_equal(base_v, frozen_v).item()


def test_zero_video_sigma_changes_audio_output():
    model, cfg = _tiny_model()
    common = _inputs(cfg)
    _, base_a = model(**common)
    _, frozen_a = model(**common, video_sigma=mx.zeros((1,)))
    mx.eval(base_a, frozen_a)
    assert not mx.array_equal(base_a, frozen_a).item()


# ---------------------------------------------------------------------------
# (c) Sampler loops pass the per-modality sigma for frozen states only
# ---------------------------------------------------------------------------

SIGMAS = [1.0, 0.6, 0.3, 0.0]
_TEXT = dict(video_text_embeds=mx.zeros((1, 4, 16)), audio_text_embeds=mx.zeros((1, 4, 16)))


class _SpyX0Model:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, video_latent, audio_latent, **kwargs):
        self.calls.append(kwargs)
        return mx.zeros_like(video_latent), mx.zeros_like(audio_latent)


def _state(num_tokens: int, frozen: bool) -> LatentState:
    # Frozen streams are fully preserved (mask 0), as in upstream a2v / retake.
    mask = mx.zeros if frozen else mx.ones
    return LatentState(
        latent=mx.ones((1, num_tokens, 16), dtype=mx.bfloat16),
        clean_latent=mx.ones((1, num_tokens, 16), dtype=mx.bfloat16),
        denoise_mask=mask((1, num_tokens, 1), dtype=mx.bfloat16),
        frozen=frozen,
    )


def _all_passes_factory() -> MultiModalGuiderFactory:
    return MultiModalGuiderFactory.constant(
        MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0, stg_blocks=[0], modality_scale=3.0),
        negative_context=mx.zeros((1, 4, 16)),
    )


def _run(loop: str, model: _SpyX0Model, video: LatentState, audio: LatentState) -> None:
    if loop == "euler":
        denoise_loop(model, video, audio, sigmas=SIGMAS, show_progress=False, **_TEXT)
    elif loop == "ancestral":
        euler_ancestral_denoising_loop(
            SIGMAS,
            video,
            audio,
            EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0),
            model,
            noise_seed=0,
            show_progress=False,
            **_TEXT,
        )
    elif loop == "guided":
        guided_denoise_loop(
            model, video, audio, video_guider_factory=_all_passes_factory(), sigmas=SIGMAS, show_progress=False, **_TEXT
        )
    elif loop == "res2s":
        res2s_denoise_loop(model, video, audio, sigmas=SIGMAS, bongmath=False, show_progress=False, **_TEXT)
    elif loop == "res2s_guided":
        res2s_denoise_loop(
            model,
            video,
            audio,
            sigmas=SIGMAS,
            bongmath=False,
            video_guider_factory=_all_passes_factory(),
            show_progress=False,
            **_TEXT,
        )
    else:
        raise AssertionError(loop)


LOOPS = ["euler", "ancestral", "guided", "res2s", "res2s_guided"]


@pytest.mark.parametrize("loop", LOOPS)
def test_frozen_audio_passes_zero_audio_sigma_on_every_call(loop):
    model = _SpyX0Model()
    _run(loop, model, _state(12, frozen=False), _state(3, frozen=True))
    assert model.calls
    for kwargs in model.calls:
        assert "video_sigma" not in kwargs
        sig = kwargs["audio_sigma"]
        assert sig.shape == (1,)
        assert mx.array_equal(sig, mx.zeros((1,))).item()


@pytest.mark.parametrize("loop", LOOPS)
def test_frozen_video_passes_zero_video_sigma_on_every_call(loop):
    model = _SpyX0Model()
    _run(loop, model, _state(12, frozen=True), _state(3, frozen=False))
    assert model.calls
    for kwargs in model.calls:
        assert "audio_sigma" not in kwargs
        assert mx.array_equal(kwargs["video_sigma"], mx.zeros((1,))).item()


@pytest.mark.parametrize("loop", LOOPS)
def test_non_frozen_states_pass_no_modality_sigma(loop):
    model = _SpyX0Model()
    _run(loop, model, _state(12, frozen=False), _state(3, frozen=False))
    assert model.calls
    for kwargs in model.calls:
        assert "video_sigma" not in kwargs
        assert "audio_sigma" not in kwargs


def test_guided_loop_covers_every_guidance_pass():
    """cond + uncond + ptb + mod: 4 calls per step, all carrying the frozen sigma."""
    model = _SpyX0Model()
    _run("guided", model, _state(12, frozen=False), _state(3, frozen=True))
    assert len(model.calls) == 4 * (len(SIGMAS) - 1)
    assert all("audio_sigma" in kwargs for kwargs in model.calls)


# ---------------------------------------------------------------------------
# (d) Wrappers forward the kwargs to the inner model
# ---------------------------------------------------------------------------


def test_streaming_wrapper_forwards_modality_sigma():
    from ltx_core_mlx.loader.block_streaming import BlockStreamer, StreamingLTXModel

    model, cfg = _tiny_model()
    common = _inputs(cfg)
    expected_v, expected_a = model(**common, audio_sigma=mx.zeros((1,)), video_sigma=mx.array([0.25]))

    out: dict[str, mx.array] = {}
    for i, block in enumerate(model.transformer_blocks):
        for k, v in _flatten(block.parameters(), ""):
            out[f"transformer_blocks.{i}.{k}"] = v
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "blocks.safetensors"
        mx.save_safetensors(str(path), out)
        streamer = BlockStreamer(path, block_prefix="transformer_blocks.")
        wrapped = StreamingLTXModel(model, streamer)
        # Default call first so the compiled block has been traced once already.
        wrapped(**common)
        got_v, got_a = wrapped(**common, audio_sigma=mx.zeros((1,)), video_sigma=mx.array([0.25]))
        mx.eval(expected_v, expected_a, got_v, got_a)
        assert mx.allclose(expected_v, got_v, atol=1e-5, rtol=1e-5).item()
        assert mx.allclose(expected_a, got_a, atol=1e-5, rtol=1e-5).item()
        streamer.close()


def _flatten(node, prefix: str) -> list[tuple[str, mx.array]]:
    if isinstance(node, mx.array):
        return [(prefix, node)]
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else []
    flat: list[tuple[str, mx.array]] = []
    for k, v in items:
        flat.extend(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    return flat


def test_tiled_wrapper_forwards_modality_sigma():
    from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler
    from ltx_core_mlx.model.video_vae.tiling import TileCountConfig

    calls: list[dict] = []

    class _Inner:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return mx.zeros_like(kwargs["video_latent"]), mx.zeros_like(kwargs["audio_latent"])

    Fv, Hv, Wv = 1, 1, 8
    tiler = VideoModalityTiler(TileCountConfig(), latent_shape=(Fv, Hv, Wv))
    wrapped = TiledLTXModel(_Inner(), tiler)
    audio_sigma, video_sigma = mx.zeros((1,)), mx.array([0.25])
    wrapped(
        video_latent=mx.zeros((1, Fv * Hv * Wv, 8)),
        audio_latent=mx.zeros((1, 4, 8)),
        timestep=mx.array([0.5]),
        video_text_embeds=mx.zeros((1, 4, 32)),
        audio_text_embeds=mx.zeros((1, 4, 16)),
        video_positions=compute_video_positions(Fv, Hv, Wv, frame_rate=24.0),
        audio_sigma=audio_sigma,
        video_sigma=video_sigma,
    )
    assert calls
    for kwargs in calls:
        assert kwargs["audio_sigma"] is audio_sigma
        assert kwargs["video_sigma"] is video_sigma
