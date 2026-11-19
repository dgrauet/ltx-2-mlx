"""Text K/V reuse across the guidance passes of one evaluation (``TextKVCache``).

The conditioned, STG and isolated-modality passes attend to the same text context at the
same sigma, so their text cross-attention keys and values are identical; the cache computes
them once per evaluation. These tests check the loops' outputs are bit-identical with and
without it, that the reuse really happens (and only for the positive context), and the
cases where the model must not use it.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.attention import TextKVCache
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_pipelines_mlx.utils import samplers
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop, res2s_denoise_loop, text_kv_cache_for

NUM_LAYERS = 4
STG_BLOCK = 2
SIGMAS = [1.0, 0.6, 0.25, 0.0]


def _config() -> LTXModelConfig:
    return LTXModelConfig(
        num_layers=NUM_LAYERS,
        video_dim=128,
        audio_dim=64,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=32,
        audio_head_dim=16,
        av_cross_num_heads=4,
        av_cross_head_dim=16,
        video_patch_channels=64,
        audio_patch_channels=64,
        ff_mult=2.0,
        timestep_embedding_dim=64,
    )


def _model(compute_dtype: mx.Dtype | None = None) -> LTXModel:
    mx.random.seed(0)
    model = LTXModel(_config())
    model.set_dtype(mx.bfloat16)
    for block in model.transformer_blocks:
        for name in ("scale_shift_table", "audio_scale_shift_table", "prompt_scale_shift_table"):
            block[name] = 0.05 * mx.random.normal(block[name].shape)
    mx.eval(model.parameters())
    if compute_dtype is not None:
        model.set_compute_dtype(compute_dtype)
    return model


def _state(num_tokens: int, seed: int) -> LatentState:
    mx.random.seed(seed)
    latent = mx.random.normal((1, num_tokens, 64)).astype(mx.bfloat16)
    return LatentState(
        latent=latent,
        clean_latent=mx.zeros_like(latent),
        denoise_mask=mx.ones((1, num_tokens, 1), dtype=mx.bfloat16),
    )


def _factory(dim: int, negative: bool) -> MultiModalGuiderFactory:
    return MultiModalGuiderFactory.constant(
        MultiModalGuiderParams(
            cfg_scale=3.0, stg_scale=1.0, stg_blocks=[STG_BLOCK], rescale_scale=0.7, modality_scale=3.0
        ),
        negative_context=mx.zeros((1, 6, dim), dtype=mx.bfloat16) if negative else None,
    )


def _run(loop: str, model: X0Model, negative: bool = True):
    cfg = model.model.config
    mx.random.seed(2)
    text = dict(
        video_text_embeds=mx.random.normal((1, 6, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, 6, cfg.audio_dim)).astype(mx.bfloat16),
    )
    factories = dict(
        video_guider_factory=_factory(cfg.video_dim, negative),
        audio_guider_factory=_factory(cfg.audio_dim, negative),
    )
    video, audio = _state(48, 3), _state(8, 4)
    if loop == "guided":
        out = guided_denoise_loop(model, video, audio, sigmas=SIGMAS, show_progress=False, **factories, **text)
    else:
        out = res2s_denoise_loop(
            model, video, audio, sigmas=SIGMAS, bongmath=False, show_progress=False, **factories, **text
        )
    mx.eval(out.video_latent, out.audio_latent)
    return out


class _CountingDict(dict):
    def __init__(self, stats: dict) -> None:
        super().__init__()
        self.stats = stats

    def get(self, key, default=None):
        hit = super().get(key, default)
        self.stats["hits" if hit is not None else "misses"] += 1
        return hit


@pytest.fixture
def kv_stats(monkeypatch):
    """Instrument every cache the loops create: count lookups that hit and miss."""
    stats = {"hits": 0, "misses": 0, "caches": 0}
    original = samplers.text_kv_cache_for

    def instrumented(model):
        cache = original(model)
        if cache is not None:
            stats["caches"] += 1
            cache.entries = _CountingDict(stats)
        return cache

    monkeypatch.setattr(samplers, "text_kv_cache_for", instrumented)
    return stats


@pytest.mark.parametrize("loop", ["guided", "res2s"])
@pytest.mark.parametrize("compute_dtype", [None, mx.float16])
@pytest.mark.parametrize("negative", [True, False])
def test_loops_are_bit_identical_with_the_cache(loop, compute_dtype, negative, monkeypatch, kv_stats):
    model = X0Model(_model(compute_dtype))
    monkeypatch.setenv("LTX2_TEXT_KV_CACHE", "0")
    reference = _run(loop, model, negative)
    assert kv_stats["caches"] == 0
    monkeypatch.delenv("LTX2_TEXT_KV_CACHE")
    cached = _run(loop, model, negative)
    assert kv_stats["caches"] > 0 and kv_stats["hits"] > 0
    assert mx.array_equal(cached.video_latent, reference.video_latent)
    assert mx.array_equal(cached.audio_latent, reference.audio_latent)


@pytest.mark.parametrize("negative", [True, False])
def test_guided_loop_reuses_positive_context_kv(negative, monkeypatch, kv_stats):
    monkeypatch.delenv("LTX2_TEXT_KV_CACHE", raising=False)
    _run("guided", X0Model(_model()), negative)
    steps = len(SIGMAS) - 1
    assert kv_stats["caches"] == steps
    # Per step and per text cross-attention (video + audio): the conditioned pass misses in every
    # block; the STG pass (resuming at STG_BLOCK) and the modality pass hit; the unconditional
    # pass runs without the cache when it has its own negative prompt, and hits when it falls
    # back to the positive context.
    per_step_hits = (NUM_LAYERS - STG_BLOCK) + NUM_LAYERS + (0 if negative else NUM_LAYERS)
    assert kv_stats["misses"] == steps * 2 * NUM_LAYERS
    assert kv_stats["hits"] == steps * 2 * per_step_hits


def _forward_kwargs(model: LTXModel) -> dict:
    cfg = model.config
    mx.random.seed(1)
    return dict(
        video_latent=mx.random.normal((1, 48, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, 8, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([0.7]),
        video_text_embeds=mx.random.normal((1, 6, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, 6, cfg.audio_dim)).astype(mx.bfloat16),
    )


def test_model_binds_to_the_first_context_only():
    model = _model()
    kwargs = _forward_kwargs(model)
    cache = TextKVCache()
    model(**kwargs, text_kv_cache=cache)
    assert len(cache.entries) == 2 * NUM_LAYERS
    stored = dict(cache.entries)
    other = {**kwargs, "video_text_embeds": kwargs["video_text_embeds"] + 1}
    want = model(**other)
    got = model(**other, text_kv_cache=cache)  # another context: must not read the cache
    assert all(mx.array_equal(g, w) for g, w in zip(got, want, strict=True))
    assert all(cache.entries[k] is v for k, v in stored.items())


def test_model_ignores_the_cache_for_another_compute_dtype():
    model = _model()
    kwargs = _forward_kwargs(model)
    cache = TextKVCache()
    model(**kwargs, text_kv_cache=cache)  # bound in float32
    model.set_compute_dtype(mx.float16)
    want = model(**kwargs)
    got = model(**kwargs, text_kv_cache=cache)
    assert all(mx.array_equal(g, w) for g, w in zip(got, want, strict=True))


def test_model_ignores_the_cache_with_a_block_provider():
    model = _model()
    kwargs = _forward_kwargs(model)
    cache = TextKVCache()
    model(**kwargs, block_provider=lambda i: model.transformer_blocks[i], text_kv_cache=cache)
    assert cache.entries == {}


def test_cache_for_refuses_tiling_streaming_and_env(monkeypatch):
    from ltx_core_mlx.components.modality_tiling import TiledLTXModel
    from ltx_core_mlx.loader.block_streaming import StreamingLTXModel

    monkeypatch.delenv("LTX2_TEXT_KV_CACHE", raising=False)
    assert isinstance(text_kv_cache_for(X0Model(_model())), TextKVCache)
    assert TiledLTXModel.supports_text_kv_cache is False
    assert StreamingLTXModel.supports_text_kv_cache is False
    monkeypatch.setenv("LTX2_TEXT_KV_CACHE", "0")
    assert text_kv_cache_for(X0Model(_model())) is None
