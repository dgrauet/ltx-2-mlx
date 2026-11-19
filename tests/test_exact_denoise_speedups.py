"""Exact (bit-identical) denoising speedups: each one must not change a single output value."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.rope import apply_rope_interleaved, apply_rope_split


def _reference_attention(attn: Attention, x, context=None, rope=None, perturbation_mask=None):
    """The pre-reorder Attention forward: gate in head-major order, then transpose + reshape."""
    B = x.shape[0]
    kv = context if context is not None else x
    q = attn.q_norm(attn.to_q(x))
    k = attn.k_norm(attn.to_k(kv))
    v = attn.to_v(kv)
    q = q.reshape(B, -1, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, -1, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, -1, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
    if rope is not None:
        cos_f, sin_f, rtype = rope
        apply = apply_rope_split if rtype == "split" else apply_rope_interleaved
        q, k = apply(q, cos_f, sin_f), apply(k, cos_f, sin_f)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale)
    if perturbation_mask is not None:
        out = out * perturbation_mask + v * (1.0 - perturbation_mask)
    if attn.to_gate_logits is not None:
        gate = 2.0 * mx.sigmoid(attn.to_gate_logits(x))
        out = out * gate.transpose(0, 2, 1)[:, :, :, None]
    out = out.transpose(0, 2, 1, 3).reshape(B, -1, attn.num_heads * attn.head_dim)
    return attn.to_out(out)


def _attention(gated: bool, kv_dim: int | None = None) -> Attention:
    mx.random.seed(0)
    attn = Attention(query_dim=64, kv_dim=kv_dim, num_heads=4, head_dim=16, use_rope=True, apply_gated_attention=gated)
    if gated:  # zero-init logits give gate == 1 exactly: randomise so the gate does something
        attn.to_gate_logits.weight = mx.random.normal(attn.to_gate_logits.weight.shape, key=mx.random.key(9))
        attn.to_gate_logits.bias = mx.random.normal(attn.to_gate_logits.bias.shape, key=mx.random.key(10))
    mx.eval(attn.parameters())
    return attn


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("ptb", [None, 0.0, 1.0])
def test_gate_after_transpose_is_bit_identical_self_attention(gated, ptb):
    attn = _attention(gated)
    x = mx.random.normal((2, 24, 64), key=mx.random.key(1))
    half = attn.head_dim // 2
    cos = mx.random.uniform(shape=(2, attn.num_heads, 24, half), key=mx.random.key(2))
    sin = mx.random.uniform(shape=(2, attn.num_heads, 24, half), key=mx.random.key(3))
    rope = (cos, sin, "split")
    mask = None if ptb is None else mx.full((2, 1, 1, 1), ptb)
    got = attn(x, rope_freqs=rope, perturbation_mask=mask)
    want = _reference_attention(attn, x, rope=rope, perturbation_mask=mask)
    assert mx.array_equal(got, want)


def test_gate_after_transpose_is_bit_identical_cross_attention():
    attn = _attention(True, kv_dim=32)
    x = mx.random.normal((1, 20, 64), key=mx.random.key(4))
    ctx = mx.random.normal((1, 7, 32), key=mx.random.key(5))
    assert mx.array_equal(attn(x, encoder_hidden_states=ctx), _reference_attention(attn, x, context=ctx))


def test_gate_after_transpose_is_bit_identical_in_float16():
    attn = _attention(True)
    attn.set_dtype(mx.float16)
    x = mx.random.normal((1, 32, 64), key=mx.random.key(6)).astype(mx.float16)
    assert mx.array_equal(attn(x), _reference_attention(attn, x))


# ---------------------------------------------------------------------------
# Model-level caches: RoPE tables reused across forwards, masks cast once per forward
# ---------------------------------------------------------------------------

from ltx_core_mlx.model.transformer import model as model_mod  # noqa: E402
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig  # noqa: E402
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions  # noqa: E402


def _tiny_model() -> LTXModel:
    mx.random.seed(0)
    model = LTXModel(
        LTXModelConfig(
            num_layers=3,
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
    )
    model.set_dtype(mx.bfloat16)
    for block in model.transformer_blocks:
        for name in ("scale_shift_table", "audio_scale_shift_table", "prompt_scale_shift_table"):
            block[name] = 0.05 * mx.random.normal(block[name].shape)
    mx.eval(model.parameters())
    return model


def _inputs(model: LTXModel, sigma: float = 0.7) -> dict:
    cfg = model.config
    mx.random.seed(1)
    return dict(
        video_latent=mx.random.normal((1, 2 * 4 * 6, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, 8, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([sigma]),
        video_text_embeds=mx.random.normal((1, 6, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, 6, cfg.audio_dim)).astype(mx.bfloat16),
        video_positions=compute_video_positions(2, 4, 6),
        audio_positions=compute_audio_positions(8),
    )


def test_rope_cache_reuses_tables_for_the_same_positions(monkeypatch):
    model = _tiny_model()
    kwargs = _inputs(model)
    calls = {"n": 0}
    original = LTXModel._compute_rope_freqs

    def counting(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(LTXModel, "_compute_rope_freqs", counting)
    first = model(**kwargs)
    assert calls["n"] == 4  # video, audio, and the two cross-modal temporal tables
    second = model(**{**kwargs, "timestep": mx.array([0.4])})
    assert calls["n"] == 4  # same positions arrays: all four reused
    # A new positions array (even with equal values) recomputes: the cache matches on identity.
    model(**{**kwargs, "video_positions": mx.array(kwargs["video_positions"])})
    assert calls["n"] == 6  # video self + video cross
    assert len(model._rope_cache) == 4  # only the last forward's tables are kept
    mx.eval(first, second)


def test_rope_cache_output_is_bit_identical():
    kwargs = _inputs(_tiny_model())
    cached = _tiny_model()
    cached(**kwargs)  # warm the cache
    got = cached(**{**kwargs, "timestep": mx.array([0.3])})
    want = _tiny_model()(**{**kwargs, "timestep": mx.array([0.3])})  # fresh model: computed tables
    assert all(mx.array_equal(g, w) for g, w in zip(got, want, strict=True))


def test_rope_cache_is_not_a_module_parameter():
    model = _tiny_model()
    before = len(tree_flatten(model.parameters()))
    model(**_inputs(model))
    after = len(tree_flatten(model.parameters()))
    assert before == after


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_mask_cast_once_per_forward_is_bit_identical(monkeypatch, dtype):
    """Casting the self-attention / cross-attention masks once in LTXModel equals the per-block cast."""
    n_video = 2 * 4 * 6
    mx.random.seed(3)
    # A [0, 1] self-attention weight mask with blocked (0) and fractional entries, and an additive
    # Prompt-Relay-style cross mask with huge negative entries (the clamp has work to do).
    self_mask = mx.random.uniform(shape=(1, n_video, n_video))
    self_mask = mx.where(self_mask > 0.8, 0.0, self_mask)
    cross_mask = mx.where(mx.random.uniform(shape=(1, 1, n_video, 6)) > 0.5, -1e9, 0.0)

    def run() -> tuple[mx.array, mx.array]:
        model = _tiny_model()
        model.set_compute_dtype(dtype)
        kwargs = _inputs(model)
        return model(**kwargs, video_attention_mask=self_mask, video_cross_attention_mask=cross_mask)

    got = run()
    calls = {"n": 0}

    def no_hoist(dt, mask):  # the old path: every Attention call clamps and casts the float32 mask
        calls["n"] += 1
        return mask

    monkeypatch.setattr(model_mod, "cast_attention_mask", no_hoist)
    want = run()
    assert calls["n"] == 2
    assert all(mx.array_equal(g, w) for g, w in zip(got, want, strict=True))


# ---------------------------------------------------------------------------
# Fused (mx.compile) elementwise helpers in the transformer block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compute_dtype", [None, mx.float16])
@pytest.mark.parametrize("per_token", [False, True])
def test_fused_elementwise_helpers_are_bit_identical(monkeypatch, compute_dtype, per_token):
    from ltx_core_mlx.model.transformer import transformer as block_mod

    def run():
        model = _tiny_model()
        if compute_dtype is not None:
            model.set_compute_dtype(compute_dtype)
        kwargs = _inputs(model)
        if per_token:  # first latent frame preserved: per-token AdaLN path
            n = kwargs["video_latent"].shape[1]
            kwargs["video_timesteps"] = mx.concatenate([mx.zeros((1, 24)), mx.full((1, n - 24), 0.7)], axis=1)
        return model(**kwargs)

    fused = run()
    assert block_mod._gated_residual is not block_mod._gated_residual_eager  # compiled by default
    monkeypatch.setattr(block_mod, "_modulate", block_mod._modulate_eager)
    monkeypatch.setattr(block_mod, "_norm_modulate", block_mod._norm_modulate_eager)
    monkeypatch.setattr(block_mod, "_gated_residual", block_mod._gated_residual_eager)
    eager = run()
    assert all(mx.array_equal(f, e) for f, e in zip(fused, eager, strict=True))


# ---------------------------------------------------------------------------
# Per-token timestep embedding + AdaLN row grouping reused across modules and passes
# ---------------------------------------------------------------------------


def _per_token_kwargs(model: LTXModel) -> dict:
    """Inputs with per-token video and audio timesteps (first frame / first tokens preserved)."""
    kwargs = _inputs(model)
    n = kwargs["video_latent"].shape[1]
    kwargs["video_timesteps"] = mx.concatenate([mx.zeros((1, 24)), mx.full((1, n - 24), 0.7)], axis=1)
    kwargs["audio_timesteps"] = mx.concatenate([mx.zeros((1, 2)), mx.full((1, 6), 0.7)], axis=1)
    return kwargs


@pytest.fixture
def _dedupe_on_tiny(monkeypatch):
    """Let the tiny model's 48 / 8 rows reach the dedupe (it needs 1024 rows by default)."""
    monkeypatch.setattr(model_mod, "_ADALN_DEDUPE_MIN_ROWS", 1)
    monkeypatch.setattr(model_mod, "_ADALN_DEDUPE_MAX_FRAC", 1.0)


def test_per_token_grouping_runs_once_per_timesteps_array(monkeypatch, _dedupe_on_tiny):
    model = _tiny_model()
    kwargs = _per_token_kwargs(model)
    calls = {"group": 0, "embed": 0}
    original_group, original_embed = model_mod._unique_rows, LTXModel._embed_timestep_per_token

    def counting_group(flat):
        calls["group"] += 1
        return original_group(flat)

    def counting_embed(self, t):
        calls["embed"] += 1
        return original_embed(self, t)

    monkeypatch.setattr(model_mod, "_unique_rows", counting_group)
    monkeypatch.setattr(LTXModel, "_embed_timestep_per_token", counting_embed)
    first = model(**kwargs)
    # One grouping and one embedding per modality, shared by its two per-token AdaLN modules.
    assert calls == {"group": 2, "embed": 2}
    # A second pass of the same step (same timesteps arrays, e.g. the uncond pass): all reused.
    second = model(**{**kwargs, "video_text_embeds": kwargs["video_text_embeds"] * 0.5})
    assert calls == {"group": 2, "embed": 2}
    # A new array with equal values (the next step) recomputes: the cache matches on identity.
    model(**{**kwargs, "video_timesteps": mx.array(kwargs["video_timesteps"])})
    assert calls == {"group": 3, "embed": 3}
    assert len(model._per_token_cache) == 2  # only the last forward's entries are kept
    mx.eval(first, second)


@pytest.mark.parametrize("dedupe", [True, False])
def test_per_token_grouping_cache_is_bit_identical(monkeypatch, _dedupe_on_tiny, dedupe):
    if not dedupe:
        monkeypatch.setattr(model_mod, "_ADALN_DEDUPE", False)
    kwargs = _per_token_kwargs(_tiny_model())
    cached = _tiny_model()
    cached(**kwargs)  # warm the cache with the same timesteps arrays
    cached(**kwargs)  # calibration call done; this one may use the deduped GEMM
    got = cached(**{**kwargs, "timestep": mx.array([0.3])})

    reference = _tiny_model()
    reference(**{**kwargs, "video_timesteps": mx.array(kwargs["video_timesteps"])})
    reference(**{**kwargs, "video_timesteps": mx.array(kwargs["video_timesteps"])})
    # Fresh arrays each call: nothing reused, every grouping computed in place.
    want = reference(
        **{
            **kwargs,
            "timestep": mx.array([0.3]),
            "video_timesteps": mx.array(kwargs["video_timesteps"]),
            "audio_timesteps": mx.array(kwargs["audio_timesteps"]),
        }
    )
    assert all(mx.array_equal(g, w) for g, w in zip(got, want, strict=True))
