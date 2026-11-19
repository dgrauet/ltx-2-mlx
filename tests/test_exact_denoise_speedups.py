"""Exact (bit-identical) denoising speedups: each one must not change a single output value."""

from __future__ import annotations

import mlx.core as mx
import pytest

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

