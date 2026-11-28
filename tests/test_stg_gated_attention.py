"""STG bypass under the per-head gate (upstream LTX-2 v1.6.0).

A perturbed sample replaces its attention output by the value projection. The per-head gate
``2 * sigmoid(to_gate_logits(x))`` scales the attention output only: the bypassed values go to
``to_out`` ungated, for a fully perturbed batch and for the perturbed rows of a mixed one.
"""

import mlx.core as mx
import pytest

from ltx_core_mlx.model.transformer.attention import Attention

DIM, HEADS, HEAD_DIM = 16, 2, 8


def _attention() -> Attention:
    mx.random.seed(0)
    attn = Attention(query_dim=DIM, num_heads=HEADS, head_dim=HEAD_DIM, use_rope=False)
    # Zero-init gate logits give a gate of exactly 1, which hides the order of the two operations.
    attn.to_gate_logits.weight = mx.random.normal(attn.to_gate_logits.weight.shape)
    attn.to_gate_logits.bias = mx.array([1.5, -2.0])
    return attn


def _gate(attn: Attention, x: mx.array) -> mx.array:
    """Per-head gate broadcast over ``(B, N, heads * head_dim)``."""
    gate = 2.0 * mx.sigmoid(attn.to_gate_logits(x))
    return mx.repeat(gate, HEAD_DIM, axis=-1)


def test_fully_perturbed_batch_outputs_the_ungated_values() -> None:
    attn = _attention()
    x = mx.random.normal((2, 5, DIM))
    out = attn(x, perturbation_mask=mx.zeros((2, 1, 1, 1)))
    assert mx.allclose(out, attn.to_out(attn.to_v(x)), atol=1e-5).item()
    assert not mx.allclose(out, attn.to_out(attn.to_v(x) * _gate(attn, x)), atol=1e-3).item()


def test_mixed_batch_gates_the_attention_rows_only() -> None:
    attn = _attention()
    x = mx.random.normal((2, 5, DIM))
    mask = mx.array([1.0, 0.0]).reshape(2, 1, 1, 1)
    out = attn(x, perturbation_mask=mask)
    assert mx.allclose(out[0], attn(x[:1])[0], atol=1e-5).item()
    assert mx.allclose(out[1], attn.to_out(attn.to_v(x[1:]))[0], atol=1e-5).item()


@pytest.mark.parametrize("mask", [None, mx.ones((2, 1, 1, 1))])
def test_unperturbed_attention_is_still_gated(mask: mx.array | None) -> None:
    attn = _attention()
    x = mx.random.normal((2, 5, DIM))
    gated = attn(x, perturbation_mask=mask)
    attn.to_gate_logits = None
    assert not mx.allclose(gated, attn(x, perturbation_mask=mask), atol=1e-3).item()
