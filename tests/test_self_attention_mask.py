"""The [0, 1] self-attention mask reaches SDPA as a log-space additive bias.

``mask_utils`` builds multiplicative weights in [0, 1]; ``mx.fast.scaled_dot_product_attention``
adds a float mask to the logits. Handed over raw, a weight of 0 left attention untouched and
1 added +1 to the logit, so ``--conditioning-strength 0.0`` did not block the reference.
"""

from __future__ import annotations

import math

import mlx.core as mx

from ltx_core_mlx.conditioning.types.attention_strength_wrapper import ConditioningItemAttentionStrengthWrapper
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, prepare_self_attention_mask


def _tiny_model() -> LTXModel:
    mx.random.seed(0)
    cfg = LTXModelConfig(
        num_layers=2,
        video_dim=32,
        audio_dim=32,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=8,
        audio_head_dim=8,
        av_cross_num_heads=4,
        av_cross_head_dim=8,
        video_patch_channels=8,
        audio_patch_channels=8,
        ff_mult=2.0,
        timestep_embedding_dim=16,
    )
    model = LTXModel(cfg)
    mx.eval(model.parameters())
    return model


def test_prepare_self_attention_mask_values_and_shape() -> None:
    mask = mx.array([[[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]]], dtype=mx.float32)
    bias = prepare_self_attention_mask(mask, dtype=mx.bfloat16)
    assert bias is not None
    assert bias.shape == (1, 1, 3, 3)
    assert bias.dtype == mx.bfloat16
    assert float(bias[0, 0, 0, 0].item()) == 0.0
    assert abs(float(bias[0, 0, 0, 1].item()) - math.log(0.5)) < 1e-2
    assert abs(float(bias[0, 0, 1, 2].item()) - math.log(0.25)) < 2e-2
    assert float(bias[0, 0, 0, 2].item()) == float(mx.finfo(mx.bfloat16).min)


def test_bias_scales_softmax_weights_like_the_multiplicative_mask() -> None:
    """softmax(s + log w) == w * exp(s) / sum(w * exp(s)): the upstream semantics of the mask."""
    mx.random.seed(2)
    q, k, v = (mx.random.normal((1, 2, 3, 4)) for _ in range(3))
    weights = mx.array([[[1.0, 0.5, 0.0], [0.25, 1.0, 1.0], [0.0, 0.0, 1.0]]])
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=0.5, mask=prepare_self_attention_mask(weights, dtype=mx.float32)
    )
    scores = mx.exp((q @ k.transpose(0, 1, 3, 2)) * 0.5) * weights[:, None]
    expected = (scores / scores.sum(axis=-1, keepdims=True)) @ v
    assert mx.allclose(out, expected, atol=1e-5).item()


def test_prepare_self_attention_mask_passthrough_for_none_4d_and_bool() -> None:
    assert prepare_self_attention_mask(None) is None

    bias_4d = mx.zeros((1, 1, 4, 4), dtype=mx.bfloat16)
    assert prepare_self_attention_mask(bias_4d) is bias_4d

    bool_3d = mx.ones((1, 4, 4), dtype=mx.bool_)
    out_bool = prepare_self_attention_mask(bool_3d)
    assert out_bool is not None and out_bool.shape == (1, 1, 4, 4) and out_bool.dtype == mx.bool_


def test_zero_attention_strength_completely_blocks_reference_influence_in_dit() -> None:
    """With conditioning_attention_strength=0.0, changing reference tokens must not change target outputs."""
    model = _tiny_model()
    n_gen = 4
    mx.random.seed(1)
    gen_latent = mx.random.normal((1, n_gen, 8)).astype(mx.bfloat16)
    gen_pos = mx.zeros((1, n_gen, 3), dtype=mx.float32)
    text_v = mx.random.normal((1, 3, 32)).astype(mx.bfloat16)

    def _run(ref_fill: float, attn_strength: float) -> mx.array:
        base = LatentState(
            latent=gen_latent,
            clean_latent=gen_latent,
            denoise_mask=mx.ones((1, n_gen, 1), dtype=mx.bfloat16),
            positions=gen_pos,
        )
        ref_tokens = mx.full((1, n_gen, 8), ref_fill, dtype=mx.bfloat16)
        cond = ConditioningItemAttentionStrengthWrapper(
            VideoConditionByReferenceLatent(
                reference_latent=ref_tokens,
                reference_positions=gen_pos,
                strength=1.0,
            ),
            attention_mask=attn_strength,
        )
        state = cond.apply(base, (1, 2, 2))
        out_v, _ = model(
            video_latent=state.latent,
            audio_latent=None,
            timestep=mx.array([0.5], dtype=mx.bfloat16),
            video_text_embeds=text_v,
            video_positions=state.positions,
            video_attention_mask=state.attention_mask,
        )
        mx.eval(out_v)
        return out_v[:, :n_gen, :]

    # At attention_strength=0.0, drastically different reference tokens must have zero effect on target tokens.
    zero_a = _run(ref_fill=0.0, attn_strength=0.0)
    zero_b = _run(ref_fill=50.0, attn_strength=0.0)
    assert float(mx.max(mx.abs(zero_a - zero_b)).item()) < 1e-4

    # At attention_strength=1.0, reference tokens DO influence target tokens.
    full_a = _run(ref_fill=0.0, attn_strength=1.0)
    full_b = _run(ref_fill=50.0, attn_strength=1.0)
    assert float(mx.max(mx.abs(full_a - full_b)).item()) > 1e-2

    # At attention_strength=0.5, output lies strictly between 0.0 and 1.0 (not raw additive +0.5).
    half_b = _run(ref_fill=50.0, attn_strength=0.5)
    assert float(mx.max(mx.abs(half_b - zero_b)).item()) > 1e-3
    assert float(mx.max(mx.abs(half_b - full_b)).item()) > 1e-3
