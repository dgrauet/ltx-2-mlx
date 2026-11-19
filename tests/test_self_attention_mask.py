"""Self-attention conditioning masks are [0, 1] weights, applied as a log-space bias (upstream semantics).

``conditioning/mask_utils.py`` builds the mask with 1 = attend, 0 = blocked and fractions =
attention strength. Upstream ``TransformerArgsPreprocessor._prepare_self_attention_mask``
turns it into ``log(p)`` (``p <= 0`` -> dtype minimum) before the attention kernel; until this
fix the port added the raw weights as a bias, so a 0 only lowered a logit by 1.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, self_attention_bias
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions

F, H, W = 2, 4, 6
N = F * H * W


def test_bias_values():
    mask = mx.array([[[1.0, 0.5, 0.0, -1.0]]])
    bias = self_attention_bias(mask)
    assert bias.shape == (1, 1, 1, 4)
    b = bias[0, 0, 0]
    assert b[0].item() == 0.0
    assert abs(b[1].item() - (-0.6931472)) < 1e-6
    assert b[2].item() == mx.finfo(mx.float32).min
    assert b[3].item() == mx.finfo(mx.float32).min
    assert self_attention_bias(mx.array([[True, False]])).dtype == mx.bool_


def _attention() -> Attention:
    mx.random.seed(0)
    attn = Attention(query_dim=32, num_heads=2, head_dim=16, use_rope=False, apply_gated_attention=False)
    mx.eval(attn.parameters())
    return attn


def test_blocked_keys_do_not_influence_the_output():
    attn = _attention()
    x = mx.random.normal((1, 10, 32), key=mx.random.key(1))
    mask = mx.ones((1, 10, 10)).at[:, :, 7:].multiply(0.0)  # keys 7..9 blocked for every query
    bias = self_attention_bias(mask)
    changed = x.at[:, 7:, :].add(5.0)
    out = attn(x, attention_mask=bias)[:, :7]
    out_changed = attn(changed, attention_mask=bias)[:, :7]
    assert mx.allclose(out, out_changed, atol=1e-6)


def test_fractional_weight_scales_attention_probability():
    """p on a key multiplies its unnormalised attention weight by p (softmax(s + log p))."""
    attn = _attention()
    x = mx.random.normal((1, 6, 32), key=mx.random.key(2))
    p = 0.25
    mask = mx.ones((1, 6, 6)).at[:, :, 2].multiply(p)
    got = attn(x, attention_mask=self_attention_bias(mask))

    q = attn.q_norm(attn.to_q(x)).reshape(1, 6, 2, 16).transpose(0, 2, 1, 3)
    k = attn.k_norm(attn.to_k(x)).reshape(1, 6, 2, 16).transpose(0, 2, 1, 3)
    v = attn.to_v(x).reshape(1, 6, 2, 16).transpose(0, 2, 1, 3)
    w = mx.exp((q @ k.transpose(0, 1, 3, 2)) * attn.scale) * mask[:, None]
    ref = (w / w.sum(-1, keepdims=True)) @ v
    want = attn.to_out(ref.transpose(0, 2, 1, 3).reshape(1, 6, 32))
    assert mx.allclose(got, want, atol=1e-5)


def _tiny_model() -> LTXModel:
    mx.random.seed(0)
    model = LTXModel(
        LTXModelConfig(
            num_layers=2,
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
    mx.eval(model.parameters())
    return model


def _inputs(model: LTXModel) -> dict:
    cfg = model.config
    mx.random.seed(1)
    return dict(
        video_latent=mx.random.normal((1, N, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, 8, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([0.7]),
        video_text_embeds=mx.random.normal((1, 6, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, 6, cfg.audio_dim)).astype(mx.bfloat16),
        video_positions=compute_video_positions(F, H, W),
        audio_positions=compute_audio_positions(8),
    )


@pytest.mark.parametrize("compute_dtype", [None, mx.float16])
def test_model_all_ones_mask_matches_no_mask(compute_dtype):
    model = _tiny_model()
    if compute_dtype is not None:
        model.set_compute_dtype(compute_dtype)
    kwargs = _inputs(model)
    plain = model(**kwargs)
    masked = model(**kwargs, video_attention_mask=mx.ones((1, N, N)))
    for a, b in zip(plain, masked, strict=True):
        assert mx.allclose(a.astype(mx.float32), b.astype(mx.float32), atol=2e-3)


def test_model_blocks_masked_tokens():
    """Tokens blocked both ways (an appended reference with strength 0) cannot affect the others.

    Video-only: with audio, the blocked tokens would still reach the others through the
    audio<->video cross-attention (audio attends to every video token), as upstream.
    """
    model = _tiny_model()
    kwargs = {**_inputs(model), "audio_latent": None, "audio_text_embeds": None, "audio_positions": None}
    keep = N - H * W  # last latent frame acts as the "reference" block
    mask = mx.ones((1, N, N))
    mask = mask.at[:, :keep, keep:].multiply(0.0).at[:, keep:, :keep].multiply(0.0)
    out = model(**kwargs, video_attention_mask=mask)[0][:, :keep]
    perturbed = {**kwargs, "video_latent": kwargs["video_latent"].at[:, keep:, :].add(3.0)}
    out2 = model(**perturbed, video_attention_mask=mask)[0][:, :keep]
    assert mx.array_equal(out, out2)
