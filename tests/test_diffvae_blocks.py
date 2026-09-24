"""NeighborhoodAttention3D / NABlock: parameter layout, q/k pipeline, block algebra."""

from __future__ import annotations

import mlx.core as mx
import mlx.utils

from ltx_core_mlx.model.video_vae.diffusion_decoder.blocks import NABlock, NeighborhoodAttention3D
from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import KeyframeStream
from ltx_core_mlx.model.video_vae.diffusion_decoder.neighborhood_attention import joint_na3d_reference, na3d_reference
from ltx_core_mlx.model.video_vae.diffusion_decoder.rope import apply_axial_rope


def test_parameter_keys_match_the_pack_layout():
    blk = NABlock(8, 4, (3, 3, 3))
    keys = set(dict(mlx.utils.tree_flatten(blk.parameters())))
    assert keys == {
        "norm1.weight",
        "norm2.weight",
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.q_norm.weight",
        "attn.k_norm.weight",
        "attn.proj.weight",
        "attn.proj.bias",
        "mlp.w_gate.weight",
        "mlp.w_up.weight",
        "mlp.w_down.weight",
    }
    assert blk.attn.qkv.weight.shape == (24, 8) and blk.mlp.w_gate.weight.shape == (32, 8)
    assert blk.attn.q_norm.weight.shape == (4,)


def test_qkv_rope_pipeline():
    attn = NeighborhoodAttention3D(8, 4, (3, 3, 3))
    x = mx.random.normal((1, 3, 4, 5, 8))
    q, k, v = attn.qkv_rope(x)
    assert q.shape == k.shape == v.shape == (1, 3, 4, 5, 2, 4)
    raw = attn.qkv(x)
    rq, rk, rv = (a.reshape(1, 3, 4, 5, 2, 4) for a in mx.split(raw, 3, axis=-1))
    assert mx.array_equal(v, rv)
    pos = [mx.arange(n).astype(mx.float32) for n in (3, 4, 5)]
    exp_q = apply_axial_rope(attn.q_norm(rq) * 0.5, *pos, attn.split, attn.inv)  # 4 ** -0.5 = 0.5
    exp_k = apply_axial_rope(attn.k_norm(rk), *pos, attn.split, attn.inv)
    assert mx.allclose(q, exp_q, atol=1e-6).item() and mx.allclose(k, exp_k, atol=1e-6).item()


def test_attention_call_equals_reference_na_plus_proj():
    attn = NeighborhoodAttention3D(8, 4, (3, 3, 3))
    x = mx.random.normal((1, 4, 5, 6, 8))
    q, k, v = attn.qkv_rope(x)
    ref = attn.proj(na3d_reference(q, k, v, (3, 3, 3)).reshape(1, 4, 5, 6, 8))
    assert mx.allclose(attn(x), ref, atol=1e-5).item()


def test_nablock_residual_structure():
    blk = NABlock(8, 4, (3, 3, 3))
    x = mx.random.normal((1, 3, 3, 3, 8))
    y = blk(x)
    h = x + blk.attn(blk.norm1(x))
    assert mx.allclose(y, h + blk.mlp(blk.norm2(h)), atol=1e-6).item()


def _stream(p, h, w, c, times, valid):
    return KeyframeStream(mx.random.normal((1, p, h, w, c), key=mx.random.key(5)), mx.array(times), mx.array(valid))


def test_qkv_rope_t_pos_defaults_to_arange_and_accepts_fractional_positions():
    # head_dim=16 (not the file's usual 4): rope_dim_split(4) gives d_t=0, so a head_dim=4
    # attention head carries no temporal RoPE component at all and t_pos could never move q.
    attn = NeighborhoodAttention3D(32, 16, (3, 3, 3))
    y = mx.random.normal((1, 3, 4, 4, 32), key=mx.random.key(1))
    a = attn.qkv_rope(y)
    b = attn.qkv_rope(y, t_pos=mx.arange(3).astype(mx.float32))
    assert all(mx.array_equal(x, z) for x, z in zip(a, b, strict=True))
    c = attn.qkv_rope(y, t_pos=mx.array([0.0, 1.5, 2.25]))
    assert not mx.array_equal(a[0], c[0]) and mx.array_equal(a[2], c[2])  # v carries no positions


def test_attention_forward_with_keyframes_matches_the_oracle_and_shares_proj():
    attn = NeighborhoodAttention3D(8, 4, (3, 3, 3))
    x = mx.random.normal((1, 4, 5, 5, 8), key=mx.random.key(2))
    kf = _stream(2, 5, 5, 8, [0.5, 2.5], [True, True])
    o, ko = attn.forward_with_keyframes(x, kf)
    assert o.shape == x.shape and ko.shape == kf.x.shape
    q, k, v = attn.qkv_rope(x)
    kq, kk, kv = attn.qkv_rope(kf.x, t_pos=kf.times)
    ro, rko = joint_na3d_reference(q, k, v, kq, kk, kv, kf.times, kf.valid, (3, 3, 3))
    b, t, h, w = x.shape[:4]
    assert mx.allclose(o, attn.proj(ro.reshape(b, t, h, w, -1)), atol=1e-5, rtol=1e-5)
    assert mx.allclose(ko, attn.proj(rko.reshape(b, 2, h, w, -1)), atol=1e-5, rtol=1e-5)


def test_nablock_forward_with_keyframes_keeps_invalid_planes_unmasked_and_returns_a_stream():
    block = NABlock(8, 4, (3, 3, 3))
    x = mx.random.normal((1, 3, 4, 4, 8), key=mx.random.key(3))
    kf = _stream(2, 4, 4, 8, [1.0, 2.0], [True, False])
    y, out = block.forward_with_keyframes(x, kf)
    assert isinstance(out, KeyframeStream) and y.shape == x.shape and out.x.shape == kf.x.shape
    assert mx.array_equal(out.times, kf.times) and mx.array_equal(out.valid, kf.valid)
    # det blocks do not re-zero invalid planes (the decoder re-masks after each upsample)
    assert float(mx.abs(out.x[0, 1]).max()) > 0.0
    # the video path of __call__ is untouched
    assert block(x).shape == x.shape
