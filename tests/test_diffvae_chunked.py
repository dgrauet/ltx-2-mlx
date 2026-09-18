"""W-chunked stage-5 block: slab construction rules, interior equivalence, block algebra."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder.chunked import (
    ChunkedDiffusionNABlock,
    build_w_slabs,
    inject_context,
)
from ltx_core_mlx.model.video_vae.diffusion_decoder.layers import LinearPixelShuffleUpsample, SharedAdaLN
from ltx_core_mlx.model.video_vae.diffusion_decoder.neighborhood_attention import na3d_reference


def _ref_slabs(x, halo, w_chunks=4):
    """Straight transcription of upstream chunked/attn.py:356-441 in numpy."""
    xn = np.array(x)
    W = xn.shape[3]
    chunk_w = math.ceil(W / w_chunks)
    extent = chunk_w + 2 * halo
    out, left = [], None
    for i in range(w_chunks):
        cs, ce = i * chunk_w, min(W, (i + 1) * chunk_w)
        L = ce - cs
        has_right = i < w_chunks - 1
        buf = np.zeros(xn.shape[:3] + (extent,) + xn.shape[4:], dtype=xn.dtype)
        if left is not None:
            buf[:, :, :, halo - left.shape[3] : halo] = left
        buf[:, :, :, halo : halo + L] = xn[:, :, :, cs:ce]
        rf = 0
        if has_right:
            r = xn[:, :, :, ce : min(W, ce + halo)]
            rf = r.shape[3]
            buf[:, :, :, halo + L : halo + L + rf] = r
        if left is None and L > 0:
            buf[:, :, :, :halo] = buf[:, :, :, halo : halo + 1]
        miss = extent - (halo + L + rf)
        if miss > 0 and L > 0:
            buf[:, :, :, extent - miss :] = buf[:, :, :, halo + L - 1 : halo + L]
        if has_right:
            left = xn[:, :, :, ce - min(halo, L) : ce].copy()
        out.append((buf, np.arange(extent, dtype=np.float32) + (cs - halo), cs, L))
    return out


@pytest.mark.parametrize("W", [20, 21, 22, 23, 24, 56, 57])
def test_slabs_match_the_upstream_transcription(W):  # noqa: N803
    x = mx.random.normal((1, 2, 3, W, 6))
    got = build_w_slabs(x, halo=5)
    ref = _ref_slabs(x, 5)
    assert len(got) == 4
    for (buf, w_pos, cs, L), (rbuf, rpos, rcs, rL) in zip(got, ref, strict=True):
        assert (cs, L) == (rcs, rL)
        assert np.array_equal(np.array(w_pos), rpos)
        assert np.array_equal(np.array(buf), rbuf)


def test_slab_edges_are_replicated_not_mirrored():
    W = 24
    x = mx.arange(W).astype(mx.float32).reshape(1, 1, 1, W, 1)
    slabs = build_w_slabs(x, halo=5)
    first = np.array(slabs[0][0])[0, 0, 0, :, 0]
    assert first[:5].tolist() == [0.0] * 5 and first[5:11].tolist() == list(range(6))
    last = np.array(slabs[3][0])[0, 0, 0, :, 0]
    assert last[-5:].tolist() == [23.0] * 5
    assert np.array(slabs[0][1])[:6].tolist() == [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]


def test_context_inject_equals_whole_volume_formula():
    up = LinearPixelShuffleUpsample(8, (2, 2, 2), 2)
    blk = ChunkedDiffusionNABlock(4, 4, (3, 3, 3), context_channels=4)
    feat = mx.random.normal((1, 3, 2, 5, 8))
    x = mx.random.normal((1, 5, 4, 10, 4))
    context = up(feat, drop_leading_frame=True)
    got = inject_context(x, context, blk.context_proj)
    ref = x + blk.context_proj(context)
    assert mx.allclose(got, ref, atol=1e-6).item()


def test_interior_columns_equal_full_volume_attention():
    """Core columns >= halo away from the image borders must match plain na3d on the whole volume."""
    dim, hd, kernel = 8, 4, (3, 5, 5)
    blk = ChunkedDiffusionNABlock(dim, hd, kernel, context_channels=dim)
    x = mx.random.normal((1, 5, 7, 40, dim))
    mod = [mx.zeros((1, 1, 1, 1, dim))] * 7
    chunked = blk.attention_residual(x, mod)  # x + proj(NA(modulated norm1(x))) with W slabs
    y = blk.norm1(x)  # scale 0 / shift 0
    q, k, v = blk.attn.qkv_rope(y)
    full = x + blk.attn.proj(na3d_reference(q, k, v, kernel).reshape(x.shape))
    halo = kernel[2] // 2
    assert mx.allclose(chunked[:, :, :, halo:-halo], full[:, :, :, halo:-halo], atol=1e-4).item()
    assert not mx.allclose(chunked[:, :, :, :halo], full[:, :, :, :halo], atol=1e-4).item()  # border differs by design


def test_block_call_order_inject_attn_mlp():
    dim = 4
    up = LinearPixelShuffleUpsample(8, (2, 2, 2), 2)
    blk = ChunkedDiffusionNABlock(dim, 4, (3, 3, 3), context_channels=dim)
    blk.scale_shift_table = mx.random.normal((7, dim)) * 0.1
    adaln = SharedAdaLN(6, dim)
    mod = adaln(mx.random.normal((1, 6)))
    feat = mx.random.normal((1, 3, 2, 5, 8))
    x = mx.random.normal((1, 5, 4, 10, dim))
    context = up(feat, drop_leading_frame=True)
    y = blk(x, context, mod)
    p = [m + blk.scale_shift_table[i].reshape(1, 1, 1, 1, dim) for i, m in enumerate(mod)]
    h = inject_context(x, context, blk.context_proj)
    h = blk.attention_residual(h, mod)
    ref = h + blk.mlp(blk.norm2(h) * (1 + p[3]) + p[4])
    assert mx.allclose(y, ref, atol=1e-5).item()
