"""Exact 3-D neighborhood attention: window bounds, brute-force oracle, blocked implementation."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder.neighborhood_attention import (
    joint_na3d,
    joint_na3d_reference,
    na3d,
    na3d_reference,
    window_start,
    window_starts,
)


@pytest.mark.parametrize(
    ("length", "kernel", "expected"),
    [
        (10, 5, [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]),  # centred, shifted inward at both borders
        (3, 5, [0, 0, 0]),  # kernel larger than the axis: k_eff = 3, window is the whole axis
        (11, 11, [0] * 11),
        (4, 3, [0, 0, 1, 1]),
    ],
)
def test_window_start_semantics(length, kernel, expected):
    assert [window_start(length, kernel, i) for i in range(length)] == expected
    assert np.array(window_starts(length, kernel)).tolist() == expected


def _numpy_na3d(q, k, v, kernel):
    # independent brute force in numpy (fp64): softmax over the Cartesian window
    q, k, v = (np.array(a, dtype=np.float64) for a in (q, k, v))
    B, T, H, W, NH, D = q.shape
    out = np.zeros_like(q)
    Ls, ks = (T, H, W), kernel
    starts = [[window_start(L, kk, i) for i in range(L)] for L, kk in zip(Ls, ks)]
    keff = [min(kk, L) for L, kk in zip(Ls, ks)]
    for b in range(B):
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    st, sh, sw = starts[0][t], starts[1][h], starts[2][w]
                    kk = k[b, st : st + keff[0], sh : sh + keff[1], sw : sw + keff[2]].reshape(-1, NH, D)
                    vv = v[b, st : st + keff[0], sh : sh + keff[1], sw : sw + keff[2]].reshape(-1, NH, D)
                    s = np.einsum("hd,nhd->nh", q[b, t, h, w], kk)
                    s = np.exp(s - s.max(0, keepdims=True))
                    s /= s.sum(0, keepdims=True)
                    out[b, t, h, w] = np.einsum("nh,nhd->hd", s, vv)
    return out


@pytest.mark.parametrize("kernel", [(3, 5, 5), (3, 7, 7), (5, 5, 5), (3, 3, 3)])
def test_reference_matches_independent_numpy(kernel):
    mx.random.seed(0)
    q, k, v = (mx.random.normal((1, 5, 9, 9, 2, 4)) for _ in range(3))
    got = np.array(na3d_reference(q, k, v, kernel))
    assert np.allclose(got, _numpy_na3d(q, k, v, kernel), atol=1e-5)


@pytest.mark.parametrize("kernel", [(3, 5, 5), (3, 7, 7), (5, 5, 5)])
@pytest.mark.parametrize("block", [(4, 8, 8), (2, 3, 5), (1, 1, 1), (16, 16, 16)])
def test_blocked_matches_reference_including_borders(kernel, block):
    mx.random.seed(1)
    q, k, v = (mx.random.normal((1, 5, 9, 11, 2, 4)) for _ in range(3))
    got = na3d(q, k, v, kernel, block=block, max_blocks=3)
    assert mx.allclose(got, na3d_reference(q, k, v, kernel), atol=1e-5, rtol=1e-5).item()


def test_blocked_handles_axis_shorter_than_kernel():
    q, k, v = (mx.random.normal((1, 3, 4, 4, 1, 4)) for _ in range(3))
    got = na3d(q, k, v, (5, 5, 5))
    assert mx.allclose(got, na3d_reference(q, k, v, (5, 5, 5)), atol=1e-5).item()


def test_blocked_bf16_runs_and_is_close():
    q, k, v = (mx.random.normal((1, 5, 9, 9, 2, 4)).astype(mx.bfloat16) for _ in range(3))
    got = na3d(q, k, v, (3, 5, 5))
    assert got.dtype == mx.bfloat16
    ref = na3d_reference(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), (3, 5, 5))
    assert mx.allclose(got.astype(mx.float32), ref, atol=5e-2).item()


def _numpy_joint(q, k, v, kq, kk, kv, times, valid, kernel):
    """Independent brute force (fp64): explicit key sets per query, both streams."""
    from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import keyframe_video_slots, video_keyframe_slots

    q, k, v, kq, kk, kv = (np.array(a, dtype=np.float64) for a in (q, k, v, kq, kk, kv))
    times, valid = np.array(times, dtype=np.float32), np.array(valid, dtype=bool)
    _, T, H, W, NH, D = q.shape
    P = kq.shape[1]
    kt, kh, kw = kernel
    win = lambda length, kk_, i: (  # noqa: E731
        window_start(length, kk_, i),
        window_start(length, kk_, i) + min(kk_, length),
    )
    vslots = video_keyframe_slots(times, valid, T)
    kslots = keyframe_video_slots(times, valid, T)

    def attend(query, keys, vals):
        s = np.einsum("nd,knd->nk", query, keys)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return np.einsum("nk,knd->nd", p, vals)

    vout = np.zeros_like(q)
    for t in range(T):
        t0, t1 = win(T, kt, t)
        for hh in range(H):
            h0, h1 = win(H, kh, hh)
            for ww in range(W):
                w0, w1 = win(W, kw, ww)
                keys = [k[0, t0:t1, h0:h1, w0:w1].reshape(-1, NH, D)]
                vals = [v[0, t0:t1, h0:h1, w0:w1].reshape(-1, NH, D)]
                for s in vslots[t]:
                    if s >= 0:
                        keys.append(kk[0, s, h0:h1, w0:w1].reshape(-1, NH, D))
                        vals.append(kv[0, s, h0:h1, w0:w1].reshape(-1, NH, D))
                vout[0, t, hh, ww] = attend(q[0, t, hh, ww], np.concatenate(keys), np.concatenate(vals))
    kout = np.zeros_like(kq)
    for p_ in range(P):
        if not valid[p_]:
            continue
        for hh in range(H):
            h0, h1 = win(H, kh, hh)
            for ww in range(W):
                w0, w1 = win(W, kw, ww)
                keys = [kk[0, p_, h0:h1, w0:w1].reshape(-1, NH, D)]
                vals = [kv[0, p_, h0:h1, w0:w1].reshape(-1, NH, D)]
                for s in kslots[p_]:
                    if s >= 0:
                        keys.append(k[0, s, h0:h1, w0:w1].reshape(-1, NH, D))
                        vals.append(v[0, s, h0:h1, w0:w1].reshape(-1, NH, D))
                kout[0, p_, hh, ww] = attend(kq[0, p_, hh, ww], np.concatenate(keys), np.concatenate(vals))
    return vout, kout


@pytest.mark.parametrize(
    ("shape", "planes", "kernel", "times", "valid"),
    [
        ((5, 6, 7), 2, (3, 3, 3), [1.5, 4.0], [True, True]),
        ((2, 5, 5), 3, (3, 3, 3), [0.0, 0.7, 1.9], [True, False, True]),  # T < 3: one plane slot empty for the planes
        ((1, 4, 6), 1, (3, 3, 5), [7.0], [True]),  # plane far outside the temporal kernel is still visible
        ((9, 4, 4), 2, (5, 3, 3), [2.0, 2.0], [True, True]),  # tie on |dt| -> index order
        ((4, 11, 9), 2, (3, 5, 5), [1.0, 3.0], [True, True]),  # several spatial blocks
    ],
)
def test_joint_na3d_matches_the_brute_force_oracles(shape, planes, kernel, times, valid):
    T, H, W = shape
    key = mx.random.key(11)
    keys = mx.random.split(key, 6)
    q, k, v = (mx.random.normal((1, T, H, W, 2, 4), key=kk_) for kk_ in keys[:3])
    kq, kk, kv = (mx.random.normal((1, planes, H, W, 2, 4), key=kk_) for kk_ in keys[3:])
    t, va = mx.array(times, dtype=mx.float32), mx.array(valid)
    vo, ko = joint_na3d(q, k, v, kq, kk, kv, t, va, kernel, block=(2, 3, 3), max_blocks=3)
    vr, kr = joint_na3d_reference(q, k, v, kq, kk, kv, t, va, kernel)
    vn, kn = _numpy_joint(q, k, v, kq, kk, kv, t, va, kernel)
    for got, want in ((vo, vr), (vo, vn), (ko, kr), (ko, kn)):
        assert np.allclose(np.array(got), np.array(want), atol=1e-5, rtol=1e-5)
    for p_, ok in enumerate(valid):
        if not ok:
            assert np.array(mx.abs(ko[0, p_]).max()) == 0.0


def test_joint_na3d_video_reduces_to_na3d_when_no_plane_is_valid():
    q, k, v = (mx.random.normal((1, 4, 5, 5, 2, 4), key=mx.random.key(i)) for i in range(3))
    kq, kk, kv = (mx.random.normal((1, 2, 5, 5, 2, 4), key=mx.random.key(10 + i)) for i in range(3))
    vo, ko = joint_na3d(q, k, v, kq, kk, kv, mx.array([1.0, 2.0]), mx.array([False, False]), (3, 3, 3))
    assert np.allclose(np.array(vo), np.array(na3d(q, k, v, (3, 3, 3))), atol=1e-6)
    assert np.array(mx.abs(ko).max()) == 0.0


def test_joint_na3d_rejects_batches_and_spatial_mismatch():
    q = mx.zeros((2, 3, 3, 3, 1, 4))
    with pytest.raises(ValueError, match="batch"):
        joint_na3d(q, q, q, q[:, :1], q[:, :1], q[:, :1], mx.array([0.0]), mx.array([True]), (3, 3, 3))
    q1 = mx.zeros((1, 3, 3, 3, 1, 4))
    bad = mx.zeros((1, 1, 4, 3, 1, 4))
    with pytest.raises(ValueError, match="spatial"):
        joint_na3d(q1, q1, q1, bad, bad, bad, mx.array([0.0]), mx.array([True]), (3, 3, 3))
