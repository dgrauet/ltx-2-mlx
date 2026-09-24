"""Deterministic-stage blocks of the diffusion decoder: neighborhood attention + SwiGLU."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import KeyframeStream
from ltx_core_mlx.model.video_vae.diffusion_decoder.layers import RMSNorm, SwiGLU
from ltx_core_mlx.model.video_vae.diffusion_decoder.neighborhood_attention import Kernel, joint_na3d, na3d
from ltx_core_mlx.model.video_vae.diffusion_decoder.rope import apply_axial_rope, inv_freqs, rope_dim_split


class NeighborhoodAttention3D(nn.Module):
    """Fused qkv -> per-head RMSNorm on q/k -> q * head_dim**-0.5 -> axial RoPE -> na3d -> proj."""

    def __init__(self, dim: int, head_dim: int, kernel: Kernel) -> None:
        super().__init__()
        self._kernel = kernel
        self._heads = dim // head_dim
        self._head_dim = head_dim
        self._scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
        self._split = rope_dim_split(head_dim)
        self._inv = tuple(inv_freqs(d) for d in self._split)

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @property
    def heads(self) -> int:
        return self._heads

    @property
    def split(self) -> tuple[int, int, int]:
        return self._split

    @property
    def inv(self) -> tuple[mx.array, mx.array, mx.array]:
        return self._inv

    def qkv_rope(
        self, y: mx.array, w_pos: mx.array | None = None, t_pos: mx.array | None = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Project, normalise, scale and RoPE ``y``; ``t_pos`` (default ``arange(T)``) carries the planes' fractional times."""
        b, t, h, w, _ = y.shape
        q, k, v = (a.reshape(b, t, h, w, self._heads, self._head_dim) for a in mx.split(self.qkv(y), 3, axis=-1))
        q = self.q_norm(q) * self._scale
        k = self.k_norm(k)
        if t_pos is None:
            t_pos = mx.arange(t).astype(mx.float32)
        h_pos = mx.arange(h).astype(mx.float32)
        if w_pos is None:
            w_pos = mx.arange(w).astype(mx.float32)
        q = apply_axial_rope(q, t_pos, h_pos, w_pos, self._split, self._inv)
        k = apply_axial_rope(k, t_pos, h_pos, w_pos, self._split, self._inv)
        return q, k, v.astype(q.dtype)

    def forward_with_keyframes(self, x: mx.array, kf: KeyframeStream) -> tuple[mx.array, mx.array]:
        """Joint attention of the video volume and the plane stack; shared ``proj`` applied per stream."""
        q, k, v = self.qkv_rope(x)
        kq, kk, kv = self.qkv_rope(kf.x, t_pos=kf.times)
        o, ko = joint_na3d(q, k, v, kq, kk, kv, kf.times, kf.valid, self._kernel)
        b, t, h, w, _, _ = o.shape
        return self.proj(o.reshape(b, t, h, w, -1)), self.proj(ko.reshape(b, kf.num_planes, h, w, -1))

    def __call__(self, x: mx.array) -> mx.array:
        q, k, v = self.qkv_rope(x)
        o = na3d(q, k, v, self._kernel)
        b, t, h, w, _, _ = o.shape
        return self.proj(o.reshape(b, t, h, w, -1))


class NABlock(nn.Module):
    """``x += attn(norm1(x)); x += mlp(norm2(x))`` — no timestep conditioning."""

    def __init__(self, dim: int, head_dim: int, kernel: Kernel) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = NeighborhoodAttention3D(dim, head_dim, kernel)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, 4 * dim)

    def forward_with_keyframes(self, x: mx.array, kf: KeyframeStream) -> tuple[mx.array, KeyframeStream]:
        """Dual-stream block: shared norms / attention / MLP, streams meet only in the joint softmax.

        Invalid planes are not re-zeroed here (upstream); the decoder re-masks after each upsample.
        """
        o, ko = self.attn.forward_with_keyframes(self.norm1(x), dataclasses.replace(kf, x=self.norm1(kf.x)))
        x = x + o
        kx = kf.x + ko
        x = x + self.mlp(self.norm2(x))
        kx = kx + self.mlp(self.norm2(kx))
        return x, dataclasses.replace(kf, x=kx)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
