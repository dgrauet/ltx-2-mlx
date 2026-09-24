"""Stage-5 diffusion block reproducing upstream's default ``chunked_eager`` mode exactly.

The width is processed in ``w_chunks`` slabs with a ``halo = kw // 2`` on each side. Halos
come from the neighbouring slabs' pre-residual values; at the true left/right image borders
the missing halo is **edge-replicated** (first core column repeated on the left, this slab's
last core column on the right). RoPE W positions are global (pad cells get out-of-range
positions). Neighborhood attention runs on the padded slab as if it were the whole volume; only
the core columns are written back. Interior columns therefore equal full-volume attention, the
first / last ``halo`` columns do not — that is what upstream users get by default.

The context (stage-4 feature pixel-shuffled to stage-5 resolution) is upsampled once by the
caller (:meth:`NADiffusionDecoder.forward_stage_5`) and injected identically into every block,
since the upsample input and the shared upsample weights are the same across blocks.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.diffusion_decoder.blocks import NeighborhoodAttention3D
from ltx_core_mlx.model.video_vae.diffusion_decoder.layers import RMSNorm, SwiGLU
from ltx_core_mlx.model.video_vae.diffusion_decoder.neighborhood_attention import Kernel, joint_na3d, na3d

W_CHUNKS = 4  # upstream `_CHUNKED_W_CHUNKS`, not configurable


def build_w_slabs(x: mx.array, halo: int, w_chunks: int = W_CHUNKS) -> list[tuple[mx.array, mx.array, int, int]]:
    """Cut ``x`` ``(B, T, H, W, C)`` into padded W slabs (spec §5b). Returns ``(buf, w_pos, core_start, core_len)`` per slab."""
    width = x.shape[3]
    chunk_w = math.ceil(width / w_chunks)
    extent = chunk_w + 2 * halo
    slabs: list[tuple[mx.array, mx.array, int, int]] = []
    left: mx.array | None = None
    for i in range(w_chunks):
        cs, ce = i * chunk_w, min(width, (i + 1) * chunk_w)
        core_len = ce - cs
        has_right = i < w_chunks - 1
        parts: list[mx.array] = []
        core = x[:, :, :, cs:ce]
        if left is None:
            left_pad = mx.repeat(core[:, :, :, :1], halo, axis=3) if core_len > 0 else mx.zeros_like(x[:, :, :, :halo])
        else:
            left_pad = (
                mx.concatenate([mx.zeros_like(x[:, :, :, : halo - left.shape[3]]), left], axis=3)
                if left.shape[3] < halo
                else left
            )
        parts.append(left_pad)
        parts.append(core)
        right_filled = 0
        if has_right:
            right = x[:, :, :, ce : min(width, ce + halo)]
            right_filled = right.shape[3]
            if right_filled:
                parts.append(right)
        missing = extent - (halo + core_len + right_filled)
        if missing > 0:
            fill = core[:, :, :, -1:] if core_len > 0 else mx.zeros_like(x[:, :, :, :1])
            parts.append(mx.repeat(fill, missing, axis=3))
        buf = mx.concatenate(parts, axis=3)
        if has_right:
            left = x[:, :, :, ce - min(halo, core_len) : ce]
        w_pos = mx.arange(extent).astype(mx.float32) + float(cs - halo)
        slabs.append((buf, w_pos, cs, core_len))
    return slabs


def inject_context(x: mx.array, context: mx.array, context_proj: nn.Linear) -> mx.array:
    """``x + context_proj(context)`` where ``context`` is the pixel-shuffled stage-4 feature.

    Upstream re-applies ``upsamples[3]`` inside every block; the upsample is shared and its input
    is identical, so computing it once per tile (see ``NADiffusionDecoder.forward_stage_5``) is
    bit-identical and avoids eight redundant 512->2048 projections.
    """
    return x + context_proj(context)


class ChunkedDiffusionNABlock(nn.Module):
    """One stage-5 block: context inject, W-chunked NA residual, modulated SwiGLU residual."""

    def __init__(self, dim: int, head_dim: int, kernel: Kernel, context_channels: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.context_proj = nn.Linear(context_channels, dim)
        self.scale_shift_table = mx.zeros((7, dim))
        self.norm1 = RMSNorm(dim)
        self.attn = NeighborhoodAttention3D(dim, head_dim, kernel)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, 4 * dim)

    def _modulation(self, modulation: list[mx.array]) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        dim = self.scale_shift_table.shape[1]
        p = [m + self.scale_shift_table[i].reshape(1, 1, 1, 1, dim) for i, m in enumerate(modulation)]
        return p[0], p[1], p[3], p[4]  # scale_msa, shift_msa, scale_mlp, shift_mlp

    def attention_residual(self, x: mx.array, modulation: list[mx.array]) -> mx.array:
        scale_msa, shift_msa, _, _ = self._modulation(modulation)
        halo = self.kernel[2] // 2
        outs: list[mx.array] = []
        for buf, w_pos, _cs, core_len in build_w_slabs(x, halo):
            y = self.norm1(buf) * (1 + scale_msa) + shift_msa
            q, k, v = self.attn.qkv_rope(y, w_pos=w_pos)
            o = na3d(q, k, v, self.kernel)
            b, t, h, e, _, _ = o.shape
            o = self.attn.proj(o.reshape(b, t, h, e, -1))
            outs.append(o[:, :, :, halo : halo + core_len])
        return x + mx.concatenate(outs, axis=3)

    def attention_residual_with_keyframes(
        self,
        x: mx.array,
        kf_x: mx.array,
        modulation: list[mx.array],
        keyframe_times: mx.array,
        keyframe_valid: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """W-chunked joint attention residual: both streams cut into the same slabs with the same halo and ``w_pos``."""
        if tuple(kf_x.shape[2:4]) != tuple(x.shape[2:4]):
            raise ValueError(
                f"keyframe planes must share the video's spatial grid, got {kf_x.shape[2:4]} vs {x.shape[2:4]}"
            )
        scale_msa, shift_msa, _, _ = self._modulation(modulation)
        halo = self.kernel[2] // 2
        outs: list[mx.array] = []
        kouts: list[mx.array] = []
        for (buf, w_pos, _cs, core_len), (kbuf, _, _, _) in zip(
            build_w_slabs(x, halo), build_w_slabs(kf_x, halo), strict=True
        ):
            y = self.norm1(buf) * (1 + scale_msa) + shift_msa
            ky = self.norm1(kbuf) * (1 + scale_msa) + shift_msa
            q, k, v = self.attn.qkv_rope(y, w_pos=w_pos)
            kq, kk, kv = self.attn.qkv_rope(ky, w_pos=w_pos, t_pos=keyframe_times)
            o, ko = joint_na3d(q, k, v, kq, kk, kv, keyframe_times, keyframe_valid, self.kernel)
            b, t, h, e, _, _ = o.shape
            outs.append(self.attn.proj(o.reshape(b, t, h, e, -1))[:, :, :, halo : halo + core_len])
            kouts.append(self.attn.proj(ko.reshape(b, ko.shape[1], h, e, -1))[:, :, :, halo : halo + core_len])
        return x + mx.concatenate(outs, axis=3), kf_x + mx.concatenate(kouts, axis=3)

    def forward_with_keyframes(
        self,
        x: mx.array,
        context: mx.array,
        kf_x: mx.array,
        kf_context: mx.array,
        modulation: list[mx.array],
        keyframe_times: mx.array,
        keyframe_valid: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Dual-stream stage-5 block: per-stream context inject, joint attention, per-stream modulated MLP; invalid planes re-zeroed."""
        if tuple(kf_x.shape[2:4]) != tuple(x.shape[2:4]):
            raise ValueError(
                f"keyframe planes must share the video's spatial grid, got {kf_x.shape[2:4]} vs {x.shape[2:4]}"
            )
        _, _, scale_mlp, shift_mlp = self._modulation(modulation)
        x = inject_context(x, context, self.context_proj)
        kf_x = inject_context(kf_x, kf_context, self.context_proj)
        x, kf_x = self.attention_residual_with_keyframes(x, kf_x, modulation, keyframe_times, keyframe_valid)
        x = x + self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        kf_x = kf_x + self.mlp(self.norm2(kf_x) * (1 + scale_mlp) + shift_mlp)
        return x, kf_x * keyframe_valid[None, :, None, None, None].astype(kf_x.dtype)

    def __call__(self, x: mx.array, context: mx.array, modulation: list[mx.array]) -> mx.array:
        _, _, scale_mlp, shift_mlp = self._modulation(modulation)
        x = inject_context(x, context, self.context_proj)
        x = self.attention_residual(x, modulation)
        return x + self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
