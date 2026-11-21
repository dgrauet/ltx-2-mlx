"""Block-sparse video self-attention for the stage-2 refine (Sol routing), opt-in.

Sol-Attn (NVIDIA, arXiv 2607.24027; reference code in NVlabs/Sana, ``sol-engine`` branch, Apache-2.0) routes attention
per 64-token block. For every head, each 64-token block gets a query centroid and a key centroid. A query block attends
exactly to the key blocks whose centroid score passes a per-query-block threshold (``mean + tau * std`` of its centroid
scores, the "diag" threshold) and to its neighbouring blocks (``|i - j| <= 1``). Every other key block enters the same
softmax once, as its key centroid with a ``log(block length)`` bias and its value mean, which is Sol's length-weighted
correction: ``exp(s + ln L) * mean(V) = exp(s) * sum(V)``. ``tau = -inf`` routes every block, i.e. dense attention.
NVIDIA's LTX-2.5 distilled config (``models/ltx25/RTX5090/attention.py``) applies it to the stage-2 video
self-attention only, block 0 dense, with ``tau = 1.0, 1.25, 1.5`` over the three stage-2 steps.

The attention itself is MLX's steel flash-attention loop (``steel/attn/kernels/steel_attention.h``, MIT, Copyright
Apple; BQ=32, BK=16, 4 simdgroups, the shape MLX picks for head_dim 128 on non-NAX GPUs) run over the routed blocks
only, followed by the centroid tiles of the skipped blocks. The routing mask is built in MLX before the kernel. The
steel headers are read from the installed ``mlx`` package and compiled through ``mx.fast.metal_kernel``, one kernel per
dtype; if that fails for a dtype (a future MLX changes them), :func:`kernel_available` is False for it and those calls
stay on the dense kernel.

One routing rule beyond the reference, measured on LTX-2.5 stage 2: with ``temporal``, the blocks holding the same
(h, w) positions one latent frame before and after are always exact (+0.2 % density; 15-20 % lower per-layer error on
some layers).

Entry points: :func:`sparse_self_attention` (q, k, v -> output), :class:`SparseAttentionState` (what
:meth:`LTXModel.set_sparse_attention` shares with the ``attn1`` modules), :func:`sol_taus_from_env`.
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx
import mlx.core as mx

SOL_TAU_ENV = "LTX2_SOL_TAU"
#: Tokens per routing block (Sol's block size).
BLOCK = 64
#: Sequences shorter than this keep the dense kernel (routing overhead, nothing to skip).
MIN_TOKENS = 4096


def sol_taus_from_env() -> tuple[float, ...] | None:
    """Parse ``LTX2_SOL_TAU``.

    Returns:
        ``None`` (unset or ``off``: dense attention, the default) or one ``tau`` per stage-2 step; the last value
        repeats when a stage has more steps.

    Raises:
        ValueError: On anything but a comma-separated list of finite numbers.
    """
    value = os.environ.get(SOL_TAU_ENV, "").strip().lower()
    if value in ("", "off"):
        return None
    try:
        taus = tuple(float(t) for t in value.split(","))
    except ValueError:
        raise ValueError(f"{SOL_TAU_ENV}={value!r}: expected comma-separated numbers, e.g. 1.0,1.25,1.5") from None
    if not all(math.isfinite(t) for t in taus):
        raise ValueError(f"{SOL_TAU_ENV}={value!r}: values must be finite")
    return taus


@dataclass
class SparseAttentionState:
    """Shared by the video self-attention modules of one :class:`LTXModel` while sparse attention is on.

    ``taus[i]`` applies to the forwards at ``sigmas[i]``: :meth:`prepare`, called by ``LTXModel.__call__`` with the
    forward's timestep, selects it by sigma, so every forward of a step (tiles, the overflow guard's recompute) gets the
    same tau. A sigma outside the table leaves the forward dense. The last tau repeats when there are more sigmas than
    taus.

    Args:
        taus: One routing threshold per step.
        sigmas: The sigma of each step, as the sampler passes it.
        temporal: Also route exactly the blocks one latent frame before and after (see :func:`build_route`).
        min_tokens: Shorter sequences stay dense.
    """

    taus: tuple[float, ...]
    sigmas: tuple[float, ...]
    temporal: bool = True
    min_tokens: int = MIN_TOKENS
    #: Tau of the current forward; ``None`` = dense.
    tau: float | None = None
    #: Tokens of one latent frame in the current forward (0 = unknown: no temporal neighbours).
    tokens_per_frame: int = 0
    #: Attention calls that ran sparse (for logs and tests).
    calls: int = 0

    def tau_for_sigma(self, sigma: float) -> float | None:
        """The tau of the step whose sigma is within 1e-2 of ``sigma`` (bfloat16 rounding), else ``None``."""
        if not self.sigmas:
            return None
        i = min(range(len(self.sigmas)), key=lambda j: abs(self.sigmas[j] - sigma))
        if abs(self.sigmas[i] - sigma) > 1e-2:
            return None
        return self.taus[min(i, len(self.taus) - 1)]

    def prepare(self, timestep: mx.array, video_positions: mx.array | None) -> None:
        """Select this forward's tau from its timestep and read the latent frame size from the video positions.

        The generated tokens come first, frame by frame (patchifier order), so a latent frame is the leading run of
        tokens sharing the first token's time; appended conditioning tokens do not change it.
        """
        self.tau = self.tau_for_sigma(float(timestep.reshape(-1)[0].item()))
        self.tokens_per_frame = 0
        if video_positions is not None:
            t = video_positions[0, :, 0]
            later = t != t[0]
            first = mx.argmax(later)
            self.tokens_per_frame = int(first.item()) if bool(later[first].item()) else int(t.shape[0])

    def tau_for_call(self, num_tokens: int, dtype: mx.Dtype = mx.float16) -> float | None:
        """The tau a self-attention call over ``num_tokens`` tokens in ``dtype`` runs with now, or ``None`` to stay dense.

        Short sequences stay dense, and so does every call when the kernels are unavailable in ``dtype``. Counts the
        sparse calls.
        """
        if self.tau is None or num_tokens < self.min_tokens or not kernel_available(dtype):
            return None
        self.calls += 1
        return self.tau


# ---------------------------------------------------------------------------
# Metal kernels
# ---------------------------------------------------------------------------


def _include_dir() -> Path:
    for root in mlx.__path__:
        inc = Path(root) / "include"
        if (inc / "mlx/backend/metal/kernels/steel/attn/attn.h").exists():
            return inc
    raise FileNotFoundError("mlx include/mlx/backend/metal/kernels/steel/attn/attn.h not found")


def _inline(inc: Path, path: Path, seen: set[Path]) -> str:
    """The header with its local includes pasted in (``mx.fast.metal_kernel`` takes one source string)."""
    out = []
    for line in path.read_text().splitlines():
        m = re.match(r'\s*#include\s+"(.+)"', line)
        if m:
            p = inc / m.group(1)
            if p not in seen:
                seen.add(p)
                out.append(_inline(inc, p, seen))
            continue
        if line.strip() == "#pragma once":
            continue
        out.append(line)
    return "\n".join(out)


_OPS = r"""
struct MaxOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); } };
struct SumOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; } };
struct MulOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; } };
struct ExpSubOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); } };
struct DivOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x / y; } };
"""

# One key tile through MLX's steel score / softmax / output update (steel_attention.h, WN = 1, no mask / causal / sinks).
# CENT = the tile holds centroid keys of skipped blocks: centroids of routed or out-of-range blocks are masked and the
# others get the log2(length) bias; otherwise exact keys, masked past `rem` in a partial last block.
_TILE = r"""
template <bool CENT, typename T, int BK, int BD, int LDQ_tgp, int LDK_tgp, int LDV_tgp,
          typename KBlockLoader, typename VBlockLoader, typename RP, typename CBP, typename QT, typename KT, typename ST,
          typename VT, typename OT>
METAL_FUNC void sol_tile(
    const device T* kp, const device T* vp, int kld, int vld, int rem, int c_first, int nb,
    RP R, CBP CB, float scale, threadgroup T* Qs, threadgroup T* KVs,
    short Qs_offset, short Ks_offset, short Vs_offset, short sn, uint simd_group_id, uint simd_lane_id,
    thread QT& Qtile, thread KT& Ktile, thread ST& Stile, thread VT& Vtile, thread OT& Otile,
    thread float* max_score, thread float* sum_score) {
  constexpr short kFragSize = 8;
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  using MMAFrag_acc_t = BaseMMAFrag<float, kFragSize, kFragSize>;
  constexpr short kRowsPT = ST::kRowsPerThread;
  constexpr float neg_inf = Limits<float>::finite_min;
  KBlockLoader loader_k(kp, kld, KVs, simd_group_id, simd_lane_id);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (rem < BK) { loader_k.load_safe(short2(BD, rem)); } else { loader_k.load_unsafe(); }
  Stile.clear();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for (short dd = 0; dd < TD; dd++) {
    simdgroup_barrier(mem_flags::mem_none);
    Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * kFragSize]);
    Ktile.template load<T, 1, 1, LDK_tgp, 1>(&KVs[Ks_offset + dd * kFragSize * LDK_tgp]);
    simdgroup_barrier(mem_flags::mem_none);
    tile_matmad(Stile, Qtile, Ktile, Stile);
  }
  STEEL_PRAGMA_UNROLL
  for (short ii = 0; ii < ST::kElemsPerTile; ii++) { Stile.elems()[ii] *= scale; }
  if (CENT) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
        const int c = c_first + sn + j * kFragSize + jj;
        if (c >= nb || R[c]) { Stile.frag_at(0, j)[jj] = neg_inf; } else { Stile.frag_at(0, j)[jj] += CB[c]; }
      }
    }
  } else if (rem < BK) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
        if (sn + j * kFragSize + jj >= rem) { Stile.frag_at(0, j)[jj] = neg_inf; }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  VBlockLoader loader_v(vp, vld, KVs, simd_group_id, simd_lane_id);
  if (rem < BK) { loader_v.load_safe(short2(BD, rem)); } else { loader_v.load_unsafe(); }

  float new_max[kRowsPT];
  float factor[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { new_max[i] = max_score[i]; }
  Stile.template row_reduce<MaxOp>(new_max);
  Stile.template row_bin_op<ExpSubOp>(new_max);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { factor[i] = fast::exp2(max_score[i] - new_max[i]); max_score[i] = new_max[i]; }
  float sum_score_tmp[kRowsPT] = {0};
  Stile.template row_reduce<SumOp>(sum_score_tmp);
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i]; }
  Otile.template row_bin_op<MulOp>(factor);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TD; id++) {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      simdgroup_barrier(mem_flags::mem_none);
      Vtile.template load<T, 1, 1, LDV_tgp, 1>(&KVs[Vs_offset + ik * kFragSize * LDV_tgp + id * kFragSize]);
      simdgroup_barrier(mem_flags::mem_none);
      MMAFrag_acc_t::mma(Otile.frag_at(0, id), Stile.frag_at(0, ik), Vtile.frag_at(0, 0), Otile.frag_at(0, id));
    }
  }
}

// CBP / RP: MLX passes inputs with fewer than 8 elements in the constant address space, the others in device memory.
template <typename T, int BQ, int BK, int BD, int WM, typename CBP, typename RP>
METAL_FUNC void sol_attention_impl(
    const device T* Q, const device T* K, const device T* V, const device T* KC, const device T* VC,
    CBP CB, RP R, device T* O,
    const int64_t qs_b, const int64_t qs_h, const int64_t qs_t, const int64_t ks_b, const int64_t ks_h,
    const int64_t ks_t, const int64_t vs_b, const int64_t vs_h, const int64_t vs_t,
    const int H, const int N, const int nb, const float softmax_scale,
    threadgroup T* Qs, threadgroup T* KVs, uint simd_lane_id, uint simd_group_id, uint3 tid) {
  constexpr short kFragSize = 8;
  constexpr short pad = 16 / sizeof(T);
  constexpr short LDQ_tgp = BD + pad;
  constexpr short LDK_tgp = BK + pad;
  constexpr short LDV_tgp = BD + pad;
  constexpr int TGP = WM * 32;
  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, TGP>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, TGP>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, TGP>;
  using MMAFrag_acc_t = BaseMMAFrag<float, kFragSize, kFragSize>;
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;
  static_assert(BQ == WM * kFragSize, "one simdgroup matrix row per simdgroup");

  const int q0 = int(tid.x) * BQ;
  const int h = int(tid.y);
  const int b = int(tid.z);
  const ulong bh = ulong(b) * H + h;
  Q += b * qs_b + h * qs_h + q0 * qs_t;
  K += b * ks_b + h * ks_h;
  V += b * vs_b + h * vs_h;
  KC += bh * nb * BD;
  VC += bh * nb * BD;
  R += (bh * nb + q0 / 64) * nb;
  O += ((ulong(b) * N + q0) * H + h) * BD;
  const int q_rem = N - q0;

  QBlockLoader loader_q(Q, int(qs_t), Qs, simd_group_id, simd_lane_id);
  const float scale = softmax_scale * M_LOG2E_F;

  MMATile<float, 1, 1, MMAFrag_acc_t> Qtile;
  MMATile<float, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<float, 1, TK, MMAFrag_acc_t> Stile;
  MMATile<float, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<float, 1, TD, MMAFrag_acc_t> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * simd_group_id;
  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (q_rem < BQ) { loader_q.load_safe(short2(BD, q_rem)); } else { loader_q.load_unsafe(); }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
  float max_score[kRowsPT];
  float sum_score[kRowsPT] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { max_score[i] = Limits<float>::finite_min; }

#define SOL_TILE(CENT, KP, VP, KLD, VLD, REM, CF) \
  sol_tile<CENT, T, BK, BD, LDQ_tgp, LDK_tgp, LDV_tgp, KBlockLoader, VBlockLoader>( \
      KP, VP, KLD, VLD, REM, CF, nb, R, CB, scale, Qs, KVs, Qs_offset, Ks_offset, Vs_offset, sn, \
      simd_group_id, simd_lane_id, Qtile, Ktile, Stile, Vtile, Otile, max_score, sum_score)
  // Routed blocks, exact. Full blocks pass a literal BK (unchecked loads, as the dense kernel); token offsets fit in
  // int. The loop over the 4 tiles of a block is not unrolled: that measured 5x slower (code size). The diagonal block
  // is always routed, so every row has a real maximum before the centroid tiles come in.
  const int kst = int(ks_t), vst = int(vs_t);
  for (int kb = 0; kb < nb; kb++) {
    if (!R[kb]) { continue; }
    const int k0 = kb * 64;
    if (k0 + 64 <= N) {
      _Pragma("clang loop unroll(disable)")
      for (int t = 0; t < 64; t += BK) { SOL_TILE(false, K + (k0 + t) * kst, V + (k0 + t) * vst, kst, vst, BK, 0); }
    } else {
      for (int t = 0; t < N - k0; t += BK) {
        SOL_TILE(false, K + (k0 + t) * kst, V + (k0 + t) * vst, kst, vst, N - k0 - t, 0);
      }
    }
  }
  // Skipped blocks, as centroid keys; a centroid tile whose blocks are all routed is skipped.
  for (int c0 = 0; c0 < nb; c0 += BK) {
    bool any = false;
    for (int c = c0; c < min(c0 + BK, nb); c++) { any = any || !R[c]; }
    if (!any) { continue; }
    SOL_TILE(true, KC + c0 * BD, VC + c0 * BD, BD, BD, nb - c0, c0);
  }
#undef SOL_TILE

  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);
  O += (tm + sm) * H * BD + sn;
  if (q_rem < BQ) {
    const short2 dims = short2(BD - sn, q_rem - (tm + sm));
    if (dims.x <= 0 || dims.y <= 0) { return; }
    Otile.template store_safe<T, 1, 1>(O, H * BD, dims);
  } else {
    Otile.template store<T, 1, 1>(O, H * BD);
  }
}
"""

_ATTN_BODY = r"""
  constexpr short pad = 16 / sizeof(T);
  constexpr short m0 = (BK + pad) * BD;
  constexpr short m1 = BK * (BD + pad);
  threadgroup T Q_smem[BQ * (BD + pad)];
  threadgroup T KV_smem[m0 > m1 ? m0 : m1];
  sol_attention_impl<T, BQ, BK, BD, WM>(
      q, k, v, kc, vc, cb, route, out,
      q_strides[0], q_strides[1], q_strides[2], k_strides[0], k_strides[1], k_strides[2],
      v_strides[0], v_strides[1], v_strides[2],
      ip[0], ip[1], ip[2], sc[0], Q_smem, KV_smem,
      thread_index_in_simdgroup, simdgroup_index_in_threadgroup, threadgroup_position_in_grid);
"""

# One thread per (batch * head, block, dim): float32 sums of q, k and v over the 64 tokens of the block.
_SUM_BODY = r"""
  const uint H = ip[0], N = ip[1], nb = ip[2];
  const uint gid = thread_position_in_grid.x;
  const uint d = gid % BD;
  const uint c = (gid / BD) % nb;
  const uint bh = gid / (BD * nb);
  const uint b = bh / H, h = bh % H;
  const uint t0 = c * 64, t1 = min(t0 + 64, N);
  float sq = 0.0f, sk = 0.0f, sv = 0.0f;
  for (uint t = t0; t < t1; t++) {
    sq += float(q[b * q_strides[0] + h * q_strides[1] + t * q_strides[2] + d]);
    sk += float(k[b * k_strides[0] + h * k_strides[1] + t * k_strides[2] + d]);
    sv += float(v[b * v_strides[0] + h * v_strides[1] + t * v_strides[2] + d]);
  }
  qs[gid] = sq; ks[gid] = sk; vs[gid] = sv;
"""

_KERNELS: dict[str, Callable[..., Any]] = {}
#: Probe result per dtype: ``mx.fast.metal_kernel`` builds a separate kernel for each ``T``.
_AVAILABLE: dict[mx.Dtype, bool] = {}
#: Probe tolerance against ``mx.fast.scaled_dot_product_attention``, per dtype.
_PROBE_ATOL = {mx.float16: 2e-3, mx.bfloat16: 1.6e-2, mx.float32: 1e-4}
#: Probe length: 9 blocks, the last one partial. ``mx.fast.metal_kernel`` passes inputs with fewer than 8 elements in
#: ``constant`` memory, which builds a different kernel; with 9 blocks every array input but ``ip`` and ``sc`` (always
#: scalars) is in device memory, as in a real call (nb >= 64).
_PROBE_TOKENS = 8 * BLOCK + 3


def _kernels() -> dict[str, Callable[..., Any]]:
    if not _KERNELS:
        inc = _include_dir()
        header = (
            _inline(inc, inc / "mlx/backend/metal/kernels/steel/attn/attn.h", set())
            + "\nusing namespace mlx::steel;\n"
            + _OPS
            + _TILE
        )
        _KERNELS["attn"] = mx.fast.metal_kernel(
            name="ltx_sol_attention",
            input_names=["q", "k", "v", "kc", "vc", "cb", "route", "ip", "sc"],
            output_names=["out"],
            header=header,
            source=_ATTN_BODY,
            ensure_row_contiguous=False,
        )
        _KERNELS["sum"] = mx.fast.metal_kernel(
            name="ltx_sol_block_sums",
            input_names=["q", "k", "v", "ip"],
            output_names=["qs", "ks", "vs"],
            source=_SUM_BODY,
            ensure_row_contiguous=False,
        )
    return _KERNELS


def kernel_available(dtype: mx.Dtype = mx.float16) -> bool:
    """Whether the kernels build and run here in ``dtype`` (checked once per dtype, on a small exact problem).

    The check runs the kernel variant a real call in ``dtype`` runs: q, k, v laid out as in ``attn1`` (heads transposed
    out of ``(B, N, H, D)``) and long enough that every array input is in device memory (see ``_PROBE_TOKENS``).
    False when the installed MLX lacks the steel headers this module compiles against, or the kernel errors or does
    not match dense attention; callers then stay on the dense kernel and a warning is printed once per dtype.
    """
    if dtype not in _AVAILABLE:
        try:
            q, k, v = (
                mx.random.normal((1, _PROBE_TOKENS, 2, 128), key=mx.random.key(i)).astype(dtype).transpose(0, 2, 1, 3)
                for i in range(3)
            )
            out = sparse_self_attention(q, k, v, 128**-0.5, tau=-math.inf, check=False)
            # Exact fp32 reference on the CPU: MLX >= 0.32 runs fp32 GPU matmuls (dense SDPA included) in TF32 on
            # newer GPUs (M5), ~1e-3 off, which failed the fp32 check although the kernel itself is exact.
            with mx.stream(mx.cpu):
                qf, kf, vf = (x.astype(mx.float32) for x in (q, k, v))
                scores = (qf @ kf.transpose(0, 1, 3, 2)) * 128**-0.5
                ref = (mx.softmax(scores, axis=-1) @ vf).transpose(0, 2, 1, 3)
                mx.eval(ref)
            atol = _PROBE_ATOL.get(dtype, 2e-3)
            ok = bool(mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=atol).item())
            if not ok:
                print(
                    f"warning: sparse attention kernel check failed for {dtype}; staying on dense attention",
                    file=sys.stderr,
                )
            _AVAILABLE[dtype] = ok
        except Exception as exc:  # any build or run failure means "use the dense kernel"
            print(
                f"warning: sparse attention kernel unavailable for {dtype} ({exc}); staying on dense attention",
                file=sys.stderr,
            )
            _AVAILABLE[dtype] = False
    return _AVAILABLE[dtype]


# ---------------------------------------------------------------------------
# Routing and attention
# ---------------------------------------------------------------------------


def block_summaries(q: mx.array, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Per head and 64-token block: query and key centroids (float32), value means (input dtype), log2 lengths.

    Args:
        q, k, v: ``(B, H, N, D)``, last axis contiguous; any batch / head / token strides.

    Returns:
        ``(qc, kc, vmean, log2_len)``: ``(B, H, nb, D)`` float32, ``(B, H, nb, D)`` float32, ``(B, H, nb, D)`` in
        ``v.dtype``, ``(nb,)`` float32.
    """
    B, H, N, D = q.shape
    nb = -(-N // BLOCK)
    ip = mx.array([H, N, nb], dtype=mx.int32)
    shape = (B, H, nb, D)
    qs, ks, vs = _kernels()["sum"](
        inputs=[q, k, v, ip],
        template=[("BD", D)],
        grid=(B * H * nb * D, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape] * 3,
        output_dtypes=[mx.float32] * 3,
    )
    lengths = mx.minimum(BLOCK, N - mx.arange(nb) * BLOCK).astype(mx.float32)
    return qs / lengths[:, None], ks / lengths[:, None], (vs / lengths[:, None]).astype(v.dtype), mx.log2(lengths)


def build_route(
    qc: mx.array,
    kc: mx.array,
    scale: float,
    tau: float,
    num_tokens: int,
    tokens_per_frame: int = 0,
    temporal: bool = True,
) -> mx.array:
    """Which key blocks each query block attends to exactly: ``(B, H, nb, nb)`` uint8.

    Routed: centroid score above ``mean + tau * std`` of the query block's centroid scores (Sol's "diag" estimate),
    the neighbouring blocks (``|i - j| <= 1``), with ``temporal`` the blocks holding the same positions one latent
    frame (``tokens_per_frame`` tokens) earlier and later.
    ``tau = -inf`` routes everything.
    """
    nb = qc.shape[2]
    if tau == -math.inf:
        return mx.ones((*qc.shape[:2], nb, nb), dtype=mx.uint8)
    l2 = scale * math.log2(math.e)
    i = mx.arange(nb)
    k_mean = kc.mean(axis=2)
    k_var = ((kc - k_mean[:, :, None]) ** 2).mean(axis=2)
    raw_mean = (qc * k_mean[:, :, None]).sum(-1)
    raw_var = (qc * qc * k_var[:, :, None]).sum(-1)
    thr = raw_mean * l2 + tau * mx.sqrt(mx.maximum(raw_var, 0.0) * l2 * l2 + 1e-6)
    scores = (qc @ kc.swapaxes(-1, -2)) * l2
    routed = (scores > thr[..., None]) | (mx.abs(i[:, None] - i[None, :]) <= 1)[None, None]
    if temporal and tokens_per_frame:
        for d in (-tokens_per_frame, tokens_per_frame):
            lo = mx.clip((i * BLOCK + d) // BLOCK, 0, nb - 1)
            hi = mx.clip((mx.minimum(i * BLOCK + BLOCK - 1, num_tokens - 1) + d) // BLOCK, 0, nb - 1)
            valid = ((i * BLOCK + d) < num_tokens) & ((i * BLOCK + BLOCK - 1 + d) >= 0)
            near = valid[:, None] & (i[None, :] >= lo[:, None]) & (i[None, :] <= hi[:, None])
            routed = routed | near[None, None]
    return routed.astype(mx.uint8)


def sparse_self_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    tau: float,
    tokens_per_frame: int = 0,
    temporal: bool = True,
    check: bool = True,
) -> mx.array:
    """Sol attention: ``(B, H, N, D)`` q, k, v (head dim contiguous, any other strides) -> ``(B, N, H, D)``.

    ``tau = -inf`` gives exactly ``mx.fast.scaled_dot_product_attention`` (same loop, every block routed).

    Raises:
        RuntimeError: If ``check`` and the kernels are unavailable in ``q.dtype`` (see :func:`kernel_available`).
    """
    if check and not kernel_available(q.dtype):
        raise RuntimeError("sparse attention kernel unavailable")
    B, H, N, D = q.shape
    qc, kc, vmean, log2_len = block_summaries(q, k, v)
    route = build_route(qc, kc, scale, tau, N, tokens_per_frame, temporal)
    bq, bk, wm = 32, 16, 4
    ip = mx.array([H, N, kc.shape[2]], dtype=mx.int32)
    sc = mx.array([scale], dtype=mx.float32)
    return _kernels()["attn"](
        inputs=[q, k, v, kc.astype(k.dtype), vmean, log2_len, route, ip, sc],
        template=[("T", q.dtype), ("BQ", bq), ("BK", bk), ("BD", D), ("WM", wm)],
        grid=(-(-N // bq) * 32, H * wm, B),
        threadgroup=(32, wm, 1),
        output_shapes=[(B, N, H, D)],
        output_dtypes=[q.dtype],
    )[0]


__all__ = [
    "BLOCK",
    "MIN_TOKENS",
    "SOL_TAU_ENV",
    "SparseAttentionState",
    "block_summaries",
    "build_route",
    "kernel_available",
    "sol_taus_from_env",
    "sparse_self_attention",
]
