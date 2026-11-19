"""Multi-head attention with RoPE and per-head gating.

Ported from ltx-core/src/ltx_core/model/transformer/attention.py

Weight keys (relative to parent, e.g. ``attn1``):
    ``to_q.{weight,bias}``, ``to_k.{weight,bias}``, ``to_v.{weight,bias}``
    ``to_out.{weight,bias}``          -- single Linear (NOT ``to_out.0``)
    ``to_gate_logits.{weight,bias}``  -- per-head gate logits
    ``q_norm.weight``, ``k_norm.weight`` -- RMSNorm over full inner_dim
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.guidance.nag import (
    DEFAULT_NAG_ALPHA,
    DEFAULT_NAG_SCALE,
    DEFAULT_NAG_TAU,
    normalized_attention_guidance,
)
from ltx_core_mlx.model.transformer.rope import apply_rope_interleaved, apply_rope_split
from ltx_core_mlx.model.transformer.sparse_attention import SparseAttentionState, sparse_self_attention


def _cast_inputs(
    dtype: mx.Dtype,
    x: mx.array,
    encoder_hidden_states: mx.array | None,
    attention_mask: mx.array | None,
    perturbation_mask: mx.array | None,
) -> tuple[mx.array, mx.array | None, mx.array | None, mx.array | None]:
    """Cast the attention inputs to ``dtype``.

    An additive mask is clamped to the dtype's finite range first, so a large
    negative bias (e.g. -1e9) becomes the dtype minimum rather than -inf: a row
    that is fully masked then still softmaxes to a finite result, as in float32.
    """
    if encoder_hidden_states is not None:
        encoder_hidden_states = encoder_hidden_states.astype(dtype)
    if attention_mask is not None and attention_mask.dtype != mx.bool_:
        info = mx.finfo(dtype)
        attention_mask = mx.clip(attention_mask, info.min, info.max).astype(dtype)
    if perturbation_mask is not None:
        perturbation_mask = perturbation_mask.astype(dtype)
    return x.astype(dtype), encoder_hidden_states, attention_mask, perturbation_mask


class Attention(nn.Module):
    """Multi-head attention with optional RoPE and per-head gating.

    The gate is computed as ``2 * sigmoid(to_gate_logits(x))`` so that
    zero-initialised logits produce an identity gate (value 1.0).

    Args:
        query_dim: Dimension of the query input.
        kv_dim: Dimension of the key/value input.  Defaults to *query_dim*.
        out_dim: Dimension of the output projection.  Defaults to *query_dim*.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        qkv_bias: Whether to use bias in Q/K/V projections.
        use_rope: Whether to apply rotary position embeddings.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int | None = None,
        out_dim: int | None = None,
        num_heads: int = 32,
        head_dim: int = 128,
        qkv_bias: bool = True,
        use_rope: bool = True,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = True,
    ):
        super().__init__()
        if kv_dim is None:
            kv_dim = query_dim
        if out_dim is None:
            out_dim = query_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.scale = head_dim**-0.5

        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, out_dim, bias=True)

        # Per-head gate: gate = 2 * sigmoid(logits), zero-init -> gate = 1
        # Reference: only created when apply_gated_attention=True
        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(query_dim, num_heads, bias=True)
        else:
            self.to_gate_logits = None

        # QK normalization (RMSNorm over full inner_dim, applied per-token)
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)

        # Optional dtype for everything inside this module (projections, q/k norms,
        # RoPE, the attention kernel). None keeps the input dtype. Set through
        # LTXModel.set_compute_dtype, which also casts the parameters to match.
        self.compute_dtype: mx.Dtype | None = None

        # Block-sparse self-attention (Sol routing), shared by the video self-attention modules while it is on;
        # None = dense. Set through LTXModel.set_sparse_attention.
        self.sparse_attention: SparseAttentionState | None = None

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array | None = None,
        rope_freqs: mx.array | None = None,
        rope_freqs_k: mx.array | None = None,
        attention_mask: mx.array | None = None,
        perturbation_mask: mx.array | None = None,
        nag_encoder_hidden_states: mx.array | None = None,
        nag_scale: float = DEFAULT_NAG_SCALE,
        nag_alpha: float = DEFAULT_NAG_ALPHA,
        nag_tau: float = DEFAULT_NAG_TAU,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (B, N, query_dim).
            encoder_hidden_states: Cross-attention keys/values if not None.
            rope_freqs: RoPE frequencies for Q (and K if rope_freqs_k is None).
            rope_freqs_k: Separate RoPE frequencies for K (cross-attention).
            attention_mask: Optional mask broadcastable to (B, 1, Nq, Nk). With NAG it masks
                the positive attention only.
            perturbation_mask: Optional STG mask, see the STG note below.
            nag_encoder_hidden_states: Negative context for Normalized Attention Guidance
                (cross-attention only), ``(1 or B, Nk', kv_dim)``. ``None`` (default) is the
                plain attention. The same queries attend to it unmasked, and the two outputs
                are combined by :func:`~ltx_core_mlx.guidance.nag.normalized_attention_guidance`
                (float32) before the per-head gate and the output projection.
            nag_scale: NAG extrapolation factor (used with ``nag_encoder_hidden_states``).
            nag_alpha: NAG blend with the positive output.
            nag_tau: NAG clip on the per-token L1-norm ratio.

        Returns:
            Output of shape (B, N, out_dim).
        """
        B, N, _ = x.shape
        if nag_encoder_hidden_states is not None and encoder_hidden_states is None:
            raise ValueError("NAG needs a cross-attention (encoder_hidden_states is None)")
        if self.compute_dtype is not None:
            x, encoder_hidden_states, attention_mask, perturbation_mask = _cast_inputs(
                self.compute_dtype, x, encoder_hidden_states, attention_mask, perturbation_mask
            )
            if nag_encoder_hidden_states is not None:
                nag_encoder_hidden_states = nag_encoder_hidden_states.astype(self.compute_dtype)
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else x

        q = self.to_q(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # QK normalization (over full inner_dim before reshaping)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to (B, num_heads, seq_len, head_dim)
        q = q.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE -- freqs are (cos, sin, rope_type) tuple
        if self.use_rope and rope_freqs is not None:
            cos_f, sin_f, rtype = rope_freqs
            _apply = apply_rope_split if rtype == "split" else apply_rope_interleaved
            q = _apply(q, cos_f, sin_f)

            if rope_freqs_k is not None:
                cos_fk, sin_fk, _ = rope_freqs_k
            else:
                cos_fk, sin_fk = cos_f, sin_f
            k = _apply(k, cos_fk, sin_fk)

        # Scaled dot-product attention (fused Flash Attention kernel). Plain self-attention calls (no mask, no STG
        # perturbation) run block-sparse while LTXModel.set_sparse_attention has a tau for the current step.
        sparse = self.sparse_attention
        tau = None
        if (
            sparse is not None
            and encoder_hidden_states is None
            and attention_mask is None
            and perturbation_mask is None
        ):
            tau = sparse.tau_for_call(N, q.dtype)
        if sparse is not None and tau is not None:
            out = sparse_self_attention(
                q, k, v, self.scale, tau, sparse.tokens_per_frame, sparse.temporal, check=False
            ).transpose(0, 2, 1, 3)
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attention_mask)

        # NAG: the same queries against the negative context (no mask: Prompt Relay gates the
        # positive prompt only), combined with the positive output over all heads per token.
        if nag_encoder_hidden_states is not None:
            k_neg = self.k_norm(self.to_k(nag_encoder_hidden_states))
            v_neg = self.to_v(nag_encoder_hidden_states)
            k_neg = k_neg.reshape(k_neg.shape[0], -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            v_neg = v_neg.reshape(v_neg.shape[0], -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            if k_neg.shape[0] != B:
                k_neg = mx.broadcast_to(k_neg, (B, *k_neg.shape[1:]))
                v_neg = mx.broadcast_to(v_neg, (B, *v_neg.shape[1:]))
            out_neg = mx.fast.scaled_dot_product_attention(q, k_neg, v_neg, scale=self.scale)
            out = normalized_attention_guidance(out, out_neg, nag_scale, nag_alpha, nag_tau, axis=(1, 3))

        # Back to token-major (B, N, num_heads, head_dim) first: the transpose is a view, so the
        # blend / gate below write their result straight in the layout the output projection
        # reads and the final reshape is free (gating in head-major order and transposing
        # afterwards wrote one more full copy). Same elementwise arithmetic either way.
        out = out.transpose(0, 2, 1, 3)

        # STG perturbation: blend attn output with value projection
        # Reference: out = attn_out * mask + v * (1 - mask)
        if perturbation_mask is not None:
            # perturbation_mask: (B, 1, 1, 1) — 1.0=keep attn, 0.0=skip to values
            out = out * perturbation_mask + v.transpose(0, 2, 1, 3) * (1.0 - perturbation_mask)

        # Per-head gating: gate = 2 * sigmoid(logits)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, N, num_heads)
            gate = 2.0 * mx.sigmoid(gate_logits)
            out = out * gate[:, :, :, None]  # (B, N, heads, 1)

        out = out.reshape(B, -1, self.num_heads * self.head_dim)
        return self.to_out(out)
