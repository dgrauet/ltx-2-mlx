"""Gemma-4 unified text tower building blocks (norm, rotary, attention, MLP).

Port of HF ``modeling_gemma4_unified.py`` -- ``Gemma4UnifiedRMSNorm``,
``Gemma4UnifiedTextRotaryEmbedding``, ``Gemma4UnifiedTextAttention`` and
``Gemma4UnifiedTextMLP`` -- to MLX. The decoder layer and the full tower
are assembled on top of these in a separate module.

Gemma 4 alternates two attention flavors, and they differ by more than a
window size:

* ``sliding_attention``: window-``sliding_window`` causal attention,
  ``num_key_value_heads`` KV heads, ``head_dim`` per head, ``default``
  rope over ``rope_local_theta``, and a real ``v_proj``.
* ``full_attention``: plain causal attention, ``num_global_key_value_heads``
  KV heads (typically 1), ``global_head_dim`` per head, ``proportional``
  rope over ``rope_global_theta`` with ``partial_rotary_factor``, and --
  when ``attention_k_eq_v`` -- **no** ``v_proj`` at all: the values are
  the raw ``k_proj`` output (pre ``k_norm``, un-roped) pushed through a
  scale-free ``v_norm``.

Two details worth calling out because they diverge from Gemma 3:

* ``Gemma4UnifiedRMSNorm`` applies ``normed * weight``, **not** Gemma 3's
  ``normed * (1 + weight)``. Reproduced verbatim here.
* attention ``scaling`` is a hard ``1.0`` (not ``head_dim ** -0.5``); the
  q/k RMSNorms carry the scale instead.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .gemma4_config import Gemma4TextConfig

_ROPE_DEFAULT = "default"
_ROPE_PROPORTIONAL = "proportional"


class Gemma4RMSNorm(nn.Module):
    """RMSNorm as used throughout the Gemma-4 text tower.

    Normalizes in float32 and casts back to the input dtype, mirroring
    ``Gemma4UnifiedRMSNorm``. The scale is applied as a plain multiply --
    Gemma 4 does not use Gemma 3's ``(1 + weight)`` convention.

    Args:
        dim: Feature dimension being normalized (last axis).
        eps: Epsilon added to the mean square before the inverse sqrt.
        with_scale: Whether to register a learned ``weight``. The
            attention ``v_norm`` is scale-free.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = mx.ones((dim,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Normalize the last axis.

        Args:
            hidden_states: Tensor whose last axis has size ``dim``.

        Returns:
            The normalized tensor, in the input dtype.
        """
        dtype = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        normed = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        if self.with_scale:
            normed = normed * self.weight.astype(mx.float32)
        return normed.astype(dtype)


class Gemma4RotaryEmbedding(nn.Module):
    """Rotary tables for one Gemma-4 layer type.

    Covers both parametrizations the tower uses:

    * ``"default"`` -- ``inv_freq[i] = theta ** (-2i / head_dim)`` over the
      whole head dim.
    * ``"proportional"`` -- only the first ``int(partial_rotary_factor *
      head_dim // 2)`` frequencies are populated (with the same
      ``head_dim`` in the exponent denominator, *not* the rotary sub-dim);
      the remaining ones are **zero**, which makes those channels
      identity-rotated (cos 1 / sin 0) while keeping the table full width.

    Args:
        head_dim: Per-head dimension of the layer.
        theta: RoPE base frequency.
        rope_type: ``"default"`` or ``"proportional"``.
        partial_rotary_factor: Rotary fraction of ``head_dim``; only
            meaningful for ``"proportional"``.

    Raises:
        ValueError: On an unknown ``rope_type``, or a ``"default"`` type
            paired with a non-unit ``partial_rotary_factor``.
    """

    def __init__(
        self,
        head_dim: int,
        theta: float,
        rope_type: str = _ROPE_DEFAULT,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        if rope_type not in (_ROPE_DEFAULT, _ROPE_PROPORTIONAL):
            raise ValueError(f"Unsupported rope_type: {rope_type!r}")
        if rope_type == _ROPE_DEFAULT and partial_rotary_factor != 1.0:
            raise ValueError(
                f"rope_type='default' rotates the full head dim, got partial_rotary_factor={partial_rotary_factor}"
            )

        self.head_dim = head_dim
        self.theta = theta
        self.rope_type = rope_type
        self.partial_rotary_factor = partial_rotary_factor

        half = head_dim // 2
        rope_angles = int(partial_rotary_factor * head_dim // 2) if rope_type == _ROPE_PROPORTIONAL else half
        exponents = mx.arange(0, 2 * rope_angles, 2, dtype=mx.float32) / head_dim
        inv_freq = 1.0 / mx.power(mx.array(theta, dtype=mx.float32), exponents)
        if rope_angles < half:
            inv_freq = mx.concatenate([inv_freq, mx.zeros((half - rope_angles,), dtype=mx.float32)])
        # Buffer, not a parameter: frozen so it is never trained/saved.
        self._inv_freq = inv_freq
        self.freeze(keys=["_inv_freq"])

    @property
    def inv_freq(self) -> mx.array:
        """The inverse frequency table, shape ``(head_dim // 2,)``."""
        return self._inv_freq

    @classmethod
    def from_config(cls, config: Gemma4TextConfig, layer_idx: int) -> Gemma4RotaryEmbedding:
        """Build the rotary embedding for layer ``layer_idx``.

        Sliding layers use ``default`` rope over ``rope_local_theta``;
        full layers use ``proportional`` rope over ``rope_global_theta``
        with the config's ``partial_rotary_factor``.

        Args:
            config: The parsed text tower config.
            layer_idx: Zero-based layer index.

        Returns:
            The rotary embedding for that layer.
        """
        if config.layer_is_sliding(layer_idx):
            return cls(config.head_dim, config.rope_local_theta, rope_type=_ROPE_DEFAULT)
        return cls(
            config.global_head_dim,
            config.rope_global_theta,
            rope_type=_ROPE_PROPORTIONAL,
            partial_rotary_factor=config.partial_rotary_factor,
        )

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Compute the cos/sin tables for the given positions.

        Args:
            position_ids: Integer positions, shape ``(B, T)``.

        Returns:
            ``(cos, sin)``, each of shape ``(B, T, head_dim)``.
        """
        freqs = position_ids.astype(mx.float32)[..., None] * self._inv_freq[None, None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def rotate_half(x: mx.array) -> mx.array:
    """Rotate the two halves of the last axis: ``[a, b] -> [-b, a]``.

    Args:
        x: Tensor with an even-sized last axis.

    Returns:
        The half-rotated tensor.
    """
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary_pos_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary position embeddings to a ``(B, T, H, D)`` tensor.

    Args:
        x: Query or key states, shape ``(B, T, H, D)``.
        cos: Cosine table, shape ``(B, T, D)``.
        sin: Sine table, shape ``(B, T, D)``.

    Returns:
        The rotated tensor, same shape as ``x``.
    """
    cos = cos[:, :, None, :].astype(x.dtype)
    sin = sin[:, :, None, :].astype(x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def build_attention_mask(
    seq_len: int,
    sliding_window: int | None = None,
    padding_mask: mx.array | None = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Build the additive causal (optionally windowed) attention mask.

    Mirrors the HF ``create_causal_mask`` /
    ``create_sliding_window_causal_mask`` pattern: a key at position
    ``j`` is visible from query position ``i`` when ``j <= i`` and, on
    sliding layers, ``j > i - sliding_window``.

    Args:
        seq_len: Query/key sequence length.
        sliding_window: Window size for sliding layers, ``None`` for full
            attention.
        padding_mask: Optional ``(B, T)`` mask, nonzero on real tokens.
        dtype: Output dtype.

    Returns:
        An additive mask of shape ``(1, 1, T, T)`` -- or ``(B, 1, T, T)``
        when ``padding_mask`` is given -- with ``0`` on visible positions
        and ``-inf`` on masked ones.
    """
    q_idx = mx.arange(seq_len)[:, None]
    k_idx = mx.arange(seq_len)[None, :]
    visible = k_idx <= q_idx
    if sliding_window is not None:
        visible = mx.logical_and(visible, k_idx > q_idx - sliding_window)
    visible = visible[None, None, :, :]

    if padding_mask is not None:
        visible = mx.logical_and(visible, (padding_mask != 0)[:, None, None, :])

    return mx.where(visible, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))


def repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    """Repeat KV heads to match the query head count (GQA).

    Args:
        hidden_states: ``(B, n_kv_heads, T, D)`` key or value states.
        n_rep: Repetition factor (``num_key_value_groups``).

    Returns:
        ``(B, n_kv_heads * n_rep, T, D)``, each KV head repeated
        contiguously -- equivalent to ``repeat_interleave`` on axis 1.
    """
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = mx.broadcast_to(hidden_states[:, :, None, :, :], (batch, n_kv_heads, n_rep, seq_len, head_dim))
    return expanded.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class Gemma4Attention(nn.Module):
    """Gemma-4 multi-head attention, both layer flavors.

    Args:
        config: The parsed text tower config.
        layer_idx: Zero-based layer index; selects the flavor via
            ``config.layer_types``.
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.head_dim = config.layer_head_dim(layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.layer_num_kv(layer_idx)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        # K and V are tied only on the full-attention layers.
        self.k_eq_v = config.attention_k_eq_v and not self.is_sliding
        self.scaling = 1.0

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        if not self.k_eq_v:
            self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Run attention over ``hidden_states``.

        Args:
            hidden_states: ``(B, T, hidden_size)`` layer input (already
                through ``input_layernorm``).
            cos: Rotary cosine table, ``(B, T, head_dim)``.
            sin: Rotary sine table, ``(B, T, head_dim)``.
            mask: Additive attention mask broadcastable to
                ``(B, H, T, T)``.

        Returns:
            ``(B, T, hidden_size)`` attention output.
        """
        batch, seq_len, _ = hidden_states.shape

        queries = self.q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, self.head_dim)
        queries = self.q_norm(queries)
        queries = apply_rotary_pos_emb(queries, cos, sin)
        queries = queries.transpose(0, 2, 1, 3)

        keys = self.k_proj(hidden_states).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        # On k_eq_v layers the values are the *raw* k_proj output: pre
        # k_norm and un-roped, normalized only by the scale-free v_norm.
        values = (
            keys
            if self.k_eq_v
            else self.v_proj(hidden_states).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        )

        keys = self.k_norm(keys)
        keys = apply_rotary_pos_emb(keys, cos, sin)
        keys = keys.transpose(0, 2, 1, 3)

        values = self.v_norm(values).transpose(0, 2, 1, 3)

        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        attn = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scaling, mask=mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn)


class Gemma4MLP(nn.Module):
    """Gated feed-forward block with the ``gelu_pytorch_tanh`` activation.

    Args:
        config: The parsed text tower config.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the gated MLP.

        Args:
            x: ``(B, T, hidden_size)`` input.

        Returns:
            ``(B, T, hidden_size)`` output.
        """
        return self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x))


def gelu_tanh(x: mx.array) -> mx.array:
    """The ``gelu_pytorch_tanh`` activation.

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`` --
    identical to ``torch.nn.functional.gelu(approximate="tanh")``.

    Args:
        x: Input tensor.

    Returns:
        The activated tensor.
    """
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3))))
