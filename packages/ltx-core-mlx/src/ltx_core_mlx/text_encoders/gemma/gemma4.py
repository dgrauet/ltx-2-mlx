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

import json
import math
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

from .gemma4_config import Gemma4TextConfig

_ROPE_DEFAULT = "default"
_ROPE_PROPORTIONAL = "proportional"

_SLIDING_ATTENTION = "sliding_attention"
_FULL_ATTENTION = "full_attention"

# The pack's text_encoder.safetensors also carries the multimodal (vision +
# audio/video projector) tensors of the wider Gemma4Unified checkpoint that
# this text-only port does not load: 9 vision_model tensors, 1 audio
# projector, 1 multi-modal projector, and the 4 text_embedding_projection
# tensors (loaded by the connector, not the tower -- see
# ``text_encoders/gemma/embeddings_connector.py``). Any pack key that is
# neither under ``text_encoder.model.`` nor in this allowlist is a real
# format-drift signal and must fail loudly rather than being silently
# dropped.
GEMMA4_PACK_IGNORE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "text_encoder.audio_projector.embedding_projection.weight",
        "text_encoder.multi_modal_projector.embedding_projection.weight",
        "text_encoder.text_embedding_projection.audio_aggregate_embed.bias",
        "text_encoder.text_embedding_projection.audio_aggregate_embed.weight",
        "text_encoder.text_embedding_projection.video_aggregate_embed.bias",
        "text_encoder.text_embedding_projection.video_aggregate_embed.weight",
        "text_encoder.vision_model.patch_dense.bias",
        "text_encoder.vision_model.patch_dense.weight",
        "text_encoder.vision_model.patch_ln1.bias",
        "text_encoder.vision_model.patch_ln1.weight",
        "text_encoder.vision_model.patch_ln2.bias",
        "text_encoder.vision_model.patch_ln2.weight",
        "text_encoder.vision_model.pos_embedding",
        "text_encoder.vision_model.pos_norm.bias",
        "text_encoder.vision_model.pos_norm.weight",
    }
)

_TEXT_ENCODER_PACK_PREFIX = "text_encoder.model."


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
        # A padded query position can end up with zero visible keys (e.g.
        # left-padding: causal + its own key being masked leaves nothing),
        # which turns that softmax row into all -inf. 0 * NaN then poisons
        # every downstream position once k_eq_v layers reuse attention
        # output as the next layer's values. HF's masking utils sidestep
        # this the same way (``AttentionMaskConverter._unmask_unattended``):
        # unmask the diagonal on any row left fully masked. The resulting
        # hidden state at that position is unused by callers regardless.
        row_all_masked = mx.logical_not(mx.any(visible, axis=-1, keepdims=True))
        diag = (mx.arange(seq_len)[None, :] == mx.arange(seq_len)[:, None])[None, None, :, :]
        visible = mx.logical_or(visible, mx.logical_and(row_all_masked, diag))

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


class Gemma4DecoderLayer(nn.Module):
    """One Gemma-4 decoder layer: pre/post norm sandwich around attn + MLP.

    Unlike Gemma 3's single pre-norm per sublayer, Gemma 4 wraps each
    sublayer with both an input norm and a post-sublayer norm before the
    residual add, and finishes with a per-layer scalar gate (``layer_scalar``,
    a real pack tensor -- not just a constant -- even though its trained
    value is typically ~1).

    Args:
        config: The parsed text tower config.
        layer_idx: Zero-based layer index; selects the attention flavor.
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        hidden_states: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Run one decoder layer.

        Args:
            hidden_states: ``(B, T, hidden_size)`` layer input.
            cos: Rotary cosine table for this layer's attention flavor.
            sin: Rotary sine table for this layer's attention flavor.
            mask: Additive attention mask for this layer's flavor.

        Returns:
            ``(B, T, hidden_size)`` layer output.
        """
        residual = hidden_states
        attn_out = self.self_attn(self.input_layernorm(hidden_states), cos, sin, mask)
        hidden_states = residual + self.post_attention_layernorm(attn_out)

        residual = hidden_states
        mlp_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(mlp_out)

        return hidden_states * self.layer_scalar


def _safetensors_header_keys(path: Path) -> set[str]:
    """Read only the JSON header of a safetensors file (no tensor data).

    Args:
        path: Path to the ``.safetensors`` file.

    Returns:
        The set of tensor keys in the file.
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {k for k in header if k != "__metadata__"}


class Gemma4TextModel(nn.Module):
    """The Gemma-4 unified text tower: scaled embeddings + N decoder layers + final norm.

    Port of ``Gemma4UnifiedTextModel`` (text-only path: no vision/audio
    inputs, no KV cache, no shared-KV layers -- ruled out by
    :meth:`Gemma4TextConfig.from_text_encoder_config`).

    Args:
        config: The parsed text tower config.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = math.sqrt(config.hidden_size)
        self.layers = [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Keyed by layer_types string, not layer index: rotary tables only
        # depend on the attention flavor. Leading underscore keeps these
        # out of the MLX parameter tree -- their inv_freq buffers are
        # derived from config, not loaded from the pack.
        self._rotary_by_type = {
            _SLIDING_ATTENTION: Gemma4RotaryEmbedding(
                config.head_dim, config.rope_local_theta, rope_type=_ROPE_DEFAULT
            ),
            _FULL_ATTENTION: Gemma4RotaryEmbedding(
                config.global_head_dim,
                config.rope_global_theta,
                rope_type=_ROPE_PROPORTIONAL,
                partial_rotary_factor=config.partial_rotary_factor,
            ),
        }

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> list[mx.array]:
        """Run the tower and collect every layer's hidden states.

        Args:
            input_ids: Token ids, shape ``(B, T)``.
            attention_mask: Optional ``(B, T)`` mask, nonzero on real
                (non-padding) tokens. This port always left-pads.

        Returns:
            ``num_hidden_layers + 1`` hidden states, mirroring
            ``output_hidden_states=True``: the scaled token embeddings,
            then the raw (pre-final-norm) output of each decoder layer in
            order. The final RMSNorm is *not* folded into the last entry
            here -- call ``self.norm`` on it explicitly to get
            ``last_hidden_state``.
        """
        batch, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * mx.array(self.embed_scale, dtype=hidden_states.dtype)

        position_ids = mx.arange(seq_len)[None, :]
        position_ids = mx.broadcast_to(position_ids, (batch, seq_len))

        rope_tables = {
            layer_type: self._rotary_by_type[layer_type](position_ids)
            for layer_type in sorted(set(self.config.layer_types))
        }
        masks = {
            _FULL_ATTENTION: build_attention_mask(
                seq_len, sliding_window=None, padding_mask=attention_mask, dtype=hidden_states.dtype
            ),
            _SLIDING_ATTENTION: build_attention_mask(
                seq_len,
                sliding_window=self.config.sliding_window,
                padding_mask=attention_mask,
                dtype=hidden_states.dtype,
            ),
        }

        hidden_states_list = [hidden_states]
        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            cos, sin = rope_tables[layer_type]
            hidden_states = layer(hidden_states, cos, sin, masks[layer_type])
            hidden_states_list.append(hidden_states)

        return hidden_states_list

    @classmethod
    def load_from_pack(cls, model_dir: str | Path) -> Gemma4TextModel:
        """Build and load a Gemma-4 tower from an LTX-2.5 pack directory.

        Reads ``text_encoder_config.json`` to build the config, then loads
        ``text_encoder.safetensors`` -- everything under the
        ``text_encoder.model.`` prefix feeds the tower; every other key
        must be in :data:`GEMMA4_PACK_IGNORE_ALLOWLIST` (the multimodal
        tensors owned by other components) or this raises. Quantization is
        applied when ``quantize_config.json`` declares a ``text_encoder``
        component.

        Args:
            model_dir: Path to the pack directory.

        Returns:
            The loaded tower.

        Raises:
            ValueError: If ``text_encoder.safetensors`` has a key that is
                neither under the model prefix nor in the ignore allowlist
                (silent format drift -- see #52's silent-no-op class).
        """
        model_dir = Path(model_dir)

        with open(model_dir / "text_encoder_config.json") as f:
            raw_config = json.load(f)
        config = Gemma4TextConfig.from_text_encoder_config(raw_config)
        model = cls(config)

        safetensors_path = model_dir / "text_encoder.safetensors"
        all_keys = _safetensors_header_keys(safetensors_path)
        stray = {
            k for k in all_keys if not k.startswith(_TEXT_ENCODER_PACK_PREFIX) and k not in GEMMA4_PACK_IGNORE_ALLOWLIST
        }
        if stray:
            raise ValueError(
                f"{safetensors_path.name} has keys outside the text tower's "
                f"prefix ({_TEXT_ENCODER_PACK_PREFIX!r}) and the ignore "
                f"allowlist -- refusing to silently drop them: {sorted(stray)[:10]}"
            )

        weights = load_split_safetensors(safetensors_path, prefix=_TEXT_ENCODER_PACK_PREFIX)

        quantize_config_path = model_dir / "quantize_config.json"
        if quantize_config_path.exists():
            with open(quantize_config_path) as f:
                quant_config = json.load(f)
            quantization = quant_config.get("quantization", {})
            if "text_encoder" in quantization.get("components", {}):
                apply_quantization(model, weights, group_size=quantization.get("group_size", 64))

        model.load_weights(list(weights.items()))
        return model
