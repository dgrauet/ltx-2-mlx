"""Gemma-4 unified text tower config, parsed from the LTX-2.5 pack.

LTX-2.5 swaps the Gemma 3 12B text encoder for the text tower of
Gemma4UnifiedForConditionalGeneration (``gemma4_unified_text``). Unlike
Gemma 3, Gemma 4 alternates two attention flavors per layer
(``layer_types``), each with its own head_dim / KV head count / RoPE
theta, and reads them from a nested HF-style config rather than a flat
dataclass. This module parses only the fields this port's tower needs
straight out of the pack's ``text_encoder_config.json`` (full JSON, with
the tower's own fields nested under ``text_config``).

Reference: HF ``configuration_gemma4_unified.py`` (Gemma4TextConfig).
"""

from __future__ import annotations

from dataclasses import dataclass

_TEXT_MODEL_TYPE = "gemma4_unified_text"
_ROOT_MODEL_TYPE = "gemma4_unified"
_SLIDING_ATTENTION = "sliding_attention"
_FULL_ATTENTION = "full_attention"

# Sliding layers use plain (non-scaled) RoPE; full-attention layers use Gemma
# 4's "proportional" scaling (see gemma4.py's rotary embedding). A pack with
# these two flipped parses fine under every other check here -- same key set,
# same types -- and silently computes the wrong rotary tables for every
# layer, so it needs its own explicit guard.
_EXPECTED_ROPE_TYPE = {
    _SLIDING_ATTENTION: "default",
    _FULL_ATTENTION: "proportional",
}


@dataclass
class Gemma4TextConfig:
    """Gemma-4 unified text tower configuration.

    Attributes:
        gemma_version: Version tag from the root of the JSON (e.g.
            ``"gemma4-12b-ltx-v1"``), or ``None`` if absent.
        hidden_size: Model (residual stream) width.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Query head count (uniform across layers).
        head_dim: Per-head dim for sliding-attention layers.
        global_head_dim: Per-head dim for full-attention layers.
        num_key_value_heads: KV head count for sliding-attention layers.
        num_global_key_value_heads: KV head count for full-attention layers.
        intermediate_size: MLP hidden width.
        vocab_size: Token vocabulary size.
        rms_norm_eps: Epsilon for RMSNorm layers.
        sliding_window: Sliding-attention window size (tokens).
        layer_types: Per-layer attention flavor, one of
            ``"sliding_attention"`` / ``"full_attention"``.
        attention_k_eq_v: Whether K and V projections are tied (share
            weights) in attention.
        rope_local_theta: RoPE base frequency for sliding-attention layers.
        rope_global_theta: RoPE base frequency for full-attention layers.
        partial_rotary_factor: Fraction of head_dim that is rotary,
            applies to full-attention layers only.
        pad_token_id: Padding token id.
    """

    gemma_version: str | None
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    global_head_dim: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    sliding_window: int
    layer_types: tuple[str, ...]
    attention_k_eq_v: bool
    rope_local_theta: float
    rope_global_theta: float
    partial_rotary_factor: float
    pad_token_id: int

    @classmethod
    def from_text_encoder_config(cls, config: dict) -> Gemma4TextConfig:
        """Parse a Gemma4TextConfig from the pack's full text encoder JSON.

        Args:
            config: The complete ``text_encoder_config.json`` content
                (root-level dict). The tower's own fields live under
                ``config["text_config"]``; ``gemma_version`` is read from
                the root.

        Returns:
            The parsed tower configuration.

        Raises:
            ValueError: If the root ``model_type`` is not
                ``"gemma4_unified"``, if ``text_config.model_type`` is not
                ``"gemma4_unified_text"``, if ``num_kv_shared_layers`` is
                nonzero (KV-sharing across layers is not ported), if
                ``use_bidirectional_attention == "all"`` (this port always
                encodes text causally), or if either attention type's
                ``rope_parameters.*.rope_type`` does not match the expected
                flavor (sliding: ``"default"``, full: ``"proportional"``).
        """
        root_model_type = config.get("model_type")
        if root_model_type != _ROOT_MODEL_TYPE:
            raise ValueError(f"Unsupported model_type: {root_model_type!r} (expected {_ROOT_MODEL_TYPE!r})")

        text_config = config["text_config"]

        model_type = text_config.get("model_type")
        if model_type != _TEXT_MODEL_TYPE:
            raise ValueError(f"Unsupported text_config.model_type: {model_type!r} (expected {_TEXT_MODEL_TYPE!r})")

        num_kv_shared_layers = text_config.get("num_kv_shared_layers", 0)
        if num_kv_shared_layers != 0:
            raise ValueError(
                f"num_kv_shared_layers={num_kv_shared_layers!r} is not supported "
                "(KV-sharing across layers is not ported)"
            )

        use_bidirectional_attention = text_config.get("use_bidirectional_attention", "vision")
        if use_bidirectional_attention == "all":
            raise ValueError(
                'use_bidirectional_attention="all" is not supported (this port always encodes text causally)'
            )

        rope_parameters = text_config["rope_parameters"]
        full_rope = rope_parameters[_FULL_ATTENTION]
        sliding_rope = rope_parameters[_SLIDING_ATTENTION]

        for attention_type, rope_config in ((_SLIDING_ATTENTION, sliding_rope), (_FULL_ATTENTION, full_rope)):
            expected_rope_type = _EXPECTED_ROPE_TYPE[attention_type]
            actual_rope_type = rope_config.get("rope_type")
            if actual_rope_type != expected_rope_type:
                raise ValueError(
                    f"Unsupported rope_parameters.{attention_type}.rope_type: {actual_rope_type!r} "
                    f"(expected {expected_rope_type!r})"
                )

        return cls(
            gemma_version=config.get("gemma_version"),
            hidden_size=text_config["hidden_size"],
            num_hidden_layers=text_config["num_hidden_layers"],
            num_attention_heads=text_config["num_attention_heads"],
            head_dim=text_config["head_dim"],
            global_head_dim=text_config["global_head_dim"],
            num_key_value_heads=text_config["num_key_value_heads"],
            num_global_key_value_heads=text_config["num_global_key_value_heads"],
            intermediate_size=text_config["intermediate_size"],
            vocab_size=text_config["vocab_size"],
            rms_norm_eps=text_config["rms_norm_eps"],
            sliding_window=text_config["sliding_window"],
            layer_types=tuple(text_config["layer_types"]),
            attention_k_eq_v=text_config.get("attention_k_eq_v", False),
            rope_local_theta=sliding_rope["rope_theta"],
            rope_global_theta=full_rope["rope_theta"],
            partial_rotary_factor=full_rope["partial_rotary_factor"],
            pad_token_id=text_config["pad_token_id"],
        )

    def layer_is_sliding(self, i: int) -> bool:
        """Whether layer ``i`` uses sliding-window (local) attention.

        Args:
            i: Zero-based layer index.

        Returns:
            True if the layer's type is ``"sliding_attention"``.
        """
        return self.layer_types[i] == _SLIDING_ATTENTION

    def layer_head_dim(self, i: int) -> int:
        """Per-head dimension for layer ``i``, by its attention type.

        Args:
            i: Zero-based layer index.

        Returns:
            ``head_dim`` for sliding layers, ``global_head_dim`` for full
            layers.
        """
        return self.head_dim if self.layer_is_sliding(i) else self.global_head_dim

    def layer_num_kv(self, i: int) -> int:
        """KV head count for layer ``i``, by its attention type.

        Args:
            i: Zero-based layer index.

        Returns:
            ``num_key_value_heads`` for sliding layers,
            ``num_global_key_value_heads`` for full layers.
        """
        return self.num_key_value_heads if self.layer_is_sliding(i) else self.num_global_key_value_heads
