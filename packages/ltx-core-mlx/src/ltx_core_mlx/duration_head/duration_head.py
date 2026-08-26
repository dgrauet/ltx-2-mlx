"""DurationHead — predicts shot duration from frozen Connector outputs.

Ported from ltx-core/src/ltx_core/duration_head/duration_head.py.

The audio and video Connectors (``text_encoders/gemma/embeddings_connector.py``)
consume a Gemma caption embedding and emit per-token hidden states. Their output
dimension and sequence length are fixed by the production model, so a thin
regression head on top has all the signal it needs to predict the natural
duration of the implied shot -- without running the full diffusion pipeline.

The head is modality-agnostic: pass either or both of {audio, video} connector
outputs. Modality-specific input projections map each stream to a shared pooler
hidden dim, learnable modality embeddings tag the streams so an attention
pooler can tell them apart, and a small MLP turns the pooled vector into a
log-duration prediction. Targets (when training) are in log-seconds so the
loss spreads its budget evenly across orders of magnitude; at inference time
callers get seconds directly -- ``exp`` is applied inside ``__call__``.

Unlike the training-time head, this module never sees packed/multi-sample
sequences, so it has no segment-id bookkeeping. It also needs no attention
mask: the connector always substitutes learnable registers for padded
positions and marks the result as fully attendable, so every token handed to
this module is already valid.

Weight keys (under the ``duration_head.`` prefix in ``duration_head.safetensors``):
    ``video_input_proj.{weight,bias}``
    ``video_modality_emb``
    ``audio_input_proj.{weight,bias}``
    ``audio_modality_emb``
    ``attention_pooler.query_tokens``
    ``attention_pooler.cross_attn.in_proj_weight``   # torch nn.MultiheadAttention, fused (3*hidden, hidden)
    ``attention_pooler.cross_attn.in_proj_bias``     # fused (3*hidden,)
    ``attention_pooler.cross_attn.out_proj.{weight,bias}``
    ``mlp_hidden.{weight,bias}``
    ``mlp_out.{weight,bias}``
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.utils.weights import load_split_safetensors

_PACK_PREFIX = "duration_head."


class AttentionCrossAttn(nn.Module):
    """Manual multi-head cross-attention, replacing torch ``nn.MultiheadAttention``.

    Torch's ``nn.MultiheadAttention`` fuses q/k/v projections into a single
    ``in_proj_weight``/``in_proj_bias`` pair. The loader splits that fused
    weight into three separate projections so the module tree maps cleanly
    onto named submodules (``q_proj``, ``k_proj``, ``v_proj``, ``out_proj``).

    Args:
        hidden_dim: Model / embedding dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, queries: mx.array, tokens: mx.array) -> mx.array:
        """Cross-attend ``queries`` against ``tokens``.

        Args:
            queries: ``(B, Nq, hidden_dim)``.
            tokens: ``(B, Nk, hidden_dim)``.

        Returns:
            ``(B, Nq, hidden_dim)``.
        """
        batch, num_queries, _ = queries.shape
        num_tokens = tokens.shape[1]

        q = self.q_proj(queries).reshape(batch, num_queries, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(tokens).reshape(batch, num_tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(tokens).reshape(batch, num_tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v

        out = out.transpose(0, 2, 1, 3).reshape(batch, num_queries, self.num_heads * self.head_dim)
        return self.out_proj(out)


class AttentionPooler(nn.Module):
    """Cross-attend ``num_queries`` learnable tokens against ``tokens``.

    Produces a fixed-shape ``(B, num_queries, hidden_dim)`` output regardless
    of the input sequence length. Every position in ``tokens`` is attendable
    -- see the module docstring for why no mask is needed.

    Args:
        hidden_dim: Model / embedding dimension.
        num_queries: Number of learnable pooling queries.
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.query_tokens = mx.zeros((num_queries, hidden_dim))
        self.cross_attn = AttentionCrossAttn(hidden_dim=hidden_dim, num_heads=num_heads)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Pool ``tokens`` down to ``num_queries`` fixed slots.

        Args:
            tokens: ``(B, N, hidden_dim)``.

        Returns:
            ``(B, num_queries, hidden_dim)``.
        """
        batch = tokens.shape[0]
        queries = mx.broadcast_to(self.query_tokens[None, :, :], (batch, self.num_queries, self.hidden_dim))
        return self.cross_attn(queries, tokens)


class DurationHead(nn.Module):
    """Predict duration in seconds from one or both Connector outputs.

    Args:
        video_cross_attention_dim: Video connector output dim.
        audio_cross_attention_dim: Audio connector output dim.
        pooler_hidden_dim: Shared hidden dim both modalities are projected into.
        num_queries: Number of learnable pooling queries.
        num_pooler_heads: Attention heads used by the pooler.
        mlp_hidden_dim: Hidden width of the output MLP.
    """

    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pooler_hidden_dim = pooler_hidden_dim

        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = mx.zeros((pooler_hidden_dim,))

        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = mx.zeros((pooler_hidden_dim,))

        self.attention_pooler = AttentionPooler(
            hidden_dim=pooler_hidden_dim,
            num_queries=num_queries,
            num_heads=num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden_dim)
        self.mlp_out = nn.Linear(mlp_hidden_dim, 1)

    def __call__(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
    ) -> mx.array:
        """Predict duration in seconds.

        The regression target is trained in log-seconds (spreads the loss
        budget evenly across orders of magnitude -- see the module
        docstring), so the MLP output is a log-duration internally; ``exp``
        is applied here so callers always get seconds.

        Args:
            video_tokens: ``(B, T_v, video_cross_attention_dim)``, or ``None``.
            audio_tokens: ``(B, T_a, audio_cross_attention_dim)``, or ``None``.

        Returns:
            Duration prediction in seconds, shape ``(B,)``.

        Raises:
            ValueError: If neither ``video_tokens`` nor ``audio_tokens`` is given.
        """
        if video_tokens is None and audio_tokens is None:
            raise ValueError("DurationHead.__call__ requires at least one of video_tokens / audio_tokens")

        token_groups: list[mx.array] = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)

        tokens = mx.concatenate(token_groups, axis=1)
        pooled = self.attention_pooler(tokens)
        pooled_flat = pooled.reshape(pooled.shape[0], -1)
        hidden = nn.gelu_approx(self.mlp_hidden(pooled_flat))
        log_duration = self.mlp_out(hidden).squeeze(-1)
        return mx.exp(log_duration)


def load_duration_head(path: str | Path) -> DurationHead:
    """Build and load a :class:`DurationHead` from a pack's ``duration_head.safetensors``.

    Strips the ``duration_head.`` prefix and splits torch
    ``nn.MultiheadAttention``'s fused ``in_proj_weight``/``in_proj_bias``
    into separate q/k/v projections so the flat weight dict maps onto the
    named submodules (``attention_pooler.cross_attn.{q_proj,k_proj,v_proj}``).

    Args:
        path: Path to ``duration_head.safetensors`` (the file itself, not
            a directory).

    Returns:
        The loaded :class:`DurationHead`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"duration head weights not found at {path}")

    raw = load_split_safetensors(path, prefix=_PACK_PREFIX)

    hidden_dim = raw["video_modality_emb"].shape[0]
    video_dim = raw["video_input_proj.weight"].shape[1]
    audio_dim = raw["audio_input_proj.weight"].shape[1]
    mlp_hidden_dim = raw["mlp_hidden.weight"].shape[0]
    num_queries = raw["attention_pooler.query_tokens"].shape[0]
    num_heads = 4  # fixed by upstream architecture; not encoded in the pack

    in_proj_weight = raw.pop("attention_pooler.cross_attn.in_proj_weight")  # (3*hidden, hidden)
    in_proj_bias = raw.pop("attention_pooler.cross_attn.in_proj_bias")  # (3*hidden,)
    q_w, k_w, v_w = mx.split(in_proj_weight, 3, axis=0)
    q_b, k_b, v_b = mx.split(in_proj_bias, 3, axis=0)
    raw["attention_pooler.cross_attn.q_proj.weight"] = q_w
    raw["attention_pooler.cross_attn.q_proj.bias"] = q_b
    raw["attention_pooler.cross_attn.k_proj.weight"] = k_w
    raw["attention_pooler.cross_attn.k_proj.bias"] = k_b
    raw["attention_pooler.cross_attn.v_proj.weight"] = v_w
    raw["attention_pooler.cross_attn.v_proj.bias"] = v_b

    model = DurationHead(
        video_cross_attention_dim=video_dim,
        audio_cross_attention_dim=audio_dim,
        pooler_hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_pooler_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    model.load_weights(list(raw.items()))
    return model
