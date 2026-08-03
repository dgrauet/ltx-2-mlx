"""Adaptive Layer Norm.

Ported from ltx-core/src/ltx_core/model/transformer/adaln.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.transformer.timestep_embedding import TimestepEmbedder


class AdaLayerNormSingle(nn.Module):
    """Adaptive Layer Norm that produces modulation parameters from timestep.

    Weight keys:
        ``emb.timestep_embedder.linear1.{weight,bias}``
        ``emb.timestep_embedder.linear2.{weight,bias}``
        ``linear.{weight,bias}``

    Args:
        dim: Model dimension (output of timestep MLP and input to linear).
        num_params: Number of modulation parameters to produce.
        timestep_dim: Dimension of the input timestep embedding. Defaults to dim.
    """

    def __init__(self, dim: int, num_params: int = 6, timestep_dim: int | None = None):
        super().__init__()
        self.num_params = num_params
        t_dim = timestep_dim or dim
        self.emb = TimestepEmbedder(t_dim, dim)
        self.linear = nn.Linear(dim, num_params * dim)

    def __call__(self, timestep_emb: mx.array) -> tuple[mx.array, mx.array]:
        """Compute adaptive norm parameters.

        Args:
            timestep_emb: Timestep embedding of shape (B, timestep_dim).

        Returns:
            Tuple of (modulation_params, embedded_timestep) where:
            - modulation_params: shape (B, num_params * dim)
            - embedded_timestep: shape (B, dim) -- intermediate embedding
              used by the output block for adaptive scale/shift.
        """
        embedded = self.emb(timestep_emb)
        params = self.linear(nn.silu(embedded))
        return params, embedded


class PerTokenAdaLNParams:
    """Per-token AdaLN parameters held in deduplicated form.

    With per-token timesteps the AdaLN projection produces a
    ``(B, N, num_params * dim)`` float32 tensor (the sinusoidal timestep
    embedding is float32 and MLX promotes), which stays resident for the whole
    transformer stack. At video resolutions that is gigabytes -- 2.95 GB for
    the 9-parameter video head alone at 20 000 tokens -- even though the tensor
    only ever holds a handful of distinct rows, one per conditioning group.

    This carries the distinct rows plus the inverse index instead, and expands
    a single ``(B, N, dim)`` parameter slice at a time, inside the block that
    consumes it. Blocks obtain those slices through
    ``BasicAVTransformerBlock._unpack_adaln``.

    Expansion is bitwise identical to gathering up front: a gather is a pure
    row permutation and the elementwise scale-shift-table add commutes with it
    exactly (both operate element by element, with no reduction whose order
    could change).
    """

    __slots__ = ("batch", "inverse", "num_params", "tokens", "unique")

    def __init__(
        self,
        unique: mx.array,
        inverse: mx.array,
        batch: int,
        tokens: int,
        num_params: int,
    ):
        """
        Args:
            unique: (U, num_params * dim) distinct rows of the projection.
            inverse: (B * N,) row index into ``unique`` for every token.
            batch: B.
            tokens: N.
            num_params: Number of modulation parameters packed per row.
        """
        self.unique = unique
        self.inverse = inverse
        self.batch = int(batch)
        self.tokens = int(tokens)
        self.num_params = int(num_params)

    # -- mx.array-compatible surface, so callers can introspect without
    #    materialising anything --------------------------------------------
    @property
    def ndim(self) -> int:
        return 3

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.batch, self.tokens, int(self.unique.shape[-1]))

    @property
    def dtype(self):
        return self.unique.dtype

    @property
    def unique_rows(self) -> int:
        return int(self.unique.shape[0])

    def gather(self) -> mx.array:
        """Materialise the full ``(B, N, num_params * dim)`` tensor."""
        return mx.take(self.unique, self.inverse, axis=0).reshape(self.batch, self.tokens, -1)

    def unpack(self, table: mx.array, num_params: int, dim: int) -> list[mx.array]:
        """Expand per parameter: ``[(B, N, dim)] * num_params``.

        Equivalent, bit for bit, to unpacking ``self.gather()`` the eager way
        (reshape to ``(B, N, P, dim)``, add ``table``, slice each ``P``).
        """
        packed = self.unique.reshape(-1, self.num_params, dim)[:, :num_params, :]
        packed = packed + table[None, :num_params, :]
        return [
            mx.take(packed[:, i, :], self.inverse, axis=0).reshape(self.batch, self.tokens, dim)
            for i in range(num_params)
        ]
