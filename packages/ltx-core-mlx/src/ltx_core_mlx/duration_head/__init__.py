"""Duration head: predicts shot duration in seconds from Connector outputs."""

from ltx_core_mlx.duration_head.duration_head import (
    AttentionCrossAttn,
    AttentionPooler,
    DurationHead,
    load_duration_head,
)

__all__ = [
    "AttentionCrossAttn",
    "AttentionPooler",
    "DurationHead",
    "load_duration_head",
]
