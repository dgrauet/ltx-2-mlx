"""Checkpoint-shape detection helpers shared across generation pipelines.

LTX-2.5 packs are distinguished from LTX-2.3 packs by their transformer
config, not by a file naming convention -- ``is_ltx25_pack`` reads the same
``embedded_config.json`` / ``config.json`` that :class:`LTXModelConfig`
already parses at load time, so the detection stays in sync with the
loader by construction instead of duplicating the field mapping.
"""

from __future__ import annotations

from pathlib import Path

from ltx_core_mlx.model.transformer.model import LTXModelConfig


def is_ltx25_pack(model_dir: str | Path) -> bool:
    """Detect whether a checkpoint directory holds an LTX-2.5 weight pack.

    LTX-2.5 checkpoints ship ``ff_bias=false`` in their transformer config
    (2.3 checkpoints omit the field, which defaults to ``True``). This is
    the same signal :class:`LTXModelConfig.from_checkpoint_config` uses to
    decide whether the feed-forward layers carry a bias term, so reusing it
    here keeps pack detection isomorphic with the loader instead of
    re-deriving a second heuristic.

    Args:
        model_dir: Directory containing the checkpoint's config files
            (``embedded_config.json`` or ``config.json``).

    Returns:
        ``True`` if the checkpoint config declares ``ff_bias=false``
        (LTX-2.5), ``False`` otherwise -- including when no config file is
        found, in which case :meth:`LTXModelConfig.from_checkpoint_dir`
        warns on stderr and falls back to the LTX-2.3-shaped defaults
        (``ff_bias=True``).
    """
    return LTXModelConfig.from_checkpoint_dir(model_dir).ff_bias is False
