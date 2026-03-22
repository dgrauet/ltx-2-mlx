"""Quantised Gemma text encoder loading utilities for MLX.

Ported from ltx-trainer (Lightricks). In MLX, quantised Gemma models are
loaded directly via ``mlx-lm`` — no bitsandbytes required. This module is
a thin wrapper matching the reference API.

Example usage::

    from ltx_trainer_mlx.gemma_8bit import load_quantized_gemma
    text_encoder = load_quantized_gemma("mlx-community/gemma-3-12b-it-8bit")
"""

from __future__ import annotations

import logging
from pathlib import Path

from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

logger = logging.getLogger(__name__)


def load_quantized_gemma(
    gemma_model_path: str | Path,
    max_tokens: int = 1024,
) -> GemmaLanguageModel:
    """Load Gemma text encoder using mlx-lm with quantisation.

    MLX handles quantisation natively through pre-quantised model weights
    (e.g. 4-bit or 8-bit models from ``mlx-community``). The
    ``GemmaLanguageModel`` from ``ltx_core_mlx`` loads these directly.

    Args:
        gemma_model_path: Path to local Gemma model directory or HuggingFace
            repo ID (e.g. ``"mlx-community/gemma-3-12b-it-4bit"``).
        max_tokens: Maximum number of tokens for the tokeniser.

    Returns:
        ``GemmaLanguageModel`` with quantised weights loaded.
    """
    logger.info("Loading quantised Gemma model from %s", gemma_model_path)

    model = GemmaLanguageModel(
        model_id=str(gemma_model_path),
        max_tokens=max_tokens,
    )

    return model
