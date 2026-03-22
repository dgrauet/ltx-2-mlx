"""Quantization utilities for LTX-2 MLX training.

Ported from ltx-trainer (Lightricks). Replaces optimum-quanto with MLX native
``nn.quantize``, using the pattern from ``ltx_core_mlx/utils/weights.py``.

Only ``nn.Linear`` layers inside ``transformer_blocks`` are quantized.
Non-quantizable layers (adaln, proj_out, patchify_proj, connectors, VAE,
vocoder) are left in full precision.
"""

from __future__ import annotations

import logging
from typing import Literal

import mlx.nn as nn
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

logger = logging.getLogger(__name__)

QuantizationOptions = Literal[
    "int8",
    "int4",
]

# Modules to skip entirely during quantisation.
SKIP_ROOT_MODULES = {
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
    "audio_caption_projection",
    "caption_projection",
    "adaln_single",
    "audio_adaln_single",
    "prompt_adaln_single",
    "audio_prompt_adaln_single",
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_v2a_gate_adaln_single",
    "time_proj",
    "timestep_embedder",
    "scale_shift_table",
    "audio_scale_shift_table",
}


def quantize_model(
    model: nn.Module,
    precision: QuantizationOptions,
    group_size: int = 64,
) -> nn.Module:
    """Quantize a model using MLX native quantisation.

    Quantises only ``nn.Linear`` layers inside ``transformer_blocks``,
    skipping layers listed in ``SKIP_ROOT_MODULES``.

    Args:
        model: The ``nn.Module`` to quantise.
        precision: Quantisation bit width (``"int8"`` or ``"int4"``).
        group_size: Group size for quantisation.

    Returns:
        The quantised model (mutated in-place).
    """
    bits = 8 if precision == "int8" else 4

    if not hasattr(model, "transformer_blocks"):
        logger.warning("Model has no transformer_blocks; applying quantisation to all Linear layers")
        nn.quantize(model, group_size=group_size, bits=bits)
        return model

    # Block-by-block quantisation for memory efficiency
    blocks = model.transformer_blocks
    logger.debug("Quantising %d transformer blocks to %d-bit (group_size=%d)", len(blocks), bits, group_size)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Quantising transformer blocks", total=len(blocks))

        for block in blocks:
            nn.quantize(block, group_size=group_size, bits=bits)
            progress.advance(task)

    return model
