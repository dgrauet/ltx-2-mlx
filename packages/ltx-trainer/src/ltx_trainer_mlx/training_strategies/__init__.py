"""Training strategies for different conditioning modes.

This package implements the Strategy Pattern to handle different training modes:
- Text-to-video training (standard generation, optionally with audio)
- Video-to-video training (IC-LoRA mode with reference videos)

Each strategy encapsulates the specific logic for preparing model inputs
and computing loss.
"""

from __future__ import annotations

import logging

from ltx_trainer_mlx.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModalityInputs,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer_mlx.training_strategies.text_to_video import (
    TextToVideoConfig,
    TextToVideoStrategy,
)
from ltx_trainer_mlx.training_strategies.video_to_video import (
    VideoToVideoConfig,
    VideoToVideoStrategy,
)

logger = logging.getLogger(__name__)

# Type alias for all strategy config types
TrainingStrategyConfig = TextToVideoConfig | VideoToVideoConfig

__all__ = [
    "DEFAULT_FPS",
    "VIDEO_SCALE_FACTORS",
    "ModalityInputs",
    "ModelInputs",
    "TextToVideoConfig",
    "TextToVideoStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "VideoToVideoConfig",
    "VideoToVideoStrategy",
    "get_training_strategy",
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.

    The strategy is determined by the ``name`` field in the configuration.

    Args:
        config: Strategy-specific configuration with a ``name`` field.

    Returns:
        The appropriate training strategy instance.

    Raises:
        ValueError: If strategy name is not supported.
    """
    match config:
        case TextToVideoConfig():
            strategy = TextToVideoStrategy(config)
        case VideoToVideoConfig():
            strategy = VideoToVideoStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    audio_mode = "(audio enabled)" if getattr(config, "with_audio", False) else "(audio disabled)"
    logger.debug(
        "Using %s training strategy %s",
        strategy.__class__.__name__,
        audio_mode,
    )
    return strategy
