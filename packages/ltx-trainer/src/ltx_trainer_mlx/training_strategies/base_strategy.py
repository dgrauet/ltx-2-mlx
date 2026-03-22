"""Base class for training strategies.

This module defines the abstract base class that all training strategies must
implement, along with the base configuration class.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core_mlx.utils.positions import (
    VIDEO_SPATIAL_SCALE,
    VIDEO_TEMPORAL_SCALE,
    compute_audio_positions,
    compute_video_positions,
)

# Default frames per second for video missing FPS metadata
DEFAULT_FPS: float = 24.0

# VAE scale factors for LTX-2
VIDEO_SCALE_FACTORS: dict[str, int] = {
    "temporal": VIDEO_TEMPORAL_SCALE,
    "spatial": VIDEO_SPATIAL_SCALE,
}


class TrainingStrategyConfigBase:
    """Base configuration class for training strategies.

    All strategy-specific configuration classes should inherit from this.

    Attributes:
        name: Unique name identifying the training strategy type.
    """

    name: Literal["text_to_video", "video_to_video"]

    def __init__(self, name: str) -> None:
        self.name = name


@dataclass
class ModalityInputs:
    """Container for a single modality's inputs to the transformer.

    Mirrors the reference ``Modality`` dataclass fields, using ``mx.array``.

    Attributes:
        enabled: Whether this modality is active.
        latent: Patchified latent tokens, shape ``(B, T, D)``.
        sigma: Current sigma value, shape ``(B,)``.
        timesteps: Per-token timesteps, shape ``(B, T)``.
        positions: Positional coordinates, shape ``(1, T, 3)`` for video
            or ``(1, T, 1)`` for audio.
        context: Text conditioning embeddings.
        context_mask: Optional mask for text context tokens.
        attention_mask: Optional self-attention mask, shape ``(B, T, T)``.
    """

    enabled: bool
    latent: mx.array
    sigma: mx.array
    timesteps: mx.array
    positions: mx.array
    context: mx.array
    context_mask: mx.array | None = None
    attention_mask: mx.array | None = None


@dataclass
class ModelInputs:
    """Container for model inputs, targets, and loss masks.

    Attributes:
        video: Video modality inputs.
        audio: Audio modality inputs (``None`` for video-only training).
        video_targets: Velocity targets for video, shape ``(B, T_v, C)``.
        audio_targets: Velocity targets for audio (``None`` for video-only).
        video_loss_mask: Boolean mask for video loss, shape ``(B, T_v)``.
            ``True`` = compute loss for this token.
        audio_loss_mask: Boolean mask for audio loss (``None`` for video-only).
        ref_seq_len: For IC-LoRA: length of the reference sequence prepended
            to the video latent sequence.
    """

    video: ModalityInputs
    audio: ModalityInputs | None

    video_targets: mx.array
    audio_targets: mx.array | None

    video_loss_mask: mx.array
    audio_loss_mask: mx.array | None

    ref_seq_len: int | None = None


class TrainingStrategy(ABC):
    """Abstract base class for training strategies.

    Each strategy encapsulates the logic for a specific training mode,
    handling input preparation and loss computation.
    """

    def __init__(self, config: TrainingStrategyConfigBase) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Strategy-specific configuration.
        """
        self.config = config
        self._video_patchifier = VideoLatentPatchifier()
        self._audio_patchifier = AudioPatchifier()

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components.

        Override in subclasses that support audio training.  The trainer uses
        this to determine whether to load audio VAE and vocoder.
        """
        return False

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Get the required data sources for this training strategy.

        Returns:
            Either a list of data directory names (where output keys match
            directory names) or a dictionary mapping data directory names
            to custom output keys for the dataset.
        """

    @abstractmethod
    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        sigma_sampler: Any,
    ) -> ModelInputs:
        """Prepare training inputs from a raw data batch.

        Args:
            batch: Raw batch data from the dataset.
            sigma_sampler: Callable that returns sampled sigma values.

        Returns:
            ``ModelInputs`` containing modality objects and training targets.
        """

    @abstractmethod
    def compute_loss(
        self,
        video_pred: mx.array,
        audio_pred: mx.array | None,
        inputs: ModelInputs,
    ) -> mx.array:
        """Compute the training loss.

        Args:
            video_pred: Video prediction from the transformer model.
            audio_pred: Audio prediction (``None`` for video-only).
            inputs: The prepared model inputs containing targets and masks.

        Returns:
            Scalar loss value.
        """

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get strategy-specific metadata to include in checkpoint files.

        Override in subclasses to add custom metadata (e.g. parameters that
        a downstream inference pipeline may need).

        Returns:
            Dictionary of metadata key-value pairs (JSON-serializable).
        """
        return {}

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_video_positions(
        num_frames: int,
        height: int,
        width: int,
        fps: float,
    ) -> mx.array:
        """Generate video position embeddings.

        Uses ``compute_video_positions`` from ``ltx_core_mlx.utils.positions``
        which returns pixel-space midpoints with causal fix, temporal/fps.

        Args:
            num_frames: Number of latent frames.
            height: Latent height.
            width: Latent width.
            fps: Frames per second.

        Returns:
            Position tensor of shape ``(1, F*H*W, 3)``.
        """
        return compute_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )

    @staticmethod
    def _get_audio_positions(num_time_steps: int) -> mx.array:
        """Generate audio position embeddings.

        Uses ``compute_audio_positions`` from ``ltx_core_mlx.utils.positions``
        which returns real-time seconds.

        Args:
            num_time_steps: Number of audio time steps ``T``.

        Returns:
            Position tensor of shape ``(1, T, 1)``.
        """
        return compute_audio_positions(num_tokens=num_time_steps)

    # ------------------------------------------------------------------
    # Timestep / mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_per_token_timesteps(
        conditioning_mask: mx.array,
        sampled_sigma: mx.array,
    ) -> mx.array:
        """Create per-token timesteps based on conditioning mask.

        Args:
            conditioning_mask: Boolean mask of shape ``(B, T)`` where ``True``
                indicates a conditioning token (timestep=0).
            sampled_sigma: Sampled sigma values of shape ``(B,)``
                or ``(B, 1)``.

        Returns:
            Timesteps tensor of shape ``(B, T)``.
        """
        # Expand sigma to match conditioning mask shape [B, T]
        expanded_sigma = mx.broadcast_to(
            sampled_sigma.reshape(-1, 1),
            conditioning_mask.shape,
        )
        # Conditioning tokens get 0, target tokens get sampled sigma
        return mx.where(conditioning_mask, mx.zeros_like(expanded_sigma), expanded_sigma)

    @staticmethod
    def _create_first_frame_conditioning_mask(
        batch_size: int,
        sequence_length: int,
        height: int,
        width: int,
        first_frame_conditioning_p: float = 0.0,
    ) -> mx.array:
        """Create conditioning mask for first-frame conditioning.

        Args:
            batch_size: Batch size.
            sequence_length: Total sequence length.
            height: Latent height.
            width: Latent width.
            first_frame_conditioning_p: Probability of conditioning on the
                first frame.

        Returns:
            Boolean mask of shape ``(B, T)`` where ``True`` indicates
            first-frame tokens (if conditioning is active for this call).
        """
        mask = mx.zeros((batch_size, sequence_length), dtype=mx.bool_)

        if first_frame_conditioning_p > 0 and random.random() < first_frame_conditioning_p:
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                # Set first-frame tokens to True
                ones = mx.ones((batch_size, first_frame_end_idx), dtype=mx.bool_)
                zeros = mx.zeros((batch_size, sequence_length - first_frame_end_idx), dtype=mx.bool_)
                mask = mx.concatenate([ones, zeros], axis=1)

        return mask
