"""Text-to-video training strategy.

This strategy implements standard text-to-video generation training where:
- Only target latents are used (no reference videos)
- Standard noise application and loss computation
- Supports first-frame conditioning
- Optionally supports joint audio-video training
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import mlx.core as mx

from ltx_trainer_mlx.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModalityInputs,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

logger = logging.getLogger(__name__)


class TextToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for text-to-video training strategy.

    Attributes:
        name: Strategy identifier (always ``"text_to_video"``).
        first_frame_conditioning_p: Probability of conditioning on the first
            frame during training.
        with_audio: Whether to include audio in training (joint audio-video).
        audio_latents_dir: Directory name for audio latents when
            ``with_audio`` is ``True``.
    """

    name: Literal["text_to_video"]
    first_frame_conditioning_p: float
    with_audio: bool
    audio_latents_dir: str

    def __init__(
        self,
        *,
        first_frame_conditioning_p: float = 0.1,
        with_audio: bool = False,
        audio_latents_dir: str = "audio_latents",
    ) -> None:
        super().__init__(name="text_to_video")
        self.first_frame_conditioning_p = first_frame_conditioning_p
        self.with_audio = with_audio
        self.audio_latents_dir = audio_latents_dir


class TextToVideoStrategy(TrainingStrategy):
    """Text-to-video training strategy.

    Implements regular video generation training where:
    - Only target latents are used (no reference videos).
    - Standard noise application and loss computation.
    - Supports first-frame conditioning.
    - Optionally supports joint audio-video training when ``with_audio=True``.
    """

    config: TextToVideoConfig

    def __init__(self, config: TextToVideoConfig) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Text-to-video configuration.
        """
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Text-to-video requires latents and text conditions.

        When ``with_audio`` is ``True``, also requires audio latents.
        """
        sources: dict[str, str] = {
            "latents": "latents",
            "conditions": "conditions",
        }
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        sigma_sampler: Any,
    ) -> ModelInputs:
        """Prepare inputs for text-to-video training.

        Args:
            batch: Raw batch containing ``"latents"`` and ``"conditions"``
                (and optionally ``"audio_latents"``).
            sigma_sampler: Callable returning sigma values for noise schedule.

        Returns:
            Prepared ``ModelInputs`` for the transformer.
        """
        # Get pre-encoded latents [B, C, F, H, W]
        latents = batch["latents"]
        video_latents: mx.array = latents["latents"]

        # Video dimensions (assume uniform across batch)
        num_frames: int = int(latents["num_frames"][0].item())
        height: int = int(latents["height"][0].item())
        width: int = int(latents["width"][0].item())

        # Patchify: [B, C, F, H, W] -> [B, seq_len, C]
        video_latents, _ = self._video_patchifier.patchify(video_latents)

        # Handle FPS
        fps_arr = latents.get("fps", None)
        fps: float
        if fps_arr is not None:
            fps = float(fps_arr[0].item())
        else:
            fps = DEFAULT_FPS

        # Text embeddings (already processed by embedding connectors)
        conditions = batch["conditions"]
        video_prompt_embeds: mx.array = conditions["video_prompt_embeds"]
        audio_prompt_embeds: mx.array = conditions["audio_prompt_embeds"]
        prompt_attention_mask: mx.array = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]

        # First-frame conditioning mask
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Sample noise and sigmas
        sigmas = sigma_sampler.sample_for(video_latents)  # (B,) or (B, 1)
        video_noise = mx.random.normal(video_latents.shape)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.reshape(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For conditioning tokens, use clean latents
        cond_mask_expanded = video_conditioning_mask[:, :, None]  # (B, T, 1)
        noisy_video = mx.where(cond_mask_expanded, video_latents, noisy_video)

        # Velocity targets: noise - clean
        video_targets = video_noise - video_latents

        # Per-token timesteps
        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.reshape(-1))

        # Video positions
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )

        # Create video ModalityInputs
        video_modality = ModalityInputs(
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Video loss mask: True for non-conditioning tokens
        video_loss_mask = ~video_conditioning_mask

        # Handle audio if enabled
        audio_modality: ModalityInputs | None = None
        audio_targets: mx.array | None = None
        audio_loss_mask: mx.array | None = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: mx.array,
        audio_prompt_embeds: mx.array,
        prompt_attention_mask: mx.array,
        batch_size: int,
    ) -> tuple[ModalityInputs, mx.array, mx.array]:
        """Prepare audio inputs for joint audio-video training.

        Args:
            batch: Raw batch data containing ``"audio_latents"``.
            sigmas: Sampled sigma values (same schedule as video).
            audio_prompt_embeds: Audio context embeddings.
            prompt_attention_mask: Attention mask for context.
            batch_size: Batch size.

        Returns:
            Tuple of ``(audio_modality, audio_targets, audio_loss_mask)``.
        """
        # Audio latents [B, C, T, F]
        audio_data = batch["audio_latents"]
        audio_latents: mx.array = audio_data["latents"]

        # Patchify: [B, 8, T, 16] -> [B, T, 128]
        audio_latents, _ = self._audio_patchifier.patchify(audio_latents)
        audio_seq_len = audio_latents.shape[1]

        # Sample audio noise
        audio_noise = mx.random.normal(audio_latents.shape)

        # Apply noise (same sigma as video)
        sigmas_expanded = sigmas.reshape(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        # Velocity targets
        audio_targets = audio_noise - audio_latents

        # Audio timesteps: all tokens use the sampled sigma (no conditioning)
        audio_timesteps = mx.broadcast_to(sigmas.reshape(-1, 1), (batch_size, audio_seq_len))

        # Audio positions
        audio_positions = self._get_audio_positions(num_time_steps=audio_seq_len)

        # Create audio ModalityInputs
        audio_modality = ModalityInputs(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Audio loss mask: all tokens contribute (no conditioning)
        audio_loss_mask = mx.ones((batch_size, audio_seq_len), dtype=mx.bool_)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: mx.array,
        audio_pred: mx.array | None,
        inputs: ModelInputs,
    ) -> mx.array:
        """Compute masked MSE loss for video and optionally audio.

        Args:
            video_pred: Video prediction from the transformer.
            audio_pred: Audio prediction (``None`` for video-only).
            inputs: Prepared model inputs with targets and masks.

        Returns:
            Scalar loss value.
        """
        # Video loss: masked MSE
        video_loss = (video_pred - inputs.video_targets) ** 2
        video_loss_mask = inputs.video_loss_mask[:, :, None].astype(video_loss.dtype)
        # Normalize by mask density
        video_loss = video_loss * video_loss_mask / mx.mean(video_loss_mask)
        video_loss = mx.mean(video_loss)

        # If no audio, return video loss only
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        # Audio loss (no conditioning mask, plain MSE)
        audio_loss = mx.mean((audio_pred - inputs.audio_targets) ** 2)

        return video_loss + audio_loss
