"""Video-to-video training strategy for IC-LoRA.

This strategy implements training with reference video conditioning where:
- Reference latents (clean) are concatenated with target latents (noised).
- Video coordinates handle both reference and target sequences.
- Loss is computed only on the target portion.
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


class VideoToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for video-to-video (IC-LoRA) training strategy.

    Attributes:
        name: Strategy identifier (always ``"video_to_video"``).
        first_frame_conditioning_p: Probability of conditioning on the first
            frame during training.
        reference_latents_dir: Directory name for reference video latents.
    """

    name: Literal["video_to_video"]
    first_frame_conditioning_p: float
    reference_latents_dir: str

    def __init__(
        self,
        *,
        first_frame_conditioning_p: float = 0.1,
        reference_latents_dir: str = "reference_latents",
    ) -> None:
        super().__init__(name="video_to_video")
        self.first_frame_conditioning_p = first_frame_conditioning_p
        self.reference_latents_dir = reference_latents_dir


class VideoToVideoStrategy(TrainingStrategy):
    """Video-to-video training strategy for IC-LoRA.

    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised).
    - Video coordinates handle both reference and target sequences.
    - Loss is computed only on the target portion.

    Attributes:
        reference_downscale_factor: The inferred downscale factor of reference
            videos.  Computed from the first batch and cached for metadata
            export.
    """

    config: VideoToVideoConfig
    reference_downscale_factor: int | None

    def __init__(self, config: VideoToVideoConfig) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Video-to-video configuration.
        """
        super().__init__(config)
        self.reference_downscale_factor = None  # Inferred from first batch

    def get_data_sources(self) -> dict[str, str]:
        """IC-LoRA training requires latents, conditions, and reference latents."""
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.config.reference_latents_dir: "ref_latents",
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        sigma_sampler: Any,
    ) -> ModelInputs:
        """Prepare inputs for IC-LoRA training with reference videos.

        Args:
            batch: Raw batch containing ``"latents"``, ``"conditions"``,
                and ``"ref_latents"``.
            sigma_sampler: Callable returning sigma values for noise schedule.

        Returns:
            Prepared ``ModelInputs`` for the transformer.
        """
        # Target latents [B, C, F, H, W]
        latents = batch["latents"]
        target_latents: mx.array = latents["latents"]
        ref_latents_data = batch["ref_latents"]
        ref_latents: mx.array = ref_latents_data["latents"]

        # Dimensions
        num_frames: int = int(latents["num_frames"][0].item())
        height: int = int(latents["height"][0].item())
        width: int = int(latents["width"][0].item())

        ref_frames: int = int(ref_latents_data["num_frames"][0].item())
        ref_height: int = int(ref_latents_data["height"][0].item())
        ref_width: int = int(ref_latents_data["width"][0].item())

        # Infer reference downscale factor
        reference_downscale_factor = _infer_reference_downscale_factor(
            target_height=height,
            target_width=width,
            ref_height=ref_height,
            ref_width=ref_width,
        )

        # Cache / validate scale factor
        if self.reference_downscale_factor is None:
            self.reference_downscale_factor = reference_downscale_factor
        elif self.reference_downscale_factor != reference_downscale_factor:
            raise ValueError(
                f"Inconsistent reference downscale factor across batches. "
                f"First batch had factor={self.reference_downscale_factor}, "
                f"but current batch has factor={reference_downscale_factor}. "
                f"All training samples must use the same reference/target "
                f"resolution ratio."
            )

        # Patchify: [B, C, F, H, W] -> [B, seq_len, C]
        target_latents, _ = self._video_patchifier.patchify(target_latents)
        ref_latents, _ = self._video_patchifier.patchify(ref_latents)

        # Handle FPS
        fps_arr = latents.get("fps", None)
        fps: float
        if fps_arr is not None:
            fps = float(fps_arr[0].item())
        else:
            fps = DEFAULT_FPS

        # Text embeddings (video-to-video uses only video embeddings)
        conditions = batch["conditions"]
        prompt_embeds: mx.array = conditions["video_prompt_embeds"]
        prompt_attention_mask: mx.array = conditions["prompt_attention_mask"]

        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]

        # Conditioning masks ---
        # Reference tokens: always conditioning (timestep = 0)
        ref_conditioning_mask = mx.ones((batch_size, ref_seq_len), dtype=mx.bool_)

        # Target tokens: check for first-frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=height,
            width=width,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Combined mask
        conditioning_mask = mx.concatenate([ref_conditioning_mask, target_conditioning_mask], axis=1)

        # Sample noise and sigmas for target
        sigmas = sigma_sampler(target_latents)  # (B,) or (B, 1)
        noise = mx.random.normal(target_latents.shape)
        sigmas_expanded = sigmas.reshape(-1, 1, 1)

        # Apply noise to target
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise

        # For first-frame conditioning in target, use clean latents
        target_cond_expanded = target_conditioning_mask[:, :, None]
        noisy_target = mx.where(target_cond_expanded, target_latents, noisy_target)

        # Velocity targets (only for target portion)
        targets = noise - target_latents

        # Concatenate reference (clean) + target (noisy)
        combined_latents = mx.concatenate([ref_latents, noisy_target], axis=1)

        # Per-token timesteps
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.reshape(-1))

        # Positions for reference and target ---
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            fps=fps,
        )

        # Scale reference positions to match target coordinate space
        if reference_downscale_factor != 1:
            # positions shape: (1, ref_seq_len, 3) where dim -1 is [time, height, width]
            # Scale height (index 1) and width (index 2), keep time (index 0) unchanged
            scale = mx.array([1.0, float(reference_downscale_factor), float(reference_downscale_factor)])
            ref_positions = ref_positions * scale[None, None, :]

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )

        # Concatenate along sequence dimension (axis=1)
        positions = mx.concatenate([ref_positions, target_positions], axis=1)

        # Create video ModalityInputs
        video_modality = ModalityInputs(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Loss mask: only non-conditioning target tokens
        ref_loss_mask = mx.zeros((batch_size, ref_seq_len), dtype=mx.bool_)
        target_loss_mask = ~target_conditioning_mask
        video_loss_mask = mx.concatenate([ref_loss_mask, target_loss_mask], axis=1)

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
        )

    def compute_loss(
        self,
        video_pred: mx.array,
        _audio_pred: mx.array | None,
        inputs: ModelInputs,
    ) -> mx.array:
        """Compute masked loss only on the target portion.

        Args:
            video_pred: Full prediction including reference + target tokens.
            _audio_pred: Unused (video-to-video is video-only).
            inputs: Prepared model inputs with targets and masks.

        Returns:
            Scalar loss value.
        """
        # Extract target portion of prediction
        ref_seq_len = inputs.ref_seq_len
        assert ref_seq_len is not None
        target_pred = video_pred[:, ref_seq_len:, :]

        # Target portion of loss mask
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:]

        # MSE loss
        loss = (target_pred - inputs.video_targets) ** 2

        # Apply and normalize by mask density
        loss_mask = target_loss_mask[:, :, None].astype(loss.dtype)
        loss = loss * loss_mask / mx.mean(loss_mask)

        return mx.mean(loss)

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get metadata for checkpoint files.

        Includes ``reference_downscale_factor`` so inference pipelines know
        the expected scale factor for reference videos.
        """
        metadata: dict[str, Any] = {}
        if self.reference_downscale_factor is not None:
            metadata["reference_downscale_factor"] = self.reference_downscale_factor
        return metadata


def _infer_reference_downscale_factor(
    target_height: int,
    target_width: int,
    ref_height: int,
    ref_width: int,
) -> int:
    """Infer the reference downscale factor from target and reference dims.

    Args:
        target_height: Target latent height.
        target_width: Target latent width.
        ref_height: Reference latent height.
        ref_width: Reference latent width.

    Returns:
        Integer downscale factor (uniform across both spatial dimensions).

    Raises:
        ValueError: If dimensions are not exact multiples or scaling is
            non-uniform.
    """
    if target_height == ref_height and target_width == ref_width:
        return 1

    if target_height % ref_height != 0 or target_width % ref_width != 0:
        raise ValueError(
            f"Target dimensions ({target_height}x{target_width}) must be exact "
            f"multiples of reference dimensions ({ref_height}x{ref_width})"
        )

    scale_h = target_height // ref_height
    scale_w = target_width // ref_width

    if scale_h != scale_w:
        raise ValueError(
            f"Reference scale must be uniform. Got height scale {scale_h} and "
            f"width scale {scale_w}. Target: {target_height}x{target_width}, "
            f"Reference: {ref_height}x{ref_width}"
        )

    if scale_h < 1:
        raise ValueError(
            f"Reference dimensions ({ref_height}x{ref_width}) cannot be larger "
            f"than target dimensions ({target_height}x{target_width})"
        )

    return scale_h
