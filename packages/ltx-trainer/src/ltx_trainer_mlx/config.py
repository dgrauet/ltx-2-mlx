"""Configuration schema for LTX-2 MLX trainer.

Ported from ltx-trainer (Lightricks). Removes torch/CUDA/DDP/FSDP dependencies;
keeps Pydantic structure intact for config validation.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model and training mode."""

    model_path: str | Path = Field(
        ...,
        description="Model path - local path to safetensors checkpoint file",
    )

    text_encoder_path: str | Path | None = Field(
        default=None,
        description="Path to text encoder (required for LTX-2/Gemma models)",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str | Path) -> str | Path:
        """Validate that model_path is an existing local path (not a URL)."""
        is_url = str(v).startswith(("http://", "https://"))

        if is_url:
            raise ValueError(f"Model path cannot be a URL: {v}")

        if not Path(v).exists():
            raise ValueError(f"Model path does not exist: {v}")

        return v


class LoraConfig(ConfigBaseModel):
    """Configuration for LoRA fine-tuning."""

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=["to_k", "to_q", "to_v", "to_out.0"],
        description="List of modules to target with LoRA",
    )


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters."""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    weight_decay: float = Field(
        default=0.0,
        description="Weight decay (L2 regularization) coefficient",
    )

    optimizer_type: Literal["adamw"] = Field(
        default="adamw",
        description="Type of optimizer to use for training (MLX supports AdamW)",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
    ] = Field(
        default="linear",
        description="Type of learning rate scheduler",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing."""

    preprocessed_data_root: str = Field(
        description="Path to folder containing preprocessed training data",
    )

    num_dataloader_workers: int = Field(
        default=0,
        description="Number of background processes for data loading (0 for synchronous, default for MLX)",
        ge=0,
    )


class ValidationConfig(ConfigBaseModel):
    """Configuration for validation during training."""

    prompts: list[str] = Field(
        default_factory=list,
        description="List of prompts to use for validation",
    )

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    reference_videos: list[str] | None = Field(
        default=None,
        description="List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

    reference_downscale_factor: int = Field(
        default=1,
        description="Downscale factor for reference videos in IC-LoRA validation. "
        "When > 1, reference videos are processed at 1/n resolution.",
        ge=1,
    )

    video_dims: tuple[int, int, int] = Field(
        default=(960, 544, 97),
        description="Dimensions of validation videos (width, height, frames). "
        "Width and height must be divisible by 32. Frames must satisfy frames % 8 == 1.",
    )

    @field_validator("video_dims")
    @classmethod
    def validate_video_dims(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate video dimensions for LTX-2 compatibility."""
        width, height, frames = v

        if width % 32 != 0:
            raise ValueError(f"Width ({width}) must be divisible by 32")
        if height % 32 != 0:
            raise ValueError(f"Height ({height}) must be divisible by 32")
        if frames % 8 != 1:
            raise ValueError(f"Frames ({frames}) must satisfy frames % 8 == 1 for LTX-2 (e.g., 1, 9, 17, 25, ...)")

        return v

    frame_rate: float = Field(
        default=25.0,
        description="Frame rate for validation videos",
        gt=0,
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=50,
        description="Number of inference steps for validation",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    videos_per_prompt: int = Field(
        default=1,
        description="Number of videos to generate per validation prompt",
        gt=0,
    )

    guidance_scale: float = Field(
        default=4.0,
        description="CFG guidance scale to use during validation",
        ge=1.0,
    )

    stg_scale: float = Field(
        default=1.0,
        description="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG.",
        ge=0.0,
    )

    stg_blocks: list[int] | None = Field(
        default=[29],
        description="Which transformer blocks to perturb for STG.",
    )

    stg_mode: Literal["stg_av", "stg_v"] = Field(
        default="stg_av",
        description="STG mode: 'stg_av' skips both audio and video self-attention, "
        "'stg_v' skips only video self-attention.",
    )

    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio in validation samples.",
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    include_reference_in_output: bool = Field(
        default=False,
        description="For video-to-video training: concatenate the original reference video side-by-side "
        "with the generated output.",
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of images (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if len(v) != num_prompts:
            raise ValueError(f"Number of images ({len(v)}) must match number of prompts ({num_prompts})")

        for image_path in v:
            if not Path(image_path).exists():
                raise ValueError(f"Image path '{image_path}' does not exist")

        return v

    @field_validator("reference_videos")
    @classmethod
    def validate_reference_videos(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of reference videos (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if len(v) != num_prompts:
            raise ValueError(f"Number of reference videos ({len(v)}) must match number of prompts ({num_prompts})")

        for video_path in v:
            if not Path(video_path).exists():
                raise ValueError(f"Reference video path '{video_path}' does not exist")

        return v

    @model_validator(mode="after")
    def validate_scaled_reference_dimensions(self) -> "ValidationConfig":
        """Validate that scaled reference dimensions are valid when reference_downscale_factor > 1."""
        if self.reference_downscale_factor > 1:
            width, height, _frames = self.video_dims

            if width % self.reference_downscale_factor != 0:
                raise ValueError(
                    f"Width {width} is not evenly divisible by reference_downscale_factor "
                    f"{self.reference_downscale_factor}."
                )
            if height % self.reference_downscale_factor != 0:
                raise ValueError(
                    f"Height {height} is not evenly divisible by reference_downscale_factor "
                    f"{self.reference_downscale_factor}."
                )

            scaled_width = width // self.reference_downscale_factor
            scaled_height = height // self.reference_downscale_factor

            if scaled_width % 32 != 0:
                raise ValueError(
                    f"Scaled reference width {scaled_width} (from {width} / {self.reference_downscale_factor}) "
                    f"is not divisible by 32."
                )
            if scaled_height % 32 != 0:
                raise ValueError(
                    f"Scaled reference height {scaled_height} (from {height} / {self.reference_downscale_factor}) "
                    f"is not divisible by 32."
                )

        return self


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training."""

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )

    precision: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Precision to use when saving checkpoint weights.",
    )


class HubConfig(ConfigBaseModel):
    """Configuration for Hugging Face Hub integration."""

    push_to_hub: bool = Field(default=False, description="Whether to push the model weights to the Hugging Face Hub")
    hub_model_id: str | None = Field(
        default=None, description="Hugging Face Hub repository ID (e.g., 'username/repo-name')"
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        """Validate that hub_model_id is not None when push_to_hub is True."""
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="ltx-trainer-mlx",
        description="W&B project name",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training."""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )


class TrainingStrategyConfig(ConfigBaseModel):
    """Configuration for training strategy.

    Simplified from the reference discriminated union; extend as needed
    when video-to-video and other strategies are ported.
    """

    name: Literal["text_to_video", "video_to_video"] = Field(
        default="text_to_video",
        description="Training strategy name.",
    )

    generate_audio: bool = Field(
        default=True,
        description="Whether to train the audio branch alongside video.",
    )


class LtxTrainerConfig(ConfigBaseModel):
    """Unified configuration for LTX-2 MLX training."""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    training_strategy: TrainingStrategyConfig = Field(
        default_factory=TrainingStrategyConfig,
        description="Training strategy configuration.",
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # General configuration
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def validate_strategy_compatibility(self) -> "LtxTrainerConfig":
        """Validate that training strategy and other configurations are compatible."""
        # Check that reference videos are provided when using video_to_video strategy
        if (
            self.training_strategy.name == "video_to_video"
            and self.validation.interval
            and not self.validation.reference_videos
        ):
            raise ValueError(
                "reference_videos must be provided in validation config when using video_to_video strategy"
            )

        # Check that LoRA config is provided when training mode is lora
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")

        # Check that LoRA config is provided when using video_to_video strategy
        if self.training_strategy.name == "video_to_video" and self.model.training_mode != "lora":
            raise ValueError("Training mode must be 'lora' when using video_to_video strategy")

        return self
