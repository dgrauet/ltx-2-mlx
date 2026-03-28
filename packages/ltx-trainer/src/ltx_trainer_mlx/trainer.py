"""LTX-2 trainer for MLX on Apple Silicon.

Ported from ltx-trainer (Lightricks). Replaces Accelerate/DDP/FSDP with
direct MLX training:
- ``mlx.optimizers.AdamW`` for optimisation
- ``mlx.nn.value_and_grad()`` for gradient computation
- Single-device (unified CPU/GPU memory on Apple Silicon)
- ``safetensors`` for checkpointing

The overall training loop structure mirrors the reference implementation.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from pydantic import BaseModel
from safetensors.numpy import save_file as save_safetensors

from ltx_trainer_mlx.config import LtxTrainerConfig
from ltx_trainer_mlx.config_display import print_config
from ltx_trainer_mlx.gpu_utils import free_gpu_memory, free_gpu_memory_context, get_gpu_memory_gb
from ltx_trainer_mlx.model_loader import (
    load_feature_extractor,
    load_text_encoder,
)
from ltx_trainer_mlx.model_loader import (
    load_model as load_ltx_model,
)
from ltx_trainer_mlx.progress import TrainingProgress
from ltx_trainer_mlx.utils import save_image
from ltx_trainer_mlx.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer_mlx.video_utils import save_video

logger = logging.getLogger(__name__)

StepCallback = Callable[[int, int, list[Path]], None]

MEMORY_CHECK_INTERVAL = 200


def _materialize(x: Any) -> None:
    """Force MLX lazy compute graph to materialise.

    ``mx.eval`` is MLX's graph-evaluation primitive (analogous to
    synchronising a CUDA stream) -- it is **not** Python's ``eval``.
    """
    # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
    mx.eval(x)


class TrainingStats(BaseModel):
    """Statistics collected during training."""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_memory_gb: float
    batch_size: int


class LtxvTrainer:
    """Main trainer for LTX-2 fine-tuning on Apple Silicon via MLX.

    Orchestrates model loading, LoRA setup, training loop, validation
    sampling, checkpointing, and W&B logging -- all on a single device
    using MLX's unified memory model.
    """

    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        print_config(trainer_config)

        self._cached_validation_embeddings = self._load_text_encoder_and_cache_embeddings()
        self._load_models()
        self._collect_trainable_params()
        self._load_checkpoint()

        self._init_timestep_sampler()
        self._global_step = -1
        self._checkpoint_paths: list[Path] = []
        self._init_wandb()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """Start the training process.

        Returns:
            Tuple of (saved_model_path, training_stats).
        """
        cfg = self._config
        start_mem = get_gpu_memory_gb()

        train_start_time = time.time()

        # Set random seed
        mx.random.seed(cfg.seed)
        logger.debug("Using seed: %d", cfg.seed)

        self._init_optimizer()
        self._init_dataloader()
        data_iter = iter(self._dataloader)

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        logger.info("Starting training...")

        progress_enabled = not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=cfg.optimization.steps,
        )

        if disable_progress_bars:
            logger.warning("Progress bars disabled. Intermediate status messages will be logged instead.")

        self._global_step = 0
        peak_mem = start_mem
        sampled_videos_paths: list[Path] | None = None

        # Build the loss function for value_and_grad
        loss_fn = self._build_loss_fn()
        loss_and_grad_fn = nn.value_and_grad(self._transformer, loss_fn)

        with progress:
            # Initial validation
            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                sampled_videos_paths = self._sample_videos(progress)
                if sampled_videos_paths and self._config.wandb.log_validation_videos:
                    self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

            accum_steps = cfg.optimization.gradient_accumulation_steps
            accumulated_grads: Any = None

            for step in range(cfg.optimization.steps * accum_steps):
                # Get next batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start_time = time.time()

                is_optimization_step = (step + 1) % accum_steps == 0
                if is_optimization_step:
                    self._global_step += 1

                # Forward + backward
                loss, grads = loss_and_grad_fn(batch)
                _materialize(loss)

                # Accumulate gradients
                if accum_steps > 1:
                    if accumulated_grads is None:
                        accumulated_grads = grads
                    else:
                        accumulated_grads = nn.utils.tree_map(lambda a, b: a + b, accumulated_grads, grads)

                if is_optimization_step:
                    # Average accumulated gradients
                    if accum_steps > 1 and accumulated_grads is not None:
                        grads = nn.utils.tree_map(lambda g: g / accum_steps, accumulated_grads)
                        accumulated_grads = None

                    # Gradient clipping
                    if cfg.optimization.max_grad_norm > 0:
                        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=cfg.optimization.max_grad_norm)
                        _materialize(grad_norm)

                    # Optimizer step
                    self._optimizer.update(self._transformer, grads)
                    _materialize((self._optimizer.state, self._transformer.parameters()))

                # Learning rate scheduling
                if self._lr_schedule is not None and is_optimization_step:
                    lr = self._lr_schedule(self._global_step)
                    self._optimizer.learning_rate = lr

                # Validation
                if (
                    cfg.validation.interval
                    and self._global_step > 0
                    and self._global_step % cfg.validation.interval == 0
                    and is_optimization_step
                ):
                    sampled_videos_paths = self._sample_videos(progress)
                    if sampled_videos_paths and self._config.wandb.log_validation_videos:
                        self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

                # Save checkpoint
                if (
                    cfg.checkpoints.interval
                    and self._global_step > 0
                    and self._global_step % cfg.checkpoints.interval == 0
                    and is_optimization_step
                ):
                    self._save_checkpoint()

                # Step callback
                if step_callback and is_optimization_step:
                    step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths or [])

                # Update progress and log metrics
                current_lr = self._optimizer.learning_rate
                if isinstance(current_lr, mx.array):
                    current_lr = float(current_lr.item())
                step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps

                loss_val = float(loss.item())
                progress.update_training(
                    loss=loss_val,
                    lr=current_lr,
                    step_time=step_time,
                    advance=is_optimization_step,
                )

                # Log to W&B
                if is_optimization_step:
                    self._log_metrics(
                        {
                            "train/loss": loss_val,
                            "train/learning_rate": current_lr,
                            "train/step_time": step_time,
                            "train/global_step": self._global_step,
                        }
                    )

                # Fallback logging
                if disable_progress_bars and self._global_step % 20 == 0:
                    elapsed = time.time() - train_start_time
                    pct = self._global_step / cfg.optimization.steps
                    if pct > 0:
                        total_est = elapsed / pct
                        total_time = f"{total_est // 3600:.0f}h {(total_est % 3600) // 60:.0f}m"
                    else:
                        total_time = "calculating..."
                    logger.info(
                        "Step %d/%d - Loss: %.4f, LR: %.2e, Time/Step: %.2fs, Total: %s",
                        self._global_step,
                        cfg.optimization.steps,
                        loss_val,
                        current_lr,
                        step_time,
                        total_time,
                    )

                # Memory check
                if step % MEMORY_CHECK_INTERVAL == 0:
                    current_mem = get_gpu_memory_gb()
                    peak_mem = max(peak_mem, current_mem)

        # Final stats
        total_time_seconds = time.time() - train_start_time
        steps_per_second = cfg.optimization.steps / total_time_seconds
        samples_per_second = steps_per_second * cfg.optimization.batch_size

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_memory_gb=peak_mem,
            batch_size=cfg.optimization.batch_size,
        )

        saved_path = self._save_checkpoint()

        self._log_training_stats(stats)

        if cfg.hub.push_to_hub:
            from ltx_trainer_mlx.hf_hub_utils import push_to_hub

            push_to_hub(saved_path, sampled_videos_paths or [], self._config)

        if self._wandb_run is not None:
            self._log_metrics(
                {
                    "stats/total_time_minutes": stats.total_time_seconds / 60,
                    "stats/steps_per_second": stats.steps_per_second,
                    "stats/samples_per_second": stats.samples_per_second,
                    "stats/peak_memory_gb": stats.peak_memory_gb,
                }
            )
            self._wandb_run.finish()

        return saved_path, stats

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------

    def _build_loss_fn(self) -> Callable:
        """Build the loss function for value_and_grad.

        Returns a closure that takes a batch and returns a scalar loss.
        The closure captures ``self`` for access to the training strategy
        and feature extractor.
        """
        strategy = self._training_strategy
        feature_extractor = self._feature_extractor

        def loss_fn(batch: dict[str, Any]) -> mx.array:
            conditions = batch["conditions"]

            # Get text embeddings
            video_features = conditions.get("video_prompt_embeds", conditions.get("prompt_embeds"))
            audio_features = conditions.get("audio_prompt_embeds", video_features)
            mask = conditions["prompt_attention_mask"]

            # Apply feature extractor to get final embeddings
            if feature_extractor is not None:
                video_embeds, audio_embeds = feature_extractor(
                    video_features,
                    audio_features=audio_features,
                    attention_mask=mask,
                )
            else:
                video_embeds = video_features
                audio_embeds = audio_features

            conditions["video_prompt_embeds"] = video_embeds
            conditions["audio_prompt_embeds"] = audio_embeds

            # Use strategy to prepare inputs
            model_inputs = strategy.prepare_training_inputs(batch, self._timestep_sampler)

            # Forward pass — call the inner LTXModel (returns velocity,
            # not x0) since training targets are velocity (noise - clean).
            video_inputs = model_inputs.video
            audio_inputs = model_inputs.audio

            call_kwargs: dict = dict(
                video_latent=video_inputs.latent,
                video_text_embeds=video_inputs.context,
                video_positions=video_inputs.positions,
                timestep=video_inputs.sigma,
                video_timesteps=video_inputs.timesteps,
            )

            if audio_inputs is not None:
                call_kwargs["audio_latent"] = audio_inputs.latent
                call_kwargs["audio_text_embeds"] = audio_inputs.context
                call_kwargs["audio_positions"] = audio_inputs.positions
                call_kwargs["audio_timesteps"] = audio_inputs.timesteps
            else:
                # Dummy audio for the joint model — must match expected token count
                # Audio tokens = round(pixel_frames / fps * 25)
                from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count

                latent_data = batch["latents"]
                num_lat_frames = int(latent_data["num_frames"][0].item())
                pixel_frames = (num_lat_frames - 1) * 8 + 1
                fps_val = float(latent_data.get("fps", mx.array([24.0]))[0].item())
                audio_tokens = compute_audio_token_count(pixel_frames, fps=fps_val)

                B = video_inputs.latent.shape[0]
                call_kwargs["audio_latent"] = mx.zeros((B, audio_tokens, 128), dtype=mx.bfloat16)
                call_kwargs["audio_text_embeds"] = conditions["audio_prompt_embeds"]
                call_kwargs["audio_positions"] = compute_audio_positions(audio_tokens)
                call_kwargs["audio_timesteps"] = mx.zeros((B, audio_tokens))

            # Call LTXModel directly which returns velocity (v),
            # NOT X0Model which converts to x0 predictions
            video_pred, audio_pred = self._transformer(**call_kwargs)

            # Compute loss
            return strategy.compute_loss(video_pred, audio_pred, model_inputs)

        return loss_fn

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        """Load text encoder + feature extractor, compute and cache validation embeddings.

        When using precomputed data (no validation prompts), the feature
        extractor is NOT loaded — precomputed conditions already contain
        projected embeddings.
        """
        cfg = self._config
        has_validation = bool(cfg.validation.interval and cfg.validation.prompts)

        if not has_validation:
            # No validation prompts — skip text encoder and feature extractor entirely.
            # Precomputed conditions already contain projected video/audio embeddings.
            self._feature_extractor = None
            logger.debug("No validation prompts. Skipping text encoder and feature extractor.")
            return None

        # Load text encoder
        logger.debug("Loading text encoder...")
        text_encoder = load_text_encoder(
            gemma_model_path=cfg.model.text_encoder_path,
        )

        # Load feature extractor (connector)
        logger.debug("Loading feature extractor...")
        self._feature_extractor = load_feature_extractor(
            model_dir=cfg.model.model_path,
        )

        # Cache validation embeddings
        logger.info("Pre-computing embeddings for %d validation prompts...", len(cfg.validation.prompts))
        cached_embeddings = []
        for prompt in cfg.validation.prompts:
            all_hs, attn_mask = text_encoder.encode_all_layers(prompt)
            video_embeds, audio_embeds = self._feature_extractor(all_hs, attention_mask=attn_mask)

            cached_embeddings.append(
                CachedPromptEmbeddings(
                    video_context=video_embeds,
                    audio_context=audio_embeds,
                )
            )

        # Unload text encoder (feature extractor stays for validation)
        del text_encoder
        free_gpu_memory()

        logger.debug("Validation prompt embeddings cached. Text encoder unloaded.")
        return cached_embeddings

    def _load_models(self) -> None:
        """Load the LTX-2 model components."""
        from ltx_trainer_mlx.training_strategies import get_training_strategy

        self._training_strategy = get_training_strategy(self._config.training_strategy)

        load_audio = self._training_strategy.requires_audio or self._config.validation.generate_audio
        need_vae_encoder = (
            self._config.validation.images is not None or self._config.validation.reference_videos is not None
        )

        has_validation = bool(self._config.validation.interval and self._config.validation.prompts)

        components = load_ltx_model(
            model_dir=self._config.model.model_path,
            with_video_vae_encoder=need_vae_encoder,
            with_video_vae_decoder=has_validation,
            with_audio_vae_decoder=load_audio and has_validation,
            with_vocoder=load_audio and has_validation,
            with_text_encoder=False,  # Handled separately above
        )

        self._transformer = components.transformer
        self._vae_decoder = components.video_vae_decoder
        self._vae_encoder = components.video_vae_encoder
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder

        # Freeze all models -- LoRA or full unfreezing handled in _collect_trainable_params
        self._transformer.freeze()
        if self._vae_decoder is not None:
            self._vae_decoder.freeze()
        if self._vae_encoder is not None:
            self._vae_encoder.freeze()
        if self._audio_vae is not None:
            self._audio_vae.freeze()
        if self._vocoder is not None:
            self._vocoder.freeze()

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            self._transformer.unfreeze()
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        # Count trainable params
        num_trainable = sum(p.size for _, p in nn.utils.tree_flatten(self._transformer.trainable_parameters()))
        logger.debug("Trainable params count: %s", f"{num_trainable:,}")

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer.

        Supports both ``nn.Linear`` and ``nn.QuantizedLinear`` targets.
        For quantized models (q4/q8), the LoRA adapter wraps the existing
        quantized layer, keeping weights in low-bit while LoRA params
        train in full precision.
        """
        from mlx_lm.tuner.lora import LoRALinear

        lora_cfg = self._config.lora
        if lora_cfg is None:
            raise ValueError("LoRA config is required for LoRA training mode")

        logger.debug("Adding LoRA adapter with rank %d", lora_cfg.rank)

        # Apply LoRA to matching linear layers (Linear or QuantizedLinear)
        lora_layers = _find_lora_targets(self._transformer, lora_cfg.target_modules)

        for layer_path, module in lora_layers:
            # Create LoRA wrapper with matching dimensions
            if isinstance(module, nn.QuantizedLinear):
                in_dims = module.scales.shape[-1] * module.group_size
                out_dims = module.weight.shape[0]
                has_bias = hasattr(module, "bias") and module.bias is not None
            else:
                in_dims = module.weight.shape[-1]
                out_dims = module.weight.shape[0]
                has_bias = hasattr(module, "bias") and module.bias is not None

            lora_linear = LoRALinear(
                input_dims=in_dims,
                output_dims=out_dims,
                r=lora_cfg.rank,
                scale=lora_cfg.alpha / lora_cfg.rank,
                bias=has_bias,
            )
            # Replace the inner nn.Linear with the original (possibly quantized) layer
            lora_linear.linear = module

            _set_module_by_path(self._transformer, layer_path, lora_linear)

        # Freeze everything, then unfreeze only LoRA params
        self._transformer.freeze()
        self._transformer.unfreeze(
            keys=["lora_a", "lora_b"],
            strict=False,
        )

        logger.debug("LoRA applied to %d layers", len(lora_layers))

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config."""
        if not self._config.model.load_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning("Could not find checkpoint at %s", self._config.model.load_checkpoint)
            return

        logger.info("Loading checkpoint from %s", checkpoint_path)

        weights = mx.load(str(checkpoint_path))

        if self._config.model.training_mode == "lora":
            # Filter to LoRA weights only and strip prefix
            lora_weights = {}
            for k, v in weights.items():
                k = k.replace("diffusion_model.", "", 1)
                if "lora_a" in k or "lora_b" in k:
                    lora_weights[k] = v
            if lora_weights:
                self._transformer.load_weights(list(lora_weights.items()), strict=False)
                logger.info("LoRA checkpoint loaded successfully")
        else:
            self._transformer.load_weights(list(weights.items()), strict=True)
            logger.info("Full model checkpoint loaded successfully")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            return checkpoint_path

        if checkpoint_path.is_dir():
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))
            if not checkpoints:
                checkpoints = list(checkpoint_path.rglob("*step_*.npz"))
            if not checkpoints:
                return None

            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            return max(checkpoints, key=_get_step_num)

        return None

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate schedule."""
        opt_cfg = self._config.optimization

        self._optimizer = optim.AdamW(
            learning_rate=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
        )

        self._lr_schedule = self._create_schedule()

    def _create_schedule(self) -> Callable[[int], float] | None:
        """Create a learning rate schedule function.

        Returns:
            A function mapping step -> learning_rate, or ``None`` for constant.
        """
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        lr = self._config.optimization.learning_rate
        params = dict(self._config.optimization.scheduler_params)

        if scheduler_type == "constant" or scheduler_type is None:
            return None

        if scheduler_type == "linear":
            start_factor = params.get("start_factor", 1.0)
            end_factor = params.get("end_factor", 0.1)

            def linear_schedule(step: int) -> float:
                t = min(step / max(steps, 1), 1.0)
                factor = start_factor + (end_factor - start_factor) * t
                return lr * factor

            return linear_schedule

        if scheduler_type == "cosine":
            eta_min = params.get("eta_min", 0.0)

            def cosine_schedule(step: int) -> float:
                t = min(step / max(steps, 1), 1.0)
                return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * t)) / 2

            return cosine_schedule

        if scheduler_type == "cosine_with_restarts":
            t_0 = params.get("T_0", steps // 4)
            t_mult = params.get("T_mult", 1)
            eta_min = params.get("eta_min", 5e-5)

            def cosine_restarts_schedule(step: int) -> float:
                cycle_len = t_0
                remaining = step
                while remaining >= cycle_len:
                    remaining -= cycle_len
                    cycle_len = int(cycle_len * t_mult)
                t = remaining / max(cycle_len, 1)
                return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * t)) / 2

            return cosine_restarts_schedule

        if scheduler_type == "polynomial":
            power = params.get("power", 1.0)

            def polynomial_schedule(step: int) -> float:
                t = min(step / max(steps, 1), 1.0)
                return lr * (1 - t) ** power

            return polynomial_schedule

        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader."""
        data_sources = self._training_strategy.get_data_sources()

        from ltx_trainer_mlx.datasets import PrecomputedDataset

        self._dataset = PrecomputedDataset(
            self._config.data.preprocessed_data_root,
            data_sources=data_sources,
        )
        logger.debug("Loaded dataset with %d samples", len(self._dataset))

        self._dataloader = _simple_dataloader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
        )

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on config."""
        from ltx_trainer_mlx.timestep_samplers import SAMPLERS

        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @free_gpu_memory_context(after=True)
    def _sample_videos(self, progress: TrainingProgress) -> list[Path] | None:
        """Run validation by generating videos from validation prompts."""
        cfg = self._config
        generate_audio = cfg.validation.generate_audio
        inference_steps = cfg.validation.inference_steps

        free_gpu_memory()

        sampling_ctx = progress.start_sampling(
            num_prompts=len(cfg.validation.prompts),
            num_steps=inference_steps,
        )

        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
        )

        output_dir = Path(cfg.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths: list[Path] = []
        width, height, num_frames = cfg.validation.video_dims

        for prompt_idx, prompt in enumerate(cfg.validation.prompts):
            sampling_ctx.start_video(prompt_idx)

            condition_image = None
            if cfg.validation.images is not None:
                condition_image = cfg.validation.images[prompt_idx]

            cached_embeddings = (
                self._cached_validation_embeddings[prompt_idx]
                if self._cached_validation_embeddings is not None
                else None
            )

            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=cfg.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=cfg.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=cfg.validation.guidance_scale,
                seed=cfg.validation.seed,
                condition_image=condition_image,
                generate_audio=generate_audio,
                cached_embeddings=cached_embeddings,
                stg_scale=cfg.validation.stg_scale,
                stg_blocks=cfg.validation.stg_blocks,
                stg_mode=cfg.validation.stg_mode,
            )

            video, audio = sampler.generate(config=gen_config)

            ext = "png" if num_frames == 1 else "mp4"
            output_path = output_dir / f"step_{self._global_step:06d}_{prompt_idx + 1}.{ext}"
            if num_frames == 1:
                save_image(video, output_path)
            else:
                save_video(
                    video_array=video,
                    output_path=output_path,
                    fps=cfg.validation.frame_rate,
                    audio=audio,
                    audio_sample_rate=48000 if audio is not None else None,
                )
            video_paths.append(output_path)

        sampling_ctx.cleanup()
        logger.info("Validation samples for step %d saved in %s", self._global_step, output_dir)
        return video_paths

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> Path:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"

        save_dir = Path(self._config.output_dir) / "checkpoints"
        save_dir.mkdir(exist_ok=True, parents=True)
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_path = save_dir / filename

        if is_lora:
            state_dict: dict[str, np.ndarray] = {}
            for name, param in nn.utils.tree_flatten(self._transformer.trainable_parameters()):
                key = f"diffusion_model.{name}"
                # Convert mlx_lm LoRA format to standard ComfyUI/diffusers format:
                # mlx_lm: .lora_a (in_dim, rank), .lora_b (rank, out_dim)
                # standard: .lora_A.weight (rank, in_dim), .lora_B.weight (out_dim, rank)
                if key.endswith(".lora_a"):
                    key = key[: -len(".lora_a")] + ".lora_A.weight"
                    param = mx.transpose(param)
                elif key.endswith(".lora_b"):
                    key = key[: -len(".lora_b")] + ".lora_B.weight"
                    param = mx.transpose(param)
                state_dict[key] = np.array(param.astype(mx.float32))
            save_safetensors(state_dict, str(saved_path))
        else:
            state_dict = {}
            for name, param in nn.utils.tree_flatten(self._transformer.parameters()):
                state_dict[name] = np.array(param)
            save_safetensors(state_dict, str(saved_path))

        logger.info(
            "%s weights for step %d saved in %s",
            prefix.capitalize(),
            self._global_step,
            saved_path.relative_to(self._config.output_dir),
        )

        self._checkpoint_paths.append(saved_path)
        self._cleanup_checkpoints()
        return saved_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        keep_n = self._config.checkpoints.keep_last_n
        if 0 < keep_n < len(self._checkpoint_paths):
            to_remove = self._checkpoint_paths[:-keep_n]
            for old in to_remove:
                if old.exists():
                    old.unlink()
                    logger.info("Removed old checkpoint: %s", old)
            self._checkpoint_paths = self._checkpoint_paths[-keep_n:]

    def _save_config(self) -> None:
        """Save the training configuration as YAML."""
        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)
        logger.info("Training configuration saved to: %s", config_path)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        logger.info(
            "Training Statistics:\n"
            " - Total time: %.1f minutes\n"
            " - Training speed: %.2f steps/second\n"
            " - Samples/second: %.2f\n"
            " - Peak memory: %.2f GB",
            stats.total_time_seconds / 60,
            stats.steps_per_second,
            stats.samples_per_second,
            stats.peak_memory_gb,
        )

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled:
            self._wandb_run = None
            return

        try:
            import wandb as _wandb

            wandb_config = self._config.wandb
            run = _wandb.init(
                project=wandb_config.project,
                entity=wandb_config.entity,
                name=Path(self._config.output_dir).name,
                tags=wandb_config.tags,
                config=self._config.model_dump(),
            )
            self._wandb_run = run
        except ImportError:
            logger.warning("wandb not installed. Disabling W&B logging.")
            self._wandb_run = None

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        """Log validation samples to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        try:
            import wandb as _wandb

            is_image = sample_paths and sample_paths[0].suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            media_cls = _wandb.Image if is_image else _wandb.Video
            samples = [media_cls(str(path), caption=prompt) for path, prompt in zip(sample_paths, prompts, strict=True)]
            self._wandb_run.log({"validation_samples": samples}, step=self._global_step)
        except ImportError:
            pass


# =============================================================================
# Helper utilities
# =============================================================================


def _find_lora_targets(
    model: nn.Module,
    target_names: list[str],
) -> list[tuple[str, nn.Linear]]:
    """Find all Linear layers matching target module names.

    Args:
        model: The model to search.
        target_names: List of module name suffixes to target.

    Returns:
        List of (full_path, module) pairs.
    """
    results: list[tuple[str, nn.Linear]] = []
    for path, module in model.named_modules():
        if isinstance(module, nn.Linear | nn.QuantizedLinear):
            name_parts = path.split(".")
            if any(part in target_names for part in name_parts):
                results.append((path, module))
    return results


def _set_module_by_path(model: nn.Module, path: str, new_module: nn.Module) -> None:
    """Set a module in a model by its dotted path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _simple_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool = True,
) -> Any:
    """Create a simple iterable data loader.

    MLX does not have a DataLoader class, so we implement a simple
    generator-based one.
    """

    class _DataLoader:
        def __init__(self, ds: Any, bs: int, do_shuffle: bool) -> None:
            self._dataset = ds
            self._batch_size = bs
            self._shuffle = do_shuffle

        def __iter__(self) -> Any:
            indices = list(range(len(self._dataset)))
            if self._shuffle:
                import random

                random.shuffle(indices)

            for i in range(0, len(indices), self._batch_size):
                batch_indices = indices[i : i + self._batch_size]
                if len(batch_indices) < self._batch_size:
                    continue  # Drop last incomplete batch
                batch = [self._dataset[j] for j in batch_indices]
                yield _collate_batch(batch)

    return _DataLoader(dataset, batch_size, shuffle)


def _collate_batch(samples: list[dict]) -> dict:
    """Collate a list of sample dicts into a batched dict.

    Stacks ``mx.array`` values along batch dimension, passes through others.
    """
    if not samples:
        return {}

    batch: dict[str, Any] = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        if isinstance(values[0], mx.array):
            batch[key] = mx.stack(values, axis=0)
        elif isinstance(values[0], dict):
            batch[key] = _collate_batch(values)
        else:
            batch[key] = values
    return batch
