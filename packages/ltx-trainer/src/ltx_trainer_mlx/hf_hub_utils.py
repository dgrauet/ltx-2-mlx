"""HuggingFace Hub upload utilities for training artifacts.

Ported from ltx-trainer (Lightricks). Replaces ``torch.save`` with
``mx.save`` for LoRA weight upload. Video-to-GIF conversion uses imageio.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
from rich.progress import Progress, SpinnerColumn, TextColumn

from ltx_trainer_mlx.config import LtxTrainerConfig

logger = logging.getLogger(__name__)


def push_to_hub(weights_path: Path, sampled_videos_paths: list[Path], config: LtxTrainerConfig) -> None:
    """Push the trained LoRA weights to HuggingFace Hub.

    Args:
        weights_path: Path to the saved weights file.
        sampled_videos_paths: Paths to validation sample videos.
        config: Training configuration.
    """
    if not config.hub.hub_model_id:
        logger.warning("HuggingFace hub_model_id not specified, skipping push to hub")
        return

    api = HfApi()

    # Save original progress bar state
    original_progress_state = are_progress_bars_disabled()
    disable_progress_bars()

    try:
        try:
            repo = create_repo(
                repo_id=config.hub.hub_model_id,
                repo_type="model",
                exist_ok=True,
            )
            repo_id = repo.repo_id
            logger.info("Successfully created HuggingFace model repository at: %s", repo.url)
        except Exception as e:
            logger.error("Failed to create HuggingFace model repository: %s", e)
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                try:
                    # Copy weights
                    task_copy = progress.add_task("Copying weights...", total=None)
                    weights_dest = temp_path / weights_path.name
                    shutil.copy2(weights_path, weights_dest)
                    progress.update(task_copy, description="Weights copied")

                    # Create model card and save samples
                    task_card = progress.add_task("Creating model card and samples...", total=None)
                    _create_model_card(
                        output_dir=temp_path,
                        videos=sampled_videos_paths,
                        config=config,
                    )
                    progress.update(task_card, description="Model card and samples created")

                    # Upload everything at once
                    task_upload = progress.add_task("Pushing files to HuggingFace Hub...", total=None)
                    api.upload_folder(
                        folder_path=str(temp_path),
                        repo_id=repo_id,
                        repo_type="model",
                    )
                    progress.update(task_upload, description="Files pushed to HuggingFace Hub")
                    logger.info("Successfully pushed files to HuggingFace Hub")

                except Exception as e:
                    logger.error("Failed to process and push files to HuggingFace Hub: %s", e)
                    raise

    finally:
        if not original_progress_state:
            enable_progress_bars()


def convert_video_to_gif(video_path: Path, output_path: Path) -> None:
    """Convert a video file to GIF format.

    Args:
        video_path: Path to the source video.
        output_path: Path to save the GIF.
    """
    try:
        import imageio

        reader = imageio.get_reader(str(video_path))
        fps = reader.get_meta_data()["fps"]

        writer = imageio.get_writer(
            str(output_path),
            fps=min(fps, 15),
            loop=0,
        )

        for frame in reader:
            writer.append_data(frame)

        writer.close()
        reader.close()
    except Exception as e:
        logger.error("Failed to convert video to GIF: %s", e)


def _create_model_card(
    output_dir: str | Path,
    videos: list[Path],
    config: LtxTrainerConfig,
) -> Path:
    """Generate and save a model card for the trained model.

    Args:
        output_dir: Directory to write the model card.
        videos: Paths to validation sample videos.
        config: Training configuration.

    Returns:
        Path to the generated model card.
    """
    repo_id = config.hub.hub_model_id
    pretrained_model_name_or_path = config.model.model_path
    validation_prompts = config.validation.prompts
    output_dir = Path(output_dir)

    model_name = repo_id.split("/")[-1] if repo_id else "ltx-lora"
    base_model_name = str(pretrained_model_name_or_path)

    # Format validation prompts and create grid layout
    prompts_text = ""
    sample_grid: list[str] = []

    if validation_prompts and videos:
        prompts_text = "Example prompts used during validation:\n\n"
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True, parents=True)

        cells: list[str] = []
        for i, (prompt, video) in enumerate(zip(validation_prompts, videos, strict=False)):
            if video.exists():
                prompts_text += f"- `{prompt}`\n"
                gif_path = samples_dir / f"sample_{i}.gif"
                try:
                    convert_video_to_gif(video, gif_path)
                    cell = (
                        f"![example{i + 1}](./samples/sample_{i}.gif)"
                        "<br>"
                        '<details style="max-width: 300px; margin: auto;">'
                        f"<summary>Prompt</summary>"
                        f"{prompt}"
                        "</details>"
                    )
                    cells.append(cell)
                except Exception as e:
                    logger.error("Failed to process video %s: %s", video, e)

        num_cells = len(cells)
        if num_cells > 0:
            num_cols = min(4, num_cells)
            num_rows = (num_cells + num_cols - 1) // num_cols

            for row in range(num_rows):
                start_idx = row * num_cols
                end_idx = min(start_idx + num_cols, num_cells)
                row_cells = cells[start_idx:end_idx]
                formatted_row = "| " + " | ".join(row_cells) + " |"
                sample_grid.append(formatted_row)

    grid_text = "\n".join(sample_grid) if sample_grid else ""

    # Build model card content
    model_card_content = f"""---
tags:
- ltx-video
- lora
- mlx
base_model: {base_model_name}
---

# {model_name}

Fine-tuned LoRA weights for LTX-2 video generation, trained with ltx-trainer-mlx on Apple Silicon.

## Training Details

- **Base model**: {base_model_name}
- **Training type**: {"LoRA fine-tuning" if config.model.training_mode == "lora" else "Full model fine-tuning"}
- **Training steps**: {config.optimization.steps}
- **Learning rate**: {config.optimization.learning_rate}
- **Batch size**: {config.optimization.batch_size}

{prompts_text}

{grid_text}
"""

    model_card_path = output_dir / "README.md"
    model_card_path.write_text(model_card_content)

    return model_card_path
