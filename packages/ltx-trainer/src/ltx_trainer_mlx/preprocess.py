"""Data preprocessing for LTX-2 MLX training.

Encodes raw videos and captions into precomputed latents and text embeddings
for use with ``PrecomputedDataset``.

Output structure::

    output_dir/
      .precomputed/
        latents/
          latent_0000.safetensors   # {latents, num_frames, height, width, fps}
          latent_0001.safetensors
        conditions/
          condition_0000.safetensors # {video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask}
          condition_0001.safetensors
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file as save_safetensors

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.utils.memory import aggressive_cleanup

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _force_eval(*arrays: mx.array) -> None:
    """Force MLX lazy compute graph evaluation (NOT Python eval)."""
    # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
    mx.eval(*arrays)


def preprocess_dataset(
    videos_dir: str,
    output_dir: str,
    model_dir: str,
    gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
    target_height: int | None = None,
    target_width: int | None = None,
    max_frames: int = 97,
    captions_dir: str | None = None,
    caption_ext: str = ".txt",
) -> None:
    """Preprocess a directory of videos into training-ready latents and conditions.

    Args:
        videos_dir: Directory containing video files.
        output_dir: Output directory for preprocessed data.
        model_dir: Model directory containing VAE encoder weights.
        gemma_model_id: Gemma model for text encoding.
        target_height: Resize height (must be divisible by 32). None = auto from video.
        target_width: Resize width (must be divisible by 32). None = auto from video.
        max_frames: Maximum frames per video (must satisfy frames % 8 == 1).
        captions_dir: Directory with .txt caption files matching video stems.
            If None, uses video filename as caption.
        caption_ext: Extension for caption files.
    """
    # Set Metal cache limit early to prevent GPU watchdog timeouts on 32GB Macs.
    # Without this, loading Gemma 12B (~7GB) triggers "Impacting Interactivity".
    mx.set_cache_limit(mx.device_info()["memory_size"])

    # Resolve HuggingFace repo ID to local path
    model_dir = _resolve_model_dir(model_dir)

    videos_path = Path(videos_dir)
    if not videos_path.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_path}")

    # Discover video files
    video_files = sorted(f for f in videos_path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file())
    if not video_files:
        raise ValueError(f"No video files found in {videos_path}")

    print(f"Found {len(video_files)} videos in {videos_path}")

    # Setup output directories
    precomputed = Path(output_dir) / ".precomputed"
    latents_dir = precomputed / "latents"
    conditions_dir = precomputed / "conditions"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    # Resolve captions
    captions = _resolve_captions(video_files, captions_dir, caption_ext)

    # Phase 1: Encode text (load Gemma + connector, encode all captions, then free)
    print("Phase 1: Encoding text prompts...")
    _encode_all_captions(
        captions=captions,
        conditions_dir=conditions_dir,
        model_dir=model_dir,
        gemma_model_id=gemma_model_id,
    )

    # Phase 2: Encode videos (load VAE encoder, encode all videos, then free)
    print("Phase 2: Encoding video latents...")
    _encode_all_videos(
        video_files=video_files,
        latents_dir=latents_dir,
        model_dir=model_dir,
        target_height=target_height,
        target_width=target_width,
        max_frames=max_frames,
    )

    print(f"\nPreprocessing complete! {len(video_files)} samples saved to {precomputed}")
    print(f"  Latents:    {latents_dir}")
    print(f"  Conditions: {conditions_dir}")


def _resolve_captions(
    video_files: list[Path],
    captions_dir: str | None,
    caption_ext: str,
) -> list[str]:
    """Resolve captions for each video file."""
    captions: list[str] = []

    if captions_dir is not None:
        captions_path = Path(captions_dir)
        for video_file in video_files:
            caption_file = captions_path / f"{video_file.stem}{caption_ext}"
            if caption_file.exists():
                captions.append(caption_file.read_text().strip())
            else:
                caption = video_file.stem.replace("_", " ").replace("-", " ")
                logger.warning("No caption file for %s, using filename: '%s'", video_file.name, caption)
                captions.append(caption)
    else:
        for video_file in video_files:
            caption = video_file.stem.replace("_", " ").replace("-", " ")
            captions.append(caption)
        print("  No captions directory provided. Using video filenames as captions.")

    return captions


def _encode_all_captions(
    captions: list[str],
    conditions_dir: Path,
    model_dir: str,
    gemma_model_id: str,
) -> None:
    """Encode all captions and save as conditions files.

    Uses a two-phase approach to stay within 32GB memory:
    1. Load Gemma + connector together, encode + project each caption, save, free both
    Deduplicates identical captions to avoid redundant encoding.
    """
    from ltx_trainer_mlx.model_loader import load_feature_extractor, load_text_encoder

    # Check which outputs already exist
    needed: list[int] = []
    for i in range(len(captions)):
        output_path = conditions_dir / f"condition_{i:04d}.safetensors"
        if output_path.exists():
            print(f"  [{i + 1}/{len(captions)}] Skipping (exists): {output_path.name}")
        else:
            needed.append(i)

    if not needed:
        print("  All conditions already encoded.")
        return

    # Load Gemma first, then connector
    text_encoder = load_text_encoder(gemma_model_path=gemma_model_id)
    aggressive_cleanup()
    feature_extractor = load_feature_extractor(model_dir=model_dir)
    aggressive_cleanup()

    # Encode unique captions and project in one pass
    unique_results: dict[str, dict[str, np.ndarray]] = {}

    for i in needed:
        caption = captions[i]
        output_path = conditions_dir / f"condition_{i:04d}.safetensors"

        if caption not in unique_results:
            print(f"  [{i + 1}/{len(captions)}] Encoding: '{caption[:80]}{'...' if len(caption) > 80 else ''}'")

            all_hs, attn_mask = text_encoder.encode_all_layers(caption)
            _force_eval(*all_hs, attn_mask)

            video_embeds, audio_embeds = feature_extractor(all_hs, attention_mask=attn_mask)
            _force_eval(video_embeds, audio_embeds)

            unique_results[caption] = {
                "video_prompt_embeds": np.array(video_embeds[0].astype(mx.float32)),
                "audio_prompt_embeds": np.array(audio_embeds[0].astype(mx.float32)),
                "prompt_attention_mask": np.array(attn_mask[0].astype(mx.float32)),
            }

            del all_hs, video_embeds, audio_embeds, attn_mask
            aggressive_cleanup()
        else:
            print(f"  [{i + 1}/{len(captions)}] Reusing cached encoding")

        save_safetensors(unique_results[caption], str(output_path))

    del text_encoder, feature_extractor
    aggressive_cleanup()
    print("  Text encoding complete.")


def _encode_all_videos(
    video_files: list[Path],
    latents_dir: Path,
    model_dir: str,
    target_height: int | None,
    target_width: int | None,
    max_frames: int,
) -> None:
    """Encode all videos and save as latent files."""
    from ltx_trainer_mlx.model_loader import load_video_vae_encoder

    vae_encoder = load_video_vae_encoder(model_dir=model_dir)
    vae_encoder.freeze()

    for i, video_file in enumerate(video_files):
        output_path = latents_dir / f"latent_{i:04d}.safetensors"
        if output_path.exists():
            print(f"  [{i + 1}/{len(video_files)}] Skipping (exists): {output_path.name}")
            continue

        print(f"  [{i + 1}/{len(video_files)}] Encoding: {video_file.name}")

        try:
            _encode_single_video(
                video_file=video_file,
                output_path=output_path,
                vae_encoder=vae_encoder,
                target_height=target_height,
                target_width=target_width,
                max_frames=max_frames,
            )
        except Exception as e:
            logger.error("Failed to encode %s: %s", video_file.name, e)
            continue

        if i % 5 == 0:
            aggressive_cleanup()

    del vae_encoder
    aggressive_cleanup()
    print("  Video encoding complete.")


def _encode_single_video(
    video_file: Path,
    output_path: Path,
    vae_encoder: object,
    target_height: int | None,
    target_width: int | None,
    max_frames: int,
) -> None:
    """Encode a single video file into VAE latents."""
    from ltx_trainer_mlx.video_utils import read_video

    video, actual_fps = read_video(video_file, max_frames=max_frames)
    num_frames = video.shape[0]

    # Ensure frames % 8 == 1
    valid_frames = ((num_frames - 1) // 8) * 8 + 1
    if valid_frames < 1:
        raise ValueError(f"Video too short: {num_frames} frames")
    video = video[:valid_frames]
    num_frames = valid_frames

    # Determine target dimensions
    _, _, h, w = video.shape  # (F, C, H, W)
    if target_height is not None and target_width is not None:
        h, w = target_height, target_width
    else:
        h = (h // 32) * 32
        w = (w // 32) * 32

    if h == 0 or w == 0:
        raise ValueError(f"Video dimensions too small after rounding to 32: original shape {video.shape}")

    # Resize if needed
    if video.shape[2] != h or video.shape[3] != w:
        video = _resize_video(video, h, w)

    # Convert to [-1, 1] for VAE and reshape: (F, C, H, W) -> (1, C, F, H, W)
    video = video * 2.0 - 1.0
    video = video.transpose(1, 0, 2, 3)  # (C, F, H, W)
    video = video[None]  # (1, C, F, H, W)
    video = video.astype(mx.bfloat16)

    # Encode with VAE
    latent = vae_encoder.encode(video)
    _force_eval(latent)

    # Compute latent shape
    F_lat, H_lat, W_lat = compute_video_latent_shape(num_frames, h, w)

    save_safetensors(
        {
            "latents": np.array(latent[0].astype(mx.float32)),  # [C, F, H, W]
            "num_frames": np.array([F_lat], dtype=np.int32),
            "height": np.array([H_lat], dtype=np.int32),
            "width": np.array([W_lat], dtype=np.int32),
            "fps": np.array([actual_fps], dtype=np.float32),
        },
        str(output_path),
    )


def _resize_video(video: mx.array, target_h: int, target_w: int) -> mx.array:
    """Resize video frames using PIL Lanczos interpolation.

    Args:
        video: Video tensor of shape (F, C, H, W) in [0, 1].
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized video tensor of shape (F, C, target_h, target_w).
    """
    from PIL import Image

    frames = []
    video_np = np.array(video)
    for i in range(video_np.shape[0]):
        # (C, H, W) -> (H, W, C) for PIL
        frame = video_np[i].transpose(1, 2, 0)
        frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(frame_uint8)
        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
        frame_resized = np.array(img_resized).astype(np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        frames.append(frame_resized.transpose(2, 0, 1))

    return mx.array(np.stack(frames))


def _resolve_model_dir(model_dir: str) -> str:
    """Resolve a model directory path, downloading from HuggingFace if needed.

    Args:
        model_dir: Local path or HuggingFace repo ID.

    Returns:
        Resolved local path to the model directory.
    """
    model_path = Path(model_dir)
    if model_path.exists():
        return str(model_path)

    # Assume it's a HuggingFace repo ID — download/resolve
    from huggingface_hub import snapshot_download

    print(f"  Resolving model: {model_dir}")
    local_path = snapshot_download(model_dir)
    return local_path
