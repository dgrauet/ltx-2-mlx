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
        audio_latents/               # only when with_audio=True
          latent_0000.safetensors    # {latents}  -- (8, T, 16) audio VAE latent
          latent_0001.safetensors

Note: audio latent files share the exact filename of their video counterpart
(``latent_XXXX.safetensors``) because ``PrecomputedDataset`` matches non-condition
sources by identical relative path.
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
    with_audio: bool = False,
    frame_rate: float | None = None,
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
        with_audio: If True, also encode each clip's audio into ``audio_latents/``
            for joint audio-video training. Clips that lack an audio stream are
            skipped (and will be dropped from the dataset, since all sources must
            match).
        frame_rate: Override the frame rate written into the latents and used to
            size audio tokens. None = use each clip's probed fps.
    """
    # Set Metal cache limit early to prevent GPU watchdog timeouts on 32GB Macs.
    # Without this, loading Gemma 12B (~7GB) triggers "Impacting Interactivity".
    mx.set_cache_limit(mx.device_info()["memory_size"])

    # Resolve HuggingFace repo ID to local path
    model_dir = _resolve_model_dir(model_dir)

    videos_path = Path(videos_dir)
    if not videos_path.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_path}")

    # Discover video files (recursive, so per-source subfolders from `ltx slice` work)
    video_files = sorted(f for f in videos_path.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file())
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
        frame_rate=frame_rate,
    )

    # Phase 3: Encode audio (optional; load audio VAE encoder, encode all, then free)
    audio_latents_dir = precomputed / "audio_latents"
    if with_audio:
        print("Phase 3: Encoding audio latents...")
        audio_latents_dir.mkdir(parents=True, exist_ok=True)
        _encode_all_audio(
            video_files=video_files,
            latents_dir=latents_dir,
            audio_latents_dir=audio_latents_dir,
            model_dir=model_dir,
            frame_rate=frame_rate,
        )

    print(f"\nPreprocessing complete! {len(video_files)} samples saved to {precomputed}")
    print(f"  Latents:    {latents_dir}")
    print(f"  Conditions: {conditions_dir}")
    if with_audio:
        print(f"  Audio:      {audio_latents_dir}")


def _resolve_captions(
    video_files: list[Path],
    captions_dir: str | None,
    caption_ext: str,
) -> list[str]:
    """Resolve captions for each video file."""
    captions: list[str] = []

    captions_path = Path(captions_dir) if captions_dir is not None else None
    for video_file in video_files:
        # 1) explicit captions dir (match by stem), 2) sibling .txt next to the clip
        #    (the layout `ltx slice` produces), 3) fall back to a cleaned filename.
        caption_file = None
        if captions_path is not None and (captions_path / f"{video_file.stem}{caption_ext}").exists():
            caption_file = captions_path / f"{video_file.stem}{caption_ext}"
        elif video_file.with_suffix(caption_ext).exists():
            caption_file = video_file.with_suffix(caption_ext)

        if caption_file is not None:
            captions.append(caption_file.read_text().strip())
        else:
            caption = video_file.stem.replace("_", " ").replace("-", " ")
            logger.warning("No caption file for %s, using filename: '%s'", video_file.name, caption)
            captions.append(caption)

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
    frame_rate: float | None = None,
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
                frame_rate=frame_rate,
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
    frame_rate: float | None = None,
) -> None:
    """Encode a single video file into VAE latents."""
    from ltx_trainer_mlx.video_utils import read_video

    video, actual_fps = read_video(video_file, max_frames=max_frames)
    # frame_rate override applies to both the saved fps (used for video positions)
    # and the audio token count, keeping the two modalities aligned.
    saved_fps = frame_rate if frame_rate is not None else actual_fps
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
            "fps": np.array([saved_fps], dtype=np.float32),
        },
        str(output_path),
    )


def _encode_all_audio(
    video_files: list[Path],
    latents_dir: Path,
    audio_latents_dir: Path,
    model_dir: str,
    frame_rate: float | None = None,
) -> None:
    """Encode each clip's audio track into audio VAE latents.

    Frame count and fps are read from the already-written video latent metadata
    (Phase 2), so the audio duration is guaranteed to match the encoded video.
    Clips that lack an audio stream are skipped, dropping them from the dataset.
    """
    from ltx_core_mlx.model.audio_vae import encode_audio
    from ltx_core_mlx.model.audio_vae.processor import AudioProcessor
    from ltx_core_mlx.utils.audio import load_audio
    from ltx_core_mlx.utils.positions import compute_audio_token_count
    from ltx_trainer_mlx.model_loader import load_audio_vae_encoder

    encoder = load_audio_vae_encoder(model_dir=model_dir)
    encoder.freeze()
    processor = AudioProcessor(sample_rate=16000)

    for i, video_file in enumerate(video_files):
        output_path = audio_latents_dir / f"latent_{i:04d}.safetensors"
        if output_path.exists():
            print(f"  [{i + 1}/{len(video_files)}] Skipping (exists): {output_path.name}")
            continue

        video_latent_path = latents_dir / f"latent_{i:04d}.safetensors"
        if not video_latent_path.exists():
            logger.warning("No video latent for %s — skipping audio (video encode failed?)", video_file.name)
            continue

        meta = mx.load(str(video_latent_path))
        num_latent_frames = int(meta["num_frames"].item())
        # Invert F_lat = (valid_frames - 1) // 8 + 1
        valid_frames = (num_latent_frames - 1) * 8 + 1
        fps = frame_rate if frame_rate is not None else float(meta["fps"].item())
        duration = valid_frames / fps

        print(f"  [{i + 1}/{len(video_files)}] Encoding audio: {video_file.name} ({duration:.2f}s)")

        try:
            _encode_single_audio(
                video_file=video_file,
                output_path=output_path,
                encoder=encoder,
                processor=processor,
                duration=duration,
                target_tokens=compute_audio_token_count(valid_frames, fps),
                encode_audio_fn=encode_audio,
                load_audio_fn=load_audio,
            )
        except Exception as e:
            logger.error("Failed to encode audio for %s: %s", video_file.name, e)
            continue

        if i % 5 == 0:
            aggressive_cleanup()

    del encoder
    aggressive_cleanup()
    print("  Audio encoding complete.")


def _encode_single_audio(
    video_file: Path,
    output_path: Path,
    encoder: object,
    processor: object,
    duration: float,
    target_tokens: int,
    encode_audio_fn: object,
    load_audio_fn: object,
) -> None:
    """Encode a single clip's audio into an audio VAE latent of ``target_tokens`` length."""
    audio = load_audio_fn(video_file, target_sample_rate=16000, max_duration=duration, mono=False)
    if audio is None:
        logger.warning("No audio stream in %s — skipping (sample dropped from dataset)", video_file.name)
        return

    latent = encode_audio_fn(audio.waveform, 16000, encoder, processor)  # (1, 8, T, 16)
    latent = _fit_audio_tokens(latent, target_tokens)
    _force_eval(latent)

    save_safetensors(
        {"latents": np.array(latent[0].astype(mx.float32))},  # (8, target_tokens, 16)
        str(output_path),
    )


def _fit_audio_tokens(latent: mx.array, target_tokens: int) -> mx.array:
    """Trim or edge-pad the audio latent time axis to ``target_tokens``.

    Args:
        latent: Audio latent of shape ``(1, 8, T, 16)``.
        target_tokens: Canonical token count from ``compute_audio_token_count``.

    Returns:
        Latent of shape ``(1, 8, target_tokens, 16)``.
    """
    t = latent.shape[2]
    if t == target_tokens:
        return latent
    if t > target_tokens:
        return latent[:, :, :target_tokens, :]
    # Pad by repeating the last token (avoids injecting normalized-zero artifacts).
    pad = mx.repeat(latent[:, :, -1:, :], target_tokens - t, axis=2)
    return mx.concatenate([latent, pad], axis=2)


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


# Files preprocessing actually loads. We deliberately exclude the ~20 GB
# transformer variants and LoRAs — they are only needed at train time, not for
# encoding latents/conditions, so fetching the full repo here wastes ~80 GB.
_PREPROCESS_MODEL_PATTERNS = [
    "connector.safetensors",  # Gemma feature extractor (load_feature_extractor)
    "vae_encoder.safetensors",  # video VAE encoder (load_video_vae_encoder)
    "audio_vae.safetensors",  # audio VAE encoder (load_audio_vae_encoder)
    "*.json",  # configs / index, small
]


def _resolve_model_dir(model_dir: str) -> str:
    """Resolve a model directory path, downloading from HuggingFace if needed.

    When given a HuggingFace repo ID, only the files preprocessing loads are
    fetched (see ``_PREPROCESS_MODEL_PATTERNS``) — not the full repo.

    Args:
        model_dir: Local path or HuggingFace repo ID.

    Returns:
        Resolved local path to the model directory.
    """
    model_path = Path(model_dir)
    if model_path.exists():
        return str(model_path)

    # Assume it's a HuggingFace repo ID — download only what preprocessing needs.
    from huggingface_hub import snapshot_download

    print(f"  Resolving model (encoder files only): {model_dir}")
    local_path = snapshot_download(model_dir, allow_patterns=_PREPROCESS_MODEL_PATTERNS)
    return local_path
