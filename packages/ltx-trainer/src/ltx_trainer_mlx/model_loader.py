"""Model loader for LTX-2 MLX trainer.

Ported from ltx-trainer (Lightricks). Replaces ``SingleGPUModelBuilder``
with direct loading via ``load_split_safetensors`` + ``apply_quantization``,
matching the patterns in ``ti2vid_one_stage.py``.

Example usage::

    components = load_model("/path/to/model_dir")
    text_encoder = load_text_encoder("mlx-community/gemma-3-12b-it-4bit")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.transformer.model import LTXModel
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder, VideoEncoder
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors, remap_audio_vae_keys

logger = logging.getLogger(__name__)


@dataclass
class LtxModelComponents:
    """Container for all LTX-2 model components."""

    transformer: LTXModel
    video_vae_encoder: VideoEncoder | None = None
    video_vae_decoder: VideoDecoder | None = None
    audio_vae_decoder: AudioVAEDecoder | None = None
    vocoder: VocoderWithBWE | None = None
    text_encoder: GemmaLanguageModel | None = None
    feature_extractor: GemmaFeaturesExtractorV2 | None = None


# =============================================================================
# Individual Component Loaders
# =============================================================================


def load_transformer(
    model_dir: str | Path,
    transformer_file: str | None = None,
) -> LTXModel:
    """Load the LTX transformer model from split safetensors.

    Args:
        model_dir: Directory containing transformer safetensors.
        transformer_file: Override filename. If None, auto-detects:
            ``transformer.safetensors``, ``transformer-distilled.safetensors``,
            or ``transformer-dev.safetensors`` (in that order).

    Returns:
        Loaded ``LTXModel``.
    """
    model_dir = Path(model_dir)
    logger.debug("Loading transformer from %s", model_dir)

    if transformer_file is not None:
        tf_path = model_dir / transformer_file
    else:
        # Auto-detect transformer file
        for candidate in [
            "transformer.safetensors",
            "transformer-distilled.safetensors",
            "transformer-dev.safetensors",
        ]:
            tf_path = model_dir / candidate
            if tf_path.exists():
                break
        else:
            raise FileNotFoundError(f"No transformer safetensors found in {model_dir}")

    logger.info("Loading transformer from %s", tf_path.name)
    model = LTXModel()
    weights = load_split_safetensors(tf_path, prefix="transformer.")
    apply_quantization(model, weights)
    model.load_weights(list(weights.items()))
    aggressive_cleanup()

    return model


def load_video_vae_encoder(
    model_dir: str | Path,
) -> VideoEncoder:
    """Load the video VAE encoder.

    Args:
        model_dir: Directory containing ``vae_encoder.safetensors``.

    Returns:
        Loaded ``VideoEncoder``.
    """
    model_dir = Path(model_dir)
    logger.debug("Loading video VAE encoder from %s", model_dir)

    model = VideoEncoder()
    weights = load_split_safetensors(model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
    # Remap underscore-prefixed per-channel stats keys
    weights = {
        k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
        for k, v in weights.items()
    }
    model.load_weights(list(weights.items()))
    aggressive_cleanup()

    return model


def load_video_vae_decoder(
    model_dir: str | Path,
) -> VideoDecoder:
    """Load the video VAE decoder.

    Args:
        model_dir: Directory containing ``vae_decoder.safetensors``.

    Returns:
        Loaded ``VideoDecoder``.
    """
    model_dir = Path(model_dir)
    logger.debug("Loading video VAE decoder from %s", model_dir)

    model = VideoDecoder()
    weights = load_split_safetensors(model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
    model.load_weights(list(weights.items()))
    aggressive_cleanup()

    return model


def load_audio_vae_decoder(
    model_dir: str | Path,
) -> AudioVAEDecoder:
    """Load the audio VAE decoder.

    Args:
        model_dir: Directory containing ``audio_vae.safetensors``.

    Returns:
        Loaded ``AudioVAEDecoder``.
    """
    model_dir = Path(model_dir)
    logger.debug("Loading audio VAE decoder from %s", model_dir)

    model = AudioVAEDecoder()
    audio_weights = load_split_safetensors(model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder.")
    # Also load per_channel_statistics
    all_audio = load_split_safetensors(model_dir / "audio_vae.safetensors", prefix="audio_vae.")
    for k, v in all_audio.items():
        if k.startswith("per_channel_statistics."):
            audio_weights[k] = v
    audio_weights = remap_audio_vae_keys(audio_weights)
    model.load_weights(list(audio_weights.items()))
    aggressive_cleanup()

    return model


def load_vocoder(
    model_dir: str | Path,
) -> VocoderWithBWE:
    """Load the vocoder with bandwidth extension.

    Args:
        model_dir: Directory containing ``vocoder.safetensors``.

    Returns:
        Loaded ``VocoderWithBWE``.
    """
    model_dir = Path(model_dir)
    logger.debug("Loading vocoder from %s", model_dir)

    model = VocoderWithBWE()
    weights = load_split_safetensors(model_dir / "vocoder.safetensors", prefix="vocoder.")
    model.load_weights(list(weights.items()))
    aggressive_cleanup()

    return model


def load_text_encoder(
    gemma_model_path: str | Path,
) -> GemmaLanguageModel:
    """Load the Gemma text encoder via mlx-lm.

    Handles both local paths and HuggingFace repo IDs (e.g.
    ``"mlx-community/gemma-3-12b-it-4bit"``).

    Args:
        gemma_model_path: Path to Gemma model directory or HF repo ID.

    Returns:
        Loaded ``GemmaLanguageModel``.
    """
    logger.debug("Loading Gemma text encoder from %s", gemma_model_path)

    model = GemmaLanguageModel()
    model.load(str(gemma_model_path))
    aggressive_cleanup()

    return model


def load_feature_extractor(
    model_dir: str | Path,
) -> GemmaFeaturesExtractorV2:
    """Load the text embedding feature extractor (connector).

    Args:
        model_dir: Directory containing ``connector.safetensors``.

    Returns:
        Loaded ``GemmaFeaturesExtractorV2``.
    """
    model_dir = Path(model_dir)
    logger.debug("Loading feature extractor from %s", model_dir)

    model = GemmaFeaturesExtractorV2()
    connector_weights = load_split_safetensors(model_dir / "connector.safetensors", prefix="connector.")
    model.connector.load_weights(list(connector_weights.items()))
    aggressive_cleanup()

    return model


# =============================================================================
# Combined Component Loader
# =============================================================================


def load_model(
    model_dir: str | Path,
    gemma_model_path: str | Path | None = None,
    with_video_vae_encoder: bool = False,
    with_video_vae_decoder: bool = True,
    with_audio_vae_decoder: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
) -> LtxModelComponents:
    """Load LTX-2 model components from a model directory.

    This is a convenience function that loads multiple components at once.
    For loading individual components, use the dedicated ``load_*`` functions.

    Args:
        model_dir: Directory containing safetensors weight files.
        gemma_model_path: Path to Gemma model directory or HF repo ID.
            Required if ``with_text_encoder=True``.
        with_video_vae_encoder: Whether to load the video VAE encoder.
        with_video_vae_decoder: Whether to load the video VAE decoder.
        with_audio_vae_decoder: Whether to load the audio VAE decoder.
        with_vocoder: Whether to load the vocoder.
        with_text_encoder: Whether to load the text encoder.

    Returns:
        ``LtxModelComponents`` containing all loaded model components.
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    logger.info("Loading LTX-2 model from %s", model_dir)

    # Load transformer
    logger.debug("Loading transformer...")
    transformer = load_transformer(model_dir)

    # Load video VAE encoder
    video_vae_encoder = None
    if with_video_vae_encoder:
        logger.debug("Loading video VAE encoder...")
        video_vae_encoder = load_video_vae_encoder(model_dir)

    # Load video VAE decoder
    video_vae_decoder = None
    if with_video_vae_decoder:
        logger.debug("Loading video VAE decoder...")
        video_vae_decoder = load_video_vae_decoder(model_dir)

    # Load audio VAE decoder
    audio_vae_decoder = None
    if with_audio_vae_decoder:
        logger.debug("Loading audio VAE decoder...")
        audio_vae_decoder = load_audio_vae_decoder(model_dir)

    # Load vocoder
    vocoder = None
    if with_vocoder:
        logger.debug("Loading vocoder...")
        vocoder = load_vocoder(model_dir)

    # Load text encoder
    text_encoder = None
    feature_extractor = None
    if with_text_encoder:
        if gemma_model_path is None:
            raise ValueError("gemma_model_path must be provided when with_text_encoder=True")
        logger.debug("Loading Gemma text encoder...")
        text_encoder = load_text_encoder(gemma_model_path)
        logger.debug("Loading feature extractor...")
        feature_extractor = load_feature_extractor(model_dir)

    return LtxModelComponents(
        transformer=transformer,
        video_vae_encoder=video_vae_encoder,
        video_vae_decoder=video_vae_decoder,
        audio_vae_decoder=audio_vae_decoder,
        vocoder=vocoder,
        text_encoder=text_encoder,
        feature_extractor=feature_extractor,
    )
