"""Text encoders — Gemma 3 wrapper with dual video/audio projections."""

from ltx_core_mlx.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import (
    GemmaFeaturesExtractorV2,
    TextEmbeddingProjection,
    TextEncoderConnector,
)

__all__ = [
    "Embeddings1DConnector",
    "GemmaFeaturesExtractorV2",
    "GemmaLanguageModel",
    "TextEmbeddingProjection",
    "TextEncoderConnector",
]
