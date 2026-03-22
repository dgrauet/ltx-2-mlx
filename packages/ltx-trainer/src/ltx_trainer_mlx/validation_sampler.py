"""Validation sampling for LTX-2 training using MLX.

Ported from ltx-trainer (Lightricks). Uses existing pipeline infrastructure
(``denoise_loop``, ``X0Model``, etc.) from ``ltx_pipelines_mlx``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    VideoConditionByLatentIndex,
    create_initial_state,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.samplers import denoise_loop
from ltx_trainer_mlx.progress import SamplingContext

if TYPE_CHECKING:
    from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
    from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
    from ltx_core_mlx.model.transformer.model import LTXModel
    from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder, VideoEncoder
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
    from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2

logger = logging.getLogger(__name__)


@dataclass
class CachedPromptEmbeddings:
    """Pre-computed text embeddings for a validation prompt.

    These embeddings are computed once at training start and reused for all
    validation runs, avoiding the need to load the full Gemma text encoder
    during validation.
    """

    video_context: mx.array  # [1, seq_len, hidden_dim]
    audio_context: mx.array  # [1, seq_len, hidden_dim]


@dataclass
class GenerationConfig:
    """Configuration for video/audio generation during validation."""

    prompt: str
    negative_prompt: str = ""
    height: int = 544
    width: int = 960
    num_frames: int = 97
    frame_rate: float = 25.0
    num_inference_steps: int = 8
    guidance_scale: float = 4.0
    seed: int = 42
    condition_image: str | None = None
    reference_video: str | None = None
    reference_downscale_factor: int = 1
    generate_audio: bool = True
    include_reference_in_output: bool = False
    cached_embeddings: CachedPromptEmbeddings | None = None
    stg_scale: float = 0.0
    stg_blocks: list[int] | None = None
    stg_mode: Literal["stg_av", "stg_v"] = "stg_av"


class ValidationSampler:
    """Generates validation samples during training using MLX.

    Supports:
    - Text-to-video generation
    - Image-to-video generation (first frame conditioning)
    - Optional audio generation

    Text embeddings can be provided either via a full text encoder or
    pre-computed ``CachedPromptEmbeddings``.
    """

    def __init__(
        self,
        transformer: LTXModel,
        vae_decoder: VideoDecoder,
        vae_encoder: VideoEncoder | None,
        text_encoder: GemmaLanguageModel | None = None,
        feature_extractor: GemmaFeaturesExtractorV2 | None = None,
        audio_decoder: AudioVAEDecoder | None = None,
        vocoder: VocoderWithBWE | None = None,
        sampling_context: SamplingContext | None = None,
    ):
        """Initialize the validation sampler.

        Args:
            transformer: LTX-2 transformer model.
            vae_decoder: Video VAE decoder.
            vae_encoder: Video VAE encoder (for image conditioning). Can be
                ``None`` if not needed.
            text_encoder: Gemma text encoder (optional if ``cached_embeddings``
                is provided in config).
            feature_extractor: Feature extractor for text embeddings.
            audio_decoder: Optional audio VAE decoder.
            vocoder: Optional vocoder.
            sampling_context: Optional progress tracking context.
        """
        self._transformer = transformer
        self._vae_decoder = vae_decoder
        self._vae_encoder = vae_encoder
        self._text_encoder = text_encoder
        self._feature_extractor = feature_extractor
        self._audio_decoder = audio_decoder
        self._vocoder = vocoder
        self._sampling_context = sampling_context

        self._video_patchifier = VideoLatentPatchifier()
        self._audio_patchifier = AudioPatchifier()

    def generate(
        self,
        config: GenerationConfig,
    ) -> tuple[mx.array, mx.array | None]:
        """Generate a video (and optionally audio) sample.

        Args:
            config: Generation configuration.

        Returns:
            Tuple of (video, audio) where:
            - video: ``mx.array`` of shape ``(C, F, H, W)`` in ``[0, 1]``
            - audio: ``mx.array`` of shape ``(C, samples)`` or ``None``
        """
        self._validate_config(config)
        return self._generate_standard(config)

    def _generate_standard(self, config: GenerationConfig) -> tuple[mx.array, mx.array | None]:
        """Standard generation (text-to-video or image-to-video)."""
        # Get prompt embeddings
        video_embeds, audio_embeds = self._get_prompt_embeddings(config)

        # Compute latent shapes
        F, H, W = compute_video_latent_shape(config.num_frames, config.height, config.width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(config.num_frames, fps=config.frame_rate)
        audio_shape = (1, audio_T, 128)

        # Compute positions
        video_positions = compute_video_positions(F, H, W, fps=config.frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # Create initial noise states
        video_state = create_initial_state(video_shape, config.seed, positions=video_positions)
        audio_state = create_initial_state(audio_shape, config.seed + 1, positions=audio_positions)

        # Apply image conditioning if provided
        if config.condition_image is not None:
            video_state = self._apply_image_conditioning(video_state, config, (F, H, W))

        # Run denoising
        sigmas = DISTILLED_SIGMAS[: config.num_inference_steps + 1]
        x0_model = X0Model(self._transformer)

        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
            show_progress=False,
        )
        # Force evaluation of the compute graph
        _force_eval(output.video_latent)
        if output.audio_latent is not None:
            _force_eval(output.audio_latent)

        # Update progress
        if self._sampling_context is not None:
            for _ in range(config.num_inference_steps):
                self._sampling_context.advance_step()

        # Unpatchify video
        video_latent = self._video_patchifier.unpatchify(output.video_latent, (F, H, W))

        # Decode video
        video_output = self._decode_video(video_latent)

        # Decode audio
        audio_output = None
        if config.generate_audio and self._audio_decoder is not None and self._vocoder is not None:
            audio_latent = self._audio_patchifier.unpatchify(output.audio_latent)
            audio_output = self._decode_audio(audio_latent)

        return video_output, audio_output

    def _apply_image_conditioning(
        self,
        video_state: LatentState,
        config: GenerationConfig,
        spatial_dims: tuple[int, int, int],
    ) -> LatentState:
        """Apply first-frame image conditioning to the video state."""
        assert config.condition_image is not None
        assert self._vae_encoder is not None

        # Prepare image for encoding: (1, 3, H, W) in [-1, 1]
        image = prepare_image_for_encoding(config.condition_image, config.height, config.width)
        # Add frame dim: (1, 3, 1, H, W)
        image = image[:, :, None, :, :]

        # Encode with VAE
        encoded = self._vae_encoder(image)
        _force_eval(encoded)

        # Patchify the encoded image (single frame)
        patchified, _ = self._video_patchifier.patchify(encoded)

        # Apply conditioning at first frame
        cond = VideoConditionByLatentIndex(
            frame_indices=[0],
            clean_latent=patchified,
            strength=1.0,
        )
        return cond.apply(video_state, spatial_dims)

    def _get_prompt_embeddings(self, config: GenerationConfig) -> tuple[mx.array, mx.array]:
        """Get prompt embeddings from cache or encode on-the-fly.

        Returns:
            Tuple of (video_embeds, audio_embeds).
        """
        if config.cached_embeddings is not None:
            cached = config.cached_embeddings
            return cached.video_context, cached.audio_context

        # Encode on-the-fly
        if self._text_encoder is None or self._feature_extractor is None:
            raise ValueError("text_encoder and feature_extractor are required when no cached embeddings provided")

        all_hidden_states, attention_mask = self._text_encoder.encode_all_layers(config.prompt)
        video_embeds, audio_embeds = self._feature_extractor(all_hidden_states, attention_mask=attention_mask)
        return video_embeds, audio_embeds

    def _decode_video(self, latent: mx.array) -> mx.array:
        """Decode video latent to pixel space.

        Args:
            latent: Video latent of shape ``(B, C, F, H, W)``.

        Returns:
            Video tensor of shape ``(C, F, H, W)`` in ``[0, 1]``.
        """
        latent = latent.astype(mx.bfloat16)
        decoded = self._vae_decoder(latent)
        _force_eval(decoded)
        # Normalise from [-1, 1] to [0, 1]
        decoded = mx.clip((decoded + 1.0) / 2.0, 0.0, 1.0)
        return decoded[0]  # Remove batch dim

    def _decode_audio(self, latent: mx.array) -> mx.array:
        """Decode audio latent to waveform.

        Args:
            latent: Audio latent of shape ``(B, 8, T, 16)``.

        Returns:
            Audio waveform of shape ``(C, samples)``.
        """
        assert self._audio_decoder is not None
        assert self._vocoder is not None

        latent = latent.astype(mx.bfloat16)
        decoded_audio = self._audio_decoder(latent)
        _force_eval(decoded_audio)
        aggressive_cleanup()

        waveform = self._vocoder(decoded_audio)
        _force_eval(waveform)

        return waveform[0]  # Remove batch dim

    def _validate_config(self, config: GenerationConfig) -> None:
        """Validate generation configuration."""
        if config.height % 32 != 0 or config.width % 32 != 0:
            raise ValueError(f"height and width must be divisible by 32, got {config.height}x{config.width}")
        if config.num_frames % 8 != 1:
            raise ValueError(f"num_frames must satisfy num_frames %% 8 == 1, got {config.num_frames}")
        if config.generate_audio and (self._audio_decoder is None or self._vocoder is None):
            raise ValueError("Audio generation requires audio_decoder and vocoder")
        if config.condition_image is not None and self._vae_encoder is None:
            raise ValueError("Image conditioning requires vae_encoder")
        if config.cached_embeddings is None and self._text_encoder is None:
            raise ValueError("Either text_encoder or config.cached_embeddings must be provided")


def _force_eval(arr: mx.array) -> None:
    """Force MLX compute graph evaluation.

    This is the MLX equivalent of synchronising a CUDA stream -- it
    materialises all lazy computations for the given array.
    """
    mx.eval(arr)
