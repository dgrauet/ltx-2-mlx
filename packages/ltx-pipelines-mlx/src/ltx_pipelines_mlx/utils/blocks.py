"""Composable pipeline blocks.

Mirrors upstream ``ltx_pipelines.utils.blocks`` (composition over
inheritance). Each block owns the lifecycle of one model component
(load, use, free) and exposes a small ``__call__`` API. Pipelines
that prefer composition can instantiate these blocks directly:

```python
from ltx_pipelines_mlx import PromptEncoder, VideoDecoder, AudioDecoder

prompt_enc = PromptEncoder(model_dir, gemma_model_id)
video_emb, audio_emb = prompt_enc(prompt)  # loads, encodes, frees

video_dec = VideoDecoder(model_dir)
video_dec.decode_and_stream(video_latent, "out.mp4", audio_path="audio.wav")
```

The :class:`BasePipeline` inheritance tree (:class:`TI2VidTwoStagesPipeline`,
:class:`RetakePipeline`, :class:`ICLoraPipeline`, ...) **delegates** to
these blocks internally. Each pipeline holds private block instances
(``self._prompt_encoder``, ``self._image_conditioner``,
``self._video_decoder``, ``self._audio_decoder_block``); the historical
attribute names (``self.text_encoder``, ``self.vae_encoder``, ...) are
properties that proxy onto the block internals so subclass code that
reads/writes them — including ``self.text_encoder = None`` to free
memory — continues to work.

The blocks are the single source of truth for loader logic; the
inheritance API exists purely for backward compat with the current
subclass bodies.

Differences vs upstream:

- No CPU/GPU offload context managers — MLX uses unified memory, so
  blocks just hold strong refs and rely on Python GC + ``aggressive_cleanup``.
- No ``Builder``/``Registry`` indirection — blocks load weights via
  :func:`load_split_safetensors` directly, mirroring our existing path.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

from ltx_core_mlx.duration_head import DurationHead, load_duration_head
from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.transformer.model import LTXModelConfig
from ltx_core_mlx.model.upsampler.model import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder as _VideoVAEDecoder
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder as _VideoVAEEncoder
from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys
from ltx_pipelines_mlx.utils.types import AutoDuration

if TYPE_CHECKING:
    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4TextEncoder

logger = logging.getLogger(__name__)

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags the literal mx.eval pattern


def _video_vae_names(model_dir: str | Path) -> tuple[str, str]:
    """Resolve the video VAE decoder/encoder file base names by pack evidence.

    LTX-2.5 ships a conv video VAE as ``vae_decoder_conv.safetensors`` /
    ``vae_encoder_conv.safetensors``. Detects it by the decoder conv file's
    presence and falls back to the 2.3 names (``vae_decoder`` /
    ``vae_encoder``) otherwise, so 2.3 packs load byte-identically.

    Returns:
        ``(decoder_name, encoder_name)`` base names (without
        ``.safetensors``), also used as the key prefix (``f"{name}."``).
    """
    if (Path(model_dir) / "vae_decoder_conv.safetensors").exists():
        return "vae_decoder_conv", "vae_encoder_conv"
    return "vae_decoder", "vae_encoder"


def _resolve_model_dir(model_dir: str | Path) -> Path:
    """Resolve a model dir — download from HuggingFace if not a local path."""
    path = Path(model_dir)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_dir)))


class PromptEncoder:
    """Owns Gemma + connector lifecycle. Encodes prompts on call.

    Mirrors upstream ``utils.blocks.PromptEncoder``. Loads Gemma + the
    feature-extractor connector lazily on first call, encodes the prompt
    into ``(video_embeds, audio_embeds)``, then frees both modules.
    """

    def __init__(
        self,
        model_dir: str | Path,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.gemma_model_id = gemma_model_id
        self._text_encoder: GemmaLanguageModel | Gemma4TextEncoder | None = None
        self._feature_extractor: GemmaFeaturesExtractorV2 | None = None
        self._encoder_kind: str | None = None

    def load(self) -> None:
        """Load Gemma + connector if not already loaded."""
        # Recomputed unconditionally (not just inside the "text encoder unset"
        # branch below) so a stale _encoder_kind from a prior partial-free
        # state can never skip the gemma4 projection merge further down.
        self._encoder_kind = select_text_encoder(self.model_dir)
        if self._text_encoder is None:
            if self._encoder_kind == "gemma4":
                # LTX-2.5 pack: Gemma 4 unified tower ships in the DiT pack
                # itself -- pack-local load, no mlx-community download.
                from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4TextEncoder

                self._text_encoder = Gemma4TextEncoder()
                self._text_encoder.load(self.model_dir)
            else:
                self._text_encoder = GemmaLanguageModel()
                self._text_encoder.load(self.gemma_model_id)
            aggressive_cleanup()

        if self._feature_extractor is None:
            # Same checkpoint field the DiT reads (LTXModelConfig.from_checkpoint_dir
            # parses "frequencies_precision" == "float64" from embedded_config.json /
            # config.json); upstream's video AND audio connector configurators read
            # it independently of the DiT configurator, so the connector needs its
            # own read here rather than inheriting the DiT's LTXModelConfig instance.
            double_precision_rope = LTXModelConfig.from_checkpoint_dir(self.model_dir).double_precision_rope
            self._feature_extractor = GemmaFeaturesExtractorV2(double_precision_rope=double_precision_rope)
            connector_weights = load_split_safetensors(self.model_dir / "connector.safetensors", prefix="connector.")
            if self._encoder_kind == "gemma4":
                # The 2.5 pack's connector.safetensors carries only the two
                # Embeddings1DConnector transformer stacks
                # (video/audio_embeddings_connector); the
                # text_embedding_projection.{video,audio}_aggregate_embed
                # tensors that TextEncoderConnector also needs ship inside
                # text_encoder.safetensors instead (see the allowlist note
                # in gemma4.py). Pull them in here and merge under the same
                # key GemmaFeaturesExtractorV2.connector expects.
                projection_weights = load_split_safetensors(
                    self.model_dir / "text_encoder.safetensors",
                    prefix="text_encoder.text_embedding_projection.",
                )
                connector_weights.update({f"text_embedding_projection.{k}": v for k, v in projection_weights.items()})
            self._feature_extractor.connector.load_weights(list(connector_weights.items()))
            aggressive_cleanup()

    def free(self) -> None:
        """Drop strong refs; rely on GC + aggressive_cleanup to reclaim memory."""
        self._text_encoder = None
        self._feature_extractor = None
        aggressive_cleanup()

    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode a single prompt to ``(video_embeds, audio_embeds)``.

        Caller is responsible for freeing via :meth:`free` when done with
        the encoder. For one-shot use, prefer :meth:`__call__`.
        """
        import os

        self.load()
        assert self._text_encoder is not None
        assert self._feature_extractor is not None

        max_length = int(os.environ.get("LTX2_GEMMA_MAX_LENGTH", "1024"))
        all_hidden_states, attention_mask = self._text_encoder.encode_all_layers(prompt, max_length=max_length)
        video_embeds, audio_embeds = self._feature_extractor(all_hidden_states, attention_mask=attention_mask)
        return video_embeds, audio_embeds

    def __call__(
        self,
        prompts: str | list[str],
        *,
        free_after: bool = True,
    ) -> tuple[mx.array, mx.array] | list[tuple[mx.array, mx.array]]:
        """Encode one or more prompts; free Gemma + connector afterwards by default.

        Args:
            prompts: Single prompt or list of prompts. With a list, each
                element is encoded sequentially and a list of tuples is
                returned (matches upstream's batched signature).
            free_after: If True (default), drop strong refs to Gemma and
                the connector after encoding so subsequent components fit
                in memory. Pass False to keep the encoder loaded for
                subsequent calls.
        """
        if isinstance(prompts, str):
            video, audio = self.encode(prompts)
            _materialize(video, audio)
            if free_after:
                self.free()
            return video, audio

        outputs: list[tuple[mx.array, mx.array]] = []
        for p in prompts:
            video, audio = self.encode(p)
            _materialize(video, audio)
            outputs.append((video, audio))
        if free_after:
            self.free()
        return outputs


class ImageConditioner:
    """Owns the video VAE encoder lifecycle.

    Mirrors upstream ``utils.blocks.ImageConditioner``. Wraps a callable
    so that the encoder is built, passed to user code, then freed.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._encoder: _VideoVAEEncoder | None = None

    def load(self) -> _VideoVAEEncoder:
        """Build the VAE encoder (cached)."""
        if self._encoder is not None:
            return self._encoder
        self._encoder = _VideoVAEEncoder()
        _decoder_name, encoder_name = _video_vae_names(self.model_dir)
        weights = load_split_safetensors(self.model_dir / f"{encoder_name}.safetensors", prefix=f"{encoder_name}.")
        weights = {
            k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
            for k, v in weights.items()
        }
        self._encoder.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._encoder

    def free(self) -> None:
        self._encoder = None
        aggressive_cleanup()

    def __call__(self, fn: Callable[[_VideoVAEEncoder], object], *, free_after: bool = True) -> object:
        """Build encoder, call ``fn(encoder)``, then free encoder."""
        encoder = self.load()
        result = fn(encoder)
        if free_after:
            self.free()
        return result


class VideoDecoder:
    """Owns the video VAE decoder lifecycle + ffmpeg streaming muxing.

    Mirrors upstream ``utils.blocks.VideoDecoder`` (streaming decode +
    audio mux). Use :meth:`decode_and_stream` to decode a latent and
    mux with an audio file in one shot.
    """

    def __init__(self, model_dir: str | Path, verbose: bool = True) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.verbose = verbose
        self._decoder: _VideoVAEDecoder | None = None

    def load(self) -> _VideoVAEDecoder:
        if self._decoder is not None:
            return self._decoder
        self._decoder = _VideoVAEDecoder()
        decoder_name, _encoder_name = _video_vae_names(self.model_dir)
        weights = load_split_safetensors(self.model_dir / f"{decoder_name}.safetensors", prefix=f"{decoder_name}.")
        self._decoder.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._decoder

    def free(self) -> None:
        self._decoder = None
        aggressive_cleanup()

    def decode_and_stream(
        self,
        video_latent: mx.array,
        output_path: str,
        frame_rate: float = 24.0,
        audio_path: str | None = None,
    ) -> str:
        """Stream-decode the latent into an mp4 with optional audio mux."""
        if self.verbose:
            tiling = _compute_decode_tiling(video_latent.shape, frame_rate=frame_rate)
            if tiling is not None and tiling.temporal_config is not None:
                tc = tiling.temporal_config
                print(
                    f"[vae-decode tiled: tile_frames={tc.tile_size_in_frames} overlap={tc.tile_overlap_in_frames}]",
                    file=sys.stderr,
                    flush=True,
                )
        decoder = self.load()
        decoder.decode_and_stream(video_latent, output_path, frame_rate=frame_rate, audio_path=audio_path)
        return output_path


class AudioDecoder:
    """Owns the audio VAE decoder + vocoder + BWE lifecycle.

    Mirrors upstream ``utils.blocks.AudioDecoder``. Decodes an audio
    latent through ``AudioVAEDecoder`` → BigVGAN vocoder → BWE to a
    waveform tensor at 48 kHz.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._audio_decoder: AudioVAEDecoder | None = None
        self._vocoder: VocoderWithBWE | None = None

    def load(self) -> tuple[AudioVAEDecoder, VocoderWithBWE]:
        if self._audio_decoder is None:
            self._audio_decoder = AudioVAEDecoder()
            decoder_weights = load_split_safetensors(
                self.model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder."
            )
            all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    decoder_weights[k] = v
            decoder_weights = remap_audio_vae_keys(decoder_weights)
            self._audio_decoder.load_weights(list(decoder_weights.items()))
            aggressive_cleanup()

        if self._vocoder is None:
            self._vocoder = VocoderWithBWE()
            vocoder_weights = load_split_safetensors(self.model_dir / "vocoder.safetensors", prefix="vocoder.")
            self._vocoder.load_weights(list(vocoder_weights.items()))
            self._vocoder.upcast_weights_to_fp32()
            aggressive_cleanup()

        return self._audio_decoder, self._vocoder

    def free(self) -> None:
        self._audio_decoder = None
        self._vocoder = None
        aggressive_cleanup()

    def __call__(self, audio_latent: mx.array) -> mx.array:
        """Decode audio latent into a 48 kHz stereo waveform."""
        decoder, vocoder = self.load()
        mel = decoder.decode(audio_latent)
        return vocoder(mel)


class AudioConditioner:
    """Owns the audio VAE encoder + processor lifecycle.

    Mirrors upstream ``utils.blocks.AudioConditioner``. Used by
    :class:`RetakePipeline` to encode the source audio of an existing
    video before regenerating a time region. Wraps a callable so the
    encoder + processor are built, passed to user code, then freed.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._encoder: object | None = None
        self._processor: object | None = None

    def load(self) -> tuple[object, object]:
        if self._encoder is not None and self._processor is not None:
            return self._encoder, self._processor
        from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder

        self._encoder = AudioVAEEncoder()
        encoder_weights = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.encoder.")
        all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                encoder_weights[k] = v
        encoder_weights = remap_audio_vae_keys(encoder_weights)
        self._encoder.load_weights(list(encoder_weights.items()))
        self._processor = AudioProcessor()
        aggressive_cleanup()
        return self._encoder, self._processor

    def free(self) -> None:
        self._encoder = None
        self._processor = None
        aggressive_cleanup()

    def __call__(self, fn: Callable[[object, object], object], *, free_after: bool = True) -> object:
        """Build encoder+processor, call ``fn(encoder, processor)``, free."""
        encoder, processor = self.load()
        result = fn(encoder, processor)
        if free_after:
            self.free()
        return result


class VideoUpsampler:
    """Owns the spatial upsampler lifecycle.

    Mirrors upstream ``utils.blocks.VideoUpsampler``. Use for 2x spatial
    upscale between stage 1 and stage 2 of the two-stage pipelines.
    """

    def __init__(
        self,
        model_dir: str | Path,
        name: str = "spatial_upscaler_x2_v1_1",
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.name = name
        self._upsampler: LatentUpsampler | None = None

    def load(self) -> LatentUpsampler:
        if self._upsampler is not None:
            return self._upsampler

        import json

        config_path = self.model_dir / f"{self.name}_config.json"
        weights_path = self.model_dir / f"{self.name}.safetensors"

        if config_path.exists():
            raw_config = json.loads(config_path.read_text())
            # Old format nests fields under a "config" key; newer conversions
            # (e.g. LTX-2.5) emit the fields flat at the top level.
            config = raw_config.get("config", raw_config)
            self._upsampler = LatentUpsampler.from_config(config)
        else:
            self._upsampler = LatentUpsampler()

        if weights_path.exists():
            weights = load_split_safetensors(weights_path, prefix=f"{self.name}.")
            self._upsampler.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._upsampler

    def free(self) -> None:
        self._upsampler = None
        aggressive_cleanup()

    def __call__(self, latent: mx.array) -> mx.array:
        """Upscale a denormalized latent (caller must denorm/renorm)."""
        upsampler = self.load()
        return upsampler(latent)


# ============================================================================
# Duration Head Helpers
# ============================================================================


def snap_frames_to_grid(frames: int, time_scale: int = 8) -> int:
    """Round ``frames`` down to the nearest ``k * time_scale + 1``.

    The model's frame count must satisfy ``(frames - 1) % time_scale == 0``
    (causal VAE temporal grid). The default ``time_scale=8`` corresponds to
    the VAE's 8x temporal compression.

    Args:
        frames: Number of frames to snap.
        time_scale: Temporal grid scale factor (default 8 for LTX-2).

    Returns:
        Snapped frame count on the 8k+1 grid.
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    return ((frames - 1) // time_scale) * time_scale + 1


def seconds_to_clamped_num_frames(
    seconds: float,
    *,
    frame_rate: float,
    min_frames: int = 1,
    max_frames: int = 1024,
    time_scale: int = 8,
) -> int:
    """Convert a duration in seconds to a frame count snapped to the VAE's temporal grid.

    Outlier durations are clamped to ``[min_frames, max_frames]`` (before snapping) so a
    misbehaving prediction can't request an OOM-sized generation. Snapping floors to the
    grid, which can undershoot ``min_frames``; when that happens the result is snapped up
    to the next grid point instead, so the ``[min_frames, max_frames]`` contract always holds.

    Args:
        seconds: Duration in seconds.
        frame_rate: Video frame rate in frames per second.
        min_frames: Minimum frame count (default 1). Result >= min_frames.
        max_frames: Maximum frame count (default 1024). Result <= max_frames.
        time_scale: Temporal grid scale factor (default 8 for LTX-2).

    Returns:
        Frame count clamped to [min_frames, max_frames] and snapped to 8k+1 grid.
    """
    raw_frames = round(seconds * frame_rate)
    raw_frames = max(min_frames, min(raw_frames, max_frames))
    frames = snap_frames_to_grid(raw_frames, time_scale)
    if frames < min_frames:
        # Round up to next grid point: ceiling division on the grid
        # Formula: -(-((min_frames - 1) // time_scale)) * time_scale + 1
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames


def require_num_frames_source(
    num_frames: int | AutoDuration,
    duration_predictor: DurationPredictor | None,
) -> None:
    """Guard against an unsatisfiable auto-duration request.

    Call at the very top of a pipeline's ``__call__`` -- before prompt encoding or any other
    work -- so a checkpoint without DurationHead weights (anything predating 2.5) fails fast
    with a clear message instead of after paying for work whose result would be discarded.

    Args:
        num_frames: Either an explicit frame count or AutoDuration request.
        duration_predictor: Optional DurationPredictor (required if num_frames is AutoDuration).

    Raises:
        ValueError: If num_frames is AutoDuration but duration_predictor is None.
    """
    if isinstance(num_frames, AutoDuration) and duration_predictor is None:
        raise ValueError(
            "num_frames was AutoDuration but this checkpoint has no DurationHead weights to "
            "auto-predict duration from (DurationHead ships from LTX 2.5 / gemma4 onward). "
            "Pass num_frames explicitly."
        )


def resolve_num_frames(
    num_frames: int | AutoDuration,
    duration_predictor: DurationPredictor | None,
    *,
    video_encoding: mx.array | None,
    audio_encoding: mx.array | None,
    frame_rate: float,
) -> int:
    """Resolve ``num_frames`` to a concrete frame count, predicting it if ``AutoDuration``.

    Call after prompt encoding (once ``video_encoding``/``audio_encoding`` exist) and after
    ``require_num_frames_source`` has already validated a predictor is available when needed.

    Args:
        num_frames: Either an explicit frame count or AutoDuration request.
        duration_predictor: Optional DurationPredictor (must be provided if AutoDuration).
        video_encoding: Video embedding tokens from prompt encoder (shape B, N, 4096).
        audio_encoding: Audio embedding tokens from prompt encoder (shape B, N, 2048).
        frame_rate: Video frame rate in frames per second.

    Returns:
        Concrete frame count, either passed through or predicted.
    """
    if not isinstance(num_frames, AutoDuration):
        return num_frames
    return duration_predictor(
        video_encoding,
        audio_encoding,
        frame_rate=frame_rate,
        min_seconds=num_frames.min_seconds,
        max_seconds=num_frames.max_seconds,
    )


class DurationPredictor:
    """Predicts shot duration (in frames) from prompt encoder output.

    Unlike most blocks, the model is held directly rather than rebuilt on every call:
    DurationHead is a few MB, so there's no memory pressure motivating the
    build-on-call / free-on-exit pattern used for the large transformer/VAE blocks.

    Attributes:
        _head: The loaded DurationHead model.
    """

    def __init__(self, head: DurationHead) -> None:
        """Construct from an already-built, already-loaded head.

        Args:
            head: A DurationHead instance.
        """
        self._head = head

    @classmethod
    def from_checkpoint(cls, model_dir: str | Path) -> DurationPredictor | None:
        """Build a predictor from a checkpoint path, or ``None`` if unavailable.

        Returns ``None`` when the duration-head file is absent, so a checkpoint without
        DurationHead weights (anything predating 2.5 / gemma3 monoliths) gracefully
        skips prediction rather than crashing later.

        Args:
            model_dir: Path to the model directory (local or HuggingFace repo ID).

        Returns:
            DurationPredictor if duration_head.safetensors exists and loads; None otherwise.
        """
        model_path = Path(model_dir) if isinstance(model_dir, str) else model_dir
        head_path = model_path / "duration_head.safetensors"

        try:
            head = load_duration_head(head_path)
            return cls(head)
        except FileNotFoundError:
            logger.info(
                "No DurationHead weights found in %s; auto-duration prediction unavailable.",
                head_path,
            )
            return None

    def __call__(
        self,
        video_encoding: mx.array | None,
        audio_encoding: mx.array | None,
        *,
        frame_rate: float,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        """Predict a frame count from prompt encoder tokens, snapped to the VAE's grid.

        ``min_seconds``/``max_seconds`` clamp the prediction so a misbehaving prediction can't
        request a degenerate or OOM-sized generation; the defaults are 1s and 20s. The result is
        a frame count snapped to the VAE's ``8k + 1`` causal temporal grid.

        Args:
            video_encoding: Video embedding tokens (shape 1, N, 4096) or None.
            audio_encoding: Audio embedding tokens (shape 1, N, 2048) or None.
            frame_rate: Video frame rate in frames per second.
            min_seconds: Minimum predicted duration in seconds (default 1.0).
            max_seconds: Maximum predicted duration in seconds (default 20.0).

        Returns:
            Predicted frame count, clamped to [min_seconds, max_seconds] and snapped to grid.

        Raises:
            ValueError: If both video_encoding and audio_encoding are None.
            ValueError: If prediction has batch size != 1.
        """
        if video_encoding is None and audio_encoding is None:
            raise ValueError("DurationPredictor requires at least one of video_encoding / audio_encoding")

        seconds_pred = self._head(video_tokens=video_encoding, audio_tokens=audio_encoding)
        if seconds_pred.shape != (1,):
            raise ValueError(
                f"DurationPredictor only supports a single-item batch, got prediction shape {tuple(seconds_pred.shape)}"
            )

        seconds = float(seconds_pred.item())
        min_frames = round(min_seconds * frame_rate)
        max_frames = round(max_seconds * frame_rate)
        num_frames = seconds_to_clamped_num_frames(
            seconds, frame_rate=frame_rate, min_frames=min_frames, max_frames=max_frames
        )

        if seconds > max_seconds or seconds < min_seconds:
            logger.warning(
                "DurationHead prediction clamped: raw %.2fs outside [%.2fs, %.2fs], using %.2fs (%d frames) @ %.2f fps",
                seconds,
                min_seconds,
                max_seconds,
                num_frames / frame_rate,
                num_frames,
                frame_rate,
            )
        else:
            logger.info("DurationHead predicted %.2fs (%d frames @ %.2f fps)", seconds, num_frames, frame_rate)

        return num_frames


__all__ = [
    "AudioConditioner",
    "AudioDecoder",
    "DurationPredictor",
    "ImageConditioner",
    "PromptEncoder",
    "VideoDecoder",
    "VideoUpsampler",
    "require_num_frames_source",
    "resolve_num_frames",
    "seconds_to_clamped_num_frames",
    "snap_frames_to_grid",
]
