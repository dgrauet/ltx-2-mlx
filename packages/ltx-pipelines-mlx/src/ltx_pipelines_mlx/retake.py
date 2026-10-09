"""Retake / Extend pipeline — edit an existing video.

Matches the reference architecture: single stage at the source resolution.
Default: dev model + CFG guidance, 30 steps, cfg=3.0, full multi-modal
guidance. ``RetakePipeline(distilled=True)`` (``retake --distilled``) runs
upstream's default mode instead: the distilled transformer on the fixed
8-step distilled sigma table, one forward per step, no negative prompt
(upstream ``RetakePipeline(distilled=True)``: ``DISTILLED_SIGMAS`` +
``SimpleDenoiser``, deterministic Euler).

The class hosts both:

- ``retake`` / ``retake_from_video`` — regenerate a temporal region of
  an existing video (matches upstream ``ltx_pipelines.retake``).
- ``extend`` / ``extend_from_video`` — append or prepend frames to an
  existing video. No upstream equivalent; folded into this file rather
  than a separate ``ExtendPipeline`` class so the file structure stays
  isomorphic with upstream's ``retake.py``.

On the default dev path both modes share the same dev transformer, text
encoding (positive + negative for CFG), and ``guided_denoise_loop``
invocation — they only differ in how the ``denoise_mask`` is constructed
and whether new tokens are appended to the source latent. ``retake`` also
has a distilled mode (``distilled=True``, upstream's default); the
distilled ``extend`` lives in ``ExtendDistilledPipeline``
(``extend_distilled.py``), and this class's ``extend`` refuses it.

Ported from ltx-pipelines/src/ltx_pipelines/retake.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.mask_utils import first_frame_keyframes_mask
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    TemporalRegionMask,
    noise_latent_state,
)
from ltx_core_mlx.model.audio_vae import encode_audio
from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
from ltx_core_mlx.model.video_vae.tiling import TilingConfig, prepare_tiles_for_encoding
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.image import load_video_frames
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx._base import BasePipeline, reject_negative_prompt
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, ltx2_schedule, shorten_schedule
from ltx_pipelines_mlx.utils.progress import phase
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput, denoise_loop, guided_denoise_loop

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern

# Reference defaults (LTX_2_3_PARAMS)
DEFAULT_CFG_SCALE = 3.0
DEFAULT_STG_SCALE = 1.0  # upstream LTX_2_3_PARAMS default


def encode_video_tensor(encoder, video_tensor: mx.array) -> mx.array:
    """VAE-encode a source clip, materialized while ``encoder`` is loaded.

    Upstream encodes the source with ``tiled_encode(frames, TileSizeConfig.default())``
    (768/64 px, 80/24 frames). A source that fits one tile is encoded untiled, which is
    the same computation, so small sources keep their exact latents; a larger one is
    tiled like upstream (704x1280x49: 23.7 GB Metal peak instead of 31.4 GB untiled).
    The latent is evaluated here: ``mx.synchronize`` does not evaluate a lazy graph, so
    a lazy encode would run inside the first denoising step, with the encoder weights,
    the source pixels and the DiT all resident at once.

    Args:
        encoder: The loaded video VAE encoder.
        video_tensor: ``(1, 3, F, H, W)`` pixels in ``[-1, 1]``.

    Returns:
        ``(1, 128, F', H', W')`` bf16 latent.
    """
    tiling = TilingConfig.default()
    if len(prepare_tiles_for_encoding(video_tensor.shape, tiling)) > 1:
        # fp32 blend buffers; back to the encoder's bf16 like upstream's ``.to(dtype)``.
        latent = encoder.tiled_encode(video_tensor, tiling).astype(mx.bfloat16)
    else:
        latent = encoder.encode(video_tensor)
    _materialize(latent)
    return latent


def encode_source_audio_latent(
    pipe: BasePipeline,
    video_path: str,
    *,
    num_frames: int,
    frame_rate: float,
    has_audio: bool,
) -> mx.array:
    """Encode a source clip's audio track (materialized, encoder freed under ``low_memory``).

    A clip without audio, or whose audio cannot be read, gets a zero latent sized to the clip.

    Returns:
        ``(1, 8, T, 16)`` bf16 audio latent.
    """
    audio_latent: mx.array | None = None
    if has_audio:
        audio_data = load_audio(
            video_path,
            target_sample_rate=16000,
            max_duration=num_frames / frame_rate,
        )
        if audio_data is not None:

            def _encode_audio(enc, proc) -> mx.array:
                latent = encode_audio(audio_data.waveform, audio_data.sample_rate, enc, proc)
                _materialize(latent)
                return latent

            audio_latent = pipe.audio_conditioner(_encode_audio, free_after=pipe.low_memory)
            if pipe.low_memory:
                aggressive_cleanup()
        elif pipe.low_memory:
            pipe.audio_conditioner.free()
    elif pipe.low_memory:
        pipe.audio_conditioner.free()

    if audio_latent is None:
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_latent = mx.zeros((1, 8, audio_T, 16), dtype=mx.bfloat16)
    return audio_latent


@dataclass(frozen=True)
class _SourceMeta:
    """Source video metadata kept after the encoder is freed."""

    height: int
    width: int
    frame_rate: float
    num_frames: int


class RetakePipeline(BasePipeline):
    """Retake pipeline: regenerate a time segment while preserving the rest.

    Uses the dev (non-distilled) model with CFG guidance by default, or the
    distilled model without guidance when ``distilled=True``. Single-stage
    pipeline (no upsampler).

    Args:
        model_dir: Path to model weights.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        low_ram_streaming: Stream transformer blocks from mmap'd safetensors
            (``--low-ram``); mirrors upstream RetakePipeline's ``offload_mode``.
        dev_transformer: Dev transformer filename.
        distilled: Retake with the distilled transformer (``transformer.safetensors``
            or ``transformer-distilled*.safetensors``, resolved as
            :meth:`DistilledPipeline.load` does) on the distilled sigma table,
            one forward per step, no negative prompt and no CFG / STG. Upstream
            ``RetakePipeline``'s ``distilled`` flag, which defaults to ``True``
            there; here it defaults to ``False`` so the dev path stays the
            default. :meth:`extend` raises on a distilled pipeline: the distilled extend is
            :class:`~ltx_pipelines_mlx.extend_distilled.ExtendDistilledPipeline`.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        dev_transformer: str = "transformer-dev.safetensors",
        distilled: bool = False,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
        )
        self._dev_transformer = dev_transformer
        self.distilled = distilled
        # Source frame rate, recorded by _encode_source_video so CLI-level
        # decode helpers can forward it to _decode_and_save_video without
        # re-probing the file.
        self.source_frame_rate: float | None = None

    def _encode_source_video(
        self,
        video_path: str | Path,
    ) -> tuple[mx.array, mx.array, _SourceMeta]:
        """Probe the source video, encode video + audio latents via blocks, free.

        Composes :class:`ImageConditioner` (video VAE encoder) and
        :class:`AudioConditioner` (audio VAE + processor) blocks rather
        than reaching for inherited ``self._load_*`` methods.
        """
        video_path = str(video_path)
        info = probe_video_info(video_path)
        # Round down to VAE-compatible frame count (1 + 8k).
        k = max(1, (info.num_frames - 1) // 8)
        vae_compatible_frames = 1 + k * 8

        # Encode video via ImageConditioner (loads → encodes → frees)
        video_tensor = load_video_frames(video_path, info.height, info.width, vae_compatible_frames)
        video_latent = self.image_conditioner(
            lambda encoder: encode_video_tensor(encoder, video_tensor), free_after=self.low_memory
        )
        if self.low_memory:
            del video_tensor
            aggressive_cleanup()

        audio_latent = encode_source_audio_latent(
            self, video_path, num_frames=vae_compatible_frames, frame_rate=info.fps, has_audio=info.has_audio
        )

        meta = _SourceMeta(
            height=info.height,
            width=info.width,
            frame_rate=info.fps,
            num_frames=vae_compatible_frames,
        )
        self.source_frame_rate = meta.frame_rate
        return video_latent, audio_latent, meta

    def retake_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
        regenerate_audio: bool = True,
        negative_prompt: str | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video file.

        Args:
            prompt: Text prompt for the regenerated segment.
            video_path: Path to the source video file.
            start_frame: First latent frame to regenerate (inclusive).
            end_frame: Last latent frame to regenerate (exclusive).
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30). With ``distilled``,
                the step count on the 8-step distilled table (see :meth:`retake`).
            cfg_scale: CFG guidance scale (default: 3.0). Unused with ``distilled``.
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).
                Unused with ``distilled``.
            regenerate_audio: If True, regenerate audio in the retake region.
                If False, preserve original audio entirely.
            negative_prompt: Negative prompt for CFG. ``None`` (default) uses
                ``DEFAULT_NEGATIVE_PROMPT``; any string (including ``""``) is
                encoded verbatim.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        video_latent, audio_latent, meta = self._encode_source_video(video_path)
        return self.retake(
            prompt=prompt,
            source_video_latent=video_latent,
            source_audio_latent=audio_latent,
            start_frame=start_frame,
            end_frame=end_frame,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
            frame_rate=meta.frame_rate,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            regenerate_audio=regenerate_audio,
            negative_prompt=negative_prompt,
        )

    def extend_from_video(
        self,
        prompt: str,
        video_path: str | Path,
        extend_frames: int,
        direction: str = "after",
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
        negative_prompt: str | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Append or prepend ``extend_frames`` latent frames to a source video.

        Mirrors :meth:`retake_from_video`'s shape but instead of regenerating
        an interior region, grows the latent (and audio) on one side.

        Args:
            prompt: Text prompt describing the new frames.
            video_path: Path to the source video file.
            extend_frames: Number of latent frames to add.
            direction: ``"before"`` or ``"after"``.
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).
            negative_prompt: Negative prompt for CFG. ``None`` (default) uses
                ``DEFAULT_NEGATIVE_PROMPT``; any string (including ``""``) is
                encoded verbatim.

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).
        """
        video_latent, audio_latent, meta = self._encode_source_video(video_path)
        return self.extend(
            prompt=prompt,
            source_video_latent=video_latent,
            source_audio_latent=audio_latent,
            extend_frames=extend_frames,
            direction=direction,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
            frame_rate=meta.frame_rate,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            negative_prompt=negative_prompt,
        )

    def retake(
        self,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        start_frame: int,
        end_frame: int,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
        regenerate_audio: bool = True,
        negative_prompt: str | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Regenerate a time segment of a video.

        Args:
            prompt: Text prompt for the regenerated segment.
            source_video_latent: (B, C, F, H, W) source video latent.
            source_audio_latent: (B, 8, T, 16) source audio latent.
            start_frame: First latent frame to regenerate (inclusive).
            end_frame: Last latent frame to regenerate (exclusive).
            height: Video height.
            width: Video width.
            num_frames: Total number of pixel frames.
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30). With ``distilled``,
                the step count on the 8-step distilled table: sigma 1.0 and then
                the last ``num_steps`` sigmas (``shorten_schedule(..., keep="start")``,
                as ``generate --distilled --stage1-steps``); 8 or more runs the
                whole table.
            cfg_scale: CFG guidance scale (default: 3.0). Unused with ``distilled``.
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).
                Unused with ``distilled``.
            regenerate_audio: If True, regenerate audio in the retake region.
            negative_prompt: Negative prompt for CFG. ``None`` (default) uses
                ``DEFAULT_NEGATIVE_PROMPT``; any string (including ``""``) is
                encoded verbatim. Must stay ``None`` with ``distilled`` (no CFG).

        Returns:
            Tuple of (video_latent, audio_latent).

        Raises:
            ValueError: when ``negative_prompt`` is set on a distilled pipeline.
        """
        if self.distilled:
            reject_negative_prompt(negative_prompt, type(self).__name__)
            # --- Text encoding (positive only, no CFG) + distilled transformer ---
            video_embeds, audio_embeds = self._encode_positive_prompt(prompt)
            neg_video_embeds = neg_audio_embeds = None
            if self.dit is None:
                self.dit = self._load_distilled_transformer()
        else:
            # --- Text encoding (positive + negative for CFG) ---
            video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(
                prompt, negative_prompt
            )

            # --- Load dev transformer ---
            if self.dit is None:
                self.dit = self._load_dev_transformer()
        assert self.dit is not None

        F, H, W = compute_video_latent_shape(num_frames, height, width)
        tokens_per_frame = H * W

        # Patchify source
        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)

        # Compute positions
        video_positions = compute_video_positions(F, H, W, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        # Create video state with temporal mask (1 = regenerate, 0 = preserve).
        # Upstream ``retake.py:267-274`` also takes a ``regenerate_video`` flag
        # and sets ``frozen=not regenerate_video``; our port always regenerates
        # the video region (no ``regenerate_video`` parameter), so video is
        # never a whole-stream-frozen conditioning here.
        region = TemporalRegionMask(start_frame, end_frame)
        denoise_mask = region.create_mask(F, tokens_per_frame)

        video_state = LatentState(
            latent=source_tokens,
            clean_latent=source_tokens,
            denoise_mask=denoise_mask,
            positions=video_positions,
            keyframes_mask=first_frame_keyframes_mask(denoise_mask, tokens_per_frame),
        )
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)

        # Audio state
        if regenerate_audio:
            # Apply same temporal mask to audio
            audio_tokens_per_video_frame = audio_T / F
            audio_start = round(start_frame * audio_tokens_per_video_frame)
            audio_end = round(end_frame * audio_tokens_per_video_frame)

            audio_mask = mx.zeros((1, audio_T, 1), dtype=mx.bfloat16)
            audio_mask = audio_mask.at[:, audio_start:audio_end, :].add(
                mx.ones((1, audio_end - audio_start, 1), dtype=mx.bfloat16)
            )
        else:
            # Preserve all audio (mask=0)
            audio_mask = mx.zeros((1, audio_T, 1), dtype=mx.bfloat16)

        # Upstream ``retake.py:275-281``: ``frozen=initial_audio_latent is not
        # None and not regenerate_audio``. ``source_audio_latent`` is always
        # provided here (zeros when the source has no audio track), so the
        # condition collapses to ``not regenerate_audio``.
        audio_state = LatentState(
            latent=audio_tokens,
            clean_latent=audio_tokens,
            denoise_mask=audio_mask,
            positions=audio_positions,
            frozen=not regenerate_audio,
        )
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        if self.distilled:
            output = self._distilled_denoise(video_state, audio_state, video_embeds, audio_embeds, (F, H, W), num_steps)
        else:
            output = self._guided_denoise(
                video_state,
                audio_state,
                video_embeds,
                audio_embeds,
                neg_video_embeds,
                neg_audio_embeds,
                (F, H, W),
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
            )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent

    def _encode_positive_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Load the text encoder, encode the prompt alone, materialize, free the encoder.

        The positive half of :meth:`_encode_text_with_negative`, for the distilled
        mode (upstream encodes ``[prompt]`` only when ``distilled``).
        """
        self._load_text_encoder()
        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(prompt)
            _materialize(video_embeds, audio_embeds)
        self.text_encoder = None
        self.feature_extractor = None
        aggressive_cleanup()
        return video_embeds, audio_embeds

    def _load_distilled_transformer(self) -> LTXModel:
        """Load the distilled transformer, resolved as :meth:`DistilledPipeline.load` does."""
        return self._load_transformer_with_optional_streaming(self._resolve_distilled_transformer(self.model_dir))

    def _guided_denoise(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        neg_video_embeds: mx.array,
        neg_audio_embeds: mx.array,
        latent_dims: tuple[int, int, int],
        *,
        num_steps: int,
        cfg_scale: float,
        stg_scale: float,
    ) -> DenoiseOutput:
        """Dev model + CFG / STG: the dynamic schedule and the LTX_2_3_PARAMS guiders."""
        F, H, W = latent_dims
        num_tokens = F * H * W
        sigmas = ltx2_schedule(num_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        # Build guidance (ref LTX_2_3_PARAMS)
        video_gp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        audio_gp = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )

        video_factory = create_multimodal_guider_factory(video_gp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_gp, negative_context=neg_audio_embeds)

        self._pre_denoise_flush(video_state, audio_state)
        return guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas,
            on_step=self._stepwise_hook(F, H, W),
        )

    def _distilled_denoise(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        latent_dims: tuple[int, int, int],
        num_steps: int,
    ) -> DenoiseOutput:
        """Distilled model: fixed distilled sigmas, one forward per step, deterministic Euler.

        Upstream's distilled retake (``DISTILLED_SIGMAS`` + ``SimpleDenoiser`` on the
        default Euler loop). Unlike ``generate --distilled``, upstream does not switch
        retake to the ancestral sampler on LTX-2.5 checkpoints, and neither does this.
        """
        F, H, W = latent_dims
        sigmas = shorten_schedule(DISTILLED_SIGMAS, num_steps, keep="start")
        self._pre_denoise_flush(video_state, audio_state)
        return denoise_loop(
            model=X0Model(self.dit),
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
            on_step=self._stepwise_hook(F, H, W),
        )

    def extend(
        self,
        prompt: str,
        source_video_latent: mx.array,
        source_audio_latent: mx.array,
        extend_frames: int,
        direction: str = "after",
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = DEFAULT_STG_SCALE,
        negative_prompt: str | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Extend a video by adding ``extend_frames`` latent frames.

        Same dev model + CFG flow as :meth:`retake`; the difference lives
        only in how the denoise mask + clean latent are constructed (new
        tokens are appended/prepended rather than regenerated in place).

        Args:
            prompt: Text prompt for the new frames.
            source_video_latent: ``(B, C, F, H, W)`` source video latent.
            source_audio_latent: ``(B, 8, T, 16)`` source audio latent.
            extend_frames: Number of latent frames to add.
            direction: ``"before"`` or ``"after"``.
            height: Source video height.
            width: Source video width.
            num_frames: Total source pixel-frame count.
            frame_rate: Source frame rate (used to size the audio extension).
            seed: Random seed.
            num_steps: Number of denoising steps (default: 30).
            cfg_scale: CFG guidance scale (default: 3.0).
            stg_scale: STG guidance scale (default: 1.0, upstream LTX_2_3_PARAMS).
            negative_prompt: Negative prompt for CFG. ``None`` (default) uses
                ``DEFAULT_NEGATIVE_PROMPT``; any string (including ``""``) is
                encoded verbatim.

        Returns:
            Tuple of (extended_video_latent, extended_audio_latent).

        Raises:
            NotImplementedError: on a ``distilled`` pipeline (``extend`` runs the dev model only).
        """
        if self.distilled:
            raise NotImplementedError(
                "RetakePipeline.extend has no distilled mode; use extend_distilled.ExtendDistilledPipeline "
                "(extend --distilled)."
            )
        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(
            prompt, negative_prompt
        )

        if self.dit is None:
            self.dit = self._load_dev_transformer()
        assert self.dit is not None

        B = source_video_latent.shape[0]
        F_source = source_video_latent.shape[2]
        _, H, W = compute_video_latent_shape(1, height, width)
        F_total = F_source + extend_frames
        tokens_per_frame = H * W

        source_tokens, _ = self.video_patchifier.patchify(source_video_latent)
        audio_tokens, audio_T = self.audio_patchifier.patchify(source_audio_latent)

        # Compute total pixel frames for audio token count (each latent frame ≈ 8 pixel frames).
        total_pixel_frames = num_frames + extend_frames * 8
        extend_audio_T = max(0, compute_audio_token_count(total_pixel_frames, frame_rate=frame_rate) - audio_T)
        audio_total_T = audio_T + extend_audio_T

        new_video_shape = (B, extend_frames * tokens_per_frame, 128)

        if direction == "after":
            video_denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, source_tokens.shape[1], 1), dtype=mx.bfloat16),
                    mx.ones((B, new_video_shape[1], 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_video = mx.concatenate(
                [source_tokens, mx.zeros(new_video_shape, dtype=mx.bfloat16)],
                axis=1,
            )
            audio_denoise_mask = mx.concatenate(
                [
                    mx.zeros((B, audio_T, 1), dtype=mx.bfloat16),
                    mx.ones((B, extend_audio_T, 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_audio = mx.concatenate(
                [audio_tokens, mx.zeros((B, extend_audio_T, 128), dtype=mx.bfloat16)],
                axis=1,
            )
        else:  # before
            video_denoise_mask = mx.concatenate(
                [
                    mx.ones((B, new_video_shape[1], 1), dtype=mx.bfloat16),
                    mx.zeros((B, source_tokens.shape[1], 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_video = mx.concatenate(
                [mx.zeros(new_video_shape, dtype=mx.bfloat16), source_tokens],
                axis=1,
            )
            audio_denoise_mask = mx.concatenate(
                [
                    mx.ones((B, extend_audio_T, 1), dtype=mx.bfloat16),
                    mx.zeros((B, audio_T, 1), dtype=mx.bfloat16),
                ],
                axis=1,
            )
            clean_audio = mx.concatenate(
                [mx.zeros((B, extend_audio_T, 128), dtype=mx.bfloat16), audio_tokens],
                axis=1,
            )

        video_positions = compute_video_positions(F_total, H, W, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_total_T)

        # No ``frozen`` marking here (nor for audio below): upstream's
        # ``LatentState.frozen`` covers a *whole stream* that is pure
        # conditioning; ``extend`` always regenerates a new segment on both
        # modalities, so neither stream is ever entirely preserved.
        video_state = LatentState(
            latent=clean_video,
            clean_latent=clean_video,
            denoise_mask=video_denoise_mask,
            positions=video_positions,
            keyframes_mask=first_frame_keyframes_mask(video_denoise_mask, H * W),
        )
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)

        audio_state = LatentState(
            latent=clean_audio,
            clean_latent=clean_audio,
            denoise_mask=audio_denoise_mask,
            positions=audio_positions,
        )
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        num_tokens = F_total * H * W
        sigmas = ltx2_schedule(num_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        video_gp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        audio_gp = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )

        video_factory = create_multimodal_guider_factory(video_gp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_gp, negative_context=neg_audio_embeds)

        self._pre_denoise_flush(video_state, audio_state)
        output = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas,
            on_step=self._stepwise_hook(F_total, H, W),
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F_total, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent
