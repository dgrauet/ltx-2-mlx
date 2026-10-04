"""SDR -> HDR IC-LoRA in one denoise stage (upstream ``ltx_pipelines/hdr_ic_lora.py``, v1.4).

The model works in ACEScct; the output is a BT.2020/HLG 10-bit master plus an EXR sequence.
LTX-2.5 packs only (the LoRA ``Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR`` and DFR seam keyframes
need ``use_keyframes_abs_pos_embedding`` and the keyframe-trained diffusion video decoder).

No prompt, Gemma, audio stream or upsampler: the scene embeddings come precomputed from a
``.safetensors`` file (``video_context``), the DiT runs video-only, and the IC-LoRA reference is the
VAE-encoded SDR clip at full resolution (downscale 1).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.keyframe_slots import extract_generated_keyframes
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import DecodeKeyframes
from ltx_core_mlx.model.video_vae.tiling import TilingConfig
from ltx_core_mlx.utils.ffmpeg import find_ffmpeg, find_ffprobe, probe_video_info
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_video_positions

from ._base import BasePipeline
from .dfr import decode_keyframes_from_slots
from .dfr_layout import resolve_canvas
from .iclora_utils import reference_conditioning_from_latent
from .scheduler import DISTILLED_SIGMAS
from .utils import hdr_media
from .utils._orchestration import resolve_model_dir
from .utils.blocks import ImageConditioner, VideoDecoder
from .utils.generation import is_ltx25_pack
from .utils.hdr_media import (
    EXRColorSpace,
    EXRVideoInput,
    VideoInput,
    align_resolution,
    encode_hdr_outputs,
    load_exr_as_hdr_conditioning,
    load_video_as_hdr_conditioning,
    read_exr,
)
from .utils.helpers import create_noised_state, generated_keyframe_conditionings
from .utils.progress import phase
from .utils.samplers import denoise_loop

logger = logging.getLogger(__name__)

HDR_LORA_REPO = "Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR"
"""Gated HuggingFace repo holding the SDR-to-HDR IC-LoRA and its scene embeddings."""
HDR_LORA_FILENAME = "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors"
HDR_SCENE_EMBEDDINGS_FILENAME = "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors"

_materialize = getattr(mx, "eval")  # noqa: B009 -- mx.eval is the MLX graph materialiser

MIN_RESOLUTION = 32
ALIGNMENT_DIVISOR = 32

#: Spatial area (H x W) above which the conditioning encode uses ``tiled_encode`` (upstream
#: ``TILED_VAE_ENCODE_PIXEL_THRESHOLD``, an empirical VRAM gate, not derived from the VAE).
TILED_VAE_ENCODE_PIXEL_THRESHOLD = 512 * 768

#: Full distilled schedule (8 Euler steps, 9 sigma values including 1.0 and 0.0); upstream
#: ``DISTILLED_SIGMA_VALUES``, value-identical to the LTX-2.5 distilled table.
DEFAULT_DENOISE_SIGMAS = list(DISTILLED_SIGMAS)
#: Same pin as ``dfr.ANCHOR_KEYFRAME_STRENGTH`` (upstream ``dfr_pipeline._ANCHOR_KEYFRAME_STRENGTH``).
DEFAULT_KEYFRAME_STRENGTH = 0.95
_SNAP_CONDITIONING_FPS_ABOVE = 30.0
#: This pipeline snaps high-fps sources to 30, not DFR's 60 (upstream ``_HIGH_FPS_CONDITIONING_FPS``).
_HIGH_FPS_CONDITIONING_FPS = 30.0

#: The SDR-to-HDR IC-LoRA is applied at a fixed strength (upstream ``LoraPathStrengthAndSDOps(..., 1.0)``).
_HDR_LORA_STRENGTH = 1.0


def conditioning_fps(playback_fps: float) -> float:
    """RoPE fps for this pipeline (upstream ``_conditioning_fps``): above 30 snaps to 30."""
    if playback_fps > _SNAP_CONDITIONING_FPS_ABOVE:
        return _HIGH_FPS_CONDITIONING_FPS
    return playback_fps


def _dfr_seam_pixel_positions(num_frames: int, *, high_quality_hdr: bool) -> list[int]:
    """DFR x8-border segment seams from :func:`resolve_canvas`, clipped to the real clip.

    A 1-frame source is a valid 8k+1 clip with no interior seams, so this returns ``[]`` rather
    than calling :func:`resolve_canvas`, which rejects it (upstream ``_dfr_seam_pixel_positions``).
    """
    if num_frames < 2:
        return []
    _canvas, _segment, positions = resolve_canvas(num_frames)
    positions = [int(p) for p in positions if int(p) < num_frames]
    if high_quality_hdr:
        positions = [2 * p for p in positions]
    return positions


def dfr_seam_roles(num_frames: int, *, high_quality_hdr: bool) -> tuple[list[int], list[int]]:
    """Generated HDR slot indices and SDR guide indices for an HDR DFR run.

    Every seam takes **both** roles (upstream ``_seam_roles``): a generated HDR slot and, at the
    same position, a 1-frame SDR guide.
    """
    positions = _dfr_seam_pixel_positions(num_frames, high_quality_hdr=high_quality_hdr)
    return list(positions), list(positions)


def load_video_context(path: str | Path) -> mx.array:
    """Load ``video_context`` (or trainer ``video_prompt_embeds``) from ``.safetensors`` (upstream ``_load_video_context``).

    Returns:
        The context as ``(1, N, D)`` (an unbatched ``(N, D)`` tensor gets a batch axis).

    Raises:
        FileNotFoundError: the file does not exist.
        KeyError: neither key is in the file.
    """
    emb_path = Path(path)
    if not emb_path.is_file():
        raise FileNotFoundError(
            f"Text embeddings not found: {emb_path} (expected a local {HDR_SCENE_EMBEDDINGS_FILENAME}). "
            f"{_download_hint()}"
        )
    tensors = mx.load(str(emb_path))
    assert isinstance(tensors, dict)
    for name in ("video_context", "video_prompt_embeds"):
        if name in tensors:
            context = tensors[name]
            # The official scene-emb file stores ``(N, 4096)`` with no batch axis; the DiT's
            # cross-attention takes ``(B, N, D)``.
            return context[None] if context.ndim == 2 else context
    raise KeyError(f"video_context/video_prompt_embeds not found in {emb_path} (keys={sorted(tensors)})")


def require_hdr_export_tools() -> None:
    """Fail fast when the HDR export cannot run (it only starts after the whole denoise).

    Raises:
        ImportError: OpenEXR (the optional ``hdr`` extra) is not installed.
        RuntimeError: the local ffmpeg has no ``libx265`` encoder (the HLG master is HEVC).
    """
    hdr_media._openexr()
    result = subprocess.run([find_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not any(
        len(fields) > 1 and fields[1] == "libx265" for fields in (line.split() for line in result.stdout.splitlines())
    ):
        raise RuntimeError(
            "hdr-ic-lora writes a 10-bit HEVC (HLG) master but this ffmpeg has no libx265 encoder "
            f"({find_ffmpeg()}); install an ffmpeg built with libx265 (e.g. brew install ffmpeg)."
        )


def _download_hint() -> str:
    return (
        f"Fetch both files of the gated repo {HDR_LORA_REPO} once (accept its licence on HuggingFace first): "
        f"huggingface-cli download {HDR_LORA_REPO} {HDR_LORA_FILENAME} {HDR_SCENE_EMBEDDINGS_FILENAME} "
        f"--local-dir hdr-lora, then pass --hdr-lora hdr-lora/{HDR_LORA_FILENAME} "
        f"--text-embeddings hdr-lora/{HDR_SCENE_EMBEDDINGS_FILENAME}."
    )


def _resolve_hdr_lora(hdr_lora: str) -> str:
    """Accept only an existing local ``.safetensors`` file; never touch the network.

    A HuggingFace repo id is refused up front: the SDR-to-HDR repo holds two ``.safetensors`` (the
    LoRA and the scene embeddings), so a snapshot would download both and still be ambiguous.

    Raises:
        ValueError: ``hdr_lora`` is not a ``.safetensors`` path (e.g. a repo id).
        FileNotFoundError: ``hdr_lora`` names a ``.safetensors`` file that does not exist.
    """
    if not hdr_lora.endswith(".safetensors"):
        raise ValueError(
            f"--hdr-lora takes a local .safetensors file ({HDR_LORA_FILENAME}), not {hdr_lora!r}. {_download_hint()}"
        )
    if not Path(hdr_lora).is_file():
        raise FileNotFoundError(
            f"HDR IC-LoRA file not found: {hdr_lora} (expected {HDR_LORA_FILENAME}). {_download_hint()}"
        )
    return hdr_lora


def _require_ltx25_pack(model_dir: str) -> None:
    """Refuse a non-2.5 pack before the (multi-GB) snapshot download.

    A local directory is checked in place; for a HuggingFace repo id only ``embedded_config.json``
    is fetched.

    Raises:
        ValueError: ``model_dir`` is not an LTX-2.5 pack.
    """
    local = Path(model_dir)
    config_dir = local if local.exists() else Path(hf_hub_download(model_dir, "embedded_config.json")).parent
    if not is_ltx25_pack(config_dir):
        raise ValueError("hdr-ic-lora needs an LTX-2.5 pack (the SDR-to-HDR IC-LoRA is 2.5-only)")


def _count_video_frames(path: Path) -> int:
    """Frame count of the first video stream (upstream ``get_videostream_metadata``).

    Uses the container's ``nb_frames`` when present; otherwise decodes and counts
    (``-count_frames``) instead of estimating from ``duration * fps``.
    """

    def _ffprobe(*extra: str, entry: str) -> str:
        result = subprocess.run(
            [find_ffprobe(), "-v", "error", "-select_streams", "v:0", *extra]
            + ["-show_entries", f"stream={entry}", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed on {path}: {result.stderr}")
        streams = json.loads(result.stdout).get("streams", [])
        return str(streams[0].get(entry, "")) if streams else ""

    declared = _ffprobe(entry="nb_frames")
    if declared.isdigit() and int(declared) > 0:
        return int(declared)
    counted = _ffprobe("-count_frames", entry="nb_read_frames")
    return int(counted) if counted.isdigit() else 0


class HDRICLoraPipeline(BasePipeline):
    """IC-LoRA from SDR to HDR in one denoise stage (upstream ``HDRICLoraPipeline``).

    The model works in ACEScct; export is an HLG MP4 plus an EXR sequence. ``keyframe_strength``
    (default :data:`DEFAULT_KEYFRAME_STRENGTH`, matching upstream's CLI default) puts a generated
    HDR slot and a 1-frame SDR guide on every DFR seam and decodes keyframe-aware; ``None`` is
    plain IC-LoRA.

    Args:
        model_dir: LTX-2.5 pack (local path or HuggingFace repo id).
        hdr_lora: Local SDR-to-HDR IC-LoRA ``.safetensors`` file (:data:`HDR_LORA_FILENAME`).
        text_embeddings: ``.safetensors`` holding the precomputed ``video_context``.
        low_ram_streaming: Stream transformer blocks (``--low-ram``); the LoRA is then fused per
            block bind instead of in place.

    Raises:
        ImportError: OpenEXR (the ``hdr`` extra) is missing.
        RuntimeError: ffmpeg has no ``libx265`` encoder.
        FileNotFoundError: the text embeddings or the LoRA ``.safetensors`` do not exist.
        KeyError: the embeddings file holds neither ``video_context`` nor ``video_prompt_embeds``.
        ValueError: ``hdr_lora`` is not a ``.safetensors`` path (e.g. a repo id), or ``model_dir`` is not
            an LTX-2.5 pack.

    All of these are checked before the model pack is downloaded or anything is loaded.
    """

    video_decoder = "diffusion"
    generate_audio = False

    def __init__(
        self,
        model_dir: str,
        hdr_lora: str,
        text_embeddings: str,
        *,
        low_ram_streaming: bool = False,
    ) -> None:
        # Cheap refusals first: the export tooling (otherwise only hit after the whole denoise), the
        # embeddings, the LoRA and the pack generation, all before the multi-GB pack snapshot.
        require_hdr_export_tools()
        logger.info("Loading text embeddings from %s", text_embeddings)
        video_context = load_video_context(text_embeddings)
        lora_path = _resolve_hdr_lora(hdr_lora)
        _require_ltx25_pack(model_dir)

        resolved = resolve_model_dir(model_dir)
        super().__init__(model_dir=str(resolved), low_memory=True, low_ram_streaming=low_ram_streaming)
        self._is_25 = True
        self.video_context = video_context
        # Upstream: ``loras=(LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)``; the
        # transformer loader fuses it (or attaches a ``BlockLoraSource`` under ``--low-ram``).
        self._pending_loras = [(lora_path, _HDR_LORA_STRENGTH)]

        # Upstream ``vae_dtype = torch.float32``: both VAE ends run in fp32, the DiT in bf16.
        self.image_conditioner = ImageConditioner(self.model_dir, dtype=mx.float32)
        self.video_decoder_block = VideoDecoder(
            self.model_dir, verbose=self.verbose, video_decoder="diffusion", dtype=mx.float32
        )

    def load(self) -> None:
        """Load the distilled DiT with the HDR IC-LoRA (no decoders: the decode loads its own)."""
        if self.dit is None:
            path = self._resolve_safetensors(self.model_dir, "transformer-distilled")
            self.dit = self._load_transformer_with_optional_streaming(path)
        self._loaded = True

    # ------------------------------------------------------------------

    @staticmethod
    def _probe(video: VideoInput | EXRVideoInput) -> tuple[int, int, int, float]:
        """``(frames, width, height, fps)`` of the source (upstream ``get_videostream_metadata``)."""
        if isinstance(video, EXRVideoInput):
            files = sorted(Path(video.dir).glob("*.exr"))
            if not files:
                raise RuntimeError(f"No EXR frames found in {video.dir}")
            height, width = read_exr(files[0]).shape[:2]
            return len(files), width, height, float(video.frame_rate)
        info = probe_video_info(str(video.path))
        return _count_video_frames(Path(video.path)), info.width, info.height, info.fps

    @staticmethod
    def _load_acescct_conditioning(
        video: VideoInput | EXRVideoInput, height: int, width: int, num_frames: int
    ) -> mx.array:
        """Load the IC-LoRA reference as ACEScct ``(1, 3, F, H, W)`` float32 in VAE range."""
        if isinstance(video, EXRVideoInput):
            frames = load_exr_as_hdr_conditioning(video.dir, height, width, num_frames, color_space=video.color_space)
            source: Path = Path(video.dir)
        else:
            frames = load_video_as_hdr_conditioning(
                video.path, height, width, num_frames, gamma_encoded=video.gamma_encoded
            )
            source = Path(video.path)
        stacked = list(frames)
        if len(stacked) != num_frames:
            raise ValueError(
                f"Loaded {len(stacked)} frames from {source} but the source was probed at {num_frames} frames "
                "(truncated or unreadable stream?)."
            )
        # (F, H, W, 3) -> (1, 3, F, H, W)
        return mx.array(np.stack(stacked, axis=0).astype(np.float32)).transpose(3, 0, 1, 2)[None]

    def _create_reference_conditionings(
        self,
        video_encoder,
        pixels: mx.array,
        *,
        conditioning_strength: float,
        gen_h: int,
        gen_w: int,
        guides_kf: Sequence[int],
        keyframe_strength: float | None,
        frame_rate: float,
    ) -> tuple[list, mx.array]:
        """VAE-encode the IC-LoRA reference, plus 1-frame SDR seam guides (upstream ``_create_reference_conditionings``).

        Returns:
            ``(conditionings, reference tokens (1, N, 128) bf16)``.
        """
        # Upstream picks the tiled path above the area gate with the pipeline's resolved
        # ``tiling_config``; the MLX conv encoder has no AUTO_TILING resolver, so it tiles with
        # ``TilingConfig.default()`` (upstream ``TileSizeConfig.default()``, 768/64 px, 80/24 frames).
        use_tiled = gen_h * gen_w > TILED_VAE_ENCODE_PIXEL_THRESHOLD

        def encode(x: mx.array) -> mx.array:
            # fp32 encode, then ``encoded.to(self.dtype)`` (bf16) like upstream.
            out = video_encoder.tiled_encode(x, TilingConfig.default()) if use_tiled else video_encoder.encode(x)
            return out.astype(mx.bfloat16)

        reference = encode(pixels)
        _materialize(reference)
        latent_f, latent_h, latent_w = reference.shape[2], reference.shape[3], reference.shape[4]
        conditionings: list = [
            reference_conditioning_from_latent(
                reference, frame_rate=frame_rate, downscale_factor=1, strength=conditioning_strength
            )
        ]
        ref_tokens, _ = self.video_patchifier.patchify(reference)
        if not guides_kf:
            return conditionings, ref_tokens
        if keyframe_strength is None:
            raise ValueError("SDR seam guides require keyframe_strength")
        # Upstream ``_keyframe_conditionings_from_pixel_frames``: each seam frame encoded as a true 1-frame latent.
        num_frames = pixels.shape[2]
        for frame_idx in guides_kf:
            idx = int(frame_idx)
            if idx < 0 or idx >= num_frames:
                raise ValueError(f"Seam frame_idx={idx} out of range for pixel video T={num_frames}")
            encoded = encode(pixels[:, :, idx : idx + 1])
            tokens, _ = self.video_patchifier.patchify(encoded)
            _materialize(tokens)
            conditionings.append(
                VideoConditionByKeyframeIndex(
                    frame_idx=idx,
                    keyframe_latent=tokens,
                    spatial_dims=(latent_f, latent_h, latent_w),
                    frame_rate=frame_rate,
                    strength=keyframe_strength,
                    num_pixel_frames=1,
                )
            )
        return conditionings, ref_tokens

    def generate(
        self,
        video: VideoInput | EXRVideoInput,
        *,
        seed: int,
        conditioning_strength: float = 1.0,
        high_quality_hdr: bool = False,
        keyframe_strength: float | None = DEFAULT_KEYFRAME_STRENGTH,
        denoise_sigmas: list[float] | None = None,
    ) -> tuple[Iterator[np.ndarray], float]:
        """Generate ACEScct HDR from an MP4/MOV or EXR-frame folder (upstream ``__call__``).

        Validation, conditioning encode and the denoise run eagerly; the decode is lazy and runs
        as the returned iterator is consumed.

        Args:
            video: :class:`VideoInput` (MP4/MOV) or :class:`EXRVideoInput` (EXR folder).
            seed: RNG seed (initial noise and diffusion-decoder noise).
            conditioning_strength: IC-LoRA reference strength.
            high_quality_hdr: Duplicate each conditioning frame, generate ``2N-1`` frames, then keep
                every other output frame (~2x cost, fewer temporal artifacts).
            keyframe_strength: ``None`` for plain IC-LoRA; a float puts a generated slot and a
                1-frame SDR guide on every ``resolve_canvas`` seam at that strength.
            denoise_sigmas: Override :data:`DEFAULT_DENOISE_SIGMAS`.

        Returns:
            ``(chunks, fps)``: an iterator of ACEScct ``[0, 1]`` ``(F, H, W, 3)`` float32 chunks
            (cropped to the source size, HQ-decimated) and the source frame rate.

        Raises:
            ValueError: frame count off the 8k+1 grid, resolution below 32 px after alignment, odd
                source width/height, or fewer frames loaded than probed.
        """
        num_frames, width, height, fps = self._probe(video)

        # Assert 8k+1 frame count (upstream ``video_scale.time``).
        if num_frames < 1 or (num_frames - 1) % 8 != 0:
            snapped = max(1, ((num_frames - 1) // 8) * 8 + 1)
            raise ValueError(
                f"Video frame count must satisfy 8k+1 (e.g. 97, 193). "
                f"Got {num_frames}; use a video with {snapped} frames."
            )

        # Pad W/H up to a multiple of 32, cropped back after decode.
        gen_w, gen_h, crop_w, crop_h = align_resolution(width, height, divisor=ALIGNMENT_DIVISOR)
        if gen_h < MIN_RESOLUTION or gen_w < MIN_RESOLUTION:
            raise ValueError(
                f"Resolution ({width}x{height}) too small after alignment "
                f"(got {gen_w}x{gen_h}, need >= {MIN_RESOLUTION})."
            )

        # The HLG master is 4:2:0: refuse odd source dimensions up front (upstream fails later, in
        # the encoder, after the whole denoise).
        if width % 2 or height % 2:
            raise ValueError(f"Source is {width}x{height}; the HLG 4:2:0 master needs even width and height.")

        # Generate 2N-1 frames for high-quality HDR, N frames otherwise.
        gen_frames = 2 * num_frames - 1 if high_quality_hdr else num_frames

        generated_kf: list[int] = []
        guides_kf: list[int] = []
        if keyframe_strength is not None:
            generated_kf, guides_kf = dfr_seam_roles(num_frames, high_quality_hdr=high_quality_hdr)
            self._require_generated_keyframes_support(generated_kf)
            if not generated_kf and not guides_kf:
                logger.warning(
                    "[HDR IC-LoRA] keyframe_strength=%.2f but this %d-frame clip has no DFR seams "
                    "after clipping the canvas; running plain IC-LoRA.",
                    keyframe_strength,
                    num_frames,
                )
            else:
                logger.info(
                    "[HDR IC-LoRA] DFR seams @ strength=%.2f -> generated %s | SDR kf %s",
                    keyframe_strength,
                    generated_kf,
                    guides_kf,
                )

        cfps = conditioning_fps(fps)
        sigmas = list(denoise_sigmas) if denoise_sigmas is not None else list(DEFAULT_DENOISE_SIGMAS)

        acescct_sdr = self._load_acescct_conditioning(video, gen_h, gen_w, num_frames)
        if high_quality_hdr:
            # BCTHW: duplicate each frame, trim to gen_frames (upstream ``repeat_interleave(2, dim=2)[:, :, :gen_frames]``).
            acescct_sdr = mx.repeat(acescct_sdr, 2, axis=2)[:, :, :gen_frames]

        with phase("Encoding the SDR reference (fp32 VAE)", verbose=self.verbose):
            result = self.image_conditioner(
                functools.partial(
                    self._create_reference_conditionings,
                    pixels=acescct_sdr,
                    conditioning_strength=conditioning_strength,
                    gen_h=gen_h,
                    gen_w=gen_w,
                    guides_kf=guides_kf,
                    keyframe_strength=keyframe_strength,
                    frame_rate=cfps,
                )
            )
        conditionings, ref_tokens = result  # type: ignore[misc]
        del acescct_sdr
        aggressive_cleanup()

        latent_f = (gen_frames - 1) // 8 + 1
        latent_h, latent_w = gen_h // 32, gen_w // 32
        # Upstream ``_run_diffusion_stage``: generated slots go last, after the reference and the guides.
        conditionings = [
            *conditionings,
            *generated_keyframe_conditionings(generated_kf, gen_frames, frame_rate=cfps),
        ]

        self.load()
        assert self.dit is not None
        # Upstream ``ModalitySpec(latent=source_conditioning.latent, noise_scale=sigmas[0])``: the
        # reference latent seeds the generation tokens, then is noised at ``sigmas[0]`` (pure noise at 1.0).
        video_state = create_noised_state(
            base_shape=ref_tokens.shape,
            conditionings=conditionings,
            spatial_dims=(latent_f, latent_h, latent_w),
            positions=compute_video_positions(latent_f, latent_h, latent_w, frame_rate=cfps),
            seed=seed,
            sigma=sigmas[0],
            initial_latent=ref_tokens,
        )
        # Deterministic Euler, video-only (upstream ``SimpleDenoiser(video_context, None)``, no audio modality).
        output = denoise_loop(
            X0Model(self.dit),
            video_state=video_state,
            audio_state=None,
            video_text_embeds=self.video_context.astype(mx.bfloat16),
            audio_text_embeds=None,
            sigmas=sigmas,
        )
        num_tokens = latent_f * latent_h * latent_w
        slots = extract_generated_keyframes(
            output.video_latent, video_state.generated_keyframe_layout, self.video_patchifier, (latent_h, latent_w)
        )
        latent = self.video_patchifier.unpatchify(
            output.video_latent[:, :num_tokens, :], (latent_f, latent_h, latent_w)
        )
        _materialize(latent, *([] if slots is None else [slots]))
        del output, video_state, conditionings, ref_tokens

        # The latents are materialized: free the DiT before the fp32 diffusion decode.
        self.dit = None
        self._loaded = False
        aggressive_cleanup()

        # Upstream decodes with ``dtype=vae_dtype`` (fp32): cast explicitly rather than rely on the sampler state.
        latent = latent.astype(mx.float32)
        decode_kf = decode_keyframes_from_slots(slots, generated_kf, gen_frames)
        if decode_kf is not None:
            # Upstream casts the keyframe planes to ``vae_dtype`` (fp32) before the decode.
            decode_kf = dataclasses.replace(decode_kf, latents=decode_kf.latents.astype(mx.float32))
        chunks = self._iter_decoded(
            latent,
            seed=seed,
            keyframes=decode_kf,
            crop_h=crop_h,
            crop_w=crop_w,
            high_quality_hdr=high_quality_hdr,
        )
        return chunks, fps

    def _iter_decoded(
        self,
        latent: mx.array,
        *,
        seed: int,
        keyframes: DecodeKeyframes | None,
        crop_h: int,
        crop_w: int,
        high_quality_hdr: bool,
    ) -> Iterator[np.ndarray]:
        """Decode lazily to FHWC ``[0, 1]`` chunks; crop the pad; HQ keeps every other frame (upstream ``_decode_video``).

        Upstream decimates the concatenated clip (``decoded[::2]``); here the chunks stream, so the
        global frame index is tracked and only even indices are kept across chunk boundaries.
        """
        try:
            if keyframes is not None:
                # Upstream docs (``hdr.md``): a decoder without a trained ``type_emb`` loads with zeros and the
                # keyframe stream then silently does nothing. Refuse instead of shipping a no-op decode.
                type_emb = self.video_decoder_block.load()._decoder.type_emb
                if bool(mx.all(type_emb == 0).item()):
                    raise RuntimeError(
                        "the pack's diffusion video decoder has no trained type_emb; keyframe-aware decode would "
                        "silently do nothing (use --no-keyframes)"
                    )
            index = 0
            with phase("Decoding HDR video (fp32 diffusion decoder)", verbose=self.verbose):
                for chunk in self.video_decoder_block.iter_frames(latent, seed=seed, keyframes=keyframes):
                    chunk = chunk[:, :crop_h, :crop_w]
                    if high_quality_hdr:
                        start = index
                        index += chunk.shape[0]
                        chunk = chunk[(-start) % 2 :: 2]
                    if chunk.shape[0]:
                        yield chunk
        finally:
            # Also on an error or an abandoned iterator (``GeneratorExit``): never leave the fp32 decoder resident.
            self.video_decoder_block.free()

    def generate_and_save(
        self,
        video: VideoInput | EXRVideoInput,
        output_path: str,
        *,
        exr_color_space: EXRColorSpace = EXRColorSpace.ACESCG,
        **generate_kwargs,
    ) -> Path:
        """Generate, then write the HLG master to ``output_path`` and the EXR sequence beside it.

        Returns:
            The EXR directory.
        """
        chunks, fps = self.generate(video, **generate_kwargs)
        # Upstream ``encode_video(..., fps=round(fps), color_space=args.exr_colorspace)``.
        return encode_hdr_outputs(chunks, output_path, round(fps), exr_color_space)


__all__ = [
    "ALIGNMENT_DIVISOR",
    "DEFAULT_DENOISE_SIGMAS",
    "DEFAULT_KEYFRAME_STRENGTH",
    "MIN_RESOLUTION",
    "TILED_VAE_ENCODE_PIXEL_THRESHOLD",
    "HDRICLoraPipeline",
    "conditioning_fps",
    "dfr_seam_roles",
    "load_video_context",
    "require_hdr_export_tools",
]
