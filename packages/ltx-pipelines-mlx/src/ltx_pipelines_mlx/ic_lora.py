"""IC-LoRA pipeline — reference video conditioning with two-stage generation.

Ported from ltx-pipelines/src/ltx_pipelines/ic_lora.py

Two-stage video generation pipeline with In-Context (IC) LoRA support.
Allows conditioning the generated video on control signals such as depth maps,
human pose, or image edges via the video_conditioning parameter.
The specific IC-LoRA model should be provided via the loras parameter.
Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
by 2x and refines with additional denoising steps for higher quality output.
Both stages use distilled models for efficiency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.attention_strength_wrapper import (
    ConditioningItemAttentionStrengthWrapper,
)
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
    create_initial_state,
    noise_latent_state,
)
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.loader import (
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraStateDictWithStrength,
    SafetensorsStateDictLoader,
    StateDict,
    apply_loras,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.video import load_video_frames_normalized
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop

logger = logging.getLogger(__name__)


class ICLoraPipeline(TextToVideoPipeline):
    """Two-stage video generation pipeline with IC-LoRA reference conditioning.

    Conditions the generated video on a reference video (e.g., depth, pose, edges)
    via VideoConditionByReferenceLatent. Stage 1 generates at half resolution with
    the IC-LoRA fused into the transformer, then Stage 2 upscales and refines
    without the LoRA (clean distilled model).

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        lora_paths: List of (lora_path, strength) tuples for IC-LoRA weights.
        low_memory: Aggressive memory management.
    """

    def __init__(
        self,
        model_dir: str,
        lora_paths: list[tuple[str, float]] | None = None,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
    ):
        super().__init__(model_dir, gemma_model_id=gemma_model_id, low_memory=low_memory)
        self.vae_encoder: VideoEncoder | None = None
        self.upsampler: LatentUpsampler | None = None

        # Resolve LoRA paths (download from HuggingFace if needed)
        self._lora_paths = [(_resolve_lora_path(p), s) for p, s in (lora_paths or [])]

        # Read reference downscale factor from LoRA metadata.
        # IC-LoRAs trained with low-resolution reference videos store this factor
        # so inference can resize reference videos to match training conditions.
        self.reference_downscale_factor = 1
        for lora_path, _ in self._lora_paths:
            scale = _read_lora_reference_downscale_factor(lora_path)
            if scale != 1:
                if self.reference_downscale_factor not in (1, scale):
                    raise ValueError(
                        f"Conflicting reference_downscale_factor values in LoRAs: "
                        f"already have {self.reference_downscale_factor}, but {lora_path} "
                        f"specifies {scale}. Cannot combine LoRAs with different reference scales."
                    )
                self.reference_downscale_factor = scale

    def load(self) -> None:
        """Load generation components: DiT, VAE encoder, upsampler.

        Does NOT load decoders (VAE decoder, audio decoder, vocoder) to save
        memory. Those are loaded on-demand in generate_and_save().
        """
        if self._loaded:
            return

        model_dir = self.model_dir

        # DiT (largest component)
        if self.dit is None:
            from ltx_core_mlx.model.transformer.model import LTXModel

            self.dit = LTXModel()
            transformer_path = model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = model_dir / "transformer-distilled.safetensors"
            transformer_weights = load_split_safetensors(transformer_path, prefix="transformer.")
            apply_quantization(self.dit, transformer_weights)
            self.dit.load_weights(list(transformer_weights.items()))
            aggressive_cleanup()

        # VAE encoder (for encoding control videos and I2V images)
        self._load_vae_encoder()

        # Upsampler (for Stage 2)
        if self.upsampler is None:
            import json

            name = "spatial_upscaler_x2_v1_1"
            config_path = model_dir / f"{name}_config.json"
            weights_path = model_dir / f"{name}.safetensors"
            if config_path.exists():
                config = json.loads(config_path.read_text()).get("config", {})
                self.upsampler = LatentUpsampler.from_config(config)
            else:
                self.upsampler = LatentUpsampler()
            if weights_path.exists():
                weights = load_split_safetensors(weights_path, prefix=f"{name}.")
                self.upsampler.load_weights(list(weights.items()))
            aggressive_cleanup()

        self._loaded = True

    def _fuse_loras(self) -> None:
        """Fuse all LoRA weights into the transformer.

        Reads LoRA files, applies ComfyUI key remapping, fuses deltas into
        model weights, and re-quantizes.
        """
        if not self._lora_paths:
            return

        assert self.dit is not None

        import mlx.utils

        model_weights = dict(mlx.utils.tree_flatten(self.dit.parameters()))
        model_sd = StateDict(sd=model_weights, size=0, dtype=set())

        loader = SafetensorsStateDictLoader()
        lora_sds = []
        for lora_path, strength in self._lora_paths:
            lora_sd = loader.load(lora_path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            lora_sds.append(LoraStateDictWithStrength(state_dict=lora_sd, strength=strength))
            logger.info(f"Loaded LoRA: {lora_path} (strength={strength})")

        fused_sd = apply_loras(model_sd=model_sd, lora_sd_and_strengths=lora_sds)

        apply_quantization(self.dit, fused_sd.sd)
        self.dit.load_weights(list(fused_sd.sd.items()))
        aggressive_cleanup()

        logger.info(f"Fused {len(self._lora_paths)} LoRA(s) into transformer")

    def _reload_clean_transformer(self) -> None:
        """Reload the transformer without LoRA for Stage 2.

        The reference implementation uses separate ModelLedgers for Stage 1
        (with LoRA) and Stage 2 (clean distilled). We achieve the same by
        discarding the fused transformer and reloading from disk.
        """
        from ltx_core_mlx.model.transformer.model import LTXModel

        self.dit = None
        aggressive_cleanup()

        self.dit = LTXModel()
        transformer_path = self.model_dir / "transformer.safetensors"
        if not transformer_path.exists():
            transformer_path = self.model_dir / "transformer-distilled.safetensors"
        transformer_weights = load_split_safetensors(transformer_path, prefix="transformer.")
        apply_quantization(self.dit, transformer_weights)
        self.dit.load_weights(list(transformer_weights.items()))
        aggressive_cleanup()
        logger.info("Reloaded clean transformer for Stage 2")

    def _create_conditionings(
        self,
        images: list[tuple[str, int, float]] | None,
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: mx.array | None = None,
    ) -> list[object]:
        """Create conditioning items for video generation.

        Builds image conditionings (I2V) and IC-LoRA reference video conditionings.
        Matches the reference implementation's _create_conditionings().

        Args:
            images: Optional list of (image_path, frame_index, strength) for I2V.
            video_conditioning: List of (video_path, strength) for IC-LoRA reference.
            height: Stage output height (pixels).
            width: Stage output width (pixels).
            num_frames: Number of pixel frames.
            conditioning_attention_strength: Scalar attention weight in [0, 1].
            conditioning_attention_mask: Optional pixel-space mask (1, 1, F, H, W).

        Returns:
            List of conditioning items. IC-LoRA conditionings are appended last.
        """
        assert self.vae_encoder is not None

        conditionings: list[object] = []

        # Image conditionings (I2V)
        if images:
            for img_path, frame_idx, strength in images:
                img_tensor = prepare_image_for_encoding(img_path, height, width)
                img_tensor = img_tensor[:, :, None, :, :]
                img_latent = self.vae_encoder.encode(img_tensor)
                img_tokens = img_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
                conditionings.append(
                    VideoConditionByLatentIndex(
                        frame_indices=[frame_idx],
                        clean_latent=img_tokens,
                        strength=strength,
                    )
                )

        # IC-LoRA reference video conditionings
        scale = self.reference_downscale_factor
        if scale != 1 and (height % scale != 0 or width % scale != 0):
            raise ValueError(
                f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
            )
        # Compute VAE-compatible reference resolution: the VAE encoder requires
        # spatial dims divisible by 32. Derive from latent shape (same approach
        # as keyframe pipeline half-resolution).
        _, ref_H_lat, ref_W_lat = compute_video_latent_shape(num_frames, height // scale, width // scale)
        ref_height = ref_H_lat * 32
        ref_width = ref_W_lat * 32

        for video_path, strength in video_conditioning:
            # Load video at scaled-down resolution (if scale > 1)
            video = load_video_frames_normalized(video_path, ref_height, ref_width, num_frames)
            # Normalize to [-1, 1] for VAE encoding
            video = (video * 2.0 - 1.0).astype(mx.bfloat16)
            encoded_video = self.vae_encoder.encode(video)
            mx.eval(encoded_video)

            # Derive reference latent dims
            ref_F = encoded_video.shape[2]
            ref_H = encoded_video.shape[3]
            ref_W = encoded_video.shape[4]
            ref_positions = compute_video_positions(ref_F, ref_H, ref_W)

            # Patchify reference to tokens
            ref_tokens = encoded_video.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)

            # Build attention mask for ConditioningItemAttentionStrengthWrapper
            if conditioning_attention_mask is not None:
                latent_mask = _downsample_mask_to_latent(
                    mask=conditioning_attention_mask,
                    target_f=ref_F,
                    target_h=ref_H,
                    target_w=ref_W,
                )
                attn_mask = latent_mask * conditioning_attention_strength
            elif conditioning_attention_strength < 1.0:
                attn_mask = conditioning_attention_strength
            else:
                attn_mask = None

            cond = VideoConditionByReferenceLatent(
                reference_latent=ref_tokens,
                reference_positions=ref_positions,
                downscale_factor=scale,
                strength=strength,
            )
            if attn_mask is not None:
                cond = ConditioningItemAttentionStrengthWrapper(
                    conditioning=cond,
                    attention_mask=attn_mask,
                )
            conditionings.append(cond)

        if video_conditioning:
            logger.info(f"Added {len(video_conditioning)} video conditioning(s)")

        return conditionings

    def generate(
        self,
        prompt: str,
        video_conditioning: list[tuple[str, float]],
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        images: list[tuple[str, int, float]] | None = None,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: mx.array | None = None,
        skip_stage_2: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """Generate video with IC-LoRA reference conditioning.

        Args:
            prompt: Text prompt.
            video_conditioning: List of (video_path, strength) tuples for IC-LoRA
                reference video conditioning (e.g., depth maps, poses, edges).
            height: Output video height.
            width: Output video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1.
            stage2_steps: Denoising steps for stage 2.
            images: Optional list of (image_path, frame_index, strength) for I2V.
            conditioning_attention_strength: Scale factor for IC-LoRA conditioning
                attention. 0.0 = ignore, 1.0 = full conditioning. Default 1.0.
            conditioning_attention_mask: Optional pixel-space mask (1, 1, F, H, W)
                matching reference video dimensions. Values in [0, 1].
            skip_stage_2: Skip upscale + refine, output at half resolution.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )

        # Load text encoder, encode, free, then load generation components.
        # Done manually instead of _encode_text_and_load() to avoid loading
        # decoders (which we don't need until generate_and_save).
        self._load_text_encoder()
        video_embeds, audio_embeds = self._encode_text(prompt)
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_embeds, audio_embeds)
        if self.low_memory:
            self.text_encoder = None
            self.feature_extractor = None
            aggressive_cleanup()

        # Load DiT + VAE encoder + upsampler (no decoders)
        self.load()

        assert self.dit is not None
        assert self.vae_encoder is not None

        # Fuse LoRA into transformer for Stage 1
        self._fuse_loras()

        # --- Stage 1: Half-resolution generation with IC-LoRA ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions_1)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Encode conditionings before denoising (reduce peak memory)
        stage_1_conditionings = self._create_conditionings(
            images=images,
            video_conditioning=video_conditioning,
            height=half_h,
            width=half_w,
            num_frames=num_frames,
            conditioning_attention_strength=conditioning_attention_strength,
            conditioning_attention_mask=conditioning_attention_mask,
        )

        # Apply image conditionings (replace tokens at frame index)
        image_conds = [c for c in stage_1_conditionings if isinstance(c, VideoConditionByLatentIndex)]
        if image_conds:
            video_state = apply_conditioning(video_state, image_conds, (F, H_half, W_half))

        # Apply IC-LoRA reference conditionings (append tokens)
        for cond in stage_1_conditionings:
            if isinstance(cond, VideoConditionByReferenceLatent | ConditioningItemAttentionStrengthWrapper):
                video_state = cond.apply(video_state, (F, H_half, W_half))

        # Denoise stage 1
        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        output_1 = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Extract only generation tokens (exclude appended reference tokens)
        gen_tokens = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens, (F, H_half, W_half))

        if skip_stage_2:
            audio_latent = self.audio_patchifier.unpatchify(output_1.audio_latent)
            return video_half, audio_latent

        # --- Stage 2: Upscale + refine (no IC-LoRA, clean distilled model) ---
        # Upscale with denormalize/renormalize wrapping (matching reference).
        # Reference: un_normalize -> upsampler -> normalize using VAE encoder stats.
        # Without this, the upsampler produces garbage for Stage 2.
        assert self.upsampler is not None
        assert self.vae_encoder is not None
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)  # back to (B,C,F,H,W)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
        video_up_mlx = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_up_mlx.transpose(0, 4, 1, 2, 3)  # back to (B,C,F,H,W)
        mx.eval(video_upscaled)

        # Derive full-resolution latent dims from the ACTUAL upscaled shape,
        # not the target height/width (which may round differently).
        # The upsampler doubles H_half and W_half, so H_full = H_half * 2.
        H_full = H_half * 2
        W_full = W_half * 2
        enc_h_full = H_full * 32
        enc_w_full = W_full * 32

        # Encode I2V images at upscaled resolution (if any) before freeing encoder
        conditionings_2 = []
        if images:
            for img_path, frame_idx, strength in images:
                img_tensor = prepare_image_for_encoding(img_path, enc_h_full, enc_w_full)
                img_tensor = img_tensor[:, :, None, :, :]
                ref_latent = self.vae_encoder.encode(img_tensor)
                ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
                conditionings_2.append(
                    VideoConditionByLatentIndex(
                        frame_indices=[frame_idx],
                        clean_latent=ref_tokens,
                        strength=strength,
                    )
                )

        # Free VAE encoder + upsampler + fused DiT before loading clean transformer
        if self.low_memory:
            self.vae_encoder = None
            self.upsampler = None
        # Reload clean transformer without LoRA (matches reference: separate model ledgers)
        self._reload_clean_transformer()

        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens_up.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens_up * (1.0 - start_sigma)

        video_positions_2 = compute_video_positions(F, H_full, W_full)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens_up,
            denoise_mask=mx.ones((1, video_tokens_up.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        # Apply I2V conditioning at full resolution for stage 2
        if conditionings_2:
            video_state_2 = apply_conditioning(video_state_2, conditionings_2, (F, H_full, W_full))

        # Audio refined in stage 2
        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = LatentState(
            latent=audio_tokens_1,
            clean_latent=audio_tokens_1,
            denoise_mask=mx.ones((1, audio_tokens_1.shape[1], 1), dtype=audio_tokens_1.dtype),
            positions=audio_positions,
        )
        audio_state_2 = noise_latent_state(audio_state_2, sigma=start_sigma, seed=seed + 2)

        output_2 = denoise_loop(
            model=x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output_2.video_latent, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        video_conditioning: list[tuple[str, float]],
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        images: list[tuple[str, int, float]] | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
    ) -> str:
        """Generate IC-LoRA conditioned video+audio and save to file.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            video_conditioning: List of (video_path, strength) for IC-LoRA reference.
            height: Output video height.
            width: Output video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1.
            stage2_steps: Denoising steps for stage 2.
            images: Optional list of (image_path, frame_index, strength) for I2V.
            conditioning_attention_strength: Attention strength for conditioning.
            skip_stage_2: Skip upscale + refine.

        Returns:
            Path to the output video file.
        """
        video_latent, audio_latent = self.generate(
            prompt=prompt,
            video_conditioning=video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            images=images,
            conditioning_attention_strength=conditioning_attention_strength,
            skip_stage_2=skip_stage_2,
        )

        # Free generation components to make room for decoders
        if self.low_memory:
            self.dit = None
            self.text_encoder = None
            self.feature_extractor = None
            self.upsampler = None
            self.vae_encoder = None
            self._loaded = False
            aggressive_cleanup()

        # Load decoders on-demand (not loaded during generate to save memory)
        self._load_decoders()

        return self._decode_and_save_video(video_latent, audio_latent, output_path)


def _resolve_lora_path(path: str) -> str:
    """Resolve a LoRA path — download from HuggingFace if needed.

    Supports:
        - Local file paths: returned as-is if they exist.
        - HuggingFace repo IDs: downloads the repo and finds the .safetensors file.
          Example: "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"

    Args:
        path: Local path or HuggingFace repo ID.

    Returns:
        Resolved local path to the .safetensors file.
    """
    local = Path(path)
    if local.exists():
        return str(local)

    # Try HuggingFace download
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading LoRA from HuggingFace: {path}")
    repo_dir = Path(snapshot_download(path))

    # Find the .safetensors file in the downloaded repo
    safetensors_files = list(repo_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {repo_dir}")
    if len(safetensors_files) > 1:
        logger.warning(f"Multiple .safetensors files found, using first: {safetensors_files[0].name}")
    return str(safetensors_files[0])


def _read_lora_reference_downscale_factor(lora_path: str) -> int:
    """Read reference_downscale_factor from LoRA safetensors metadata.

    Some IC-LoRA models are trained with reference videos at lower resolution than
    the target output. The downscale factor indicates the ratio between target and
    reference resolutions (e.g., factor=2 means reference is half the resolution).

    Args:
        lora_path: Path to the LoRA .safetensors file.

    Returns:
        The reference downscale factor (1 if not specified in metadata).
    """
    try:
        with safe_open(lora_path, framework="numpy") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_downscale_factor", 1))
    except Exception as e:
        logger.warning(f"Failed to read metadata from LoRA file '{lora_path}': {e}")
        return 1


def _downsample_mask_to_latent(
    mask: mx.array,
    target_f: int,
    target_h: int,
    target_w: int,
) -> mx.array:
    """Downsample a pixel-space mask to latent space using VAE scale factors.

    Handles causal temporal downsampling: the first frame is kept separately
    (temporal scale factor = 1 for the first frame), while the remaining
    frames are downsampled by the VAE's temporal scale factor.

    Args:
        mask: Pixel-space mask of shape (B, 1, F_pixel, H_pixel, W_pixel).
            Values in [0, 1].
        target_f: Target latent temporal dim.
        target_h: Target latent spatial height.
        target_w: Target latent spatial width.

    Returns:
        Flattened latent-space mask of shape (B, F_lat * H_lat * W_lat).
    """
    # Use numpy for interpolation (MLX doesn't have F.interpolate with area mode)
    mask_np = np.array(mask)
    b, _, f_pix, h_pix, w_pix = mask_np.shape

    # Step 1: Spatial downsampling via area interpolation per frame
    from PIL import Image as PILImage

    spatial_down = np.zeros((b, 1, f_pix, target_h, target_w), dtype=np.float32)
    for bi in range(b):
        for fi in range(f_pix):
            frame = mask_np[bi, 0, fi]
            img = PILImage.fromarray((frame * 255).astype(np.uint8))
            img = img.resize((target_w, target_h), PILImage.Resampling.BOX)
            spatial_down[bi, 0, fi] = np.array(img).astype(np.float32) / 255.0

    # Step 2: Causal temporal downsampling
    first_frame = spatial_down[:, :, :1, :, :]  # (B, 1, 1, H_lat, W_lat)

    if f_pix > 1 and target_f > 1:
        t = (f_pix - 1) // (target_f - 1)  # temporal downscale factor
        assert (f_pix - 1) % (target_f - 1) == 0, (
            f"Pixel frames ({f_pix}) not compatible with latent frames ({target_f}): "
            f"(f_pix - 1) must be divisible by (target_f - 1)"
        )
        rest = spatial_down[:, :, 1:, :, :]  # (B, 1, f_pix-1, H, W)
        # Reshape to groups and average
        rest = rest.reshape(b, 1, target_f - 1, t, target_h, target_w)
        rest = rest.mean(axis=3)  # (B, 1, target_f-1, H, W)
        latent_mask = np.concatenate([first_frame, rest], axis=2)
    else:
        latent_mask = first_frame

    # Flatten to (B, F_lat * H_lat * W_lat)
    latent_mask = latent_mask.reshape(b, target_f * target_h * target_w)
    return mx.array(latent_mask)


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> mx.array:
    """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W).

    The mask video is loaded, resized to (height, width), converted to
    grayscale, and normalised to [0, 1].

    Args:
        mask_path: Path to the mask video file.
        height: Target height in pixels.
        width: Target width in pixels.
        num_frames: Maximum number of frames to load.

    Returns:
        Tensor of shape (1, 1, F, H, W) with values in [0, 1].
    """
    # load_video_frames_normalized returns (1, 3, F, H, W) in [0, 1]
    mask_video = load_video_frames_normalized(mask_path, height, width, num_frames)
    # Take mean over channels for grayscale: (1, 3, F, H, W) -> (1, 1, F, H, W)
    mask = mask_video.mean(axis=1, keepdims=True)
    return mx.clip(mask, 0.0, 1.0)
