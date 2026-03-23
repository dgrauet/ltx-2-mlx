"""Keyframe interpolation pipeline — two-stage interpolation between reference frames.

Matches reference architecture: stage 1 at half resolution with optional CFG guidance,
neural upscale 2x, then stage 2 refinement at full resolution.

Keyframe images are re-encoded by the VAE at each stage's resolution (matching
reference ``image_conditionings_by_adding_guiding_latent``), rather than
downsampling pre-encoded latents.

Ported from ltx-pipelines keyframe_interpolation.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from PIL import Image

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, create_initial_state, noise_latent_state
from ltx_core_mlx.loader.fuse_loras import apply_loras
from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.ti2vid_two_stages import TwoStagePipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop


def _flatten_dict(d: dict, prefix: str, out: dict[str, mx.array]) -> None:
    """Flatten a nested dict to dot-separated keys."""
    for k, v in d.items():
        full_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, f"{full_key}.", out)
        else:
            out[full_key] = v


def _encode_keyframe(
    vae_encoder: VideoEncoder,
    image: Image.Image | str,
    height: int,
    width: int,
) -> mx.array:
    """Encode a keyframe image at a specific resolution.

    If the target resolution produces odd latent dims (not divisible by 64),
    encodes at the next 64-multiple and crops the latent tokens to match the
    target latent grid. This avoids the VAE encoder's space-to-depth stride=2
    constraint while producing tokens at the correct spatial dimensions.

    Args:
        vae_encoder: VAE encoder.
        image: PIL Image or path.
        height: Target pixel height.
        width: Target pixel width.

    Returns:
        Patchified keyframe tokens (1, H_lat*W_lat, 128).
    """
    target_lat_h = height // 32
    target_lat_w = width // 32

    # VAE encoder needs even latent dims (space-to-depth stride=2).
    # Encode at the next valid resolution and crop if needed.
    enc_h = height if target_lat_h % 2 == 0 else (target_lat_h + 1) * 32
    enc_w = width if target_lat_w % 2 == 0 else (target_lat_w + 1) * 32

    img_tensor = prepare_image_for_encoding(image, enc_h, enc_w)
    latent = vae_encoder.encode(img_tensor[:, :, None, :, :])
    mx.eval(latent)

    # Crop to target latent dims if we encoded at a larger resolution
    # latent shape: (1, 128, 1, H_enc, W_enc) in BCFHW
    latent = latent[:, :, :, :target_lat_h, :target_lat_w]

    # (1, 128, 1, H', W') -> (1, H'*W', 128) tokens
    tokens = latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
    return tokens


def _remap_lora_keys(lora_sd: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap LoRA keys from ComfyUI/diffusion_model format to MLX model format.

    Applies the LTXV_LORA_COMFY_RENAMING_MAP replacements plus additional
    MLX-specific key remapping:
    - ``diffusion_model.`` → ```` (strip prefix)
    - ``.to_out.0.`` → ``.to_out.`` (no Sequential index)
    - ``.ff.net.0.proj.`` → ``.ff.proj_in.`` (MLX naming)
    - ``.ff.net.2.`` → ``.ff.proj_out.`` (MLX naming)
    - ``.linear_1.`` → ``.linear1.`` (MLX AdaLN naming)
    - ``.linear_2.`` → ``.linear2.`` (MLX AdaLN naming)
    """
    remapped: dict[str, mx.array] = {}
    for key, value in lora_sd.items():
        new_key = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(key)
        # MLX timestep embedder uses linear1/linear2 (no underscore)
        new_key = new_key.replace(".linear_1.", ".linear1.").replace(".linear_2.", ".linear2.")
        # audio_ff uses same net.0.proj / net.2 pattern as ff, but underscore prefix
        # prevents the .ff.net replacement from matching
        new_key = new_key.replace("audio_ff.net.0.proj.", "audio_ff.proj_in.")
        new_key = new_key.replace("audio_ff.net.2.", "audio_ff.proj_out.")
        remapped[new_key] = value
    return remapped


class KeyframeInterpolationPipeline(TwoStagePipeline):
    """Two-stage keyframe interpolation pipeline.

    Stage 1: Generate at half resolution with keyframe conditioning + optional CFG.
             When ``dev_transformer`` is specified, uses the non-distilled model
             for higher quality interpolation (matching the reference).
    Stage 2: Neural upscale 2x, re-apply keyframe conditioning at full resolution,
             refine with distilled model (dev + LoRA fusion, or standalone distilled).

    Args:
        model_dir: Path to model weights.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Filename of the dev (non-distilled) transformer weights
            inside model_dir (e.g. ``transformer-dev.safetensors``). When provided,
            stage 1 uses this model and stage 2 fuses the distilled LoRA on top.
        distilled_lora: Filename of the distilled LoRA weights inside model_dir.
            Required when ``dev_transformer`` is set.
        distilled_lora_strength: Strength for the distilled LoRA fusion (default 1.0).
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        dev_transformer: str | None = None,
        distilled_lora: str | None = None,
        distilled_lora_strength: float = 1.0,
    ):
        super().__init__(model_dir, gemma_model_id=gemma_model_id, low_memory=low_memory)
        self._dev_transformer = dev_transformer
        self._distilled_lora = distilled_lora
        self._distilled_lora_strength = distilled_lora_strength

    def _load_dev_transformer(self) -> LTXModel:
        """Load the dev (non-distilled) transformer weights."""
        assert self._dev_transformer is not None
        dev_path = self.model_dir / self._dev_transformer
        dit = LTXModel()
        weights = load_split_safetensors(dev_path, prefix="transformer.")
        apply_quantization(dit, weights)
        dit.load_weights(list(weights.items()))
        aggressive_cleanup()
        return dit

    def _fuse_distilled_lora(self, dit: LTXModel) -> None:
        """Fuse distilled LoRA weights into a loaded transformer in-place."""
        assert self._distilled_lora is not None
        lora_path = self.model_dir / self._distilled_lora
        lora_raw = dict(mx.load(str(lora_path)))
        lora_remapped = _remap_lora_keys(lora_raw)

        # Flatten model parameters to dot-separated keys using MLX tree_flatten
        import mlx.utils

        flat_params = mlx.utils.tree_flatten(dit.parameters())
        flat_model = {k: v for k, v in flat_params if isinstance(v, mx.array)}

        model_sd = StateDict(sd=flat_model, size=0, dtype=set())
        lora_sd = StateDict(sd=lora_remapped, size=0, dtype=set())
        lora_with_strength = LoraStateDictWithStrength(lora_sd, self._distilled_lora_strength)

        fused = apply_loras(model_sd, [lora_with_strength])
        dit.load_weights(list(fused.sd.items()))
        aggressive_cleanup()

    def _load_vae_encoder(self) -> VideoEncoder:
        """Load VAE encoder for keyframe encoding."""
        vae_encoder = VideoEncoder()
        enc_weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
        enc_weights = {
            k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
            for k, v in enc_weights.items()
        }
        vae_encoder.load_weights(list(enc_weights.items()))
        return vae_encoder

    def interpolate(
        self,
        prompt: str,
        keyframe_images: list[Image.Image | str],
        keyframe_indices: list[int],
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        fps: float = 24.0,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        negative_prompt_embeds: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video interpolating between keyframes using two-stage pipeline.

        Args:
            prompt: Text prompt.
            keyframe_images: List of keyframe images (PIL or paths).
            keyframe_indices: Pixel frame indices for each keyframe (0-based).
            height: Final video height.
            width: Final video width.
            num_frames: Total number of pixel frames.
            fps: Frame rate.
            seed: Random seed.
            stage1_steps: Stage 1 denoising steps.
            stage2_steps: Stage 2 denoising steps.
            video_guider_params: Video guidance params. Defaults to LTX_2_3_PARAMS.
            audio_guider_params: Audio guidance params. Defaults to LTX_2_3_PARAMS.
            negative_prompt_embeds: Optional (video_neg, audio_neg) for CFG.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        # Use exact half dimensions like the reference. The VAE encoder
        # handles odd latent dims via encode-at-larger + crop in _encode_keyframe.
        half_h = height // 2
        half_w = width // 2

        # Compute the actual upscaled resolution (upsampler doubles spatial latent dims).
        # This determines the full-res keyframe encoding resolution.
        F_half, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        H_up, W_up = H_half * 2, W_half * 2
        up_h, up_w = H_up * 32, W_up * 32  # pixel resolution after upscaling

        # --- Encode keyframes at both resolutions ---
        vae_encoder = self._load_vae_encoder()

        kf_tokens_half = []
        for img in keyframe_images:
            tokens = _encode_keyframe(vae_encoder, img, half_h, half_w)
            kf_tokens_half.append(tokens)

        kf_tokens_full = []
        for img in keyframe_images:
            tokens = _encode_keyframe(vae_encoder, img, up_h, up_w)
            kf_tokens_full.append(tokens)

        # Keep per_channel_statistics for upsampler normalization (reference:
        # upsample_video wraps with un_normalize/normalize). These are tiny (128 floats each).
        self._encoder_mean = vae_encoder.per_channel_statistics.mean_of_means
        self._encoder_std = vae_encoder.per_channel_statistics.std_of_means

        # Force evaluation before freeing encoder
        mx.eval(*(kf_tokens_half + kf_tokens_full))
        del vae_encoder
        aggressive_cleanup()

        # --- Text encoding (load Gemma, encode, free) ---
        use_dev = self._dev_transformer is not None
        if self.text_encoder is None or self.feature_extractor is None:
            model_dir = self.model_dir
            if self.text_encoder is None:
                from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

                self.text_encoder = GemmaLanguageModel()
                self.text_encoder.load(self._gemma_model_id)
                aggressive_cleanup()
            if self.feature_extractor is None:
                from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2

                self.feature_extractor = GemmaFeaturesExtractorV2()
                conn_weights = load_split_safetensors(model_dir / "connector.safetensors", prefix="connector.")
                self.feature_extractor.connector.load_weights(list(conn_weights.items()))
                aggressive_cleanup()

        video_embeds, audio_embeds = self._encode_text(prompt)
        mx.eval(video_embeds, audio_embeds)
        aggressive_cleanup()  # Free Gemma graph before second encode

        # Encode negative prompt only when guidance is explicitly requested.
        # NOTE: The reference uses guided denoising with CFG/STG/modality/rescale
        # by default, but the MLX guided_denoise_loop currently produces text overlay
        # artifacts with CFG+rescale. Until this is investigated, default to simple
        # denoising which produces clean transitions (validated by test-transition-7).
        neg_video_embeds = None
        neg_audio_embeds = None
        use_guidance = video_guider_params is not None or audio_guider_params is not None
        if use_guidance:
            from ltx_pipelines_mlx.utils.constants import DEFAULT_NEGATIVE_PROMPT

            neg_video_embeds, neg_audio_embeds = self._encode_text(DEFAULT_NEGATIVE_PROMPT)
            mx.eval(neg_video_embeds, neg_audio_embeds)
            aggressive_cleanup()

        # Free text encoder before loading transformer
        self.text_encoder = None
        self.feature_extractor = None
        aggressive_cleanup()

        # --- Load transformer (dev or distilled) + upsampler only ---
        if use_dev:
            self.dit = self._load_dev_transformer()
        elif self.dit is None:
            self.dit = LTXModel()
            transformer_path = self.model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = self.model_dir / "transformer-distilled.safetensors"
            weights = load_split_safetensors(transformer_path, prefix="transformer.")
            apply_quantization(self.dit, weights)
            self.dit.load_weights(list(weights.items()))
            aggressive_cleanup()

        if self.upsampler is None:
            self._load_upsampler()

        assert self.dit is not None
        assert self.upsampler is not None

        # --- Stage 1: Half resolution with keyframe conditioning ---
        F = F_half  # already computed above
        video_shape_1 = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, fps=fps)
        audio_positions = compute_audio_positions(audio_T)

        video_state_1 = create_initial_state(video_shape_1, seed, positions=video_positions_1)
        audio_state_1 = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Apply keyframe conditioning at half resolution
        for tokens, kf_idx in zip(kf_tokens_half, keyframe_indices):
            kf_condition = VideoConditionByKeyframeIndex(
                frame_idx=kf_idx,
                keyframe_latent=tokens,
                spatial_dims=(F, H_half, W_half),
                fps=fps,
            )
            video_state_1 = kf_condition.apply(video_state_1, (F, H_half, W_half))

        # Stage 1 sigma schedule: dev model uses LTX2Scheduler (dynamic schedule),
        # distilled model uses predefined DISTILLED_SIGMAS.
        if use_dev:
            s1_steps = stage1_steps or 30  # Reference default: LTX_2_3_PARAMS.num_inference_steps=30
            sigmas_1 = ltx2_schedule(s1_steps)  # Default num_tokens=4096 matches reference
        else:
            sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        # Stage 1 denoising: guided when params provided, simple otherwise
        if use_guidance:
            video_neg = negative_prompt_embeds[0] if negative_prompt_embeds else neg_video_embeds
            audio_neg = negative_prompt_embeds[1] if negative_prompt_embeds else neg_audio_embeds
            video_factory = create_multimodal_guider_factory(video_guider_params, negative_context=video_neg)
            audio_factory = create_multimodal_guider_factory(audio_guider_params, negative_context=audio_neg)

            output_1 = guided_denoise_loop(
                model=x0_model,
                video_state=video_state_1,
                audio_state=audio_state_1,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                video_guider_factory=video_factory,
                audio_guider_factory=audio_factory,
                sigmas=sigmas_1,
            )
        else:
            output_1 = denoise_loop(
                model=x0_model,
                video_state=video_state_1,
                audio_state=audio_state_1,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_1,
            )
        if self.low_memory:
            aggressive_cleanup()

        # Extract generated tokens (without appended keyframe tokens)
        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]

        # --- Fuse distilled LoRA for stage 2 (if using dev model) ---
        if use_dev and self._distilled_lora:
            self._fuse_distilled_lora(self.dit)

        # --- Upscale (reference: upsample_video with per-channel normalization) ---
        from ltx_core_mlx.model.upsampler import upsample_video

        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))
        video_upscaled = upsample_video(video_half, self._encoder_mean, self._encoder_std, self.upsampler)
        mx.eval(video_upscaled)
        if self.low_memory:
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: Upscaled resolution with keyframe conditioning ---
        # H_up/W_up already computed above from H_half*2, W_half*2
        H_full, W_full = H_up, W_up
        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)

        # Noise the upscaled latent
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens_up.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens_up * (1.0 - start_sigma)

        video_positions_2 = compute_video_positions(F, H_full, W_full, fps=fps)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens_up,
            denoise_mask=mx.ones((1, video_tokens_up.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        # Apply keyframe conditioning at full resolution
        for tokens, kf_idx in zip(kf_tokens_full, keyframe_indices):
            kf_condition = VideoConditionByKeyframeIndex(
                frame_idx=kf_idx,
                keyframe_latent=tokens,
                spatial_dims=(F, H_full, W_full),
                fps=fps,
            )
            video_state_2 = kf_condition.apply(video_state_2, (F, H_full, W_full))

        # Audio state for stage 2
        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = LatentState(
            latent=audio_tokens_1,
            clean_latent=audio_tokens_1,
            denoise_mask=mx.ones((1, audio_tokens_1.shape[1], 1), dtype=audio_tokens_1.dtype),
            positions=audio_positions,
        )
        audio_state_2 = noise_latent_state(audio_state_2, sigma=start_sigma, seed=seed + 2)

        # Stage 2 denoising: simple (no CFG), matching reference
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

        # Free transformer + upsampler before decode phase
        if self.low_memory:
            self.dit = None
            self.upsampler = None
            aggressive_cleanup()

        # Extract generated tokens (without appended keyframe tokens)
        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        keyframe_images: list[Image.Image | str] | None = None,
        keyframe_indices: list[int] | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        fps: float = 24.0,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        **kwargs: object,
    ) -> str:
        """Generate two-stage keyframe interpolation and save to file.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            keyframe_images: Keyframe images (PIL or paths).
            keyframe_indices: Pixel frame indices for each keyframe.
            height: Final video height.
            width: Final video width.
            num_frames: Total number of pixel frames.
            fps: Frame rate.
            seed: Random seed.
            stage1_steps: Stage 1 denoising steps.
            stage2_steps: Stage 2 denoising steps.
            cfg_scale: CFG guidance scale for stage 1.

        Returns:
            Path to output video file.
        """
        if keyframe_images is None or keyframe_indices is None:
            raise ValueError("keyframe_images and keyframe_indices are required")

        video_latent, audio_latent = self.interpolate(
            prompt=prompt,
            keyframe_images=keyframe_images,
            keyframe_indices=keyframe_indices,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )

        # Free any remaining heavy components from generation phase
        if self.low_memory:
            self.dit = None
            self.text_encoder = None
            self.feature_extractor = None
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        # --- Decode phase: load decoders on demand ---
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.weights import remap_audio_vae_keys

        model_dir = self.model_dir

        if self.audio_decoder is None:
            self.audio_decoder = AudioVAEDecoder()
            audio_weights = load_split_safetensors(model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder.")
            all_audio = load_split_safetensors(model_dir / "audio_vae.safetensors", prefix="audio_vae.")
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    audio_weights[k] = v
            audio_weights = remap_audio_vae_keys(audio_weights)
            self.audio_decoder.load_weights(list(audio_weights.items()))
            aggressive_cleanup()

        if self.vocoder is None:
            self.vocoder = VocoderWithBWE()
            vocoder_weights = load_split_safetensors(model_dir / "vocoder.safetensors", prefix="vocoder.")
            self.vocoder.load_weights(list(vocoder_weights.items()))
            aggressive_cleanup()

        mel = self.audio_decoder.decode(audio_latent)
        waveform = self.vocoder(mel)
        if self.low_memory:
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()

        import tempfile

        audio_path = tempfile.mktemp(suffix=".wav")
        self._save_waveform(waveform, audio_path, sample_rate=48000)

        if self.vae_decoder is None:
            self.vae_decoder = VideoDecoder()
            vae_weights = load_split_safetensors(model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
            self.vae_decoder.load_weights(list(vae_weights.items()))
            aggressive_cleanup()

        self.vae_decoder.decode_and_stream(video_latent, output_path, fps=fps, audio_path=audio_path)

        Path(audio_path).unlink(missing_ok=True)
        aggressive_cleanup()

        return output_path
