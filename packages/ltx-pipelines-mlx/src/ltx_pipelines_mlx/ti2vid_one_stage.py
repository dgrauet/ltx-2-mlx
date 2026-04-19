"""Text-to-Video and Image-to-Video pipelines — prompt (+ optional image) to video+audio.

Ported from ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from PIL import Image

from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import (
    VideoConditionByLatentIndex,
    apply_conditioning,
    create_initial_state,
)
from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder, VideoEncoder
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors, remap_audio_vae_keys
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines_mlx.utils.samplers import denoise_loop


class TextToVideoPipeline:
    """Text-to-Video generation pipeline.

    Generates video+audio from a text prompt using the LTX-2.3 model.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        low_memory: If True, aggressively free memory between stages.
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
    ):
        self.model_dir = self._resolve_model_dir(model_dir)
        self._gemma_model_id = gemma_model_id
        self.low_memory = low_memory
        self._loaded = False

        self._dev_transformer: str | None = None

        # Components (lazy-loaded)
        self.dit: LTXModel | None = None
        self.vae_decoder: VideoDecoder | None = None
        self.vae_encoder: VideoEncoder | None = None
        self.audio_decoder: AudioVAEDecoder | None = None
        self.vocoder: VocoderWithBWE | None = None
        self.text_encoder: GemmaLanguageModel | None = None
        self.feature_extractor: GemmaFeaturesExtractorV2 | None = None
        self.audio_encoder: object | None = None
        self.audio_processor: object | None = None
        self.video_patchifier = VideoLatentPatchifier()
        self.audio_patchifier = AudioPatchifier()

    @staticmethod
    def _resolve_model_dir(model_dir: str) -> Path:
        """Resolve model directory — download from HF if needed."""
        path = Path(model_dir)
        if path.exists():
            return path
        # Try HuggingFace download
        return Path(snapshot_download(model_dir))

    @staticmethod
    def _fuse_pending_loras(
        transformer_weights: dict[str, mx.array],
        lora_paths: list[tuple[str, float]],
    ) -> dict[str, mx.array]:
        """Fuse LoRA deltas into transformer weights before model loading.

        Args:
            transformer_weights: Raw transformer state dict.
            lora_paths: List of (path, strength) tuples.

        Returns:
            Modified state dict with LoRA deltas fused.
        """
        from ltx_core_mlx.loader.fuse_loras import apply_loras
        from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
        from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core_mlx.loader.sft_loader import SafetensorsStateDictLoader

        model_sd = StateDict(sd=transformer_weights, size=0, dtype=set())
        loader = SafetensorsStateDictLoader()

        lora_sds = []
        for lora_path, strength in lora_paths:
            lora_sd = loader.load(lora_path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            lora_sds.append(LoraStateDictWithStrength(state_dict=lora_sd, strength=strength))
            print(f"  Fusing LoRA: {lora_path} (strength={strength:.2f})")

        fused_sd = apply_loras(model_sd=model_sd, lora_sd_and_strengths=lora_sds)
        return fused_sd.sd

    # ------------------------------------------------------------------
    # Shared component loading methods (used by subclass pipelines)
    # ------------------------------------------------------------------

    def _load_text_encoder(self) -> None:
        """Load Gemma text encoder and feature extractor connector if not already loaded."""
        if self.text_encoder is None:
            self.text_encoder = GemmaLanguageModel()
            self.text_encoder.load(self._gemma_model_id)
            aggressive_cleanup()

        if self.feature_extractor is None:
            self.feature_extractor = GemmaFeaturesExtractorV2()
            connector_weights = load_split_safetensors(self.model_dir / "connector.safetensors", prefix="connector.")
            self.feature_extractor.connector.load_weights(list(connector_weights.items()))
            aggressive_cleanup()

    def _encode_text_with_negative(self, prompt: str) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Load text encoder, encode prompt + negative prompt, materialize, free encoder.

        Returns:
            Tuple of (video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds).
        """
        self._load_text_encoder()

        video_embeds, audio_embeds = self._encode_text(prompt)
        neg_video_embeds, neg_audio_embeds = self._encode_text(DEFAULT_NEGATIVE_PROMPT)
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds)

        # Free text encoder before loading heavy components
        self.text_encoder = None
        self.feature_extractor = None
        aggressive_cleanup()

        return video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds

    def _load_vae_encoder(self) -> None:
        """Load VAE encoder with key remapping if not already loaded."""
        if self.vae_encoder is not None:
            return
        self.vae_encoder = VideoEncoder()
        enc_weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
        enc_weights = {
            k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
            for k, v in enc_weights.items()
        }
        self.vae_encoder.load_weights(list(enc_weights.items()))
        aggressive_cleanup()

    def _load_audio_encoder(self) -> None:
        """Load audio VAE encoder + processor if not already loaded."""
        if self.audio_encoder is not None:
            return
        from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder

        self.audio_encoder = AudioVAEEncoder()
        encoder_weights = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.encoder.")
        all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                encoder_weights[k] = v
        encoder_weights = remap_audio_vae_keys(encoder_weights)
        self.audio_encoder.load_weights(list(encoder_weights.items()))
        self.audio_processor = AudioProcessor()
        aggressive_cleanup()

    def _load_decoders(self) -> None:
        """Load VAE decoder, audio decoder, and vocoder if not already loaded."""
        model_dir = self.model_dir

        if self.vae_decoder is None:
            self.vae_decoder = VideoDecoder()
            vae_weights = load_split_safetensors(model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
            self.vae_decoder.load_weights(list(vae_weights.items()))
            aggressive_cleanup()

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
            self.vocoder.upcast_weights_to_fp32()
            aggressive_cleanup()

    def _load_dev_transformer(self) -> LTXModel:
        """Load the dev (non-distilled) transformer weights.

        Requires ``self._dev_transformer`` to be set by the subclass.
        """
        assert self._dev_transformer is not None, "_dev_transformer must be set before calling _load_dev_transformer()"
        dev_path = self.model_dir / self._dev_transformer
        if not dev_path.exists():
            raise FileNotFoundError(
                f"Dev transformer not found: {dev_path}\n"
                "This pipeline requires the dev model for CFG guidance.\n"
                "Use: --model dgrauet/ltx-2.3-mlx-q8"
            )
        dit = LTXModel()
        weights = load_split_safetensors(dev_path, prefix="transformer.")
        apply_quantization(dit, weights)
        dit.load_weights(list(weights.items()))
        aggressive_cleanup()
        return dit

    def _decode_and_save_video(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        output_path: str,
        fps: float = 24.0,
    ) -> str:
        """Decode audio + video latents and save to file.

        Decodes audio first (smaller), saves to temp WAV, then streams
        video decode to ffmpeg with audio muxing.

        Args:
            video_latent: Video latent tensor.
            audio_latent: Audio latent tensor.
            output_path: Path to output video file.
            fps: Frame rate.

        Returns:
            Path to the output video file.
        """
        import tempfile

        # Decode audio first (smaller)
        assert self.audio_decoder is not None
        assert self.vocoder is not None
        mel = self.audio_decoder.decode(audio_latent)
        waveform = self.vocoder(mel)
        if self.low_memory:
            aggressive_cleanup()

        # Save audio to temp file
        audio_path = tempfile.mktemp(suffix=".wav")
        self._save_waveform(waveform, audio_path, sample_rate=48000)

        # Decode video and stream to ffmpeg
        assert self.vae_decoder is not None
        self.vae_decoder.decode_and_stream(video_latent, output_path, fps=fps, audio_path=audio_path)

        # Cleanup temp audio
        Path(audio_path).unlink(missing_ok=True)
        aggressive_cleanup()

        return output_path

    # ------------------------------------------------------------------
    # Original one-stage pipeline methods
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all model components from disk.

        In low_memory mode, components are loaded in stages to avoid
        exceeding Metal memory. Gemma (7GB) + connector (6GB) are loaded
        first for text encoding, then freed before loading the
        transformer (10.5GB).
        """
        if self._loaded:
            return

        model_dir = self.model_dir

        # Stage 1: Text encoder + connector (loaded first, freed after encode)
        self._load_text_encoder()

        # Stage 2: DiT (largest component — load after text encoding frees Gemma)
        if self.dit is None:
            if self.low_memory:
                # Free text encoder before loading transformer to fit in RAM
                self.text_encoder = None
                aggressive_cleanup()

            self.dit = LTXModel()
            transformer_path = model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                # Fallback: try transformer-distilled.safetensors (mlx-forge dual-variant layout)
                transformer_path = model_dir / "transformer-distilled.safetensors"
            transformer_weights = load_split_safetensors(transformer_path, prefix="transformer.")

            # Fuse pending LoRA weights before loading into model
            pending_loras = getattr(self, "_pending_loras", None)
            if pending_loras:
                transformer_weights = self._fuse_pending_loras(transformer_weights, pending_loras)

            apply_quantization(self.dit, transformer_weights)
            self.dit.load_weights(list(transformer_weights.items()))
            aggressive_cleanup()

        # Stage 3: VAE + audio (smaller components)
        self._load_decoders()

        self._loaded = True

    def _encode_text_and_load(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode text, then finish loading remaining components.

        In low_memory mode this ensures Gemma is freed before the
        transformer is loaded, keeping peak memory under control.
        """
        # Load text encoder + connector if not already loaded
        self._load_text_encoder()

        # Encode text
        video_embeds, audio_embeds = self._encode_text(prompt)
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(video_embeds, audio_embeds)

        # Free text encoder before loading heavy components
        if self.low_memory:
            self.text_encoder = None
            self.feature_extractor = None
            aggressive_cleanup()

        # Load remaining components (DiT, VAE, vocoder)
        self.load()

        return video_embeds, audio_embeds

    def _encode_text(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode text prompt to video and audio embeddings."""
        assert self.text_encoder is not None
        assert self.feature_extractor is not None

        # Extract ALL 49 layer hidden states with attention mask
        all_hidden_states, attention_mask = self.text_encoder.encode_all_layers(prompt)
        video_embeds, audio_embeds = self.feature_extractor(all_hidden_states, attention_mask=attention_mask)
        return video_embeds, audio_embeds

    def generate(
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video and audio latents.

        Args:
            prompt: Text prompt.
            height: Video height in pixels.
            width: Video width in pixels.
            num_frames: Number of video frames.
            seed: Random seed.
            num_steps: Number of denoising steps (defaults to 8).

        Returns:
            Tuple of (video_latent, audio_latent) in spatial format.
        """
        # Encode text first (loads Gemma + connector, then frees Gemma)
        video_embeds, audio_embeds = self._encode_text_and_load(prompt)

        assert self.dit is not None

        # Compute latent shapes
        F, H, W = compute_video_latent_shape(num_frames, height, width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        # Compute positions for RoPE
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        # Create initial noise with positions
        video_state = create_initial_state(video_shape, seed, positions=video_positions)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Denoise
        sigmas = DISTILLED_SIGMAS[: num_steps + 1] if num_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
        )
        if self.low_memory:
            aggressive_cleanup()

        # Unpatchify
        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> str:
        """Generate and save video+audio to file.

        Args:
            prompt: Text prompt.
            output_path: Path to output video file.
            height: Video height.
            width: Video width.
            num_frames: Number of frames.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Path to the output video file.
        """
        video_latent, audio_latent = self.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_steps=num_steps,
        )

        # Free transformer + text encoder to make room for VAE decode
        if self.low_memory:
            self.dit = None
            self.text_encoder = None
            self.feature_extractor = None
            self._loaded = False
            aggressive_cleanup()

        return self._decode_and_save_video(video_latent, audio_latent, output_path)

    @staticmethod
    def _save_waveform(waveform: mx.array, path: str, sample_rate: int = 48000) -> None:
        """Save waveform to WAV file.

        Args:
            waveform: (B, C, T) or (B, T) waveform.
            path: Output path.
            sample_rate: Sample rate in Hz.
        """
        import wave

        import numpy as np

        # Take first batch item
        wav = waveform[0]
        if wav.ndim == 2:
            num_channels = wav.shape[0]
            wav = wav.T  # (T, C)
        else:
            num_channels = 1
            wav = wav[:, None]  # (T, 1)

        wav_np = np.array(wav.astype(mx.float32), dtype=np.float32)
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767).astype(np.int16)

        with wave.open(path, "w") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(wav_int16.tobytes())


class ImageToVideoPipeline(TextToVideoPipeline):
    """Image-to-Video generation pipeline.

    Extends TextToVideoPipeline to condition on a reference image.
    The first frame is encoded and preserved during denoising.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        low_memory: If True, aggressively free memory between stages.
    """

    def load(self) -> None:
        """Load all model components including VAE encoder."""
        super().load()
        self._load_vae_encoder()

    def generate_from_image(
        self,
        prompt: str,
        image: Image.Image | str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video conditioned on a reference image.

        Args:
            prompt: Text prompt.
            image: Reference image (PIL Image or path).
            height: Video height.
            width: Video width.
            num_frames: Number of frames.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Tuple of (video_latent, audio_latent).
        """
        # I2V needs both VAE encoder (for image) and text encoder (for prompt).
        # In low_memory mode, load and use each before loading the transformer.

        # Step 1: Load VAE encoder, encode image, free
        self._load_vae_encoder()

        img_tensor = prepare_image_for_encoding(image, height, width)
        ref_latent = self.vae_encoder.encode(img_tensor[:, :, None, :, :])
        mx.eval(ref_latent)
        if self.low_memory:
            self.vae_encoder = None
            aggressive_cleanup()

        # Step 2: Encode text, then load remaining components
        video_embeds, audio_embeds = self._encode_text_and_load(prompt)
        assert self.dit is not None

        # Compute shapes
        F, H, W = compute_video_latent_shape(num_frames, height, width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        # Compute positions for RoPE
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        # Create initial state with positions
        video_state = create_initial_state(video_shape, seed, positions=video_positions)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Apply I2V conditioning: preserve first frame
        ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens)
        video_state = apply_conditioning(video_state, [condition], (F, H, W))

        # Denoise
        sigmas = DISTILLED_SIGMAS[: num_steps + 1] if num_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)

        return video_latent, audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        image: Image.Image | str | None = None,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        num_steps: int | None = None,
    ) -> str:
        """Generate and save I2V video+audio.

        Args:
            prompt: Text prompt.
            output_path: Output video path.
            image: Reference image. If None, falls back to T2V.
            height: Video height.
            width: Video width.
            num_frames: Number of frames.
            seed: Random seed.
            num_steps: Number of denoising steps.

        Returns:
            Path to output video.
        """
        if image is None:
            return super().generate_and_save(prompt, output_path, height, width, num_frames, seed, num_steps)

        video_latent, audio_latent = self.generate_from_image(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_steps=num_steps,
        )

        return self._decode_and_save_video(video_latent, audio_latent, output_path)
