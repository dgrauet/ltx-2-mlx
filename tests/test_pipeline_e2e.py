"""End-to-end pipeline tests with real weights.

Tests the full audio pipeline (VAE decode -> vocoder -> WAV file)
and a single transformer forward pass.

Run with: pytest tests/test_pipeline_e2e.py -v -s
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.utils.memory import aggressive_cleanup, get_memory_stats
from tests.conftest import MODEL_DIR

skip_no_weights = pytest.mark.skipif(MODEL_DIR is None, reason="q8 model weights not found")


@skip_no_weights
class TestAudioPipelineE2E:
    """Full audio pipeline: random latent -> audio VAE -> vocoder -> WAV."""

    def test_latent_to_wav(self):
        from mlx.utils import tree_unflatten

        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

        # Load audio VAE
        audio_decoder = AudioVAEDecoder()
        audio_weights = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.decoder.")
        all_audio = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                audio_weights[k] = v
        audio_weights = remap_audio_vae_keys(audio_weights)
        audio_decoder.update(tree_unflatten(list(audio_weights.items())))
        print(f"\nLoaded audio VAE ({len(audio_weights)} params)")

        # Load vocoder
        vocoder = VocoderWithBWE()
        vocoder_weights = load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.")
        vocoder.load_weights(list(vocoder_weights.items()))
        print(f"Loaded vocoder+BWE ({len(vocoder_weights)} params)")

        # Random audio latent: ~1 second at 24fps
        mx.random.seed(42)
        audio_latent = mx.random.normal((1, 8, 25, 16)).astype(mx.bfloat16)
        print(f"Audio latent shape: {audio_latent.shape}")

        # Step 1: Decode latent -> mel
        mel = audio_decoder.decode(audio_latent)
        mx.synchronize()
        print(f"Mel shape: {mel.shape}")
        aggressive_cleanup()

        # Step 2: Vocoder mel -> 16kHz waveform (just base vocoder for speed)
        # mel is (B, 2, T, 64) — vocoder expects (B, T, 128) = stereo mel concatenated
        B_mel = mel.shape[0]
        T_mel = mel.shape[2]
        mel_concat = mel.transpose(0, 2, 1, 3).reshape(B_mel, T_mel, -1)  # (B, T, 128)
        wav_16k_raw = vocoder._run_base_vocoder(mel_concat)  # (B, T_audio, 2)
        mx.synchronize()

        wav_16k = wav_16k_raw.transpose(0, 2, 1)  # (B, 2, T_audio)
        print(f"Waveform 16kHz shape: {wav_16k.shape}")
        aggressive_cleanup()

        # Step 3: Save to WAV
        wav_np = np.array(wav_16k[0].transpose(1, 0), dtype=np.float32)  # (T, 2)
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(wav_int16.tobytes())

        file_size = Path(wav_path).stat().st_size
        duration = wav_int16.shape[0] / 16000
        print(f"WAV file: {wav_path}")
        print(f"  Size: {file_size / 1024:.1f} KB")
        print(f"  Duration: {duration:.2f}s, Channels: 2, Sample rate: 16kHz")

        assert file_size > 100, "WAV file too small"
        assert duration > 0.1, "WAV too short"

        # Cleanup
        Path(wav_path).unlink(missing_ok=True)
        aggressive_cleanup()
        print("Audio pipeline E2E test PASSED")


@skip_no_weights
class TestTransformerForwardPass:
    """Single forward pass through the real 19B transformer (q8)."""

    def test_single_step(self):
        from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

        print(f"\nMemory before load: {get_memory_stats()}")

        config = LTXModelConfig()
        model = LTXModel(config)

        weights = load_split_safetensors(MODEL_DIR / "transformer.safetensors", prefix="transformer.")
        apply_quantization(model, weights)
        model.load_weights(list(weights.items()), strict=True)
        mx.synchronize()

        print(f"Loaded transformer ({len(weights)} params)")
        print(f"Memory after load: {get_memory_stats()}")

        # Very small input: 1 video token, 1 audio token, 1 text token
        B = 1
        mx.random.seed(42)
        video_latent = mx.random.normal((B, 4, 128)).astype(mx.bfloat16)
        audio_latent = mx.random.normal((B, 2, 128)).astype(mx.bfloat16)
        timestep = mx.array([0.5])
        video_text = mx.random.normal((B, 2, 4096)).astype(mx.bfloat16)
        audio_text = mx.random.normal((B, 2, 2048)).astype(mx.bfloat16)

        print("Running single forward pass (tiny input)...")
        video_out, audio_out = model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=timestep,
            video_text_embeds=video_text,
            audio_text_embeds=audio_text,
        )
        mx.synchronize()

        print(f"Video output: {video_out.shape} {video_out.dtype}")
        print(f"Audio output: {audio_out.shape} {audio_out.dtype}")
        print(f"Memory after forward: {get_memory_stats()}")

        assert video_out.shape == (B, 4, 128)
        assert audio_out.shape == (B, 2, 128)

        # Verify output is not all zeros (model actually computed something)
        assert float(mx.abs(video_out).max()) > 0
        assert float(mx.abs(audio_out).max()) > 0

        del model, weights
        aggressive_cleanup()
        print("Transformer forward pass PASSED")
