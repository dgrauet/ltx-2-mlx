"""Smoke tests for pipeline components with real weights.

These tests load real model weights and run forward passes with random inputs
to verify the components work end-to-end (no correctness check on outputs).

Run with: pytest tests/test_pipeline_smoke.py -v -s
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.utils.memory import aggressive_cleanup
from tests.conftest import MODEL_DIR

skip_no_weights = pytest.mark.skipif(MODEL_DIR is None, reason="q8 model weights not found")


@skip_no_weights
class TestVAEDecoderSmoke:
    """Test VAE decoder forward pass with real weights."""

    def test_decode_random_latent(self):
        """Load VAE decoder weights and decode a tiny random latent."""
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.weights import load_split_safetensors

        decoder = VideoDecoder()
        weights = load_split_safetensors(MODEL_DIR / "vae_decoder.safetensors", prefix="vae_decoder.")
        decoder.load_weights(list(weights.items()))
        print(f"\nLoaded VAE decoder ({len(weights)} params)")

        # Tiny latent: 1 frame, 2x2 spatial = very small
        # BCFHW format: (1, 128, 1, 2, 2)
        mx.random.seed(42)
        latent = mx.random.normal((1, 128, 1, 2, 2)).astype(mx.bfloat16)

        print("Decoding tiny latent (1, 128, 1, 2, 2)...")
        pixels = decoder.decode(latent)
        mx.synchronize()
        print(f"Output pixels shape: {pixels.shape}, dtype: {pixels.dtype}")

        # Check output is valid
        assert pixels.ndim == 5, f"Expected 5D output, got {pixels.ndim}D"
        assert pixels.shape[0] == 1  # batch
        assert pixels.shape[1] == 3 or pixels.shape[1] == 48  # RGB or pre-pixel-shuffle channels

        aggressive_cleanup()
        print("VAE decode smoke test PASSED")


@skip_no_weights
class TestAudioVAESmoke:
    """Test audio VAE decoder forward pass with real weights."""

    def test_decode_random_latent(self):
        """Load audio VAE decoder and decode a tiny random latent."""
        from mlx.utils import tree_unflatten

        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

        decoder = AudioVAEDecoder()
        weights = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.decoder.")
        all_audio = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                weights[k] = v
        weights = remap_audio_vae_keys(weights)
        decoder.update(tree_unflatten(list(weights.items())))
        print(f"\nLoaded audio VAE decoder ({len(weights)} params)")

        # Tiny audio latent: (B=1, C1=8, T=4, C2=16)
        mx.random.seed(42)
        latent = mx.random.normal((1, 8, 4, 16)).astype(mx.bfloat16)

        print("Decoding tiny audio latent (1, 8, 4, 16)...")
        mel = decoder.decode(latent)
        mx.synchronize()
        print(f"Output mel shape: {mel.shape}, dtype: {mel.dtype}")

        assert mel.ndim == 4
        assert mel.shape[0] == 1  # batch
        assert mel.shape[1] == 2  # stereo

        aggressive_cleanup()
        print("Audio VAE decode smoke test PASSED")


@skip_no_weights
class TestVocoderSmoke:
    """Test vocoder forward pass with real weights."""

    def test_mel_to_waveform(self):
        """Load vocoder and convert a random mel spectrogram to waveform."""
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.utils.weights import load_split_safetensors

        model = VocoderWithBWE()
        weights = load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.")
        model.load_weights(list(weights.items()))
        print(f"\nLoaded vocoder+BWE ({len(weights)} params)")

        # Tiny mel: (B=1, T=10, n_mels=128) for stereo input
        mx.random.seed(42)
        mel = mx.random.normal((1, 10, 128)).astype(mx.bfloat16)

        print("Running base vocoder on tiny mel (1, 10, 128)...")
        # Just test the base vocoder (not full BWE pipeline)
        wav = model._run_base_vocoder(mel)
        mx.synchronize()
        print(f"Output waveform shape: {wav.shape}, dtype: {wav.dtype}")

        assert wav.ndim == 3  # (B, T_audio, 2)
        assert wav.shape[0] == 1

        aggressive_cleanup()
        print("Vocoder smoke test PASSED")
