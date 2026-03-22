"""Full generation test — DiT denoising + VAE decode + audio pipeline.

Uses random text embeddings (no Gemma) to test the complete pipeline
from noise to video+audio output files.

Run with: pytest tests/test_generate.py -v -s
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_unflatten

from ltx_core_mlx.utils.memory import aggressive_cleanup, get_memory_stats
from tests.conftest import MODEL_DIR

skip_no_weights = pytest.mark.skipif(MODEL_DIR is None, reason="q8 weights not found")


@skip_no_weights
class TestFullGeneration:
    """Generate a tiny video+audio from random noise with real weights."""

    def test_generate_tiny_video(self):
        import shutil
        import wave

        from ltx_core_mlx.conditioning.types.latent_cond import LatentState
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors, remap_audio_vae_keys
        from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
        from ltx_pipelines_mlx.utils.samplers import denoise_loop

        # ---- Parameters ----
        seed = 42
        num_video_tokens = 8  # very small
        num_audio_tokens = 4  # very small
        num_text_tokens = 4  # minimal text
        num_steps = 4  # half the full schedule

        print(f"\n{'=' * 60}")
        print("FULL GENERATION TEST (tiny, random text embeddings)")
        print(f"{'=' * 60}")

        # ---- 1. Load DiT ----
        print("\n[1/5] Loading transformer (q8)...")
        config = LTXModelConfig()
        dit = LTXModel(config)
        t_weights = load_split_safetensors(MODEL_DIR / "transformer.safetensors", prefix="transformer.")
        apply_quantization(dit, t_weights)
        dit.load_weights(list(t_weights.items()), strict=True)
        del t_weights
        mx.synchronize()
        aggressive_cleanup()
        print(f"  Loaded. Memory: {get_memory_stats()['active_gb']:.2f} GB")

        # ---- 2. Denoise ----
        print("\n[2/5] Denoising (4 steps, tiny tokens)...")
        mx.random.seed(seed)
        x0_model = X0Model(dit)

        video_noise = mx.random.normal((1, num_video_tokens, 128)).astype(mx.bfloat16)
        audio_noise = mx.random.normal((1, num_audio_tokens, 128)).astype(mx.bfloat16)

        video_state = LatentState(
            latent=video_noise,
            clean_latent=mx.zeros_like(video_noise),
            denoise_mask=mx.ones((1, num_video_tokens, 1), dtype=mx.bfloat16),
        )
        audio_state = LatentState(
            latent=audio_noise,
            clean_latent=mx.zeros_like(audio_noise),
            denoise_mask=mx.ones((1, num_audio_tokens, 1), dtype=mx.bfloat16),
        )

        # Random text embeddings (bypass Gemma)
        video_text = mx.random.normal((1, num_text_tokens, 4096)).astype(mx.bfloat16) * 0.01
        audio_text = mx.random.normal((1, num_text_tokens, 2048)).astype(mx.bfloat16) * 0.01

        sigmas = DISTILLED_SIGMAS[:num_steps]
        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_text,
            audio_text_embeds=audio_text,
            sigmas=sigmas,
            show_progress=True,
        )
        mx.synchronize()

        print(f"  Video tokens: {output.video_latent.shape}")
        print(f"  Audio tokens: {output.audio_latent.shape}")

        # Free transformer
        del dit, x0_model
        aggressive_cleanup()
        print(f"  Freed transformer. Memory: {get_memory_stats()['active_gb']:.2f} GB")

        # ---- 3. VAE decode video ----
        print("\n[3/5] VAE decoding video...")
        vae = VideoDecoder()
        v_weights = load_split_safetensors(MODEL_DIR / "vae_decoder.safetensors", prefix="vae_decoder.")
        vae.load_weights(list(v_weights.items()))
        del v_weights
        mx.synchronize()

        # Reshape tokens to spatial: (1, N, 128) -> (1, 128, F, H, W)
        # Use N=8 tokens as 2 frames x 2x2 spatial
        F, H, W = 2, 2, 2
        video_latent = output.video_latent.reshape(1, F, H, W, 128).transpose(0, 4, 1, 2, 3)
        pixels = vae.decode(video_latent)
        mx.synchronize()
        print(f"  Pixels: {pixels.shape} {pixels.dtype}")

        del vae
        aggressive_cleanup()

        # ---- 4. Audio decode + vocoder ----
        print("\n[4/5] Audio decode + vocoder...")
        audio_dec = AudioVAEDecoder()
        a_weights = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.decoder.")
        all_audio = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                a_weights[k] = v
        a_weights = remap_audio_vae_keys(a_weights)
        audio_dec.update(tree_unflatten(list(a_weights.items())))
        del a_weights

        # Reshape audio tokens: (1, 4, 128) -> (1, 8, 4, 16)
        audio_latent = output.audio_latent.reshape(1, num_audio_tokens, 8, 16).transpose(0, 2, 1, 3)
        mel = audio_dec.decode(audio_latent)
        mx.synchronize()
        print(f"  Mel: {mel.shape}")

        del audio_dec
        aggressive_cleanup()

        # Vocoder
        vocoder = VocoderWithBWE()
        voc_weights = load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.")
        vocoder.load_weights(list(voc_weights.items()))
        del voc_weights

        # Mel (1, 2, T, 64) -> concat -> (1, T, 128) for vocoder
        B_m, C_m, T_m, M_m = mel.shape
        mel_concat = mel.transpose(0, 2, 1, 3).reshape(B_m, T_m, C_m * M_m)
        wav_16k = vocoder._run_base_vocoder(mel_concat)  # (1, T_audio, 2)
        mx.synchronize()
        print(f"  Waveform 16kHz: {wav_16k.shape}")

        del vocoder
        aggressive_cleanup()

        # ---- 5. Save outputs ----
        print("\n[5/5] Saving outputs...")
        output_dir = Path(tempfile.mkdtemp(prefix="ltx_2_mlx_test_"))

        # Save audio WAV
        wav_path = output_dir / "audio.wav"
        wav_np = np.array(wav_16k[0], dtype=np.float32)
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767).astype(np.int16)
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(wav_int16.tobytes())

        # Save video frames as raw video via ffmpeg
        video_path = output_dir / "video.mp4"
        if shutil.which("ffmpeg"):
            # pixels: (1, C, F_out, H_out, W_out) -> frames
            pix = mx.clip(pixels[0], -1.0, 1.0)
            pix = ((pix + 1.0) * 127.5).astype(mx.uint8)
            # (C, F, H, W) -> (F, H, W, C)
            frames = pix.transpose(1, 2, 3, 0)
            mx.synchronize()

            num_frames_out = frames.shape[0]
            h_out, w_out = frames.shape[1], frames.shape[2]

            # Scale up to at least 64x64 for codec compatibility
            scale_h = max(64, h_out)
            scale_w = max(64, w_out)
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{w_out}x{h_out}",
                "-pix_fmt",
                "rgb24",
                "-r",
                "24",
                "-i",
                "-",
                "-i",
                str(wav_path),
                "-vf",
                f"scale={scale_w}:{scale_h}:flags=neighbor",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(video_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for i in range(num_frames_out):
                    frame = np.array(frames[i])
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"  ffmpeg warning: {stderr.decode()[-200:]}")

        print(f"\n  Output directory: {output_dir}")
        print(f"  Audio: {wav_path} ({wav_path.stat().st_size / 1024:.1f} KB)")
        if video_path.exists():
            print(f"  Video: {video_path} ({video_path.stat().st_size / 1024:.1f} KB)")
            print(f"  Resolution: {w_out}x{h_out}, Frames: {num_frames_out}")

        audio_duration = wav_int16.shape[0] / 16000
        print(f"  Audio duration: {audio_duration:.2f}s")

        assert wav_path.stat().st_size > 100, "WAV file too small"
        if video_path.exists():
            assert video_path.stat().st_size > 100, "Video file too small"

        print(f"\n{'=' * 60}")
        print("GENERATION TEST PASSED")
        print(f"{'=' * 60}")
