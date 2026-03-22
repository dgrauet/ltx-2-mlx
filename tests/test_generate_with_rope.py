"""Generation test with RoPE positions and full 8-step distilled schedule.

Uses random text embeddings but real weights + positions.
Tests a slightly larger generation than the tiny test.

Run with: pytest tests/test_generate_with_rope.py -v -s
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_unflatten

from ltx_core_mlx.utils.memory import aggressive_cleanup
from tests.conftest import MODEL_DIR

skip_no_weights = pytest.mark.skipif(MODEL_DIR is None, reason="q8 weights not found")


@skip_no_weights
class TestGenerateWithRoPE:
    """Generate with RoPE positions and full 8-step schedule."""

    def test_generate_small_video(self):
        import shutil

        from ltx_core_mlx.conditioning.types.latent_cond import LatentState
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors, remap_audio_vae_keys
        from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
        from ltx_pipelines_mlx.utils.samplers import denoise_loop

        # Small but realistic latent dimensions
        # 2 frames x 2x3 spatial = 12 video tokens
        F, H, W = 2, 2, 3
        num_video_tokens = F * H * W  # 12
        num_audio_tokens = 8
        num_text_tokens = 4
        seed = 42

        print(f"\n{'=' * 60}")
        print(f"GENERATION WITH RoPE ({F}f x {H}h x {W}w, 8 steps)")
        print(f"{'=' * 60}")

        # 1. Load transformer
        print("\n[1/5] Loading transformer...")
        config = LTXModelConfig()
        dit = LTXModel(config)
        t_weights = load_split_safetensors(MODEL_DIR / "transformer.safetensors", prefix="transformer.")
        apply_quantization(dit, t_weights)
        dit.load_weights(list(t_weights.items()), strict=True)
        del t_weights
        mx.synchronize()
        aggressive_cleanup()

        # 2. Compute positions
        print("[2/5] Computing positions...")
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(num_audio_tokens)
        print(f"  Video positions: {video_positions.shape}")
        print(f"  Audio positions: {audio_positions.shape}")

        # 3. Denoise (full 8-step schedule)
        print("[3/5] Denoising (8 steps)...")
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

        video_text = mx.random.normal((1, num_text_tokens, 4096)).astype(mx.bfloat16) * 0.01
        audio_text = mx.random.normal((1, num_text_tokens, 2048)).astype(mx.bfloat16) * 0.01

        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_text,
            audio_text_embeds=audio_text,
            sigmas=DISTILLED_SIGMAS,
            video_positions=video_positions,
            audio_positions=audio_positions,
            show_progress=True,
        )
        mx.synchronize()
        print(f"  Output: video={output.video_latent.shape}, audio={output.audio_latent.shape}")

        del dit, x0_model
        aggressive_cleanup()

        # 4. VAE decode
        print("[4/5] VAE decoding...")
        vae = VideoDecoder()
        v_weights = load_split_safetensors(MODEL_DIR / "vae_decoder.safetensors", prefix="vae_decoder.")
        vae.load_weights(list(v_weights.items()))
        del v_weights

        video_latent = output.video_latent.reshape(1, F, H, W, 128).transpose(0, 4, 1, 2, 3)
        pixels = vae.decode(video_latent)
        mx.synchronize()
        print(f"  Pixels: {pixels.shape}")
        del vae
        aggressive_cleanup()

        # 5. Audio + save
        print("[5/5] Audio + save...")
        audio_dec = AudioVAEDecoder()
        a_weights = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.decoder.")
        all_audio = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                a_weights[k] = v
        a_weights = remap_audio_vae_keys(a_weights)
        audio_dec.update(tree_unflatten(list(a_weights.items())))
        del a_weights

        audio_latent = output.audio_latent.reshape(1, num_audio_tokens, 8, 16).transpose(0, 2, 1, 3)
        mel = audio_dec.decode(audio_latent)
        mx.synchronize()
        del audio_dec
        aggressive_cleanup()

        vocoder = VocoderWithBWE()
        voc_weights = load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.")
        vocoder.load_weights(list(voc_weights.items()))
        del voc_weights

        B_m, C_m, T_m, M_m = mel.shape
        mel_concat = mel.transpose(0, 2, 1, 3).reshape(B_m, T_m, C_m * M_m)
        wav_16k = vocoder._run_base_vocoder(mel_concat)
        mx.synchronize()
        del vocoder
        aggressive_cleanup()

        # Save
        output_dir = Path(tempfile.mkdtemp(prefix="ltx_2_mlx_rope_"))
        wav_path = output_dir / "audio.wav"
        wav_np = np.array(wav_16k[0], dtype=np.float32)
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767).astype(np.int16)
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(wav_int16.tobytes())

        video_path = output_dir / "video.mp4"
        if shutil.which("ffmpeg"):
            pix = mx.clip(pixels[0], -1.0, 1.0)
            pix = ((pix + 1.0) * 127.5).astype(mx.uint8)
            frames = pix.transpose(1, 2, 3, 0)
            mx.synchronize()
            num_frames_out = frames.shape[0]
            h_out, w_out = frames.shape[1], frames.shape[2]
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
                    proc.stdin.write(np.array(frames[i]).tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.communicate()

        print(f"\n  Output: {output_dir}")
        if video_path.exists():
            print(f"  Video: {video_path.stat().st_size / 1024:.1f} KB ({num_frames_out} frames, {w_out}x{h_out})")
        print(f"  Audio: {wav_path.stat().st_size / 1024:.1f} KB ({wav_int16.shape[0] / 16000:.2f}s)")
        print(f"\n{'=' * 60}")
        print("GENERATION WITH RoPE PASSED")
        print(f"{'=' * 60}")
