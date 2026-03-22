"""Full T2V generation with a real text prompt via Gemma 3.

Loads Gemma 3 12B 4-bit → extracts hidden states → feature extractor →
connector → DiT denoising → VAE decode → video + audio files.

Run with: pytest tests/test_generate_real_prompt.py -v -s
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

from ltx_core_mlx.utils.memory import aggressive_cleanup, get_memory_stats
from tests.conftest import MODEL_DIR

GEMMA_MODEL = "mlx-community/gemma-3-12b-it-4bit"

skip_no_weights = pytest.mark.skipif(MODEL_DIR is None, reason="q8 weights not found")


@skip_no_weights
class TestRealPromptGeneration:
    """Generate video+audio from a real text prompt."""

    def test_t2v_real_prompt(self):
        import shutil

        from ltx_core_mlx.conditioning.types.latent_cond import LatentState
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
        from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
        from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors, remap_audio_vae_keys
        from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
        from ltx_pipelines_mlx.utils.samplers import denoise_loop

        prompt = "A cat playing piano in a jazz club, cinematic lighting"
        F, H, W = 2, 2, 3
        num_video_tokens = F * H * W
        num_audio_tokens = 8
        seed = 42

        print(f"\n{'=' * 60}")
        print(f'T2V WITH REAL PROMPT: "{prompt}"')
        print(f"{'=' * 60}")

        # ---- 1. Text encoding via Gemma ----
        print("\n[1/6] Loading Gemma 3 12B 4-bit...")
        lm = GemmaLanguageModel()
        lm.load(GEMMA_MODEL)
        mx.synchronize()
        print(f"  Gemma loaded. Memory: {get_memory_stats()['active_gb']:.1f} GB")

        print("  Extracting hidden states...")
        all_hidden_states, attention_mask = lm.encode_all_layers(prompt, max_length=256)
        mx.synchronize()
        print(f"  Got {len(all_hidden_states)} layers, seq_len={all_hidden_states[0].shape[1]}")

        # Free Gemma
        del lm
        aggressive_cleanup()
        print(f"  Freed Gemma. Memory: {get_memory_stats()['active_gb']:.1f} GB")

        # ---- 2. Feature extraction + connector ----
        print("\n[2/6] Loading feature extractor + connector...")
        extractor = GemmaFeaturesExtractorV2()
        conn_weights = load_split_safetensors(MODEL_DIR / "connector.safetensors", prefix="connector.")
        extractor.connector.load_weights(list(conn_weights.items()))
        del conn_weights
        mx.synchronize()

        print("  Running feature extraction + connector...")
        video_embeds, audio_embeds = extractor(all_hidden_states, attention_mask=attention_mask)
        mx.synchronize()
        print(f"  Video embeds: {video_embeds.shape}")
        print(f"  Audio embeds: {audio_embeds.shape}")

        del extractor, all_hidden_states
        aggressive_cleanup()
        print(f"  Freed text encoder. Memory: {get_memory_stats()['active_gb']:.1f} GB")

        # ---- 3. Load DiT ----
        print("\n[3/6] Loading transformer (q8)...")
        config = LTXModelConfig()
        dit = LTXModel(config)
        t_weights = load_split_safetensors(MODEL_DIR / "transformer.safetensors", prefix="transformer.")
        apply_quantization(dit, t_weights)
        dit.load_weights(list(t_weights.items()), strict=True)
        del t_weights
        mx.synchronize()
        aggressive_cleanup()
        print(f"  Loaded. Memory: {get_memory_stats()['active_gb']:.1f} GB")

        # ---- 4. Denoise ----
        print("\n[4/6] Denoising (8 steps)...")
        mx.random.seed(seed)
        x0_model = X0Model(dit)

        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(num_audio_tokens)

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

        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=DISTILLED_SIGMAS,
            video_positions=video_positions,
            audio_positions=audio_positions,
            show_progress=True,
        )
        mx.synchronize()
        print(f"  Output: video={output.video_latent.shape}, audio={output.audio_latent.shape}")

        del dit, x0_model, video_embeds, audio_embeds
        aggressive_cleanup()

        # ---- 5. VAE decode ----
        print("\n[5/6] VAE decoding...")
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

        # ---- 6. Audio + save ----
        print("\n[6/6] Audio + save...")
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
        output_dir = Path(tempfile.mkdtemp(prefix="ltx_2_mlx_real_"))
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
            num_frames_out, h_out, w_out = frames.shape[0], frames.shape[1], frames.shape[2]
            scale_h, scale_w = max(64, h_out), max(64, w_out)
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
            print(f"  Video: {video_path.stat().st_size / 1024:.1f} KB ({num_frames_out}f, {w_out}x{h_out})")
        print(f"  Audio: {wav_path.stat().st_size / 1024:.1f} KB ({wav_int16.shape[0] / 16000:.2f}s)")

        print(f"\n{'=' * 60}")
        print("T2V WITH REAL PROMPT — PASSED")
        print(f"{'=' * 60}")
