# CLAUDE.md — ltx-2-mlx

## Project Overview

Pure MLX port of [LTX-2](https://github.com/Lightricks/LTX-2/) (Lightricks) for Apple Silicon. Three-package monorepo mirroring the reference structure:

- **ltx-core-mlx** (`ltx_core_mlx`) — model library: DiT, VAE, audio, text encoder, conditioning
- **ltx-pipelines-mlx** (`ltx_pipelines_mlx`) — generation pipelines: T2V, I2V, retake, extend, keyframe, two-stage
- **ltx-trainer** (`ltx_trainer_mlx`) - ltx-2 training, democratized.

Loads pre-converted MLX weights from the [LTX-2.3 MLX collection on HuggingFace](https://huggingface.co/collections/dgrauet/ltx-23). Weight conversion is handled by [mlx-forge](https://github.com/dgrauet/mlx-forge).

---

## Tech Stack

- Python 3.11+, `uv` workspace (monorepo with `packages/*`)
- MLX (`mlx>=0.31.0`) — Apple Silicon ML framework (unified CPU/GPU memory)
- `mlx-lm>=0.31.0` — for Gemma 3 text encoder loading
- `safetensors`, `huggingface-hub`, `numpy`
- Linter/formatter: ruff

---

## Architecture

```
packages/
├── ltx-core-mlx/                          # ltx_core_mlx
│   └── src/ltx_core_mlx/
│       ├── components/                    # Shared pipeline components
│       │   ├── guiders.py                 # Guidance strategies
│       │   └── patchifiers.py             # VideoLatentPatchifier, AudioPatchifier
│       │
│       ├── conditioning/                  # Latent conditioning system
│       │   ├── mask_utils.py              # build/update/resolve attention masks
│       │   └── types/
│       │       ├── attention_strength_wrapper.py # Attention strength wrapping
│       │       ├── latent_cond.py         # LatentState, VideoConditionByLatentIndex
│       │       ├── keyframe_cond.py       # VideoConditionByKeyframeIndex
│       │       └── reference_video_cond.py # VideoConditionByReferenceLatent (IC-LoRA)
│       │
│       ├── guidance/                      # Guidance utilities
│       │   └── perturbations.py           # Noise perturbation strategies
│       │
│       ├── loader/                        # Weight loading & LoRA fusion
│       │   ├── fuse_loras.py              # LoRA weight fusion
│       │   ├── primitives.py              # Loading primitives
│       │   ├── sd_ops.py                  # Safetensors loading operations
│       │   └── sft_loader.py              # Split safetensors loader
│       │
│       ├── model/
│       │   ├── audio_vae/                 # Audio VAE + vocoder + BWE
│       │   │   ├── audio_vae.py           # AudioVAEDecoder, AudioResBlock, AudioAttnBlock
│       │   │   ├── encoder.py             # AudioVAEEncoder
│       │   │   ├── vocoder.py             # BigVGANVocoder, SnakeBeta, Activation1d
│       │   │   ├── bwe.py                 # VocoderWithBWE, HannSincResampler, MelSTFT
│       │   │   └── processor.py           # AudioProcessor (STFT + mel filterbank)
│       │   │
│       │   ├── transformer/               # Diffusion Transformer (DiT)
│       │   │   ├── model.py               # LTXModel, X0Model, LTXModelConfig
│       │   │   ├── transformer.py         # BasicAVTransformerBlock (joint audio+video)
│       │   │   ├── attention.py           # Multi-head attention + RoPE + per-head gating
│       │   │   ├── feed_forward.py        # Gated MLP blocks
│       │   │   ├── rope.py                # Rotary position embeddings (SPLIT type)
│       │   │   ├── adaln.py               # AdaLayerNormSingle (9-param)
│       │   │   └── timestep_embedding.py  # Sinusoidal + MLP timestep encoding
│       │   │
│       │   ├── upsampler/                 # Neural latent upscaler
│       │   │   └── model.py               # LatentUpsampler, SpatialRationalResampler
│       │   │
│       │   └── video_vae/                 # Video VAE
│       │       ├── video_vae.py           # VideoDecoder (streaming), VideoEncoder
│       │       ├── convolution.py         # Conv3dBlock (causal + reflect padding)
│       │       ├── resnet.py              # ResBlock3d, ResBlockStage
│       │       ├── sampling.py            # DepthToSpaceUpsample, pixel_shuffle_3d
│       │       ├── tiling.py              # Tiled VAE encoding/decoding
│       │       ├── normalization.py       # pixel_norm (RMS)
│       │       └── ops.py                 # PerChannelStatistics
│       │
│       ├── text_encoders/                 # Text encoding (Gemma 3)
│       │   └── gemma/
│       │       ├── embeddings_connector.py  # Embeddings1DConnector (RoPE + registers)
│       │       ├── feature_extractor.py     # GemmaFeaturesExtractorV2 (video/audio projections)
│       │       └── encoders/
│       │           ├── base_encoder.py      # Gemma 3 12B wrapper via mlx-lm
│       │           └── prompts/             # System prompt templates
│       │               ├── gemma_t2v_system_prompt.txt
│       │               └── gemma_i2v_system_prompt.txt
│       │
│       └── utils/
│           ├── positions.py   # compute_video_positions, compute_audio_positions
│           ├── weights.py     # load_split_safetensors, apply_quantization
│           ├── memory.py      # aggressive_cleanup, get_memory_stats
│           ├── image.py       # prepare_image_for_encoding
│           ├── video.py       # Video processing utilities
│           ├── audio.py       # Audio processing utilities
│           └── ffmpeg.py      # find_ffmpeg, probe_video_info
│
├── ltx-pipelines-mlx/                    # ltx_pipelines_mlx
│   └── src/ltx_pipelines_mlx/
│       ├── ti2vid_one_stage.py            # T2V/I2V: one-stage generation
│       ├── ti2vid_two_stages.py           # Two-stage: half res → upscale → refine
│       ├── ti2vid_two_stages_hq.py        # Two-stage HQ variant
│       ├── a2vid_two_stage.py             # Audio-to-video two-stage pipeline
│       ├── retake.py                      # Retake: regenerate a time segment
│       ├── extend.py                      # Extend: add frames before/after
│       ├── keyframe_interpolation.py      # Keyframe interpolation
│       ├── ic_lora.py                     # IC-LoRA reference-based generation
│       ├── scheduler.py                   # DISTILLED_SIGMAS, STAGE_2_SIGMAS
│       ├── cli.py                         # CLI entry point
│       └── utils/
│           ├── samplers.py                # Sampling utilities (Euler denoising)
│           ├── constants.py               # Pipeline constants
│           └── res2s.py                   # Second-stage resolution utilities
│
└── ltx-trainer/                           # ltx_trainer_mlx
    └── src/ltx_trainer_mlx/
        ├── trainer.py                     # Main training loop
        ├── config.py                      # Training configuration
        ├── config_display.py              # Config pretty-printing
        ├── datasets.py                    # Dataset loading and processing
        ├── model_loader.py                # Model loading for training
        ├── quantization.py                # Training-time quantization
        ├── timestep_samplers.py           # Timestep sampling strategies
        ├── captioning.py                  # Auto-captioning utilities
        ├── validation_sampler.py          # Validation sampling during training
        ├── gemma_8bit.py                  # 8-bit Gemma encoder for training
        ├── gpu_utils.py                   # GPU/Metal memory utilities
        ├── hf_hub_utils.py                # HuggingFace Hub integration
        ├── progress.py                    # Training progress tracking
        ├── video_utils.py                 # Video processing for training
        ├── utils.py                       # General training utilities
        └── training_strategies/           # Pluggable training strategies
            ├── base_strategy.py           # Base strategy interface
            ├── text_to_video.py           # T2V training strategy
            └── video_to_video.py          # V2V training strategy
```

---

## LTX-2.3 Model Architecture

- **Type**: Diffusion Transformer (DiT), 19B params, joint audio+video single-pass
- **Transformer**: 48 layers × 32 heads × 128-dim = 4096-dim (video), 32 heads × 64-dim = 2048-dim (audio)
- **VAE**: Temporal 8×, Spatial 32× compression → 128-channel latent
- **Text encoder**: Gemma 3 12B → dual projections (video 4096-dim, audio 2048-dim) via Embeddings1DConnector
- **Vocoder**: BigVGAN v2 with SnakeBeta activation (log-scale alpha/beta) + anti-aliased resampling
- **BWE**: Residual bandwidth extension (base 16kHz → Hann-sinc 3× resample → causal MelSTFT → BWE generator → 48kHz)
- **Distilled**: 8 steps (predefined sigma schedule), no classifier-free guidance

### Key Shapes

| Component | Input | Output |
|-----------|-------|--------|
| Text encoder | token_ids (1, 1024) | video_embeds (1, 1024, 4096), audio_embeds (1, 1024, 2048) |
| Transformer (video) | latent (B, F×H×W, 128) | velocity (B, F×H×W, 128) |
| Transformer (audio) | latent (B, T, 128) | velocity (B, T, 128) |
| Video VAE decoder | latent (B, 128, F', H', W') | pixels (B, 3, F, H, W) |
| Audio VAE decoder | latent (B, 8, T, 16) | mel (B, 2, T', 64) |
| Vocoder | mel (B, 2, T', 64) | waveform (B, 2, T_audio) @ 16kHz |
| BWE | waveform 16kHz | waveform 48kHz |
| Upsampler | latent (B, 128, F, H, W) | latent (B, 128, F, 2H, 2W) |

### Audio Token Count

Audio tokens per video: `round(num_pixel_frames / fps * 25)` where 25 = sample_rate(16000) / hop_length(160) / downsample_factor(4).

---

## Weight Format

Weights are pre-converted by [mlx-forge](https://github.com/dgrauet/mlx-forge) and hosted on HuggingFace. This package only **loads** weights — it never converts them.

### Available Variants

| Variant | HuggingFace | Size | Notes |
|---------|-------------|------|-------|
| Distilled bf16 | [dgrauet/ltx-2.3-mlx-distilled](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled) | ~42GB | Full precision, requires 64GB+ RAM |
| Distilled int8 | [dgrauet/ltx-2.3-mlx-distilled-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q8) | ~21GB | Recommended for 32GB+ |
| Distilled int4 | [dgrauet/ltx-2.3-mlx-distilled-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q4) | ~12GB | Lower quality, fits 16GB |
| Dev + distilled q8 | [dgrauet/ltx-2.3-mlx-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8) | ~26GB | Dev transformer + distilled LoRA for keyframe |

### MLX Layout Conventions

| Layer Type | PyTorch | MLX | Notes |
|-----------|---------|-----|-------|
| Linear | (O, I) | (O, I) | No transpose |
| Conv1d | (O, I, K) | (O, K, I) | Pre-converted by mlx-forge |
| Conv2d | (O, I, H, W) | (O, H, W, I) | Pre-converted |
| Conv3d | (O, I, D, H, W) | (O, D, H, W, I) | Pre-converted |
| ConvTranspose1d | (I, O, K) | (O, K, I) | Pre-converted by mlx-forge |
| Norm layers | (D,) | (D,) | No transpose |

**All weights must be in MLX format on disk.** If a weight file contains PyTorch-format tensors, fix it in mlx-forge — don't work around it here.

### Config Corrections

| Parameter | config.json | Actual | Evidence |
|-----------|-------------|--------|----------|
| `cross_attention_adaln` | `false` | **`true`** | Weights have 9 AdaLN params + `prompt_scale_shift_table` per block |

### Quantization

- Only `nn.Linear` inside `transformer_blocks` → int8 (group_size=64)
- Non-quantizable (must stay bf16): `adaln_single`, `proj_out`, `patchify_proj`, connectors, VAE, vocoder
- MLX can only quantize Linear and Embedding — never Conv layers

### Split Safetensors

| File | Prefix | Content |
|------|--------|---------|
| `transformer.safetensors` | `transformer.` | DiT blocks (quantized) |
| `connector.safetensors` | N/A | Text embeddings connectors |
| `vae_decoder.safetensors` | `vae_decoder.` | Video VAE decoder + per-channel stats |
| `vae_encoder.safetensors` | `vae_encoder.` | Video VAE encoder + per-channel stats |
| `audio_vae.safetensors` | `audio_vae.` | Audio VAE decoder + per-channel stats |
| `vocoder.safetensors` | `vocoder.` | Base vocoder + BWE generator + mel STFT |
| `spatial_upscaler_x2_v1_1.safetensors` | N/A | 2x spatial upsampler |
| `spatial_upscaler_x1_5_v1_0.safetensors` | N/A | 1.5x spatial upsampler |
| `temporal_upscaler_x2_v1_0.safetensors` | N/A | 2x temporal upsampler |

### Key Remapping (mlx-forge)

Applied during weight conversion, not at load time:

| PyTorch Key Pattern | MLX Key Pattern | Context |
|---------------------|-----------------|---------|
| `ff.net.0.proj.*` | `ff.proj_in.*` | Feed-forward input |
| `ff.net.2.*` | `ff.proj_out.*` | Feed-forward output |
| `attn.to_out.0.*` | `attn.to_out.*` | Main transformer attention (removes Sequential) |
| `_mean_of_means` | `mean_of_means` | Audio VAE stats (MLX `_` prefix = private) |
| `_std_of_means` | `std_of_means` | Audio VAE stats |

Note: Connector attention **keeps** Sequential wrapping (`to_out.0.*` stays).

---

## Critical Rules

### 1. Metal Memory Management (NON-NEGOTIABLE)

```python
from ltx_core_mlx.utils.memory import aggressive_cleanup
aggressive_cleanup()  # gc.collect() + mx.clear_cache()
```

Call between **every pipeline stage**. MLX Metal cache grows unbounded without explicit cleanup.

### 2. Streaming VAE Decode (NON-NEGOTIABLE)

Never decode all video frames in RAM. Stream frame-by-frame to ffmpeg:

```python
for i in range(num_frames):
    frame = decoder.decode_frame(latents[:, :, i:i+1])
    ffmpeg_proc.stdin.write(frame_to_bytes(frame))
    del frame
    if i % 8 == 0:
        aggressive_cleanup()
```

### 3. Reference Implementation is ltx-core

**ALWAYS** port from [ltx-core](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core) (Lightricks official), NOT from mlx-video.

Key reference paths:
- `packages/ltx-core/src/ltx_core/model/transformer/` — DiT architecture
- `packages/ltx-core/src/ltx_core/model/audio_vae/` — Audio VAE + vocoder + BWE
- `packages/ltx-core/src/ltx_core/model/video_vae/` — Video VAE
- `packages/ltx-core/src/ltx_core/conditioning/` — Conditioning system
- `packages/ltx-core/src/ltx_core/components/` — Schedulers, patchifiers, guiders
- `packages/ltx-pipelines/src/ltx_pipelines/` — Pipeline implementations

### 4. No Weight Conversion in This Package

Weight conversion is handled by [mlx-forge](https://github.com/dgrauet/mlx-forge). This package loads pre-converted weights only.

### 5. Positions Must Be in Pixel-Space

Video positions use pixel-space coordinates with causal fix, divided by fps:
- Temporal: `midpoint(max(0, i*8 - 7), i*8 + 1) / fps`
- Spatial: `h * 32 + 16`, `w * 32 + 16`

Audio positions use real-time seconds: `midpoint(max(0, (i-3)*4) * 0.01, max(0, (i-2)*4) * 0.01)`

Never use raw latent indices as positions.

### 6. Per-Token Timesteps for Conditioning

When conditioning (I2V, retake, extend), use per-token timesteps `sigma * denoise_mask`:
- X0Model denoising: `x0 = x_t - per_token_sigma * v` (preserved tokens get sigma=0 → x0=x_t)
- AdaLN: reshape per-token params as `(B, N, P, dim)` not `(B*N, P, dim)`

---

## Conditioning System

### Core Types
- `LatentState(latent, clean_latent, denoise_mask, positions?, attention_mask?)` — generation state
- `denoise_mask`: `1.0` = denoise (generate), `0.0` = preserve (keep clean)
- `positions`: (B, N, num_axes) pixel-space positions for RoPE
- `attention_mask`: (B, N, N) optional self-attention mask [0,1]

### Conditioning Items
- `VideoConditionByLatentIndex(frame_indices, clean_latent, strength)` — replace tokens at frame index (I2V)
- `VideoConditionByKeyframeIndex(indices, latents, positions, strength)` — append tokens (interpolation)
- `VideoConditionByReferenceLatent(latent, positions, downscale_factor, strength)` — append reference (IC-LoRA)
- `TemporalRegionMask(start_frame, end_frame)` — time-range masking (retake)

### Attention Mask System
- `mask_utils.build_attention_mask()` — block-structured (B, N+M, N+M) mask
- `mask_utils.update_attention_mask()` — incremental mask building for conditioning items
- Conditioning items call `update_attention_mask` when appending tokens

### Diffusion Loop
```python
# denoise_loop resolves positions/attention_mask from LatentState automatically
# Per-step: video_timesteps = sigma * denoise_mask (preserved regions get sigma=0)
# Per-step: x0 = apply_denoise_mask(x0, clean_latent, mask) → blend before Euler step
# Noising: noise_latent_state() blends clean*(1-mask) + noisy*mask
```

---

## Audio Pipeline

### Full Chain
```
Audio latent (B, 8, T, 16)
    → Audio VAE decoder (causal Conv2d + PixelNorm + AttnBlock) → mel (B, 2, T', 64)
    → BigVGAN v2 vocoder (SnakeBeta log-scale + anti-aliased) → waveform @ 16kHz
    → BWE (Hann-sinc 3× resample + causal MelSTFT + BigVGAN residual) → waveform @ 48kHz
```

### Key Implementation Details
- **SnakeBeta**: weights stored in log-scale, forward applies `exp(alpha)` and `exp(beta)`
- **Audio VAE Conv2d**: causal padding on height axis (time), reflect padding NOT used (zeros)
- **Audio VAE upsample**: drop first row after causal conv for temporal alignment
- **BWE resampler**: Hann-windowed sinc, 43 taps, rolloff=0.99 (NOT Kaiser)
- **BWE MelSTFT**: causal left-only padding (352, 0), NOT symmetric
- **BWE generator**: `apply_final_activation=False` (no tanh on residual)

---

## CLI Commands

Entry point: `uv run ltx-2-mlx <command>`. Available commands:

| Command | Pipeline | Description |
|---------|----------|-------------|
| `generate` | T2V / I2V | One-stage or two-stage generation (auto-selects based on `--image`) |
| `keyframe` | Keyframe interpolation | Two-stage interpolation between start/end frames |
| `retake` | Retake | Regenerate a time segment of an existing video |
| `extend` | Extend | Add frames before or after an existing video |
| `a2v` | Audio-to-video | Two-stage audio-conditioned generation |
| `enhance` | Prompt enhancement | Enhance a text prompt using Gemma (no video generation) |
| `info` | Model info | Show model configuration and memory estimates |

Common flags: `--model`, `--prompt`, `--output`, `--height`, `--width`, `--frames`, `--seed`, `--steps`, `--cfg-scale`, `--stg-scale`, `--enhance-prompt`.

---

## Guidance System (STG / CFG / Modality)

The non-distilled (dev) model uses multi-modal guidance with up to 4 forward passes per step:

| Pass | Purpose | Controlled By |
|------|---------|--------------|
| Conditioned | Normal generation | Always runs |
| Unconditional | CFG (classifier-free guidance) | `cfg_scale != 1.0` |
| Perturbed | STG (spatio-temporal guidance) | `stg_scale != 0.0` |
| Modality-isolated | Cross-modal guidance | `modality_scale != 1.0` |

Default reference params for keyframe: `cfg_scale=3.0`, `stg_scale=1.0`, `stg_blocks=[28]`, `rescale_scale=0.7`, `modality_scale=3.0`.

**Memory impact**: Each extra pass doubles/triples/quadruples memory. On 32GB Mac with dev model at 480x704: CFG-only supports 33 frames, full guidance (4 passes) supports ~17 frames.

**STG perturbation masks**: Self-attention masks are 4D `(B,1,1,1)` for use inside attention where tensors are `(B,H,N,D)`. Cross-modal masks (A2V/V2A) are 3D `(B,1,1)` for use outside attention where outputs are `(B,N,dim)`. Mixing these up causes silent shape corruption via broadcasting.

---

## Keyframe Interpolation Pipeline

Two-stage pipeline requiring the dev (non-distilled) model + CFG. The distilled model hallucinates during interpolation.

### Stage 1: Half Resolution + CFG
1. Compute half-res latent dims: `H_half = (height//2) // 32`, `W_half = (width//2) // 32`
2. Encode keyframes at VAE-compatible resolution: `H_half * 32` x `W_half * 32`
3. Create empty LatentState → apply `VideoConditionByKeyframeIndex` → noise (order matters!)
4. Denoise with dev model + CFG (20 steps, dynamic schedule)

### Stage 2: Upscale + Refine
1. Denormalize latent → neural upsampler (2x spatial) → re-normalize (using VAE encoder stats)
2. Fuse distilled LoRA into dev model
3. Re-encode keyframes at upscaled resolution, apply conditioning
4. Denoise with distilled schedule (3 steps)

### Key Files
- `keyframe_interpolation.py` — `KeyframeInterpolationPipeline` (extends `TwoStagePipeline`)
- `conditioning/types/keyframe_cond.py` — `VideoConditionByKeyframeIndex` (appends tokens, builds attention mask)
- `model/upsampler/model.py` — `LatentUpsampler` (Conv3d + PixelShuffle2D)

---

## Conventions

- Python 3.11+
- Mandatory type hints on all functions
- Google-style docstrings
- ruff for formatting/linting
- Tests in `tests/` using pytest
- Conventional commits (feat:, fix:, docs:, refactor:)
- Package imports: `ltx_core_mlx.*` for core, `ltx_pipelines_mlx.*` for pipelines

---

## Resources

- **ltx-core**: [GitHub](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core)
- **ltx-pipelines**: [GitHub](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines)
- **MLX**: [Docs](https://ml-explore.github.io/mlx/) · [GitHub](https://github.com/ml-explore/mlx)
- **mlx-forge**: [GitHub](https://github.com/dgrauet/mlx-forge) — weight conversion
- **Pre-converted weights**: [HuggingFace collection](https://huggingface.co/collections/dgrauet/ltx-23)
