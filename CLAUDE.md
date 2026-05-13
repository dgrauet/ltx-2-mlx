# CLAUDE.md — ltx-2-mlx

## Project Overview

Pure MLX port of [LTX-2](https://github.com/Lightricks/LTX-2/) (Lightricks) for Apple Silicon. Three-package monorepo mirroring the reference structure:

- **ltx-core-mlx** (`ltx_core_mlx`) — model library: DiT, VAE, audio, text encoder, conditioning
- **ltx-pipelines-mlx** (`ltx_pipelines_mlx`) — generation pipelines: T2V, I2V, retake, extend, keyframe, IC-LoRA, two-stage
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
│       ├── _base.py                       # BasePipeline + ImageToVideoPipeline (private base classes)
│       ├── ti2vid_one_stage.py            # TI2VidOneStagePipeline (dev one-stage + CFG, full target res)
│       ├── ti2vid_two_stages.py           # Two-stage: half res → upscale → refine
│       ├── ti2vid_two_stages_hq.py        # Two-stage HQ variant
│       ├── a2vid_two_stage.py             # Audio-to-video two-stage pipeline
│       ├── retake.py                      # RetakePipeline: regenerate a time segment + extend (append/prepend)
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
| bf16 | [dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx) | ~42GB | Full precision, fits 32GB with `--low-ram`, 64GB+ otherwise. For >4s HD or 1080p: stack with `--tile-spatial 2`. |
| int8 | [dgrauet/ltx-2.3-mlx-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8) | ~26GB | Recommended for 32GB+; fits 16GB with `--low-ram`. Stack with tiling for HD on Mac Studio. |
| int4 | [dgrauet/ltx-2.3-mlx-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) | ~12GB | Lower quality, fits 16GB |

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

Full pipeline ↔ option matrix lives at [docs/PIPELINES.md](docs/PIPELINES.md). Quick reference:

Entry point: `uv run ltx-2-mlx <command>`. Available commands:

| Command | Pipeline | Tier | Description |
|---------|----------|------|-------------|
| `generate` | T2V / I2V (mode flag required) | Stable | `--one-stage` (dev+CFG @ target), `--two-stage` (dev+CFG+upscale, recommended), `--two-stages-hq` (res_2s+CFG+upscale), `--distilled` (distilled+upscale, fastest). `--image` for I2V on any mode. |
| `keyframe` | Keyframe interpolation | Stable | Two-stage interpolation between start/end frames |
| `ic-lora` | IC-LoRA | Stable | Two-stage generation with control video conditioning (depth, canny, pose, motion tracks) |
| `hdr-ic-lora` | HDR IC-LoRA | Stable | Two-stage HDR generation via IC-LoRA + LogC3 inverse (saves SDR mp4 + linear-HDR `.npz`) |
| `a2v` | Audio-to-video | Beta | Two-stage audio-conditioned generation (Euler + CFG). Sync quality depends on prompt-audio alignment. |
| `retake` | Retake | Beta | Regenerate a time segment of an existing video (dev model + CFG) |
| `extend` | Extend | Beta | Add frames before or after an existing video (dev model + CFG) |
| `lipdub` | LipDub | Experimental | Lip-dub a reference video → re-sync visuals to source audio. Output audio is a VAE+vocoder reconstruction (audible artifacts on rich music). Uses pre-1.0 LipDub IC-LoRA. |
| `enhance` | Prompt enhancement | Stable | Enhance a text prompt using Gemma (no video generation) |
| `info` | Model info | Stable | Show model configuration and memory estimates |
| `train` | Training | Stable | Train a LoRA or full model from YAML config (requires ltx-trainer-mlx) |
| `preprocess` | Data preprocessing | Stable | Encode raw videos into latents + conditions for training |

All pipelines except one-stage T2V/I2V use the dev model with CFG guidance. Common flags: `--model`, `--prompt`, `--output`, `--seed`, `--quiet`. Tier semantics + promotion criteria live in [docs/PIPELINE_MATURITY.md](docs/PIPELINE_MATURITY.md).

### Low-RAM Example

```bash
# bf16 inference on a 32 GB Mac via block streaming
ltx-2-mlx generate \
  --model dgrauet/ltx-2.3-mlx \
  --prompt "a fox in the forest" \
  --low-ram \
  -H 480 -W 704 -f 33 -o fox.mp4

# q8 inference fits 16 GB Macs
ltx-2-mlx generate \
  --model dgrauet/ltx-2.3-mlx-q8 \
  --prompt "a fox in the forest" \
  --low-ram -o fox.mp4
```

`--low-ram` is supported on `generate` (one-stage / `--two-stage` / `--two-stages-hq`), `a2v`, `keyframe`, and `ic-lora`. Bind-time LoRA fusion handles ic-lora's control LoRAs and custom `--distilled-lora-strength`. The `generate --lora` flag is still incompatible (use a pre-fused safetensors via mlx-forge). See `## Block Streaming` below for details.

### IC-LoRA Example

```bash
# Union Control (depth, canny, pose)
ltx-2-mlx ic-lora \
  --prompt "a person walking" \
  --lora Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control 1.0 \
  --video-conditioning depth_map.mp4 1.0 \
  -o output.mp4

# Motion Track Control
ltx-2-mlx ic-lora \
  --prompt "particles moving" \
  --lora Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control 1.0 \
  --video-conditioning tracks.mp4 1.0 \
  -o output.mp4
```

Flags: `--lora PATH STRENGTH` (repeatable, supports HF repo IDs), `--video-conditioning PATH STRENGTH` (repeatable), `--conditioning-strength`, `--skip-stage-2`, `--image`.

#### Static-scene I2V recipe (preserve identity)

The `generate` modes (`--one-stage`, `--two-stage`, `--two-stages-hq`, `--distilled`) don't preserve the input image's identity over 4 sec with descriptive prompts — even with multi-anchor I2V (`--image PATH 0 1.0 --image PATH 96 1.0`) or STG=1.0. The model uses the anchor as initialization but generates freely afterwards, drifting toward the prompt's distribution. Same behavior upstream.

The **upstream-iso pattern** for static-scene identity preservation is `ic-lora` with a control video replicated from the source image. Validated on Phoenix Q15 (1280×704×97 in 15 min on M2 Pro 32 GB, identity preserved throughout):

```bash
# 1. Generate canny control video by replicating the source image
ffmpeg -y -loop 1 -i input_image.jpg \
  -vf "scale=W:H:force_original_aspect_ratio=increase,crop=W:H,edgedetect=mode=canny:low=0.1:high=0.4,format=yuv420p" \
  -frames:v N -r 24 -c:v libx264 -preset veryfast -crf 18 \
  control_canny.mp4

# 2. ic-lora with Union Control + canny + I2V anchor
ltx-2-mlx ic-lora \
  -p "your cinematic prompt..." \
  --lora Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control 1.0 \
  --video-conditioning control_canny.mp4 1.0 \
  --image input_image.jpg \
  --low-ram -W W -H H -f N --seed ... \
  -o output.mp4
```

Why it works: `VideoConditionByReferenceLatent` (used by ic-lora) adds conditioning tokens **frame-by-frame** rather than just at the anchor positions. The replicated canny edges give the model a stable spatial structure throughout the clip, and the Union Control LoRA is trained to respect that structure. Faster than `--two-stage` modes (~4× shorter wall-clock) because distilled defaults (8+3 steps, no CFG = 1 forward/step).

Alternatives via Union Control LoRA: depth maps (need external depth model like Depth-Anything-V2) or pose maps (OpenPose). Canny edges work for general scenes and require only ffmpeg.

### HDR IC-LoRA Example

```bash
# V2V HDR — upgrade an existing SDR video to linear HDR
ltx-2-mlx hdr-ic-lora \
  --prompt "cinematic golden hour" \
  --lora Lightricks/LTX-2.3-22b-IC-LoRA-HDR 1.0 \
  --video-conditioning source_sdr.mp4 1.0 \
  --low-ram -o out.mp4

# T2V HDR — pure text-to-video with HDR output (no conditioning)
ltx-2-mlx hdr-ic-lora \
  --prompt "a sunset over the ocean, vivid HDR" \
  --lora Lightricks/LTX-2.3-22b-IC-LoRA-HDR 1.0 \
  --low-ram -o out.mp4
```

Outputs both `out.mp4` (SDR preview, tonemapped) and `out.hdr.npz` (float32 `(F, H, W, 3)` linear HDR tensor for EXR/TIFF conversion). Auto-detects the LoRA's HDR transform (`logc3`) and `reference_downscale_factor` from safetensors metadata. Flags identical to `ic-lora`, with `--video-conditioning` made optional (matches upstream's empty-list path → pure T2V HDR mode).

### Two-Stage Example

```bash
# Two-stage with Euler sampler (auto-selects q8 model)
ltx-2-mlx generate \
  --prompt "a scene description" \
  --two-stage -o output.mp4

# HQ with res_2s second-order sampler (higher quality, ~2x slower)
ltx-2-mlx generate \
  --prompt "a scene description" \
  --two-stages-hq -o output.mp4

# With I2V conditioning
ltx-2-mlx generate \
  --prompt "animate this" \
  --two-stage --image photo.jpg -o output.mp4
```

Flags: `--two-stage` (Euler), `--two-stages-hq` (res_2s), `--cfg-scale` (default 3.0), `--stg-scale` (default 0.0), `--stage1-steps` (default 30 standard, 15 HQ), `--stage2-steps` (default 3), `--image`.

### Audio-to-Video Example

```bash
# A2V with reference image
ltx-2-mlx a2v \
  --prompt "a singer performing" \
  --audio music.wav --image photo.jpg -o output.mp4

# A2V HQ (res_2s sampler)
ltx-2-mlx a2v \
  --prompt "a singer performing" \
  --audio music.wav --two-stages-hq -o output.mp4
```

Flags: `--audio` (required), `--image` (optional I2V), `--two-stages-hq` (res_2s), `--cfg-scale`, `--stg-scale`, `--stage1-steps` (default 30 standard, 15 HQ), `--fps`.

### Retake / Extend Example

```bash
# Retake: regenerate latent frames 2-5
ltx-2-mlx retake \
  --prompt "a different action" \
  --video source.mp4 --start 2 --end 5 -o retake.mp4

# Extend: add 4 latent frames after
ltx-2-mlx extend \
  --prompt "continue the scene" \
  --video source.mp4 --extend-frames 4 -o extended.mp4
```

Flags: `--steps` (default 30), `--cfg-scale` (default 3.0), `--stg-scale` (default 0.0), `--no-regen-audio` (retake only).

### Training Example

```bash
# 1. Preprocess videos into latents + conditions
ltx-2-mlx preprocess \
  --videos ./my_training_videos \
  --captions ./my_captions \
  --model dgrauet/ltx-2.3-mlx-q8 \
  -o ./preprocessed_data

# 2. Train LoRA from YAML config
ltx-2-mlx train --config packages/ltx-trainer/configs/lora_t2v.yaml
```

Flags for `preprocess`: `--height`, `--width` (resize, must be divisible by 32), `--max-frames` (default 97), `--captions` (directory with .txt files matching video stems), `--caption-ext`.

Flags for `train`: `--config` (required, path to YAML config). See `packages/ltx-trainer/configs/` for examples.

---

## Guidance System (STG / CFG / Modality)

The non-distilled (dev) model uses multi-modal guidance with up to 4 forward passes per step:

| Pass | Purpose | Controlled By |
|------|---------|--------------|
| Conditioned | Normal generation | Always runs |
| Unconditional | CFG (classifier-free guidance) | `cfg_scale != 1.0` |
| Perturbed | STG (spatio-temporal guidance) | `stg_scale != 0.0` |
| Modality-isolated | Cross-modal guidance | `modality_scale != 1.0` |

Default reference params (LTX_2_3_PARAMS): `cfg_scale=3.0`, `stg_scale=1.0`, `stg_blocks=[28]`, `rescale_scale=0.7`, `modality_scale=3.0`. Audio: `cfg_scale=7.0`.
HQ params (LTX_2_3_HQ_PARAMS): `cfg_scale=3.0`, `stg_scale=0.0`, `stg_blocks=[]`, `rescale_scale=0.45`. Audio: `cfg_scale=7.0`, `rescale_scale=1.0`.

**Default `stg_scale=0.0`**: STG requires a 3rd forward pass per step. On 32GB Mac, this causes OOM for videos longer than ~33 frames at 480x704. All pipelines default to `stg_scale=0.0` (CFG-only) for 32GB compatibility. Use `--stg-scale 1.0` for short videos only.

**Memory impact**: Each extra pass doubles/triples/quadruples memory. On 32GB Mac with dev model at 480x704: CFG-only supports ~97 frames at half-res (two-stage), full guidance (4 passes) supports ~17 frames.

**STG perturbation masks**: Self-attention masks are 4D `(B,1,1,1)` for use inside attention where tensors are `(B,H,N,D)`. Cross-modal masks (A2V/V2A) are 3D `(B,1,1)` for use outside attention where outputs are `(B,N,dim)`. Mixing these up causes silent shape corruption via broadcasting.

---

## Keyframe Interpolation Pipeline

Two-stage pipeline requiring the dev (non-distilled) model + CFG. The distilled model hallucinates during interpolation.

### Stage 1: Half Resolution + CFG
1. Compute half-res latent dims: `H_half = (height//2) // 32`, `W_half = (width//2) // 32`
2. Encode keyframes at VAE-compatible resolution: `H_half * 32` x `W_half * 32`
3. Create empty LatentState → apply `VideoConditionByKeyframeIndex` → noise (order matters!)
4. Denoise with dev model + CFG (30 steps, dynamic schedule)

### Stage 2: Upscale + Refine
1. Denormalize latent → neural upsampler (2x spatial) → re-normalize (using VAE encoder stats)
2. Fuse distilled LoRA into dev model
3. Re-encode keyframes at upscaled resolution, apply conditioning
4. Denoise with distilled schedule (3 steps)

### Key Files
- `keyframe_interpolation.py` — `KeyframeInterpolationPipeline` (extends `TI2VidTwoStagesPipeline`)
- `conditioning/types/keyframe_cond.py` — `VideoConditionByKeyframeIndex` (appends tokens, builds attention mask)
- `model/upsampler/model.py` — `LatentUpsampler` (Conv3d + PixelShuffle2D)

---

## Two-Stage Pipeline (T2V / I2V)

Two-stage pipeline for higher-resolution generation. Requires the dev model + distilled LoRA (`dgrauet/ltx-2.3-mlx-q8`).

### Architecture (matching reference)

- **Stage 1**: Dev model + CFG guidance at half resolution
  - `--two-stage`: Euler sampler (`guided_denoise_loop`)
  - `--two-stages-hq`: res_2s second-order sampler (`res2s_denoise_loop` with guidance)
  - Dynamic sigma schedule via `ltx2_schedule` (default 30 steps standard, 15 HQ)
  - Optional I2V conditioning (re-encoded at half-res)
- **Stage 2**: Dev + distilled LoRA fused, simple Euler (no CFG)
  - `STAGE_2_SIGMAS` (default 3 steps)
  - I2V conditioning re-encoded at full resolution
  - Denormalize → neural upsampler 2x → re-normalize before Stage 2

### Critical Implementation Details

- **Dev model required**: The distilled model produces flat/low-quality output at half resolution without CFG. Two-stage always uses the dev model.
- **Upsampler denorm/renorm**: Same as IC-LoRA/keyframe — the neural upsampler operates in un-normalized latent space. Without denorm/renorm, Stage 2 produces grid artifacts.
- **Stage 2 dims from upscaled shape**: `H_full = H_half * 2`, not `compute_video_latent_shape(height)`, to avoid RoPE shape mismatch.
- **Decoders loaded on-demand**: VAE decoder + audio + vocoder loaded in `generate_and_save()` after freeing DiT, keeping peak memory under 32GB.
- **Text encoding before DiT**: In low_memory mode, Gemma is loaded → encode prompt + negative prompt → free Gemma → load DiT. Both positive and negative embeddings must be materialized before freeing.
- **res_2s + guidance**: `res2s_denoise_loop` accepts optional `video_guider_factory`/`audio_guider_factory` for CFG/STG/modality guidance. Each res_2s step does 2 model evaluations (substep + step), each with full guidance passes.
- **Memory budget (32GB Mac)**: 33 frames at 480x704 with CFG-only (2 forward passes per step). STG adds a 3rd pass and may not fit.

### Key Files
- `ti2vid_two_stages.py` — `TI2VidTwoStagesPipeline` (Euler + CFG, extends `BasePipeline`)
- `ti2vid_two_stages_hq.py` — `TI2VidTwoStagesHQPipeline` (res_2s + CFG, extends `TI2VidTwoStagesPipeline`)
- `utils/samplers.py` — `res2s_denoise_loop` (with guidance support), `guided_denoise_loop`
- `scheduler.py` — `ltx2_schedule`, `STAGE_2_SIGMAS`

### TeaCache (opt-in stage 1 acceleration)

Timestep-aware residual caching (Liu et al., *Timestep Embedding Aware Cache*) for `TI2VidTwoStagesPipeline.generate_two_stage`. Engine lives in `mlx-arsenal>=0.2.4` (`TeaCacheController`); LTX-2-specific calibrated coefficients + threshold live in `ti2vid_two_stages.py` (`LTX2_TEACACHE_COEFFICIENTS`, `LTX2_TEACACHE_THRESH`).

```python
pipeline.generate_and_save(
    prompt=...,
    enable_teacache=True,           # default False
    teacache_thresh=0.5,            # optional override
)
```

Equivalent CLI flag (works on both `--two-stage` and `--two-stages-hq`):

```bash
ltx-2-mlx generate --prompt "..." --two-stage --enable-teacache -o out.mp4
ltx-2-mlx generate --prompt "..." --two-stages-hq --enable-teacache --teacache-thresh 1.0 -o out.mp4
```

The HQ path uses the res_2s sampler, which does two model evaluations per outer step (stage 1 at `sigma`, stage 2 at the substep after SDE noise injection). The TeaCache decision is made **once per outer step** on stage 1's gate signal; on skip both stages reuse cached residuals via `block_stack_override`. Cache payload shape: `{"stage1": {cond: (v,a), uncond: (v,a), ...}, "stage2": {...}}`.

Decision per step is made on **block 0's modulated input** of the conditioned pass; on skip, the entire transformer block stack is bypassed (head + prelude still run). With CFG enabled (default), residuals are cached as a per-pass dict (`{"cond": (v,a), "uncond": (v,a)}`) so all guidance passes skip together.

**Calibration**: 5-prompt × 30-step run on a fresh host (commit `245fd5f`). The robust fitter (`scripts/fit_teacache_poly.py`) picked degree 1 — higher degrees are non-monotone on the observed delta range. Polynomial: `y = 1.364 * x + 0.409`.

**Empirical speedup** (seed 81647281, 480×704×97, MLX bf16 q8):
- Baseline: 1374s
- TeaCache (thresh=0.5): 942s — **1.46x speedup, 31% time saved**, visually validated.

**HQ (`--two-stages-hq`) speedup** (seed 81647281, 384×576×65, MLX bf16 q8,
HQ-specific calibrated coefficients in `ti2vid_two_stages_hq.py`):
- Baseline: 1370s
- TeaCache (thresh=1.0): 768s — **1.78x speedup, 44% time saved**

HQ outperforms Euler in raw speedup because:
- Pearson correlation between input and output L1 deltas is 0.62 on
  res_2s (vs 0.41 on Euler) — the polynomial is more predictive.
- res_2s does 2 forwards/step, so each skipped step saves ~2x what an
  Euler skip saves.
- Threshold 1.0 lands at the "cliff" in HQ skip-rate-vs-threshold; ~50%
  of interior steps skip in practice.

Calibration is **scheduler-specific** — Euler coefficients (calibrated
on `guided_denoise_loop`) produce ~0% skip on `res2s_denoise_loop` and
vice versa. Use `scripts/calibrate_teacache.py --two-stages-hq` to recalibrate
res_2s, and edit `LTX2_HQ_TEACACHE_COEFFICIENTS` /
`LTX2_HQ_TEACACHE_THRESH` in `ti2vid_two_stages_hq.py`.

**Tuning**: thresh 0.5 is the conservative default. Push higher for more skip / more speed, with quality risk:
- thresh 1.0 → ~55% skip, ~2× speedup expected
- thresh 1.5 → ~69% skip, ~3× expected, quality drift visible

LTX-2 stage 1 has weak per-step input/output L1 correlation (Pearson 0.41) and high mean output drift (~0.56), which is why thresholds are nettement higher than upstream DiTs (HunyuanVideo 0.15, Flux 0.4).

**Key files**:
- `mlx_arsenal.diffusion.TeaCacheController` (engine)
- `ti2vid_two_stages.py:LTX2_TEACACHE_COEFFICIENTS` (calibration constants)
- `scripts/calibrate_teacache.py` (calibration runner — saves raw deltas in JSON)
- `scripts/fit_teacache_poly.py` (offline robust polyfit; tries deg 1-N, picks lowest stable)
- Hooks in `transformer/model.py` (`tap` + `block_stack_override` on `LTXModel.__call__`)
- `samplers.py:guided_denoise_loop` (`teacache=` kwarg with per-pass `_run_pass` dispatcher)

**Re-calibration**: run on fresh host. ~22 min/prompt × 5 prompts ≈ 1h45. Then `python -m ltx_pipelines_mlx.scripts.fit_teacache_poly <calibration.json>` to validate stability and produce a paste-ready snippet.

---

## IC-LoRA Pipeline

Two-stage pipeline for control-conditioned video generation using official Lightricks IC-LoRAs.
Uses the distilled model (no CFG) with LoRA fused for Stage 1 only.

### Supported IC-LoRAs

| LoRA | HuggingFace | Control Types | ref_downscale |
|------|-------------|---------------|---------------|
| Union Control | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) | Canny edges, depth maps, human pose | 2 |
| Motion Track | [Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) | Colored spline trajectories (BGR) | 2 |

### Pipeline Flow

1. **LoRA resolution**: `_resolve_lora_path()` downloads from HuggingFace if needed
2. **Metadata**: `reference_downscale_factor` read from LoRA safetensors metadata
3. **Text encoding**: Gemma + connector, freed before loading DiT
4. **Stage 1**: Load DiT + fuse LoRA + VAE-encode control video at ref resolution + denoise (8 steps)
5. **Upscale**: Denormalize → neural upsampler → re-normalize (VAE encoder per-channel stats)
6. **Stage 2**: Reload **clean** transformer (no LoRA) + denoise (3 steps, distilled sigmas)
7. **Decode**: Free DiT, load decoders on-demand, stream video+audio

### Critical Implementation Details

- **Memory**: Decoders (VAE decoder, audio, vocoder) loaded on-demand in `generate_and_save()`, NOT during `generate()`. This keeps peak memory under 32GB.
- **Stage 2 clean transformer**: After Stage 1, the LoRA-fused transformer is deleted and a fresh distilled transformer is loaded. Matches reference's separate `ModelLedger`s.
- **Upsampler denorm/renorm**: The neural upsampler must receive denormalized latents (`vae_encoder.denormalize_latent()` before, `normalize_latent()` after). Without this, Stage 2 produces garbage.
- **Reference resolution**: Must be 32-aligned for VAE encoder. Computed via `compute_video_latent_shape(num_frames, h // scale, w // scale)` then `* 32`.
- **Stage 2 positions**: Derived from actual upscaled dims (`H_half * 2`, `W_half * 2`), not target `height/width` (which may round differently).
- **Motion Track BGR**: Control videos use BGR channel order (matches IC-LoRA training format).
- **LoRA key remapping**: Uses `LTXV_LORA_COMFY_RENAMING_MAP` (ComfyUI/diffusers → MLX keys). All 480 LoRA targets match model keys.

### Key Files
- `ic_lora.py` — `ICLoraPipeline` (extends `BasePipeline`)
- `conditioning/types/reference_video_cond.py` — `VideoConditionByReferenceLatent`
- `conditioning/types/attention_strength_wrapper.py` — `ConditioningItemAttentionStrengthWrapper`
- `loader/fuse_loras.py` — LoRA weight fusion with quantization support
- `loader/sd_ops.py` — `LTXV_LORA_COMFY_RENAMING_MAP`

---

## Block Streaming (`--low-ram`)

Stream transformer blocks from a memory-mapped safetensors file so peak Metal memory stays at ~one block instead of ~num_layers blocks. MLX-native equivalent of upstream PyTorch's CUDA-stream-based block_streaming, but ~10x simpler (~250 lines vs ~840) because the upstream "CPU pinned + GPU pool" model doesn't apply to Apple Silicon's unified memory.

### How it works

Three pieces combine to make per-block streaming work without tripping the macOS Metal "Impacting Interactivity" watchdog:

1. **`mx.set_cache_limit(0)`** in pipeline `__init__`: tells MLX to immediately return freed Metal buffers to the OS instead of keeping them in a heap. Without this, MLX retains "recently freed" buffers and the page-cache can't evict mmap'd safetensors pages between forwards.

2. **`mx.compile(shared_block, inputs=shared_block)`** in `StreamingLTXModel`: pre-compiles the shared block forward. The `inputs=block` annotation tells the compiler that the block parameters can vary between calls — `streamer.bind()` rebinds weights without invalidating the compiled graph. Compiled kernels dispatch fast enough that 48 sequential `mx.eval` syncs no longer trip the watchdog.

3. **Per-block `mx.eval` sync** inside `LTXModel.__call__` when `block_provider` is set: forces the lazy compute graph to materialize between blocks so the previous block's weights become evictable. Without sync, the graph holds refs to all 48 blocks within one forward.

The pipeline drops `transformer_blocks[1:]` before quantization so only block 0 is materialized; quantization scales/biases for that one block fit in ~430 MB (q8) / ~875 MB (bf16). The streamer rebinds block i's weights into block 0 for each forward iteration.

### Empirical

LTX-2.3 q8 distilled, 48 blocks, Nv=192 (256x384x9 frames):
- Without streaming: peak Metal ~10-12 GB (transformer alone).
- **With `--low-ram`: peak Metal ~2.76 GB** (after Gemma freed). ~75% reduction.
- Latency: ~5% slower per step (compile overhead).

LTX-2.3 bf16 distilled, 480x704x33: confirmed runs end-to-end on M2 Pro 32 GB. Without streaming this would OOM (44 GB transformer alone).

### Coverage

`--low-ram` is supported and end-to-end-validated on:
- `generate` (one-stage T2V/I2V)
- `generate --two-stage` (Euler + CFG)
- `generate --two-stages-hq` (res_2s + CFG)
- `a2v` (audio-to-video)
- `keyframe` (interpolation)
- `ic-lora` (control video conditioning, via bind-time LoRA fusion)

Validated runs on M2 Pro 32 GB:
- bf16 HQ at 480x704x97 (4 sec): 49:38 — would OOM without streaming.
- q8 HQ at 480x704x97 (4 sec): 44:38.
- q8 one-stage 480x704x33 (1.3 sec): 2:31.

For two-stage / HQ / a2v / keyframe at default LoRA strength 1.0, the Stage 1 → Stage 2 transition swaps the streamer from ``transformer-dev.safetensors`` to the pre-fused ``transformer-distilled.safetensors`` (mlx-forge produces this at LoRA strength 1.0). For custom ``--distilled-lora-strength`` (or any non-1.0 strength), the streamer keeps the dev model and attaches a ``BlockLoraSource`` that fuses the LoRA delta at each ``bind()`` (dequantize → ``W + B @ A * strength`` → re-quantize for q4/q8).

For `ic-lora`, each control LoRA is attached as a ``BlockLoraSource`` to the streaming wrapper instead of being fused in-place at load time. Stage 2 just clears the source list rather than reloading the whole transformer.

`mx.compile` cannot trace `BatchedPerturbationConfig` (the dataclass passed for STG / modality-isolation passes), so `StreamingLTXModel.__call__` falls back to the eager block whenever ``perturbations`` is non-None. Eager + per-block sync still works thanks to `set_cache_limit(0)`.

### Limitations

- The `generate --lora` flag (per-pipeline LoRA on the one-stage path) is still incompatible with `--low-ram`. It pre-fuses LoRAs into the weight dict at load time, before streaming is set up. Either drop `--low-ram` or pre-fuse via mlx-forge.
- Bind-time fusion at custom strength is slower than the strength-1.0 swap path: dequantize + re-quantize for every linear in every block, every step. ~50ms per linear × ~50 linears × 48 blocks × num_forward_passes. For 30 stage-1 steps this adds noticeable wall-clock. For typical strengths between 0.8 and 1.2 the strength-1.0 swap output is visually indistinguishable, so prefer it when possible.
- The compiled-block forward differs from eager by ~1 fp32 ULP (kernel fusion). `tests/test_block_streaming.py::test_wrapper_matches_baseline` uses `mx.allclose(atol=1e-5, rtol=1e-5)` to capture this.

### Key Files

- `loader/block_streaming.py` — `BlockStreamer` (mmap'd safetensors + per-block key map + bind with eviction + auto-reload) and `StreamingLTXModel` (drop-in wrapper).
- `model/transformer/model.py` — `block_provider` parameter on `LTXModel.__call__` + per-block sync.
- `_base.py` — `BasePipeline.low_ram_streaming` constructor flag wired to `--low-ram` CLI.
- `tests/test_block_streaming.py` — 7 unit tests covering bind, block_provider hook, wrapper, eviction + auto-reload, bind-time LoRA fusion.

---

## Modality Tiling (`--tile-frames N --tile-spatial M`)

Splits the patchified video token sequence into spatial+temporal tiles so each tile is denoised independently, then blends results back with trapezoidal weights. Tackles a different memory bottleneck than block streaming:

- **Block streaming**: caps **weight memory** (transformer params).
- **Modality tiling**: caps **activation memory during forward**, dominated by the O(N²) attention scores tensor.

For long / HD videos, attention activations can exceed working-set even with weights streamed. Tiling splits N into N/k per tile so peak attention memory drops by ``k²``.

### Why on Apple Silicon

Per-layer attention scores tensor size scales with token count squared:

- 480x704x33 (Nv=1650): ~350 MB / layer.
- 480x704x97 (Nv=3168): ~1.3 GB / layer.
- 720x1280x97 (Nv≈9000): ~10 GB / layer.
- 1080p 8s+: doesn't fit any current Apple Silicon.

For the latter targets, ``--tile-spatial 2`` (4 spatial tiles) cuts attention activation by 4x. Combined with ``--low-ram`` (weights ~3 GB), 1080p / 8s+ inference becomes feasible on Mac Studio 64-128 GB.

### Architecture

Mirrors upstream ``ltx_core.modality_tiling.VideoModalityTilingHelper`` API verbatim:

- ``Modality`` dataclass (``model/transformer/modality.py``): bundles latent + sigma + timesteps + positions + context + masks. The canonical input/output type for tiling helpers.
- ``VideoModalityTiler.tile_modality(modality, tile, normalize_positions=True) -> (Modality, TileContext)``: slices token-level state to a tile, normalizes positions, builds keep mask + per-cond-token blend weights.
- ``VideoModalityTiler.blend(tile_output, tile, ctx, output=None)`` accumulates the tile contribution into the full token buffer with trapezoidal blend masks at overlaps.
- ``TiledLTXModel`` wraps ``LTXModel`` (or ``StreamingLTXModel``); intercepts ``__call__``, builds Modality from kwargs, iterates tiles, blends video output, averages audio output across tiles. Pipelines stay unchanged.

### Position layout divergence (documented)

Upstream uses ``(B, num_axes, T, 2)`` interval positions per token; we use ``(B, T, num_axes)`` point coords (consistent with the rest of our codebase). Conditioning-token overlap test is therefore "point in [tile_start, tile_end] on every axis" instead of upstream's interval-overlap. Mathematically equivalent for non-degenerate intervals.

### CLI

| Flag | Default | Effect |
|------|---------|--------|
| `--tile-frames N` | 1 | Number of temporal tiles |
| `--tile-spatial M` | 1 | Number of spatial tiles per axis (M*M total) |
| `--tile-overlap K` | 2 | Token-grid overlap between tiles. Larger overlap = smoother blend but more redundant compute |

Total tiles = ``N * M * M``. Default ``1*1*1`` = no tiling.

Coverage: ``generate`` (one-stage / ``--two-stage`` / ``--two-stages-hq``), ``a2v``, ``keyframe`` via ``TI2VidTwoStagesPipeline.tile_count`` (inherited). ``ic-lora`` could be wired on the same model with the same primitive but isn't yet.

### Tradeoff

Tiling adds wall-clock overhead (each tile is a separate model forward + kernel dispatch). On 32 GB Mac at typical Nv (1650-3168), tiling overhead dominates over memory benefit — use ``--low-ram`` alone. On Mac Studio 64-128 GB targeting 1080p / 8s+ where attention activations OOM otherwise, tiling unblocks the run at the cost of ~2-3x latency.

Validation: with conservative config (``--tile-frames 2 --tile-overlap 4`` on 480x704x33), output is **PSNR 228 dB / bit-identical** to non-tiled baseline (overlap saturates the tile coverage, blend math averages back to identity). Confirms blend correctness; not a stress test of tile boundaries.

### Key Files

- `components/modality_tiling.py` — `VideoModalityTiler` + `TiledLTXModel`.
- `model/transformer/modality.py` — `Modality` dataclass (isomorphic with upstream).
- `model/video_vae/tiling.py` — token-grid primitives (`split_by_count`, `identity_mapping_operation`, `TileCountConfig`).
- `tests/test_modality_tiling.py` — 8 unit tests bit-exact at 1e-6 (tile/blend round-trips, position normalization, cond overlap, wrapper baseline).

---

## HDR IC-LoRA Pipeline

Two-stage IC-LoRA pipeline that produces **linear HDR video output** via LogC3 inverse compression. Subclasses `ICLoraPipeline` so it inherits low-RAM streaming, modality tiling, bind-time LoRA fusion, and the standard two-stage decode flow. Mirrors upstream `ltx_pipelines.hdr_ic_lora.HDRICLoraPipeline` 1:1 for clean PR diff tracking.

### Mechanism

The HDR LoRA is trained so the VAE decoder output (mapped to `[0, 1]`) holds a LogC3-compressed signal. `LogC3.decompress` recovers the linear HDR signal in `[0, ∞)`. Conditioning input is treated as standard SDR (clamp `[0, 1]` then `to_vae_range`).

| Mode | Support | Notes |
|---|---|---|
| **V2V** (SDR ref → HDR) | ✅ primary | `--video-conditioning source.mp4 1.0` |
| **T2V** (pure text → HDR) | ✅ | `--video-conditioning` omitted (matches upstream's empty-list path); LoRA runs out-of-distribution but functional |
| **I2V + V2V** | ✅ | Add `--image photo.jpg` on top |
| **One-stage HDR** | ❌ | Upstream is two-stage by design (stage 1 half-res + stage 2 upscale) — not provided |
| `high_quality_hdr` mode | ⏳ deferred | Upstream's 2× frame oversample + drop-alternate; not yet ported |
| EXR sequence output | ⏳ deferred | We save `.hdr.npz` (`(F, H, W, 3)` fp32); user tooling converts to EXR/TIFF |

### Auto-detection

`HdrLoraConfig` is read from the LoRA's safetensors metadata via `read_hdr_lora_config()`:

| Metadata key | Effect |
|---|---|
| `hdr_transform` | Names the transform (only `"logc3"` supported); presence triggers HDR mode |
| `use_hdr_transform` | Legacy boolean fallback |
| `reference_downscale_factor` | Spatial downscale for V2V conditioning (matches LoRA training recipe) |

`HDRICLoraPipeline.__init__` raises `ValueError` if no HDR LoRA is detected. Pass `hdr_lora_config=HdrLoraConfig(...)` explicitly to override.

### Outputs

- `<output>` (e.g. `out.mp4`): SDR preview, tonemapped via standard streaming VAE decode (clips highlights at 1.0).
- `<output>.hdr.npz`: float32 `(F, H, W, 3)` linear HDR tensor. User tooling converts to EXR / TIFF / OpenEXR sequences.

### Validated

End-to-end run on M2 Pro 32 GB, dev model + HDR LoRA fused, q8:
- 480×704×9 short test: 83 s — `pixels > 1.0`: 0.65%, range `[-0.017, 7.85]`.
- 704×448×89 cosmic with `--low-ram`: 6:43 — 12.84% highlights, range `[-0.017, 55.08]`.
- 1280×704×97 Lisbon T2V with `--low-ram`: 15:28 — 14.74% highlights, range `[-0.017, 55.08]`.

### Key Files

- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/hdr_ic_lora.py` — `HDRICLoraPipeline` subclass with `_decode_to_hdr` + `generate_and_save` saving HDR npz.
- `packages/ltx-core-mlx/src/ltx_core_mlx/hdr.py` — `LogC3` compress/decompress + `apply_hdr_decode_postprocess`.
- `packages/ltx-core-mlx/src/ltx_core_mlx/loader/hdr_metadata.py` — `HdrLoraConfig` + `read_hdr_lora_config`.
- `tests/test_hdr.py`, `tests/test_hdr_metadata.py` — 16 unit tests.

---

## Metal Watchdog Mitigation

The macOS GPU watchdog (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`, ~10 s per Metal command buffer) trips when:

1. A single command buffer takes too long (one giant lazy-graph dispatch).
2. Many small command buffers queue behind system processes (mds_stores, knowledgeconstructiond, Spotlight, Siri).

LTX-2 mitigates both by inserting `mx.eval` at strategic points so each command buffer is small enough to fit one watchdog window but few enough that queue contention doesn't dominate. These guards apply on **all Apple Silicon Macs** — the previous 48 GB threshold was empirically wrong; M2 Max 64 GB machines exhibit `MTLCommandBufferErrorInternal` (code 14) crashes without them:

- **Gemma forward**: per-layer eval (48 layers × ~100 ms each). Override via `LTX2_GEMMA_EVAL_EVERY=N` (default `1`).
- **TextEmbeddingProjection**: per-output-projection eval (splits the 188160→4096 video matmul from the 188160→2048 audio matmul).
- **Embeddings1DConnector**: per-block eval (8 video + 8 audio blocks).
- **LTX DiT block loop**: eval every `LTX2_DIT_EVAL_EVERY` blocks (default `8`). Splits the 48-block forward into 6 command buffers of ~6 blocks each (~1–2 s/buffer), well within the watchdog window.

Mac Studio / M-series Ultra users who have never seen a watchdog crash may recover full lazy-graph pipelining by setting `LTX2_GEMMA_EVAL_EVERY=0` and `LTX2_DIT_EVAL_EVERY=0`. This disables all eval guards and restores maximum throughput at the cost of watchdog safety on machines where those guards were needed.

### `LTX2_GEMMA_MAX_LENGTH`

Caps the padded Gemma sequence length (default `1024`). Reducing to `512` halves Gemma forward time but **shifts left-padded RoPE positions away from the LTX training distribution** — quality risk. Use only as a last resort on heavily contended systems.

```bash
LTX2_GEMMA_MAX_LENGTH=512 ltx-2-mlx hdr-ic-lora ...
```

### Pipeline-load ordering

The fix that actually unblocked production-quality generation was `1a30f74`: every pipeline's `load()` method previously called `_load_text_encoder()` to load Gemma → free → load DiT, but the wrapping `generate_*()` methods had ALREADY encoded the prompt and freed Gemma BEFORE calling `load()`. Loading Gemma twice (7.5 GB mmap each time) right before the 10 GB DiT thrashed the Metal heap and caused the watchdog crash. The text encoder lifecycle now lives entirely in `generate_*()` methods; `load()` no longer touches Gemma.

---

## Release Process

The project is pre-1.0 (`0.x.y`). Under that scheme the `0.y` segment serves
as the major version: **breaking changes bump `y`, strictly additive changes
bump `z`**. This is documented at the top of [CHANGELOG.md](CHANGELOG.md)
and applies to every consumer of the package.

### Branch & PR discipline

- `main` is **protected**: PR required, CI green required (`lint` +
  `syntax` strict, plus `test (3.11)` / `test (3.12)` / `commitlint`),
  no force-push, no deletion. Direct push is blocked.
- One PR = one logical concern. Split mixed work into multiple PRs so
  each carries a coherent version bump (see PR #8/#9/#10 splitting the
  upstream PR #212 sync into additive / default-changes / new-pipeline).
- PR titles use conventional-commits prefixes (`feat:`, `fix:`,
  `chore:`, `refactor:`, `docs:`). `commitlint` enforces this.

### Versioning rules

| Change type | Bump | Example |
|---|---|---|
| New pipeline (additive) | `z` | `0.12.0 → 0.12.1` (LipDub) |
| New helper / primitive (additive) | `z` | `0.11.0 → 0.11.1` (diffusion_steps) |
| Internal refactor, public API unchanged | `z` | `0.11.0 → 0.11.1` (ic_lora delegation) |
| Default value change (potentially breaking for callers relying on defaults) | `y` | `0.11.1 → 0.12.0` (TilingConfig / rope defaults) |
| Removal / signature change | `y` | `0.9.x → 0.10.0` (ImageToVideoPipeline removal) |

### Release artifacts

Every release on `main` produces, in this order:

1. **Merged squash commit** with conventional-commits subject + body.
2. **Annotated git tag** `vX.Y.Z` on the merge commit (created locally
   with `git tag -a vX.Y.Z -m '...'`, pushed via `git push origin vX.Y.Z`).
3. **GitHub Release** auto-created by `.github/workflows/release.yml` (it
   listens on `v*` tags). After creation, edit the release notes to use
   the corresponding `CHANGELOG.md` section verbatim via
   `gh release edit vX.Y.Z --notes-file …`.
4. **CHANGELOG entry** at the top of the file under
   `## [X.Y.Z] - YYYY-MM-DD` with `### Added` / `### Changed` /
   `### Fixed` / `### Removed` sections as needed. Plain prose first
   paragraph explaining the why; details below.
5. **`pyproject.toml` version bump** in the workspace root **and** in
   every sub-package (`packages/ltx-core-mlx/`, `packages/ltx-pipelines-mlx/`,
   `packages/ltx-trainer/`) — all four must stay in sync.
6. **`.release-please-manifest.json` bump** so release-please can drive
   future automated releases from the correct base.

### Pre-releases (release candidates)

When a release introduces visible behavioural changes that downstream
apps must validate before adopting, cut a **pre-release** first:

- Tag as `vX.Y.Z-rc.N` (e.g. `v0.12.0-rc.1`).
- Mark as **pre-release** in GitHub Releases UI (or `--prerelease`
  on `gh release create`).
- Promote to the stable tag only after downstream validation.

### Maturity tiers

Pipelines are classified Stable / Beta / Experimental in
[docs/PIPELINE_MATURITY.md](docs/PIPELINE_MATURITY.md). CLI `--help`
output for non-Stable subcommands carries a `[beta]` or
`[experimental]` tag. Tier promotion criteria + per-tier stability
guarantees live in that doc.

### Downstream communication

**Out of scope.** External communication with consumer apps / integrators
is owned by Damien. The release process produces the artifacts above —
notifying consumers, drafting changelog announcements, scheduling
upgrades, etc., is not done from this repo.

---

## Conventions

- Python 3.11+
- Mandatory type hints on all functions
- Google-style docstrings
- ruff for formatting/linting (pre-commit + CI lint job)
- Tests in `tests/` using pytest
- **Conventional commits** (`feat:`, `fix:`, `chore:`, `docs:`,
  `refactor:`, `feat!:` / `fix!:` for breaking) — enforced by
  `commitlint` in CI.
- Package imports: `ltx_core_mlx.*` for core, `ltx_pipelines_mlx.*` for pipelines.
- One PR = one concern + one version bump (see Release Process above).

---

## Resources

- **ltx-core**: [GitHub](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core)
- **ltx-pipelines**: [GitHub](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-pipelines)
- **MLX**: [Docs](https://ml-explore.github.io/mlx/) · [GitHub](https://github.com/ml-explore/mlx)
- **mlx-forge**: [GitHub](https://github.com/dgrauet/mlx-forge) — weight conversion
- **Pre-converted weights**: [HuggingFace collection](https://huggingface.co/collections/dgrauet/ltx-23)
