# CLAUDE.md — ltx-2-mlx

## Project Overview

Pure MLX port of [LTX-2](https://github.com/Lightricks/LTX-2/) (Lightricks) for Apple Silicon. Three-package monorepo mirroring the reference structure:

- **ltx-core-mlx** (`ltx_core_mlx`) — model library: DiT, VAE, audio, text encoder, conditioning
- **ltx-pipelines-mlx** (`ltx_pipelines_mlx`) — generation pipelines: T2V, I2V, A2V, retake, extend, keyframe, IC-LoRA, HDR IC-LoRA, LipDub, one-stage, two-stage, distilled, DFR
- **ltx-trainer** (`ltx_trainer_mlx`) - ltx-2 training, democratized.

Loads pre-converted MLX weights from the [LTX-2.3](https://huggingface.co/collections/dgrauet/ltx-23) and [LTX 2.5](https://huggingface.co/collections/dgrauet/ltx-25-6a90c410ff65a75f8aeae402) MLX collections on HuggingFace. Weight conversion is handled by [mlx-forge](https://github.com/dgrauet/mlx-forge).

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
│       ├── color/                         # HDR colour science: hlg.py, primaries.py, yuv.py
│       ├── hdr.py                         # ACEScct working space + sRGB EOTF (HDR IC-LoRA)
│       ├── duration_head/                 # DurationHead (2.5 auto-duration)
│       │
│       ├── components/                    # Shared pipeline components
│       │   ├── guiders.py                 # Guidance strategies
│       │   ├── diffusion_steps.py         # Euler / res_2s / Euler-ancestral / CFG++ step primitives
│       │   ├── modality_tiling.py         # VideoModalityTiler, TiledLTXModel
│       │   └── patchifiers.py             # VideoLatentPatchifier, AudioPatchifier
│       │
│       ├── conditioning/                  # Latent conditioning system
│       │   ├── mask_utils.py              # build/update/resolve attention masks
│       │   ├── prompt_relay.py            # Prompt Relay (--segment) cross-attention mask
│       │   └── types/
│       │       ├── attention_strength_wrapper.py # Attention strength wrapping
│       │       ├── latent_cond.py         # LatentState, VideoConditionByLatentIndex
│       │       ├── keyframe_cond.py       # VideoConditionByKeyframeIndex
│       │       ├── keyframe_slots.py      # VideoGeneratedKeyframeSlots (2.5)
│       │       ├── reference_audio_cond.py # Audio reference conditioning (LipDub)
│       │       └── reference_video_cond.py # VideoConditionByReferenceLatent (IC-LoRA)
│       │
│       ├── guidance/                      # Guidance utilities
│       │   ├── nag.py                     # NAG (negative prompt without CFG): NAGConfig, NAGGuidance, the combine
│       │   └── perturbations.py           # Noise perturbation strategies
│       │
│       ├── loader/                        # Weight loading & LoRA fusion
│       │   ├── block_streaming.py         # BlockStreamer, StreamingLTXModel (--low-ram)
│       │   ├── helpers.py                 # Checkpoint metadata (version parser)
│       │   ├── fuse_loras.py              # LoRA weight fusion
│       │   ├── lora_adapters.py           # Unfused LoRAs: run-time adapters (LTX2_LORA_MODE=unfused)
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
│       │   │   ├── modality.py            # Modality dataclass (tiling I/O)
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
│       │       ├── diffusion_decoder/     # NADiffusionDecoder (2.5 --video-decoder diffusion)
│       │       ├── convolution.py         # Conv3dBlock (causal + reflect padding)
│       │       ├── resnet.py              # ResBlock3d, ResBlockStage
│       │       ├── sampling.py            # DepthToSpaceUpsample, pixel_shuffle_3d
│       │       ├── tiling.py              # Tiled VAE encoding/decoding
│       │       ├── normalization.py       # pixel_norm (RMS)
│       │       └── ops.py                 # PerChannelStatistics
│       │
│       ├── text_encoders/                 # Text encoding (Gemma 3 on 2.3, Gemma 4 on 2.5)
│       │   └── gemma/
│       │       ├── embeddings_connector.py  # Embeddings1DConnector (RoPE + registers)
│       │       ├── feature_extractor.py     # GemmaFeaturesExtractorV2 (video/audio projections)
│       │       ├── gemma4.py, gemma4_config.py # Gemma 4 text tower (2.5 packs)
│       │       └── encoders/
│       │           ├── base_encoder.py      # Gemma 3 12B wrapper via mlx-lm
│       │           ├── gemma4_encoder.py    # Gemma 4 encoder (2.5 packs)
│       │           ├── encoder_configurator.py # select_text_encoder (Gemma 3 vs 4)
│       │           └── prompts/             # System prompt templates
│       │               ├── gemma_t2v_system_prompt.txt
│       │               └── gemma_i2v_system_prompt.txt
│       │
│       └── utils/
│           ├── diffusion.py   # to_velocity / to_denoised
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
│       ├── _base.py                       # BasePipeline (private composition facade)
│       ├── ti2vid_one_stage.py            # TI2VidOneStagePipeline (dev one-stage + CFG, full target res)
│       ├── ti2vid_two_stages.py           # Two-stage: half res → upscale → refine
│       ├── ti2vid_two_stages_hq.py        # Two-stage HQ variant
│       ├── distilled.py                   # DistilledPipeline (2.3 / 2.5 dispatch, _stage1 / _stage2)
│       ├── dfr.py                         # DFRPipeline (generate --dfr, 2.5)
│       ├── dfr_layout.py                  # DFR canvas + temporal tile plan
│       ├── a2vid_two_stage.py             # Audio-to-video two-stage pipeline
│       ├── a2vid_distilled.py             # Audio-to-video on the distilled path (a2v --distilled)
│       ├── retake.py                      # RetakePipeline: regenerate a time segment + extend (append/prepend)
│       ├── extend_distilled.py            # ExtendDistilledPipeline (extend --distilled: chunk-style continuation)
│       ├── keyframe_interpolation.py      # Keyframe interpolation
│       ├── ic_lora.py                     # IC-LoRA reference-based generation
│       ├── iclora_utils.py                # Shared IC-LoRA helpers (metadata, reference conditioning)
│       ├── hdr_ic_lora.py                 # HDRICLoraPipeline (ACEScct SDR-to-HDR, 2.5)
│       ├── lipdub.py                      # LipDubPipeline
│       ├── scheduler.py                   # DISTILLED_SIGMAS, STAGE_2_SIGMAS
│       ├── cli.py                         # CLI entry point
│       ├── scripts/                       # TeaCache calibration + polyfit
│       └── utils/
│           ├── samplers.py                # Sampling loops (Euler, ancestral, res_2s, guided)
│           ├── constants.py               # Pipeline constants (guidance params)
│           ├── res2s.py                   # res_2s second-order solver coefficients (phi functions)
│           ├── blocks.py                  # Model-component blocks (encoders, decoders, DurationPredictor)
│           ├── _orchestration.py          # Multi-component flows shared across pipelines
│           ├── helpers.py, types.py       # Upstream-named orchestration helpers + types
│           ├── args.py                    # Multi-image I2V input parsing
│           ├── generation.py              # is_ltx25_pack
│           ├── media_io.py                # Image / video / audio I/O
│           ├── hdr_media.py               # HDR input / EXR / HLG I/O
│           ├── estimate.py                # [estimate] cost lines
│           ├── progress.py                # [phase] markers
│           ├── stepwise.py                # Stepwise previews
│           └── watchdog.py                # macOS GPU-watchdog failure recognition
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

- **Type**: Diffusion Transformer (DiT), ~22B params (upstream naming `ltx-2.3-22b-*`), joint audio+video single-pass
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

The DiT also runs video-only: `LTXModel` / `X0Model` and the Euler loop accept `audio_state=None` (#181), which `hdr-ic-lora` uses.

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

LTX-2.5 packs (same variant semantics; carry the Gemma-4 text tower, conv VAE pair and DurationHead):

| Variant | HuggingFace | Size | Notes |
|---------|-------------|------|-------|
| bf16 | [dgrauet/ltx-2.5-mlx](https://huggingface.co/dgrauet/ltx-2.5-mlx) | ~120GB | Full precision (dev + distilled + Gemma-4 in bf16); needs `--low-ram` on 32GB |
| int8 | [dgrauet/ltx-2.5-mlx-q8](https://huggingface.co/dgrauet/ltx-2.5-mlx-q8) | ~75GB | Recommended; validated e2e on all 2.5 pipelines |
| int4 | [dgrauet/ltx-2.5-mlx-q4](https://huggingface.co/dgrauet/ltx-2.5-mlx-q4) | ~47GB | Lower quality; validated (contracts + deterministic distilled render, ~97s/render) |

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
- Loaders derive `(bits, group_size)` from tensor shapes → any group_size (32/64/128) + bit width (int4/int8) loads & fuses. Single home: `utils/weights.py::derive_quant_params` (exact-consistency validated); shared by load-time quant + LoRA fusion.

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
| `transformer-dev.safetensors` / `transformer-distilled.safetensors` | `transformer.` | Dev and pre-fused distilled DiT (two-stage, `--low-ram` stage swap; possibly versioned, e.g. `-1.1`) |
| `text_encoder.safetensors` (+ `text_encoder_config.json`) | `text_encoder.` | Gemma-4 text tower (2.5 only) |
| `duration_head.safetensors` | `duration_head.` | DurationHead (2.5 only) |
| `vae_decoder_conv.safetensors` / `vae_encoder_conv.safetensors` | `vae_decoder_conv.` / `vae_encoder_conv.` | Conv video VAE (2.5 only; replaces `vae_decoder` / `vae_encoder`) |
| `vae_decoder_av.safetensors` | `vae_decoder_av.` | Diffusion video decoder (2.5 only) |
| `spatial_upscaler_x2_v1_0.safetensors` | N/A | 2x spatial upsampler (2.5 packs) |

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
    frame = decoder.decode_frame(latents[:, :, i : i + 1])
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
- `frozen`: `True` marks a stream that is conditioning only (upstream `LatentState.frozen`); it always carries an
  all-zero `denoise_mask`. `create_noised_state(..., frozen=True)` builds one. Set where upstream sets
  `frozen=True`: a2v audio (both stages), lipdub stage-2 audio, `--two-stage` stage-2 audio (upstream v1.4.0
  `freeze_audio=True`), retake audio with `--no-regen-audio`.

### Frozen streams and per-modality sigma
Upstream gives each modality its own `Modality.sigma` and forces it to 0 for a frozen stream
(`modality_from_latent_state`). That sigma drives the modality's **prompt AdaLN** and the **other**
modality's **cross-attention gate** (the A→V gate on the video side reads the audio sigma, the V→A gate
reads the video sigma). `LTXModel.__call__` takes optional `video_sigma` / `audio_sigma` (default: the
global `timestep`); the four sampler loops pass `0` for a frozen state on every pass (CFG / STG /
modality passes, both res_2s evaluations) and pass nothing otherwise, so renders without a frozen stream
are byte-identical. Per-token timesteps (`sigma * denoise_mask`) already zeroed the frozen stream's
9-param AdaLN and x0. a2v stage 2 additionally used to re-noise its audio (`sigma=start_sigma`, all-ones
mask) and denoise it with the video; it is now frozen like upstream (`frozen=True, noise_scale=0.0`).
Before this fix the prompt AdaLNs and both gates saw the step sigma: **a2v,
lipdub stage 2 and retake `--no-regen-audio` outputs shift** (a2v q8 512×768×49 seed 5: PSNR 20.8 dB vs
the old render; the muxed audio is unchanged by construction — a2v muxes the input wav, retake
`--no-regen-audio` already preserved it; retake `--no-regen-audio` on a 49-frame clip, latent frames 2-4: 50.1 dB,
audio identical; `--distilled` byte-identical).

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

The user-facing pipelines guide (decision tree, per-pipeline cards, flag matrix) lives at [docs/PIPELINES.md](docs/PIPELINES.md); `tests/test_docs_flags.py` keeps its flag lists in sync with `cli.py`. Quick reference:

Entry point: `uv run ltx-2-mlx <command>`. Available commands:

| Command | Pipeline | Tier | Description |
|---------|----------|------|-------------|
| `generate` | T2V / I2V (mode flag required) | Stable | `--one-stage` (dev+CFG @ target), `--two-stage` (dev+CFG+upscale, recommended), `--two-stages-hq` (res_2s+CFG+upscale), `--distilled` (distilled+upscale, fastest), `--dfr` (2.5 packs, experimental; distilled + detailing IC-LoRA). `--image` for I2V on any mode. `--segment` for Prompt Relay temporal prompt gating. `-f/--frames` defaults to auto-predicted duration on 2.5 packs (via `DurationHead`) and is **required** on 2.3 packs (immediate `ValueError` before any Gemma load if omitted). `--auto-duration MIN:MAX` overrides the predictor's clamp range on 2.5 packs. `--no-audio` skips audio decode + mux (mp4 with no audio track; video unchanged, audio latents still generated jointly). `--num-generated-keyframes N` (2.5 packs) adds N generated keyframe slots to stage 1 for fast motion. `--dfr` (2.5 packs) adds `--temporal-upscalings {0,1,2}` for post-hoc temporal x2 refine rounds (default 0) and `--spatial-upscalings {1,2}` for a full-res spatial detailing epilogue (default 1). |
| `keyframe` | Keyframe interpolation | Stable | Two-stage interpolation between start/end frames |
| `ic-lora` | IC-LoRA | Stable | Two-stage generation with control video conditioning (depth, canny, pose, motion tracks) |
| `hdr-ic-lora` | HDR IC-LoRA | Experimental | Single-stage ACEScct SDR-to-HDR IC-LoRA (LTX-2.5 packs only): HLG BT.2020 10-bit mp4 + ACEScg EXR frames. Takes `--input`, `--hdr-lora`, `--text-embeddings` (no prompt) |
| `a2v` | Audio-to-video | Beta | Two-stage audio-conditioned generation (Euler + CFG). Sync quality depends on prompt-audio alignment. `--distilled` (Experimental): the distilled path, no CFG, input audio frozen in both stages |
| `retake` | Retake | Beta | Regenerate a time segment of an existing video (dev model + CFG). `--distilled` (Experimental): the distilled model on the 8-step distilled table, no CFG (upstream's default retake mode) |
| `extend` | Extend | Beta | Add frames before or after an existing video (dev model + CFG). `--distilled` (Experimental): append on the distilled two-stage path, the source's last 25 frames pinned (upstream chunk continuation) |
| `lipdub` | LipDub | Experimental | Lip-dub a reference video → re-sync visuals to source audio. Output audio is a VAE+vocoder reconstruction (audible artifacts on rich music). Uses pre-1.0 LipDub IC-LoRA. Stage 2 uses stage 1's generated audio as its reference (upstream Dub-It); the stage-1 reference is the source audio sliced or zero-padded to the clip window (#174). |
| `enhance` | Prompt enhancement | Stable | Enhance a text prompt using Gemma (no video generation) |
| `info` | Model info | Stable | Lists the pack's safetensors files and a naive RAM estimate (total weights × 1.3; ignores `--low-ram` and which transformer a mode loads, so it overstates 2.5 packs) |
| `train` | Training | Stable | Train a LoRA or full model from YAML config (requires ltx-trainer-mlx) |
| `preprocess` | Data preprocessing | Stable | Encode raw videos into latents + conditions for training |
| `slice` | Training data | Stable | Slice long videos into normalized training clips (audio retained) |

`generate --one-stage/--two-stage/--two-stages-hq`, `keyframe`, `a2v`, `retake` and `extend` use the dev model with CFG; `generate --distilled/--dfr`, `a2v --distilled`, `retake --distilled`, `extend --distilled`, `ic-lora` (unless `--dev-transformer`), `hdr-ic-lora` and `lipdub` use the distilled model without CFG. Common flags: `--model`, `--prompt`, `--output`, `--seed`, `--quiet` (`hdr-ic-lora` has its own set: no `--prompt`). CFG modes (`generate --one-stage/--two-stage/--two-stages-hq`, `keyframe`, `a2v`, `retake`, `extend`) take `--negative-prompt TEXT` (default: upstream `DEFAULT_NEGATIVE_PROMPT`; `""` is encoded verbatim; always one global prompt, even with `--segment`); `generate --distilled` / `--dfr` reject it (no CFG) unless `--nag` applies it through Normalized Attention Guidance (see "Negative prompt without CFG" below), and the distilled-sampler IC-LoRA family (`ic-lora`, `hdr-ic-lora`, `lipdub`) does not expose it. Every denoising stage prints an `[estimate]` work line (steps × passes × tokens = forwards) on stderr before step 1 and a time projection after the first computed step, refined once after the second (`utils/estimate.py`; retake/extend note that cost follows total clip length). Tier semantics + promotion criteria live in [docs/PIPELINE_MATURITY.md](docs/PIPELINE_MATURITY.md).

### Low-RAM Example

```bash
# bf16 inference on a 32 GB Mac via block streaming
ltx-2-mlx generate --two-stage \
  --model dgrauet/ltx-2.3-mlx \
  --prompt "a fox in the forest" \
  --low-ram \
  -H 480 -W 704 -f 33 --frame-rate 24 -o fox.mp4

# q8 inference fits 16 GB Macs (2.3 packs require -f explicitly — no DurationHead)
ltx-2-mlx generate --distilled \
  --model dgrauet/ltx-2.3-mlx-q8 \
  --prompt "a fox in the forest" \
  --low-ram -f 97 --frame-rate 24 -o fox.mp4
```

`--low-ram` is supported on every `generate` mode (incl. `--distilled` and `--dfr`), `a2v`, `keyframe`, `ic-lora`, `hdr-ic-lora`, `lipdub`, `retake` and `extend`. Bind-time LoRA fusion handles ic-lora's control LoRAs, custom `--distilled-lora-strength`, and `generate --lora` (community LoRAs). See `## Block Streaming` below for details.

### IC-LoRA Example

```bash
# Union Control (depth, canny, pose)
ltx-2-mlx ic-lora \
  --prompt "a person walking" \
  --lora Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control 1.0 \
  --video-conditioning depth_map.mp4 1.0 \
  --frame-rate 24 -o output.mp4

# Motion Track Control
ltx-2-mlx ic-lora \
  --prompt "particles moving" \
  --lora Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control 1.0 \
  --video-conditioning tracks.mp4 1.0 \
  --frame-rate 24 -o output.mp4
```

Flags: `--lora PATH STRENGTH` (repeatable, supports HF repo IDs), `--video-conditioning PATH STRENGTH` (repeatable), `--conditioning-strength`, `--skip-stage-2`, `--image`.

**Dev mode** (`--dev-transformer FILE`, opt-in — default stays distilled): fuses the distilled LoRA (`--distilled-lora`, `--distilled-lora-strength` default 0.5) **alongside** the task IC-LoRA in the same pass and keeps both across stages (Comfy Union-Control recipe). Missing `--dev-transformer` hard-fails (no silent distilled fallback). `--single-stage`: full-res one-pass (no upsampler / Stage 2), takes precedence over `--skip-stage-2`.

**Cheap topologies** (0.14.17, community PR #68): `--upsample-only` = half-res gen (control applied throughout) → 2× latent upsample → decode directly, no refine — fast rough draft (~3× faster than `--single-stage`). Add `--refine-steps N` for a **control-aware refine** after the upsample: the control is re-encoded at full res and re-appended with the IC-LoRA **kept fused** (no clean-model reload), unlike the legacy Stage 2 which is control-blind. `N` maps onto the distilled sigma tail (`N=3` == `STAGE_2_SIGMAS` depth; capped at 8); each refine step is ~8× a Stage 1 step (full-res gen + appended control tokens, O(n²) attention). `--refine-steps` without `--upsample-only` is silently ignored. Both flags ignored under `--single-stage`/`--skip-stage-2`.

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
  --low-ram -W W -H H -f N --frame-rate 24 --seed ... \
  -o output.mp4
```

Why it works: `VideoConditionByReferenceLatent` (used by ic-lora) adds conditioning tokens **frame-by-frame** rather than just at the anchor positions. The replicated canny edges give the model a stable spatial structure throughout the clip, and the Union Control LoRA is trained to respect that structure. Faster than `--two-stage` modes (~4× shorter wall-clock) because distilled defaults (8+3 steps, no CFG = 1 forward/step).

Alternatives via Union Control LoRA: depth maps (need external depth model like Depth-Anything-V2) or pose maps (OpenPose). Canny edges work for general scenes and require only ffmpeg.

### HDR IC-LoRA Example

```bash
# SDR mp4 -> HLG master + ACEScg EXR frames (LTX-2.5 pack required; LoRA repo is gated)
huggingface-cli download Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR --local-dir hdr-lora
ltx-2-mlx hdr-ic-lora \
  --model dgrauet/ltx-2.5-mlx-q8 \
  --input source_sdr.mp4 \
  --hdr-lora hdr-lora/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors \
  --text-embeddings hdr-lora/ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors \
  --low-ram -o out.mp4

# EXR-frame input (needs --frame-rate), ACEScct EXR sidecar
ltx-2-mlx hdr-ic-lora --input frames_acescg/ --input-colorspace acescg --frame-rate 24 \
  --exr-colorspace acescct --hdr-lora ... --text-embeddings ... --low-ram -o out.mp4
```

Flags: `--input PATH` (MP4/MOV, or a directory of `*.exr` frames), `--output-path/-o`, `--hdr-lora PATH`, `--text-embeddings PATH` (all required), `--input-colorspace {srgb_gamma,srgb,acescg,acescct}` (default `srgb_gamma`; MP4/MOV take `srgb_gamma`/`srgb`, EXR folders take `srgb`/`acescg`/`acescct`), `--exr-colorspace {srgb_linear,acescg,acescct}` (default `acescg`), `--frame-rate` (EXR folders only, forbidden for MP4/MOV), `--seed`, `--high-quality` (2x frames internally, ~2x slower), `--no-keyframes`, `--keyframe-strength` (0.95), plus `--model` (default `dgrauet/ltx-2.5-mlx-q8`), `--low-ram`, `--quiet`. Output length matches the source (frame count must be 8k+1). Removed vs the old LogC3 command: `--prompt`, `-H/-W/-f`, `--lora`, `--video-conditioning`, `--image`, `--stage1-steps`, `--stage2-steps`, `--skip-stage-2`, `--conditioning-strength`, `--tile-*`.

### Two-Stage Example

```bash
# Two-stage with Euler sampler (auto-selects q8 model; -f required on 2.3 packs)
ltx-2-mlx generate \
  --prompt "a scene description" \
  --two-stage -f 97 --frame-rate 24 -o output.mp4

# HQ with res_2s second-order sampler (higher quality, ~2x slower)
ltx-2-mlx generate \
  --prompt "a scene description" \
  --two-stages-hq -f 97 --frame-rate 24 -o output.mp4

# With I2V conditioning
ltx-2-mlx generate \
  --prompt "animate this" \
  --two-stage --image photo.jpg -f 97 --frame-rate 24 -o output.mp4

# On a 2.5 pack, -f can be omitted — duration is auto-predicted from the prompt/image
# via the DurationHead (see "LTX-2.5" section below), or clamped with --auto-duration MIN:MAX
ltx-2-mlx generate \
  --model /path/to/ltx-2.5-mlx-q8 \
  --prompt "a scene description" \
  --distilled --auto-duration 2:6 --frame-rate 24 -o output.mp4
```

Flags: `--two-stage` (Euler), `--two-stages-hq` (res_2s), `--cfg-scale` (default 3.0), `--stg-scale` (default 1.0; 0.0 on `--two-stages-hq`), `--stage1-steps` (default 30 standard, 15 HQ), `--stage2-steps` (default 3; a shorter count takes the **last** N sigmas of the stage-2 table via `scheduler.shorten_schedule(..., keep="tail")`, so it always ends at σ=0 — a stage 1 on the fixed distilled table uses `keep="start"`: σ=1.0 then the last N sigmas), `--image`, `-f/--frames` (required on 2.3 packs; optional on 2.5 packs — auto-predicted when omitted), `--auto-duration MIN:MAX` (2.5 packs only — overrides the predictor's clamp range; explicit `-f` wins over `--auto-duration` if both are given, with a warning).

### Prompt Relay (`--segment`)

Community port (WhatDreamsCost/Kijai) — sequence local prompts over time within one generation. A training-free additive Gaussian penalty on the video→text cross-attention (`attn2`) gates each local prompt's token range to a slice of the timeline; the global `--prompt` still applies to every frame.

```bash
ltx-2-mlx generate --distilled --prompt "cinematic, a woman in a bedroom" \
  --segment "sitting on the bed" \
  --segment "standing and turning to the window" --frame-rate 24 -o out.mp4
```

Flags: `--segment "TEXT" [LEN_FRAMES]` (LEN_FRAMES in latent frames; repeatable, timeline order; omit LEN to auto-distribute), `--relay-epsilon` (default 1e-3, smaller = sharper), `--relay-strength` (default 1.0). Works on all `generate` modes; on CFG modes the mask applies to the **conditional pass only** (never the negative). **Not compatible with modality tiling** (raises). Correctness hinges on the Gemma connector front-packing valid tokens to column *i* (`_replace_padding_with_registers`) — token *i* in encode order → column *i* in the `Nk` axis. Inert on the default path (no `--segment` → `video_cross_attention_mask=None`, byte-identical output). Key files: `conditioning/prompt_relay.py`; `video_cross_attention_mask` kwarg threaded `LTXModel → BasicAVTransformerBlock → attn2`.

**On 2.5 packs.** The Gemma-4 text encoder tokenizes with the pack's HuggingFace `tokenizer.json`, which (unlike the
mlx-lm Gemma-3 tokenizer of the 2.3 packs) adds no `<bos>` and no `<eos>`; `map_token_ranges` measures with the same
`encode` the encoder runs, so its ranges are the encoder's columns either way. `tests/test_prompt_relay_ltx25.py`
(runs when the local pack is found, like the other `LTX25_Q8_DIR` tests) pins that chain on the real tokenizer: each
range decodes to exactly its local prompt (quoted speech, non-ASCII, repeated spaces), the ranges index the valid
tokens `Gemma4TextEncoder.tokenize` left-pads, and after `_replace_padding_with_registers` the mask's penalised
columns are exactly those tokens. It fails if the encoder starts prepending `<bos>` (upstream's `LTXGemmaTokenizer`
does on Gemma 4; this port's encoder does not), which would shift every range by one column.

Validated end to end on the 2.5 q8 pack (M4 Pro 48 GB, `--distilled` and `--dfr`, 512×768×97 = 13 latent frames,
seed 5, global "a woman in a kitchen, static camera" + `--segment` "she smiles at the camera" / "she turns around and
walks away toward the window"; the no-relay arm encodes the same combined text as one `--prompt`). A face detector
(insightface, score ≥ 0.6) gives the last frame where her face is visible:

| run | segment lengths (latent frames) | last frame with her face | stage-1 / stage-2 s/step | peak footprint |
|---|---|---:|---:|---:|
| `--distilled`, no relay | — | 46 | 7.23 / 29.25 | 25.7 GB |
| `--distilled` | 7 / 6 (auto) | 64 | 7.34 / 29.65 | 25.7 GB |
| `--distilled` | 3 / 10 | 36 | 7.36 / 29.15 | 25.7 GB |
| `--distilled` | 10 / 3 | 69 | 7.36 / 29.15 | 25.7 GB |
| `--dfr`, no relay | — | 65 | 8.87 / 46.75 | 39.0 GB |
| `--dfr` | 7 / 6 (auto) | 65 | 8.86 / 47.30 | 39.5 GB |
| `--dfr` | 3 / 10 | 47 | 8.84 / 47.00 | 39.2 GB |

The turn follows the segment lengths (36 → 64 → 69). It lands where the second segment's plateau starts, not on the
nominal boundary: with the default `--relay-epsilon` both local prompts are strongly penalised in the latent frames
between the two plateaus (half-width `L // 2 - 2`), so the global prompt alone drives that stretch. Without relay the
2.5 model already plays the two sentences in order (its native multishot) and turns at frame 46; on `--dfr` the auto
split turns at frame 65 with or without relay, and a 3 / 10 split moves the turn to frame 47. Overhead: one mask build per stage (≈3 ms, 10 MB at 4,992 stage-2
tokens), within the 1 s resolution of the step timer.

### Negative prompt without CFG: NAG (`--nag`)

Experimental. Normalized Attention Guidance ([paper code](https://github.com/ChenDarYen/Normalized-Attention-Guidance),
LTX-2 port: kijai's `LTX2_NAG` node in `ComfyUI-KJNodes/nodes/ltxv_nodes.py`) applies a negative prompt on the distilled
paths, which run at CFG 1 and have no unconditional pass to extrapolate from. Inside every text cross-attention
(`attn2`, and `audio_attn2` unless `--nag-video-only`) the queries attend to the positive and to the negative context;
the two outputs are combined per token before the per-head gate and `to_out`:

```
z~ = z+ * s - z- * (s - 1);  r = ||z~||_1 / ||z+||_1 (over all heads);  z^ = z~ * min(1, tau / r);  z = alpha * z^ + (1 - alpha) * z+
```

```bash
ltx-2-mlx generate --distilled --nag --negative-prompt "mouth opened very wide, exaggerated mouth movements" \
  -p "..." -H 768 -W 512 -f 97 --frame-rate 24 -o out.mp4
```

Flags: `--nag` (needs `--negative-prompt`; `--distilled` and `--dfr` only, refused on the CFG modes), `--nag-scale` (11.0,
`>= 1`, `1` = off), `--nag-alpha` (0.25), `--nag-tau` (2.5), `--nag-video-only`; the defaults are the node's. Without
`--nag`, `--negative-prompt` stays refused on the distilled paths (it is not silently given another mechanism), and both
`nag` kwargs stay `None`: renders are byte-identical. Python: `generate_two_stage(..., negative_prompt=..., nag=NAGConfig())`
on `DistilledPipeline` / `DFRPipeline`. Every pass of the run is guided: both stages, and on `--dfr` the temporal rounds
and the spatial epilogue.

How it runs: the negative prompt is encoded once per run (`BasePipeline._encode_nag`, right after the prompt, +5.5 s on
the 2.5 pack) and travels as one `NAGGuidance` (a `NamedTuple`, so it crosses the `--low-ram` compiled block) through
`Stage1Result.nag` → the Euler / ancestral loops → `LTXModel` → `BasicAVTransformerBlock` → `Attention`
(`nag_encoder_hidden_states`). The block gives the negative context the same per-step prompt AdaLN modulation as the
positive (`prompt_scale_shift_table`), then the attention projects its K/V (once per block per step; the modulation
depends on sigma, so it cannot be hoisted) and runs a second `scaled_dot_product_attention` with the same queries and
**no mask** (Prompt Relay gates the positive prompt only). The combine runs in float32 whatever the compute dtype and
returns the attention's dtype. The loops drop the audio negative for an absent or frozen audio stream. Sol
(`LTX2_SOL_TAU`) is untouched: it covers `attn1` (141 stage-2 calls ran sparse with and without NAG).

Differences from the node, deliberate: (1) the node hands its patched forward the raw connector output, so on 2.3 / 2.5
checkpoints (`cross_attention_adaln`) its negative skips the modulation the positive gets; here both are modulated (on
the mouth test below the node's variant reduced the mean opening less, 0.243 → 0.172 against 0.145, and its largest
opening, 0.457, exceeded the unguided render's); (2) float32 combine (an L1 sum over 4,096 features overflows float16);
(3) `scale == 1` is off (the paper's `nag_scale > 1` gate) where the node uses `0`; (4) no NAG with CFG (the node allows
both), the audio negative comes from the same prompt (the node takes an optional separate audio conditioning).

Validated on the 2.5 q8 pack (M4 Pro 48 GB, AC power, `--distilled` 768×512×97, one run per arm). Mouth opening = inner-lip
gap / distance between the eye centres (insightface 3D-68 landmarks, every frame), negative `"mouth opened very wide,
exaggerated mouth movements, screaming"`:

| prompt, seed | off: mean / max / frames > 0.30 | NAG: mean / max / frames > 0.30 | Whisper WER off / NAG | SyncNet LSE-C off / NAG |
|---|---|---|---|---|
| man laughing "No way, we actually did it!", 5 | 0.243 / 0.435 / 19.6 % | 0.145 / 0.352 / 3.1 % | 0 / 0 | 6.14 / 4.75 |
| same, 11 | 0.169 / 0.298 / 0 % | 0.099 / 0.260 / 0 % | 0 / 0 | 4.66 / 4.23 |
| vlogger "Hi everyone, welcome back…", 5 | 0.199 / 0.458 / 15.5 % | 0.185 / 0.433 / 12.4 % | — | — |
| same, 11 | 0.222 / 0.441 / 22.7 % | 0.136 / 0.320 / 3.1 % | — | — |

The opening shrinks on all four; the speech is unchanged and the lips still follow it, with a lower sync confidence (smaller
mouth movements). A hands prompt (counting on the fingers, negative `"extra fingers, missing fingers, fused fingers,
distorted hands"`) changed the gestures but not the motion-blurred, merged fingers: NAG does not fix hands here. No harm on
an ordinary clip (kitchen dialogue, negative `"blurry, distorted hands, extra fingers, deformed face"`, seed 5): WER 0 in
all three arms (off / NAG / NAG video-only), LSE-C 6.94 / 7.10 / 6.83, the same audio level (−27.7 / −26.7 / −27.8 dB);
the NAG render is a different sample (lighter, softer look; MUSIQ 73.3 / 65.1 / 67.8, DOVER 0.859 / 0.854 / 0.860; `--nag-scale 5`
scores the same MUSIQ, 65.0). On seed 11 the words were the same and the NAG render was louder (−27.1 → −20.7 dB).

Cost (steps 2+, the step timer has 1 s resolution):

| run | stage 1 s/step off → NAG | stage 2 s/step off → NAG | whole render | peak footprint |
|---|---:|---:|---:|---:|
| `--distilled` (float32 compute) | 7.36 → 8.36 (+14 %) | 29.20 → 31.25 (+7 %) | 175 → 194 s | 25.7 → 25.6 GB |
| `--nag-video-only` | → 8.10 | → 30.85 | 193 s | 25.8 GB |
| `LTX2_COMPUTE_DTYPE=float16` | 6.21 → 7.24 | 24.95 → 26.60 | 154 → 171 s | 24.9 → 25.4 GB |
| `LTX2_SOL_TAU=1.0,1.25,1.5` | 7.36 → 8.36 | 26.45 → 28.50 | 167 → 199 s | 25.8 → 25.8 GB |
| `--low-ram` | 7.41 → 8.46 | 29.20 → 30.80 | 171 → 192 s | 22.4 → 23.2 GB |
| with `--segment` (Prompt Relay; measured with the audio mask on as well) | 7.34 → 8.36 | 29.60 → 31.65 | 176 → 195 s | 25.7 → 25.6 GB |
| `--dfr` 768×512×49 | 5.09 → 6.10 | 23.95 → 26.05 | 149 → 160 s | 39.3 → 39.0 GB |

The extra work is a second K/V projection of the 1,024-token text context and a second attention per cross-attention call,
so its share falls as the video grows: one `attn2` call (micro-benchmark, random q8 weights) costs +17.7 ms at 1,248
video tokens (×1.57), +36 ms at 4,992 (×1.41), +99 ms at 17,856 (×1.35), i.e. +0.85 / +1.7 / +4.8 s per 48-block forward;
`audio_attn2` +3–4 ms. Float16 margin on real weights: the largest |value| entering or leaving the combine was 434, with
no non-finite output; the tau clip acted on 26–38 % of the tokens.

Key files: `guidance/nag.py` (`NAGConfig`, `NAGGuidance`, `normalized_attention_guidance`), `model/transformer/attention.py`
(`nag_encoder_hidden_states`), `transformer.py` (modulation), `utils/samplers.py` (`_nag_for_states`), `_base.py`
(`resolve_distilled_nag`, `_encode_nag`). Tests: `tests/test_nag.py`.

### Multi-Anchor I2V (`--image` repeatable)

All `generate` modes (`--one-stage`, `--two-stage`, `--two-stages-hq`, `--distilled`, `--dfr`) support multiple `--image` flags. Each anchor takes `PATH FRAME_IDX STRENGTH` where `FRAME_IDX` is the **pixel frame index** (0-based; for a 97-frame video the last frame is 96).

- `frame_idx=0` → `VideoConditionByLatentIndex`: hard-replaces the first latent frame (strongly preserved)
- `frame_idx>0` → `VideoConditionByKeyframeIndex`: appends soft reference tokens at that temporal position
- Optional 4th value `CRF`: the H.264 CRF the image is re-compressed at before encoding (`0` = none). Omitted, it resolves from the checkpoint like upstream (`ImageConditioner.resolve_crf`, reading `model_version` from the VAE encoder's safetensors metadata): **33** before LTX-2.4 (2.3 packs carry no version → 33), **18** from 2.4 on (2.5 packs). Applies to every pipeline that takes images, `keyframe` included; an explicit CRF always wins.
- Image preprocessing mirrors upstream `load_image_and_preprocess` (`utils/media_io.py`): EXIF orientation + embedded ICC profile → sRGB, odd dims cropped to even, the CRF round trip encoded as 4:2:0 (`yuv420p`, bilinear RGB↔YUV, slice threads, as PyAV does upstream), then a PyTorch-style bilinear resize on floats (no antialias, no uint8 re-quantization) and `x / 127.5 - 1`. Until #188's audit we encoded 4:4:4 and resized with PIL Lanczos on uint8 (2.3 q8 latent corr 0.918 vs upstream on a 1672×941 image; 0.9993 after). Control / reference videos (`ic-lora`, `lipdub`, the `ic-lora` attention-mask video) go through the same resize: `media_io.decode_video_by_frame` (native size, no autorotate, like PyAV) + `media_io.video_preprocess` (upstream names), so a control whose aspect differs from the render is center-cropped, not stretched (#194; ffmpeg's `scale` filter used to stretch it, bicubic, on uint8). Torch golden: `tests/parity_video_preprocess_reference.py`. Torch golden: `tests/parity_image_preprocess_reference.py`.
- `FRAME_IDX` can also be `last` (or `end`) or a negative number counted back from the end (`-1` = last frame). `resolve_frame_indices` (`utils/args.py`) turns it into a pixel index once the pipeline knows the frame count, so an end anchor stays on the final frame when `--auto-duration` picks the length (a fixed `96` would not). Resolved in every pipeline that takes `--image`, before the keyframe-token bookkeeping that tests `frame_idx > 0`. On `--dfr` it is resolved against the requested length before the clip is padded to whole keyframe segments (`resolve_stage1_frames`, `distilled.py`), so `last` lands on the final frame of the trimmed output. `-num_frames` resolves to `0` and therefore gets the hard first-latent replace above, not a soft keyframe. An index outside the clip (before frame 0, or `>= num_frames`) raises a `ValueError` that names the image.

```bash
# Anchor both ends — model animates the transition
ltx-2-mlx generate \
  --prompt "woman stands up from chair and walks to kitchen sink" \
  --two-stage \
  --image sitting.jpg 0 1.0 \
  --image standing.jpg 96 1.0 \
  --image at_sink.jpg 136 1.0 \
  -f 137 --frame-rate 24 -o output.mp4

# Seamless loop — same image at start and end
ltx-2-mlx generate \
  --prompt "flowing water rippling" \
  --two-stage \
  --image frame.jpg 0 1.0 \
  --image frame.jpg 96 1.0 \
  -f 97 --frame-rate 24 -o loop.mp4

# End anchor with a predicted length (2.5 packs): `last` follows the frame count
ltx-2-mlx generate \
  --prompt "a door slowly swings shut" \
  --distilled \
  --image open.jpg 0 1.0 \
  --image closed.jpg last 1.0 \
  --auto-duration 2:6 --frame-rate 24 -o door.mp4
```

**Mode recommendations for multi-anchor:** `--two-stage` or `--two-stages-hq` (dev model + CFG) respects anchors most faithfully. `--distilled` (8 steps, no CFG) also honors them — soft keyframe anchors are hints, not law, so the model may drift from them at longer durations, but a distilled start+end smoke test (512×512×25) tracked both anchors cleanly, and so did a 768×512×97 comparison against `keyframe` on 2.5 (see the `keyframe` card in [docs/PIPELINES.md](docs/PIPELINES.md#keyframe)). `--one-stage` works but is slower than `--two-stage` at large resolutions.

**Frame count grid:** frame counts live on the 8k+1 grid; an off-grid request is floored to it with a warning before any latent is sized (`snap_num_frames`, #177; e.g. 87 → 81). `hdr-ic-lora` instead rejects an off-grid source. Valid counts: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, …

### Audio-to-Video Example

```bash
# A2V with reference image
ltx-2-mlx a2v \
  --prompt "a singer performing" \
  --audio music.wav --image photo.jpg --frame-rate 24 -o output.mp4
```

Flags: `--audio` (required), `--audio-start` (default 0 s), `--frame-rate` (required, mirrors upstream `frame_rate=`), `--image` (optional I2V), `--cfg-scale` (default 3.0), `--stg-scale` (default 1.0), `--negative-prompt`, `--stage1-steps` (default 30), `--stage2-steps` (default 3), `--enable-teacache` / `--teacache-thresh` (LTX-2.3 packs), `--distilled` (experimental, see below).

```bash
# Distilled A2V: no dev model, no CFG (Lightricks' LTX-2.5_A2V_Two_Stage_Distilled workflow)
ltx-2-mlx a2v --distilled \
  --prompt "a woman talks to the camera" \
  --audio vocals.wav --image photo.jpg -H 1280 -W 704 -f 121 --frame-rate 24 -o output.mp4
```

`A2VidDistilledPipeline` (`a2vid_distilled.py`) is `DistilledPipeline` with two hooks swapped: `_stage1_audio_state` / `_stage2_audio_state` return the encoded input track as a frozen state (`create_noised_state(..., frozen=True)`, zero mask, per-modality sigma 0) instead of noise / the re-noised stage-1 audio, as upstream's `freeze_audio=True` does in both stages. Stage 2 re-attaches the source tokens rather than stage 1's output (identical anyway: a frozen stream comes out of both loops bit for bit). The default hook bodies are the old inline calls, so `generate --distilled` and `--dfr` are byte-identical. The audio encoding and the source-waveform mux are shared with `a2v` (`encode_source_audio` / `decode_with_source_audio` in `a2vid_two_stage.py`). Flags: the `a2v` ones minus `--cfg-scale`, `--stg-scale`, `--negative-prompt` and `--enable-teacache` (rejected up front); `--stage1-steps` defaults to 8. Validated (M1 Max, 2.5 q8, 7 Oct): `generate --distilled`, `--distilled --low-ram`, `--dfr` and `a2v` are byte-identical (sha256) to main at 512×768×49, seed 5; `a2v --distilled` takes 109 s against 631 s for `a2v` at that size (default environment), 87 s with `--low-ram` (fp16), ~400 s at 704×1280×121 and 942 s at 704×1280×241 with three image anchors (fp16).

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

Flags: `--steps` (default 30), `--cfg-scale` (default 3.0), `--stg-scale` (default 1.0), `--no-regen-audio` (retake only), `--distilled` (both; see below), `--image` (`extend --distilled` only).

```bash
# Distilled retake: upstream RetakePipeline's default mode (distilled model, no CFG)
ltx-2-mlx retake --distilled \
  --prompt "a different action" \
  --video source.mp4 --start 2 --end 5 -o retake.mp4
```

`RetakePipeline(distilled=True)` encodes the prompt alone (`reject_negative_prompt` refuses a negative one), loads the distilled transformer the way `DistilledPipeline.load` resolves it (`transformer.safetensors`, else `transformer-distilled*.safetensors`, through `_load_transformer_with_optional_streaming`, so `--low-ram` and `LTX2_COMPUTE_DTYPE` apply) and runs `denoise_loop` on `shorten_schedule(DISTILLED_SIGMAS, steps, keep="start")`, like upstream's distilled retake (`DISTILLED_SIGMAS` + `SimpleDenoiser`, deterministic Euler; upstream does not switch retake to the ancestral sampler on 2.5 checkpoints, and an ancestral run of the same retake measured no difference). The window mask, the source noising and `--no-regen-audio` (audio frozen) are shared with the dev path, which is unchanged (`_guided_denoise` holds its guider setup). `RetakePipeline.extend` raises on a distilled pipeline; the distilled extend is `ExtendDistilledPipeline` (below). CLI: `--cfg-scale`, `--stg-scale` and `--negative-prompt` are rejected before anything loads; `--steps` defaults to the whole table (8). `LTX2_SOL_TAU` does not apply (one stage, no stage 2). Retake re-rolls the window; it follows a new action in the prompt only weakly (not a semantic editor). Validated (M1 Max, 2.5 q8, 7 Oct): at 512×768×49, seed 5, `retake --distilled` takes 148 s (14.4 s/step, Metal peak 23.4 GB) against 1793 s for `retake`, 120 s with `LTX2_COMPUTE_DTYPE=float16` (11.6 s/step), 148 s with `--low-ram` (22.4 GB footprint); 704×1280×121 float16: 695 s (76 s/step, 30.8 GB Metal peak while denoising), 716 s with `--low-ram` (9.5 GB while denoising). On an M4 Pro 48 GB the same 704×1280×121 float16 retake runs resident in 770 s.

```bash
# Distilled extend: continue a clip on the distilled two-stage path (upstream chunk continuation)
ltx-2-mlx extend --distilled \
  --prompt "what happens next" \
  --video source.mp4 --extend-frames 12 -o longer.mp4
```

`ExtendDistilledPipeline` (`extend_distilled.py`) mirrors upstream's chunk continuation (`ltx_pipelines.chunks`, used by `DistilledPipeline.stream_chunks`): a new window of `4 + extend_frames` latent frames starts with the previous window's last `next_video_carry_frames` (25 pixel frames = 4 latent frames) of video latent and the matching audio tokens (`round(25 / fps × 25)`) pinned at index 0, strength 1.0, in both stages (`VideoConditionByLatentIndex` / `AudioConditionByLatentIndex` there; here `VideoConditionByLatentIndex` over the patchified tokens for both). The previous window is the source clip, encoded at full resolution (the stage-2 carry and the output's first part) and at half resolution (the stage-1 carry; upstream's stage-1 carry is the previous stage-1 output, which a finished clip does not have). The rest is `DistilledPipeline`: `_stage1` (a new hook, `_stage1_video_conditionings`, appends the pinned tail; it returns nothing by default, so every other distilled path is unchanged), `_upsample_latent`, `_stage2` with the full-resolution tail as `extra_conditionings`, and the `_stage1_audio_state` / `_stage2_audio_state` hooks for the audio tail; sigmas, the ancestral sampler on 2.5, `--low-ram`, `LTX2_COMPUTE_DTYPE` and `LTX2_SOL_TAU` (stage 2) carry over. The window's first 4 latent frames are the source's last 4, so its other frames are appended to the source latent and the whole clip is decoded once (upstream decodes each chunk and crossfades the overlap; here the decoder sees one continuous latent). CLI: `extend --distilled`; `--direction before`, `--steps`, `--cfg-scale`, `--stg-scale` and `--negative-prompt` are rejected before anything loads. `--image PATH [FRAME STRENGTH [CRF]]` (repeatable; the dev `extend` rejects it) anchors the new frames, as upstream passes images to each chunk: FRAME counts the appended frames (`0` = first new frame, `last` / `-1` = the last), `resolve_frame_indices` resolves it against `8 × extend_frames`, then it is shifted by the 25 carried frames and handed to `_stage1` as `images`, so every anchor is a `VideoConditionByKeyframeIndex` guide, re-encoded at full resolution in `_stage2`, its tokens cut before decoding (as on `generate --distilled`). Without `--image` `_stage1` gets `images=None` exactly as before. The source size must be on the two-stage grid (multiples of 64). Validated (M1 Max, 2.5 q8, 8 Oct): `extend` (dev + CFG), `generate --distilled`, `--dfr` and `retake --distilled` byte-identical (sha256) at 512×768×49, seed 5. `extend --distilled` at 512×768×49 + 2 latent frames: 103 s against 2398 s for `extend` (default environment). 704×1280×241 + 12 latent frames, float16: 589 s (source encode 79 s, Metal peak 32.6 GB; stage 1 145 s; stage 2 252 s, 80 s/step; decode 108 s); 693 s with `--low-ram` (same bytes, 31.7 GB footprint against 56.6 GB); stage-2 step 80 → 61 s with `LTX2_SOL_TAU=1.0,1.25,1.5`. `extend` on a 5 s cut of the same clip would denoise for ~5 h 50 min (176 s/forward × 120) against 502 s for the whole `extend --distilled` run. The frame change at the join is 6.3 against the clip's 95th percentile of 6.8 (5 s source: 3.1 against 5.8). The model sees only the 25 carried frames: on the 5 s source, whose last second shows her looking down, the face that comes up afterwards is a different-looking face (ArcFace to the reference 0.42 against 0.64 before). With that clip's frame 0 (front-facing) as an anchor at the last new frame, strength 0.7: new part 0.55 (0.65–0.78 from 1 s after the join on), Gemini "same woman" 7/10 instead of 2/10; on the 10 s clip 0.67 against 0.61, earrings and hair colour kept. No burn (saturation +4 %, contrast unchanged, brightness −5 % over the last half second); one frame change about twice its neighbours at the start of the last latent frame (8.7 and 8.3 against a p99 of 7.6 and 7.3). The anchored runs took 544 s and 612 s against 502 s and 589 s.

Source encode (`encode_video_tensor` / `encode_source_audio_latent` in `retake.py`: retake, extend and `extend --distilled`): the video and audio latents are evaluated while their encoder is loaded, before it is freed (`mx.synchronize` used to leave the VAE encode lazy, so it ran inside the first denoising step with the encoder, the source pixels and the DiT resident together). A source larger than one `TilingConfig.default()` tile (768/64 px, 80/24 frames) is encoded with `tiled_encode`, as upstream (`video_latent_from_file` → `tiled_encode(TileSizeConfig.default())`); a source that fits one tile keeps the untiled encode, the same computation. Sources larger than one tile therefore render differently from 0.16.2 (tile blending); sources within one tile are byte-identical. `VideoEncoder.tiled_encode` evaluates its accumulators after each tile instead of scheduling every tile in one graph at the end (same latents bit for bit; also used by `hdr-ic-lora`). Measured on an M1 Max 64 GB (2.5 q8): `retake` at 512×768×49 is byte-identical (sha256) to before, peak memory footprint 52.9 → 39.7 GB; the tiled encode of a 704×1280×121 source peaks at 31.2 GB Metal (38.5 GB without the per-tile evaluation; 704×1280×49: 31.4 GB untiled, 23.7 GB tiled).

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

Flags for `preprocess`: `--height`, `--width` (resize, must be divisible by 32), `--max-frames` (default 97), `--captions` (directory with .txt files matching video stems), `--caption-ext`, `--with-audio` (also encode audio latents for joint audio-video training), `--frame-rate` (override the fps probed per clip), `--gemma`.

Flags for `train`: `--config` (required, path to YAML config), `--low-ram` (gradient checkpointing; fits the dev model on 64 GB). See `packages/ltx-trainer/configs/` for examples.

**Programmatic hooks** (`LtxvTrainer(cfg).train(...)`, `ltx_trainer_mlx/trainer.py`):

- `step_callback: StepCallback` — `(step, total_steps, sampled_video_paths)` once per optimizer step, after validation/checkpointing. Arity is fixed at 3 (downstream apps pin it); exceptions propagate and abort the run.
- `metrics_callback: MetricsCallback` — `Callable[[StepMetrics], None]`, once per optimizer step right after the update + LR-scheduler tick (before validation/checkpointing). `StepMetrics` (frozen dataclass): `step` (1-based), `total_steps`, `loss` (mean of the step's gradient-accumulation micro-batch losses), `lr` (rate the optimizer applied for this update, read before `update`), `step_time_s` (first micro-batch forward to materialized update), `peak_memory_gb` (`mx.get_peak_memory()` in GiB: MLX high-water mark, not RSS; can exceed physical RAM). No extra GPU sync: the loop already materializes and `.item()`s every micro-batch loss. A raising `metrics_callback` is logged (warning + traceback) and disabled for the rest of the run; training continues. Tests: `tests/test_trainer_metrics_callback.py` (tiny real loop, no weights).

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

**`stg_scale` defaults differ by pipeline**: the dev + CFG pipelines (`--two-stage`, `--one-stage`, `a2v`, `keyframe`, `retake`/`extend`) default to `stg_scale=1.0` (`ti2vid_two_stages.py:417`, `retake.py:64` `DEFAULT_STG_SCALE`; guidance params in `utils/constants.py`); `--two-stages-hq` defaults to `stg_scale=0.0` (`ti2vid_two_stages_hq.py:113`). STG requires a 3rd forward pass per step. On 32GB Mac, this causes OOM for videos longer than ~33 frames at 480x704. Pass `--stg-scale 0` on 32GB machines for long clips.

**Memory impact**: Each extra pass doubles/triples/quadruples memory. On 32GB Mac with dev model at 480x704: CFG-only supports ~97 frames at half-res (two-stage), full guidance (4 passes) supports ~17 frames.

**Distilled paths (CFG 1)**: no unconditional pass; a negative prompt acts only through NAG (`--nag`, inside the text cross-attentions; see "Negative prompt without CFG: NAG").

**STG perturbation masks**: Self-attention masks are 4D `(B,1,1,1)` for use inside attention where tensors are `(B,H,N,D)`. Cross-modal masks (A2V/V2A) are 3D `(B,1,1)` for use outside attention where outputs are `(B,N,dim)`. Mixing these up causes silent shape corruption via broadcasting.

---

## Keyframe Interpolation Pipeline

Two-stage pipeline requiring the dev (non-distilled) model + CFG. The distilled model hallucinates during interpolation.
On 2.5 packs, `generate --distilled` with a start and an end `--image` is a faster alternative that tracked both images in a 768×512×97 comparison (numbers on the `keyframe` card in [docs/PIPELINES.md](docs/PIPELINES.md#keyframe)).

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
- **Stage 2**: Dev + distilled LoRA fused, no CFG: Euler on `--two-stage`, res_2s on `--two-stages-hq` (as upstream; two forwards per step)
  - `STAGE_2_SIGMAS` (default 3 steps)
  - I2V conditioning re-encoded at full resolution
  - Denormalize → neural upsampler 2x → re-normalize before Stage 2
  - Audio: on `--two-stage`, stage 2 keeps stage 1's audio as a frozen conditioning stream (upstream v1.4.0 `freeze_audio=True`, #173); on `--two-stages-hq` stage 2 re-noises the stage-1 audio and refines it with the video, as upstream does

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
    enable_teacache=True,  # default False
    teacache_thresh=0.5,  # optional override
)
```

Equivalent CLI flag (works on `--two-stage`, `--two-stages-hq` and `a2v`, whose stage 1 is the same Euler + CFG loop and reuses the Euler coefficients via `TI2VidTwoStagesPipeline._make_stage1_teacache`; ~1.30× at 30 steps on a2v, where the default `stg_scale=1.0` also caches the STG pass; refused on LTX-2.5 packs):

```bash
ltx-2-mlx generate --prompt "..." --two-stage -f 97 --frame-rate 24 --enable-teacache -o out.mp4
ltx-2-mlx generate --prompt "..." --two-stages-hq -f 97 --frame-rate 24 --enable-teacache --teacache-thresh 1.0 -o out.mp4
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

**Same seed, different sample.** TeaCache skips steps early in stage 1, where the layout is decided, so a TeaCache render is **not** a faster copy of the plain render at the same seed. On unanchored T2V it comes out as a different scene: different subject, outfit and framing (2.3 q8, 384×576×25, seed 5, default thresh: SSIM 0.70 `--two-stage`, 0.73 `--two-stages-hq`). Anchors limit the drift: `keyframe` keeps both anchors and diverges in between (SSIM 0.74). A tight a2v close-up keeps the same face and framing (SSIM 0.89), and its lip-sync follows the speech/silence pattern exactly like the plain render. A wide a2v shot changes composition entirely. Each render is clean on its own. Don't use TeaCache to preview a seed you intend to finish without it.

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
- **Half-res dims floored to mult-32** (0.14.17): `gen_h = (height//2//32)*32` — raw `height//2` crashed the VAE encoder's `space_to_depth` for non-mult-64 targets with `--image` (incl. default 480). Latent-neutral; output snaps down at non-mult-64 dims (requested 480 high → 448 out).
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
- `generate --distilled` and `generate --dfr` (the DFR detailing LoRA attaches as a `BlockLoraSource`)
- `a2v` (audio-to-video) and `a2v --distilled`
- `keyframe` (interpolation)
- `ic-lora` (control video conditioning, via bind-time LoRA fusion)
- `hdr-ic-lora` (HDR LoRA as a `BlockLoraSource`)
- `retake` / `extend` (dev model + CFG; mirrors upstream RetakePipeline's `offload_mode`), `retake --distilled` and `extend --distilled`

`lipdub` inherits the `ic-lora` path (its LoRA attaches as a `BlockLoraSource`). Validated on 2.3 q8 with the DubIt LoRA (576×320, 73-frame speaking reference, seed 5): 334 s streamed vs 305 s resident, 42.6 dB between the two outputs (the compiled-block ULP difference); at this small size the run's peak footprint (13.4 GB both) is set by text encoding, not the transformer.

Validated runs on M2 Pro 32 GB:
- bf16 HQ at 480x704x97 (4 sec): 49:38 — would OOM without streaming.
- q8 HQ at 480x704x97 (4 sec): 44:38.
- q8 one-stage 480x704x33 (1.3 sec): 2:31.

For two-stage / HQ / `a2v` (dev path) / keyframe at default LoRA strength 1.0, the Stage 1 → Stage 2 transition swaps the streamer from ``transformer-dev.safetensors`` to the pre-fused ``transformer-distilled.safetensors`` (mlx-forge produces this at LoRA strength 1.0). For custom ``--distilled-lora-strength`` (or any non-1.0 strength), the streamer keeps the dev model and attaches a ``BlockLoraSource`` that fuses the LoRA delta at each ``bind()`` (dequantize → ``W + B @ A * strength`` → re-quantize for q4/q8).

For `ic-lora`, each control LoRA is attached as a ``BlockLoraSource`` to the streaming wrapper instead of being fused in-place at load time. Stage 2 just clears the source list rather than reloading the whole transformer.

`mx.compile` cannot trace `BatchedPerturbationConfig` (the dataclass passed for STG / modality-isolation passes), so `StreamingLTXModel.__call__` falls back to the eager block whenever ``perturbations`` is non-None. Eager + per-block sync still works thanks to `set_cache_limit(0)`.

### Limitations

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
- ``VideoModalityTiler(tiling, latent_shape, seams=())``: ``seams`` (interior latent frames) cut the temporal split on those cells with a rectangular mask — the lead-in is context only and the earlier tile keeps the seam cell (upstream ``seam_split`` / ``split_at_seams``); without interior seams or with one frame tile it keeps the count split with trapezoid ramps.
- ``VideoModalityTiler.tile_modality(modality, tile, normalize_positions=True) -> (Modality, TilingContext)``: slices token-level state to a tile (keep indices + per-cond-token blend weights ``1 / tiles keeping it``, from one keep table per call); ``normalize_positions`` subtracts the tile's generated interval start (upstream).
- ``TiledLTXModel(inner, tiler, normalize_positions=False)``: the ``--tile-*`` wrapper keeps positions global (default); the DFR spatial epilogue uses ``True`` like upstream.
- ``VideoModalityTiler.blend(tile_output, tile, ctx, output=None)`` accumulates the tile contribution into the full token buffer with trapezoidal blend masks at overlaps.
- ``TiledLTXModel`` wraps ``LTXModel`` (or ``StreamingLTXModel``); intercepts ``__call__``, builds Modality from kwargs, iterates tiles, blends video output, averages audio output across tiles. Pipelines stay unchanged.

### Position layout divergence (documented)

Upstream uses ``(B, num_axes, T, 2)`` interval positions per token; we use ``(B, T, num_axes)`` midpoints (consistent with the rest of our codebase). A conditioning token is kept by a tile iff its midpoint lies in the **closed** extent of the tile's exact generated intervals (temporal ``[max(0, 8f0−7), 8(f1−1)+1)/fps`` with the causal fix, fps from the frame-0 midpoint ``0.5/fps``; spatial 32-px cells), or its time is negative. For every token lattice we append — 1-frame keyframes / slots, 32-px cells, ×2 reference cells, 8-frame reference latents — this equals upstream's ``start < tile_end and end > tile_start`` exactly; it would differ for a reference ``downscale_factor >= 3`` or multi-frame keyframe tokens (no current caller). Before the fix the test used the generated tokens' midpoints as the extent, which dropped keyframe tokens on a tile's last seam and on the canvas's last frame, so conditioned ``generate --tile-*`` renders (keyframe anchors / slots, DFR references) change; unconditioned ones are byte-identical (e2e, 2.5 q8 `--distilled --tile-frames 2` 512×768×49: sha identical to main; with `--image … 48 1.0` — an anchor on the last frame — frame 48 matches the anchor at 38.4 dB vs 23.6 dB on main, where the tiler had dropped it). ``split_by_count`` now raises when the tile size is ``<= overlap`` (upstream guard): very small latents with ``--tile-*`` error instead of silently running untiled.

### CLI

| Flag | Default | Effect |
|------|---------|--------|
| `--tile-frames N` | 1 | Number of temporal tiles |
| `--tile-spatial M` | 1 | Number of spatial tiles per axis (M*M total) |
| `--tile-overlap K` | 2 | Token-grid overlap between tiles. Larger overlap = smoother blend but more redundant compute |

Total tiles = ``N * M * M``. Default ``1*1*1`` = no tiling.

Coverage: ``generate`` only (one-stage / ``--two-stage`` / ``--two-stages-hq`` / ``--distilled`` / ``--dfr``), via ``TI2VidTwoStagesPipeline.tile_count``. ``a2v``, ``keyframe``, ``ic-lora`` and ``hdr-ic-lora`` don't read ``tile_count``, so their CLIs don't register the ``--tile-*`` flags (they used to accept and silently ignore them).

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

Port of upstream v1.4 `ltx_pipelines.hdr_ic_lora.HDRICLoraPipeline`: a **single-stage ACEScct SDR-to-HDR** IC-LoRA on **LTX-2.5 packs only** (2.3 packs are refused at construction; there is no 2.3 HDR LoRA path any more). Subclasses `BasePipeline`; no Gemma, no audio, no upsampler, no prompt.

**Breaking (release notes):** `hdr-ic-lora` is now upstream's single-stage ACEScct SDR-to-HDR pipeline on LTX-2.5 packs; LogC3 / LTX-2.3 HDR and the `.hdr.npz` output are gone.

### Install and early refusals

The EXR writer needs the optional `hdr` extra: `pip install 'ltx-pipelines-mlx[hdr]'` (or `uv sync --extra hdr` in this repo); the HLG master needs an ffmpeg with the `libx265` encoder (Homebrew's has it). `HDRICLoraPipeline.__init__` checks, in order and **before the pack snapshot download**: OpenEXR importable + `libx265` listed by `ffmpeg -encoders` (`require_hdr_export_tools`), the `--text-embeddings` file loads, the LoRA resolves (a `.safetensors` path must exist locally, no HF call), and the pack is 2.5 (local dir checked in place; for a repo id only `embedded_config.json` is fetched). `generate` then refuses odd source width/height (4:2:0 master; upstream fails later in the encoder) and a short read (loaded frames != probed count; the count is decoded with `-count_frames` when the container has no `nb_frames`) before the DiT load.

### Flow

1. **Inputs**: `--hdr-lora` (`ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors` from the gated `Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR`; accept its licence once, then download it with `huggingface-cli download`) and `--text-embeddings` (the scene-emb `.safetensors` shipped in the same repo, key `video_context`). Both must be local files: anything else (e.g. a repo id) is refused before any download, with the exact `huggingface-cli download` command to run.
2. **Source**: MP4/MOV (`srgb_gamma` = display sRGB, EOTF applied; `srgb` = linear Rec.709) or an EXR-frame folder (`srgb` / `acescg` / `acescct`, `--frame-rate` mandatory). Converted to ACEScct by the input transform (`utils/hdr_media.py`). Frame count must be 8k+1; the output length matches the source.
3. **Conditioning order**: the VAE-encoded SDR clip at full resolution is the IC-LoRA reference (downscale 1, strength 1.0 by default). With seam keyframes on (default, `--keyframe-strength` 0.95) every DFR `resolve_canvas` x8 border gets a generated HDR slot plus a 1-frame SDR guide; `--no-keyframes` gives plain IC-LoRA.
4. **Denoise**: one stage, the 8-step distilled schedule, distilled transformer with the LoRA at fixed strength 1.0 (fused, or a `BlockLoraSource` under `--low-ram`). `--high-quality` duplicates each conditioning frame, generates 2N-1 frames and keeps every other one.
5. **Precision**: both VAE ends (image encoder, diffusion video decoder) run in **fp32**, the DiT in bf16 (upstream `vae_dtype = float32`); the denoised latent is cast to fp32 explicitly before the decode.
6. **Decode**: the keyframe-aware **diffusion** decoder streams chunks (`iter_frames`); the DiT is freed first.

### Outputs

- `<output>.mp4`: BT.2020 / HLG 10-bit HEVC master (always, whatever `--exr-colorspace`).
- `<stem>_<exr-colorspace>_exr/frame_*.exr`: EXR sidecar beside it (scene-linear ACEScg by default, `srgb_linear`, or `acescct` log codes).

### Status

Experimental (see docs/PIPELINE_MATURITY.md). Unit tests run without weights (`tests/test_hdr_ic_lora.py`, `tests/test_hdr_media.py`).

**Validated** on real weights (M2 Pro 32 GB, LTX-2.5 q8, `--low-ram`, seed 5, LoRA `ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0` + its scene-emb file, a sunlit-kitchen SDR clip at 768×512, 24 fps):

| run | wall-clock | denoise | decode (fp32 diffusion) | peak footprint |
|---|---:|---:|---:|---:|
| 49 frames, seam keyframes (default; 6912 tokens) | 1097 s | 8 × 83 s | 419 s | 13.4 GB |
| 49 frames, `--no-keyframes` (5376 tokens) | 761 s | | | |
| 25 frames, `--high-quality` (49 generated, 6144 tokens; 25 written) | 1013 s | | 412 s | 13.4 GB |

- HLG master: HEVC `yuv420p10le`, `bt2020` / `arib-std-b67` / `bt2020nc`, every frame. Its inverse-OETF luminance correlates 0.98 with the EXR frame (converted to Rec.2020).
- EXR (ACEScg): no NaN, min ≥ 0. Default run: max 13.0, 15 % of pixels above 1.0. The source's near-white areas (the window) average 3.25 against 0.42 elsewhere: real highlights. `--no-keyframes`: max 10.1, window 1.86. `--high-quality`: max 23.1, window 4.16.
- Same picture as the SDR source. The LoRA lifts mid-tones by ~1.3× against the source's sRGB-EOTF linear (median ratio 1.36 default, 1.25 without keyframes); with that gain compensated, clipped mid-tones sit at 22.9 dB / 25.1 dB PSNR.
- The official scene-emb file stores `video_context` as `(1024, 4096)` with no batch axis; `load_video_context` adds it.

### Key Files

- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/hdr_ic_lora.py` — `HDRICLoraPipeline`, `dfr_seam_roles`, `load_video_context`.
- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/hdr_media.py` — input/EXR/HLG media IO (`VideoInput`, `EXRVideoInput`, `EXRColorSpace`, `encode_hdr_outputs`).
- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/cli.py` — `hdr-ic-lora` parser, `_resolve_hdr_input` (upstream messages verbatim), `_cmd_hdr_ic_lora`.

---

## LTX-2.5

`generate --distilled --model <2.5-pack-dir>` runs end-to-end on a local
LTX-2.5 pack. No new CLI flag — generation is auto-detected from the pack's
`embedded_config.json` (`ff_bias is False` ⇔ 2.5) via
`is_ltx25_pack()`/`LTXModelConfig.from_checkpoint_dir()`, resolved once in
`TI2VidTwoStagesPipeline.__init__` (declared as `_is_25: bool = False` on
`BasePipeline`, inherited by `DistilledPipeline`) as `self._is_25`. The pack carries its own
Gemma-4 text encoder (`text_encoder.safetensors` + `text_encoder_config.json`,
selected by `select_text_encoder()`) — no `mlx-community` Gemma 3 download on
that path. 2.3 packs are byte-identical to before.

### Sampler

- Stage 1: `euler_ancestral_denoising_loop` (`EulerAncestralDiffusionStep(eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE)`, 8 steps) on `LTX_2_5_DISTILLED_SIGMAS`.
- Stage 2: also `euler_ancestral_denoising_loop` (same eta / s_noise, `STAGE_2` renoise) on `LTX_2_5_STAGE_2_DISTILLED_SIGMAS`, as upstream since v1.4.0 (it used to keep stage 2 deterministic). This also covers DFR's stage 2, which reuses `_stage2`.
- Ancestral noise is seeded from `seed + ANCESTRAL_NOISE_SEED_OFFSET` (10000) for stage 1 and `seed + ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET` (20000) for stage 2, to decorrelate both from the initial-latent draw and from each other. 2.3 packs stay deterministic on both stages.
- Stage 2 upscaler resolves to `spatial_upscaler_x2_v1_0.safetensors` (vs `v1_1` on 2.3), falling back to the 2.3 stems; hard error only when none exists (#42 style).

### Auto-Duration (`DurationHead`, `-f` optional on 2.5)

2.5 packs ship a `duration_head.safetensors` (`packages/ltx-core-mlx/src/ltx_core_mlx/duration_head/`)
that predicts a clip length in seconds from the encoded prompt (+ image,
when present). `generate`'s `-f/--frames` default changed from a hardcoded
`97` to `AutoDuration()` (`DEFAULT_AUTO_DURATION`, clamp `[1.0, 20.0]`
seconds) across the five `generate` modes (`--one-stage`, `--distilled`,
`--two-stage`, `--two-stages-hq`, `--dfr`). `keyframe`, `a2v` and `ic-lora`
keep `-f 97`. `retake`, `extend`, `lipdub` and `hdr-ic-lora` take their
length from the source video and have no `-f`.

- **On 2.5 packs**: omitting `-f` predicts the duration right after prompt
  encoding (`DurationPredictor.from_checkpoint(self.model_dir)`, built once
  in `BasePipeline.__init__`) and snaps it to the model's frame grid
  (`(num_frames - 1) % 8 == 0`). A `[auto-duration] predicted N frames
  (S.SSs @ FPS fps)` line prints to stderr when a prediction actually runs.
  `--auto-duration MIN:MAX` overrides the clamp range (seconds); explicit
  `-f` always wins over `--auto-duration` if both are given, with a stderr
  warning.
- **On 2.3 packs** (no `DurationHead` weights): omitting `-f` raises
  `ValueError: ... Pass num_frames explicitly.` **immediately** —
  `require_num_frames_source()` runs before any Gemma load or other work,
  so there's no wasted encode/download on the failure path. `-f` is
  effectively mandatory on 2.3.

```bash
# 2.5: let the DurationHead pick the length from the prompt
ltx-2-mlx generate --model /path/to/ltx-2.5-mlx-q8 --distilled \
  -p "a heavy wooden door creaks slowly open" -H 512 -W 512 --frame-rate 24 -o out.mp4

# 2.5: clamp the predicted duration to 2-4 seconds
ltx-2-mlx generate --model /path/to/ltx-2.5-mlx-q8 --distilled \
  -p "a heavy wooden door creaks slowly open" -H 512 -W 512 --frame-rate 24 \
  --auto-duration 2:4 -o out.mp4

# 2.3: -f is required — this fails fast with no Gemma/network activity
ltx-2-mlx generate --model dgrauet/ltx-2.3-mlx-q8 --distilled \
  -p "a heavy wooden door creaks slowly open" -H 512 -W 512 --frame-rate 24 -o out.mp4
# ValueError: num_frames was AutoDuration but this checkpoint has no DurationHead
# weights to auto-predict duration from (DurationHead ships from LTX 2.5 / gemma4
# onward). Pass num_frames explicitly.
```

Key files: `packages/ltx-core-mlx/src/ltx_core_mlx/duration_head/duration_head.py`
(`DurationHead`, `load_duration_head`); `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/types.py`
(`AutoDuration`, `DEFAULT_AUTO_DURATION`); `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/blocks.py`
(`DurationPredictor`, `require_num_frames_source`, `resolve_num_frames`,
`seconds_to_clamped_num_frames`); `_base.py::BasePipeline._require_num_frames_source`
/ `_resolve_num_frames`; `cli.py::_parse_auto_duration` / `_resolve_num_frames_arg`.

### Generated keyframe slots (`--num-generated-keyframes N`, 2.5 packs)

Port of upstream ``VideoGeneratedKeyframeSlots`` + the keyframe absolute-position
embedding. Extra single-pixel-frame token slots are appended to the stage-1
sequence at evenly spaced interior pixel frames (``linspace(0, F-1, N+2)``
rounded, endpoints excluded) with ``denoise_mask=1`` and a temporal RoPE span
of exactly one pixel frame; the model generates their content and conditions
the surrounding video on them, which relaxes the effective 8× temporal
compression where motion is fast. Cost: one latent frame of tokens per slot.
Stage 2 needs no slots (the effect is baked into the stage-1 latent). The
denoised slot content is extracted as ``(B, C, K, H, W)`` into
``BasePipeline.generated_keyframes`` before the conditioning tokens are cut; the
standard pipelines don't decode it (DFR does, with `--video-decoder diffusion`).

**Keyframe marker on every 2.5 render.** Upstream marks the target's *first
latent frame* in ``LatentState.keyframes_mask`` unconditionally (the causal
encoder makes it cover 1 pixel frame) and adds the learned
``keyframes_abs_pos_embedding`` to marked tokens right after ``patchify_proj``.
The 2.5 packs carry a small non-zero embedding (norm ≈ 0.05); before this port
we loaded it and never applied it, so 2.5 renders were slightly off upstream on
frame 0. Now applied on every 2.5 state (both stages, retake/extend included),
which shifts 2.5 outputs; 2.3 packs have no such parameter and stay
byte-identical. ``video_keyframes_mask`` kwarg on ``LTXModel.__call__``
(``None`` = exact no-op) is threaded from the state by all four sampler loops,
the TeaCache gate probe, ``Modality`` and the tiling wrapper.

**I2V frame-0 anchor was dropping the marker (fixed, DFR sub-project 4).**
``VideoConditionByLatentIndex.apply`` (the ``--image PATH 0 STRENGTH`` anchor)
rebuilt the latent state without carrying ``keyframes_mask`` /
``generated_keyframe_layout`` / ``generated_keyframes`` through, so the
first-frame keyframe marker above was silently dropped on every 2.5 I2V
render. It now carries the whole state through (``dataclasses.replace``, as
upstream's ``clone()`` + in-place write). 2.5 I2V outputs change (frame 0 now
gets the learned embedding, matching upstream); 2.3 packs are byte-identical.
``VideoConditionByReferenceLatent.apply`` and
``VideoConditionByKeyframeIndex.apply`` had the same hole for the
generated-keyframe-slot layout specifically — only visible on the DFR path,
where a reference conditioning follows the slots — and are fixed the same way.

Key files: ``conditioning/types/keyframe_slots.py`` (item + extraction),
``conditioning/mask_utils.py`` (``first_frame_keyframes_mask`` /
``extend_keyframes_mask``), ``model/transformer/model.py``
(``apply_keyframes_absolute_embedding``), ``utils/helpers.py``
(``evenly_spaced_keyframe_positions`` / ``generated_keyframe_conditionings``),
``BasePipeline._require_generated_keyframes_support`` (fails before any Gemma
load on packs without the embedding). Tests: ``tests/test_keyframe_slots.py``.

**Multishot prompting.** LTX-2.5's "native multishot" is a model capability, not
a pipeline feature: describe the shots in order in one prompt (see
[Lightricks' prompting guide](https://docs.ltx.video/open-source-model/usage-guides/prompting-guide)).
On this runtime, `--segment` (Prompt Relay) additionally gates local prompts to
time ranges when the model does not cut where you want.

### Two-Stage on LTX-2.5

`generate --two-stage --model <2.5-pack-dir>` runs the dev model + CFG
two-stage pipeline (half-res Stage 1 → upsample → distilled Stage 2 refine)
end-to-end on a local LTX-2.5 pack, same `is_ltx25_pack()` auto-detection as
the distilled path. T2V and I2V (`--image`) both work. `--two-stages-hq`
also runs on 2.5 packs (validated end to end; since #175 its stage 2 also runs res_2s).

```bash
ltx-2-mlx generate --model /path/to/ltx-2.5-mlx-q8 --two-stage --low-ram \
  -p "a heavy wooden door creaks slowly open" -H 512 -W 512 -f 49 \
  --frame-rate 24 -o out.mp4
```

### DFR base path (`generate --dfr`, 2.5 packs, experimental)

Port of upstream `DFRPipeline` (spatial_upscalings 1/2, temporal_upscalings 0/1/2 — complete). Runs on top of `DistilledPipeline`'s
`_stage1` / `_upsample_latent` / `_stage2` split.

**Canvas layout** (`dfr_layout.py`). The requested clip is padded to a whole number of keyframe
segments before stage 1 runs: candidate segment lengths are 24 and 32 pixel frames
(`SEGMENT_CANDIDATES`), and `choose_segment_length` picks whichever needs the least padding,
the larger candidate on a tie. One generated keyframe slot is placed at every segment boundary
(`[segment, 2*segment, ...]`) up to the padded length. The output is trimmed back to the requested
duration after stage 2 — the padding never reaches the saved file, only its keyframe slots do
(e.g. a 137-frame request pads to a 145-frame / 6-slot canvas and still writes 137 frames).

**Two stages.** Both stages, and the temporal rounds below, condition the transformer (RoPE
video positions, re-encoded I2V anchors, and the stage-2 slot/reference conditionings) at the
snapped `conditioning_fps(frame_rate)` (upstream `_conditioning_fps`: above 30 fps snaps to 60),
while audio token count/positions and the actual playback fps stay at the real `frame_rate`.
Matches upstream, which threads `fps=_conditioning_fps(frame_rate)` / `audio_fps=frame_rate`
through both `self.stage(...)` calls. At `frame_rate <= 30` (the default 24) `conditioning_fps`
is the identity, so nothing changes numerically; `--distilled` (outside DFR) is untouched.
- **Stage 1** (`_stage1`, half resolution, distilled/ancestral on 2.5): runs on the padded canvas
  with the keyframe slots injected via the same `--num-generated-keyframes` mechanism
  ([Generated keyframe slots](#generated-keyframe-slots---num-generated-keyframes-n-25-packs)),
  driven internally rather than by the CLI flag (`--num-generated-keyframes` is refused on
  `--dfr`). Optional I2V anchors (`--image`) apply as usual.
- **Stage 2** (`_stage2`, full resolution, ancestral on 2.5 like `--distilled`): the stage-1 video latent and its
  extracted keyframe-slot latents are each upsampled once (2× spatial, matching upstream's
  single-call-per-tensor shape), then denoised with two extra conditionings appended:
  `VideoGeneratedKeyframeSlots` (the upsampled slots, at the same canvas pixel-frame positions)
  and an IC-LoRA reference built from the pre-upsample stage-1 latent
  (`iclora_utils.reference_conditioning_from_latent`, strength 1.0, downscale factor read from
  the detailing LoRA's own metadata).

**Detailing LoRA attach.** The LoRA repo is gated on HuggingFace: `_resolve_detailing_lora` runs
before any model load and turns `GatedRepoError` into a `PermissionError` naming the licence page
(accept it once with the logged-in account). `_attach_detailing_lora` then resolves (downloads on first use) and attaches
`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` at a fixed strength of 0.5
(`DETAILING_LORA_STRENGTH`, not a user knob) to the resident distilled transformer right before
stage 2, mirroring `ICLoraPipeline._fuse_loras`: under `--low-ram` it appends a `BlockLoraSource`
(fused per block bind, or run-time adapters with `LTX2_LORA_MODE=unfused`); otherwise it fuses in place and re-quantizes, since stage 1 is finished
and the transformer is never reused clean afterward.

**Audio.** A stage-2 audio state is created and denoised jointly with the video (as upstream), but
its result is discarded; the shipped audio is stage 1's (unpatchified and trimmed to the
requested duration in audio tokens), matching upstream. Under `--low-ram`, stage 2's per-forward
time also includes the per-bind detailing LoRA fusion (dequantize -> fuse -> requantize on every
block bind of every step).

**Keyframe-aware decode.** `_decode_and_save_video` (overridden on `DFRPipeline`) builds a
`DecodeKeyframes` from the stage-2 slot latents (`BasePipeline.generated_keyframes`) and their
canvas pixel-frame positions (`decode_keyframes_from_slots`, `dfr.py`), dropping any slot whose
position falls outside the trimmed clip (the canvas padding). With `--video-decoder diffusion`
the slots are decoded as a second stream — joint neighborhood attention with the video stream, 2
nearest planes per video frame (and vice versa), plane noise seeded from the tile key's split —
where every joint window is upstream `joint_eager`'s **centered window clipped to the volume**
(`[i - k//2, i - k//2 + k) ∩ [0, L)`, fewer keys at a border), not natten's shift-inward window of
the plain path (`na3d`, untouched) —
and a tiled decode selects each tile's planes (inside the tile plus one neighbour on each side).
The conv decoder ignores the slots with a warning. Keyframe-aware renders are not
pixel-comparable to a plain render at the same seed (the extra stream changes every activation).

**Temporal rounds (`--temporal-upscalings {1,2}`).** Each round temporally x2-upsamples the
stage-2 video latent with the temporal latent upsampler (pack `temporal_upscaler_x2_v1_0.safetensors`,
resolved by `_resolve_temporal_upsampler_path` — a local `--temporal-upsampler-path` override skips
that lookup), then cuts the doubled timeline into `2**round` keyframe-seam tiles (`TemporalTilePlan`,
`dfr_layout.py`) at the carried keyframe positions (`seams = [2 * p for p in carry_positions]`) —
tiling is a hard split with **no overlap** (kept runs are disjoint), not a blend. A non-first tile
**starts on a keyframe plane with a pinned prefix** (upstream v1.4.0 `TilePrefix` / `tile_prefix` /
`lead_in_carryover`, ours in `dfr_layout.py` / `dfr.py::lead_in_latent`): cell 0 is the plane at the
last plane position before its seam (usually the previous tile's fresh mid-segment slot), cells
`1 .. (seam - plane) / 8` are the previous tile's finished output up to the seam, pinned by a
strength-1 `VideoConditionByLatentIndex` at index 0 (mask 0, so they *are* that output at every
step), and only the cells after the seam are kept. A tile is denoised as its own clip, whose cell 0
the model reads as one pixel frame: starting on a plane keeps content, shape and RoPE time in
agreement (the old mid-canvas lead-in ran 7 frames ahead). Each tile is re-denoised independently with the **distilled transformer, detailing
LoRA detached** (`_detach_detailing_lora`, run once before round 1: under `--low-ram` this drops the
`BlockLoraSource` from the streamer, otherwise it reloads a clean transformer) on the last 4 denoising
steps of the distilled sigma schedule (`TEMPORAL_SIGMAS = LTX_2_5_DISTILLED_SIGMAS[4:]`, 5 sigma
entries bracketing 4 steps) via ancestral Euler
(`EulerAncestralDiffusionStep(eta=TEMPORAL_ANCESTRAL_ETA=0.5)`, noise seed `seed + 1000*round + tile`).
Conditioning per tile: the carried keyframes after the tile's resume point (seam + 1) anchor it at
`ANCHOR_KEYFRAME_STRENGTH = 0.95` (soft, not a hard replace; anchors inside the pinned prefix are
dropped), plus fresh generated-keyframe slots at the *canvas* segment midpoints that fall in the tile,
and user images rebased on the tile window (prefix included). Audio is **frozen**, not re-denoised: stage 1's audio latent is windowed to the tile's
time range and resampled to the tile's new token count (`resample_audio_time`,
`audio_latent_for_tile`) as a `frozen` state (all-zero `denoise_mask`, sigma 0 for the audio prompt
AdaLN and the A→V gate — see "Frozen streams and per-modality sigma"), as upstream. The
transformer's conditioning fps is snapped by
`conditioning_fps()` — RoPE fps above 30 snaps to 60 (`_MAX_CONDITIONING_FPS = 60.0`); the actual
playback fps (`frame_rate * 2**temporal_upscalings`) is unchanged. After each round,
the carry bag (`generated_keyframes` / `generated_keyframe_positions`) is every plane on that round's
grid: the scaled anchors plus each tile's new slots, added as the tile finishes (existing planes win); the last round's bag
is what the keyframe-aware decode (`--video-decoder diffusion`) consumes instead of the stage-2 slots
— the conv decoder ignores it exactly as it ignores the stage-2 slots. Output frame count is
`(requested - 1) * 2**T + 1` at `frame_rate * 2**T` fps. Rounds refuse `--segment` (Prompt Relay) and
`--tile-frames` / `--tile-spatial` (modality tiling) up front, before any Gemma load.

**Temporal rounds validated** (M2 Pro 32 GB, LTX-2.5 q8, `--low-ram --no-audio`, seed 5, 512×768, `-f 121`,
pinned-prefix / no-overlap plan with global slots, #179): `--temporal-upscalings 1`: 241 frames at 48 fps in
1599 s; frames 0–79 are bit-identical to the pre-#179 render (the first tile is unchanged); the seam at
144→145 has a frame-to-frame change of 5.48 against the clip's 99th percentile of 6.09.
`--temporal-upscalings 2`: 481 frames at 96 fps in 4028 s; seam changes 3.31 / 3.08 / 3.42 against a 99th
percentile of 3.92. One isolated stall-then-jump at a latent border inside a round-2 tile (around frame 249
of the 96 fps render) comes from the model's temporal upsampler itself: it is already there when round 2
skips denoising (upsample only), and our temporal `LatentUpsampler` matches upstream torch to 1.6e-5 on the
pack weights — model-authentic, not a port bug; the round-2 tile denoise neither creates nor removes it.
Measured before #179/#180 (old overlapping lead-in tiles): `--dfr` without rounds byte-identical (sha256) to
`main`; T=1 round 1 = 2 tiles in 1073 s (~135 s per ancestral step on a 145-frame tile), max RSS 11.8 GB; T=2
round 2 = 4 tiles in 2393 s, max RSS 11.1 GB; round 1 with `--video-decoder diffusion`: the 10-plane carry
bag reaches the decoder (auto-tiled 3×1×3), 3382 s, 13.8 GB peak Metal. The first diffusion-decoder and
T=2 attempts were killed by the macOS GPU watchdog (display active) and passed with
`AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1`.

**Spatial epilogue (`--spatial-upscalings 2`).** With `spatial_upscalings=2` (the CLI's
`--spatial-upscalings {1,2}`, default 1) the dims are floored to multiples of 128 px (a stderr
warning when this changes the requested size), stage 1 runs at H/4 and stage 2 plus every temporal
round run at H/2 instead of the default H/2 / full res split — one extra spatial halving deferred
to a final epilogue. After stage 2 (and any temporal rounds) finish, `_run_spatial_epilogue`
details the H/2 latent up to full resolution (upstream v1.4.0 `run_spatial_epilogue`): the carry
keyframe bag is decoded one plane at a time with the render's own decoder (conv or diffusion, seeded
`seed + 4000 + i`), Lanczos-upsampled ×2 in RGB, and re-encoded as strength-1.0 keyframe
conditionings at full resolution. When no user image sits at frame 0 of the final grid, the H/2
latent's first frame is decoded the same way (next seed) as an **opening plane** that anchors frame
0 of the first window and is not shipped. The H/2 video latent is spatially upsampled once more and
re-denoised **window by window** (`plan_epilogue_windows`): one window per last-round temporal tile
(the whole canvas without rounds), run sequentially, each non-first window starting on the last carry
plane before its seam with its lead-in pinned to the previous window's finished output, exactly as in
the temporal rounds. Per window: the carry planes after its resume point, the opening plane (first
window), user images on the final grid, an IC-LoRA reference built from the pre-upsample H/2 latent
cropped to the window, and stage 1's audio windowed and frozen. Each window runs the stage-2 sigma
table with the distilled transformer + detailing LoRA (0.5) in two phases (`epilogue_sigma_phases`):
one ancestral step on a **2×2** spatial grid, then the same conditionings re-applied to that output
(no new noise) and the remaining steps on a **4×4** grid (a one-step table stays 2×2); spatial
tiles overlap 10 cells and blend after every step through
`X0Model(TiledLTXModel(..., normalize_positions=True))`, windows are never blended. Seeds: initial
noise `seed + 2000 + 100 * window`, ancestral `seed + 30000 + pass` (window i: coarse 2i, fine
2i + 1). Upstream's own T=0 path builds `TemporalTilePlan([])`, whose seam split refuses an empty
boundary list; we treat it as one window. The re-encoded keyframes (not the
pre-epilogue slots) become the decoder keyframes for the final keyframe-aware decode. `--dfr
--spatial-upscalings 2` refuses `--tile-frames` / `--tile-spatial` (modality tiling collides with
the epilogue's own tiling) and `--segment` (Prompt Relay), both up front in the CLI before any
Gemma load, mirroring the temporal-rounds refusals above.

**Spatial epilogue validated** (M2 Pro 32 GB, LTX-2.5 q8, `--dfr --low-ram --no-audio`, seed 5, `-f 49`;
per-window pinned passes, 2×2 coarse then 4×4, overlap 10, ancestral, #180): `--spatial-upscalings 2` at
1536×1024: 2272 s total, epilogue 1941 s, peak Metal 11.2 GB; with `--temporal-upscalings 1` (97 frames @
48 fps): 5015 s total, epilogue 4251 s, peak Metal 16.8 GB. No visible tile seam; the temporal seam's
frame-to-frame change is 1.20 against the clip's 99th percentile of 1.50. Measured before #179/#180 (single
2×2 window, overlap 12): `--spatial-upscalings 1` byte-identical (sha256) to main; max RSS 12.6 GB at
`--spatial-upscalings 2`; mean |Laplacian| 1.268 vs 1.202 for a direct `--spatial-upscalings 1` render at
1536×1024 (1190 s).

**Keyframe decode validated** (M2 Pro 32 GB, LTX-2.5 q8, `--low-ram --no-audio`, seed 5, 512×768×49,
baselines at the pre-keyframe base): `--dfr` (conv) and `--distilled --video-decoder diffusion` are
byte-identical (sha256) to the baselines; `--dfr --video-decoder diffusion` now decodes 2 planes
(`keyframes=2@[24, 48]`) in a 157.4 s decode phase vs 99.7 s plain (+58 %), peak Metal 10.73 vs
10.58 GB, PSNR 47.0 dB vs the plain decode (smooth per frame, no seam or flicker at the slot frames,
no visible sharpness change on this fog scene); forced 1×3×4 tiles: 470.7 s, 3.74 GB, 49.8 dB vs
untiled; `-f 137`: the slot at 144 is dropped, 5 planes, auto-tiled 2×1×2, 137 frames written. Torch
parity (upstream a95ab85): every plain and keyframe boundary ≤ 5.3e-6.

**Validated** (PR #150 e2e, M2 Pro 32 GB, LTX-2.5 q8, `--low-ram --no-audio`, seed 5):
`--distilled` at 512×768×49 is byte-identical (sha256) to `main` at 194 s, confirming the
`_stage1`/`_stage2` split is additive. `--dfr` at 512×768×49: 276.9 s total (stage 1: 8 steps at
10.7 s/forward over 864 video tokens; stage 2: 3 steps at 53.5 s/forward over 4128 tokens — target
+ 2 slots + the half-res reference), peak Metal 14.0 GB, max RSS 10.8 GB; frame 24 visibly sharper
(tree crowns, haze texture) than the plain distilled render at the same seed. `--dfr --image`
(I2V, frame-0 anchor) at the same shape: 283.8 s, first frame matches the source image. `--dfr -f
137`: canvas pads to 145 frames / 6 slots, 782.6 s, output trimmed back to 137 frames. `--dfr` at
768×1152×25 with `--video-decoder diffusion`: 464.5 s, decoded untiled, peak Metal 22.2 GB when first
measured — that was the decode promoted to fp32 by the fp32 stage-2 latent the sampler produces on
conditioned states (per-token sigma path); the diffusion decoder now casts its input to its weights'
dtype on entry (like the conv decoder) and the same decode peaks at ~12 GB (5.8 GB at 512×768×25).

**Key files:** `dfr_layout.py` (canvas), `dfr.py` (`DFRPipeline`), `distilled.py` (`_stage1` /
`_upsample_latent` / `_stage2`), `iclora_utils.py` (`reference_conditioning_from_latent`),
`tests/test_dfr*.py`.

### v1 limits (2.5 packs)

| Feature | Status |
|---|---|
| `--two-stage` (dev model + CFG) | supported (see above) |
| `--two-stages-hq` (res_2s + CFG) | supported — validated e2e on 2.5 (deterministic, audio at healthy 2.3-level loudness) |
| `DurationHead` / auto-duration (`-f` optional) | supported — `-f` defaults to `AutoDuration()` on `--one-stage`/`--distilled`/`--two-stage`/`--two-stages-hq`/`--dfr`; `--auto-duration MIN:MAX` overrides the clamp. Absent on 2.3 packs, where omitting `-f` now raises immediately (see "Auto-Duration" above) |
| `keyframe` | supported — validated e2e on 2.5 (deterministic, audio -38.3 dB; requires `--dev-transformer transformer-dev.safetensors`) |
| `a2v` | supported — validated e2e on 2.5 (deterministic, conditioned audio faithfully reconstructed at -36.2 dB). `a2v --distilled` (Experimental): validated e2e on 2.5 q8 at 512×768×49 (± `--low-ram`) and 704×1280×121/241 |
| `retake`, `extend` | supported — validated e2e on 2.5 (retake deterministic ×2; extend +N latent frames). `--low-ram` wired (mirrors upstream `offload_mode`): 49-frame retake that OOM'd now peaks at 13.8 GB. `retake --distilled` (Experimental): validated e2e on 2.5 q8 at 512×768×49 and 704×1280×121; `extend --distilled` (Experimental): validated e2e on 2.5 q8 at 704×1280, 10 s + 4 s (see Retake / Extend Example) |
| `ic-lora`, `lipdub` | not yet supported (no official 2.5 task IC-LoRAs published yet) |
| `hdr-ic-lora` | supported, **2.5 only** (upstream v1.4 SDR-to-HDR IC-LoRA, ACEScct); Experimental, validated end to end on real weights (see "HDR IC-LoRA Pipeline" › Status) |
| `enhance` / `--enhance-prompt` | raises `NotImplementedError` (`_guard_enhance_not_gemma4`) — Gemma 3 only |
| `--enable-teacache` | raises `ValueError` — 2.3 polynomial isn't calibrated for 2.5 |
| Prompt Relay | supported — validated e2e on 2.5 `--distilled` and the `--dfr` base path (see "Prompt Relay"); token ranges pinned against the pack's Gemma-4 tokenizer |
| Modality tiling | validated on 2.5 `--distilled` (see "Position layout divergence" under Modality Tiling) |
| NAG (`--nag`, negative prompt on `--distilled` / `--dfr`) | experimental — validated e2e on 2.5 `--distilled` and `--dfr` (see "Negative prompt without CFG: NAG") |
| Generated keyframe slots (`--num-generated-keyframes N`) | supported on `generate` (the four non-DFR modes, stage 1 only; `--dfr` places its own); refused up front on 2.3 packs (no `use_keyframes_abs_pos_embedding`) |
| DFR (`DFRPipeline`) | complete — shipped as `generate --dfr`: base path (spatial detailing with the official 2.5 detailing IC-LoRA), keyframe-aware decode on `--video-decoder diffusion`, temporal rounds (`--temporal-upscalings {1,2}`), and the spatial epilogue (`--spatial-upscalings {1,2}`) |
| Diffusion video decoder | opt-in `--video-decoder diffusion` (experimental; tiled automatically above the decode budget, `--diffvae-tile` override); conv remains default |

The IC-LoRA family (`ic-lora` / `lipdub`) lands once
Lightricks publishes the official 2.5 task IC-LoRAs.

### Conv VAE decode budget and auto tiling

`VideoDecoder.decode_and_stream` estimates the peak of a decode as **~750 bytes per output
pixel-frame** (`CONV_DECODE_BYTES_PER_PIXEL_FRAME`, measured on an M2 Pro 32 GB with the 2.5 q8
pack: 13.3 GB for 512×768×49, 10.2 GB for 768×1152×17; issue #142 measured 33.9 GB for
768×512×121 on an M3 Max) plus the fp32 accumulation buffers of the tiled path. The peak sits in
the last up-blocks at full pixel resolution — the previous block-3 estimate was ~80× too low and
the conv decoder effectively never tiled. When the estimate exceeds `LTX2_VAE_DECODE_BUDGET_GB`
(default half of unified memory) `_compute_decode_tiling` walks a ladder: temporal tiles from
upstream's default 80 frames / 24 overlap down to 40 frames, then spatial tiles 768/64 → 512/32 →
256/32 px (upstream `TileSizeConfig.default()` is 768/64) at 40 frames, then 16–32-frame tiles at
256 px as a last resort (a warning if even that exceeds the budget). MLX's allocator cache is
disabled for the duration of the decode (`decode_cache_limit`; opt out with
`LTX2_VAE_DECODE_KEEP_CACHE=1`): pixels are identical, only the free-list retention changes.
Spatial-only `TilingConfig`s decode correctly (they crashed before). Verbose runs print
`[vae-decode tiling] frames=… px=… est. X GB budget=Y GB` and the measured peak Metal memory.

Validated (M2 Pro 32 GB, 2.5 q8, `--distilled --low-ram --no-audio`, seed 5): 512×768×49 stays
untiled at the default 16 GB budget and is byte-identical (sha256) to the v0.15.7 render, 7.0 s
decode, 14.0 GB peak Metal memory (13.5 estimated); 768×1152×25 untiled 15.7 GB peak (15.4
estimated) vs forced tiled at an 8 GB budget (768/64 spatial, 2×2 tiles) 8.9 GB peak, 10.7 s vs
8.0 s, PSNR 45.0 dB, seam lines at most 1.8/255 mean abs diff (nothing visible); 768×1152×49 at
16 GB picks 40-frame × 768-px tiles: 8.8 GB peak (12.6 estimated), 23 s on a random latent.
Disabling the allocator cache left the pixels identical and the decode 20 % faster (9.2 → 7.2 s).

### Diffusion video decoder (`--video-decoder diffusion`, 2.5 packs, experimental)

Port of upstream `NADiffusionDecoder` (`vae_decoder_av.safetensors`, already in every 2.5 pack):
a Linear-only transformer decoder — four deterministic neighborhood-attention stages on the latent
grid with pixel-shuffle upsamples, then eight AdaLN-modulated diffusion blocks at pixel/4
resolution (kernel 11×11×11), one evaluation at `t = 1` on pure noise, `x0` output = pixels.
Reproduces upstream's **default** `chunked_eager` mode exactly: stage-5 attention runs on four
width slabs with a 5-cell halo, edge-replicated at the true image borders (the first/last 20 px
of each row differ from full-volume attention — same as upstream). Neighborhood attention is exact
blocked dense attention with a boolean window mask (`diffusion_decoder/neighborhood_attention.py`).
Tiled decode (upstream `diffusion_tiling.py`, ported in `diffusion_decoder/tiling.py`): stages 1–3
run once on the whole latent; stages 4–5 run per tile on the stage-4 grid (one cell = 2 frames ×
8 × 8 px) with recommended overlaps of 40 frames / 160 px, per-tile fresh noise (key
`seed + 30000 + tile_index`), trapezoid blending in an fp16 accumulator and temporal-group
streaming to ffmpeg. Tiling is automatic: the decode is untiled when its estimated activations
(`stage-5 tokens × 256 × 2 × 17.5` (calibrated on MLX, see the e2e numbers) + fp16 output) fit
`LTX2_VAE_DECODE_BUDGET_GB` (default: half of unified memory, shared with the conv decoder), otherwise the least-redundant tile on the (8, 32, 32) px grid ≥ 80 frames / 320 px
that fits is chosen. `--diffvae-tile FRAMES HEIGHT WIDTH` overrides it (0 = axis untiled);
`--diffvae-tile 0 0 0` forces one tile, the only case where the `LTX2_DIFFVAE_MAX_TOKENS` guard
(default 1,204,224) still applies. Overlaps make tiled decodes cost several times the untiled
token count (the `[diffvae tiling]` stderr line prints the redundancy factor); a bigger budget
means fewer, larger tiles. Tiled and untiled renders of the same seed differ in fine texture
(different noise), as upstream. The decoder runs in its weights' dtype (bf16 packs) whatever dtype the
caller's latent has and restores that dtype on the output, mirroring the conv decoder: an fp32 latent
(every conditioned render's sampler state ends fp32) would otherwise promote every activation and
double the peak. Decoder noise seed = `seed + 30000`; not
bit-comparable with torch's generator. Parity: per-stage torch goldens
(`tests/parity_diffvae_reference.py`, disposable env) at 1e-4 (det stages) / 1e-3 (diffusion).
Conv stays the default. `decode`/`tiled_decode` accept an optional `keyframes: DecodeKeyframes`
(DFR's stage-2 slots, threaded from `BasePipeline._decode_and_save_video`): the keyframe planes
are denormalised, offset by the decoder's own `type_emb` (distinguishes plane tokens from video
tokens in the shared weights) and projected through the same `conv_in`, then carried as a second
stream through stages 1–5 via `forward_*_with_keyframes`, mixing with the video stream only
inside `joint_na3d`'s softmax (`diffusion_decoder/keyframes.py`). `keyframes=None` is the exact
no-op plain path above. Key files: `model/video_vae/diffusion_decoder/`, `utils/blocks.py::_DiffusionVideoDecoder`.

E2E validated on the 2.5 q8 pack (M2 Pro 32 GB, `--low-ram`, distilled two-stage, seed 5).
At 384×576×25: conv 93.4s total (2.6s decode phase, ~11.8 GB peak RSS) vs diffusion 129.4s total
(39.1s decode phase, ~10.4 GB peak RSS); PSNR conv-vs-diffusion 39.25 dB, diffusion frame
visibly sharper on fine edges (flower petals) at matched latents/seed. Larger diffusion decodes
also complete: 512×768×25 (614,400 stage-5 tokens) in 175.3s total / 50.8s decode phase /
~11.3 GB peak RSS, and 512×768×49 (1,204,224 tokens) in 289.8s total / 102.5s decode phase /
~10.5 GB peak RSS. `na3d` materializes its accumulator once per block group, which bounds live
Metal buffers and is what lifted the earlier `[metal::malloc] Resource limit (499000) exceeded`
ceiling. 512×768×49 is the largest shape measured and is now the `LTX2_DIFFVAE_MAX_TOKENS`
default.

Tiling validated end to end on the same pack (M2 Pro 32 GB, `--distilled --low-ram --no-audio`,
seed 5). At 512×768×49 the default 16 GB budget decodes untiled, byte-identical (sha256) to the
pre-tiling decoder: 101.4s decode phase, 10.58 GB peak Metal memory. The same target forced to a
80/320/320 tile size splits into 1×3×4 = 12 tiles (redundancy ×5.1): 287.1s, 3.65 GB peak, PSNR
45.84 dB against the untiled decode with no seam visible (the seam-profile spikes don't line up
with tile boundaries). 512×768×97 untiled at a 12 GB budget — the calibration run, previously
refused by the token guard — takes 193.6s at 20.07 GB peak, which is what drove the stage-5 memory
coefficient recalibration above. 768×1152×49 at an 8 GB budget splits into 1×4×6 = 24 tiles
(redundancy ×5.0): 627.6s, 4.28 GB peak Metal memory, 11.25 GB max RSS. As with the untiled
decoder, tiled and untiled decodes of the same seed differ in fine texture (different per-tile
noise), matching upstream.

### Key Files

- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/distilled.py` — `DistilledPipeline` (2.3/2.5 dispatch), `ANCESTRAL_*` constants.
- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/generation.py` — `is_ltx25_pack()`.
- `packages/ltx-core-mlx/src/ltx_core_mlx/text_encoders/gemma/encoders/encoder_configurator.py` — `select_text_encoder()`, `check_gemma_version()`.
- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py` — `_resolve_upsampler_path()` (`_is_25` resolved in `__init__` via `is_ltx25_pack`; the `_is_25: bool = False` class attribute lives on `BasePipeline`).
- `tests/test_ltx25_distilled.py` — routing, sigma-table, TeaCache-guard, upsampler-resolution tests.

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

### AdaLN dedupe switches (per-token timesteps only)

The per-token AdaLN path deduplicates identical sigma rows into one shrunken
GEMM (bitwise-verified per module by a one-time calibration; the calibrating
call returns the exact reference) and defers the per-token gather into the
blocks. Both are on by default and bit-identical by construction:

- `LTX2_ADALN_DEDUPE=0` — disable the dedupe entirely.
- `LTX2_ADALN_LAZY=0` — keep the dedupe but materialise the gather eagerly
  (the `--low-ram` streamer does this automatically: the lazy carrier cannot
  cross the compiled block, and the compiled block is what keeps the 48
  per-block syncs under the Metal watchdog).
- `LTX2_ADALN_DEDUPE_MIN_ROWS` (default 1024), `LTX2_ADALN_DEDUPE_MAX_FRAC`
  (default 0.5) — thresholds below/above which the dedupe is not attempted.
- `LTX2_ADALN_DEDUPE_DEBUG=1` — log calibration verdicts.

### DiT compute dtype (`LTX2_COMPUTE_DTYPE`)

The DiT runs in float32 by default, although its weights are bf16: the `scale_shift_table`s are
stored F32 and the per-token AdaLN parameters are F32 (the sinusoidal timestep embedding is
float32), so `rms(x) * (1 + scale) + shift` promotes the modulated activations, the residual stream
becomes float32 after block 0, and every projection and attention call follows. Upstream PyTorch
casts the tables to the timestep dtype and runs the whole block in bf16.

`LTX2_COMPUTE_DTYPE=float16` (Python: `LTXModel.set_compute_dtype(mx.float16)`; pipelines apply the
variable when they load a DiT, including `--low-ram`, where `BlockStreamer.bind` casts each block as
it is bound) runs the inside of every attention and feed-forward module in float16: projections,
q/k norms, RoPE and the attention kernel. The residual stream and the AdaLN modulation stay float32,
and the float32 gate multiply promotes each module output back before the residual add. The modules'
float parameters (quantization scales/biases, Linear biases, q/k norm weights) are cast at load:
float16 activations against bf16 scales would promote to float32 inside `quantized_matmul` and gain
nothing. The AdaLN tables keep their F32 dtype. `bfloat16` gives the upstream precision; unset or
`float32` is today's path, untouched. In-place LoRA fusion (`dfr`'s detailing LoRA, `ic-lora`, and the
distilled-LoRA fusion that starts stage 2 of `--two-stage`, `--two-stages-hq`, `a2v` and `keyframe`
in `TI2VidTwoStagesPipeline._fuse_distilled_lora`) re-quantizes from a float32 weight; `fuse_loras`
gives the new scales/biases the dtype the old ones had (until that fix they came out float32 on every
path). Every such call site also calls `BasePipeline._recast_after_inplace_fusion()` afterwards, so a
compute dtype set before the fusion is re-applied to whatever the fusion replaced. (`--low-ram` swaps
the streamer to the pre-fused distilled transformer instead and keeps casting at bind time.)

Why float16: on an M1 GPU (no native bf16), int8 `quantized_matmul` at the stage-2 shapes
(17,856 tokens × 4096) runs at 6.6 TFLOPS with float32 activations, 5.8 with bf16 and 8.1 with
float16, and `mx.fast.scaled_dot_product_attention` (32 × 17,856 × 128) at 4.5 / 6.0 / 7.6 TFLOPS.
Measured on an M1 Max 64 GB, 2.5 q8 pack, `--distilled` I2V 576×1024×241, seed 7:

| | stage-2 step | whole render | output vs float32 (one stage-2 forward) |
|---|---:|---:|---|
| float32 (default) | 141 s | 701 s | — |
| `bfloat16` | 136 s | 698 s | rel. L2 6.4e-2, cos 0.99796 |
| `float16` | 107 s | 563 s | rel. L2 1.7e-2, cos 0.99985 |

Float16 margin: on every forward of two full renders (I2V 576×1024×121 and a flat-sky T2V
1024×576×97, all 11 forwards each) the largest value inside any attention/FF op was 2,512, against
float16's 65,504; the residual stream reaches ~14,000, which is why it stays float32. As a guard,
`LTXModel.__call__` checks the output when a compute dtype is set and, if it is not finite, prints
a warning, drops the setting for the rest of the run and recomputes that forward. On a resident model
the recompute is **not** the float32 path: the parameters already cast stay float16-rounded, only the
activations are no longer cast. Under `--low-ram` the `StreamingLTXModel` wrapper owns the setting (the
inner model holds a weak reference to it), so the guard drops it there: later binds stop casting, the
compiled block is retraced, and the recompute rebinds the stored weights, i.e. it is the default path.
0.48 % of the q8 pack's quantization scales are below float16's normal range (they belong to
near-zero weight groups) and lose precision in the cast.

Tests: `tests/test_compute_dtype.py` (tiny int8 model with F32 tables and per-token timesteps, the
production dtype layout): default untouched, which parameters are cast, module output dtypes,
closeness to float32 with float16 closer than bf16, the overflow guard (resident and streamed), streaming
parity, the recast after in-place LoRA fusion (including the two-stage distilled-LoRA call site), env
parsing.

### LoRA mode (`LTX2_LORA_MODE`)

On a quantized pack, fusing a LoRA in place (`apply_loras`) dequantizes each targeted weight, adds
`strength * B @ A` and re-quantizes to the pack's int8 / group 64 (or int4). The re-quantization
error is of the same order as a small LoRA's update. Measured on the LTX-2.5 Ingredients IC-LoRA: the
update is 0.25-3.6 % of the weight's norm and the error 16-216 % of the update itself; on the detailing
LoRA at strength 0.5, three layers: 82 %, 36 % and 194 %. Most of what such a LoRA should change is lost.

`LTX2_LORA_MODE=unfused` (Python: `ltx_core_mlx.loader.attach_loras`) leaves the weights alone and
turns every targeted linear into an adapter that computes `base(x) + (x @ A^T) @ B^T` (strength
folded into `B` in float32; the ranks of several LoRAs on one layer are stacked). The adapter is the
original layer with `lora_a` / `lora_b` added (its class subclasses the layer's class and shares its
arrays), so the weight names do not change: an in-place fusion that runs later still reaches the
weight under an adapter, and `LTXModel.set_compute_dtype` casts the factors with the other float
parameters of their attention / feed-forward module. The factors are created in the DiT's compute
dtype when one is set (no `_recast_after_inplace_fusion` is needed: no weight was re-quantized), else
kept in the LoRA file's dtype, which the float32 activations promote. `AttachedLoras.detach()` puts
the original module objects back in place, with the same arrays. The handle holds no model memory (weak
references; the replaced layers it keeps are emptied while the adapter holds their parameters), so a
pipeline that frees its DiT with LoRAs attached really frees it. Unset or `fused` is today's path,
untouched. The value is parsed once, when the pipeline is built (`BasePipeline.lora_mode`), so a typo
fails before any work rather than when the first LoRA is attached.

| call site | unfused | notes |
|---|---|---|
| `ic-lora` / `hdr-ic-lora` / `lipdub` task LoRAs (`ICLoraPipeline._fuse_loras`) | adapters | Stage 2 of the legacy distilled path detaches them in place instead of reloading the DiT; a second call replaces them |
| `--dfr` detailing LoRA (`DFRPipeline._attach_detailing_lora`) | adapters | the temporal rounds detach them in place (no DiT reload); the spatial epilogue re-attaches |
| `generate --lora` on a resident DiT (`_pending_loras`) | adapters | the DiT is loaded as usual, then the LoRAs are attached |
| distilled LoRA (stage 2 of `--two-stage`, `--two-stages-hq`, `a2v`, `keyframe`; `ic-lora` dev mode) | **fused** | rank 384 / 450 on every block linear: as adapters it would stay resident and add work to every forward |
| `--low-ram` (task LoRAs: `--lora`, IC-LoRAs, detailing LoRA) | adapters on the streamed block | see "Unfused LoRAs under `--low-ram`" below; the distilled LoRA still fuses at bind |

Cost: per adapted layer, `rank * (in + out)` multiply-adds per token against `in * out` for the layer
(rank 128 on a 4096 x 4096 projection: +6 %; on the 4096 x 16384 feed-forward: +4 %), and the factors
stay resident (the LoRA file's size, e.g. 1.3 GB for a rank-128 IC-LoRA in bf16).

**Unfused LoRAs under `--low-ram`** (#192). The streamed pipelines add each task LoRA as a
`BlockLoraSource(fuse=False)` (`streamed_lora_fuse(lora_mode)`); `fuse=True` sources (the default, and
always the distilled LoRA) keep the bind-time fusion. Before each forward,
`StreamingLTXModel._sync_lora_adapters` lays out adapters on the shared block for the current
`fuse=False` sources, through `attach_loras` with zero factors: per targeted layer, the sources' ranks
stacked in order, each at its largest rank over all blocks. It acts only when that set changes (a
pipeline adds or drops a source), detaching the previous layout and retracing the compiled block.
`BlockStreamer.bind` then loads each block's factors next to its weights
(`_lora_factors_for_block`): `B` scaled by the strength in float32, zeros where a block has no factors
for a layer or a smaller rank, so the parameter tree and the compiled graph stay fixed. The quantized
weights are bound as stored (no dequantize / re-quantize per bind), and `cast_dtype` casts the
factors with the module's other float parameters. Tests: `tests/test_unfused_lora_streaming.py`
(streamed vs resident `attach_loras` on the tiny model in bf16 and q8, rank padding and a layer
missing from a block, two stacked sources, fused + unfused sources together, clearing the sources
restores the base output and plain layers, compute dtype).

Tests: `tests/test_unfused_lora.py` (adapter vs dense `W + B @ A`; on a q8 layer and on the tiny q8
DiT, fuse + re-quantize vs unfused against a float32 reference; stacking and reverse-order detach;
detach restores the original objects and the exact output; a freed model is not kept alive; weight
names unchanged; compute dtype followed; an in-place fusion still reaches an adapted layer; the
ic-lora / dfr / `--lora` call sites; the default path identical to the fusion; env parsing).

### Block-sparse stage-2 attention (`LTX2_SOL_TAU`)

Opt-in port of NVIDIA's Sol-Attn routing (arXiv 2607.24027; reference code in NVlabs/Sana, `sol-engine` branch,
Apache-2.0) to the video self-attention of the distilled stage 2, the scope NVIDIA's own LTX-2.5 distilled config uses
(`models/ltx25/RTX5090/attention.py`): `attn1` only, block 0 dense, one threshold per stage-2 step.

```bash
LTX2_SOL_TAU=1.0,1.25,1.5 ltx-2-mlx generate --distilled ...   # NVIDIA's values; unset or "off" = dense (default)
```

How it works (`model/transformer/sparse_attention.py`):
- Every head's sequence is cut into 64-token blocks with a query and a key centroid each. A query block attends exactly
  to the key blocks whose centroid score passes `mean + tau * std` of its centroid scores (Sol's "diag" threshold), to
  its neighbours (`|i - j| <= 1`) and, one rule beyond the reference, to the blocks holding the same (h, w) positions
  one latent frame before and after (+0.2 % density). Every other key block enters the same softmax once, as its key
  centroid with a `log(length)` bias and its value mean: `exp(s + ln L) * mean(V) = exp(s) * sum(V)`, Sol's
  length-weighted correction.
- The routing mask is built in MLX; the attention is MLX's own steel flash-attention loop (`steel_attention.h`, BQ=32,
  BK=16, 4 simdgroups) over the routed blocks only, then the centroid tiles. The steel headers come from the installed
  `mlx` and are compiled through `mx.fast.metal_kernel`. `tau = -inf` routes every block and is bit-identical to
  `mx.fast.scaled_dot_product_attention` at head_dim 128 in float16, bfloat16 and float32. `kernel_available(dtype)`
  checks the kernel once per dtype, the first time a call in that dtype would run sparse, on the variant a real call
  builds (9 blocks, so every array input is in device memory: `mx.fast.metal_kernel` passes inputs with fewer than 8
  elements in `constant` memory, which is another kernel). If it fails to build or does not match dense attention,
  a warning is printed once for that dtype and its calls stay dense.
- `LTXModel.set_sparse_attention(state)` attaches one `SparseAttentionState` to `attn1` of blocks 1 and up. Each forward
  calls `state.prepare(timestep, video_positions)`: the tau is picked by matching the forward's sigma against the
  stage-2 table (so tiles, guidance passes and the float16 overflow guard's recompute all get their step's tau; a sigma
  outside the table runs dense), and the latent frame size is the leading run of tokens sharing the first token's time
  (appended conditioning tokens, e.g. DFR's reference, do not change it). Calls with an attention mask, an STG
  perturbation, cross-attention, or fewer than 4096 tokens stay dense.
- `DistilledPipeline._stage2` turns it on around the stage-2 loop and off afterwards (also on error), so it covers
  the stage 2 of every pipeline built on it: `generate --distilled`, `generate --dfr`, `a2v --distilled` and
  `extend --distilled`. Stage 1, the DFR temporal rounds and spatial epilogue, and
  the other pipelines (single-stage `retake --distilled` included) are untouched. Under `--low-ram` the streamed model attaches the state to whichever block is bound
  and runs the eager shared block while it is on (the compiled block would replay the tau and routing of the step it
  was traced on).

Measured on an M1 Max 64 GB (MLX 0.32.2, 2.5 q8 pack), `LTX2_SOL_TAU=1.0,1.25,1.5` against unset, same command and
seed, one run per arm (stage 1 is the same work in both arms and took the same time):

| run | stage-2 tokens | stage-2 step, dense → sparse | whole render | peak footprint |
|---|---:|---:|---:|---:|
| `--distilled` I2V 576×1024×241, `LTX2_COMPUTE_DTYPE=float16` | 17,856 | 103 → 81 s (−21 %) | 534 → 470 s (−12 %) | 39.8 → 40.3 GB |
| same, default compute (float32) | 17,856 | 139 → 105 s (−24 %) | 684 → 583 s (−15 %) | 40.8 → 42.0 GB |
| `--distilled` I2V 1088×1920×121, float16 | 32,640 | 238 → 153 s (−36 %) | 1148 → 890 s (−22 %) | 53.2 → 53.4 GB |
| `--dfr` I2V 576×1024×121, float16 | 14,400 | 76 → 64 s (−16 %) | 374 → 339 s (−9 %) | 40.7 → 40.4 GB |
| `--distilled --low-ram` I2V 576×1024×121, float16 | 9,216 | 46 → 41 s (−11 %) | 262 → 246 s (−6 %) | 25.1 → 25.1 GB |
| `--distilled` T2V 1024×576×121, float16 | 9,216 | 43 → 36 s (−16 %) | 241 → 222 s (−8 %) | 25.6 → 25.9 GB |

The gain grows with the token count because only the attention gets cheaper (feed-forward, cross-attention and the
projections are untouched). One attention call on real stage-2 q/k/v (28,160 tokens,
float16): dense 1.72 s, every block routed 1.78 s (bit-identical), `tau = 1.0 / 1.25 / 1.5` ×4.1–4.9 / ×5.0–6.1 /
×5.9–7.7, at 16–20 % routed density for `tau = 1.0`.

Quality, sparse against the dense render of the same seed (evals suite: ArcFace to the start image, Whisper WER,
SyncNet LSE-C, background warping error after optical flow):

| run | PSNR / SSIM | ArcFace median, dense / sparse | WER, dense / sparse | LSE-C, dense / sparse | background warp ×1000, dense / sparse |
|---|---:|---:|---:|---:|---:|
| I2V 576×1024×241, float16 | 30.3 dB / 0.924 | 0.62 / 0.56 | 0 / 0 | 3.22 / 3.25 | 0.093 / 0.103 |
| same, float32 | 30.8 dB / 0.927 | 0.64 / 0.59 | 0 / 0 | 3.25 / 3.42 | 0.096 / 0.098 |
| I2V 1088×1920×121 | 31.7 dB / 0.954 | 0.55 / 0.53 | 0 / 0 | 5.60 / 5.87 | 0.243 / 0.262 |
| `--dfr` 576×1024×121 | 32.3 dB / 0.938 | 0.73 / 0.64 | 0.06 / 0.06 | 0.80 / 0.78 | 0.158 / 0.180 |
| `--low-ram` 576×1024×121 | 30.8 dB / 0.925 | 0.67 / 0.66 | 0 / 0.06 | 0.53 / 0.65 | 0.138 / 0.137 |
| T2V 1024×576×121, flat wall and sky | 42.5 dB / 0.993 | — | — | — | 0.023 / 0.024 |

Frames look the same; the differences are in fine, moving detail (strands of hair, the timing of a breaking wave), with
no block artifacts on flat areas. Sparse attention costs some identity against the start image (ArcFace 0.01–0.09
lower): later frames see the conditioning frame only through its centroids. Keeping the first latent frame exact for
every query (a "sink") recovered a third to all of that, but made a stage-2 step up to 8.5 % slower (166 s instead of
153 s at 32,640 tokens) and raised the background warping error in three of five I2V runs, with no visible difference
in the clips, so it is not included.

Tests: `tests/test_sparse_attention.py` (kernel vs dense at `tau = -inf` and vs a plain-MLX Sol reference at finite
tau, routing rules, sigma-to-tau selection, the frame size, the module guards, the model and streamed wiring, the
pipeline switch, env parsing, the unavailable-kernel fallback).

### `LTX2_GEMMA_MAX_LENGTH`

Caps the padded Gemma sequence length (default `1024`). Reducing to `512` halves Gemma forward time but **shifts left-padded RoPE positions away from the LTX training distribution** — quality risk. Use only as a last resort on heavily contended systems.

```bash
LTX2_GEMMA_MAX_LENGTH=512 ltx-2-mlx generate --two-stage ...
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

### Release artifacts (automated via release-please)

Releases are driven by **release-please** (`.github/workflows/release-please.yml`),
mirroring the setup in mlx-forge and smeltr. You no longer hand-bump versions or
hand-cut tags — you just merge conventional-commit feature PRs into `main`.

The flow:

1. **Merge feature PRs** into `main` with conventional-commit subjects
   (`feat:` → `z` bump, `fix:` → `z` bump, `feat!:`/`fix!:` → `y` bump). The
   bump rules in the table above still hold; release-please derives them from
   the commit type. `commitlint` enforces the prefixes.
2. **release-please opens / updates a `chore(main): release X.Y.Z` PR**
   automatically on every push to `main`. It aggregates the unreleased commits
   into the `CHANGELOG.md` entry and bumps the version in all four pyprojects:
   the workspace root via the `python` release-type, and the three
   sub-packages (`ltx-core-mlx`, `ltx-pipelines-mlx`, `ltx-trainer`) via the
   `extra-files` TOML `$.project.version` entries in `release-please-config.json`.
   All four stay in sync by construction.
3. **`relock-on-release.yml` resyncs `uv.lock`** on that release PR
   (gated to the Bot-authored `release-please--` branch), so the lockfile
   always matches the released version — this is what the 0.14.12 release
   missed by hand.
4. **Merging the release PR** makes release-please push the annotated
   `vX.Y.Z` tag on the merge commit **and** create the GitHub Release with the
   CHANGELOG section as notes. No manual `git tag` / `gh release` step.

The release PR is authored by the release GitHub App (`RELEASE_APP_ID` /
`RELEASE_APP_PRIVATE_KEY` secrets), not the default `GITHUB_TOKEN`, so its PR
triggers CI and the relock workflow (a default-token push cannot trigger other
workflows). `.release-please-manifest.json` tracks the last released version;
do not edit it by hand.

**Manual fallback** (only if release-please is broken): bump the four pyprojects
+ `uv.lock` + manifest, add the CHANGELOG entry, open a `chore(release):` PR,
then tag with `git tag -a vX.Y.Z` and `git push origin vX.Y.Z`
(`.github/workflows/release.yml` no longer exists — the tag-listening release
job was superseded by release-please). `scripts/validate_versions.py` checks
four-pyproject coherence for this path.

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
- **Pre-converted weights**: HuggingFace collections for [LTX-2.3](https://huggingface.co/collections/dgrauet/ltx-23) and [LTX-2.5](https://huggingface.co/collections/dgrauet/ltx-25-6a90c410ff65a75f8aeae402)
