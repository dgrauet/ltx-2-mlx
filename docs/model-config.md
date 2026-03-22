# Model Configuration

Correct configuration values for LTX-2.3, including cases where `config.json` is wrong.

---

## Config Corrections

| Parameter | config.json Value | Actual Value | Evidence |
|-----------|-------------------|--------------|----------|
| `cross_attention_adaln` | `false` | **`true`** | Weights have 9 AdaLN params + `prompt_scale_shift_table` per block |

This is the only known incorrect value in config.json. All other values are correct.

---

## Transformer Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_layers` | 48 | Transformer blocks |
| `dim` | 4096 | Video hidden dimension |
| `audio_dim` | 2048 | Audio hidden dimension |
| `num_heads` | 32 | Video attention heads |
| `audio_heads` | 32 | Audio attention heads |
| `head_dim` (video) | 128 | 4096 / 32 |
| `head_dim` (audio) | 64 | 2048 / 32 |
| `ff_inner_dim` | 16384 | 4096 * 4 |
| `audio_ff_inner_dim` | 8192 | 2048 * 4 |
| `cross_attention_adaln` | **true** | Config says false, weights say true |
| `apply_gated_attention` | true | Per-head gating: `2 * sigmoid(logits)` |
| `norm_type` | rms_norm | Parameterless: `x / sqrt(mean(x^2) + eps)` |
| `timestep_scale_multiplier` | 1000 | Scales input timestep before embedding |

### Positional Embedding

| Parameter | Value | Notes |
|-----------|-------|-------|
| `positional_embedding_theta` | 10000.0 | RoPE base frequency |
| `positional_embedding_max_pos` (video) | [20, 2048, 2048] | [temporal, height, width] |
| `positional_embedding_max_pos` (audio) | [20] | [temporal] |
| `use_middle_indices_grid` | true | Positions are midpoints (i + 0.5) |
| `positional_embedding_type` | INTERLEAVED | Main transformer RoPE mode |

### AdaLN Parameters

- `AdaLayerNormSingle` produces 9 parameters per block (not 6).
- Split: `[0-2]` self-attn, `[3-5]` feed-forward, `[6-8]` cross-attn.
- Separate `prompt_scale_shift_table(2, dim)` for text cross-attention KV modulation.
- Separate `scale_shift_table_a2v_ca_video(5, dim)` / `_audio(5, dim)` for AV cross-attention.

---

## VAE Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| `norm_layer` | pixel_norm | Parameterless RMS norm |
| `patch_size` | 4 | Final pixel shuffle spatial factor |
| `causal_decoder` | false | |
| `timestep_conditioning` | false | VAE is not timestep-conditioned |
| `decoder_base_channels` | 128 | Internal = 128 * 8 = 1024 |
| Temporal compression | 8x | |
| Spatial compression | 32x | |
| Latent channels | 128 | |

### Frame Count Formula

- 6 latent frames -> 41 pixel frames.
- General: `pixel_frames = (latent_frames * temporal_factor) - (temporal_factor - 1)`.
- Each temporal DepthToSpace doubles frames, then removes first frame (`x[:, 1:]`).

---

## Text Encoder / Connector Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Gemma 3 12B | Loaded via mlx-lm |
| Max tokens | 1024 | |
| Video projection dim | 4096 | Matches transformer dim |
| Audio projection dim | 2048 | Matches audio_dim |
| `connector_rope_type` | SPLIT | Different from main transformer (INTERLEAVED) |
| `connector_num_learnable_registers` | 128 | Appended to connector sequence |

---

## Audio Config

### Vocoder (BigVGAN v2, 16kHz)

| Parameter | Value |
|-----------|-------|
| Upsample ratios | [5, 2, 2, 2, 2, 2] |
| Upsample kernel sizes | [11, 4, 4, 4, 4, 4] |
| Total upsample factor | 160x (hop_length=160 at 16kHz) |
| Activation | SnakeBeta + anti-aliased resampling |
| ResBlocks per stage | 3 x AMPBlock1 |

### BWE (16kHz -> 48kHz)

| Parameter | Value |
|-----------|-------|
| Resampler | 3x upsample (kaiser-sinc) |
| BWE generator ratios | [6, 5, 2, 2, 2] |
| BWE total factor | 240x |
| MelSTFT mel_basis | (64, 257) |
| MelSTFT STFT basis | (514, 1, 512) |
| Output formula | `clamp(resampled_base + bwe_residual, -1, 1)` |

### Audio VAE

| Parameter | Value |
|-----------|-------|
| Latent shape | (B, 8, T, 16) |
| Mel output shape | (B, 2, T', 64) |
| Stereo handling | Channels concatenated before vocoder: (B, 2, T, 64) -> (B, T, 128) |

---

## Upsampler Config

| Variant | Mid Channels | Blocks | Special |
|---------|-------------|--------|---------|
| `spatial_x2` | 1024 | 4 | Conv2d per-frame + PixelShuffle2D(2) |
| `spatial_x1.5` | 1024 | 4 | Conv2d + PixelShuffle2D(3) + BlurDownsample(stride=2, 5x5 binomial) |
| `temporal_x2` | 512 | 4 | Conv3d + PixelShuffle3D(temporal=2) + frame removal |

All use GroupNorm (not PixelNorm) in their ResBlocks.

---

## Diffusion Config (Distilled)

| Parameter | Value |
|-----------|-------|
| Steps | 8 |
| Sigma schedule | Predefined `DISTILLED_SIGMAS` |
| Classifier-free guidance | None (distilled model) |
| Prediction type | Velocity (v-prediction) |
