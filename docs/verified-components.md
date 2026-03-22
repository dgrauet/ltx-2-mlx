# Verified Components

Summary of all components verified correct by comparison with the PyTorch reference
implementation in [ltx-core](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core).

---

## Verification Status

| Component | File | Status | Max Diff vs Reference |
|-----------|------|--------|-----------------------|
| Timestep Embedding | `model/timestep_embedding.py` | Verified | 5e-05 |
| RoPE | `model/rope.py` | Verified | 4e-05 (cos/sin), 0 (freq grid) |
| Transformer Block | `model/transformer.py` | Verified | Identical (block 0) |
| Attention | `model/attention.py` | Verified | Identical (self-attn) |
| Output Block | `model/model.py` | Verified | Identical |
| VAE Decoder | `vae/decoder.py` | Verified | Correct output shapes and content |
| Upsampler | `upsampler/upsampler.py` | Verified | All 3 variants load and produce correct shapes |
| Audio Pipeline | `audio/` | Verified | Correct format and layout |
| Text Encoder | `text_encoder/` | Not yet verified | -- |

---

## Timestep Embedding

**File**: `src/ltx_2_mlx/model/timestep_embedding.py`

### `get_timestep_embedding`

- Sinusoidal embedding with **`[sin, cos]` order** (not `[cos, sin]`).
- Divisor: `(half_dim - downscale_freq_shift)` where `downscale_freq_shift=1`.
- Verified: max diff **5e-05** vs PyTorch reference.

### `AdaLayerNormSingle`

- Returns tuple `(params, embedded_timestep)`.
- `embedded_timestep` is reused in the output block for adaptive scale/shift.
- Architecture: `TimestepEmbedder(linear1 -> SiLU -> linear2) -> SiLU -> linear -> (params, embedded)`.

### Weight Keys

```
emb.timestep_embedder.linear1.weight / .bias
emb.timestep_embedder.linear2.weight / .bias
linear.weight / .bias
```

---

## RoPE (Rotary Position Embeddings)

**File**: `src/ltx_2_mlx/model/rope.py`

### Frequency Grid

- `generate_freq_grid(theta, num_pos_dims, inner_dim)`: log-spaced from `pi/2` to `theta * pi/2`.
- Number of frequencies: `inner_dim // (2 * num_pos_dims)`.
- Verified: **diff 0** vs PyTorch reference.

### Fractional Positions

- `position / max_pos` mapped to `[-1, 1]` via `frac * 2 - 1`.
- With `use_middle_indices_grid=True`, positions are midpoints: `i + 0.5`.
- Verified: **diff 0** vs PyTorch reference.

### Two Modes

| Mode | Usage | cos/sin Shape |
|------|-------|---------------|
| INTERLEAVED | Main transformer (default) | `(B, num_heads, N, head_dim)` |
| SPLIT | Text connector | `(B, num_heads, N, head_dim//2)` |

- **INTERLEAVED**: `repeat_interleave(2)` on cos/sin, pad to `inner_dim`.
- **SPLIT**: pad raw freqs to `inner_dim//2`, compute cos/sin directly.
- Verified: cos/sin max diff **4e-05** vs reference.

### Config Values

| Parameter | Video | Audio |
|-----------|-------|-------|
| `positional_embedding_theta` | 10000.0 | 10000.0 |
| `positional_embedding_max_pos` | [20, 2048, 2048] | [20] |

---

## Transformer Block

**File**: `src/ltx_2_mlx/model/transformer.py`

### `BasicAVTransformerBlock`

- Full block 0 output verified **identical** to PyTorch reference (range [-121, 576] matches exactly).

### 9 AdaLN Parameters

| Index | Purpose |
|-------|---------|
| 0-2 | Self-attention: shift, scale, gate |
| 3-5 | Feed-forward: shift, scale, gate |
| 6-8 | Cross-attention: shift_q, scale_q, gate |

### Key Implementation Details

- **Cross-attention AdaLN is ACTIVE** (`cross_attention_adaln=True`) despite config.json saying `false`. Confirmed by weights having 9 params + `prompt_scale_shift_table`.
- **Normalization**: `rms_norm` (parameterless) -- `x / sqrt(mean(x^2) + eps)`.

### Operation Order

1. **Self-attention**: `x + attn1(rms_norm(x) * (1+scale) + shift) * gate`
2. **Text cross-attention**: query modulated by indices [6-8], KV modulated by `prompt_scale_shift_table + prompt_adaln_params`, gated by gate[8].
3. **AV cross-attention**: 5-param table per side `[scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate]` + separate gate adaln.
4. **Feed-forward**: indices [3-5], `rms_norm(x) * (1+scale) + shift -> proj_in -> gelu_approx -> proj_out`, gated.

### Per-Head Gating

- Inside attention: `2 * sigmoid(to_gate_logits(x))`.

### Weight Keys Per Block

```
attn1.*                                  # Video self-attention
audio_attn1.*                            # Audio self-attention
attn2.*                                  # Video text cross-attention
audio_attn2.*                            # Audio text cross-attention
audio_to_video_attn.*                    # A->V cross-attention
video_to_audio_attn.*                    # V->A cross-attention
ff.proj_in.*                             # Video feed-forward
ff.proj_out.*                            # Video feed-forward
audio_ff.*                               # Audio feed-forward
scale_shift_table(9, dim)                # Video AdaLN params
audio_scale_shift_table(9, dim)          # Audio AdaLN params
prompt_scale_shift_table(2, dim)         # Video text cross-attn KV modulation
audio_prompt_scale_shift_table(2, dim)   # Audio text cross-attn KV modulation
scale_shift_table_a2v_ca_video(5, dim)   # AV cross-attn video side
scale_shift_table_a2v_ca_audio(5, dim)   # AV cross-attn audio side
```

---

## Output Block

**File**: `src/ltx_2_mlx/model/model.py`

### Adaptive Output Projection

```python
scale_shift_values = scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
# Then:
x = rms_norm(x) * (1 + scale) + shift -> proj_out
```

- `scale_shift_table` shape: `(2, dim)` at model top level.
- `embedded_timestep` comes from `AdaLayerNormSingle`.

---

## Attention

**File**: `src/ltx_2_mlx/model/attention.py`

- Self-attention output verified to **match reference exactly**.
- QK normalization with `nn.RMSNorm` **before** head reshape.
- RoPE applied **after** head reshape (equivalent to reference which applies before reshape).
- Per-head gating: `2 * sigmoid(to_gate_logits(x))`.
- `to_out` is a direct `nn.Linear` (not Sequential-wrapped like in the connector).

---

## VAE Decoder

**File**: `src/ltx_2_mlx/vae/decoder.py`

### Verified Behavior

- Produces colorful, spatially varied images from random latents.
- Correct frame count: **6 latent frames -> 41 pixel frames** (not 48).

### Architecture Details

- **PixelNorm**: parameterless RMS norm in ResBlocks.
- **ResBlock pattern**: pre-activation `norm -> silu -> conv1 -> norm -> silu -> conv2 + skip`.
- **CausalConv3d**: replicates first frame for temporal padding (not zero-pad).
- **Pixel shuffle**: C-outermost ordering `(B, D, H, W, C, tf, sf_h, sf_w)` matching PyTorch `rearrange("b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)")`.
- **Final pixel shuffle**: `sf=4, tf=1` (not `tf=16`).
- **Temporal frame removal**: `x[:, 1:]` after each temporal upsampling (tf > 1).

### Channel Flow

```
128 -> conv_in -> 1024
  -> ResBlocks + DepthToSpace(2,2,2) -> 512
  -> DepthToSpace(2,2,2) -> 512
  -> DepthToSpace(2,1,1) -> 256
  -> DepthToSpace(1,2,2) -> 128
  -> conv_out(48)
  -> pixel_shuffle(sf=4)
  -> 3 RGB
```

---

## Upsampler

**File**: `src/ltx_2_mlx/upsampler/upsampler.py`

### Three Verified Variants

| Variant | Op Type | Upsample Method | Config |
|---------|---------|-----------------|--------|
| `spatial_x2` | Conv2d per-frame | PixelShuffle2D(2) | mid=1024, 4 blocks |
| `spatial_x1.5` | Conv2d | PixelShuffle2D(3) + BlurDownsample(stride=2, kernel=5x5 binomial) | mid=1024, rational_resampler=true |
| `temporal_x2` | Conv3d | PixelShuffle3D(temporal=2) + frame removal | mid=512 |

### Architecture

```
initial_conv -> GroupNorm -> SiLU -> 4x ResBlock(GroupNorm) -> upsampler -> 4x post_ResBlock(GroupNorm) -> final_conv
```

### ResBlock

```
conv1 -> norm1 -> SiLU -> conv2 -> norm2 -> SiLU(x + residual)
```

---

## Audio Pipeline

**File**: `src/ltx_2_mlx/audio/`

### Verified Details

- **Vocoder stereo handling**: mel channels concatenated `(B, 2, T, 64) -> (B, T, 128)` before vocoder (not processed as separate channels).
- **Anti-aliased filters**: stored in MLX Conv1d format `(O, K, I)`, pre-transposed by mlx-forge.
- **STFT basis**: stored in MLX Conv1d format, pre-transposed by mlx-forge.
- **Audio VAE per-channel stats**: public names `mean_of_means` / `std_of_means` (MLX treats `_`-prefixed attributes as private). Remapped via `remap_audio_vae_keys()`.
