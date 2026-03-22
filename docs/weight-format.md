# Weight Format

All weights are pre-converted to MLX layout by [mlx-forge](https://github.com/dgrauet/mlx-forge)
and loaded as-is by this package. No weight conversion happens at runtime.

---

## Layout Conventions

| Layer Type | PyTorch Shape | MLX Shape | Notes |
|-----------|---------------|-----------|-------|
| Linear | (O, I) | (O, I) | No transpose needed |
| Conv1d | (O, I, K) | (O, K, I) | Pre-converted by mlx-forge |
| Conv2d | (O, I, H, W) | (O, H, W, I) | Pre-converted |
| Conv3d | (O, I, D, H, W) | (O, D, H, W, I) | Pre-converted |
| ConvTranspose1d | (I, O, K) | (O, K, I) | Pre-converted |
| Norm layers | (D,) | (D,) | No change |
| Buffers (filters, basis, kernel) | PyTorch layout | MLX layout | Also transposed by mlx-forge |

---

## Weight Files

### Core Model (6 files)

| File | Prefix | Content |
|------|--------|---------|
| `transformer.safetensors` | `transformer.` | DiT blocks (quantized in q8/q4 variants) |
| `connector.safetensors` | N/A | Text embeddings connectors |
| `vae_decoder.safetensors` | `vae_decoder.` | Video VAE decoder + per-channel stats |
| `vae_encoder.safetensors` | `vae_encoder.` | Video VAE encoder + per-channel stats |
| `audio_vae.safetensors` | `audio_vae.` | Audio VAE decoder + per-channel stats |
| `vocoder.safetensors` | `vocoder.` | Base vocoder + BWE generator + mel STFT |

### Upsampler (3 files)

| File | Content |
|------|---------|
| `spatial_upscaler_x2_v1_1.safetensors` | 2x spatial upsampler (Conv2d) |
| `spatial_upscaler_x1_5_v1_0.safetensors` | 1.5x spatial upsampler (Conv2d + blur downsample) |
| `temporal_upscaler_x2_v1_0.safetensors` | 2x temporal upsampler (Conv3d) |

---

## Key Remapping (mlx-forge)

These remappings are applied during weight conversion by mlx-forge, not at load time.

### Transformer Block Feed-Forward

| PyTorch Key | MLX Key |
|-------------|---------|
| `net.0.proj.weight` | `proj_in.weight` |
| `net.0.proj.bias` | `proj_in.bias` |
| `net.2.weight` | `proj_out.weight` |
| `net.2.bias` | `proj_out.bias` |

### Transformer Attention `to_out`

| PyTorch Key | MLX Key |
|-------------|---------|
| `to_out.0.weight` | `to_out.weight` |
| `to_out.0.bias` | `to_out.bias` |

The main transformer attention removes the Sequential wrapping from `to_out`.

### Connector Attention `to_out`

The connector **keeps** the Sequential wrapping:

| PyTorch Key | MLX Key |
|-------------|---------|
| `to_out.0.weight` | `to_out.0.weight` |
| `to_out.0.bias` | `to_out.0.bias` |

This is intentional -- the connector's attention uses a different code path than the main transformer's attention.

### Audio VAE Per-Channel Stats

| PyTorch Key | MLX Key | Reason |
|-------------|---------|--------|
| `_mean_of_means` | `mean_of_means` | MLX treats `_`-prefixed attributes as private |
| `_std_of_means` | `std_of_means` | Same reason |

Remapped at load time via `remap_audio_vae_keys()`.

---

## Transformer Weight Keys

### Top-Level

```
adaln_single.emb.timestep_embedder.linear1.weight    (dim*4, dim)
adaln_single.emb.timestep_embedder.linear1.bias      (dim*4,)
adaln_single.emb.timestep_embedder.linear2.weight     (dim, dim*4)
adaln_single.emb.timestep_embedder.linear2.bias       (dim,)
adaln_single.linear.weight                            (9*dim, dim)
adaln_single.linear.bias                              (9*dim,)
scale_shift_table                                     (2, dim)
proj_out.weight                                       (patch_out, dim)
proj_out.bias                                         (patch_out,)
patchify_proj.weight                                  (dim, 1, 1, 1, patch_in)
patchify_proj.bias                                    (dim,)
```

Where `dim=4096` for video, `dim=2048` for audio paths.

### Per-Block (48 blocks)

```
transformer_blocks.{i}.scale_shift_table                      (9, dim)
transformer_blocks.{i}.audio_scale_shift_table                (9, audio_dim)
transformer_blocks.{i}.prompt_scale_shift_table               (2, dim)
transformer_blocks.{i}.audio_prompt_scale_shift_table         (2, audio_dim)
transformer_blocks.{i}.scale_shift_table_a2v_ca_video         (5, dim)
transformer_blocks.{i}.scale_shift_table_a2v_ca_audio         (5, audio_dim)

# Video self-attention
transformer_blocks.{i}.attn1.to_q.weight                     (dim, dim)
transformer_blocks.{i}.attn1.to_k.weight                     (dim, dim)
transformer_blocks.{i}.attn1.to_v.weight                     (dim, dim)
transformer_blocks.{i}.attn1.to_out.weight                   (dim, dim)
transformer_blocks.{i}.attn1.to_out.bias                     (dim,)
transformer_blocks.{i}.attn1.q_norm.weight                   (dim,)
transformer_blocks.{i}.attn1.k_norm.weight                   (dim,)
transformer_blocks.{i}.attn1.to_gate_logits.weight           (num_heads, dim)

# Audio self-attention
transformer_blocks.{i}.audio_attn1.to_q.weight               (audio_dim, audio_dim)
transformer_blocks.{i}.audio_attn1.to_k.weight               (audio_dim, audio_dim)
transformer_blocks.{i}.audio_attn1.to_v.weight               (audio_dim, audio_dim)
transformer_blocks.{i}.audio_attn1.to_out.weight             (audio_dim, audio_dim)
transformer_blocks.{i}.audio_attn1.to_out.bias               (audio_dim,)
transformer_blocks.{i}.audio_attn1.q_norm.weight             (audio_dim,)
transformer_blocks.{i}.audio_attn1.k_norm.weight             (audio_dim,)
transformer_blocks.{i}.audio_attn1.to_gate_logits.weight     (audio_heads, audio_dim)

# Video text cross-attention
transformer_blocks.{i}.attn2.to_q.weight                     (dim, dim)
transformer_blocks.{i}.attn2.to_k.weight                     (dim, dim)
transformer_blocks.{i}.attn2.to_v.weight                     (dim, dim)
transformer_blocks.{i}.attn2.to_out.weight                   (dim, dim)
transformer_blocks.{i}.attn2.to_out.bias                     (dim,)
transformer_blocks.{i}.attn2.q_norm.weight                   (dim,)
transformer_blocks.{i}.attn2.k_norm.weight                   (dim,)
transformer_blocks.{i}.attn2.to_gate_logits.weight           (num_heads, dim)

# Audio text cross-attention
transformer_blocks.{i}.audio_attn2.to_q.weight               (audio_dim, audio_dim)
transformer_blocks.{i}.audio_attn2.to_k.weight               (audio_dim, audio_dim)
transformer_blocks.{i}.audio_attn2.to_v.weight               (audio_dim, audio_dim)
transformer_blocks.{i}.audio_attn2.to_out.weight             (audio_dim, audio_dim)
transformer_blocks.{i}.audio_attn2.to_out.bias               (audio_dim,)
transformer_blocks.{i}.audio_attn2.q_norm.weight             (audio_dim,)
transformer_blocks.{i}.audio_attn2.k_norm.weight             (audio_dim,)
transformer_blocks.{i}.audio_attn2.to_gate_logits.weight     (audio_heads, audio_dim)

# Audio-to-Video cross-attention
transformer_blocks.{i}.audio_to_video_attn.to_q.weight       (dim, dim)
transformer_blocks.{i}.audio_to_video_attn.to_k.weight       (dim, audio_dim)
transformer_blocks.{i}.audio_to_video_attn.to_v.weight       (dim, audio_dim)
transformer_blocks.{i}.audio_to_video_attn.to_out.weight     (dim, dim)
transformer_blocks.{i}.audio_to_video_attn.to_out.bias       (dim,)
transformer_blocks.{i}.audio_to_video_attn.q_norm.weight     (dim,)
transformer_blocks.{i}.audio_to_video_attn.k_norm.weight     (dim,)
transformer_blocks.{i}.audio_to_video_attn.to_gate_logits.weight  (num_heads, dim)

# Video-to-Audio cross-attention
transformer_blocks.{i}.video_to_audio_attn.to_q.weight       (audio_dim, audio_dim)
transformer_blocks.{i}.video_to_audio_attn.to_k.weight       (audio_dim, dim)
transformer_blocks.{i}.video_to_audio_attn.to_v.weight       (audio_dim, dim)
transformer_blocks.{i}.video_to_audio_attn.to_out.weight     (audio_dim, audio_dim)
transformer_blocks.{i}.video_to_audio_attn.to_out.bias       (audio_dim,)
transformer_blocks.{i}.video_to_audio_attn.q_norm.weight     (audio_dim,)
transformer_blocks.{i}.video_to_audio_attn.k_norm.weight     (audio_dim,)
transformer_blocks.{i}.video_to_audio_attn.to_gate_logits.weight  (audio_heads, audio_dim)

# Video feed-forward
transformer_blocks.{i}.ff.proj_in.weight                     (dim*4, dim)
transformer_blocks.{i}.ff.proj_in.bias                       (dim*4,)
transformer_blocks.{i}.ff.proj_out.weight                    (dim, dim*4)
transformer_blocks.{i}.ff.proj_out.bias                      (dim,)

# Audio feed-forward
transformer_blocks.{i}.audio_ff.proj_in.weight               (audio_dim*4, audio_dim)
transformer_blocks.{i}.audio_ff.proj_in.bias                 (audio_dim*4,)
transformer_blocks.{i}.audio_ff.proj_out.weight              (audio_dim, audio_dim*4)
transformer_blocks.{i}.audio_ff.proj_out.bias                (audio_dim,)
```

Where `dim=4096`, `audio_dim=2048`, `num_heads=32`, `audio_heads=32`.

---

## Quantization (q8/q4 Variants)

- Only `nn.Linear` layers inside `transformer_blocks` are quantized.
- Quantization: int8 with `group_size=64` (q8), int4 with `group_size=64` (q4).
- **Never quantized** (must stay bf16):
  - `adaln_single` (timestep embedding)
  - `proj_out` (output projection)
  - `patchify_proj` (input patch embedding)
  - Connectors
  - VAE (encoder + decoder)
  - Vocoder + BWE
- MLX can only quantize `Linear` and `Embedding` layers, never Conv layers.

---

## Connector Weight Keys

```
video_connector.*          # Embeddings1DConnector for video projection
audio_connector.*          # Embeddings1DConnector for audio projection
video_feature_extractor.*  # GemmaFeaturesExtractorV2 video path
audio_feature_extractor.*  # GemmaFeaturesExtractorV2 audio path
```

Connector attention layers keep Sequential wrapping (`to_out.0.*`), unlike main transformer attention.
