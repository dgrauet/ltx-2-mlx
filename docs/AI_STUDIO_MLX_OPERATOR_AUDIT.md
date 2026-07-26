# AI Studio MLX operator audit

Observed 2026-07-26 against clean `ltx-2-mlx` commit
`36ab612919ccba4b90321019a2927633dd837d35`, package version 0.14.18,
MLX/MLX Metal 0.31.1, and the cached BF16
`dgrauet/ltx-2.3-mlx` snapshot
`baa5f235ea04fd9c95899d751295c4fd825ee4e2`.

This is a static operator and synchronization audit of AI Studio's BF16
`DistilledPipeline` path. It keeps the UI's 193-frame default separate from the
controlled 129-frame benchmark used for the quantitative trace envelope. No
generation or Metal capture was run for this audit. Operator multiplicities
below are derived from source; performance rankings are hypotheses until
measured with a Metal trace.

## Request envelope

The AI Studio **UI default** is distilled, 8 stage-1 steps, 193 frames at
24 fps, source-oriented 768×512 or 512×768 output, and low-RAM streaming off
unless the user selects it. This is a product-default reference, not the
measured benchmark
(`ai_studio/templates/index.html:3189-3214,3230-3238`;
`ltx23_mlx_single.py:78-80,111-118`).

For either orientation, that 193-frame UI default has:

| Quantity | Stage 1 | Stage 2 |
|---|---:|---:|
| Pixel resolution | 384×256 | 768×512 |
| Video latent shape `(F,H,W)` | `(25,8,12)` | `(25,16,24)` |
| Video tokens `Nv` | 2,400 | 9,600 |
| Audio tokens `Na` | 201 | 201 |
| Text tokens `Nt` | up to 1,024 | up to 1,024 |
| DiT forwards | 8 | 3 |

The **controlled benchmark** is distilled, 129 frames at 24 fps, 768×512
output, and the same 8+3 denoising schedule
(`docs/AI_STUDIO_MLX_BASELINE_2026-07-26.md:25-37`). Its operator shapes are:

| Quantity | Stage 1 | Stage 2 |
|---|---:|---:|
| Pixel resolution | 384×256 | 768×512 |
| Video latent shape `(F,H,W)` | `(17,8,12)` | `(17,16,24)` |
| Video tokens `Nv` | 1,632 | 6,528 |
| Audio tokens `Na` | 134 | 134 |
| Text tokens `Nt` | up to 1,024 | up to 1,024 |
| DiT forwards | 8 | 3 |

The shape formulas are
`ceil(num_frames/8), height/32, width/32` and
`round((num_frames/fps) * 25)` audio tokens
(`ltx_core_mlx/components/patchifiers.py:102-124`;
`ltx_core_mlx/utils/positions.py:11-31`).

Warm-run medians for orientation are:

| Controlled 129-frame metric | Warm median |
|---|---:|
| Total | 69.792485 s |
| Stage 1 half-resolution denoise | 19.113536 s |
| Stage 2 full-resolution denoise | 31.774556 s |
| Decode and mux | 12.310893 s |
| Peak MLX memory | 40.3603609 GiB |

The timer spans are pipeline phases, not operator attribution. Stage 1 encloses
the complete first denoise loop, stage 2 encloses the complete second loop, and
decode includes video VAE, audio VAE/vocoder, host conversion, and muxing
(`ltx_pipelines_mlx/distilled.py:224-248,315-334`;
`ltx_pipelines_mlx/_base.py:347-367`;
`ltx_pipelines_mlx/utils/progress.py:31-65`). These medians show where request
time accumulates at phase granularity; they do **not** establish which
attention, linear, normalization, FF, synchronization, or other operation
inside the DiT consumed that time.

Additional nonzero-index keyframes append conditioning tokens and Prompt Relay
adds a video-to-text attention bias. They are not part of this ordinary
single-source baseline.

“BF16 request” refers to the LTX transformer and latent path. The prompt model
is 4-bit Gemma, `X0Model` performs its final subtraction in fp32, and the
vocoder intentionally runs in fp32.

## End-to-end operator path

```text
Pillow/NumPy/ffmpeg source preprocessing
  → video VAE encode at half resolution
  → patch/reshape source conditioning
  → noised video + audio latent tokens
  → 8 × X0Model(LTXModel(48 × BasicAVTransformerBlock))
  → latent unpatchify
  → normalize/denormalize + 2× latent upsampler
  → video VAE encode source again at full resolution
  → patchify upscaled latent
  → 3 × X0Model(LTXModel(48 × BasicAVTransformerBlock))
  → video/audio unpatchify
  → audio VAE decoder → BigVGAN → BWE
  → video VAE decoder
  → NumPy/WAV + RGB pipe → ffmpeg
```

`DistilledPipeline` implements the half-resolution denoise, latent upsample,
full-resolution denoise, and final unpatchify directly
(`ltx_pipelines_mlx/distilled.py:149-340`).

## `X0Model` and `LTXModel`

### `X0Model`

Each denoise step calls `X0Model`, which invokes `LTXModel` once and converts
its velocity prediction to `x0`:

```text
x0 = noisy_latent - sigma * velocity
```

The subtraction and multiplication are explicitly promoted to fp32, then cast
back to the input latent dtype
(`ltx_core_mlx/model/transformer/model.py:631-705`). This is MLX array math;
there is no NumPy or host transition.

### `LTXModel` prelude

For every one of the 11 DiT forwards, `LTXModel`:

1. casts video/audio latents and text embeddings to BF16;
2. projects 128-channel video and audio tokens to 4096- and 2048-wide hidden
   states with `nn.Linear`;
3. computes scalar or per-token sinusoidal timestep embeddings;
4. runs eight AdaLN MLP/projection modules for video, audio, prompt, and
   bidirectional audio/video gates;
5. computes four RoPE frequency sets: video self, audio self, video temporal
   cross-modal, and audio temporal cross-modal; and
6. enters the 48-block stack.

Evidence:
`ltx_core_mlx/model/transformer/model.py:179-220,316-475`.

The BF16 primary-image path has per-token video timesteps because the source
frame is preserved by a denoise mask. Audio normally uses the scalar timestep.
AdaLN therefore broadcasts either per-batch or per-token scale, shift, and
gate tensors (`model.py:239-275,404-432`).

## One `BasicAVTransformerBlock`

Every block performs these eight branches in order:

| Order | Branch | Query tokens | Key/value tokens | RoPE | Attention primitive |
|---:|---|---|---|---|---|
| 1 | Video self-attention | video | video | 3D | fused MLX SDPA |
| 2 | Audio self-attention | audio | audio | 1D | fused MLX SDPA |
| 3 | Video → text cross-attention | video | text | none | fused MLX SDPA |
| 4 | Audio → text cross-attention | audio | text | none | fused MLX SDPA |
| 5 | Audio → video cross-modal | video | audio | temporal 1D | fused MLX SDPA |
| 6 | Video → audio cross-modal | audio | video | temporal 1D | fused MLX SDPA |
| 7 | Video feed-forward | video | — | — | Linear → GELU approx → Linear |
| 8 | Audio feed-forward | audio | — | — | Linear → GELU approx → Linear |

The block definitions and order are explicit in
`ltx_core_mlx/model/transformer/transformer.py:29-153,261-397`.

Before the branches, the block unpacks timestep AdaLN tensors and adds its
learned scale/shift tables. Each attention/FF input uses affine-free
`mx.fast.rms_norm`, then elementwise scale/shift modulation. Each residual is
multiplied by its AdaLN gate before addition
(`transformer.py:155-205,261-395`).

### Main attention implementation

Every one of the six attention branches uses the same `Attention` module:

1. three independent `nn.Linear` Q/K/V projections;
2. fused `nn.RMSNorm` on Q and K;
3. reshape/transpose to `(B,heads,N,head_dim)`;
4. compiled RoPE on Q/K when applicable;
5. `mx.fast.scaled_dot_product_attention`;
6. per-head sigmoid gating; and
7. output transpose/reshape and `nn.Linear`.

Evidence:
`ltx_core_mlx/model/transformer/attention.py:36-143`.

The source calls the MLX primitive a fused Flash Attention kernel
(`attention.py:126-127`). The actual Metal kernel selected for a particular
shape/dtype is an MLX backend decision and must be confirmed in a capture.
There is no Python loop over heads or tokens and no explicit `Nv×Nv` score
tensor in this main DiT implementation.

### Per-request multiplicity

One block contains:

- 6 fused SDPA calls;
- 30 attention linears: Q, K, V, output, and gate for each branch;
- 4 feed-forward linears;
- 8 direct affine-free fast RMS norms around the block branches;
- 12 fast RMS norms inside the six attention modules; and
- 8 compiled RoPE applications: Q and K for four RoPE-enabled attentions.

For either 8+3 distilled envelope:

| Operation | Count |
|---|---:|
| Complete DiT forwards | 11 |
| Transformer-block executions | 528 |
| Fused main-DiT SDPA calls | 3,168 |
| Feed-forward networks | 1,056 |
| Block-local `nn.Linear` calls | 17,952 |
| Block/attention fast RMS norms | 10,560 |
| Compiled main-DiT RoPE applications | 4,224 |

These counts exclude the top-level patch/output projections, AdaLN MLPs,
Gemma, connector, VAE, upsampler, and vocoder.

In the controlled benchmark, stage 2 has only three forwards but its 6,528
video tokens are four times stage 1's 1,632. Video self-attention pair count
scales quadratically: the source-derived stage-2 `Nv² × steps` term is exactly
six times the stage-1 term. The same 6× ratio holds for the 193-frame UI
default because its stage-2 spatial token count is also 4× stage 1. This makes
stage-2 video attention the first trace target, not proof that it is the
wall-clock bottleneck; the warm phase medians do not attribute stage-2 time
within the DiT.

### Feed-forward and output head

Each modality's FF is
`nn.Linear → nn.gelu_approx → nn.Linear`
(`ltx_core_mlx/model/transformer/feed_forward.py:16-32`).

After block 48, video and audio each use:

1. `mx.fast.layer_norm`;
2. output AdaLN scale/shift; and
3. one linear projection back to 128 latent channels.

Evidence:
`ltx_core_mlx/model/transformer/model.py:559-594`.

## MLX primitives already fused or compiled

### Explicit fused primitives

| Source path | Primitive | Active use |
|---|---|---|
| `model/transformer/attention.py:126-127` | `mx.fast.scaled_dot_product_attention` | all six attentions in all 48 DiT blocks |
| `model/transformer/transformer.py:178-180` | `mx.fast.rms_norm` | eight affine-free block pre-norms |
| `model/transformer/model.py:591-594` | `mx.fast.layer_norm` | video/audio DiT output heads |
| `model/video_vae/normalization.py:11-17` | `mx.fast.rms_norm` | video VAE PixelNorm |
| `model/audio_vae/audio_vae.py:33-35` | `mx.fast.rms_norm` | audio VAE PixelNorm |
| `text_encoders/gemma/embeddings_connector.py:219-225` | `mx.fast.rms_norm` | connector pre/output normalization |

Pinned MLX `nn.RMSNorm` delegates directly to `mx.fast.rms_norm`, and
`nn.LayerNorm` delegates to `mx.fast.layer_norm`
(`.venv/lib/python3.11/site-packages/mlx/nn/layers/normalization.py:69-143`).
Thus attention Q/K normalization is also on the fused fast path.

Pinned MLX `nn.Linear` uses `mx.addmm` when a bias is present
(`.venv/lib/python3.11/site-packages/mlx/nn/layers/linear.py:26-71`).
The main DiT attention and FF linears all have bias, so their matrix
multiplication and bias addition already enter MLX as one `addmm` primitive.
This does not fuse Q, K, and V into one projection; they remain three linears.

`nn.Conv1d/2d/3d` map to the corresponding MLX convolution primitives, with a
separate lazy bias addition
(`.venv/lib/python3.11/site-packages/mlx/nn/layers/convolution.py:76-82,156-162,228-232`).

### Explicit `mx.compile`

The repository has only these explicit compile sites:

- SPLIT and INTERLEAVED RoPE application functions are decorated with
  `@mx.compile`
  (`ltx_core_mlx/model/transformer/rope.py:156-205`).
- Low-RAM block streaming compiles one shared
  `BasicAVTransformerBlock` with `mx.compile(shared, inputs=shared)`, allowing
  its parameters to be rebound for each of 48 layers
  (`ltx_core_mlx/loader/block_streaming.py:327-374`).

The ordinary AI Studio default has low-RAM **off**, so it uses 48 eager block
instances. RoPE remains compiled; the entire eager block, full DiT forward,
upsampler, video VAE, audio VAE, and vocoder are not explicitly compiled.
MLX still builds lazy graphs and performs its normal primitive-level backend
fusion.

The 4-bit Gemma implementation supplied by pinned `mlx-lm` uses its SDPA
wrapper, which maps the unquantized-cache case to
`mx.fast.scaled_dot_product_attention`
(`.venv/lib/python3.11/site-packages/mlx_lm/models/gemma3_text.py:35-101`;
`.venv/lib/python3.11/site-packages/mlx_lm/models/base.py:108-137`).
Gemma's residual helper is also compiled shapeless
(`gemma3_text.py:104-160`).

### MLX's inspectable Metal layer and M5 tuning

The installed MLX 0.31.1 distribution ships 98 public/open Metal kernel header
files under
`.venv/lib/python3.11/site-packages/mlx/include/mlx/backend/metal/`, together
with the compiled
`.venv/lib/python3.11/site-packages/mlx/lib/mlx.metallib`. The headers include
Steel attention, GEMM, split-K GEMM, convolution, reduction, softmax, and other
kernel implementations. This gives us a supported, inspectable implementation
layer above Apple's private Metal driver and compiler binaries; investigating
MLX's dispatch and kernels does not require reverse-engineering those Apple
binaries.

The installed API also exposes library-scoped
`mlx.core.metal.start_capture(path)` and `stop_capture()`
(`.venv/lib/python3.11/site-packages/mlx/core/metal.pyi:25-35`). That is the
preferred way to capture narrowly bounded LTX phases or exact operator
reproductions without placing the entire process under a system-wide trace.

[MLX v0.31.1's release notes](https://github.com/ml-explore/mlx/releases/tag/v0.31.1)
include the merged initial M5 Pro/Max tuning work in
[MLX PR #3211](https://github.com/ml-explore/mlx/pull/3211). The PR reports
approximately 52.7–59.6 TFLOP/s on an M5 Max for representative float16 GEMMs
whose dimensions use 4096 and 14336, and explicitly leaves split-K tuning for a
separate follow-up. LTX's repeated 4096→16384→4096 feed-forward and projection
shapes are close to, but not the same as, those published 4096/14336 cases.
Therefore custom GEMM work should begin by capturing and microbenchmarking the
exact LTX shapes to verify which MLX route, tile, and split-K policy 0.31.1
selects. The existence of M5 tuning is evidence against starting with a GEMM
rewrite, not evidence that LTX's shapes already take the best route.

## RoPE

The main DiT precomputes log-spaced SPLIT RoPE frequencies once per forward,
not once per block. It computes normalized fractional positions, cos/sin
tensors, and per-head layouts with MLX array operations
(`ltx_core_mlx/model/transformer/rope.py:22-153`).

Each RoPE-enabled attention then invokes the compiled SPLIT function separately
for Q and K. The function slices the two half-head components, performs the
rotation with elementwise multiply/add/subtract, and concatenates the result
(`rope.py:186-205`).

Potential frequency caching across denoise steps is a hypothesis: positions and
shapes are stable within a stage, but the current `LTXModel` recomputes all four
frequency sets on every forward. A trace must first show this work is material
relative to attention/GEMM.

## Prompt encoder and connector

The prompt path runs once before the DiT:

- Gemma produces the embedding plus 48 layer outputs, with one `mx.eval` per
  layer by default
  (`text_encoders/gemma/encoders/base_encoder.py:105-163`).
- The 49×3840 hidden states are projected separately to 4096 video and 2048
  audio dimensions, with a materialization after each large projection
  (`text_encoders/gemma/feature_extractor.py:68-90`).
- Separate video and audio connectors each run eight transformer blocks
  (`feature_extractor.py:119-189`).

Unlike the main DiT and Gemma, connector attention manually computes
`Q @ Kᵀ`, adds the mask, applies `mx.softmax`, then multiplies by V
(`text_encoders/gemma/embeddings_connector.py:65-119`). It therefore does not
call fused MLX SDPA. This is a real operator difference, but it runs for only
16 connector blocks once per generation. Replacing it with fused SDPA is a
high-value correctness/performance experiment only if a trace shows material
time or memory and the mask/RoPE parity test passes.

## Patchify and unpatchify

Transformer video/audio patchifiers are transpose+reshape operations:

- video `(B,C,F,H,W) ↔ (B,F·H·W,C)`;
- audio `(B,8,T,16) ↔ (B,T,128)`.

Evidence:
`ltx_core_mlx/components/patchifiers.py:18-99`.

Video VAE pixel patchification and depth-to-space are also explicit
reshape/transpose sequences
(`ltx_core_mlx/model/video_vae/sampling.py:18-124`).
They do not call a custom kernel. Whether a transpose is a cheap view or causes
a contiguous materialization depends on its consumer and MLX's graph lowering;
source inspection alone cannot label it a copy or a hotspot.

## 2× latent upsampler

The cached `spatial_upscaler_x2_v1_1` is:

- 128 input channels;
- 1024 middle channels;
- Conv3d → GroupNorm → SiLU;
- four pre-upsample 3D residual blocks;
- per-frame Conv2d to 4× channels plus 2D pixel shuffle;
- four post-upsample 3D residual blocks; and
- final Conv3d back to 128 channels.

The model snapshot config is
`spatial_upscaler_x2_v1_1_config.json:2-11`.
The implementation is
`ltx_core_mlx/model/upsampler/model.py:106-132,207-359`.

No upsampler function is explicitly compiled and it has no direct `mx.fast`
call. Its convolutions are standard MLX convolution operators; GroupNorm is
the standard MLX module; SiLU and residual arithmetic remain in MLX's lazy
graph. Pixel shuffle is reshape/transpose/reshape.

`DistilledPipeline` materializes the normalized upscaled latent before it
constructs stage 2 (`ltx_pipelines_mlx/distilled.py:249-284`).

## Video VAE

### I2V encoder

The ordinary source image is VAE-encoded twice, once at each stage's spatial
resolution. The encoder performs:

1. channel-last transpose and 4×4 spatial patchification;
2. Conv3d from 48 to 128 channels;
3. 18 PixelNorm/SiLU/two-Conv3d residual blocks distributed through five
   stages;
4. four space-to-depth downsamplers with Conv3d and a group-mean skip;
5. final PixelNorm/SiLU/Conv3d;
6. discard the non-mean output channel; and
7. per-channel latent normalization.

Architecture and live forward:
`ltx_core_mlx/model/video_vae/video_vae.py:526-634`;
residual primitive:
`ltx_core_mlx/model/video_vae/resnet.py:15-72`;
padding/convolution:
`ltx_core_mlx/model/video_vae/convolution.py:12-96`.

Video PixelNorm is fused RMSNorm. The Conv3d, padding, reshape, mean, SiLU, and
residual operators are ordinary MLX primitives and lazy elementwise graphs.

### Decoder

The decoder performs Conv3d 128→1024, then:

- 18 two-Conv3d PixelNorm/SiLU residual blocks;
- four Conv3d depth-to-space upsamplers;
- three temporal first-frame drops;
- final PixelNorm/SiLU/Conv3d 128→48; and
- 4× spatial unpatchify to RGB.

Architecture and forward:
`ltx_core_mlx/model/video_vae/video_vae.py:140-289`.

The decoder estimates its activation peak and switches to temporal tiling only
above the configurable 8 GiB budget
(`video_vae.py:66-82,291-440`). Both the controlled 768×512×129 benchmark and
the 768×512×193 UI default are below that static estimate, so the source selects
the untiled path for either envelope. It materializes the complete decoded
pixel tensor once, then streams one materialized uint8 frame at a time
(`video_vae.py:309-315,418-523`).

For larger tiled requests, the decoder additionally materializes after each
upsample stage, each decoded tile, and each accumulation-buffer update
(`video_vae.py:275-277,342-415`). Its tiled accumulation and weight buffers are
currently fp32, with a source TODO to evaluate BF16
(`video_vae.py:336-340`). That is outside the ordinary default path.

## Audio VAE and vocoder

The audio decoder converts `(B,8,T,16)` latent to a stereo mel tensor using
Conv2d, PixelNorm, SiLU, nine residual blocks, two nearest-repeat+Conv2d
upsamples, and a final Conv2d
(`ltx_core_mlx/model/audio_vae/audio_vae.py:255-329`).

The file contains a manual `Q @ Kᵀ → softmax → @ V` `AudioAttnBlock`
(`audio_vae.py:125-155`), but the LTX-2.3 distilled configuration explicitly
sets `mid_block_add_attention=False` and `attn_resolutions=[]`. No audio-VAE
attention block executes in this audited request
(`audio_vae.py:267-284`).

The vocoder:

1. upcasts mel and all weights to fp32;
2. runs a six-stage BigVGAN base vocoder;
3. uses repeated ConvTranspose1d, anti-aliased SnakeBeta activation, Conv1d,
   and residual blocks;
4. computes a BWE mel transform using Conv1d STFT bases and a mel matrix
   multiply;
5. runs a five-stage BigVGAN BWE generator;
6. applies a 3× sinc resampler implemented with zero insertion and Conv1d; and
7. sums/clips the BWE residual and resampled base waveform.

Evidence:
`ltx_core_mlx/model/audio_vae/vocoder.py:38-232,240-339`;
`ltx_core_mlx/model/audio_vae/bwe.py:28-116,144-210,218-405`.

There is no explicit compile or fast primitive in the vocoder. It uses standard
MLX convolution, transposed convolution, matrix multiplication, and
elementwise operators. Its 108 sequential convolutions are a plausible trace
target, not a proven request bottleneck.

## Explicit evaluation and synchronization boundaries

### Prompt and load

| Boundary | Default count | Evidence |
|---|---:|---|
| Gemma per-layer `mx.eval` | 48 | `text_encoders/gemma/encoders/base_encoder.py:143-161` |
| Large text projection materialization | 2 | `text_encoders/gemma/feature_extractor.py:80-88` |
| Video/audio connector per-block eval | 16 | `text_encoders/gemma/embeddings_connector.py:335-343` |
| Final prompt embeddings eval | 1 | `ltx_pipelines_mlx/distilled.py:149-156` |
| Transformer parameter materialization | 1 | `ltx_pipelines_mlx/utils/_orchestration.py:89-123` |

### Denoising

| Boundary | Default count | Evidence |
|---|---:|---|
| Pre-denoise state `mx.eval` | 2 | `ltx_pipelines_mlx/_base.py:503-517` |
| Eager DiT eval every 8 of 48 blocks | 66 across 11 forwards | `ltx_core_mlx/model/transformer/model.py:32-40,482-557` |
| End-of-step `mx.async_eval` | 11 | `ltx_pipelines_mlx/utils/samplers.py:135-174` |
| Upscaled latent materialization | 1 | `ltx_pipelines_mlx/distilled.py:249-261` |

When low-RAM streaming is enabled, the block provider forces an eval after
**every** block so the previous block's mapped weights can be evicted: 528
block-level evals for the 8+3 request
(`ltx_core_mlx/model/transformer/model.py:549-557`;
`ltx_core_mlx/loader/block_streaming.py:345-374`).

### Decode/output

The untiled default video decoder materializes its complete pixel tensor once,
then performs one `mx.eval(frame_hwc)` per output frame before CPU/ffmpeg access
(`ltx_core_mlx/model/video_vae/video_vae.py:309-315,494-519`).

`aggressive_cleanup()` calls Python garbage collection and
`mx.clear_cache()` (`ltx_core_mlx/utils/memory.py:8-15`). It is a cache/lifetime
operation, not a substitute for the explicit eval boundaries above.

These barriers are intentional watchdog and memory controls. Their throughput
cost should be captured, but removing them without a contended stability test
would reverse proven crash mitigations.

## Python, NumPy, and host/media boundaries

No Python or NumPy operation occurs inside the 48-block DiT arithmetic. Python
loops select 48 blocks and 11 denoise steps; their tensor operations remain
lazy MLX graphs executed by the backend.

The ordinary I2V request does cross these host boundaries:

| Boundary | Exact behavior | Evidence |
|---|---|---|
| Source image decode | Pillow → uint8 NumPy | `ltx_pipelines_mlx/utils/media_io.py:57-64` |
| Training-distribution preprocessing | NumPy `tobytes` → ffmpeg H.264 encode → in-memory MP4 → ffmpeg probe/decode → NumPy copy | `media_io.py:67-189,203-224` |
| Resize/crop | Pillow Lanczos and crop | `media_io.py:226-245` |
| Host → MLX source tensor | NumPy float32 normalize → `mx.array` → BF16 | `media_io.py:248-281` |
| Repeated conditioning | the entire image load/preprocess/encode path runs once at half and once at full resolution | `ltx_pipelines_mlx/distilled.py:182-201,267-279` |
| Audio MLX → host | full waveform becomes float32 NumPy, int16 NumPy, then `tobytes()` into a temporary WAV | `ltx_pipelines_mlx/utils/_orchestration.py:143-161,188-196` |
| Video MLX → host | each RGB frame is materialized, wrapped in `memoryview`, copied to `bytes`, and written to ffmpeg stdin | `ltx_core_mlx/model/video_vae/video_vae.py:494-519` |
| Codec/mux | separate ffmpeg process performs H.264 CRF-18/AAC output | `video_vae.py:463-489` |

Apple Silicon unified memory means there is no discrete PCIe host/device copy
for ordinary MLX array use. The NumPy arrays, `bytes`, temporary WAV, and
subprocess pipes above are still real allocations and serialization
boundaries.

## Metal capture status

A full Metal System Trace was too intrusive to characterize this workload. Both
the production request and a reduced 384×256×33 q8 request stalled at
`Preparing Stage 1` under that capture mode. In the reduced attempt, the trace
contained 243 application command buffers ending at approximately trace
`t=3.73 s`, followed by no later LTX GPU submissions before the process was
terminated at 180 seconds.

Those attempts provide no valid hotspot, kernel-duration, utilization, or
command-buffer-gap inference: the tracing method changed forward progress
before the denoise phase being investigated. The replacement is a
library-scoped capture using `mlx.core.metal.start_capture()` /
`stop_capture()` around a narrowly selected stage or exact-shape operator
reproduction, followed by an uncaptured parity/timing control. That replacement
subsequently succeeded for a stage-2 micro workload; its structural findings
and artifact manifest are recorded in
[`AI_STUDIO_MLX_METAL_CAPTURE_2026-07-26.md`](AI_STUDIO_MLX_METAL_CAPTURE_2026-07-26.md).

## Ranked Metal/CPU trace targets

Every item below is a **hypothesis**, not a finding.

1. **Stage-2 video self-attention and its Q/K/V/output projections.** The
   controlled benchmark has 6,528 stage-2 video tokens and an `Nv²` term. Trace
   the already-fused SDPA separately from the four large linears and gate. A
   custom attention kernel is not the starting assumption because MLX SDPA is
   already used.
2. **Stage-2 video FF and AdaLN/residual graph.** The 4096→16384→4096 FF runs
   48 times per forward. Determine whether time is in `addmm`, GELU, memory
   traffic, or command-buffer boundaries. For the GEMMs, record whether MLX
   0.31.1 routes the exact LTX token/4096/16384 shapes through its M5-tuned
   path, which tile it selects, and whether split-K is selected or would help;
   PR #3211's 4096/14336 results do not answer those LTX-shape questions.
3. **DiT eval cadence and queue gaps.** Compare the default six command buffers
    per forward with one carefully varied cadence on the M5 Max. The acceptance
    test must include macOS contention and watchdog-free marathons.
    The isolated cadence-16 matched probe found no material end-to-end gain and
    retained the watchdog-safe default of 8; see
    [AI_STUDIO_MLX_EVAL_CADENCE_PROBE_2026-07-26.md](AI_STUDIO_MLX_EVAL_CADENCE_PROBE_2026-07-26.md).
4. **Stage-1 DiT.** It has eight forwards and can dominate nonquadratic work
   despite fewer tokens. Use identical attention-branch labels to compare it
   with stage 2.
5. **Cold prompt path.** Separate 4-bit Gemma, the two 188160-wide projection
   matmuls, and the 16 manual connector-attention blocks. If connector score
   tensors are material, test `mx.fast.scaled_dot_product_attention` parity.
6. **Video VAE decoder peak Conv3d stages.** Capture per-upsample-stage time and
   allocation high-water marks. At larger shapes, separately capture fp32 tile
   accumulation; it is not active in the ordinary default.
7. **1024-channel latent upsampler.** Split Conv3d residual work, per-frame
   Conv2d, GroupNorm, and pixel-shuffle lowering. Test compilation only after
   identifying stable shapes and compile amortization.
8. **Vocoder/BWE.** Measure the fp32 ConvTranspose1d/Conv1d chain, anti-aliased
   activations, and BWE generator. It is long but runs only once.
9. **CPU/media boundaries.** Time the duplicated CRF-33 source preprocessing,
   complete waveform copy/WAV write, per-frame `bytes` allocation, and ffmpeg.
   These may dominate startup/finalization without appearing in a Metal trace.
10. **RoPE, patchify, and transpose materializations.** Low priority unless a
    trace shows repeated kernels or copies. RoPE application is already
    compiled and patchifiers contain no arithmetic-heavy loop.

## Candidate experiments after profiling

These are also hypotheses:

- capture and microbenchmark the exact LTX `addmm` shapes to verify M5
  routing, tiling, and split-K behavior before considering a custom GEMM;
- cache the four stable RoPE frequency tensors across denoise steps within each
  stage;
- pack Q/K/V projections only if three separate `addmm` calls are a measured
  bottleneck and weight-layout parity is maintainable;
- compile an eager `BasicAVTransformerBlock` per stable shape bucket while
  retaining watchdog eval boundaries;
- replace connector manual attention with fused MLX SDPA after mask and output
  parity tests;
- stream/cache the two deterministic source-image preprocessing products
  within one request;
- remove the `bytes(memoryview(...))` copy if direct buffer writes are safe
  after the required frame eval;
- stream audio directly to the muxer instead of constructing NumPy int16 and a
  temporary WAV; and
- use BF16 VAE tile accumulation only for tiled requests, after blend-quality
  and memory tests.

The main DiT's SDPA, RMSNorm, LayerNorm, biased linears, and RoPE already use
fused or compiled MLX primitives. A custom Metal program should therefore be
authorized only when a capture identifies a specific remaining operation,
shape, and memory behavior that MLX 0.31.1 does not handle well. The first
capture path is MLX's library-scoped API over its inspectable kernels, not
reverse-engineering Apple's private driver/compiler binaries and not repeating
the intrusive full-system trace.
