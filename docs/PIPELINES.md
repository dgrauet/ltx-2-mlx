# Pipelines guide — which pipeline, which flags

Current as of **v0.15.8**. Architecture and internals: [CLAUDE.md](../CLAUDE.md).
Stability tiers: [PIPELINE_MATURITY.md](PIPELINE_MATURITY.md).
User-facing overview: [README.md](../README.md).

Resolutions are written width × height in prose. On the command line the parser
takes `--height, -H` first and `--width, -W` second.

## Which pipeline?

Start from what you have:

- **Text only** → `generate --distilled` (fastest, 2.5 packs recommended) · `generate --two-stage` (dev + CFG, best default quality) · `generate --two-stages-hq` (res_2s sampler, slower) · `generate --one-stage` (native resolution ≤ 704 × 480, no upsampler).
- **Text + one or more images** → the same four `generate` modes with `--image PATH FRAME STRENGTH` (repeatable).
- **Two images to interpolate between** → `keyframe`.
- **A control video** (depth / canny / pose / motion tracks) → `ic-lora`; **HDR output** → `hdr-ic-lora`.
- **An audio track to drive the video** → `a2v`.
- **An existing video to change** → `retake` (a time range) · `extend` (add frames) · `lipdub` (re-sync lips to audio, experimental).
- **A prompt to improve first** → the `enhance` subcommand.

Then pick by constraint: 16–32 GB machines add `--low-ram`; 1080p or clips over 8 s
add `--tile-spatial 2`; a sharper decode on 2.5 packs adds `--video-decoder diffusion`;
**maximum detail on 2.5 packs** → `generate` `--dfr` (experimental).

`generate` has no implicit mode. One of `--one-stage`, `--two-stage`, `--two-stages-hq`,
`--distilled` or `--dfr` is mandatory, because each maps 1:1 to an upstream pipeline class.

## Pipelines

One card per subcommand or mode. "Own flags" are the flags specific to that card.
Everything else is in [Common flags](#common-flags).

### `generate --distilled`

- **Produces:** T2V / I2V mp4 with audio. Half-res distilled pass, 2× latent upsample, 3-step distilled refine.
- **Packs:** 2.3 and 2.5. On 2.5 stage 1 uses the ancestral sampler. **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`. `--frames` is required on 2.3 packs and auto-predicted on 2.5.
- **Own flags:** `--stage1-steps` (8), `--stage2-steps` (3). No CFG, so `--cfg-scale`, `--stg-scale` and TeaCache do not apply.
- **Example:** `ltx-2-mlx generate --distilled -p "a fox in the forest" -H 512 -W 768 -f 49 --frame-rate 24 -o fox.mp4`
- **Notes:** fastest mode. Quality sits slightly below the dev + CFG variants.

### `generate --two-stage`

- **Produces:** T2V / I2V mp4 with audio. Dev model + CFG at half resolution, 2× upsample, distilled 3-step refine.
- **Packs:** 2.3 and 2.5. **Tier:** Stable. Best default quality per minute.
- **Required:** `--prompt`, `--output`, `--frame-rate`; `--frames` on 2.3 packs.
- **Own flags:** `--stage1-steps` (30), `--stage2-steps` (3), `--cfg-scale` (3.0), `--stg-scale` (1.0; each unit above 0 adds one extra forward pass per step — pass `--stg-scale 0` on 32 GB Macs for long clips), `--dev-transformer` (`transformer-dev.safetensors`), `--distilled-lora`, `--distilled-lora-strength` (1.0), `--enable-teacache`, `--teacache-thresh`.
- **Example:** `ltx-2-mlx generate --two-stage -p "a fox in the forest" -H 480 -W 704 -f 97 --frame-rate 24 --low-ram -o fox.mp4`
- **Cost (M2 Pro 32 GB, q8):** 704 × 480, 97 frames ≈ 1374 s, or ≈ 942 s with `--enable-teacache`. [Details](../CLAUDE.md#teacache-opt-in-stage-1-acceleration).

### `generate --two-stages-hq`

- **Produces:** the same two-stage output with the second-order res_2s sampler in stage 1.
- **Packs:** 2.3 and 2.5. **Tier:** Stable. Roughly twice the stage-1 cost of `--two-stage`.
- **Required:** identical to `--two-stage`.
- **Own flags:** identical to `--two-stage`, except `--stage1-steps` defaults to 15 and `--stg-scale` to 0.0. Each step runs two model evaluations.
- **Example:** `ltx-2-mlx generate --two-stages-hq -p "a fox in the forest" -H 480 -W 704 -f 97 --frame-rate 24 --low-ram -o fox.mp4`
- **Cost (M2 Pro 32 GB, `--low-ram`):** 704 × 480, 97 frames ≈ 44 min at q8, ≈ 50 min at bf16. TeaCache at `--teacache-thresh 1.0` cuts a 576 × 384, 65-frame run from 1370 s to 768 s.

### `generate --one-stage`

- **Produces:** T2V / I2V mp4 with audio in a single dev + CFG pass at the target resolution. No upsampler, no stage 2.
- **Packs:** 2.3 and 2.5. **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`; `--frames` on 2.3 packs.
- **Own flags:** `--steps` (30), `--cfg-scale` (3.0), `--stg-scale` (1.0), `--dev-transformer`. No stage-2 or TeaCache flags.
- **Example:** `ltx-2-mlx generate --one-stage -p "a fox in the forest" -H 480 -W 704 -f 33 --frame-rate 24 --low-ram -o fox.mp4`
- **Cost (M2 Pro 32 GB, q8, `--low-ram`):** 704 × 480, 33 frames ≈ 2 min 31 s.
- **Notes:** pick it for native resolutions up to 704 × 480, or when you would rather not depend on the neural upsampler. `--two-stage` is faster at larger targets.

### `generate` `--dfr` *(experimental, LTX 2.5 packs only)*

- **Produces:** T2V / I2V mp4 (+ audio) — the DFR base path: half-res distilled stage with keyframe slots on a segment-aligned canvas, 2× latent upsample, then a full-res detailing stage with the official detailing IC-LoRA (`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`, strength 0.5) guided by the stage-1 latent. With `--video-decoder diffusion` the stage-2 keyframe slots are decoded as a keyframe-aware second stream (`--video-decoder conv`, the default, ignores them with a warning). `--temporal-upscalings T` adds T temporal x2 refine rounds on top: `(N-1)*2**T+1` frames at `frame_rate*2**T` fps, audio carried over unchanged from stage 1 (frozen, not re-denoised). The spatial epilogue is not ported yet.
- **Packs:** 2.5 only. **Tier:** Experimental.
- **Required:** `--prompt`, `--output`, `--frame-rate` (`-f` optional: auto-predicted).
- **Own flags:** `--detailing-lora PATH_OR_REPO` (official LoRA, downloaded on first use — **gated repo**: accept the licence once at [huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) with the account `huggingface-cli login` uses, or the run stops before any model load; a local `.safetensors` path skips the download), `--stage1-steps` (8), `--stage2-steps` (3), `--image PATH FRAME STRENGTH` (repeatable), `--temporal-upscalings {0,1,2}` (0), `--temporal-upsampler-path PATH` (default: the pack's `temporal_upscaler_x2_v1_0.safetensors`). Not accepted: `--num-generated-keyframes` (slots come from the canvas), `--enable-teacache`, `--cfg-scale`, `--stg-scale`; with `--temporal-upscalings` set, also not accepted: `--segment` (Prompt Relay), `--tile-frames` / `--tile-spatial`.
- **Example:** `ltx-2-mlx generate --dfr --model /path/to/ltx-2.5-mlx-q8 -p "a fox in the forest" -H 512 -W 768 -f 49 --frame-rate 24 --low-ram -o fox.mp4`
- **Cost (M2 Pro 32 GB, q8, `--low-ram`):** 768 × 512, 49 frames ≈ 277 s (8-step stage 1 at ~10.7 s/forward over 864 video tokens, 3-step stage 2 at ~53.5 s/forward over 4128 tokens — target + 2 keyframe slots + the half-res reference), peak Metal 14.0 GB, max RSS 10.8 GB. Frame 24 is visibly sharper (tree crowns, haze texture) than the plain `--distilled` render at the same seed. `--image` (I2V, frame 0 anchor) adds ~7 s. A 137-frame request pads to a 145-frame canvas (6 slots) and costs ≈ 783 s. `--video-decoder diffusion` at 1152 × 768, 25 frames runs untiled at ≈ 465 s, 12 GB peak Metal (the decoder now runs in its own bf16 whatever dtype the pipeline hands it; the first measurement, 22 GB, was the decode promoted to fp32 by an fp32 stage-2 latent). The keyframe-aware diffusion decode costs +58 % on the decode phase at 768 × 512 × 49 (157 s vs 100 s, 2 planes), same peak Metal. Temporal rounds at 768 × 512, 121 frames: `--temporal-upscalings 1` (241 frames @ 48 fps) ≈ 29 min, `--temporal-upscalings 2` (481 @ 96) ≈ 70 min; each 145-frame round tile costs ~9–10 min (4 ancestral steps).
- **Notes:** on I2V, the first-frame keyframe marker is now applied consistently (see [Details](../CLAUDE.md#dfr-base-path-generate---dfr-25-packs-experimental)). `--segment` (Prompt Relay) auto-distributes over the padded canvas, not the requested duration (e.g. 145 frames for `-f 137`), so segment boundaries shift by the padding before the tail is trimmed — pass explicit segment lengths for exact boundaries. [Details](../CLAUDE.md#dfr-base-path-generate---dfr-25-packs-experimental).

### `keyframe`

- **Produces:** an interpolation between a start image and an end image. Dev model + CFG at half res, then a distilled refine.
- **Packs:** 2.3 and 2.5 (2.5 needs `--dev-transformer transformer-dev.safetensors`). **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`, `--start`, `--end` (image paths).
- **Own flags:** `--frames` (97), `--start-strength` (1.0), `--end-strength` (1.0), `--stage1-steps`, `--stage2-steps`, `--cfg-scale` (3.0 video, 7.0 audio), `--stg-scale` (1.0), `--dev-transformer`, `--distilled-lora`, `--lora-strength` (1.0).
- **Example:** `ltx-2-mlx keyframe -p "the camera pushes in" --start a.jpg --end b.jpg -f 97 --frame-rate 24 -o out.mp4`
- **Notes:** the distilled model hallucinates on interpolation, so this pipeline always uses the dev model. Lower `--end-strength` gives the prompt more freedom to diverge from the end fixture. [Details](../CLAUDE.md#keyframe-interpolation-pipeline).

### `ic-lora`

- **Produces:** control-conditioned video from a control clip (canny, depth, pose, motion tracks) plus an official Lightricks IC-LoRA.
- **Packs:** 2.3 only. No official 2.5 task IC-LoRAs exist yet. **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`, `--lora PATH STRENGTH`, `--video-conditioning PATH STRENGTH`.
- **Own flags:** `--frames` (97), `--conditioning-strength` (1.0), `--skip-stage-2`, `--upsample-only`, `--refine-steps`, `--single-stage`, `--stage1-steps`, `--stage2-steps`, `--dev-transformer`, `--distilled-lora`, `--distilled-lora-strength` (0.5 in dev mode), `--image`.
- **Example:**
  ```
  ltx-2-mlx ic-lora -p "a person walking" \
    --lora Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control 1.0 \
    --video-conditioning depth.mp4 1.0 \
    -H 704 -W 1280 -f 97 --frame-rate 24 --low-ram -o out.mp4
  ```
- **Cost (M2 Pro 32 GB, q8, `--low-ram`):** 1280 × 704, 97 frames ≈ 15 min.
- **Topologies:** the default is half-res generation then a control-blind stage-2 refine. `--single-stage` generates at full resolution in one pass and tracks the control most tightly. `--upsample-only` skips the refine for a fast draft, and adding `--refine-steps N` runs a control-aware refine instead. `--skip-stage-2` stops at half resolution.
- **Notes:** `--dev-transformer` switches on dev mode, which fuses the distilled LoRA alongside the task LoRA. This is also the recipe that preserves a source image's identity across a whole clip. [Details](../CLAUDE.md#ic-lora-pipeline) · [static-scene recipe](../CLAUDE.md#static-scene-i2v-recipe-preserve-identity).

### `hdr-ic-lora`

- **Produces:** two files. An SDR mp4 preview at `--output`, and a `.hdr.npz` float32 linear-HDR tensor next to it.
- **Packs:** 2.3 only. **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`, `--lora PATH STRENGTH` (an HDR IC-LoRA).
- **Own flags:** same as `ic-lora`, except `--video-conditioning` is optional. Omit it for pure text-to-HDR. `--frames` (97), `--conditioning-strength`, `--skip-stage-2`, `--stage1-steps`, `--stage2-steps`, `--image`.
- **Example:**
  ```
  ltx-2-mlx hdr-ic-lora -p "cinematic golden hour" \
    --lora Lightricks/LTX-2.3-22b-IC-LoRA-HDR 1.0 \
    --video-conditioning source_sdr.mp4 1.0 \
    -f 97 --frame-rate 24 --low-ram -o out.mp4
  ```
- **Cost (M2 Pro 32 GB, `--low-ram`):** 1280 × 704, 97 frames ≈ 15 min 28 s.
- **Notes:** the HDR transform and the reference downscale factor are read from the LoRA's safetensors metadata. Converting the npz to EXR or TIFF is left to your own tooling. [Details](../CLAUDE.md#hdr-ic-lora-pipeline).

### `a2v`

- **Produces:** video driven by an existing audio track, optionally anchored on an image. Dev model + CFG, two stages.
- **Packs:** 2.3 and 2.5. **Tier:** Beta.
- **Required:** `--prompt`, `--output`, `--frame-rate`, `--audio`.
- **Own flags:** `--frames` (97), `--audio-start` (0 s), `--stage1-steps` (30), `--stage2-steps` (3), `--cfg-scale` (3.0), `--stg-scale` (1.0), `--image`.
- **Example:** `ltx-2-mlx a2v -p "a singer performing" --audio music.wav -i photo.jpg -f 97 --frame-rate 24 -o out.mp4`
- **Notes:** sync quality depends on how well the prompt matches the audio. Audio CFG runs at 7.0.

### `retake`

- **Produces:** the source video with one latent-frame range regenerated. Dev model + CFG, single stage.
- **Packs:** 2.3 and 2.5. **Tier:** Beta.
- **Required:** `--prompt`, `--output`, `--video`, `--start`, `--end` (latent frame indices, end exclusive).
- **Own flags:** `--steps` (30), `--cfg-scale` (3.0), `--stg-scale` (1.0), `--no-regen-audio`. Resolution and frame count come from the source clip, so `--height`, `--width`, `--frames` and `--frame-rate` are absent.
- **Example:** `ltx-2-mlx retake -p "a different action" -v source.mp4 --start 2 --end 5 --low-ram -o retake.mp4`
- **Cost:** follows the **total** clip length, not the size of the regenerated window. Preserved frames are still computed and attended over on every pass.

### `extend`

- **Produces:** the source video with latent frames appended before or after it. Same pipeline class as `retake`.
- **Packs:** 2.3 and 2.5. **Tier:** Beta.
- **Required:** `--prompt`, `--output`, `--video`, `--extend-frames`.
- **Own flags:** `--direction` (`after`), `--steps` (30), `--cfg-scale` (3.0), `--stg-scale` (1.0).
- **Example:** `ltx-2-mlx extend -p "continue the scene" -v source.mp4 --extend-frames 4 --low-ram -o extended.mp4`
- **Cost:** same rule as `retake` — the whole clip is denoised each step.

### `lipdub`

- **Produces:** a reference clip re-synced so the lips follow its own audio track.
- **Packs:** 2.3 only. **Tier:** Experimental, on a pre-1.0 LipDub IC-LoRA.
- **Required:** `--prompt`, `--output`, `--reference-video`, `--lora PATH STRENGTH` (exactly one).
- **Own flags:** `--reference-strength` (1.0), `--stage1-steps`, `--stage2-steps`. The frame count and frame rate come from the reference video, so `--frames` and `--frame-rate` are absent.
- **Example:** `ltx-2-mlx lipdub -p "a person speaking" --reference-video clip.mp4 --lora <lipdub-lora> 1.0 -o out.mp4`
- **Notes:** the output audio is a VAE and vocoder reconstruction, audibly degraded on rich music. Remux the original audio when fidelity matters. `--low-ram` is accepted by the parser but not wired for this pipeline.

### Utilities

These subcommands do not generate video and have no column in the matrix below.

- **`enhance`** rewrites a prompt with Gemma and prints it. Takes `--prompt`, `--gemma`, `--seed`, and `--mode`. Gemma 3 only, so it raises on 2.5 packs, which ship Gemma 4.
- **`info`** prints the configuration and memory estimate of a model directory. Takes `--model`.
- **`train`** trains a LoRA or a full model from a YAML config file. Takes a config path and `--low-ram`.
- **`preprocess`** encodes raw videos into latents and conditions for training. Takes a video directory, a caption directory and extension, an output directory, `--model`, `--gemma`, `--height`, `--width`, `--frame-rate`, a maximum frame count, and an audio switch.
- **`slice`** cuts long source videos into training clips. Takes an output directory plus clip-selection options (timecodes, interval, sampling, minimum length, maximum clip count, head and tail trims, resolution, fit mode, frame rate, encoder quality, caption template).

Run `ltx-2-mlx <subcommand> --help` for the exact spelling of these options.

## Common flags

### Inputs and outputs

| Flag | Default | Effect | Applies to |
|---|---|---|---|
| `--prompt`, `-p` | required | Text prompt. | all |
| `--output`, `-o` | required | Output video path (`.mp4`). | all |
| `--model`, `-m` | `dgrauet/ltx-2.3-mlx-q8` | Weights, as a HuggingFace repo id or a local pack directory. 2.5 support is auto-detected from the pack. | all |
| `--gemma` | `mlx-community/gemma-3-12b-it-4bit` | Text encoder for 2.3 packs. 2.5 packs carry their own Gemma 4 tower and ignore it. | all |
| `--seed`, `-s` | -1 (random) | Random seed. Pass a fixed value for reproducible runs. | all |
| `--quiet`, `-q` | off | Suppress the progress output described below. | all |
| `--height`, `-H` | 480 | Output height in pixels. Non-multiples of 64 round down on two-stage paths. | all except `retake` / `extend` |
| `--width`, `-W` | 704 | Output width in pixels. Same rounding rule. | all except `retake` / `extend` |
| `--frame-rate` | required | Output frame rate. LTX-2.3 was trained at 24; values far from that drift out of distribution. | all except `retake` / `extend` / `lipdub` |
| `--frames`, `-f` | 97, or auto on `generate` with a 2.5 pack | Frame count. Must satisfy `(frames - 1) % 8 == 0`. On `generate` with a 2.3 pack, omitting it fails immediately. | all except `retake` / `extend` / `lipdub` |
| `--auto-duration MIN:MAX` | 1:20 | Clamp, in seconds, for the duration predicted by the 2.5 DurationHead. Ignored with a warning when `--frames` is given. [Details](../CLAUDE.md#auto-duration-durationhead--f-optional-on-25). | `generate` on 2.5 packs |
| `--image`, `-i` | — | Reference image: `PATH [FRAME_IDX STRENGTH [CRF]]`. Repeatable, so you can anchor several pixel frames. Frame 0 replaces the first latent frame; later indices act as soft keyframes. On 2.5 packs, a frame-0 anchor now also carries the learned keyframe marker through, which shifts 2.5 I2V output slightly (2.3 unaffected). [Details](../CLAUDE.md#multi-anchor-i2v---image-repeatable). | `generate` modes, `ic-lora`, `hdr-ic-lora`, `a2v` |
| `--num-generated-keyframes N` | 0 | Add N generated keyframe slots at evenly spaced interior frames in stage 1, which relaxes the temporal compression where motion is fast. Each slot costs a latent frame of tokens. Refused up front on 2.3 packs. [Details](../CLAUDE.md#generated-keyframe-slots---num-generated-keyframes-n-25-packs) | `generate` modes on 2.5 packs |
| `--no-audio` | off | Skip the audio decode and mux. Video is unchanged; the DiT still produces audio latents jointly. | `generate` modes |
| `--video-decoder {conv,diffusion}` | `conv` | Video VAE decoder. `diffusion` is sharper and slower and needs a 2.5 pack. [Details](../CLAUDE.md#diffusion-video-decoder---video-decoder-diffusion-25-packs-experimental). | `generate` modes |
| `--diffvae-tile FRAMES HEIGHT WIDTH` | auto | Diffusion-decoder tile size, in pixel frames and pixels (multiples of 2 and 8; `0` leaves an axis untiled, `0 0 0` forces a single tile). | with `--video-decoder diffusion` |
| `--lora PATH STRENGTH` | — | Extra LoRA weights, repeatable. A local `.safetensors` file or a HuggingFace repo id. Required on the IC-LoRA family. | `generate` modes, `ic-lora`, `hdr-ic-lora`, `lipdub` |
| `--enhance-prompt` | off | Rewrite the prompt with Gemma before generating. Gemma 3 only, so it raises on 2.5 packs. | `generate` modes |
| `--dev-transformer` | `transformer-dev.safetensors` on `generate`, unset elsewhere | Filename of the dev (non-distilled) transformer inside the pack. On `keyframe` and `ic-lora` there is no default, and on `ic-lora` passing it switches dev mode on. | `generate` modes, `keyframe`, `ic-lora` |
| `--distilled-lora` | resolved from the pack; `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` on `ic-lora` | Filename of the distilled LoRA used by the refine stage. | `generate` modes, `keyframe`, `ic-lora` |
| `--distilled-lora-strength` | 1.0 (0.5 on `ic-lora` dev mode) | Strength of that LoRA. Any value other than 1.0 forces bind-time fusion under `--low-ram`. | `generate` modes, `ic-lora` |

### Memory

| Flag or variable | Default | Effect | Applies to |
|---|---|---|---|
| `--low-ram` | off | Stream transformer blocks from mmap'd safetensors. Cuts transformer peak Metal memory by about 75%, at roughly 5% more time per step. Targets 16 GB Macs at q8 and 32 GB Macs at bf16. [Details](../CLAUDE.md#block-streaming---low-ram). | every pipeline except `lipdub` |
| `--tile-frames N` | 1 | Split the video tokens into N temporal tiles, denoised independently and blended back. Caps the quadratic attention activation. [Details](../CLAUDE.md#modality-tiling---tile-frames-n---tile-spatial-m). | `generate` modes, `keyframe`, `ic-lora`, `hdr-ic-lora`, `a2v` |
| `--tile-spatial M` | 1 | Split into M × M spatial tiles. Total tiles are `tile-frames × M²`. | same as above |
| `--tile-overlap K` | 2 | Token-grid overlap between adjacent tiles. More overlap means a smoother blend and more redundant compute. | when tiling is active |
| `LTX2_VAE_DECODE_BUDGET_GB` | half of unified memory | Peak-memory budget for the VAE decode. Drives the conv decoder's automatic tiling and the diffusion decoder's tile sizing. Raise it on 64–256 GB machines for fewer, larger tiles. [Details](../CLAUDE.md#conv-vae-decode-budget-and-auto-tiling). | video decode |
| `LTX2_VAE_DECODE_KEEP_CACHE` | unset | Keep the MLX allocator cache during the decode. It is disabled by default for the decode only; pixels are identical either way and the process footprint is tens of GB lower on long clips. | video decode |

Tiling costs wall-clock time and only pays for itself when attention activations
would otherwise not fit. On a 32 GB Mac at typical token counts, prefer `--low-ram` alone.

### Speed

| Flag or variable | Default | Effect | Applies to |
|---|---|---|---|
| `--enable-teacache` | off | Timestep-aware residual caching in stage 1. About 1.46× on the Euler sampler and 1.78× on res_2s. [Details](../CLAUDE.md#teacache-opt-in-stage-1-acceleration). | `generate --two-stage`, `generate --two-stages-hq` |
| `--teacache-thresh F` | 0.5 Euler, 1.0 res_2s | How aggressively steps are skipped. Higher is faster and lossier. Ignored without `--enable-teacache`. | with `--enable-teacache` |
| `LTX2_GEMMA_EVAL_EVERY` | 1 | Per-layer flush cadence in the Gemma forward, which keeps each Metal command buffer under the macOS GPU watchdog deadline. Set to `0` only if you have never seen a watchdog crash. [Details](../CLAUDE.md#metal-watchdog-mitigation). | all pipelines |
| `LTX2_DIT_EVAL_EVERY` | 8 | Same guard for the DiT block loop: flush every N of the 48 blocks. `0` disables it. | all pipelines |
| `LTX2_GEMMA_MAX_LENGTH` | 1024 | Cap on the padded Gemma sequence length. Lowering it halves the encode time but shifts the text positions away from what the model was trained with (quality risk). Last resort. | all pipelines |
| `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` | unset | An AGX driver knob, not an LTX-2 variable, and never set automatically. It relaxes the watchdog eviction timeout for the process that sets it, working around a macOS 26.x and MLX 0.31.x regression. The UI may stutter during the run, and on some machines running with the display off is the only reliable workaround. | all pipelines |

### Previews and prompt gating

| Flag | Default | Effect | Applies to |
|---|---|---|---|
| `--stepwise-image-output-dir DIR` | off | Experimental. Every N steps, decode a short window of the in-progress prediction and write it to this directory as an animated WebP. One file per step, written once. It keeps the VAE decoder resident, which raises peak memory and works against `--low-ram`. | all generating pipelines |
| `--stepwise-interval N` | 1 | Preview every N denoising steps. The final step is always previewed. | with `--stepwise-image-output-dir` |
| `--stepwise-frames N` | 8 | Latent frames per preview. The VAE upsamples time 8×, so N latent frames give 8N−7 pixel frames, about 2.3 s at the default. Cost is independent of clip length. `1` gives a single still. | with `--stepwise-image-output-dir` |
| `--stepwise-frame I` | middle | Latent frame the preview window is centred on. Negative values count from the end. The middle is the default because frame 0 is the clean conditioning image on I2V runs. | with `--stepwise-image-output-dir` |
| `--segment "TEXT" [LEN]` | — | Prompt Relay: a local prompt gated to a slice of the timeline, repeatable in timeline order. The global `--prompt` still applies everywhere. Not compatible with modality tiling. [Details](../CLAUDE.md#prompt-relay---segment). | `generate` modes |
| `--relay-epsilon` | 1e-3 | Prompt Relay falloff. Smaller is sharper temporal gating. | with `--segment` |
| `--relay-strength` | 1.0 | Prompt Relay penalty multiplier. Higher isolates segments more strictly. | with `--segment` |

## Flags by pipeline

Columns are the cards above. `enhance`, `info`, `train`, `preprocess` and `slice`
are utilities and have no column.

| Flag | distilled | two-stage | hq | one-stage | dfr | keyframe | ic-lora | hdr-ic-lora | a2v | retake | extend | lipdub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--prompt` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--output` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--model` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--gemma` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--seed` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--quiet` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--height` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| `--width` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| `--frame-rate` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `--frames` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `--auto-duration` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--image` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `--num-generated-keyframes` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--no-audio` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--video-decoder` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--diffvae-tile` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--lora` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| `--enhance-prompt` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--dev-transformer` | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--distilled-lora` | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--distilled-lora-strength` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--detailing-lora` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--temporal-upscalings` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--temporal-upsampler-path` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--low-ram` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `--tile-frames` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `--tile-spatial` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `--tile-overlap` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `--enable-teacache` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--teacache-thresh` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--stepwise-image-output-dir` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--stepwise-interval` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--stepwise-frames` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--stepwise-frame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--segment` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--relay-epsilon` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--relay-strength` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

A ❌ means the flag is either rejected by the parser or accepted and inert for that
mode. The four `generate` modes share one parser, so a flag marked ❌ on `distilled`
or `one-stage` will still be accepted on the command line and then ignored.
`lipdub` accepts `--low-ram` but the streaming path is not wired for it.

## Progress output (stderr)

All CLI progress goes to **stderr** so stdout stays clean for callers that pipe it:

- `[phase] ...` / `[phase] done in X.Ys` brackets the silent stages (Gemma load, prompt encode, DiT load, decoders, decode). Silenced by `--quiet`.
- Each denoising stage prints `[estimate] <stage>: N steps x P passes over V video + A audio tokens = F forwards` **before** its first step (tqdm cannot say anything until iteration 1 completes, which at tens of seconds per step is exactly when a run looks hung), then `[estimate] <stage>: ~T remaining (S s/forward)` after the first computed step (tagged `first step includes warm-up` — kernel compilation and cache warm-up make it an upper bound) and once more after the second, timed on that step alone (tagged `refined`, the number to trust). Nothing after that. Passes per step come from the guider schedule (`cond` always, `uncond` under CFG, `ptb` under STG, `mod` under modality isolation; res_2s doubles them). Multi-stage pipelines print one pair per stage; there is no cross-stage total.
- `retake` / `extend`: the estimate carries `cost follows total clip length, not the regenerated window` — preserved frames are still computed and attended over on every pass, so retaking 1 latent frame of a 10 s clip costs the same as retaking all of it.

## Multishot on LTX-2.5

"Native multishot" is a capability of the 2.5 model, not a pipeline feature: write the shots in order in a single prompt, following [Lightricks' prompting guide](https://docs.ltx.video/open-source-model/usage-guides/prompting-guide). Nothing to enable on this runtime. If the model does not cut where you want, `--segment` (Prompt Relay) gates local prompts to time ranges on top of the global prompt.

## Compatibility notes

- `generate --lora <path>` (one-stage) is **incompatible with `--low-ram`** (LoRA pre-fuse happens before streaming setup). Use `ic-lora` or pre-fuse via mlx-forge.
- `--low-ram` + custom `--distilled-lora-strength` (≠1.0) on two-stage uses bind-time LoRA fusion (slower per step but supports any strength). At strength=1.0, swaps to pre-fused `transformer-distilled.safetensors`.
- TeaCache calibration is sampler-specific (Euler vs res_2s). Don't reuse coefficients across `--two-stage` and `--two-stages-hq`.
- HDR LoRA can be combined with regular IC-LoRA control LoRAs in theory but untested — single HDR LoRA per pipeline is the validated path.
- Modality tiling overhead dominates over memory benefit at default Nv (1650-3168). Use only when targeting 1080p / 8s+ on Mac Studio 64-128 GB; on 32 GB Mac, prefer `--low-ram` alone.
- `generate` requires a mode flag (`--one-stage`, `--two-stage`, `--two-stages-hq`, or `--distilled`). There is **no implicit default** — every pipeline maps 1:1 to an upstream Lightricks/LTX-2 class.
- `generate --one-stage` vs `generate --two-stage`: same dev model + CFG, but `--one-stage` runs **once at the target resolution** (no upscaler dependency, simpler latents for downstream). `--two-stage` runs at half-res then upscales 2× and refines (typically faster overall and better at large targets). Pick `--one-stage` for native res ≤ 704 × 480 or if you don't trust the upsampler; pick `--two-stage` for everything else.
- `generate --distilled` vs `generate --two-stage`: same half-res + upscale structure, but `--distilled` skips CFG entirely (8 stage 1 steps × 1 forward instead of 30 × 2-4). Fastest mode; quality slightly below the dev+CFG variants.
- `--video-decoder diffusion` (LTX 2.5 packs only, experimental) reproduces upstream's **default** `chunked_eager` stage-5 mode exactly: the neighborhood-attention stage runs on four width slabs with a halo, so the first/last ~20 px of each row are edge-replicated rather than attending over the full volume — identical to upstream's own default-mode output, not a port shortfall. Decodes above the memory budget are tiled automatically (`--diffvae-tile` to override); conv remains the default decoder.
