# Pipelines & options matrix

Reference for all CLI subcommands of `ltx-2-mlx`, the pipeline class
backing each, and which memory / performance flags apply where.
Current as of **v0.12.1**.

For the underlying architecture and conventions, see
[CLAUDE.md](../CLAUDE.md). For the high-level user-facing overview,
see [README.md](../README.md). For production-readiness classification
of each pipeline (Stable / Beta / Experimental) and stability guarantees
by tier, see [PIPELINE_MATURITY.md](PIPELINE_MATURITY.md).

## Core pipelines

| CLI | Pipeline class | Mode(s) | Sampler stage 1 | Sampler stage 2 | Default model | CFG | STG default |
|---|---|---|---|---|---|---|---|
| `generate --one-stage` | `TI2VidOneStagePipeline` | T2V / I2V | Euler + CFG (30 steps) at **full** resolution | — | q8 + dev LoRA | ✅ | 0.0 |
| `generate --two-stage` | `TI2VidTwoStagesPipeline` | T2V / I2V | Euler + CFG (30 steps) | Euler distilled (3 steps) | q8 + dev LoRA | ✅ | 0.0 |
| `generate --two-stages-hq` | `TI2VidTwoStagesHQPipeline` | T2V / I2V | res_2s + CFG (15 steps × 2 sub-steps) | Euler distilled (3) | q8 + dev LoRA | ✅ | 0.0 |
| `generate --distilled` | `DistilledPipeline` | T2V / I2V | Euler distilled (8 steps) at half-res | Euler distilled (3) at full-res | q8 (distilled only) | ❌ | — |
| `a2v` *(beta)* | `A2VidPipelineTwoStage` | A2V (+ optional I2V) | Euler + CFG (30) | Euler distilled (3) | q8 + dev LoRA | ✅ (audio cfg=7) | 0.0 |
| `keyframe` | `KeyframeInterpolationPipeline` | start frame ↔ end frame | Euler + CFG (30) | Euler distilled (3) | q8 + dev LoRA | ✅ | 0.0 |
| `ic-lora` | `ICLoraPipeline` | V2V (control video) + optional I2V | Euler distilled (8) | Euler distilled (3) | q8 + control LoRA | ❌ | — |
| `hdr-ic-lora` | `HDRICLoraPipeline(ICLoraPipeline)` | V2V / pure T2V / +I2V → linear HDR | Euler distilled (8) | Euler distilled (3) | q8 + HDR LoRA | ❌ | — |
| `lipdub` *(exp.)* | `LipDubPipeline(ICLoraPipeline)` | V2V + ref audio → lip-dubbed | Euler distilled (8) | Euler distilled (3) | q8 + LipDub LoRA | ❌ | — |
| `retake` *(beta)* | `RetakePipeline.retake_from_video` | regenerate latent frame range | Euler dev + CFG (30) | — | dev | ✅ | 0.0 |
| `extend` *(beta)* | `RetakePipeline.extend_from_video` (same class) | append frames before/after | Euler dev + CFG (30) | — | dev | ✅ | 0.0 |
| `enhance` | Gemma rewrite | prompt → enriched prompt | — | — | Gemma 3 12B | — | — |
| `info` / `train` / `preprocess` | — | utilities | — | — | — | — | — |

## Memory / perf opt-ins (cross-pipeline)

| Flag / env var | Default | Effect | Supported pipelines |
|---|---|---|---|
| `--low-ram` | off | Block streaming: stream DiT layers from mmap'd safetensors. Peak ≈ 1 block + Gemma. ~75% transformer RAM cut. | `generate` (one-stage / `--two-stage` / `--two-stages-hq`), `a2v`, `keyframe`, `ic-lora`, `hdr-ic-lora` |
| `--tile-frames N` | 1 | Split video tokens into N temporal tiles. Caps O(N²) attention activations. | `generate` (all variants), `a2v`, `keyframe` |
| `--tile-spatial M` | 1 | Split video tokens into M×M spatial tiles. Total tiles = `tile-frames × M²`. | same as above |
| `--tile-overlap K` | 2 | Token-grid overlap (smoother blend at cost of redundant compute). | when tiling active |
| `--enable-teacache` | off | Timestep-aware residual caching. ~1.46× speedup (Euler) / ~1.78× (HQ). Conservative thresh 0.5. | `generate --two-stage`, `generate --two-stages-hq` |
| `--teacache-thresh F` | 0.5 (Euler) / 1.0 (HQ) | Skip aggressiveness. Higher = more skip = faster but quality risk. | with `--enable-teacache` |
| `LTX2_GEMMA_EVAL_EVERY=N` | 1 (per-layer eval) | Per-layer `mx.eval` cadence in Gemma forward. Keeps each Metal command buffer below the macOS GPU watchdog (~10 s) deadline. Default `1` on all Apple Silicon since PR #3 (M2 Max 64 GB also crashes without it at production resolutions). Set to `0` on Mac Studio / M-series Ultra owners who never see the watchdog crash to recover lazy-graph throughput. | all pipelines (text encoding shared) |
| `LTX2_DIT_EVAL_EVERY=N` | 8 | Flush the DiT block loop every N blocks (splits 48 blocks into 6 command buffers, ~1-2 s each). Same trade-off as `LTX2_GEMMA_EVAL_EVERY`. Set to `0` to disable. | all pipelines (DiT forward shared) |
| `LTX2_GEMMA_MAX_LENGTH=N` | 1024 | Cap Gemma padded seq_len (last-resort escape hatch). Quality risk: shifts left-padded RoPE positions. | all pipelines |

## Pipeline-specific options

| Pipeline | Specific flags |
|---|---|
| `generate --one-stage` | `--frame-rate` (required), `--stage1-steps` aliased to `num_steps` (default 30), `--cfg-scale` (3.0), `--stg-scale` (0.0), `--image`. No stage2 / TeaCache / distilled-lora flags. Common: `--lora PATH STRENGTH` (incompatible with `--low-ram`), `--enhance-prompt`. |
| `generate --two-stage` | `--frame-rate` (required), `--stage1-steps` (30), `--stage2-steps` (3), `--cfg-scale` (3.0), `--stg-scale` (0.0), `--image`, `--distilled-lora-strength` (1.0), `--enable-teacache`, `--teacache-thresh` |
| `generate --two-stages-hq` | same as two-stage but stage1 default 15 steps, res_2s sampler |
| `generate --distilled` | `--frame-rate` (required), `--stage1-steps` (8 default), `--stage2-steps` (3 default), `--image`. No CFG/STG/TeaCache flags (distilled flow). Same DiT in both stages — no LoRA swap. |
| `a2v` | `--audio` (required), `--image`, `--audio-start`, `--frame-rate` (required), all two-stage flags |
| `keyframe` | `--start` / `--end` (image paths, required), `--frame-rate` (required), all two-stage flags |
| `ic-lora` | `--frame-rate` (required), `--lora PATH STRENGTH` (required, repeatable), `--video-conditioning PATH STRENGTH` (required, repeatable), `--conditioning-strength` (1.0), `--image`, `--skip-stage-2`, `--stage1-steps`, `--stage2-steps` |
| `hdr-ic-lora` | same as `ic-lora`, but `--video-conditioning` is **optional** (omit for pure T2V HDR). Auto-detects HDR transform from LoRA metadata. Outputs `.mp4` SDR + `.hdr.npz` linear HDR fp32 |
| `lipdub` *(experimental)* | `--reference-video PATH` (required, provides visuals + audio), `--lora PATH STRENGTH` (exactly one LipDub IC-LoRA, required), `--reference-strength` (1.0), `--stage1-steps` / `--stage2-steps`. Frame count auto-derived from the reference video metadata (snapped to `8k+1`). Output audio is VAE+vocoder reconstruction — remux original audio for music-fidelity use cases. |
| `retake` | `--video` (required), `--start` / `--end` (latent frame indices, required), `--steps` (30), `--no-regen-audio`, `--cfg-scale`, `--stg-scale` |
| `extend` | `--video` (required), `--extend-frames N` (required), `--direction before|after`, `--steps`, `--cfg-scale`, `--stg-scale` |

## Compatibility notes

- `generate --lora <path>` (one-stage) is **incompatible with `--low-ram`** (LoRA pre-fuse happens before streaming setup). Use `ic-lora` or pre-fuse via mlx-forge.
- `--low-ram` + custom `--distilled-lora-strength` (≠1.0) on two-stage uses bind-time LoRA fusion (slower per step but supports any strength). At strength=1.0, swaps to pre-fused `transformer-distilled.safetensors`.
- TeaCache calibration is sampler-specific (Euler vs res_2s). Don't reuse coefficients across `--two-stage` and `--two-stages-hq`.
- HDR LoRA can be combined with regular IC-LoRA control LoRAs in theory but untested — single HDR LoRA per pipeline is the validated path.
- Modality tiling overhead dominates over memory benefit at default Nv (1650-3168). Use only when targeting 1080p / 8s+ on Mac Studio 64-128 GB; on 32 GB Mac, prefer `--low-ram` alone.
- `generate` requires a mode flag (`--one-stage`, `--two-stage`, `--two-stages-hq`, or `--distilled`). There is **no implicit default** — every pipeline maps 1:1 to an upstream Lightricks/LTX-2 class.
- `generate --one-stage` vs `generate --two-stage`: same dev model + CFG, but `--one-stage` runs **once at the target resolution** (no upscaler dependency, simpler latents for downstream). `--two-stage` runs at half-res then upscales 2× and refines (typically faster overall and better at large targets). Pick `--one-stage` for native res ≤ 480×704 or if you don't trust the upsampler; pick `--two-stage` for everything else.
- `generate --distilled` vs `generate --two-stage`: same half-res + upscale structure, but `--distilled` skips CFG entirely (8 stage 1 steps × 1 forward instead of 30 × 2-4). Fastest mode; quality slightly below the dev+CFG variants.
