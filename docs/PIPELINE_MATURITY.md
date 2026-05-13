# Pipeline maturity tiers

This document classifies each `ltx-2-mlx` pipeline by stability and production
readiness. Downstream consumers should read this before relying on a pipeline
in their app.

## Tiers

### 🟢 Stable

Bit-exact upstream-iso math, validated through multiple regression cohorts,
zero known quality regressions. Safe to rely on in production.

| Pipeline | CLI subcommand | Notes |
|---|---|---|
| `TI2VidOneStagePipeline` | `generate --one-stage` | Dev one-stage at target res |
| `TI2VidTwoStagesPipeline` | `generate --two-stage` | Dev + CFG + 2x upscale + distilled refine |
| `TI2VidTwoStagesHQPipeline` | `generate --two-stages-hq` | Same as `--two-stage` with res_2s sampler |
| `DistilledPipeline` | `generate --distilled` | Distilled-only two-stage; fastest |
| `KeyframeInterpolationPipeline` | `keyframe` | Two-stage interp dev + CFG |
| `ICLoraPipeline` | `ic-lora` | IC-LoRA reference video conditioning |
| `HDRICLoraPipeline` | `hdr-ic-lora` | Linear HDR via LogC3 inverse |

### 🟡 Beta

Works as designed, fully ported upstream-iso, but the underlying model
behaviour has visible quality limitations on some inputs. Functional, but
quality consistency depends on input alignment.

| Pipeline | CLI subcommand | Known limitations |
|---|---|---|
| `A2VidPipelineTwoStage` | `a2v` | Audio-to-video sync varies with prompt specificity; generic prompts produce visually plausible but loosely-synced output |
| `RetakePipeline` | `retake` | Regenerated segment doesn't always blend cleanly with source video at boundaries |
| `RetakePipeline` | `extend` | Appended frames inherit some seed-of-the-extension noise; quality varies |

### 🔴 Experimental

Recently ported, limited validation, or known model-level quality
limitations. The pipeline itself runs correctly (math is upstream-iso), but
the **output quality** depends on a third-party LoRA that may itself be
pre-1.0 or have known artifacts.

Pin a specific version if you depend on the current behaviour — semantic
backwards compatibility is best-effort, not guaranteed, on this tier.

| Pipeline | CLI subcommand | Status & limitations |
|---|---|---|
| `LipDubPipeline` | `lipdub` | Lip-dub uses `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` (currently v0.9). Audio output is **VAE+vocoder reconstruction** of the reference audio — perceptually similar but not bit-identical (spectral artifacts visible on rich musical content). Lip-sync quality depends on prompt-audio alignment. Workaround for music: remux original audio over the output mp4 via ffmpeg (loses fine lip-sync but preserves source music). |

## Stability guarantees by tier

| Tier | API stability | Math iso vs upstream | Output quality consistency |
|---|---|---|---|
| Stable | Breaking only on `0.y` bumps; deprecation cycle when possible | Bit-exact via regression cohort | Validated; no known quality regressions |
| Beta | Breaking only on `0.y` bumps | Bit-exact | Quality consistency depends on input alignment; expect occasional surprises |
| Experimental | API may shift on `0.z` bumps; pin if depending on it | Bit-exact | LoRA / model-level limitations propagate to output |

## Promotion criteria

- **Experimental → Beta**: pipeline has run on ≥3 distinct test scenarios
  including production-length (≥4 s) outputs; no port-level regressions; LoRA
  released at ≥1.0 OR limitations explicitly documented and tolerable.
- **Beta → Stable**: pipeline has been used by a downstream consumer for ≥1
  release cycle without quality complaints, all known limitations resolved
  or formally accepted as out-of-scope.

## How to read this in code

CLI `--help` output marks Beta pipelines with `[beta]` and Experimental ones
with `[experimental]` in the help text. Pipeline classes don't carry the
classification in code — it lives here in the doc, so a tier change doesn't
churn module-level metadata.
