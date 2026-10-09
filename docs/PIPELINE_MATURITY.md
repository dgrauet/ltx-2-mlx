# Pipeline maturity tiers

This document classifies each `ltx-2-mlx` pipeline by stability and production
readiness. Downstream consumers should read this before relying on a pipeline
in their app. For per-pipeline usage (flags, defaults, when to reach for
each one), see the cards in [docs/PIPELINES.md § Pipelines](PIPELINES.md#pipelines).

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
limitations. Some entries run correctly (math is upstream-iso) but their
**output quality** depends on a third-party LoRA that may itself be pre-1.0
or have known artifacts; others are new local paths (the distilled `a2v` /
`retake` / `extend` modes, NAG) validated on a few scenarios only.

Pin a specific version if you depend on the current behaviour — semantic
backwards compatibility is best-effort, not guaranteed, on this tier.

| Pipeline | CLI subcommand | Status & limitations |
|---|---|---|
| `A2VidDistilledPipeline` | `a2v --distilled` | Lightricks' distilled A2V workflow (input audio frozen in both stages) on the `generate --distilled` path, no dev model. Validated end to end on 2.5 q8 (M1 Max): 512×768×49 with and without `--low-ram`, 704×1280×121, and 704×1280×241 with three image anchors; SyncNet LSE-C 4.6 on a front-facing test clip. |
| `ExtendDistilledPipeline` | `extend --distilled` | Upstream's chunk continuation (the previous window's last 25 frames pinned at index 0 in both distilled stages) with the source clip as the previous window. Validated end to end on 2.5 q8 (M1 Max): 704×1280, 10 s + 4 s, float16, resident, `--low-ram` (same bytes) and with `LTX2_SOL_TAU`; a 5 s source; 512×768×49 in the default environment. The model sees only the last 25 source frames: a face hidden or turned away there can come back as a different face; an `--image` anchor on the new frames (a front-facing frame of the character) brings it back. |
| `RetakePipeline(distilled=True)` | `retake --distilled` | Upstream's default retake mode: distilled transformer, 8-step distilled table, no CFG, deterministic Euler. Validated end to end on 2.5 q8 (M1 Max): 512×768×49 with and without `--low-ram`, float16 and `--no-regen-audio`, and 704×1280×121 float16 with and without `--low-ram`. Same blending limits as `retake` at the window edges. |
| `HDRICLoraPipeline` | `hdr-ic-lora` | Upstream v1.4 single-stage ACEScct SDR-to-HDR IC-LoRA, LTX-2.5 packs only (HLG mp4 + EXR frames). Gated LoRA (`Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR`). Validated end to end on real weights (2.5 q8, 768×512, 25-49 frames; see CLAUDE.md); promotion needs ≥3 scenarios including ≥4 s output (see promotion criteria). |
| `LipDubPipeline` | `lipdub` | Lip-dub uses `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` (currently v0.9). Audio output is **VAE+vocoder reconstruction** of the reference audio — perceptually similar but not bit-identical (spectral artifacts visible on rich musical content). Lip-sync quality depends on prompt-audio alignment. Workaround for music: remux original audio over the output mp4 via ffmpeg (loses fine lip-sync but preserves source music). |
| Diffusion video decoder | `generate --video-decoder diffusion` (LTX 2.5 packs only) | Opt-in alternative to the default conv VAE decoder; sharper output, several times slower. Tiled automatically above the decode budget (upstream schedule, 40-frame / 160-px overlaps); validated up to 768×1152×49 tiled and 512×768×97 untiled on an M2 Pro 32 GB (see CLAUDE.md for timings). Still Experimental until a downstream consumer validates it. Reproduces upstream's default `chunked_eager` mode, including its edge-replicated borders. |
| `DFRPipeline` | `generate --dfr` (LTX 2.5 packs only) | DFR complete: base path (spatial ×2 detailing with the official detailing IC-LoRA), temporal rounds (`--temporal-upscalings {1,2}`, ×2/×4 frames+fps), spatial epilogue (`--spatial-upscalings 2`, H/4 → H/2 → full-res detailing), and keyframe-aware decode on `--video-decoder diffusion`. Validated e2e at 512×768×49 T2V/I2V and 137 frames (canvas padding). |
| NAG (Normalized Attention Guidance) | `generate --distilled --nag`, `generate --dfr --nag` | Negative prompt on the CFG-less distilled paths, applied inside the text cross-attentions (kijai's `LTX2_NAG` node, defaults scale 11 / alpha 0.25 / tau 2.5). Opt-in; without `--nag` renders are byte-identical. Validated e2e on 2.5 q8 (`--distilled` 768×512×97, `--dfr` 768×512×49): reduces the wide-open mouth it is told to avoid on 4/4 runs, speech unchanged; did not fix motion-blurred hands. Cost +14 % per stage-1 step and +7 % per stage-2 step at 768×512 (falls with resolution), peak memory unchanged. |

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
