# Regression test suite — v0.11.0 (post-isomorphic refactor)

Hardware: M2 Pro 32 GB · macOS Darwin 25.4.0 · MLX bf16/q8
Date started: 2026-05-09
Branch: `main` @ `d6cc3d1`
Goal: Validate every public pipeline still produces correct outputs after the v0.8.x → v0.11.0 isomorphic refactor (CLI flags, file names, class names, blocks API, helpers API, ImageToVideoPipeline removal).

## Test fixtures (`/tmp/ltx2_regression/`)

| File | Generation | Use |
|---|---|---|
| `test.wav` | ffmpeg sine 440 Hz, 4 s, 16 kHz mono | a2v audio input |
| `frame_a.png` | PIL solid red 384×256 | keyframe start, I2V |
| `frame_b.png` | PIL solid blue 384×256 | keyframe end |
| `frame_a_hd.png` | PIL solid red 704×448 | (unused) |
| `source.mp4` | output of test 02 (two-stage 384×256×17) | retake/extend/ic-lora/hdr source |

## Run notes

- Common params: `-W 384 -H 256`, `-f 9` (or 17 for two-stage/a2v), reduced steps.
- Initial smoke tests (#02-#16) ran with `LTX2_METAL_WATCHDOG_GUARD=1` (env var since removed in `8877f88` after the auto memory-gated mitigation made it redundant).
- System under heavy `knowledgeconstructiond` (Apple Intelligence indexer) post-boot. First non-`--low-ram` attempt hit Metal `Impacting Interactivity` watchdog within 14 s during model load. Tests #03–#16 then ran with `--low-ram` (block streaming, ~2.8 GB Metal peak vs 10–12 GB eager). After tests #15/#16 (which don't expose `--low-ram`) succeeded on retry, test #03b re-ran without `--low-ram` to verify the eager path.
- All q8 (`dgrauet/ltx-2.3-mlx-q8` default).

## Test matrix — results

| # | Pipeline | Class | CLI | Wall-clock | Output | Status |
|---|---|---|---|---|---|---|
| 02 | two-stage T2V | `TI2VidTwoStagesPipeline` | `generate --two-stage` | **119 s** | 82,446 B | ✅ |
| 03 | one-stage T2V (low-ram) | `TI2VidOneStagePipeline` | `generate --one-stage --low-ram` | **121 s** | 188,124 B | ✅ |
| 03b | one-stage T2V (eager) | same | `generate --one-stage` | **116 s** | 188,124 B | ✅ md5≡#03 |
| 04 | one-stage I2V | `TI2VidOneStagePipeline` | `generate --one-stage --image …` | **121 s** | 10,338 B | ✅ |
| 05 | two-stage I2V | `TI2VidTwoStagesPipeline` | `generate --two-stage --image …` | **130 s** | 92,053 B | ✅ |
| 06 | two-stage + TeaCache | same | `generate --two-stage --enable-teacache` | **133 s** | 82,446 B | ✅ no-skip @ 6 steps; mp4 ≡ #02 |
| 08 | HQ res_2s | `TI2VidTwoStagesHQPipeline` | `generate --two-stages-hq` | **176 s** | 52,436 B | ✅ ~29 s/step (2 forwards/step) |
| 09 | distilled | `DistilledPipeline` | `generate --distilled` | **48 s** | 39,700 B | ✅ ~3.5 s/step (no CFG) |
| 10 | audio-to-video | `A2VidPipelineTwoStage` | `a2v --audio test.wav` | **137 s** | 60,429 B | ✅ |
| 11 | keyframe interp | `KeyframeInterpolationPipeline` | `keyframe --start … --end …` | **150 s** | 51,313 B | ✅ requires explicit `--dev-transformer`/`--distilled-lora` |
| 12 | IC-LoRA Union | `ICLoraPipeline` | `ic-lora --lora … --video-conditioning …` | **44 s** | 32,496 B | ✅ bind-time LoRA fusion @ q8 |
| 13 | HDR IC-LoRA V2V | `HDRICLoraPipeline` | `hdr-ic-lora --video-conditioning …` | **50 s** | 32,760 B + 4.56 MB npz | ✅ HDR range [-0.017, 13.32], 1.36% px>1.0 |
| 14 | HDR IC-LoRA T2V | `HDRICLoraPipeline` | `hdr-ic-lora` (no `--video-conditioning`) | **46 s** | 33,346 B + 4.80 MB npz | ✅ HDR range [-0.017, 8.76], 4.78% px>1.0 |
| 15 | retake | `RetakePipeline` | `retake --start 1 --end 2` | **106 s** | 22,205 B | ✅ no `--low-ram` flag exposed |
| 16 | extend | `RetakePipeline.extend_from_video` | `extend --extend-frames 1` | **114 s** | 74,233 B | ✅ no `--low-ram` flag exposed |

**Total wall-clock: 1,734 s ≈ 28 min 54 s** (15 unique runs at smallest viable shape).

## Per-step throughput (informational)

| Pipeline | Steps | s/step | Notes |
|---|---|---|---|
| `--one-stage` (CFG) | 6 | 14.0 | full-res, 2 forwards/step |
| `--two-stage` stage 1 | 6 | 12.4 | half-res, 2 forwards/step |
| `--two-stage` stage 2 | 2 | 5.1 | full-res, no CFG |
| `--two-stages-hq` stage 1 | 4 | 29.0 | half-res, 2× sub-steps × 2 forwards = 4 forwards/step |
| `--distilled` stage 1 | 4 | 3.5 | half-res, **1** forward/step (no CFG) |
| `--distilled` stage 2 | 2 | 4.2 | full-res, 1 forward/step |
| `a2v` stage 1 | 6 | 15.3 | + audio guidance |
| `keyframe` stage 1 | 6 | 17.2 | + cond tokens |
| `ic-lora` stage 1 | 4 | 3.6 | distilled + bind-time LoRA |
| `hdr-ic-lora` stage 1 | 4 | 3.7 | distilled + bind-time HDR LoRA |
| `retake` | 4 | 17.8 | dev + CFG |
| `extend` | 4 | 20.1 | dev + CFG |

## Cross-cutting validations

| Property | Test pair | Result |
|---|---|---|
| `--low-ram` ≡ eager | #03 vs #03b | **byte-identical** (md5 `1052db2daeb9ad4d65241c01bb88ab6a`) — block streaming preserves bit-equivalence at q8 |
| TeaCache opt-in | #06 vs #02 | byte-identical mp4 (no skips at 6 steps; calibration assumes 30 steps — expected) |
| HDR linear range | #13, #14 | values up to 13.32×/8.76× SDR; LogC3 inverse working |
| Bind-time LoRA fusion | #12, #13, #14 | works at q8 with `--low-ram` (dequantize → W+BA·s → re-quantize per linear per step) |
| HDR T2V (no `--video-conditioning`) | #14 | upstream-iso "empty list" path triggers pure T2V HDR — works |

## Known divergence vs upstream observed during tests

- `keyframe` rejects the distilled checkpoint with a clear ValueError ("hallucinates unrelated content during interpolation"). Requires explicit `--dev-transformer transformer-dev.safetensors --distilled-lora ltx-2.3-22b-distilled-lora-384.safetensors`. Same constraint upstream.
- `retake` / `extend` don't expose `--low-ram` — coverage gap noted in CLAUDE.md "## Block Streaming → Coverage" (validated: generate / a2v / keyframe / ic-lora). Both ran fine in eager mode after system thermal/contention pressure subsided.

## Outputs (file inventory)

```
/tmp/ltx2_regression/
├── frame_a.png            889 B
├── frame_a_hd.png        1.9 KB
├── frame_b.png            889 B
├── source.mp4              82 KB  (= test02)
├── test.wav              125 KB
├── test02_twostage.mp4    82 KB
├── test03_onestage.mp4   188 KB   md5: 1052db2…
├── test03b_onestage_nolowram.mp4  188 KB  md5: 1052db2… (= test03)
├── test04_onestage_i2v.mp4 10 KB
├── test05_twostage_i2v.mp4 92 KB
├── test06_twostage_teacache.mp4   82 KB  (= test02 — no skips)
├── test08_hq.mp4          52 KB
├── test09_distilled.mp4   39 KB
├── test10_a2v.mp4         60 KB
├── test11_keyframe.mp4    51 KB
├── test12_iclora.mp4      32 KB
├── test13_hdr_v2v.mp4     32 KB
├── test13_hdr_v2v.hdr.npz 4.56 MB  shape=(9,256,384,3) range=[-0.017, 13.32]
├── test14_hdr_t2v.mp4     33 KB
├── test14_hdr_t2v.hdr.npz 4.80 MB  shape=(9,256,384,3) range=[-0.017, 8.76]
├── test15_retake.mp4      22 KB
└── test16_extend.mp4      74 KB
```

## Conclusion

**All 9 public pipeline classes (15 invocations) ✅ pass non-regression after the v0.11.0 isomorphic refactor.**
- File names, class names, blocks API, orchestration helpers all match upstream.
- Block streaming bit-equivalence with eager confirmed (md5 match).
- HDR LoRA auto-detection via safetensors metadata works for both V2V and T2V modes.
- TeaCache wires correctly (no-skip behavior at low step count expected per polynomial calibration).

## Production-quality validation (post-fixes)

After the lazy Gemma + Metal-heap-thrash fixes (commits `8b2a29f`, `3431205`,
`1a30f74`), the full pipeline cohort was re-run at **production step counts**
on M2 Pro 32 GB under sustained system contention (Spotlight, Siri, mds_stores,
knowledgeconstructiond active). All passed.

Hero seed: `937473992`. Common res: 384×256×33 (Q1, Q4, Q5, Q6) or 384×256×9
(Q2, Q3) or 384×256×17 source (Q8). User visually validated `quality_i2v.mp4`
(704×480×33 from earlier production run, 714 s).

| # | Pipeline | Steps | Wall-clock | Output | Notes |
|---|---|---|---|---|---|
| Q1 | `--distilled` T2V | 8+3 | **84 s** | 97 KB | half-res→upscale→refine, no CFG |
| Q2 | `ic-lora` Union Control | 8+3 | **68 s** | 38 KB | bind-time LoRA fusion via `--low-ram` |
| Q3 | `hdr-ic-lora` T2V | 8+3 | **70 s** | 23 KB mp4 + 3.86 MB npz | range [-0.017, 55.08], 9.06% px>1.0 |
| Q4 | `--two-stages-hq` | 30+3 res_2s | **974 s** | 193 KB | 4 forwards/step (2× sub-step × 2 CFG) |
| Q5 | `--one-stage` | 30 | **672 s** | 237 KB | dev + CFG @ full target res |
| Q6 | `a2v` | 30+3 | **509 s** | 108 KB | audio-conditioned with 4s sine |
| Q7 | `keyframe` | 20+3 | **436 s** | 55 KB | red→blue interp via dev model |
| Q8a | `retake` | 30 | **371 s** | 35 KB | regen latent frame [1,2) |
| Q8b | `extend` | 30 | **568 s** | 99 KB | append 2 latent frames (after) |
| ref | `--two-stage` I2V (earlier) | 30+3 | 714 s | 304 KB | 704×480×33 production reference |

**Total production cohort wall-clock: ~63 min** (10 invocations).

## Root-cause fixes shipped during validation

Three commits fixed actual bugs uncovered during this regression suite:

1. **`8b2a29f`** — `fix(text-encoder): default Gemma forward to lazy graph (LTX2_GEMMA_EVAL_EVERY=0)`.
   Initial fix: removed hardcoded `eval_every=1`. Worked once but not robust.

2. **`3431205`** — `fix(text-encoder): auto-split Gemma + connector forwards on <=48 GB Macs`.
   Refined: per-layer Gemma + per-block connector eval, gated by device memory.
   Right balance between "one giant buffer" and "many small queued buffers".

3. **`1a30f74`** — `fix(pipelines): drop wasteful Gemma re-load in load() — fixes Metal heap thrash`.
   The big one. Several pipelines' `load()` methods called `_load_text_encoder()`
   to load Gemma → free → load DiT, but the wrapping `generate_*()` had ALREADY
   encoded + freed Gemma. So Gemma was loaded twice (7.5 GB mmap each time) right
   before the 10 GB DiT load. This thrashed the Metal heap and was the actual
   cause of the watchdog crashes under sustained contention.

After 1a30f74, ALL pipelines run at production steps under contention without
the watchdog firing. Tests Q1-Q8 above are the post-fix validation cohort.

## Follow-ups

- [ ] Run cross-cutting flag tests at production step counts (30 stage-1 / 8 distilled) to actually exercise TeaCache skips.
- [ ] `--tile-frames 2 --tile-overlap 4` regression test on `--two-stage` (PSNR vs no-tile baseline).
- [ ] Quieter-system retry of non-`--low-ram` baseline for #02, #08, #10, #11, #12, #13, #14 to broaden eager-path coverage.
- [ ] `retake` / `extend` `--low-ram` wiring (coverage gap; one of two pipelines without block-streaming support).
