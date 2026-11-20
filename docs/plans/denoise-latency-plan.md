# Plan: cut denoising-stage latency (distilled + two-stage)

## Context

The user asked for an analysis of the denoising ("video generation") stage: find what limits latency and recommend fixes. Both paths matter equally:
- `--distilled` / `--dfr`: 8 + 3 steps, 1 forward each.
- `--two-stage`: 30 steps × up to 4 passes, then 3 steps.

Changes that alter pixels are allowed if they pass a quality check against the float32 render. Exact wins come first.

The estimates below come from code reading plus a FLOP model fitted to the timings documented in `CLAUDE.md`. Nothing has been profiled yet, so phase 0 measures before anything else changes.

### Status (2026-11-19, branch `perf/stg-prefix-fp16`)

Done on the branch, one commit each, every one bit-identical by unit test. The sha256 e2e check passes: `--distilled`, `--two-stage` (6 stage-1 steps), `--dfr`, `a2v` (6 stage-1 steps) and a float32 `--two-stage` (4 steps), 2.5 q8, 512×768×49, seed 5, write byte-identical mp4s on the branch and on `e0dc64d`. Full suite: 1965 passed, 118 skipped.

`scripts/bench_denoise.py`, M5 Pro 48 GB, 2.5 q8, mean of the timed steps (one warm-up step dropped), `e0dc64d` (STG prefix sharing and float16 default already in) → branch:

| run | tokens | float16 s/step | float32 s/step | peak memory float16 / float32 |
|---|---:|---:|---:|---:|
| distilled I2V 512×768×97 | 4,992 | 12.18 → 11.64 (−4 %) | 19.44 → 16.25 (−16 %, baseline possibly inflated by a concurrent test run) | 22.7 → 22.9 / 23.3 → 23.2 GB |
| distilled I2V 576×1024×121 | 9,216 | 24.66 → 22.71 (−8 %) | 34.05 → 31.28 (−8 %) | 25.4 → 25.7 / 26.0 → 26.2 GB |
| guided stage 1 (CFG + STG + modality) 192×320×97 | 780 | 7.68 → 6.61 (−14 %) | 9.34 → 8.64 (−8 %) | 20.9 → 23.1 / 21.0 → 25.5 GB (text K/V cache) |

STG prefix sharing alone (baseline, float16): 8.24 → 7.68 s/step (−7 %, half the ~14 % estimate: at 780 tokens the per-pass fixed costs weigh more). Float16 against float32 at baseline: −37 % / −28 % on the distilled shapes, −18 % on guided stage 1. One synchronised forward (float16, 4,992 tokens) splits as video FF 33 %, `attn1` 27 %, `attn2` 12 %, A↔V 10 %, rest 14 % (`attn1` 35 % at 9,216 tokens), so the share table below overstates attention at these sizes. Whole small e2e renders (512×768×49, dominated by load and decode) moved within noise (−5 % to +7 %).
- Phase 0: `scripts/bench_denoise.py` (real weights, random states/embeddings, `--profile-ops`).
- Item 1: STG prefix sharing.
- Item 2: text K/V shared across the guidance passes of one evaluation (`TextKVCache`, `LTX2_TEXT_KV_CACHE`); `tests/test_text_kv_cache.py`.
- Item 3: RoPE tables reused across forwards (keyed on the positions array's identity; one forward's tables kept); additive masks clamped and cast once per forward.
- Item 4: gate after the head transpose; in-place LoRA fusions evaluated block by block before the next stage on the float32 path too (the compute-dtype path already did).

Deferred: item 3's AdaLN dedupe grouping cache (touches the exactness-verification code for a few host syncs per I2V pass), item 3's all-ones-mask drop (after the mask correctness fix), item 5 (async eval guard: needs a watchdog test with the display active).

Second batch (same branch):
- Self-attention mask correctness fix (log-space bias, upstream `_prepare_self_attention_mask`); changes `ic-lora` with `--conditioning-strength` < 1 only. Not render-checked (Union-Control LoRA not local).
- Item 7: Sol in the dev stage 2 (`--two-stage`, `--two-stages-hq` with res_2s substep sigmas, `a2v`). `--two-stages-hq` 704×1024×49 (4,928 stage-2 tokens): 329 of 329 attention calls ran sparse (7 evaluations × 47 blocks); whole render 155 s dense vs 163 s sparse: no gain at this size (Sol pays off from ~9k tokens). Quality not measured.
- Item 6: `--compute-dtype` (CLI form of `LTX2_COMPUTE_DTYPE`; no `auto`, float16 is already the default).
- Item 9: fused (mx.compile) AdaLN modulations and gated residuals, bit-identical.

sha256 e2e after the second batch: the same five renders are still byte-identical to `e0dc64d`. Full suite: 1985 passed, 118 skipped. Bench (one run per arm, s/step), `e0dc64d` → first batch → second batch:

| run | float16 | float32 |
|---|---:|---:|
| distilled I2V 4,992 tokens | 12.18 → 11.64 → 9.38 (−23 %) | 19.44 → 16.25 → 14.61 (−25 %) |
| distilled I2V 9,216 tokens | 24.66 → 22.71 → 19.15 (−22 %) | 34.05 → 31.28 → 30.81 (−10 %) |
| guided stage 1, 780 tokens | 7.68 → 6.61 → 5.85 (−24 %) | 9.34 → 8.64 → 8.93 (−4 %) |

The fused helpers helped float16 far more than float32 (where the matmuls dominate); the guided float32 arm moving back up (8.64 → 8.93) is within run-to-run noise. Repeat runs are needed before quoting any single figure.

Third pass (measurement gates; nothing shipped). M5 Pro 48 GB, 2.5 q8, float16 unless noted:
- Item 10 (fused projections): dropped. One q8 `quantized_matmul` over `to_q|to_k|to_v|to_gate_logits` against four separate calls (micro-benchmark, random weights): 4.22 → 4.18 ms at 780 tokens, 23.38 → 23.36 ms at 4,992, 43.19 → 43.02 ms at 9,216, 85.25 → 83.24 ms at 17,856 (float32: at most −3 %). The projections are compute-bound at these sizes, so reading `x` once saves nothing; the outputs were not bit-identical below ~9k tokens either. Not worth the LoRA / adapter / streaming work.
- Item 11 (batched CFG): dropped. Real dev weights, one forward with cond + uncond at B=2 against two B=1 forwards: −11.0 % at 780 tokens, −6.0 % at 1,248 (the stage 1 of a 512×768×97 render), −2.5 % at 2,808; same peak memory. It covers two of the ~3.4 passes per step (STG runs 20 of 48 blocks), so ~3–4 % per stage-1 step at typical sizes, and the batched outputs differ from the B=1 ones by up to 0.23 (max abs), i.e. it changes samples. Too little for the Prompt Relay / NAG / STG-prefix / text-K/V interplay it would need.
- Item 3's all-ones-mask drop: nothing to do. A self-attention mask exists only through `ConditioningItemAttentionStrengthWrapper`, which `iclora_utils` uses only for `--conditioning-strength` < 1 or a mask video; keyframe guides, slots and references pass `attention_mask=None` and keep `None` unless a mask already exists. The mask is therefore never all-ones.

Float16-default quality gate: run (results and table in `CLAUDE.md`, "DiT compute dtype"). `--two-stage` (2.5 and 2.3), `--two-stages-hq`, `a2v`, `retake` and `extend` pass: clean renders in both dtypes, float16 20–43 % faster end to end; the guided paths land on a different sample at the same seed, `retake` / `extend` stay at ~43 dB. `ic-lora` passes on 2.3 with the Union-Control LoRA and a canny control (strength 1.0 and 0.7; the 0.7 arm also renders the self-attention mask fix without artifacts): same layout, same control adherence, −15 to −27 %. `lipdub` passes on 2.3 with Dub-It 0.9: 44.8 dB between the dtypes, −30 %. Every pipeline the float16 default touches is now gated (one seed per arm).

Fourth pass (branch `perf/denoise-remaining`, M5 Pro 48 GB, 2.5 q8, float16):
- Item 3's AdaLN grouping cache: done. The per-token timestep embedding and its row grouping (`_unique_rows`: `np.unique` plus two host syncs) are computed once per per-token timesteps array (`LTXModel._per_token_embedding_cached`, keyed on identity like the RoPE cache) and shared by the two per-token AdaLN modules of each modality and by every guidance pass of a step: once per step instead of 2 modules × up to 4 passes. Bit-identical (`tests/test_exact_denoise_speedups.py`); the saving is a few host syncs per pass, below the bench's noise. sha256 e2e: `--distilled` I2V and `--two-stage` I2V (6 stage-1 steps), 512×768×49, seed 5, byte-identical to `054cb2b`. Full suite: 1988 passed, 118 skipped.
- Item 5 (async eval guard): dropped. `mx.async_eval` at the 8-block guard, same segments, bit-identical on the tiny model. `scripts/bench_denoise.py`, two runs per arm, sync → async s/step: distilled I2V 4,992 tokens 9.66 / 10.55 → 10.71 / 10.55; guided stage 1, 780 tokens 5.75 / 5.87 → 5.96 / 5.82; same peak memory. No gain beyond run-to-run noise (the GPU is the bottleneck even at 780 tokens), so no watchdog test was run and nothing ships. The overflow guard's per-forward `.item()` is left as is for the same reason.
- Item 8 (Sol default evaluation): run; the quality gate passes, the default is not flipped here. `LTX2_SOL_TAU=1.0,1.25,1.5` against unset, 2.5 q8, float16, I2V, seed 7, one render per arm, M5 Pro 48 GB on battery. Speech prompt ("Good morning! Today we are walking through the old town."). Start image: a front-facing close-up (frame 4 of a 2.5 `--distilled` T2V render, 1024×576). The `tests/fixtures/lisbon_woman_start.png` wide shot cannot gate identity: its face is ~40×60 px and ArcFace to it falls from 0.69 on frame 0 to ~0.15 by frame 8 in the dense render too. ArcFace median over every 4th frame (insightface buffalo_l, score ≥ 0.6, a face found in every sampled frame); WER from Whisper large-v3-turbo; PSNR / SSIM of Sol against dense; motion = mean abs frame-to-frame change.

  | run (1024×576 portrait) | stage-2 tokens | stage-2 s/step dense → Sol | whole render | PSNR / SSIM | ArcFace dense → Sol | WER | motion mean / p99, dense → Sol |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `--distilled` ×121 | 9,216 | 28.1 → 30.3 (+8 %) | 114 → 121 s | 28.6 dB / 0.844 | 0.720 → 0.694 (−0.026) | 0 / 0 | 5.31 / 9.20 → 5.46 / 9.33 |
  | `--dfr` ×121 | 14,400 | 59.6 → 51.5 (−14 %) | 208 → 190 s (−9 %) | 29.4 dB / 0.832 | 0.806 → 0.790 (−0.016) | 0 / 0 | 5.39 / 9.97 → 5.52 / 9.79 |
  | `--distilled` ×241 | 17,856 | 76.8 → 64.2 (−16 %) | 272 → 249 s (−8 %) | 28.7 dB / 0.843 | 0.697 → 0.682 (−0.015) | 0 / 0 | 4.37 / 8.04 → 4.37 / 8.22 |

  The same three arms on the Lisbon wide shot (576×1024, same seed and prompt), stage-2 s/step dense → Sol: 29.8 → 26.5 (−11 %), 51.1 → 47.9 (−6 %), 73.2 → 65.8 (−10 %); whole render −9 % / −5 % / −6 %; WER 0 everywhere; PSNR 25.7–26.4 dB. Every ArcFace drop is within the 0.05 gate, `--dfr` included (the M1 Max's 0.09 did not reproduce on this image). Frames are the same picture (at frame 200 of the 17,856-token pair only a blink lands differently), with no block artifacts. The speed-up is smaller than the M1 Max's (−21–36 % per stage-2 step): the M5 Pro's dense attention is relatively faster. At 9,216 tokens it is within noise (−11 % and +8 % on the two images); from 14,400 tokens up it is −6 to −16 %. If a default ships, it should be `--distilled` / `--dfr` stage 2 from ~14k video tokens, as its own `feat!:` PR (it changes samples at the default settings). One seed per arm, on battery.

Still open: item 12 (TeaCache on 2.5: ~1h45 calibration per sampler). Item 8 needs only the decision on the default (the gate above passes); shipping it is a `feat!:` PR.

Earlier review notes:
- **Item 1 (STG prefix sharing): implemented** (now committed). `BlockPrefix`, `capture_prefix` / `resume_prefix` on `LTXModel.__call__`, `stg_block_prefix` in `samplers.py`, `LTX2_STG_PREFIX_REUSE`, `tests/test_stg_prefix_reuse.py`.
- **Item 6 overtaken: float16 is now the default compute dtype in the working tree** (`LTX2_COMPUTE_DTYPE` unset → float16; `float32` restores the old path). That flip skipped this plan's gate: measured on `--distilled`, `--dfr` and the relay/NAG runs only. Before it ships (as `feat!:`, a 0.y bump), run the quality gate on `--two-stage` stage 1 (CFG + STG), `a2v`, `retake` / `extend`, `ic-lora` and `lipdub`.
- The share table and finding 1 below were derived at the old float32 default. Re-measure them at float16 in phase 0: with cheaper matmuls and attention, elementwise work and syncs take a larger share.

### Where the time goes (one stage-2 forward: 2.5 q8, 17,856 video tokens, float32, the pre-0.19 default)

| Share | Op | Notes |
|---|---|---|
| ~40% | `attn1` attention (N²) | The only thing Sol sparse attention reduces |
| ~25% | Video feed-forward (4096→16384→4096) | Quantized GEMMs with float32 activations |
| ~25% | Other projections | `attn1` QKV/out, `attn2` Q/out, A↔V |
| ~10% | Elementwise, norms, RoPE, gather passes | ~35–40 unfused float32 passes per block; nothing is compiled on the resident path |

Pipeline-level findings:
1. **Precision.** Float32 was the default because the AdaLN tables are F32, which promotes the whole residual stream (`transformer.py:390,403`). Matmuls and attention ran at float32 speed. Float16 (−20–24% per step on M1 Max, `CLAUDE.md` compute-dtype table) is now the default in the working tree; see Status.
2. **Two-stage stage 1 does ~120 forwards, all run one after another at B=1** (`guided_denoise_loop` / `res2s_denoise_loop` in `samplers.py`).
   - The STG pass perturbs only block 28 (`stg_blocks=[28]`), yet it recomputed blocks 0–27, which are identical to the cond pass. Fixed by item 1.
   - Text K/V of the positive prompt is recomputed in the cond, ptb and mod passes of the same step. Same context, same sigma, so it is exact to share. It is ~12% of a forward at stage-1 size, but the ptb pass now runs only 20 blocks, so the recoverable share is ~5% of stage 1.
3. **Sol sparse attention is opt-in, and only wired into `DistilledPipeline._stage2`** (`distilled.py:865`). The dev stage-2 loops don't use it: `ti2vid_two_stages.py:725`, `ti2vid_two_stages_hq.py:357`, `a2vid_two_stage.py:481`.
4. **Work redone every forward that is constant within a stage:**
   - 4 RoPE tables (`model.py` `_compute_rope_freqs`, `rope.py:70-77`). The float64 grid built on the CPU stream has only `num_freqs` entries and is cheap; the real cost is the cos/sin over N tokens. A small win.
   - The AdaLN dedupe grouping: `np.unique` plus host syncs (`.item()`) per forward on I2V/keyframe runs (`model.py:230-264`).
   - The dense (B,N,N) conditioning mask: clipped and cast in every block (`attention.py:42-44`). This is ~1.3 GB per call at 17k tokens. A mask also disables Sol.
5. **Syncs and graph build.**
   - 6 blocking `mx.eval` per forward (`model.py:1122-1126`, watchdog guard). The GPU idles while Python builds the next 8-block graph.
   - With a compute dtype set (now the default), the overflow guard runs `mx.isfinite(...).item()` on every forward's output (`model.py:86`): one more host sync per forward.
   - The distilled-LoRA fusion for two-stage is lazy (`ti2vid_two_stages.py:212`), so it runs inside the first stage-2 forward.
6. **Small exact waste:**
   - Gate multiply then transpose writes a full extra (B,H,N,D) copy (`attention.py:228-231`).
   - Separate Q/K/V/gate GEMMs each re-read x.
   - The quantized FF bias add is an extra pass over (N,16384).

**Correctness finding (fix before item 3's mask changes):** the conditioning self-attention mask is built in [0,1] (`mask_utils.py:56-113`) and reaches `mx.fast.scaled_dot_product_attention` as an additive bias (`attention.py:203`); the only processing on the way is a clip and cast (`attention.py:42-44`), with no log conversion anywhere in `model.py` / `transformer.py` (confirmed). A "0" entry therefore only lowers a logit by 1 instead of masking it, so `ConditioningItemAttentionStrengthWrapper` strengths (`ic-lora`, `lipdub`) and the prev_ref ↔ new_ref "0" block are close to no-ops. Upstream ltx-core is believed to convert the mask to a log-space bias; verify against upstream, then fix it in its own `fix:` PR (it changes `ic-lora` / `lipdub` outputs). It goes before item 3's "all-ones mask → `None`": that drop is exact either way (log(1) = 0, or a constant +1 row shift), but both touch mask construction.

## Recommended approach (ordered; one PR per item, conventional commits)

### Phase 0: measure first (no behaviour change)
- **New `scripts/bench_denoise.py`.**
  - Runs a fixed shape per path, N timed steps after warm-up, prints s/forward per stage and peak memory, plus chip name, MLX version and compute dtype.
  - Fixed shapes: distilled 2.5 q8 512×768×97 and 576×1024×121 I2V; two-stage 2.5 q8 384×640×97.
  - Runs at both float16 (the new default) and `LTX2_COMPUTE_DTYPE=float32`, so the share table can be redone for each.
  - An `LTX2_PROFILE_OPS=1` mode puts `mx.eval` + timer around attn1 / attn2 / AV / FF / elementwise inside one block, to confirm the share table above.
  - Reuse `ltx_pipelines_mlx/utils/estimate.py` timing and the `scripts/mlx_upgrade_report.py` structure.
- **Baseline renders + quality metrics** for the gate. Reuse `tests/regression/capture_baseline.py` / `verify_no_regression.py`, plus the PSNR/ArcFace/WER/LSE-C evals already used in `CLAUDE.md` tables.
- **Float16-default gate** on the pipelines it was not measured on (see Status).

### Phase 1: exact speedups (byte-identical or ≤1 ULP)
1. **STG prefix sharing. Done (uncommitted).**
   - The cond pass captures the state entering `min(stg_blocks)`; the ptb pass resumes from it, in `guided_denoise_loop` and `res2s_denoise_loop`.
   - Saves 28 of 48 blocks of one of 4 passes: ~14% of two-stage stage 1 (none on `--two-stages-hq`, whose default `stg_scale` is 0).
   - Off with modality tiling, TeaCache or a calibration tap; `LTX2_STG_PREFIX_REUSE=0` disables it.
2. **Per-pass text K/V reuse within a step.**
   - Cache positive-context K/V per block index, keyed on (context id, sigma), across the cond / ptb / mod passes. Cover audio cross-attention (`audio_attn2`) too.
   - Clear it each step. Memory: 48 × 2 × 1024 × 4096 elements ≈ 0.8 GB in float16, 1.6 GB in float32, plus the audio share. That is significant on a 32 GB machine: skip or cap it there.
   - Under `--low-ram` one compiled block is rebound for every layer: key the cache by block index and keep it out of the traced graph (pass K/V in as inputs), or leave it off on that path.
   - Applies to stage 1 of `--two-stage`, `--two-stages-hq` and a2v. Expected ~5% of stage 1 now that item 1 shortened the ptb pass.
3. **Stage-invariant caches** (set up once per sampler loop, passed in or memoised on the model):
   - RoPE `(cos, sin)` already cast to the compute dtype.
   - AdaLN dedupe `idx`/`inv` keyed on the denoise mask (`model.py:188-264`), grouping `vt_emb`/`at_emb` once.
   - The attention mask cast once per forward, not per block.
   - An all-ones mask dropped to `None` (a constant row shift doesn't change softmax). This also re-enables Sol. Land after the mask correctness fix.
4. **Small kernel-level cleanups:**
   - Gate applied after the transpose: `out.transpose(0,2,1,3) * gate[..., None]` (`attention.py:228-231`).
   - Eager evaluation of the distilled-LoRA fusion block by block at the stage transition (`ti2vid_two_stages.py:212`).
5. **Async eval guard.** Try `mx.async_eval` at `model.py:1122-1126`, keeping the same 8-block command-buffer split. Ship it only if the watchdog test passes on M1/M2 with the display active. Look at folding the overflow guard's per-forward `.item()` (`model.py:86`) into the sync the sampler already does.

Expected: ~15% on two-stage stage 1 (most of it from item 1, already in the tree; ~20% is the optimistic end), ~3–6% on distilled.

### Phase 2: precision and attention (changes pixels; quality-gated)
6. **`--compute-dtype {auto,float32,float16,bfloat16}` CLI flag**, wired to the existing `LTX2_COMPUTE_DTYPE` path (`_base.py:57-69,590-640`, `model.py:603-621`).
   - The float16 default already exists in the working tree; what remains is the flag and the gate in Status.
   - `auto` picks float16. bf16 is *less* accurate than float16 here (one stage-2 forward vs float32: cos 0.99796 vs 0.99985) and fails the per-forward gate below, so `auto` picks bf16 only if a chip shows a clear speed win *and* bf16 passes the gate.
   - Shipping the default flip is a `feat!:` (0.y bump) per the release rules.
7. **Sol in the dev stage-2 loops.** Copy the `_start/_stop_stage2_sparse_attention` switching from `distilled.py:320-342,865` into `ti2vid_two_stages.py:725`, `ti2vid_two_stages_hq.py:357` and `a2vid_two_stage.py:481`. Keep it opt-in.
   - On `--two-stages-hq`, res_2s's second evaluation runs at a substep sigma. `SparseAttentionState` matches sigmas to its table within 1e-2 (`sparse_attention.py:101-102`), so those forwards would fall back to dense and the gain roughly halves. Pass the substep sigmas (with their taus) in the table.
8. **Sol default evaluation.** Benchmark `LTX2_SOL_TAU=1.0,1.25,1.5` as the default for distilled stage 2 at ≥9k tokens (measured −16–36% per stage-2 step).
   - The ArcFace drop is the gate (≤ 0.05). Measured drops: 0.01–0.06 on `--distilled`, 0.09 on `--dfr` (0.73 → 0.64), so as measured `--dfr` fails it. Scope a default to `--distilled` only, unless a `--dfr` rerun passes.
   - Recommend default-on only above a token threshold, e.g. ≥17k tokens, where the gain is largest.

### Phase 3: graph and kernel fusion (exact to ~1 ULP; measured-gated)
9. **Compiled elementwise helpers.** `mx.compile(shapeless=True)` for `rms*(1+scale)+shift` and `h + out*gate`, used at the `transformer.py` modulation and residual call sites.
   - This avoids the pytree issues that block compiling the whole resident block (perturbation configs, the lazy AdaLN carrier).
   - Target: recover most of the elementwise share (larger than ~10% at float16; see phase 0).
10. **Fused projections at load time.**
    - Concatenate `to_q|to_k|to_v|to_gate_logits` (attn1) and `to_q|to_gate_logits` (attn2, A→V) along the output axis. Quantized weight, scales and biases concat cleanly per row group.
    - Keep the original module names for LoRA fusion and adapters. Needs care with `BlockLoraSource` / unfused adapters, so it comes after a design check.
11. **Batch the CFG passes (cond+uncond, B=2) in stage 1, behind a flag.**
    - Measure at stage-1 token counts (1–5k), where weight reads, not FLOPs, dominate.
    - B=2 roughly doubles stage-1 activation memory, against the documented 32 GB limits (CFG-only ≈ 97 frames at half-res). Refuse it under `--low-ram` and when the estimate exceeds the memory budget.
    - Prompt Relay needs a zero-bias row for the negative pass.

### Phase 4 (optional, changes samples): step reduction
12. **Recalibrate TeaCache for the 2.5 dev stage 1** using the existing `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/scripts/calibrate_teacache.py` / `fit_teacache_poly.py` (~1h45 per sampler). It is refused on 2.5 today (`_base.py:352-368`). Note that TeaCache disables STG prefix sharing (item 1), so measure against the item-1 baseline.

## Critical files
- `packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/{model.py,transformer.py,attention.py,rope.py,adaln.py,feed_forward.py}`
- `packages/ltx-core-mlx/src/ltx_core_mlx/conditioning/mask_utils.py` (mask correctness fix)
- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py`
- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/{_base.py,distilled.py,ti2vid_two_stages.py,ti2vid_two_stages_hq.py,a2vid_two_stage.py,cli.py}`
- `packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/sparse_attention.py` (reuse; item 7 may extend the sigma table for res_2s substeps)
- New: `scripts/bench_denoise.py`

## Verification
- **Unit tests** per item:
  - Prefix-sharing ptb output equals the full ptb forward (`mx.array_equal`) on the tiny model (`tests/test_stg_prefix_reuse.py`, in the tree).
  - K/V and RoPE caches give identical outputs, resident and under `--low-ram`.
  - The all-ones-mask drop gives identical outputs.
  - The mask fix: a 0 entry blocks attention (output independent of the masked keys).
  - Compiled helpers within `atol=1e-5`.
  - `tests/test_docs_flags.py` updated for `--compute-dtype`.
- **Phase 1:** sha256 of `--distilled`, `--two-stage`, `--dfr` and `a2v` renders at 512×768×49, seed 5, matches `main` at the same compute dtype (the repo's standard check). Bench shows the s/forward drop.
- **Phase 2/3 quality gate:** against the float32 dense render of the same seed, check PSNR/SSIM, ArcFace median (I2V), Whisper WER and SyncNet LSE-C on the standard prompts in `CLAUDE.md`. Accept only if within the bands already seen for float16 (cos ≥ 0.9998 per forward; float16 measured 0.99985, so the margin is thin; bf16's 0.99796 fails) and Sol (ArcFace drop ≤ 0.05; `--dfr` measured 0.09).
- `uv run pytest`, `ruff check`, CI green. Record before/after tables in `CLAUDE.md` alongside the existing ones.
