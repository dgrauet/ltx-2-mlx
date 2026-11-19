# Plan: cut denoising-stage latency (distilled + two-stage)

## Context

The user asked for an analysis of the denoising ("video generation") stage: find what limits latency and recommend fixes. Both paths matter equally:
- `--distilled` / `--dfr`: 8 + 3 steps, 1 forward each.
- `--two-stage`: 30 steps × up to 4 passes, then 3 steps.

Changes that alter pixels are allowed if they pass a quality check against the float32 render. Exact wins come first.

The estimates below come from code reading plus a FLOP model fitted to the timings documented in `CLAUDE.md`. Nothing has been profiled yet, so phase 0 measures before anything else changes.

### Where the time goes (one stage-2 forward: 2.5 q8, 17,856 video tokens, float32 default)

| Share | Op | Notes |
|---|---|---|
| ~40% | `attn1` attention (N²) | The only thing Sol sparse attention reduces |
| ~25% | Video feed-forward (4096→16384→4096) | Quantized GEMMs with float32 activations |
| ~25% | Other projections | `attn1` QKV/out, `attn2` Q/out, A↔V |
| ~10% | Elementwise, norms, RoPE, gather passes | ~35–40 unfused float32 passes per block; nothing is compiled on the resident path |

Pipeline-level findings:
1. **Precision.** Float32 is the default because the AdaLN tables are F32, which promotes the whole residual stream (`transformer.py:390,403`). Matmuls and attention run at float32 speed. Float16 is already implemented but opt-in through an env var: −20–24% per step on M1 Max (`CLAUDE.md` compute-dtype table).
2. **Two-stage stage 1 does ~120 forwards, all run one after another at B=1** (`samplers.py:1358,1383,1414,1434`).
   - The STG pass perturbs only block 28 (`stg_blocks=[28]`), yet it recomputes blocks 0–27, which are identical to the cond pass.
   - Text K/V of the positive prompt is recomputed in the cond, ptb and mod passes of the same step. Same context, same sigma, so it is exact to share. It is ~12% of a forward at stage-1 size.
3. **Sol sparse attention is opt-in, and only wired into `DistilledPipeline._stage2`** (`distilled.py:865`). The dev stage-2 loops don't use it: `ti2vid_two_stages.py:725`, `ti2vid_two_stages_hq.py:357`, `a2vid_two_stage.py:481`.
4. **Work redone every forward that is constant within a stage:**
   - 4 RoPE tables, with the float64 frequency grid built on the CPU stream (`model.py:950-989`, `rope.py:70-77`).
   - The AdaLN dedupe grouping: `np.unique` plus ~4–8 host syncs per forward on I2V/keyframe runs (`model.py:204-216,916-941`).
   - The dense (B,N,N) conditioning mask: clipped and cast in every block (`attention.py:42-44`). This is ~1.3 GB per call at 17k tokens. A mask also disables Sol.
5. **Syncs and graph build.**
   - 6 blocking `mx.eval` per forward (`model.py:1077-1080`, watchdog guard). The GPU idles while Python builds the next 8-block graph.
   - The distilled-LoRA fusion for two-stage is lazy (`ti2vid_two_stages.py:212`), so it runs inside the first stage-2 forward.
6. **Small exact waste:**
   - Gate multiply then transpose writes a full extra (B,H,N,D) copy (`attention.py:228-231`).
   - Separate Q/K/V/gate GEMMs each re-read x.
   - The quantized FF bias add is an extra pass over (N,16384).

**Out-of-scope finding (correctness, not latency):** the conditioning self-attention mask is built in [0,1] (`mask_utils.py:56-113`), but it is passed to `mx.fast.scaled_dot_product_attention` as an additive bias (`attention.py:203`) with no log conversion. A "0" entry therefore only lowers a logit by 1 instead of masking it. Upstream ltx-core converts the mask to a log-space bias. Confidence is medium; it needs checking against upstream. I'll raise it as its own follow-up rather than mix it into perf PRs.

## Recommended approach (ordered; one PR per item, conventional commits)

### Phase 0: measure first (no behaviour change)
- **New `scripts/bench_denoise.py`.**
  - Runs a fixed shape per path, N timed steps after warm-up, prints s/forward per stage and peak memory, plus chip name and MLX version.
  - Fixed shapes: distilled 2.5 q8 512×768×97 and 576×1024×121 I2V; two-stage 2.5 q8 384×640×97.
  - An `LTX2_PROFILE_OPS=1` mode puts `mx.eval` + timer around attn1 / attn2 / AV / FF / elementwise inside one block, to confirm the share table above.
  - Reuse `pipes/utils/estimate.py` timing and the `scripts/mlx_upgrade_report.py` structure.
- **Baseline renders + quality metrics** for the gate. Reuse `tests/regression/capture_baseline.py` / `verify_no_regression.py`, plus the PSNR/ArcFace/WER/LSE-C evals already used in `CLAUDE.md` tables.

### Phase 1: exact speedups (byte-identical or ≤1 ULP)
1. **STG prefix sharing.**
   - `LTXModel.__call__` gets `capture_at_block` / `resume_from` (video + audio hidden states and the per-block kwargs) next to the existing `block_stack_override` hook.
   - In `guided_denoise_loop` and `res2s_denoise_loop`, the cond pass captures the state at `min(stg_blocks)` and the ptb pass resumes from it.
   - Saves 28 of 48 blocks of one of 4 passes: ~14% of two-stage stage 1.
   - TeaCache's `block_stack_override` path must keep working.
2. **Per-pass text K/V reuse within a step.**
   - Cache positive-context K/V per block, keyed on (context id, sigma), across the cond / ptb / mod passes.
   - Clear it each step, so memory is bounded at ~48 × 2 × 1024 × 4096.
   - Applies to stage 1 of `--two-stage`, `--two-stages-hq` and a2v.
3. **Stage-invariant caches** (set up once per sampler loop, passed in or memoised on the model):
   - RoPE `(cos, sin)` already cast to the compute dtype (`model.py:950-989`).
   - AdaLN dedupe `idx`/`inv` keyed on the denoise mask (`model.py:188-217`), grouping `vt_emb`/`at_emb` once.
   - The attention mask cast once per forward, not per block.
   - An all-ones mask dropped to `None` (a constant row shift doesn't change softmax). This also re-enables Sol.
4. **Small kernel-level cleanups:**
   - Gate applied after the transpose: `out.transpose(0,2,1,3) * gate[..., None]` (`attention.py:228-231`).
   - Eager evaluation of the distilled-LoRA fusion block by block at the stage transition (`ti2vid_two_stages.py:212`).
5. **Async eval guard.** Try `mx.async_eval` at `model.py:1080`, keeping the same 8-block command-buffer split. Ship it only if the watchdog test passes on M1/M2 with the display active.

Expected: ~15–20% on two-stage stage 1, ~3–6% on distilled.

### Phase 2: precision and attention (changes pixels; quality-gated)
6. **`--compute-dtype {auto,float32,float16,bfloat16}` CLI flag**, wired to the existing `LTX2_COMPUTE_DTYPE` path (`_base.py:57-69,590-640`, `model.py:603-621`).
   - `auto` picks float16 on M1/M2 (no native bf16). On M3+ it picks whichever of bf16/fp16 wins in the phase-0 bench.
   - Make `auto` the default only after the quality gate passes on all pipelines. That is a `feat!:` (0.y bump) per the release rules.
   - Expected −20–24% per step everywhere.
7. **Sol in the dev stage-2 loops.** Copy the `_start/_stop_stage2_sparse_attention` switching from `distilled.py:320-342,865` into `ti2vid_two_stages.py:725`, `ti2vid_two_stages_hq.py:357` and `a2vid_two_stage.py:481`. Keep it opt-in.
8. **Sol default evaluation.** Benchmark `LTX2_SOL_TAU=1.0,1.25,1.5` as the default for distilled stage 2 at ≥9k tokens (measured −16–36% per stage-2 step).
   - The ArcFace drop (0.01–0.09) is the gate.
   - Recommend default-on only above a token threshold, e.g. ≥17k tokens, where the gain is largest.

### Phase 3: graph and kernel fusion (exact to ~1 ULP; measured-gated)
9. **Compiled elementwise helpers.** `mx.compile(shapeless=True)` for `rms*(1+scale)+shift` and `h + out*gate`, used at the `transformer.py` modulation and residual call sites.
   - This avoids the pytree issues that block compiling the whole resident block (perturbation configs, the lazy AdaLN carrier).
   - Target: recover most of the ~10% elementwise share.
10. **Fused projections at load time.**
    - Concatenate `to_q|to_k|to_v|to_gate_logits` (attn1) and `to_q|to_gate_logits` (attn2, A→V) along the output axis. Quantized weight, scales and biases concat cleanly per row group.
    - Keep the original module names for LoRA fusion and adapters. Needs care with `BlockLoraSource` / unfused adapters, so it comes after a design check.
11. **Batch the CFG passes (cond+uncond, B=2) in stage 1, behind a flag.**
    - Measure at stage-1 token counts (1–5k), where weight reads, not FLOPs, dominate.
    - Prompt Relay needs a zero-bias row for the negative pass.

### Phase 4 (optional, changes samples): step reduction
12. **Recalibrate TeaCache for the 2.5 dev stage 1** using the existing `scripts/calibrate_teacache.py` / `fit_teacache_poly.py` (~1h45 per sampler). It is refused on 2.5 today (`_base.py:352-368`).

## Critical files
- `packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/{model.py,transformer.py,attention.py,rope.py,adaln.py,feed_forward.py}`
- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py`
- `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/{_base.py,distilled.py,ti2vid_two_stages.py,ti2vid_two_stages_hq.py,a2vid_two_stage.py,cli.py}`
- `packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/sparse_attention.py` (reuse, no change)
- New: `scripts/bench_denoise.py`

## Verification
- **Unit tests** per item:
  - Prefix-sharing ptb output equals the full ptb forward (`mx.array_equal`) on the tiny model used in `tests/test_compute_dtype.py` / `tests/test_block_streaming.py`.
  - K/V and RoPE caches give identical outputs.
  - The all-ones-mask drop gives identical outputs.
  - Compiled helpers within `atol=1e-5`.
  - `tests/test_docs_flags.py` updated for `--compute-dtype`.
- **Phase 1:** sha256 of `--distilled`, `--two-stage`, `--dfr` and `a2v` renders at 512×768×49, seed 5, matches `main` (the repo's standard check). Bench shows the s/forward drop.
- **Phase 2/3 quality gate:** against the float32 dense render of the same seed, check PSNR/SSIM, ArcFace median (I2V), Whisper WER and SyncNet LSE-C on the standard prompts in `CLAUDE.md`. Accept only if within the bands already seen for float16 (cos ≥ 0.9998 per forward) and Sol (ArcFace drop ≤ 0.05).
- `uv run pytest`, `ruff check`, CI green. Record before/after tables in `CLAUDE.md` alongside the existing ones.
