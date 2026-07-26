# AI Studio MLX library-scoped Metal capture — 2026-07-26

## Status

A library-scoped MLX Metal capture completed successfully. This is a
**structural trace**, not a benchmark: capture overhead materially changes
timing, and the bundle has not yet been analyzed in Xcode's GPU debugger for
per-dispatch durations or hardware counters.

The captured request was:

| Field | Value |
|---|---|
| Model | `dgrauet/ltx-2.3-mlx-q8` |
| Pipeline | Distilled I2V |
| Geometry | 384×256×33 at 24 fps |
| Denoising | 8 stage-1 + 3 stage-2 steps |
| Seed | 424242 |
| Prompt/image | Same pinned prompt and source image as the BF16 control |
| Capture enablement | `MTL_CAPTURE_ENABLED=1` |
| Capture selector | Stage 2 only |
| Result | Success; no swaps observed |

The pinned source and prompt are recorded in
[the BF16 control report](AI_STUDIO_MLX_BASELINE_2026-07-26.md#fixed-workload).

## Artifact manifest

| Artifact | Location / identity | Retention |
|---|---|---|
| Metal capture | Originally `/tmp/ltx-mlx-profile.nsGZ9V/q8_stage2_micro_enabled.gputrace` | 22 GB; moved to macOS Trash after this manifest was verified. It remains recoverable until Trash is emptied and is not repository evidence. |
| Capture metadata | Bundle reports one captured frame | Recorded here |
| Raw profile | [`trace_q8_stage2_micro_enabled.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/trace_q8_stage2_micro_enabled.jsonl) | Preserved in the repository |
| Raw-profile SHA-256 | `6b3119c5f95fb5bf75578c41256215a6101eafb59ae07276fab0e208380c03f1` | Verified against the temporary source |
| Original profile | `/tmp/ltx-mlx-profile.nsGZ9V/q8_stage2_capture_micro_enabled.jsonl` | Temporary |
| Output MP4 SHA-256 | `3df5c3fc92e012732113b7fb7deb00fc7d0429bb48270026fb4916c9cb5c1385` | Output is not preserved here |

## Observed run envelope

| Measurement | Captured-run value |
|---|---:|
| Profiler elapsed | 31.8966 s |
| Stage 1 denoise | 4.0938 s |
| Stage 2 phase | 20.8060 s |
| Actual MLX capture interval | 15.6530 s |
| Decode and mux | 1.8738 s |
| Peak MLX memory | 21.6191 GiB |
| Swaps | 0 |

These values establish that the selected capture completed and bound its
interval. They must not be compared with uncaptured controls as performance
results or used to estimate capture overhead.

## Inspectable Metal resources

The capture bundle exposed native MLX/Metal resource names including:

- q4 BF16 affine QMM:
  `affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0`;
- q8 float affine QMM variants matching
  `affine_qmm_t_nax_float_gs_64_b_8_bm64_bn64_bk64_wm2_wn2_alN_{true,false}_batch_0`;
- Steel float32 attention variants with head dimensions 128 and 64,
  `bq64`/`bk32`, no mask, noncausal mode, and no sinks;
- a fused NAX BF16 GEMM using `bm64`/`bn128`/`bk256`; and
- a float32 split-K GEMM resource.

Resource-name presence is not an invocation count. Metal pipeline states and
resources can be created before `start_capture()` and retained into the
captured frame; the Gemma-related q4 resource is the clearest example of why
presence alone does not prove execution during stage 2.

## What this establishes

The trace confirms that the request's heavy tensor math reaches native
MLX/Metal QMM, Steel attention, fused GEMM, and split-K-capable resources. It
does **not** identify the hottest dispatch or prove which resource variant each
LTX operation selected.

The next analysis step is to open the capture in Xcode's GPU debugger and
inspect per-dispatch timing, counters, exact LTX-shape routing, tile selection,
and split-K decisions. That evidence should guide a narrowly scoped MLX/kernel
experiment. It points toward verifying and improving MLX routing for LTX's
actual shapes—not replacing Python wholesale or rewriting already-native math
without a measured dispatch-level reason.

Related static analysis:
[AI Studio MLX operator audit](AI_STUDIO_MLX_OPERATOR_AUDIT.md).

For QA runs that need system-level scheduling and per-dispatch evidence in
addition to the library-scoped `.gputrace`, use the opt-in launcher:

```bash
uv run python scripts/capture_metal_system_trace.py \
  --output /path/to/trial.trace \
  --time-limit 5m \
  -- ltx-2-mlx generate ...
```

It invokes the installed `Metal System Trace` xctrace template with
`--no-prompt`, exports the trace table of contents, and writes a JSON manifest
containing the target argv, runtime commit/version/device identity, return
codes, and artifact hashes. It is never enabled during normal generation.
MLX itself exposes capture start/stop and allocator counters but no
programmatic per-dispatch duration API; dispatch timing must therefore be read
from the `.gputrace`/`.trace` in Xcode Instruments or an exported xctrace table.
