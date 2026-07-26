# AI Studio MLX Persistent BF16 Probe — 2026-07-26

## Status

**Exploratory and inconclusive.**

This probe sent two back-to-back requests through one persistent Python process
and one retained `DistilledPipeline`. It was intended to expose lifecycle behavior,
not to qualify a persistent-worker optimization.

There were only two requests, no interleaved BF16 controls, and no controlled
cooldown or idle-pressure recovery period. The result is not protocol-qualified
and demonstrates no speed benefit.

## Fixed Workload

Both requests used the exact pinned AI Studio-equivalent BF16 workload:

| Field | Value |
|---|---|
| Process count | One |
| Pipeline instance | One retained `DistilledPipeline` |
| Model | `dgrauet/ltx-2.3-mlx` |
| `low_memory` | `False` |
| Pipeline | Distilled image-to-video |
| Geometry | 768 × 512 |
| Frames | 129 |
| Frame rate | 24 fps |
| Denoising schedule | 8 + 3 steps |
| Seed | 424242 |
| Source image and prompt | Same pinned values as the BF16 baseline |
| Output SHA-256 | `9ceef33d97dc24801dc184eeb6e9f825653fa12d3f8123f23d216d44e3ce395c` |

Both requests reproduced the byte-identical BF16 control output.

## Preserved Evidence

The exact JSONL profiles are preserved under
[`docs/benchmarks/ai-studio-mlx-2026-07-26/`](benchmarks/ai-studio-mlx-2026-07-26/).
Generated MP4 files are not stored in this repository.

| Request | Preserved profile | SHA-256 |
|---|---|---|
| Request 1 | [`persistent_bf16_request1.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/persistent_bf16_request1.jsonl) | `7defdc7ff85b72a2e40893c610c575c71aabb8e85ee431f933c9d81f72c5408c` |
| Request 2 | [`persistent_bf16_request2.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/persistent_bf16_request2.jsonl) | `0fd420ae455d1e9150cac9f54f2af3b5f8ec3c611a39c31bc18053439cbe6fdb` |

## Observations

### End-to-end and memory

| Measurement | Request 1 | Request 2 |
|---|---:|---:|
| External wall time | 63.325969 s | 66.082885 s |
| Profiler elapsed time | 63.315309 s | 66.073691 s |
| Process RSS | 47,424,143,360 B | 47,424,520,192 B |
| Maximum process RSS | 47,590,653,952 B | 47,590,653,952 B |
| MLX active memory after request | 50.8022258 GiB | 50.8022258 GiB |
| MLX peak memory | 57.7277170 GiB | 57.7280529 GiB |
| Swap activity | Zero swaps observed | Zero swaps observed |

The second request was not faster: its external wall time was 2.756916 seconds
longer than the first request. With only two uncontrolled requests, this difference
must not be interpreted as a persistent-worker regression either.

### Pipeline phases

| Phase | Request 1 | Request 2 |
|---|---:|---:|
| Model/component loading total | approximately 5.23 s | effectively zero |
| Prompt processing | See preserved JSONL | 0.964474 s |
| Stage 1 half-resolution denoise | 17.547718 s | 19.823899 s |
| Stage 2 full-resolution denoise | 27.400601 s | 33.121827 s |
| Decode and mux | 11.038154 s | 11.205945 s |

Pipeline reuse removed model/component loading from the second request, but the
second request's longer denoising phases outweighed that saved load time in this
probe. An operator trace and controlled repeats would be required to explain why;
this report does not assign a cause.

## Retained-Memory Cost

The naive retain-everything lifecycle left approximately 50.8022258 GiB of active
MLX memory after each request. Peak MLX memory was approximately 57.73 GiB, and
the process retained roughly 47.4 billion bytes of RSS.

This is the clearest factual result of the probe. A persistent worker cannot be a
universal default based only on avoiding reload time. It needs a hardware- and
pressure-aware policy that accounts for:

- Installed unified memory and headroom required by macOS and foreground apps.
- Current memory-pressure class and swap activity.
- A bounded idle retention period.
- Explicit unload/cleanup on pressure, inactivity, errors, and application exit.
- AI Studio's one-heavy-render ledger and orphan reconciliation.
- A safe fallback to the current fresh-process lifecycle.

This report does not choose capacity thresholds. Those thresholds require
controlled measurements across supported Apple Silicon memory tiers.

## What Would Qualify the Design

A protocol-qualified persistent-worker experiment still requires:

1. An opening BF16 control cohort collected under stable conditions.
2. A declared warmup and cooldown policy.
3. At least five sequential persistent requests, plus idle-memory samples between
   requests.
4. Interleaved or restarted fresh-process controls.
5. Memory-pressure and retained-RSS growth gates.
6. A closing BF16 control cohort.
7. Full structural and audiovisual parity checks.

Until those conditions are met, this probe supports only two narrow statements:
the retained pipeline avoided second-request model loading, and the naive
retention policy held about 50.8 GiB of active MLX memory without demonstrating a
speed benefit.
