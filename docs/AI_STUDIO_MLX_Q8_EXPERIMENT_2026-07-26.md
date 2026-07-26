# AI Studio MLX q8 Experiment — 2026-07-26

## Decision

**Strict qualification result: invalid series; q8 is not accepted.**

The q8 cohort completed successfully and produced encouraging observed wall-time
and memory deltas. However, the opening and closing BF16 controls drifted beyond
the benchmark protocol's permitted bounds. The apparent q8 speed improvement is
therefore withdrawn as a qualification claim. In addition, a full blinded output
review was not completed, so the quality gate remains incomplete.

This report records measurements from this series. It does not claim that q8 is
generally faster, equivalent in quality, or ready to replace BF16 in AI Studio.

## Workload and Single Variable

The experiment used the same pinned AI Studio-equivalent ordinary I2V workload as
the opening BF16 control:

| Field | Value |
|---|---|
| Model | `dgrauet/ltx-2.3-mlx-q8` |
| Pipeline | Distilled image-to-video |
| Geometry | 768 × 512 |
| Frames | 129 |
| Frame rate | 24 fps |
| Denoising schedule | 8 + 3 steps |
| Seed | 424242 |
| Source image | Same frozen input as the BF16 control |
| Prompt | Same frozen prompt as the BF16 control |
| LoRAs / extra conditioning | None |

The intended change from the control was the model weight variant: BF16 to q8.
The command otherwise had the form documented in
`AI_STUDIO_MLX_BASELINE_2026-07-26.md`, with the q8 model identifier and
trial-specific output/profile paths.

## Preserved Evidence

The generated MP4 files are not stored in this repository. The exact JSONL
profiles are preserved under
[`docs/benchmarks/ai-studio-mlx-2026-07-26/`](benchmarks/ai-studio-mlx-2026-07-26/).

| Profile | Preserved artifact | SHA-256 |
|---|---|---|
| q8 process-cold | [`candidate_q8_cold.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/candidate_q8_cold.jsonl) | `9c35823729c03bcb755c5f114390e1ec97c872bc3a6405eb9fb07a447271fd61` |
| q8 warm 1 | [`candidate_q8_warm1.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/candidate_q8_warm1.jsonl) | `e88d54ce773f36107cfb583147f8afa3b703f8dcbfaf17907f86291f921bfb92` |
| q8 warm 2 | [`candidate_q8_warm2.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/candidate_q8_warm2.jsonl) | `3362bc5304b0baf328c50609156ef82b6208c9ec9e443af5241766b939980d54` |
| q8 warm 3 | [`candidate_q8_warm3.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/candidate_q8_warm3.jsonl) | `708bb861078a5f1a57b92ec83aa180c14999c48606525cf9d22722082d3e13ce` |
| BF16 closing 1 | [`control_bf16_closing.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/control_bf16_closing.jsonl) | `82fd3f8d7f2b9ae97da874ded8abe1b6fe4f4a7488c09f99907e82f8ff91428c` |
| BF16 closing 2 | [`control_bf16_closing2.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/control_bf16_closing2.jsonl) | `1e63fa68981343825658a6c9139781cab3f49cfa60d18eef141ac047d4fd5bfd` |
| BF16 closing 3 | [`control_bf16_closing3.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/control_bf16_closing3.jsonl) | `872d0b3edba491b74b7e18a8c22d5289960af0b9f25acee514ec751529479643` |

## q8 Cohort

All four q8 trials completed successfully. Each reported peak MLX memory of
23.719919 GiB and zero swaps.

| Trial | Cache classification | Profiler elapsed | Maximum RSS | Peak process footprint |
|---|---|---:|---:|---:|
| q8 cold | Process-cold | 65.367406 s | Not recorded in this report | Not recorded in this report |
| q8 warm 1 | Warm | 58.550597 s | 23,135,813,632 B | 32,604,458,776 B |
| q8 warm 2 | Warm | 58.552142 s | 23,102,963,712 B | 32,571,575,848 B |
| q8 warm 3 | Warm | 58.285749 s | 23,122,657,280 B | 32,625,102,712 B |

### Warm distribution

| Statistic | q8 profiler elapsed |
|---|---:|
| Median | 58.550597 s |
| Minimum | 58.285749 s |
| Maximum | 58.552142 s |
| MAD | 0.001545 s |

The q8 warm phase medians were:

| Phase | Median |
|---|---:|
| Stage 1 half-resolution denoise | 15.581063 s |
| Stage 2 full-resolution denoise | 27.564934 s |
| Decode and mux | 10.223424 s |

These are pipeline-phase measurements. They do not identify a specific operator
or kernel as the cause of any difference.

## Output Structure and Quality Evidence

The q8 outputs were structurally valid:

- H.264 video at 768 × 512, 129 frames, and 24 fps.
- AAC audio present.
- All q8 trials produced the same deterministic output SHA-256:
  `59f0e5b4b73bfab166da6cddd0f130f913fefd3459446d6ab075d561f0dd0113`.

Compared with the deterministic BF16 output, the observed metrics were:

| Metric | q8 versus BF16 |
|---|---:|
| Video PSNR | 27.266714 dB |
| Video SSIM | 0.913573 |
| Audio L1 | 0.0495076 |
| Audio STFT L1 | 0.1307196 |

A midpoint visual inspection found the q8 result visually close to BF16. A full
blinded review of the complete clip was not performed. The numerical metrics and
midpoint inspection are evidence, but they do not complete the protocol's
perceptual quality gate.

## Observed Memory Difference

Relative to the opening BF16 controls:

| Measurement | BF16 reference | q8 reference | Observed delta |
|---|---:|---:|---:|
| Peak MLX memory | 40.3603609 GiB | 23.719919 GiB | -41.23% |
| Median peak process footprint | 49,167,135,496 B | 32,604,458,776 B | -33.69% |

These are substantial observed reductions under this pinned workload. They remain
candidate evidence rather than an acceptance decision because the series failed
the strict control-drift gate and the quality review is incomplete.

## Speed Comparison and Control Drift

### Opening BF16 control

| Statistic | Profiler elapsed |
|---|---:|
| Warm median | 69.792485 s |
| Warm MAD | 0.670318 s |

### Closing BF16 control

| Trial | Profiler elapsed |
|---|---:|
| Closing 1 | 64.120412 s |
| Closing 2 | 65.380305 s |
| Closing 3 | 68.538111 s |
| Median | 65.380305 s |
| MAD | 1.259893 s |

The closing BF16 median was 6.32% faster than the opening BF16 median. The
4.412180-second median shift also exceeds twice the larger control MAD:

```text
2 × 1.259893 s = 2.519786 s
```

Both drift conditions violate the protocol. The strict series is therefore
invalid for a speed qualification.

For completeness, the q8 warm median was provisionally:

- 16.1% lower than the opening BF16 warm median.
- 10.4% lower than the closing BF16 warm median.

Those figures describe this invalid series only. They are not accepted speedup
claims.

## Conclusion

The q8 cohort completed without swaps or structural output failures and showed
lower observed MLX memory, process footprint, and wall time in this series. The
series cannot qualify q8 because:

1. Opening-to-closing BF16 drift exceeded both strict protocol bounds.
2. A complete blinded audiovisual review was not performed.

Qualification requires a new series with interleaved or restarted BF16 controls,
a comparable q8 cold-plus-warm cohort, and the complete blinded quality gate.
