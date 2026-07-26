# AI Studio MLX Opening BF16 Control Cohort — 2026-07-26

## Status

This document records the completed opening BF16 control cohort observed on
2026-07-26: one process-cold observation followed by three warm trials. It
establishes the opening control distribution for this workload, but does not by
itself accept an optimization candidate.

The run artifacts were collected under:

```text
/tmp/ltx-mlx-profile.nsGZ9V
```

The small JSONL profile is preserved durably at
[`docs/benchmarks/ai-studio-mlx-2026-07-26/control_bf16.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/control_bf16.jsonl).
The fine-grained instrumentation repeat is preserved at
[`docs/benchmarks/ai-studio-mlx-2026-07-26/control_bf16_finegrained.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/control_bf16_finegrained.jsonl).
Warm trials 2 and 3 are preserved at
[`control_bf16_warm2.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/control_bf16_warm2.jsonl)
and
[`control_bf16_warm3.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/control_bf16_warm3.jsonl).
The generated MP4 is not stored in this repository.

The opening-control requirement of the benchmark protocol is now complete.
Candidate acceptance still requires a comparable candidate cohort and a closing
BF16 control to detect environmental drift.

## Fixed Workload

| Field | Observed value |
|---|---|
| Model | `dgrauet/ltx-2.3-mlx` |
| Weight format | BF16 |
| Pipeline | Distilled image-to-video |
| Output geometry | 768 × 512 |
| Frames | 129 |
| Frame rate | 24 fps |
| Denoising schedule | 8 + 3 steps |
| Seed | 424242 |
| Result | Completed successfully |

The run used this benign source image and prompt:

```text
Input: /Users/paulericksen/Documents/ComfyUI/output/video_frames/clipboard_capture_04.png
Prompt: A slow, steady lateral camera move reveals the scene with natural subject motion and physically consistent lighting.
```

The profile does not retain the image path or prompt text, so they are recorded
explicitly here. The command had the following form:

```bash
/usr/bin/time -l ltx-2-mlx generate \
  --model dgrauet/ltx-2.3-mlx \
  --image "/Users/paulericksen/Documents/ComfyUI/output/video_frames/clipboard_capture_04.png" \
  --prompt "A slow, steady lateral camera move reveals the scene with natural subject motion and physically consistent lighting." \
  --output "<trial-output.mp4>" \
  --height 512 \
  --width 768 \
  --frames 129 \
  --frame-rate 24 \
  --steps 8 \
  --seed 424242 \
  --distilled \
  --profile-json "<trial-profile.json>"
```

## Process-Cold Observation

The original broad-instrumentation run is the process-cold observation. Its
profile predates the finer `DistilledPipeline` phase boundaries used for the three
warm trials, so its wall and memory measurements remain valid while its large
internal interval is not pooled with warm phase statistics.

### Timing

| Measurement | Observed value |
|---|---:|
| Profiler elapsed time | 72.185 s |
| `/usr/bin/time` real time | 72.34 s |
| Load Gemma | 1.7438 s |
| Prompt processing | 1.1912 s |
| Load transformer | 3.8126 s |
| Load decoders | 0.0502 s |
| Decode and mux | 11.9682 s |
| Post-transformer, pre-decoder interval | approximately 53.099 s |

The approximately 53.099-second interval was not subdivided by the instrumentation
available for this run. It must remain classified as unassigned
post-transformer/pre-decoder time; this report does not assume that the entire
interval was denoising.

This gap motivated finer `DistilledPipeline` phase instrumentation for future
trials. That later instrumentation must not be used to retroactively manufacture
phase timings for this run.

### Memory

| Measurement | Observed value |
|---|---:|
| Peak MLX memory | 40.3603609 GiB |
| Maximum RSS | 37,271,191,552 bytes |
| Peak process footprint | 49,173,459,768 bytes |
| Swap activity | Zero swaps observed |

The MLX peak, process RSS, and process footprint are distinct measurements and
should not be treated as interchangeable. The raw byte values above are retained
instead of relying on rounded unit conversions.

### Output validation

| Field | Observed value |
|---|---|
| Container size | 1,555,779 bytes |
| Video codec | H.264 |
| Video dimensions | 768 × 512 |
| Video frames | 129 |
| Video frame rate | 24 fps |
| Video duration | 5.375 s |
| Audio codec | AAC |
| Audio format | 48 kHz stereo |
| Audio duration | 5.330 s |
| A/V duration skew | 45 ms |

The observed output geometry, frame count, frame rate, stream presence, and A/V
skew satisfy the structural checks for this trial. This report does not claim
perceptual equivalence to another implementation or model variant.

### Artifact hashes

The following SHA-256 identifiers were supplied with the run:

| Artifact | SHA-256 |
|---|---|
| Input image | `6bc60da42cdd2955ca96803ee21c336662853ecae1bfc9af12463eb72ac81236` |
| Output MP4 | `9ceef33d97dc24801dc184eeb6e9f825653fa12d3f8123f23d216d44e3ce395c` |
| Profile | `659709c2f45d8dbc2017dbafb4e1fc9dad2808be0a333e9e313f7a4a9eb6f3da` |

## Historical Context

Four historical AI Studio runs with the comparable 768×512, 129-frame,
distilled-eight-step workload completed in:

```text
66.6 s, 76.8 s, 71.9 s, 67.9 s
```

Their median is 69.9 seconds. The present 72.185-second profiler result falls
within that historical minimum-to-maximum range. This is a consistency
observation only: the historical runs were not collected as a controlled
benchmark series and cannot replace the cold and warm trials required by the
protocol.

## Warm Trial 1: Fine-Grained Instrumentation Validation

This was the first warm trial and a successful repeat of the same BF16 workload
after adding finer `DistilledPipeline` phase boundaries. Its additional purpose
was to validate the instrumentation and resolve the process-cold run's large
unclassified interval. It is one member of the opening control cohort, not an
optimization comparison.

### Timing and memory

| Measurement | Observed value |
|---|---:|
| Profiler elapsed time | 69.792485 s |
| `/usr/bin/time` real time | 69.95 s |
| Sum of measured phases | 69.058496 s |
| Unclassified time | 0.733989 s |
| Peak MLX memory | 40.3603609 GiB |
| Maximum RSS | 39,003,570,176 bytes |
| Peak process footprint | 49,171,935,936 bytes |
| Swap activity | Zero swaps observed |

### Fine-grained phases

| Phase | Observed time |
|---|---:|
| Load Gemma | 2.244833 s |
| Prompt processing | 1.122842 s |
| Load transformer | 1.204760 s |
| Load VAE encoder | 0.010421 s |
| Load latent upsampler | 0.009014 s |
| Prepare Stage 1 conditioning/state | 0.335210 s |
| Stage 1 half-resolution denoise | 19.321820 s |
| Latent upscale | 0.345950 s |
| Prepare Stage 2 conditioning/state | 0.330787 s |
| Stage 2 full-resolution denoise | 31.774556 s |
| Load decoders | 0.047409 s |
| Decode and mux | 12.310893 s |

The two measured denoise phases total 51.096376 seconds, approximately 73.2% of
the profiler elapsed time. Decode and mux accounts for approximately 17.6%.
These measurements identify where wall time is concentrated at the pipeline-phase
level. They do not establish which operation inside the DiT is responsible; that
requires an operator-level trace.

### Output and evidence

The repeat output was structurally identical to the first run and byte-identical,
with the same SHA-256:

```text
9ceef33d97dc24801dc184eeb6e9f825653fa12d3f8123f23d216d44e3ce395c
```

The preserved fine-grained profile SHA-256 is:

```text
3729fdcb8ca24cec7cff47689aa4749babd0d75acde9ca4c83f626427a1ea78e
```

The repeat confirms that the finer phase events can account for all but
0.733989 seconds of this run while preserving byte-identical output.

## Opening BF16 Control Cohort

All four trials completed successfully with zero observed swaps, peak MLX memory
of 40.3603609 GiB, structurally valid output, and byte-identical MP4 output with
SHA-256
`9ceef33d97dc24801dc184eeb6e9f825653fa12d3f8123f23d216d44e3ce395c`.

| Trial | Cache classification | Profiler | `/usr/bin/time` real | Phase sum | Maximum RSS | Peak process footprint | Profile SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| Original | Process-cold, broad instrumentation | 72.185077625 s | 72.34 s | Not comparable | 37,271,191,552 B | 49,173,459,768 B | `659709c2f45d8dbc2017dbafb4e1fc9dad2808be0a333e9e313f7a4a9eb6f3da` |
| Warm 1 | Warm, fine-grained instrumentation | 69.792485 s | 69.95 s | 69.058496 s | 39,003,570,176 B | 49,171,935,936 B | `3729fdcb8ca24cec7cff47689aa4749babd0d75acde9ca4c83f626427a1ea78e` |
| Warm 2 | Warm, fine-grained instrumentation | 70.462803 s | 70.66 s | 69.717102 s | 39,025,655,808 B | 49,167,135,496 B | `19f9667fe7ae88a592a4455dce46f056215f975ca6fa849a8e82ef6b59f985a0` |
| Warm 3 | Warm, fine-grained instrumentation | 67.590930 s | 67.78 s | 66.882562 s | 38,914,916,352 B | 49,150,554,960 B | `497279c6da56b9e56584ac7b1a3d34cd4a1e76f786f7b23baac506c8e859bfc2` |

### Warm timing distribution

| Measurement | Result |
|---|---:|
| Profiler median | 69.792485 s |
| Profiler minimum | 67.590930 s |
| Profiler maximum | 70.462803 s |
| Profiler MAD | 0.670318 s |
| Stage 1 denoise median | 19.113536 s |
| Stage 2 denoise median | 31.774556 s |
| Decode/mux median | 12.310893 s |

The three warm phase measurements were:

| Warm trial | Stage 1 denoise | Stage 2 denoise | Decode/mux |
|---|---:|---:|---:|
| Warm 1 | 19.321820 s | 31.774556 s | 12.310893 s |
| Warm 2 | 19.105707 s | 32.336398 s | 12.639227 s |
| Warm 3 | 19.113536 s | 30.863152 s | 11.197417 s |

The warm medians describe this opening cohort only. They do not identify the cause
of time inside the DiT, predict another workload, or demonstrate an optimization.
Operator-level attribution still requires an operator trace.

## Conclusion

The opening BF16 control cohort is complete: one process-cold observation and
three valid warm controls produced the same audiovisual bytes without swaps or
structural failures. This supplies the opening comparison distribution for the
pinned workload.

No candidate is accepted by this report. Acceptance still requires:

1. A comparable process-cold and warm candidate cohort that changes one variable.
2. The protocol's performance, memory, output-parity, and failure gates.
3. A closing BF16 control to detect environmental drift.
4. Complete evidence defined in `AI_STUDIO_MLX_BENCHMARK_PROTOCOL.md`.
