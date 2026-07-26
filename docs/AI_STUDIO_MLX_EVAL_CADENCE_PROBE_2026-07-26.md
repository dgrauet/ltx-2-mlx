# AI Studio MLX DiT Eval-Cadence Probe — 2026-07-26

## Decision

**Rejected: no demonstrated material gain. Keep the watchdog-safe default DiT
evaluation cadence of 8 blocks.**

The candidate cadence of 16 blocks produced byte-identical output and a 1.3524%
lower measured denoising sum than its matched default-cadence control. Its
end-to-end runtime was nevertheless 0.9577% slower. This isolated pair does not
demonstrate a useful performance improvement and is far too small to establish
watchdog safety under contention.

The benchmark protocol normally forbids changing watchdog evaluation cadence.
This was a narrowly scoped diagnostic maintenance probe with a matched control,
not a proposal to relax production safety.

## Fixed Workload

Both runs used the canonical AI Studio-equivalent BF16 workload and the same
hardware, model, source image, prompt, seed, and process lifecycle:

| Field | Value |
|---|---|
| Model | `dgrauet/ltx-2.3-mlx` |
| Pipeline | Distilled image-to-video |
| Geometry | 768 × 512 |
| Frames | 129 |
| Frame rate | 24 fps |
| Denoising schedule | 8 + 3 steps |
| Seed | 424242 |
| Low-RAM streaming | Disabled |
| Candidate change | `LTX2_DIT_EVAL_EVERY=16` |
| Matched control | Default `LTX2_DIT_EVAL_EVERY=8` |

Both runs completed successfully with zero observed swaps.

## Preserved Evidence

The compact JSONL profiles are preserved under
[`docs/benchmarks/ai-studio-mlx-2026-07-26/`](benchmarks/ai-studio-mlx-2026-07-26/).
The generated MP4 files are not stored in this repository.

| Run | Output path during probe | Preserved profile | Profile SHA-256 |
|---|---|---|---|
| Candidate, cadence 16 | `/tmp/ltx-mlx-profile.nsGZ9V/bf16_eval16_probe.mp4` | [`bf16_eval16_probe.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/bf16_eval16_probe.jsonl) | `76f8fe0b10594fe44c3498087ecc83cbb27aa1506cd66387f246a91a83c1588a` |
| Matched control, cadence 8 | `/tmp/ltx-mlx-profile.nsGZ9V/bf16_eval8_matched_control.mp4` | [`bf16_eval8_matched_control.jsonl`](benchmarks/ai-studio-mlx-2026-07-26/bf16_eval8_matched_control.jsonl) | `62cba054eb43f08e70ee843cd53c363886dcaca9a3954f061911b33ae0ee55ac` |

## Measurements

### Matched results

| Measurement | Default cadence 8 | Candidate cadence 16 | Observation |
|---|---:|---:|---|
| Total profiler elapsed | 61.429360584 s | 62.017714125 s | Candidate 0.9577% slower |
| Stage 1 denoise | 16.683498583 s | 16.664831417 s | Candidate 0.018667166 s lower |
| Stage 2 denoise | 27.881221958 s | 27.297189542 s | Candidate 0.584032416 s lower |
| Denoise sum | 44.564720541 s | 43.962020959 s | Candidate 1.3524% lower |
| Decode and mux | 11.015567833 s | 10.844566583 s | Candidate 0.171001250 s lower |
| Peak MLX memory | 40.360360797 GiB | 40.360361572 GiB | No material reduction |
| Swaps | Zero | Zero | No difference observed |

The reported intervals were calculated directly from the preserved phase
timestamps:

```text
cadence 16 stage 1: 22.88613041699864 - 6.221298999967985
                    = 16.664831417030655 s
cadence 16 stage 2: 50.818744291958865 - 23.521554749982897
                    = 27.297189541975968 s
cadence 16 decode:  61.99748754198663 - 51.15292095899349
                    = 10.84456658299314 s

cadence 8 stage 1:  21.550850500003435 - 4.867351917026099
                    = 16.683498582977336 s
cadence 8 stage 2:  50.08568787499098 - 22.20446591702057
                    = 27.881221957970522 s
cadence 8 decode:   61.409312000032514 - 50.39374416699866
                    = 11.015567833033856 s
```

### Output parity

Both runs produced byte-identical output with SHA-256:

```text
9ceef33d97dc24801dc184eeb6e9f825653fa12d3f8123f23d216d44e3ce395c
```

This establishes deterministic parity for this workload and pair of runs. It does
not establish watchdog safety across machine load, longer clips, larger
resolutions, or other Apple Silicon systems.

## Interpretation

Doubling the number of blocks between explicit DiT evaluations reduced the
measured denoising sum by 0.602699582 seconds, but that difference did not survive
at the request boundary: the candidate took 0.588353541 seconds longer overall.

No operator-level trace was collected, so this report does not attribute either
difference to graph construction, command-buffer scheduling, queue contention,
or a particular DiT operator. A single matched pair is also insufficient to
separate normal run variance from cadence effects.

The explicit evaluation boundaries exist to keep Metal command buffers within the
macOS interactivity watchdog window. An isolated successful run at cadence 16
does not override that safety evidence. Since the candidate showed no material
end-to-end gain, there is no reason to spend the additional stability risk budget.

## Conclusion

Retain `LTX2_DIT_EVAL_EVERY=8` as the production default. The cadence-16 candidate
is rejected for this optimization effort because:

1. Total runtime was 0.9577% slower than the matched default control.
2. Peak MLX memory was materially unchanged.
3. The small denoising difference was not demonstrated across repeated trials.
4. Watchdog safety under macOS contention was not and should not be inferred from
   this isolated diagnostic.

No broader eval-cadence experiment is justified unless a future operator trace
shows synchronization overhead large enough to exceed normal variance and to
warrant a dedicated contended stability campaign.
