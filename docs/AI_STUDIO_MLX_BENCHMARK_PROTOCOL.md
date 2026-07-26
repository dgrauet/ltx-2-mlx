# AI Studio MLX Benchmark Protocol

## Purpose

This protocol measures the ordinary AI Studio LTX-2.3 image-to-video path before
and after MLX optimizations. It is designed to distinguish repeatable improvements
from model-cache effects, machine load, output drift, and unsafe reductions in
Metal synchronization.

The BF16 result is the control. Quantization, block streaming, TeaCache, tiling,
persistent execution, compilation, and custom kernels are separate experiments.
Do not combine experiments until each component has passed independently.

This document defines a protocol; it does not predict that any candidate will be
faster, use less memory, or preserve quality.

## Safety Rules

LTX-2.3 is a heavy unified-memory workload. Two overlapping renders, including an
orphan from a previous AI Studio process, can exhaust wired GPU memory and cause a
kernel panic rather than a recoverable out-of-memory error.

Before every trial:

1. Confirm the AI Studio render queue is idle and no restart is pending.
2. Use AI Studio's normal heavy-render ledger and orphan reconciliation. Do not
   start a direct diagnostic command while AI Studio is serving another job.
3. Confirm that exactly zero LTX render processes are active before dispatch.
   Record the process snapshot in the trial manifest.
4. Do not weaken `LTX2_GEMMA_EVAL_EVERY` or `LTX2_DIT_EVAL_EVERY`. Record their
   effective values; use the safe defaults when they are unset.
5. Run only one heavy local render at a time. Do not run Spotlight indexing,
   another ML workload, a game, video export, or a benchmark in parallel.
6. Require adequate free disk space for logs, the complete MP4, and failure
   artifacts. Do not delete diagnostics during a benchmark series.
7. After a failed trial, stop the series and reconcile all descendants before
   another render. Never assume that a failed wrapper implies its MLX child exited.

The preferred harness submits the workload through AI Studio's serialized `i2v3`
queue. A direct invocation of `ltx23_mlx_single.py` is permissible only in a
dedicated maintenance window after the same idle/orphan checks.

## Fixed Workload

Every experiment uses the following workload:

| Field | Required value |
|---|---|
| AI Studio lane | Ordinary `i2v3` |
| Conditioning | One primary source image; no additional keyframes |
| Width × height | `768 × 512` |
| Frames | `129` |
| Frame rate | `24 fps` |
| Mode | Distilled |
| Denoising steps | `8` |
| Seed | `424242` |
| Text encoder | Local |
| LoRAs | None |
| Prompt Relay | Disabled |
| Hidden conditioning | Disabled |
| Audio | Enabled; use AI Studio's normal prompt/audio policy |

Freeze one legally owned, representative source image and one prompt before the
first control run. Store their exact UTF-8 prompt text and SHA-256 hashes in the
series manifest. Do not resize, recompress, rotate, color-convert, or replace the
source between experiments.

The AI Studio-equivalent wrapper invocation is:

```bash
"/path/to/ai-studio-python" "/path/to/ltx23_mlx_single.py" \
  --image "/path/to/frozen-source.png" \
  --prompt "$(cat /path/to/frozen-prompt.txt)" \
  --output "/path/to/artifacts/<trial-id>/output.mp4" \
  --width 768 \
  --height 512 \
  --frames 129 \
  --fps 24 \
  --steps 8 \
  --distilled \
  --seed 424242 \
  --text-encoder local \
  --job-status "/path/to/artifacts/<trial-id>/status.json"
```

The harness must retain the final, fully expanded argument vector as JSON. Shell
history or the example above is not sufficient provenance.

## Machine and Artifact Provenance

Create a series manifest before the cold trial. It must contain:

- UTC start time, local time zone, operator, and series identifier.
- Mac model, Apple Silicon generation, CPU/GPU core counts, and installed RAM.
- macOS build, kernel version, uptime, power source, and Low Power Mode state.
- Repository commit, branch, dirty/clean state, and a patch artifact if dirty.
- SHA-256 of `uv.lock` and the resolved versions of Python, MLX, `mlx-lm`,
  `ltx-core-mlx`, `ltx-pipelines-mlx`, ffmpeg, and ffprobe.
- Model repository identifier and revision plus the SHA-256 of every loaded model
  config and weight shard. A Hugging Face model name alone is insufficient.
- Source image SHA-256, dimensions, pixel format, color metadata, and byte size.
- Exact prompt bytes and SHA-256.
- Exact argv and a whitelist of relevant environment variables, including all
  `LTX2_*`, `MLX_*`, and Metal diagnostic variables.
- Experiment name and the single intended difference from the BF16 control.
- Hashes for each output, status file, stdout/stderr log, metrics file, and
  diagnostic artifact after the trial completes.

Use SHA-256 consistently. Never overwrite a trial directory or mutate an artifact
after hashing it. Corrections produce a new trial identifier.

## Trial Order and Cache States

Each experiment consists of one cold trial followed by at least three warm trials.
Use more warm trials when variability is high.

### Cold trial

A cold trial starts from a fresh render process with no active or adopted LTX
worker. Do not use `purge` or other artificial cache eviction. OS file-cache state
is not controllable enough to infer; instead record uptime, prior LTX activity,
memory pressure, and whether model files are likely resident. Label this
`process-cold`, not `disk-cold`.

### Warm trials

Warm trials follow the cold trial without rebooting or purging caches. For the
current AI Studio wrapper, each trial still creates a fresh Python/MLX process, so
“warm” primarily describes OS model-page cache and stable machine state. For a
persistent-worker experiment, record both first-request and subsequent-request
timings separately.

Between trials:

1. Confirm the previous process tree is gone.
2. Wait until system memory pressure returns to its pre-series class and remains
   stable for 30 seconds.
3. Record the idle memory/process snapshot.
4. Keep the same power state and foreground application load.
5. Do not discard an outlier merely because it is slow. Mark an exclusion only
   for a documented external disturbance, and retain the original evidence.

Compare warm medians. Report minimum, maximum, median, median absolute deviation
(MAD), and every individual value. Cold results are reported separately and never
pooled into the warm median.

## Measurements

Use a monotonic clock. All samplers must timestamp records against the same trial
origin.

### Wall and phase time

Measure end-to-end wall time from immediately before wrapper dispatch through
successful close and fsync of the final output. Instrument these phases where the
pipeline exposes boundaries:

1. Process startup and argument validation.
2. Model/config discovery and weight loading.
3. Prompt encoding and connector projection.
4. Source-image preparation and VAE encode.
5. Transformer preparation/compilation.
6. Denoising, including time for every step.
7. Video VAE decode.
8. Audio VAE, vocoder, and bandwidth extension.
9. ffmpeg encode/mux and final file close.
10. Cleanup.

Phase totals must reconcile with end-to-end wall time; report unclassified time
explicitly rather than assigning it to a nearby phase.

### MLX memory

Capture `mx.get_active_memory()`, `mx.get_peak_memory()`, and
`mx.get_cache_memory()`:

- At process startup.
- Before and after every phase above.
- Before and after each explicit cleanup.
- At every denoising step boundary.
- Immediately before process exit.

Reset the MLX peak counter only once, before measured work begins. Preserve both
the MLX-reported peak and the sampled time series.

### Process and system memory

Sample at least every 250 ms:

- PID, parent PID, command, elapsed time, RSS, virtual size, and exit state for
  the wrapper and every descendant.
- Sum of descendant RSS, while retaining per-process values.
- `vm_stat` counters and system memory-pressure class.
- Swap usage and page-in/page-out deltas.

Process RSS and MLX active memory measure different things and must not be
substituted for one another. Report both. Do not use `powermetrics` unless the
benchmark environment has explicitly authorized its privileges; if used, record
its command and sampling interval.

### Output parity

For every successful trial, verify with ffprobe:

- The file opens without decode errors.
- Video is exactly `768 × 512`, 129 frames, and 24 fps.
- Duration is consistent with 129 frames at 24 fps.
- Both video and audio streams are present.
- Timestamps are monotonic and A/V duration skew is no more than 50 ms.

Retain decoded-frame and audio comparisons against the BF16 control:

- Source-conditioning check on frame zero using pixel error and SSIM.
- Per-frame SSIM and PSNR over a deterministic RGB decode.
- A temporal-difference comparison to detect frozen, duplicated, or reordered
  frames.
- Audio duration, sample rate, channel count, peak level, RMS, clipping count,
  and sample-aligned error where deterministic parity is expected.
- Side-by-side, blinded human review for identity/scene preservation, motion,
  flicker, temporal coherence, lip/action synchronization when applicable, and
  new visual or audio artifacts.

Perceptual metrics are evidence, not a substitute for review. Quantization and
cache approximations may produce acceptable output that is not pixel-identical.

## Failure and Crash Capture

A trial is failed if the wrapper or any descendant exits abnormally, stalls past
the declared timeout, produces an invalid output, or leaves a render process
behind.

Retain:

- Expanded argv, environment whitelist, start/end timestamps, exit code, and
  terminating signal.
- Untruncated stdout and stderr from the wrapper and all children.
- AI Studio status JSON and queue/ledger snapshot.
- Final process tree and the full memory time series.
- The last completed phase, denoising step, and MLX memory snapshot.
- Relevant macOS unified-log entries for Metal, IOGPU, memory pressure, jetsam,
  watchdog, and the affected PIDs over the trial window.
- Any `.ips`, panic, spin, or crash report generated for the trial.
- A partial output and ffprobe/decode report, if one exists.

On a stall, capture a non-destructive process sample before termination. Terminate
through AI Studio's normal SIGTERM grace period followed by its scoped SIGKILL
fallback. Never kill unrelated Python or MLX processes by broad name.

A kernel panic invalidates the experiment and is a hard stop. After reboot, archive
the panic report and do not resume until the overlap/orphan cause has been audited.

## Acceptance and Rejection

These are qualification thresholds, not predicted results.

### Universal gates

A candidate is rejected if any of the following occurs:

- Any crash, panic, Metal watchdog error, abnormal signal, hang, orphan, or
  unreconciled child.
- Fewer than one cold and three valid warm trials.
- Wrong dimensions, frame count, frame rate, missing audio/video stream,
  non-monotonic timestamps, decode error, or A/V skew above 50 ms.
- Source-conditioning failure, frozen/duplicated frames, or a new material visual
  or audio defect in blinded review.
- Peak system memory or memory pressure reaches a less-safe state than the control
  without a documented reason and explicit risk acceptance.
- More than 5% descendant-RSS growth across five identical sequential requests in
  the persistent-worker experiment after returning to idle.

For a performance claim, the warm median must improve by at least 5%, and the
improvement must exceed twice the larger MAD of control and candidate. Otherwise
report “no demonstrated speed improvement.” A regression above 5% is a performance
failure unless the experiment's declared purpose is memory reduction.

For a memory claim, both MLX peak memory and peak descendant RSS must improve by at
least 10%, and the candidate must not worsen warm median wall time by more than
10%, unless a candidate-specific threshold below says otherwise. Smaller changes
are reported as inconclusive.

Numerically intended-to-be-equivalent changes must meet the implementation's
documented tensor tolerances and produce effectively identical decoded output.
Approximate changes such as q8 and TeaCache additionally require blinded review.
Do not invent a single SSIM/PSNR cutoff before observing normal BF16 repeatability;
freeze quantitative parity thresholds from the BF16 repeatability study before
opening candidate results.

## Ordered Experiment Matrix

Run the matrix in this order. Restore the control configuration before each row.

| Order | Experiment | Only intended change | Candidate-specific acceptance |
|---:|---|---|---|
| 0 | Current BF16 control | None: `dgrauet/ltx-2.3-mlx`, current safe synchronization defaults | Establish valid control and freeze variance/parity thresholds |
| 1 | q8 | Model changes to `dgrauet/ltx-2.3-mlx-q8` | Universal gates; qualify speed only under the performance rule or memory only under the memory rule |
| 2 | Low RAM | Add `--low-ram` to BF16 control | At least 20% lower MLX peak and descendant RSS; no more than 15% warm-wall regression |
| 3 | TeaCache 0.5 | Enable TeaCache with threshold exactly `0.5` on BF16 | At least 5% warm-wall improvement; memory no worse than 5%; approximate-output review passes |
| 4 | Tiling | Enable one predeclared tiling configuration on BF16; record frame/spatial tile counts and overlap | At least 20% lower MLX peak; no crash; no more than 50% warm-wall regression; tile-boundary review passes |
| 5 | Persistent worker | Reuse the loaded BF16 pipeline; model/math unchanged | At least 10% lower subsequent-request median wall time; idle RSS growth no more than 5% across five requests |
| 6 | Compile candidates | One compilation boundary or graph change at a time | At least 5% end-to-end or 10% target-phase improvement; memory no worse than 5%; numerical-equivalence gates pass |
| 7 | Custom Metal kernel candidates | One operator/kernel replacement at a time | At least 5% end-to-end or 15% target-operator improvement; memory no worse than 5%; numerical-equivalence and watchdog gates pass |

Tiling is a memory experiment at this baseline size, not a presumed speed
optimization. Persistent-worker, compilation, and custom-kernel rows require an
implementation before they can be measured; until then they remain candidate
descriptions, not findings.

After any candidate passes, rerun the unchanged BF16 control. If the closing
control differs from the opening control by more than 5% or their medians differ
by more than twice the larger MAD, treat the series as environmentally drifted and
repeat it.

## Reporting

The benchmark report must include:

1. Series manifest and immutable artifact index.
2. Opening and closing BF16 controls.
3. Per-trial wall, phase, memory, pressure, and parity results.
4. Cold results separate from warm distributions.
5. All failures and excluded trials with reasons.
6. Candidate decision: accepted, rejected, or inconclusive.
7. The narrow evidence-backed claim, such as “reduced peak MLX memory under this
   workload,” rather than a general claim about all LTX workloads or Apple chips.

Never extrapolate this single 768×512×129 workload to longer clips, 1080p,
two-stage generation, additional conditioning, LoRAs, or other Apple Silicon
systems without running their own pinned protocols.
