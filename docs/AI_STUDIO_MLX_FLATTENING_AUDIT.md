# AI Studio MLX flattening audit

Observed 2026-07-26. This audit covers the ordinary AI Studio `i2v3`
LTX-2.3 path and the MLX libraries it invokes. It does not cover the retired
`i2va` path, the PyTorch ID-LoRA path, or other applications.

## Executive finding

AI Studio is already using the native MLX implementation for ordinary
LTX-2.3 generation:

`ltx-pipelines-mlx` → `ltx-core-mlx` → MLX → MLX Metal kernels → Apple Metal.

The main avoidable layering is above MLX. Every queued generation starts a
Python wrapper, which starts a second Python CLI process, which constructs and
destroys the pipeline. Python is not doing the 22B tensor arithmetic
element-by-element; MLX and Metal are. Removing Python alone therefore will not
remove the dominant Gemma, DiT, VAE, vocoder, or attention work. It can,
however, remove process launch and stdout-parsing layers, reduce repeated model
load/mmap/free churn, enable shape-aware compilation caches, and give AI Studio
much better control over memory and crash recovery.

The shortest evidence-backed path to the requested “two layers” is:

1. AI Studio as the control plane.
2. One persistent MLX engine process as the inference data plane.

Metal and the media encoder remain system runtimes rather than application
orchestration layers. A later Swift/C++ host can replace the persistent Python
worker, but it should have to beat that baseline before the much larger port is
authorized.

## Evidence labels

- **Fact** — directly observed in the pinned local source, runtime, cache, or
  checked-in test records listed below.
- **Inference** — a conclusion from those facts that has not yet been measured
  in AI Studio.
- **Hypothesis** — a candidate optimization that requires a benchmark or trace.

No current AI Studio crash was reproduced as part of this source audit. The
document deliberately distinguishes historical crash causes from the cause of
any crash happening on the current machine.

## Observation baseline

### Source revisions

| Item | Observed revision/state |
|---|---|
| `ltx-2-mlx` | clean worktree, branch `codex/ai-studio-mlx-profile`, commit `36ab612919ccba4b90321019a2927633dd837d35` (`ci: bump astral-sh/setup-uv in the github-actions group (#73)`, 2026-07-14) |
| AI Studio repository | branch `structured-prompts-2026-07-11`, commit `2f0d806bacfc798d933281e049a85260a1479d09` (`feat(queue): record WHO queued each job (initiated_by)`, 2026-07-25), 86 commits ahead of its configured upstream |
| AI Studio worktree | dirty at observation time; the line references in this audit describe the observed worktree, not necessarily the commit above |

Because the AI Studio files used by this audit were not all clean, these
SHA-256 fingerprints pin the exact inspected contents:

| File | SHA-256 |
|---|---|
| `ai_studio/app.py` | `4de3d7475a46b5c22b5e6c2430a1bb17f612ff648901b8f3064777bdc7bdf3d0` |
| `ai_studio/jobs.py` | `5a803041f7a7e95bd0e629e275547f649048f777695236095bf16ad40974425c` |
| `ai_studio/config.py` | `77bf0a6da98d6616cf1358901391552e80facc27d9f1795984072fff8da5124e` |
| `ai_studio/static/app.js` | `43c84dbf0d9399dcf6b1602477f5451be1b18bdce7bc3b878ec47c3a63344183` |
| `ai_studio/templates/index.html` | `77ef4c3fe1351a429d6083cdd41b50cd4d5939c49297a3b5b6ae4749931c7490` |
| `ltx23_mlx_single.py` | `b152501019fe281481612f2b9f4e060ade2a3f8533bb3600606b8ee3b90efe50` |

### Runtime and model revisions

| Item | Observed value |
|---|---|
| Hardware | Apple M5 Max, 128 GiB unified memory, arm64 |
| OS | macOS 26.5.2, build 25F84 |
| AI Studio configured job Python | `/Users/paulericksen/Documents/ComfyUI/.venv/bin/python`, CPython 3.12.11 |
| MLX engine Python | `/Users/paulericksen/video-models/ltx-2-mlx/.venv/bin/python`, CPython 3.11.15 |
| `ltx-core-mlx` / `ltx-pipelines-mlx` | editable local packages, version 0.14.18 |
| `mlx` / `mlx-metal` | 0.31.1 / 0.31.1 |
| `mlx-lm` / `mlx-arsenal` | 0.31.1 / 0.2.4, as locked in `uv.lock` |
| AI Studio ordinary model | `dgrauet/ltx-2.3-mlx` (BF16), snapshot `baa5f235ea04fd9c95899d751295c4fd825ee4e2` |
| Text encoder | `mlx-community/gemma-3-12b-it-4bit`, snapshot `86cc6a8dedbc456dd0e4af01a9d09f396f77e558` |

The model snapshot identifies itself as LTX 2.3.0 `AudioVideo`, with 48
transformer layers, 32 video attention heads × 128 dimensions, and 32 audio
heads × 64 dimensions. Its split manifest includes the distilled and dev
transformers, connector, video VAE encoder/decoder, audio VAE, vocoder,
spatial/temporal upscalers, and distilled LoRA; its source is
`Lightricks/LTX-2.3`.

AI Studio explicitly overrides the CLI's q8 default with the BF16
`dgrauet/ltx-2.3-mlx` repository. The library documentation estimates the
complete BF16 variant at about 42 GB, q8 at about 26 GB, and q4 at about 12 GB.
Those are model-footprint estimates, not measured peak memory for this AI
Studio worktree or this M5 Max.

## Exact ordinary `i2v3` call path

```text
Browser UI
  ai_studio/static/app.js + ai_studio/templates/index.html
        │ HTTP POST /api/queue
        ▼
AI Studio Flask process
  ai_studio/app.py → jobs.add_job(...)
        │ queue worker constructs argv
        ▼
Fresh Python 3.12 process per job
  ltx23_mlx_single.py
        │ subprocess + stdout/status translation
        ▼
Fresh Python 3.11 process per job
  .venv/bin/ltx-2-mlx generate ...
        │ console-script shim
        ▼
ltx_pipelines_mlx.cli.main()
  DistilledPipeline.generate_and_save(...)
        ▼
ltx-pipelines-mlx orchestration
  Gemma → image VAE encode → DiT stage 1 → upscale → DiT stage 2
  → audio VAE/vocoder → video VAE → ffmpeg
        ▼
ltx-core-mlx → MLX 0.31.1 → MLX Metal kernels → Apple Metal
```

Evidence:

- The UI labels LTX-2.3 as “MLX native · default” and defaults ordinary
  generation to the distilled 8-step mode
  (`ai_studio/templates/index.html:3131-3139,3190-3199,3230-3238`).
- `POST /api/queue` delegates to `jobs.add_job`
  (`ai_studio/app.py:428-435`).
- `i2v3` is an accepted video job class
  (`ai_studio/jobs.py:188-192,7731-7733`).
- The job worker maps `i2v3` to `ltx23_mlx_single.py`, builds its arguments,
  and launches it with `subprocess.Popen`
  (`ai_studio/config.py:42,47-50`;
  `ai_studio/jobs.py:2912-2951,4020-4046`).
- The wrapper hard-codes the separate
  `/Users/paulericksen/video-models/ltx-2-mlx/.venv/bin/ltx-2-mlx` executable
  and `dgrauet/ltx-2.3-mlx`, then launches the CLI as another subprocess
  (`ltx23_mlx_single.py:39-45,83-119,304-311,371`).
- The console script is a Python shim for `ltx_pipelines_mlx.cli:main`
  (`packages/ltx-pipelines-mlx/pyproject.toml:32-33`).
- `--distilled` constructs a fresh `DistilledPipeline` with
  `low_memory=True`, optional `low_ram_streaming`, and invokes
  `generate_and_save`
  (`packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/cli.py:725-758`).

The process boundaries do **not** copy model tensors between processes. Only
arguments, progress text, paths, and files cross them; the inner CLI process
owns all MLX tensors. Their main costs are interpreter/import startup,
stdout/status translation, repeated model lifecycle work, and loss of a
persistent compilation/model cache.

The wrapper passes the user-facing negative prompt to neither CFG nor the MLX
pipeline in distilled mode. That is intentional for this flow:
`DistilledPipeline` performs no CFG and encodes only the positive prompt.

## What the distilled pipeline actually does

The apparent two-stage model flow is model architecture, not redundant
application layering. Flattening the processes does not make these stages
disappear without changing model behavior.

| Order | Component/stage | Material behavior |
|---:|---|---|
| 1 | Prompt encoder | Load 4-bit Gemma and connector; encode one prompt into separate video (4096-wide) and audio (2048-wide) embeddings; materialize them. |
| 2 | Memory transition | Free Gemma before loading the DiT. This is a deliberate peak-memory and watchdog mitigation. |
| 3 | Model load | Resolve/load the distilled 48-block LTX AudioVideo transformer, video VAE encoder, and 2× latent upsampler. |
| 4 | I2V conditioning A | Decode/preprocess the source image at half-resolution and VAE-encode it into conditioning tokens. T2V skips this. |
| 5 | Stage 1 | Construct half-resolution video/audio noise states and run the distilled Euler schedule: normally 8 DiT forwards, no CFG. |
| 6 | Latent upscale | Unpatchify video tokens, denormalize, run the 2× latent upsampler, renormalize, and materialize the result. |
| 7 | I2V conditioning B | Re-preprocess and re-encode the image at full-resolution because the stage-2 conditioning latent has a different spatial shape. |
| 8 | Stage 2 | Repatchify, add stage-2 noise, and run the same distilled DiT at full resolution: normally 3 forwards, no CFG. |
| 9 | Decode transition | Unpatchify final video/audio latents; free the DiT, encoder, and upsampler before loading decoders. |
| 10 | Audio decode | Audio VAE decoder → BigVGAN vocoder → bandwidth extension; materialize/copy a 48 kHz waveform to a temporary PCM WAV. |
| 11 | Video decode | Video VAE decode, automatically temporal-tiled above its memory budget; stream RGB frames to ffmpeg. |
| 12 | Encode/mux | ffmpeg encodes H.264/yuv420p CRF 18, encodes audio to AAC, and writes the final MP4. |

The stage implementation is explicit in
`packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/distilled.py:144-333`.
The pipeline uses 8+3 single-forward denoising steps by default
(`docs/PIPELINES.md:18-20,52,70`). This is already the cheapest of the ordinary
two-stage LTX flows: the dev two-stage path normally performs CFG and therefore
multiple model forwards per step.

## Optimizations already present

### MLX-native execution

**Fact.** The transformer, text encoder, VAE, audio decoder, vocoder, latent
upsampler, schedulers, and conditioning math are MLX arrays/modules. Python
orchestrates the graph, but MLX executes the tensor operations through its
compiled backend and Metal kernels.

### Stage-aware component lifetime

**Fact.** `low_memory=True` is always set by the AI Studio CLI path. Gemma is
materialized and freed before the DiT; the DiT and encoders are freed before
the decoders; decoders are freed after saving
(`distilled.py:147-165`;
`ti2vid_two_stages.py:653-676`). This prevents simultaneous residency of the
largest component groups.

### Optional transformer block streaming

**Fact.** AI Studio exposes `--low-ram`, but only passes it when the user
selects that option (`ltx23_mlx_single.py:109-119`). With it enabled:

- MLX's cache limit is set to zero.
- Safetensors are memory-mapped.
- One shared transformer block is rebound for all 48 layers.
- The shared block is compiled using `mx.compile(shared, inputs=shared)`.
- The previous block's mapped arrays are evicted as the next block binds.
- Each block is evaluated separately.

The checked-in profile reports roughly 10–12 GB → 2.76 GB transformer Metal
peak on q8, at about a 5% time cost, and byte-identical q8 output in its
regression pair (`CLAUDE.md:758-814`;
`docs/REGRESSION_TESTS_v0.11.0.md:68-71`). Those measurements were not made on
the current AI Studio BF16/M5 Max run and must not be presented as its measured
peak.

### Lazy-graph/watchdog barriers

**Fact.** The current code intentionally breaks large lazy graphs into bounded
Metal command buffers:

- Gemma: `mx.eval` every layer by default
  (`LTX2_GEMMA_EVAL_EVERY=1`).
- Text projection: an eval between video and audio output projections.
- Connector: an eval per block.
- DiT: an eval every eight blocks by default
  (`LTX2_DIT_EVAL_EVERY=8`).
- Prompt output: explicit materialization before Gemma is freed.
- Pre-denoise state: explicit materialization before each denoise loop.
- Denoise loop: `mx.async_eval` after each Euler step.
- VAE tiles and accumulation buffers: explicit materialization between tiles.
- Output frame: explicit materialization before its memory is read by ffmpeg.

These are synchronization costs, but they are not accidental Python
inefficiencies. They were added after real Metal watchdog failures
(`CLAUDE.md:928-954`; `_base.py:503-517`; `utils/samplers.py:135-174`).
Removing all of them is not a safe optimization.

### Streaming/tiled media decode

**Fact.** Video VAE decode switches to temporal tiling when its estimated
block-3 activation would exceed `LTX2_VAE_DECODE_BUDGET_GB` (8 GB by default),
then streams frames into ffmpeg instead of retaining all decoded frames
(`ltx_core_mlx/model/video_vae/video_vae.py:300-523`).

### Optimizations present in the library but not active in this path

- **Modality tiling:** the CLI supports temporal/spatial DiT tiling, but the AI
  Studio wrapper does not pass the tile flags. The library warns that overhead
  can exceed the benefit at normal token counts and recommends it mainly for
  1080p or clips of roughly eight seconds and longer.
- **TeaCache:** supported by the guided dev two-stage samplers, not the
  distilled flow used here. Enabling a flag in AI Studio would not make it
  applicable to `DistilledPipeline`.
- **q8/q4 models:** supported and materially smaller, but AI Studio currently
  hard-codes BF16.

## Synchronization and copy audit

Apple Silicon unified memory makes “CPU tensor” versus “GPU tensor” less like
a discrete-GPU system. MLX arrays can be consumed by CPU and GPU without a
traditional PCIe transfer, but materialization, layout conversion, new host
buffers, subprocess pipes, and codec boundaries still cost time and memory.

| Boundary | Current behavior | Classification |
|---|---|---|
| Job process → wrapper process → CLI process | strings, JSON/status text, paths, and files only; no tensors | avoidable orchestration, not a tensor copy |
| Safetensors → MLX | mmap/lazy arrays; eager mode touches the full transformer, low-RAM mode binds one block at a time | required weights; residency policy is tunable |
| Input image file → Pillow/NumPy | decode, resize/crop, and H.264 CRF-33 round trip for training-distribution alignment | host media work; intentional quality behavior |
| NumPy input image → `mx.array` | creates the MLX conditioning tensor | required host/MLX boundary |
| I2V half-res → full-res conditioning | source is preprocessed/VAE-encoded again at the second spatial shape | model-stage requirement; caching/preprocessing implementation may be tunable |
| MLX stage 1 → upsampler → stage 2 | MLX arrays and transposes remain in unified memory | no explicit host copy |
| MLX audio waveform → NumPy | `np.array(wav.astype(float32))` materializes/copies the complete waveform | candidate host copy |
| NumPy audio → temporary WAV | float32 → int16 → `tobytes()` → disk file | candidate buffer and disk copy |
| MLX video frame → ffmpeg | materialize uint8 HWC, then `bytes(memoryview(frame_hwc))`, then pipe write | at least one required encoder boundary plus a likely avoidable `bytes` copy |
| ffmpeg → MP4 | separate process performs H.264/AAC encode and mux | required codec work; process can later be replaced by VideoToolbox/AVFoundation |

The current video path is streaming, so it avoids an all-frames host buffer.
The audio path is not streaming. A native media path could remove ffmpeg and
the temporary WAV, but it will not remove the need to present decoded pixels
and samples to an encoder.

## Crash history: what is and is not established

### Retired PyTorch/MPS path

**Fact.** AI Studio's wrapper and UI document that its retired PyTorch/MPS
LTX-2.3 lane leaked roughly 10–20 GB per sampling step and died around steps
5–6 (`ltx23_mlx_single.py:1-22`;
`ai_studio/static/app.js:8744-8757`). That is evidence about the retired
PyTorch/MPS implementation, not evidence that the current MLX path has the
same leak.

### Historical MLX watchdog failures

**Fact.** This MLX repository records
`kIOGPUCommandBufferCallbackErrorImpactingInteractivity` and
`MTLCommandBufferErrorInternal` code 14 failures caused by oversized lazy
command buffers and by command-buffer contention with macOS services. It also
records a concrete heap-thrash bug: some pipelines encoded/freed Gemma, then
loaded/freed the 7.5 GB Gemma mapping again immediately before loading the
roughly 10 GB q8 DiT. Commit `1a30f74` removed that duplicate load. Strategic
eval barriers and the lifecycle fix passed a production-step cohort under
contention on an M2 Pro 32 GB
(`docs/REGRESSION_TESTS_v0.11.0.md:115-160`;
`CLAUDE.md:928-954`).

### Current AI Studio crashes

**Unknown.** This audit has not established whether current failures are:

- macOS jetsam/SIGKILL under system-wide memory pressure;
- a Metal watchdog/internal error;
- BF16 eager-model residency combined with another resident workload;
- VAE decode accumulation pressure on long/high-resolution output;
- an MLX/Metal defect on the pinned M5 Max/macOS/MLX combination;
- ffmpeg/media failure; or
- an application-level cancellation/process-management bug.

The queue code contains memory-based failure heuristics, but a heuristic is not
a root-cause trace. Do not optimize custom kernels against an unclassified
crash.

## Evidence-gated flattening roadmap

### P0 — Make one failing and one successful run explain themselves

Add a stage ledger to the inner engine with timestamps, dimensions/token
counts, model variant, low-RAM state, MLX active/cache/peak memory, process RSS,
system memory pressure, and the last completed synchronization boundary.
Capture child termination status, macOS unified logs for Metal errors, and
ffmpeg stderr as separate artifacts.

**Gate:** reproduce the current failure at least three times with the same
classification, plus one control run at the same shape. Until this exists,
“memory leak,” “watchdog,” and “kernel bug” are hypotheses.

### P1 — Establish the cheapest safe configuration envelope

Benchmark the actual AI Studio presets as a matrix:

- BF16 eager;
- BF16 low-RAM;
- q8 eager;
- q8 low-RAM;
- only for shapes that need it, q8 low-RAM plus modality tiling.

Measure cold start, warm generation, each pipeline phase, peak MLX memory, peak
RSS, wall time, output similarity, and crash rate under both quiet and
deliberately contended macOS conditions.

**Gate:** select defaults by preset/hardware only after a 10-run stability
cohort and visual/metric acceptance. The existing q8/low-RAM numbers justify
testing this first; they do not justify silently changing AI Studio's quality
default.

### P2 — Replace two throwaway Python children with one persistent MLX worker

Build a long-lived engine that imports `ltx_pipelines_mlx` directly. AI Studio
should send a typed request over a local IPC boundary rather than construct CLI
argv and scrape characters from stdout. The worker should own:

- model/snapshot resolution;
- shape-bucketed pipeline and compiled-function caches;
- structured progress, memory telemetry, cancellation, and errors;
- stage-aware component eviction;
- a one-job-at-a-time queue;
- graceful self-recycle after a configurable memory/health threshold; and
- crash isolation so a Metal failure does not kill the AI Studio server.

This produces the practical two-layer architecture:

```text
AI Studio control plane  ⇄  persistent MLX engine
                                  │
                                  ├─ MLX/Metal
                                  └─ media encoder
```

Do not begin by keeping every component resident. The current component
ordering exists to keep Gemma, DiT, and decoders from stacking. Persistence
should first preserve the process, snapshot mappings, tokenizer, metadata, and
compiled shape buckets while retaining stage-aware eviction. Components can be
kept warm only where measured headroom allows.

**Gate:** identical request/output semantics, cancellation, and progress;
10-generation marathon with bounded peak and end-of-job memory; lower median
job-start latency; no reduction in crash isolation.

### P3 — Compile and schedule around stable shapes

The low-RAM path already compiles its shared transformer block. Next
experiments should be narrow:

1. Cache compiled shared blocks and shape metadata across jobs.
2. Test compilation of stable denoise-step subgraphs per
   `(model, dtype, H, W, F, conditioning mode)`.
3. Profile whether prompt connector, latent upsampler, VAE tiles, or
   audio/vocoder graphs benefit from compilation.
4. Tune `LTX2_GEMMA_EVAL_EVERY` and `LTX2_DIT_EVAL_EVERY` one boundary at a
   time on the M5 Max.

**Gate:** warm-run speedup after compile cost, equal output within the existing
MLX tolerance, no higher peak, and zero watchdog failures in a contended
marathon. Never remove all barriers merely because a quiet one-off run passes.

### P4 — Remove proven media copies

Run allocation and CPU traces around output:

- write a buffer-protocol view directly to ffmpeg if MLX buffer lifetime and
  synchronization are proven safe, avoiding `bytes(memoryview(...))`;
- stream/pipe the audio result instead of creating a complete NumPy int16
  buffer and temporary WAV;
- cache the deterministic source-image preprocessing products needed by each
  stage during one job;
- evaluate VideoToolbox/AVFoundation for direct H.264/AAC muxing if ffmpeg
  process and copy overhead is measurable.

**Gate:** a trace must show the allocation/copy is material at production
dimensions. Preserve the CRF-33 I2V preprocessing behavior unless visual
regression proves an alternative equivalent.

### P5 — Native MLX host proof of concept

A Swift/Objective-C++ or C++ host is feasible in principle, but it is a port,
not a compiler switch. It must reproduce tokenizer/Gemma integration,
safetensors and quantization loading, 48-block AudioVideo DiT, conditioning,
schedulers, VAE/audio/vocoder, LoRA handling, memory lifecycle, progress,
cancellation, and output parity. A Swift MLX or C++ MLX interface should first
be validated against the exact operations and quantized weights this repository
uses.

Start with one fixed-shape distilled stage and the existing weights. Compare it
to the persistent Python worker with model load excluded and included.

**Gate:** demonstrate a meaningful measured improvement in warm latency,
memory, reliability, packaging, or control. If the native host only removes
sub-millisecond Python dispatch around multi-second Metal work, keep Python and
invest in the actual bottleneck.

### P6 — Custom Metal kernels only for traced hotspots

Do not reverse engineer Apple's private Metal driver or precompiled Apple
kernels as the first strategy. That path is brittle across macOS/GPU revisions
and does not address orchestration or model residency.

Use Metal System Trace and MLX-level profiling to find a repeatable hotspot,
then implement the smallest candidate as an MLX-compatible custom Metal
operation or native extension. Plausible investigation areas—not established
bottlenecks—include:

- attention layout/materialization at long video token counts;
- quantized matrix operations not already using an optimal MLX kernel;
- repeated transpose/normalize/patchify boundaries around the latent
  upsampler;
- VAE tile accumulation buffers, currently fp32 with a documented BF16 TODO;
- uint8 output conversion and encoder handoff; and
- audio post-processing/resampling.

**Gate:** the Metal capture attributes significant GPU time or memory to the
exact operation; the custom operation passes numerical and visual regression
tests across target M-series GPUs; the gain remains after an MLX upgrade; and
maintenance cost is justified. If attention's quadratic token count is the
limit, tiling or a model-level attention change may outperform a handcrafted
kernel.

## Recommended first implementation slice

The first engineering slice should not port the model or write Metal. It should:

1. add the P0 stage/memory/crash ledger;
2. run the BF16/q8 × eager/low-RAM envelope at the failing AI Studio preset;
3. replace `ltx23_mlx_single.py` → CLI subprocess chaining with one persistent,
   crash-isolated MLX worker that imports `DistilledPipeline`; and
4. preserve current stage-aware unload behavior until the measurements prove
   which components can remain warm.

That slice directly removes the two throwaway Python layers, attacks repeated
load/heap churn, and produces the evidence needed to decide whether the next
dollar belongs in compilation, media I/O, a native host, or a specific Metal
kernel.
