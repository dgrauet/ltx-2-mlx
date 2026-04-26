# LTX-2 MLX — TeaCache calibration & integration

**Date:** 2026-04-26
**Target:** `TwoStagePipeline` stage 1 (dev model, non-distilled, 30 Euler steps via `guided_denoise_loop`, bf16 q8 weights, default 32 GB host config = CFG-only, 2 forward passes per step)

## Context

`mlx-arsenal` ships a generic `TeaCacheController` (timestep-aware residual
caching for diffusion transformers, Liu et al.) with presets for HunyuanVideo
and FLUX. LTX-2 is absent from the registry: upstream (`ltx-reference`) does
not ship TeaCache, so coefficients must be calibrated.

This spec defines the calibration methodology, the instrumentation needed in
the LTX-2 MLX port, and the opt-in pipeline integration that consumes the
controller at inference time.

## Goals

1. Produce model-specific polyfit coefficients + a default `rel_l1_thresh`
   for LTX-2 stage 1 (30 steps), persisted in
   `mlx_arsenal.diffusion.TEACACHE_PRESETS["ltx2"]`.
2. Wire `TeaCacheController` into `TwoStagePipeline.generate()` (the dev +
   CFG path; one-stage uses the distilled model and is out of scope per
   non-goals) as an **opt-in** flag. Default behavior unchanged.
3. Keep the calibration script memory-friendly so it runs on the user's
   32 GB host alongside the q8 model + text encoder.

## Non-goals

- Calibrating distilled (8 steps) or stage-2 (3 steps) paths. Empirical
  agreement: TeaCache has no useful margin at those step counts.
- Capturing traces from the upstream PyTorch reference. The calibration is
  done end-to-end in the MLX port to keep coefficients aligned with the
  production numerical regime (bf16 q8).
- Per-stream skip decisions (separate video and audio controllers). The
  cross-attention coupling between video and audio inside each block makes a
  half-skip unsound; both streams skip together.

## Architecture overview

Three deliverables across two repos:

```
mlx-arsenal/                        ltx-2-mlx/
├── diffusion/                      ├── packages/
│   └── teacache.py                 │   ├── ltx-core-mlx/
│       └── (relax residual type)   │   │   └── model/transformer/
│                                   │   │       ├── transformer.py [tap, teacache kwargs]
│                                   │   │       └── model.py       [plumb kwargs through]
                                    │   └── ltx-pipelines-mlx/
                                    │       ├── utils/samplers.py        [tap, teacache plumbing in guided_denoise_loop]
                                    │       ├── ti2vid_two_stages.py     [enable_teacache flag]
                                    │       └── scripts/
                                    │           └── calibrate_teacache.py [new]
```

Data flow at calibration time:

```
calibrate_teacache.py constructs `tap` closure
  └─ TwoStagePipeline.generate(prompt, stage1_steps=30, tap=capture)
      └─ guided_denoise_loop(..., tap=tap)
          └─ for step in range(30):
              model(..., tap=tap)  # tap fires only on the conditioned pass
                  └─ tap(step, video_normed, video_residual, audio_residual)
                      └─ tap closure: append delta_in / delta_out scalars
  └─ on completion: numpy.polyfit(deg=4) ; print preset snippet
```

## Components

### 1. `mlx_arsenal.diffusion.TeaCacheController` — type relaxation

Single-tensor `cache_residual(residual: mx.array)` is too narrow for
dual-stream models. Relax to `Any` so callers can store
`(video_residual, audio_residual)` tuples or dicts. The controller does not
inspect the cached value; it only stores and returns it.

Signature change:

```python
def cache_residual(self, residual) -> None: ...

@property
def previous_residual(self): ...   # whatever was cached
```

Tests stay green (existing tests pass arrays). One new test added: cache and
retrieve a tuple of arrays.

### 2. LTX-2 transformer — two optional kwargs

The transformer's forward gains two **independent** optional kwargs:

- `tap: Callable | None = None` — pure observation hook. After computing
  `video_normed_block0` (the modulated input to block 0) and after running
  the block stack, the transformer calls
  `tap(step_idx, video_normed_block0, video_residual, audio_residual)`
  where the residuals are `block_stack_output - block_stack_input` per
  stream. Used by the calibration script. Does not change control flow.
- `teacache: TeaCacheController | None = None` — control hook. After the
  prelude (patchify, embed, timestep AdaLN, RMS norm → `video_normed_block0`)
  but before the block stack, the transformer asks
  `teacache.should_compute(step_idx, video_normed_block0)`. On True it
  runs the block stack normally and calls
  `teacache.cache_residual((video_residual, audio_residual))`. On False
  it **skips the block stack** and reconstructs the outputs from the
  cached residuals. The transformer's head (final norm + projection)
  always runs.

Both can coexist (calibration with `tap` set and `teacache=None` is the
common case). The block stack itself is unchanged. Cost when both are
`None`: a single `is None` check per forward. Negligible.

Shapes: `video_normed_block0` is `(B, S_v, 4096)` (the signal currently
materialized at `transformer.py:266`); audio residual is `(B, S_a, 2048)`.

### 3. LTX-2 sampler — plumbing

`guided_denoise_loop` (the loop used by `TwoStagePipeline` stage 1)
accepts `tap` and `teacache` kwargs and forwards them to the transformer
per step (with the current `step_idx` injected). The sampler does not
interpret either; it just plumbs. The basic `denoise_loop` is left
unchanged for now (one-stage / distilled — out of scope).

### 4. Pipeline integration — opt-in flags

`TwoStagePipeline.generate()` gains:

```python
def generate(
    ...,
    enable_teacache: bool = False,
    teacache_thresh: float | None = None,
):
```

Behavior:
- `enable_teacache=False` (default): unchanged.
- `enable_teacache=True`: instantiate
  `TeaCacheController.from_preset("ltx2", num_steps=num_inference_steps,
  rel_l1_thresh=teacache_thresh)`, `reset()`, and pass it to `denoise_loop`
  via the `teacache=` kwarg. The controller is fresh per `generate()`
  call; no cross-call state.

If `TEACACHE_PRESETS["ltx2"]` is missing (calibration not yet done), the
pipeline raises a clear error pointing at the calibration script.

### 4a. Multi-pass guidance interaction (LTX-2-specific)

The dev model with CFG (default config) runs **2 model forwards per
sampling step**: conditioned + unconditional. STG/modality add more passes
(3–4) but are off by default on 32 GB hosts. TeaCache as canonically
formulated assumes one forward per step; we adapt as follows:

- **One controller per pipeline call**, gating on the **conditioned pass's**
  `video_normed_block0`. The conditioned pass is always present.
- **The skip decision is global**: if the controller says skip at step `t`,
  *all* passes for step `t` are skipped (each pass reuses its own cached
  residual from the last computed step).
- The cached state is therefore not a single residual tuple but a **dict
  keyed by pass label**:
  `{"cond": (v_res, a_res), "uncond": (v_res, a_res), ...}`.
  The arsenal `TeaCacheController` accepts this naturally once
  `cache_residual` accepts `Any` (Component 1).
- The conditioned pass runs first per step; we ask `should_compute` once
  using its modulated input. If True, we run all passes and cache them in
  the dict. If False, we reconstruct each pass's output from its cached
  residual without invoking the model at all.

Memory implication: with CFG, the rolling residual state ~doubles
(~400 MB total) — still safe inside the 32 GB envelope.

Calibration uses the conditioned pass only for delta computation
(`tap` returns conditioned-pass tensors). The other passes' residuals
are not part of the polyfit signal — they piggyback on the conditioned
pass's skip decisions at inference.

### 5. Calibration script

`packages/ltx-pipelines-mlx/scripts/calibrate_teacache.py`:

```
calibrate_teacache.py \
  --model-dir dgrauet/ltx-2.3-mlx-q8 \
  --num-steps 30 \
  --prompts prompts.txt \
  --out coefficients.json
```

Algorithm:
1. Load `TwoStagePipeline` once with the dev (non-distilled) q8 model.
   Loop over prompts.
2. For each prompt run `pipeline.generate(stage1_steps=30, stage2_steps=0,
   tap=...)` (or stop after stage 1; we only calibrate stage 1).
3. Inside the callback, on each step keep one rolling pair
   `(prev_modulated, prev_residual)` per stream. After step ≥ 1 compute:
   - `delta_in_t = mean_fp32(|inp_t - inp_{t-1}|) / mean_fp32(|inp_{t-1}|)`
   - `delta_res_t = mean_fp32(|res_t - res_{t-1}|) / mean_fp32(|res_{t-1}|)`
   The fp32 cast is only on the small reductions, not on the tensors
   themselves.
4. After all prompts: `coeffs = np.polyfit(delta_in, delta_res, deg=4)`.
   Pick a default `rel_l1_thresh` empirically (start at 0.15, the upstream
   HunyuanVideo default; the user can tune).
5. Write a JSON blob with `{"coefficients": [...], "rel_l1_thresh": ...,
   "calibration_meta": {"num_prompts": 3, "num_steps": 30, ...}}` and print
   a copy-pasteable Python snippet for `TEACACHE_PRESETS["ltx2"]`.

Memory budget per step (rolling state, not buffered history):
- `prev_modulated_video`: ~64 MB (bf16, B=1, S≈8K, D=4096)
- `prev_modulated_audio`: ~32 MB
- `prev_residual_video` / `audio` (conditioned pass): same sizes
- ~200 MB during calibration (single conditioned pass captured).
- ~400 MB during inference with CFG enabled (2 cached residual pairs:
  conditioned + unconditional). Still safe inside 32 GB.

### Calibration scope

- 3 prompts × 1 seed (start). Bumping to 5 is supported via the prompt file.
- Single resolution / frame count: the user's typical generation config
  (e.g., 480 × 704, 97 frames). Calibration is not robust to wildly
  different sequence lengths because `mean(|inp|)` scales with token count.
- Polynomial degree 4 (upstream default).

## Testing

### Unit tests (this session, on synthetic data)

- `test_tap_called_per_step`: a fake 1-block transformer wired with the
  `tap` argument; confirm the tap receives one call per step with correct
  shapes.
- `test_tap_zero_overhead_when_none`: with `tap=None` and `teacache=None`,
  the sampler / transformer's per-step overhead is unchanged (no extra
  branches fire).
- `test_skip_short_circuits_model`: with `teacache` set to a forced-skip
  controller stub, the block stack is NOT executed on skip steps, and
  outputs equal `(video_in + cached_video_res, audio_in + cached_audio_res)`.
- `test_arsenal_cache_residual_accepts_tuple`: arsenal-side test for the
  type relaxation.

### Integration validation (delegated to user)

- User runs `calibrate_teacache.py` on their host. Inspects fitted curve
  visually (delta_in → predicted delta_res should be monotonic increasing).
- User generates with `enable_teacache=True, teacache_thresh=0.15` and
  visually compares against `enable_teacache=False`. Tunes threshold up
  (more skip, more drift) or down as needed.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| bf16 noise in delta computation produces a poor polyfit | fp32 cast on the L1 reductions only |
| Sequence-length sensitivity (calibration at 480×704×97 doesn't transfer) | Document the calibration resolution in the preset metadata; flag if generate() is called at a very different shape |
| Audio/video coupling broken if streams skip independently | Force joint skip; gate on video_normed only |
| User runs out of memory loading model + capturing traces | Rolling state only (~200 MB); no per-step buffering |
| Calibration coefficients drift if model weights change (e.g., new release) | Preset metadata records the model-dir hash; pipeline warns on mismatch |

## Out of scope (explicit non-decisions)

- Per-resolution presets. We pick one config and ship one set of coefficients.
- Adaptive thresholding (varying `rel_l1_thresh` mid-generation).
- TeaCache for the audio-only path or VAE.
- ComfyUI integration (separate spec, downstream).

## Open questions to resolve in implementation

- Exact fp32-cast location: only on the reduction (`mx.mean(mx.abs(x.astype(mx.float32)))`)
  vs cast before the subtract. Likely the latter for stability with
  near-equal tensors. Decide via a quick A/B during implementation.
- Whether the boundary rule (first/last step always compute) interacts
  badly with stage 1's truncated schedule (sigma_stop > 0). Probably not —
  the rule is on `step_index`, not sigma value — but verify with one
  end-to-end run.
