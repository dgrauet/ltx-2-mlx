# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is pre-1.0 (`0.x.y`), the `0.y` segment serves as the major
version: breaking changes bump `y`, additive changes bump `z`. See
[`docs/PIPELINE_MATURITY.md`](docs/PIPELINE_MATURITY.md) for per-pipeline
stability guarantees.

## [Unreleased]

## [0.14.9] - 2026-06-02

Handles LTX-2.3 model directories that ship versioned safetensors
filenames (e.g. `transformer-distilled-1.1.safetensors`). Previously the
loader hardcoded unversioned names, so it silently ignored LTX's newer
file revisions — and failed hard if a user kept only the newer file to
save disk. Resolution is now dynamic: the alphabetically-latest versioned
file wins over the unversioned exact name when both are present. After an
upstream sync + re-forge, the code picks up the newer weights with no
changes. Thanks to [@plz12345](https://github.com/plz12345) for the
contribution (PR #32).

### Added

- `BasePipeline._resolve_safetensors(model_dir, stem)` — resolves a
  (possibly versioned) safetensors path, preferring `{stem}-*.safetensors`
  over `{stem}.safetensors` and returning the canonical exact path when
  nothing exists (clear `FileNotFoundError`). Wired into transformer,
  distilled-LoRA, and upscaler resolution across `_base`, `distilled`,
  `ic_lora`, and `ti2vid_two_stages`.
- `_load_weights` fallback in `loader/sft_loader.py` — loads extensionless
  HuggingFace cache blobs via `mx.load(path, format="safetensors")`, which
  `mx.load` cannot infer by suffix. Reachable when a model dir resolves
  through a `snapshots/` symlink to the real GUID blob name (custom or
  refined models).
- Trainer `model_loader.load_transformer` auto-detect extended to
  versioned transformer names.

### Fixed

- Upscaler file naming and key-prefix handling: supports both the new
  v1.1+ bare-key layout and the legacy stem-prefixed layout.

## [0.14.8] - 2026-05-22

Enables `generate --lora` on the `--low-ram` block-streaming path.
Previously, combining the two raised `NotImplementedError` because
LoRA fusion required a fully-materialised weight dict before
block-stream eviction started. The streaming path now attaches each
pending LoRA as a `BlockLoraSource` on the `StreamingLTXModel`
wrapper — fusion happens per-block at each `bind()`, mirroring the
pattern already used by `ICLoraPipeline._fuse_loras` for control
LoRAs. Works for community LoRAs at strength 1.0 and at custom
strengths. Thanks to [@plz12345](https://github.com/plz12345) for the
contribution (PR #30, follow-up to #20).

### Added

- `resolve_lora_path` helper in `ltx_pipelines_mlx.utils._orchestration`
  shared by the streaming and non-streaming LoRA paths. Accepts local
  `.safetensors` paths and HuggingFace repo IDs. Raises `ValueError`
  on ambiguous multi-safetensors repos (lists the candidate file
  names) and `FileNotFoundError` on empty repos.
- Audio / joint-block LoRA key remappings (`.linear_1.` → `.linear1.`,
  `.linear_2.` → `.linear2.`, `audio_ff.net.0.proj.` →
  `audio_ff.proj_in.`, `audio_ff.net.2.` → `audio_ff.proj_out.`)
  consolidated into the shared `LTXV_LORA_COMFY_RENAMING_MAP` in
  `ltx_core_mlx.loader.sd_ops` so the streaming path picks them up
  automatically (it goes through the map directly, no longer through
  `ti2vid_two_stages._remap_lora_keys`).
- Tests: streaming dispatch now pins `BlockLoraSource` ctor args and
  exercises a parametrized two-LoRA case; new `test_resolve_lora_path`
  covers the 4 resolution branches (local-exists, HF single, HF
  multi-raises, HF zero-raises); new `test_lora_renaming_map` locks
  the contract for the 4 audio/joint-block patterns.

### Changed

- CLAUDE.md "Limitations" + CLI `--low-ram` help text updated: the
  `generate --lora` flag is now compatible with `--low-ram` via
  per-block bind-time fusion.
- `_remap_lora_keys` in `ti2vid_two_stages.py` simplified to a single
  dict comprehension; `or k` fallback removed (verified dead — the
  shared map's `apply_to_key` always returns a string given
  `with_matching()` + no `allowed_keys`).
- `resolve_lora_path` writes its "Downloading LoRA from HuggingFace"
  notice to stderr via `print(..., file=sys.stderr)` instead of
  `logging.info` (which was swallowed by the absent logger config).

## [0.14.7] - 2026-05-20

Hotfix for a long-standing IC-LoRA reference-video crash. Any caller
passing an 8k-frame source file (the format LTX itself produces — its
``_decode_and_save_video`` drops the leading frame on write) into
``ICLoraPipeline.generate_and_save(video_conditioning=...)`` hit a
``space_to_depth`` reshape error at ``ltx_core_mlx/model/video_vae/sampling.py:121``
because the encoder's first temporal-stride-2 block requires a
``(1 + 8k)``-frame input. Failure was input-content-independent: a raw
RGB driving video and a canny-edges control map produced byte-identical
error numbers at the same reshape site.

### Fixed

- ``append_ic_lora_reference_video_conditionings`` in ``iclora_utils.py``
  now probes the source with ``probe_video_info``, clamps to the
  caller's target ``num_frames``, and rounds down to the nearest
  ``(1 + 8k)`` before invoking ``load_video_frames_normalized``.
  Mirrors ``RetakePipeline._encode_source_video``. Applies to every
  ``ICLoraPipeline`` subclass — including ``HDRICLoraPipeline`` (where
  the reporter observed the same crash at a different spatial scale,
  ``reference_downscale_factor=1``) and ``LipDubPipeline``'s
  video-reference path.

### Changed

- Lifted the local ``from ltx_core_mlx.components.patchifiers import
  compute_video_latent_shape`` inside ``append_ic_lora_reference_video_conditionings``
  to module scope. Same scoping anti-pattern that caused the
  ``UnboundLocalError`` previously fixed in ``ic_lora.py`` (commit
  ``23127d6``); not load-bearing here, but matches the cleanup precedent.

### Credit

Bug surfaced and diagnosed by [@R0drig0Diaz](https://github.com/R0drig0Diaz)
in [#27](https://github.com/dgrauet/ltx-2-mlx/issues/27) — with
byte-identical reproductions across two input modalities (RGB +
canny-edges) and two LoRA variants (Union Control with
``reference_downscale_factor=2``, HDR with ``reference_downscale_factor=1``),
plus the ``RetakePipeline._encode_source_video`` precedent and a 3-fix
proposal. The two scoping-trap items called out in #27 (``ic_lora.py:261``
+ ``ic_lora.py:501``) were already resolved on ``main`` since
``23127d6``; the reporter was on stale HEAD ``0e753b6``.

## [0.14.6] - 2026-05-20

Automatic temporal tiling for the video VAE decoder. The block-3
DepthToSpaceUpsample intermediate (``(B, 512, 4F, 4H, 4W)`` in bf16)
dominates peak Metal activation memory at HD or long durations and
pushed 32 GB Macs into swap on 720p+ runs beyond ~20s — even when the
rest of the pipeline (transformer streamed, decoders loaded on demand)
fit comfortably.

### Added

- ``_compute_decode_tiling(latent_shape, frame_rate)`` in
  ``ltx_core_mlx.model.video_vae.video_vae``: pure-arithmetic helper
  that derives a ``TilingConfig`` from the latent shape and the
  ``LTX2_VAE_DECODE_BUDGET_GB`` budget (default ``8.0``). Returns
  ``None`` when the full video already fits, so the no-tiling path
  has zero overhead at common resolutions. Tile size and overlap
  scale with frame rate (~1 second of pixel frames of overlap).
- ``VideoDecoder.decode_and_stream`` (pipelines wrapper) now prints
  ``[vae-decode tiled: tile_frames=N overlap=K]`` to stderr when tiling
  kicks in, gated on the pipeline's ``verbose`` flag.

### Changed

- ``VideoDecoder.tiled_decode`` (core) inserts ``mx.eval`` +
  ``aggressive_cleanup`` after each tile decode and after each
  accumulation step so prior-tile activations are freed before the
  next tile begins.
- ``VideoDecoder.decode`` gains an opt-in ``_materialize_stages``
  keyword-only flag (default ``False``) that forces ``mx.eval`` between
  the four upsample stages. Only set ``True`` by ``tiled_decode`` —
  the no-tiling fast path keeps full kernel fusion across upsample
  stages.
- ``A2VidPipelineTwoStage._upscale_and_optionally_encode`` was
  reaching into ``self.vae_decoder.decode_and_stream`` directly;
  routed through ``self.video_decoder_block.decode_and_stream`` like
  every other pipeline. Removes a stale ``assert self.vae_decoder
  is not None`` (the block's ``load()`` is idempotent) and gives a2v
  the same stderr marker as the rest.

### Note on numerical equivalence

Tiled decode is **not** bit-equivalent to a non-tiled decode at the
same configuration. The video VAE's causal ``Conv3dBlock``
(``convolution.py:62-66``) replicates each tile's first frame for
temporal padding, so isolated tiles diverge from the full-decode
context at boundaries. The trapezoidal blend mask smooths the seam
visually but does not reconstruct the exact signal. This is intrinsic
to any tiled-VAE-decode and matches the upstream behaviour — users
switching into tiled mode at 1080p+ should expect minor boundary
drift vs. the same config with enough RAM to skip tiling.

### Benchmarks (one run each, ``--distilled`` 8/3 at 480p × 15s × 25fps)

| stage | no-tile | tiled (opt-in via large input) | Δ |
|---|---|---|---|
| decode | 124.1s | 131.4s | +5.9% |
| total | 841.0s | 881.9s | +4.9% |

No-tiling fast path is unchanged.

### Credit

End-to-end PR by [@plz12345](https://github.com/plz12345) in
[#25](https://github.com/dgrauet/ltx-2-mlx/pull/25), iterated through
two rounds of review covering env-var budget gating, scoped
``BrokenPipeError`` handling, fp32→bf16 budget correction, stderr
marker plumbing, a2v consistency cleanup, and the multi-tile
integration test. Production-validated on plz12345's M5 MacBook Air
32 GB for a week prior to submission.

## [0.14.5] - 2026-05-19

CLI phase-marker coverage gap on the distilled two-stage path. The
``[Encoding prompt] ... done in X.Xs`` marker introduced in v0.13.1
was emitted by every pipeline that goes through ``BasePipeline``'s
text-encoding helper, but ``DistilledPipeline.generate_two_stage``
encodes the (positive-only) prompt inline and was missed in the v0.13.1
audit. ``[Loading text encoder]`` was emitted, then silence until
``[Loading DiT]``.

### Fixed

- Wrap ``_encode_text`` + ``_materialize`` in ``DistilledPipeline.generate_two_stage``
  with the ``phase("Encoding prompt", ...)`` context manager so the
  encoding duration is reported on stderr like every other pipeline.
  Caught by external contributor [@plz12345](https://github.com/plz12345)
  in PR #28.

## [0.14.4] - 2026-05-14

Apple Silicon Metal watchdog hardening for the denoise loop. On
M2 Max 64 GB (and any Apple Silicon under load), the macOS GPU
watchdog could fire with ``MTLCommandBufferErrorInternal`` (code 14)
at the start of a denoise loop when the accumulated pre-denoise lazy
graph — VAE encoding, conditioning blend, and noise addition — was
submitted as a single oversized Metal command buffer exceeding the
~10s watchdog window.

### Fixed

- Add ``BasePipeline._pre_denoise_flush(video_state, audio_state)``
  that calls ``mx.eval`` on the noised latent states to force-materialise
  the accumulated graph in its own command buffer before the denoise
  loop begins. Each subsequent denoise-step buffer is then within the
  watchdog window.
- Wire the flush at every denoise call site across all pipelines
  (18 sites total): ``BasePipeline.generate`` (one-stage distilled),
  ``TI2VidOneStagePipeline`` (dev one-stage + CFG), ``DistilledPipeline``
  (two-stage), ``TI2VidTwoStagesPipeline`` and ``TI2VidTwoStagesHQPipeline``
  (both stages), ``A2VidPipelineTwoStage`` (both stages),
  ``ICLoraPipeline`` and ``HDRICLoraPipeline`` (both stages, via
  inheritance), ``RetakePipeline`` (retake + extend), ``KeyframeInterpolationPipeline``
  (stage 1 dev/distilled branches + stage 2), and ``LipDubPipeline``
  (both stages).

### Credit

Initial fix and validation on M2 Max 64 GB by
[@colinbdesign](https://github.com/colinbdesign) in
[#22](https://github.com/dgrauet/ltx-2-mlx/pull/22) — covered
``BasePipeline.generate`` plus the two two-stage variants. Coverage
extended to the remaining 14 call sites in
[#23](https://github.com/dgrauet/ltx-2-mlx/pull/23).

## [0.14.3] - 2026-05-14

Accurate transformer-load phase timing. Before this patch, the
``[Loading transformer (...)] done in 0.1s`` marker reported ~0.1s for
a 10+ GB load — MLX is lazy, so ``apply_quantization`` + ``load_weights``
build a graph but defer the real work. The marker measured graph
construction, not loading.

### Fixed

- Force MLX graph materialisation immediately after ``load_weights`` in
  both the orchestration helper (``utils._orchestration.load_transformer``)
  and the LoRA-fusion path (``_base.py``). Both branches now report
  real load time (~1.8s empirically observed by the reporter on a
  typical run).

### Credit

Bug surfaced by [@plz12345](https://github.com/plz12345) in
[#18](https://github.com/dgrauet/ltx-2-mlx/pull/18). Release PR
adds the symmetric guard to the LoRA-fusion path for consistency.

## [0.14.2] - 2026-05-14

Hotfix for a long-standing latent bug: setting ``pipe._pending_loras = [...]``
from the CLI was silently dropped by every pipeline whose ``load()`` method
overrides :meth:`BasePipeline.load` — that is, ``--distilled``, ``--one-stage``,
``--two-stage``, and ``--two-stages-hq``. Only the ``BasePipeline.load()``
path (no longer reached by any T2V/I2V CLI mode) honored the flag, so T2V
generation with ``--lora`` produced output indistinguishable from a
base-model run.

### Fixed

- ``BasePipeline._load_transformer_with_optional_streaming`` now honors
  ``_pending_loras`` directly. Every pipeline whose ``load()`` routes
  through this wrapper (or through ``_load_dev_transformer``, which
  transitively calls the wrapper) automatically picks up LoRA fusion —
  no subclass-level boilerplate required. Pre-existing pipelines fixed:
  ``DistilledPipeline``, ``TI2VidOneStagePipeline``, ``TI2VidTwoStagesPipeline``,
  ``TI2VidTwoStagesHQPipeline``.
- New regression test (``tests/test_pending_loras_dispatch.py``) locks the
  contract: every pipeline ``load()`` override must route DiT construction
  through the wrapper. Catches future overrides that would silently
  reintroduce the bug.

### Credit

Bug surfaced by [@colinbdesign](https://github.com/colinbdesign) in
[#16](https://github.com/dgrauet/ltx-2-mlx/pull/16) (closed in favor of
this PR's wrapper-level fix, which is upstream-iso friendly and covers
all four affected pipelines instead of just ``DistilledPipeline``).

## [0.14.1] - 2026-05-14

Hotfix for a regression introduced by the v0.14.0 `fps` → `frame_rate`
rename. The `VideoDecoder.decode_and_stream` wrapper in
`ltx_pipelines_mlx/utils/blocks.py` was missed during the audit and
kept the old `fps=` kwarg, while the inner `ltx_core_mlx`
`VAE.decode_and_stream` now requires `frame_rate=` mandatory
keyword-only. Every decode path raised `TypeError: ... got an
unexpected keyword argument 'frame_rate'` at mux time.

### Fixed

- `VideoDecoder.decode_and_stream` wrapper accepts and forwards
  `frame_rate=` (was still `fps=`). Affects every pipeline that goes
  through the orchestration helper: `--two-stage`, `--two-stages-hq`,
  `a2v`, `keyframe`, `ic-lora`, `hdr-ic-lora`. One-stage was
  unaffected — bug was isolated to the decode hop.
  Closes [#17](https://github.com/dgrauet/ltx-2-mlx/pull/17).
  Thanks to [@plz12345](https://github.com/plz12345) for the catch +
  patch.

## [0.14.0] - 2026-05-13

Ultra-strict upstream-iso pass on the `frame_rate` parameter. Mirrors
`Lightricks/LTX-2`'s pipeline signatures byte-for-byte: every public
pipeline method now takes `frame_rate: float` as a **mandatory
keyword-only** parameter (no default, matches upstream's required
`frame_rate=` kwarg). The legacy `fps=` kwarg is renamed throughout the
pipelines layer + immediate core helpers. Closes the audit gap
identified on [issue #6](https://github.com/dgrauet/ltx-2-mlx/issues/6).

### Changed

- **Breaking**: every pipeline public method (`generate*`,
  `generate_and_save`, `interpolate`, `retake`, `extend`,
  `generate_lipdub`) renames `fps: float = 24.0` →
  `frame_rate: float` (mandatory keyword-only). 8 pipelines affected:
  `TI2VidOneStagePipeline`, `TI2VidTwoStagesPipeline`,
  `TI2VidTwoStagesHQPipeline`, `DistilledPipeline`,
  `A2VidPipelineTwoStage`, `KeyframeInterpolationPipeline`,
  `ICLoraPipeline`, `HDRICLoraPipeline`. The 2 pipelines that derive
  `frame_rate` from the source video metadata (`RetakePipeline`,
  `LipDubPipeline`) still don't expose a public kwarg, but their
  internal `extend(...)` accepts `frame_rate` keyword-only too.

- **Breaking**: every CLI subcommand that previously accepted `--fps`
  now accepts `--frame-rate` and makes it **required**. Coverage
  changes: `--frame-rate` is now also required on `generate`
  (all four modes), `ic-lora`, and `hdr-ic-lora` — fixing the silent
  24fps default that was unintentionally baked into the temporal RoPE
  on those pipelines. `retake`, `extend`, and `lipdub` derive
  `frame_rate` from the source video and need no flag.

- **Breaking** (core helpers): `compute_video_positions`,
  `compute_audio_token_count`, `decode_and_stream`,
  `VideoConditionByKeyframeIndex` constructor, `_decode_and_save_video`,
  `combined_image_conditionings` all rename `fps` → `frame_rate`. The
  `fps` field on `RetakePipeline._SourceMeta` is renamed to
  `frame_rate`. `VideoInfo.fps` (ffprobe metadata carrier) **keeps
  its `fps` name** — it describes a source video file's metadata, not
  a pipeline parameter, and aligns with how ffprobe and upstream's
  `VideoPixelShape(fps=...)` data class label the concept.

### Migration

Python API callers:

```python
# 0.13.x
pipe.generate_and_save(prompt="...", num_frames=97, fps=24.0)
# 0.14.0
pipe.generate_and_save(prompt="...", num_frames=97, frame_rate=24.0)
```

`frame_rate` is mandatory and keyword-only — positional callers and
callers relying on the implicit 24.0 default will hit a `TypeError`.

CLI users:

```bash
# 0.13.x
ltx-2-mlx a2v --audio music.wav --fps 24 ...
# 0.14.0
ltx-2-mlx a2v --audio music.wav --frame-rate 24 ...

# 0.13.x silently assumed 24
ltx-2-mlx generate --two-stage -p "..." -o out.mp4
# 0.14.0 requires it explicitly
ltx-2-mlx generate --two-stage --frame-rate 24 -p "..." -o out.mp4
```

LTX-2.3 was trained at 24 fps. Values far from 24 drift out of the
temporal RoPE training distribution — quality risk. ComfyUI exposes
the same knob with the same caveat.

## [0.13.1] - 2026-05-13

Adds CLI phase markers around previously silent long-running stages
(Gemma load + prompt encode, transformer load, decoder load, video
decode). Addresses [issue #5](https://github.com/dgrauet/ltx-2-mlx/issues/5)
— a UX-only change with no impact on math, performance, or output.

### Added

- **CLI phase markers**. Each pipeline now prints `[phase] ...` /
  `[phase] done in X.Ys` lines to **stderr** around the five silent
  stages: loading the text encoder, encoding the prompt, loading the
  transformer, loading the decoders, and decoding video + audio +
  muxing. Output goes to stderr so stdout stays clean for callers that
  pipe pipeline output. Suppressed by `--quiet`.
- `BasePipeline.verbose` constructor parameter (default `True`)
  controlling the phase markers. CLI maps `verbose=not args.quiet` after
  pipeline construction. Programmatic users can set it either via the
  constructor or by assigning `pipe.verbose = False` after the fact.
- New `ltx_pipelines_mlx.utils.progress.phase()` context manager. Small
  helper used internally by `BasePipeline` to wrap silent stages; no-op
  when `verbose=False`. Public-ish utility, but mainly an internal
  building block.

## [0.13.0] - 2026-05-13

Removes the standalone `upscale` pipeline and its `upscale` CLI subcommand.
The pipeline was a local experimental addition with no upstream counterpart
in `Lightricks/LTX-2`, kept out of scope for this MLX port.

### Removed

- **`UpscalePipeline`** class and `upscale` CLI subcommand. The pipeline
  exposed the LTX neural latent upsampler (`spatial_upscaler_x2_v1_1` /
  `spatial_upscaler_x1_5_v1_0`) as a standalone VAE-encode → upsampler →
  VAE-decode tool with no DiT. It has no upstream equivalent and was only
  ever a local experiment. Removed from `ltx_pipelines_mlx.__all__`,
  `ltx-2-mlx --help`, and the maturity matrix.

  **What still works.** The neural latent upsampler module
  (`ltx_core_mlx/model/upsampler/`) is unchanged — it's a load-bearing
  component of every two-stage pipeline (`generate --two-stage` /
  `--two-stages-hq` / `--distilled`, `keyframe`, `ic-lora`, `hdr-ic-lora`,
  `a2v`, `lipdub`). Only the standalone CLI wrapper is gone.

  **Migration.** No upstream-iso replacement exists. If you relied on
  standalone latent upscaling, pin `ltx-2-mlx==0.12.1` or re-implement
  externally on top of `ltx_core_mlx.model.upsampler.LatentUpsampler`.

## [0.12.1] - 2026-05-13

Adds `LipDubPipeline` from upstream PR #212 as a new
**[experimental tier](docs/PIPELINE_MATURITY.md) pipeline**.

### Added

- **`LipDubPipeline`** + `lipdub` CLI subcommand. Two-stage lip-dubbing
  pipeline that takes a reference video providing both visual structure
  (via IC-LoRA reference latent appends) and target audio (via the audio
  VAE encoded as `AudioConditionByReferenceLatent`). Frame count is
  auto-derived from the reference video metadata (snapped to `8k+1`).
  Stage 2 keeps the stage-1 audio latent unchanged (`frozen=True`
  semantics) and only refines the video. Exported from
  `ltx_pipelines_mlx` as `LipDubPipeline`.

  **Known limitations** (model-level, not a port bug):
  - Output audio is a **VAE+vocoder reconstruction** of the reference
    audio, perceptually similar but not bit-identical to the input.
    Spectral artifacts can be audible on rich musical content. To
    preserve the original audio bit-exact, remux the source audio over
    the output mp4 via ffmpeg post-pipeline (loses fine lip-sync but
    preserves source music).
  - Lip-sync quality depends on prompt-audio alignment. Generic prompts
    produce visually plausible but loosely-synced output.
  - Uses `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` (currently `v0.9`).
    Pin a specific app version if depending on the current behaviour.

  Classified as **Experimental** in `docs/PIPELINE_MATURITY.md`. The
  CLI `--help` output marks it as `[experimental]`.

## [0.12.0] - 2026-05-13

Upstream sync from Lightricks/LTX-2 PR #212 — surfaces two **default value
changes**. No public API additions or removals; existing callers that pass
explicit values are unaffected, but consumers that relied on the previous
defaults should retest before upgrading.

### Changed (potentially breaking for callers relying on defaults)

- **`TilingConfig.default()`** spatial config bumped from `512×512` /
  `64` overlap to **`768×768` / `64` overlap**; temporal config from
  `64` frames / `24` overlap to **`80` frames / `24` overlap**. Matches
  upstream's tradeoff (fewer tile boundaries at production resolutions).
  Our internal pipelines do not call `TilingConfig.default()` directly,
  but external consumers using this helper will get the new defaults.
- **`precompute_rope_freqs` default `rope_type`** switched from
  `"interleaved"` to `"split"`. All in-repo call sites pass `rope_type=`
  explicitly so this is a no-op for our pipelines, but external consumers
  that called `precompute_rope_freqs` without specifying `rope_type` will
  see different output (the LTX-2.3 checkpoints all use SPLIT — upstream
  switched the default to match reality).

### Added

- `AudioConditionByReferenceLatent` now exported from
  `ltx_core_mlx.conditioning` and `ltx_core_mlx.conditioning.types`
  (was previously importable only from the leaf module).

## [0.11.1] - 2026-05-13

Additive upstream sync from Lightricks/LTX-2 PR #212 (merged upstream 2026-05-11),
plus a low-risk internal refactor. No public default values change in this
release.

### Added

- `iclora_utils.py` module exposing the shared IC-LoRA helpers from upstream
  PR #212: `read_lora_reference_downscale_factor`,
  `downsample_mask_video_to_latent`, `append_ic_lora_reference_video_conditionings`.
  Used by ic-lora and the upcoming lip-dub pipeline.
- `AudioConditionByReferenceLatent` conditioning type for appending
  reference audio tokens with negative-shifted RoPE positions. Audio-side
  mirror of `VideoConditionByReferenceLatent`.
- `ltx_core_mlx.components.diffusion_steps` — `EulerDiffusionStep`,
  `Res2sDiffusionStep`, `EulerCfgPpDiffusionStep` primitives + protocol +
  `_get_ancestral_step` helper. Available as standalone primitives;
  existing samplers still inline this math (no behaviour change for
  current pipelines).
- `ltx_core_mlx.utils.diffusion` — `to_velocity` / `to_denoised` helpers
  matching upstream.

### Changed

- `ic_lora.py` refactored to delegate the IC-LoRA reference video
  conditioning to `iclora_utils.append_ic_lora_reference_video_conditionings`.
  -151 LOC net. Public API unchanged. Bit-exact regression validated
  against the pre-refactor Q20 baseline (SHA256 match).

## [0.11.0] - 2026-05-11

### Added
- `--start-strength` / `--end-strength` flags on `keyframe` CLI to expose
  per-keyframe conditioning strength (upstream-iso surface, default `1.0`).
- Static-scene I2V recipe documentation (ic-lora + canny control video),
  validated end-to-end on Phoenix Q15.
- Standalone `upscale` pipeline + CLI subcommand (VAE encode → neural
  upsampler → VAE decode, no DiT).
- Modality tiling (`--tile-frames N --tile-spatial M`) on `generate` (one-stage /
  --two-stage / --two-stages-hq), `a2v`, `keyframe`.

### Fixed
- `ic_lora` upstream-iso tightening: fix `UnboundLocalError` introduced by an
  earlier edit, 4 API alignments with upstream `ICLoraPipeline._create_conditionings`.
- Default `stg_scale=1.0` restored on standard pipelines (matches upstream
  `LTX_2_3_PARAMS`). Was hardcoded `0.0` for 32 GB Mac compat — that's now
  the user's choice, not the default.
- Strip appended keyframe tokens before unpatchify across all pipelines (fix
  for multi-anchor I2V at `frame_idx > 0`).
- Multi-image conditioning propagated to all I2V pipelines; upstream-iso
  `combined_image_conditionings` helper used end-to-end.
- Metal watchdog: drop wasteful Gemma re-load in `load()` methods (Gemma was
  being loaded twice — once before forward, once before DiT — causing 7.5 GB
  heap thrash). Production-quality generation on M2 Pro 32 GB now passes
  under sustained system contention.

### Changed
- `media_io.py` ported 1:1 upstream-iso (replaces the previous divergent
  shim).
- `prepare_image_for_encoding` applies H.264 CRF round-trip (matches
  upstream's training-distribution pre-processing).
- `metal_watchdog.py` removed; auto eval gating (per-layer Gemma /
  per-block connector) supersedes the opt-in `LTX2_METAL_WATCHDOG_GUARD` env var.

## [0.10.0] - 2026-05-08

### Removed (BREAKING)
- `ImageToVideoPipeline` removed — no upstream equivalent. I2V is now
  supported on every public pipeline via `image=` kwarg / `--image` CLI flag
  (consistent with upstream's `combined_image_conditionings` pattern).

## [0.9.x] - 2026-05 (pre-isomorphism patch series)

Series of bit-exact iso-tightening patches across pipelines (T2V, I2V, A2V,
keyframe, ic-lora, hdr-ic-lora). Detailed per-commit history in `git log v0.9.0..v0.9.8`.

## [0.x] - earlier (foundations)

Initial MLX port of LTX-2.3 from Lightricks/LTX-2 reference. Bring-up of
video VAE, audio VAE + vocoder + BWE, DiT transformer, Gemma 3 12B text
encoder, conditioning system, two-stage pipelines, distilled mode.
