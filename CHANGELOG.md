# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is pre-1.0 (`0.x.y`), the `0.y` segment serves as the major
version: breaking changes bump `y`, additive changes bump `z`. See
[`docs/PIPELINE_MATURITY.md`](docs/PIPELINE_MATURITY.md) for per-pipeline
stability guarantees.

## [Unreleased]

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
