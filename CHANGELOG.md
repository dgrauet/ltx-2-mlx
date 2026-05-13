# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is pre-1.0 (`0.x.y`), the `0.y` segment serves as the major
version: breaking changes bump `y`, additive changes bump `z`. See
[`docs/PIPELINE_MATURITY.md`](docs/PIPELINE_MATURITY.md) for per-pipeline
stability guarantees.

## [Unreleased]

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
