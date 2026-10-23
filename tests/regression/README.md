# Regression harness

Locks generation outputs at frozen seed/prompt/resolution so we can prove
that an upstream-sync fix produces the expected delta and nothing else.

## What it captures

Two goldens (see `config.py`):

| ID            | Pipeline                | Covers                                |
|---------------|-------------------------|----------------------------------------|
| `g1_two_stage`| `generate --two-stage`  | DiT, VAE, Vocoder, BWE, Euler+CFG     |
| `g2_hq`       | `generate --two-stages-hq` | + res_2s sampler                    |

Heavy artifacts (`.mp4`, `.wav`) are gitignored. Only `manifest.json`
(stats per tag) is committed.

A `g3b_hedgehog` golden runs the previously-validated hedgehog
keyframe config (seed 712577398, default 480x704, dev model + CFG).
It is **not** a regression check in the strict sense: the `keyframe`
pipeline it was captured with (`post_fix_2_kf`) produced a hold-cut-decay
pattern (frames stick on keyframe1 for ~80% of the timeline, jump, then
collapse onto keyframe2 over the last few frames) instead of smooth
interpolation. Fix 2 (`num_pixel_frames`) was applied and the pattern
persisted with only a ~5% reduction in cut amplitude — confirming the
regression was not localised to keyframe RoPE positions. The `keyframe`
command has since been removed (upstream v1.5.0); `g3b_hedgehog` now runs
its replacement, `generate --two-stage` with both images bound as guides
and `--refine-audio`. Compare against the `v0_18_0` tag (captured on that
release); the older tags come from the `keyframe` command. The replacement
holds the first image for 25 frames and cuts to the second at frame 25
(the model's behaviour on these two unrelated images, not a smooth
interpolation).

## Per-fix workflow

```bash
# 0. Lock the current behaviour as baseline.
python -m tests.regression.capture_baseline --tag pre_fix_1

# 1. Apply the fix in a feature branch.
git switch -c fix/vocoder-fp32
# ... edit code ...

# 2. Capture the new behaviour.
python -m tests.regression.capture_baseline --tag post_fix_1

# 3. Diff.
python -m tests.regression.verify_no_regression \
    --against pre_fix_1 --tag post_fix_1
```

## Expected deltas per fix

| Fix | g1_two_stage video | g1 audio (L1, STFT) | g2_hq video | g2 audio |
|-----|--------------------|----------------------|-------------|----------|
| **1 — Vocoder fp32** | PSNR = inf (identical) | sample L1 >> 1e-2, STFT L1 >> 1e-2 | identical | same delta as g1 |
| **2 — `num_pixel_frames`** | PSNR = inf | identical | PSNR = inf | identical |
| **3 — Decoder dtype guard** | PSNR = inf | identical | PSNR = inf | identical |

If a column says "identical" but PSNR is < 60 dB, the fix has leaked into
unrelated paths — **stop and investigate**.

Determinism on identical seed: `g1_two_stage` re-run produces PSNR=inf
and audio L1=0. The strict cap is achievable.

## A/B audio listening

`tests/regression/goldens/pre_fix_1__g1_two_stage.wav` vs
`post_fix_1__g1_two_stage.wav`. Upstream reports 40-90% spectral metric
gap; the fix should be audibly cleaner (less crunch on transients).

## Memory check

Run with `MLX_METAL_DEBUG=1` or capture peak memory in a wrapper —
post-fix peak must stay within +5% of baseline.
