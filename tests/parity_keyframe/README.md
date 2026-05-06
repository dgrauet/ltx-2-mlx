# Keyframe parity tests (MLX vs upstream PT)

Targeted parity tests that prove the keyframe-conditioning code path
and the sampler step logic are numerically identical to
`Lightricks/LTX-2` upstream (commit `b604d3f`). Used during the
keyframe hold-cut-decay regression investigation (2026-05-06) to rule
out the keyframe code as the source of the regression.

## What's covered

### `dump_pt.py` / `check_mlx.py` — conditioning parity

For 3 cases (start half-res, end half-res, end full-res) at typical
hedgehog dimensions, asserts bit-equality of:

- `positions` (ours: `(B, N, 3)` scalars; upstream: midpoint of
  `(B, 3, N, 2)` ranges — equivalent because the model's
  `use_middle_indices_grid=True` collapses to midpoint)
- `latent`, `clean_latent` (token concatenation)
- `denoise_mask` (per-keyframe mask blocks)
- `attention_mask` (None when starting from unmasked state — both
  sides agree)

### `dump_pt_sampler.py` / `check_mlx_sampler.py` — sampler step parity

Given a keyframe-conditioned state (gen tokens noisy, kf tokens clean,
mask=[1,...,0,...]) and a mock `denoised` model output, asserts
bit-equality across:

- `apply_denoise_mask` / `post_process_latent` (mask blend)
- `timesteps_from_mask` (`mask * sigma`)
- 1 Euler step
- 3 cumulative Euler steps
- Keyframe-token preservation across the 3 steps

## Running

PT side (in upstream venv):

```bash
cd /Users/dgrauet/sandbox/ltx-reference
uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/dump_pt.py
uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/dump_pt_sampler.py
```

MLX side (in this repo):

```bash
uv run python tests/parity_keyframe/check_mlx.py
uv run python tests/parity_keyframe/check_mlx_sampler.py
```

Both should print `ALL OK` (max_abs ≤ 1e-3).

## What this proves and does NOT prove

**Proves**: the keyframe-conditioning code path (positions, masks,
state appending) and the sampler step logic (mask blend + Euler
step) are bit-equivalent to upstream.

**Does NOT prove**: that the full keyframe pipeline produces
upstream-equivalent video. The remaining suspects for the keyframe
hold-cut-decay regression are:

1. The transformer forward pass on a keyframe-conditioned sequence
   (real weights matter — RoPE on appended tokens, AdaLN with
   per-token timesteps, attention on non-uniform mask).
2. The stage 1 → stage 2 transition (upsampler denorm/renorm,
   keyframe re-encoding at full-res).
3. The VAE decoder when decoding latents that span keyframe-influenced
   regions.

All three require real model weights and a PT vs MLX intermediate-
latent diff to investigate further.
