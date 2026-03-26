# Keyframe Interpolation — Bug Tracker

_Last updated: 2026-03-26_

## Status: All Known Bugs Fixed

Five bugs were found and resolved in the keyframe interpolation pipeline.
The pipeline now works with dev model + CFG + STG + modality guidance.

## Bugs Fixed

| # | Bug | Commit | Description |
|---|-----|--------|-------------|
| 1 | Conditioning after noising | 9c089ff | Pipeline created noise then appended keyframes. Fix: empty state -> conditioning -> noise. |
| 2 | GroupNorm pytorch_compatible | 133e1c1, ca784dd | MLX GroupNorm didn't normalize over spatial dims. Fix: `pytorch_compatible=True`. |
| 3 | Guider params not propagated | 899ae78, ecd8910, 668d4b0 | CLI built full MultiModalGuiderParams but generate_and_save dropped them. |
| 4 | Resolution rounding | 6e71b71 | Half-res rounded to 64-multiples (352->320). Fix: use integer division by 32. Output 448x704. |
| 5 | STG cross-attention mask shape | 591a3d8 | A2V/V2A masks 4D (B,1,1,1) on 3D output (B,N,dim) -> broadcasting corruption. Fix: pass 3D tensor to mask_like. |

## Test Matrix

Pipeline: dev model (q8) + CFG (scale 3.0) + distilled LoRA, seed 712577398, 33 frames.

| Fixture | Resolution | Status |
|---------|-----------|--------|
| hedgehog | 704x448 | OK |
| forest_zoom | 704x448 | OK |
| day_to_night | 704x448 | OK |
| hedgehog_seasons | 704x448 | OK |
| forest_blur | 704x448 | OK |

## Memory Limits (32GB Mac)

| Guidance Mode | Passes/Step | Max Frames (480x704) |
|--------------|-------------|---------------------|
| CFG only | 2 | 33 |
| CFG + STG + modality | 4 | 17 |

## Reproduction

```bash
# Generate fixtures from source photos
uv run python scripts/keyframe_tests/generate_fixtures.py

# Run full matrix
uv run python scripts/keyframe_tests/run_matrix.py

# Run specific pairs
uv run python scripts/keyframe_tests/run_matrix.py --pairs hedgehog,forest_zoom

# Single run
uv run ltx-2-mlx keyframe \
    --model dgrauet/ltx-2.3-mlx-q8 \
    --prompt "description" \
    --start start.png --end end.png -o out.mp4 \
    --dev-transformer transformer-dev.safetensors \
    --distilled-lora ltx-2.3-22b-distilled-lora-384.safetensors \
    --cfg-scale 3.0 --seed 712577398 --frames 33
```
