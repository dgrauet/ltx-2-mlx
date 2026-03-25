# Keyframe Interpolation Divergence Test Matrix

_Generated: 2026-03-25 19:16 UTC_

## Overview

3 known divergences between MLX and PyTorch reference keyframe interpolation:

| # | Divergence | MLX Behavior | Reference Behavior | Impact |
|---|-----------|-------------|-------------------|--------|
| 1 | Upsampler norm wrapping | Disabled (causes grid artifacts) | Enabled | Visual grid pattern |
| 2 | CFG guidance | Disabled (no negative prompt support) | Enabled (scale 3.0) | Over-saturation, text overlays |
| 3 | Output resolution | 480x640 (default) | 448x704 | Possible plaid/garden artifacts |

## Test Configurations

| Config | Description | Extra Args |
|--------|-------------|------------|
| A | Baseline — current working state | (none) |
| B | CFG guidance enabled | `--cfg-scale 3.0` |
| C | Reference resolution | `--height 448 --width 704` |
| D | CFG + reference resolution | `--cfg-scale 3.0 --height 448 --width 704` |

## Results Table

| Fixture | Config A | Config B | Config C | Config D |
|---------|---------|---------|---------|---------|
| existing | DRY-RUN | DRY-RUN | DRY-RUN | DRY-RUN |
| solid_colors | DRY-RUN | DRY-RUN | DRY-RUN | DRY-RUN |
| gradient | DRY-RUN | DRY-RUN | DRY-RUN | DRY-RUN |
| identity | DRY-RUN | DRY-RUN | DRY-RUN | DRY-RUN |
| text_overlay | DRY-RUN | DRY-RUN | DRY-RUN | DRY-RUN |
| geometric | DRY-RUN | DRY-RUN | DRY-RUN | DRY-RUN |

## Expected Artifacts by Configuration

- **Config A (baseline)**: None expected — current working state
- **Config B (cfg)**: Possible text overlays, over-saturation from classifier-free guidance
- **Config C (resolution)**: Possible plaid/garden artifacts from non-standard resolution
- **Config D (cfg+resolution)**: Combined B+C artifacts

## Notes for Manual Visual Inspection

- [ ] Check Config A videos for baseline quality (smooth transitions, no artifacts)
- [ ] Compare Config B vs A: look for text hallucinations, color over-saturation
- [ ] Compare Config C vs A: look for grid/plaid patterns, spatial artifacts
- [ ] Compare Config D vs A: check if B+C artifacts compound or cancel
- [ ] Identity fixture: all configs should produce near-static video
- [ ] Solid colors fixture: interpolation should be smooth gradient between red and blue

## Fixture Descriptions

| Fixture | Description |
|---------|-------------|
| existing | Original hand-crafted test pair from tests/fixtures/ |
| solid_colors | Solid red -> Solid blue (tests color interpolation) |
| gradient | Horizontal gradient -> Vertical gradient (tests spatial transitions) |
| identity | Same checkerboard duplicated (should produce near-static video) |
| text_overlay | White + 'START' text -> White + 'END' text (tests text coherence) |
| geometric | 8x8 checkerboard -> Diagonal stripes (tests pattern transitions) |

## Reproduction

```bash
# Generate fixtures
uv run python scripts/keyframe_tests/generate_fixtures.py

# Run full matrix
uv run python scripts/keyframe_tests/run_matrix.py

# Run specific configs/pairs
uv run python scripts/keyframe_tests/run_matrix.py --config A,B --pairs existing,solid_colors

# Dry run (print commands only)
uv run python scripts/keyframe_tests/run_matrix.py --dry-run
```

All tests use seed **712577398**, 97 frames, 8 steps.
