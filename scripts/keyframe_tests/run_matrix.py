"""Run keyframe interpolation divergence test matrix.

Runs each fixture pair against each test configuration and produces
a Markdown report summarizing results and timings.

Usage:
    uv run python scripts/keyframe_tests/run_matrix.py [--config A,B,C,D] [--pairs solid_colors,...] [--fixtures-only] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SEED = 712577398
FRAMES = 97
STEPS = 8
PROMPT = "smooth transition between two images"

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "keyframe_pairs"
EXISTING_START = ROOT / "tests" / "fixtures" / "keyframe_start.png"
EXISTING_END = ROOT / "tests" / "fixtures" / "keyframe_end.png"
OUTPUT_DIR = ROOT / "tests" / "outputs" / "keyframe_matrix"
REPORT_PATH = ROOT / "docs" / "comparison" / "keyframe_divergences.md"


@dataclass
class TestConfig:
    """A test configuration with optional extra CLI arguments."""

    name: str
    description: str
    extra_args: list[str] = field(default_factory=list)


CONFIGS: dict[str, TestConfig] = {
    "A": TestConfig(
        name="A",
        description="Baseline — current working state, no extra args",
    ),
    "B": TestConfig(
        name="B",
        description="CFG guidance enabled (--cfg-scale 3.0)",
        extra_args=["--cfg-scale", "3.0"],
    ),
    "C": TestConfig(
        name="C",
        description="Reference resolution (--height 448 --width 704)",
        extra_args=["--height", "448", "--width", "704"],
    ),
    "D": TestConfig(
        name="D",
        description="CFG + reference resolution combined",
        extra_args=["--cfg-scale", "3.0", "--height", "448", "--width", "704"],
    ),
}


@dataclass
class FixturePair:
    """A pair of start/end images for testing."""

    name: str
    start: Path
    end: Path
    description: str


def get_fixture_pairs() -> dict[str, FixturePair]:
    """Return all available fixture pairs including the existing one."""
    pairs: dict[str, FixturePair] = {}

    # Existing hand-crafted pair
    if EXISTING_START.exists() and EXISTING_END.exists():
        pairs["existing"] = FixturePair(
            name="existing",
            start=EXISTING_START,
            end=EXISTING_END,
            description="Original hand-crafted test pair from tests/fixtures/",
        )

    # Generated pairs
    generated_descriptions = {
        "solid_colors": "Solid red -> Solid blue (tests color interpolation)",
        "gradient": "Horizontal gradient -> Vertical gradient (tests spatial transitions)",
        "identity": "Same checkerboard duplicated (should produce near-static video)",
        "text_overlay": "White + 'START' text -> White + 'END' text (tests text coherence)",
        "geometric": "8x8 checkerboard -> Diagonal stripes (tests pattern transitions)",
    }

    for name, desc in generated_descriptions.items():
        start = FIXTURES_DIR / f"{name}_start.png"
        end = FIXTURES_DIR / f"{name}_end.png"
        if start.exists() and end.exists():
            pairs[name] = FixturePair(name=name, start=start, end=end, description=desc)

    return pairs


@dataclass
class RunResult:
    """Result of a single test run."""

    fixture: str
    config: str
    status: str  # "OK", "FAIL: exit code N", "SKIP", "DRY-RUN"
    elapsed_secs: float = 0.0
    output_path: str = ""
    stderr_tail: str = ""


def build_command(
    pair: FixturePair,
    config: TestConfig,
    output_path: Path,
) -> list[str]:
    """Build the CLI command for a single test run."""
    cmd = [
        "uv",
        "run",
        "ltx-2-mlx",
        "keyframe",
        "--prompt",
        PROMPT,
        "--start",
        str(pair.start),
        "--end",
        str(pair.end),
        "--output",
        str(output_path),
        "--seed",
        str(SEED),
        "--frames",
        str(FRAMES),
        "--steps",
        str(STEPS),
        *config.extra_args,
    ]
    return cmd


def run_single(
    pair: FixturePair,
    config: TestConfig,
    dry_run: bool = False,
) -> RunResult:
    """Run a single fixture x config combination."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pair.name}_{config.name}.mp4"
    cmd = build_command(pair, config, output_path)

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return RunResult(
            fixture=pair.name,
            config=config.name,
            status="DRY-RUN",
            output_path=str(output_path),
        )

    print(f"  Running {pair.name} x {config.name} ...")
    print(f"    cmd: {' '.join(cmd)}")

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per run
            cwd=str(ROOT),
        )
        elapsed = time.monotonic() - t0
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-5:]) if result.stderr else ""

        if result.returncode == 0:
            status = f"OK {elapsed:.0f}s"
        else:
            status = f"FAIL: exit code {result.returncode}"
            print(f"    FAILED (exit {result.returncode})")
            if stderr_tail:
                print(f"    stderr: {stderr_tail[:200]}")

        return RunResult(
            fixture=pair.name,
            config=config.name,
            status=status,
            elapsed_secs=elapsed,
            output_path=str(output_path),
            stderr_tail=stderr_tail,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        print(f"    TIMEOUT after {elapsed:.0f}s")
        return RunResult(
            fixture=pair.name,
            config=config.name,
            status="FAIL: timeout",
            elapsed_secs=elapsed,
            output_path=str(output_path),
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"    ERROR: {e}")
        return RunResult(
            fixture=pair.name,
            config=config.name,
            status=f"FAIL: {e}",
            elapsed_secs=elapsed,
            output_path=str(output_path),
        )


def generate_report(results: list[RunResult], pairs: dict[str, FixturePair]) -> str:
    """Generate the Markdown report from test results."""
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Build results lookup
    result_map: dict[tuple[str, str], RunResult] = {}
    for r in results:
        result_map[(r.fixture, r.config)] = r

    fixture_names = list(pairs.keys())
    config_names = sorted({r.config for r in results})

    # Results table rows
    rows = []
    for fname in fixture_names:
        cells = [fname]
        for cname in config_names:
            r = result_map.get((fname, cname))
            cells.append(r.status if r else "—")
        rows.append("| " + " | ".join(cells) + " |")

    results_table = "\n".join(rows)
    header_row = "| Fixture | " + " | ".join(f"Config {c}" for c in config_names) + " |"
    separator = "|" + "|".join("---------" for _ in range(len(config_names) + 1)) + "|"

    # Fixture descriptions table
    fixture_rows = []
    for name, pair in pairs.items():
        fixture_rows.append(f"| {name} | {pair.description} |")
    fixture_table = "\n".join(fixture_rows)

    report = f"""# Keyframe Interpolation Divergence Test Matrix

_Generated: {timestamp}_

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

{header_row}
{separator}
{results_table}

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
{fixture_table}

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

All tests use seed **{SEED}**, {FRAMES} frames, {STEPS} steps.
"""
    return report


def main() -> None:
    """Run the keyframe divergence test matrix."""
    parser = argparse.ArgumentParser(description="Run keyframe interpolation divergence test matrix")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Comma-separated configs to run (default: all). Example: A,B,C,D",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated fixture pairs (default: all). Example: existing,solid_colors",
    )
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Just generate fixtures, don't run tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    args = parser.parse_args()

    # Generate fixtures if needed
    if args.fixtures_only or not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.iterdir()):
        print("Generating fixtures...")
        from scripts.keyframe_tests.generate_fixtures import main as gen_main

        gen_main()
        if args.fixtures_only:
            return

    # Resolve pairs
    all_pairs = get_fixture_pairs()
    if args.pairs:
        selected_pair_names = [p.strip() for p in args.pairs.split(",")]
        pairs = {k: v for k, v in all_pairs.items() if k in selected_pair_names}
        missing = set(selected_pair_names) - set(pairs.keys())
        if missing:
            print(f"Warning: fixture pairs not found: {missing}")
            print(f"Available: {list(all_pairs.keys())}")
    else:
        pairs = all_pairs

    if not pairs:
        print("No fixture pairs found. Run with --fixtures-only first.")
        return

    # Resolve configs
    if args.config:
        selected_configs = [c.strip().upper() for c in args.config.split(",")]
        configs = {k: v for k, v in CONFIGS.items() if k in selected_configs}
    else:
        configs = CONFIGS

    print(f"\nTest matrix: {len(pairs)} pairs x {len(configs)} configs = {len(pairs) * len(configs)} runs")
    print(f"Pairs: {list(pairs.keys())}")
    print(f"Configs: {list(configs.keys())}")
    print()

    # Run matrix
    results: list[RunResult] = []
    for _pair_name, pair in pairs.items():
        for _config_name, config in configs.items():
            result = run_single(pair, config, dry_run=args.dry_run)
            results.append(result)
            print()

    # Generate report
    report = generate_report(results, pairs)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Report written to {REPORT_PATH}")

    # Summary
    total = len(results)
    ok_count = sum(1 for r in results if r.status.startswith("OK"))
    fail_count = sum(1 for r in results if r.status.startswith("FAIL"))
    dry_count = sum(1 for r in results if r.status == "DRY-RUN")

    if dry_count:
        print(f"\nDry run: {dry_count} commands printed")
    else:
        print(f"\nResults: {ok_count}/{total} passed, {fail_count} failed")
        if fail_count:
            print("Failed runs:")
            for r in results:
                if r.status.startswith("FAIL"):
                    print(f"  {r.fixture} x {r.config}: {r.status}")


if __name__ == "__main__":
    main()
