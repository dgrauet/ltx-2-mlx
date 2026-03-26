"""Run keyframe interpolation test matrix with realistic fixtures.

Runs each fixture pair with the dev model + CFG pipeline and produces
a Markdown report summarizing results and timings.

Usage:
    uv run python scripts/keyframe_tests/run_matrix.py [--pairs hedgehog,...] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SEED = 712577398
FRAMES = 33
PROMPT_MAP = {
    "hedgehog": "A 3D animated hedgehog character in the rain, smooth camera transition",
    "forest_zoom": "A camera slowly zooming into a cat sitting in a lush green forest",
    "day_to_night": "A forest scene transitioning from bright daylight to dark blue dusk",
    "hedgehog_seasons": "A 3D animated hedgehog transitioning from warm autumn to cold winter atmosphere",
    "forest_blur": "A cat in a forest with the camera focus gradually shifting from sharp to soft bokeh blur",
}

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "keyframe_pairs"
OUTPUT_DIR = ROOT / "tests" / "outputs" / "keyframe_matrix"
REPORT_PATH = ROOT / "docs" / "keyframe-bugs.md"

MODEL_ARGS = [
    "--model",
    "dgrauet/ltx-2.3-mlx-q8",
    "--dev-transformer",
    "transformer-dev.safetensors",
    "--distilled-lora",
    "ltx-2.3-22b-distilled-lora-384.safetensors",
    "--cfg-scale",
    "3.0",
]


@dataclass
class FixturePair:
    name: str
    start: Path
    end: Path
    prompt: str


@dataclass
class RunResult:
    fixture: str
    status: str
    elapsed_secs: float = 0.0
    output_path: str = ""
    resolution: str = ""


def get_fixture_pairs() -> dict[str, FixturePair]:
    """Return all available fixture pairs."""
    pairs: dict[str, FixturePair] = {}
    for name, prompt in PROMPT_MAP.items():
        start = FIXTURES_DIR / f"{name}_start.png"
        end = FIXTURES_DIR / f"{name}_end.png"
        if start.exists() and end.exists():
            pairs[name] = FixturePair(name=name, start=start, end=end, prompt=prompt)
    return pairs


def run_single(pair: FixturePair, dry_run: bool = False) -> RunResult:
    """Run a single fixture pair."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pair.name}.mp4"
    cmd = [
        "uv",
        "run",
        "ltx-2-mlx",
        "keyframe",
        "--prompt",
        pair.prompt,
        "--start",
        str(pair.start),
        "--end",
        str(pair.end),
        "-o",
        str(output_path),
        "--seed",
        str(SEED),
        "--frames",
        str(FRAMES),
        *MODEL_ARGS,
    ]

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return RunResult(fixture=pair.name, status="DRY-RUN", output_path=str(output_path))

    print(f"  Running {pair.name} ...")
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
        elapsed = time.monotonic() - t0

        # Get resolution via ffprobe
        resolution = ""
        if result.returncode == 0:
            try:
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "quiet",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height",
                        "-of",
                        "csv=p=0",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                )
                resolution = probe.stdout.strip()
            except Exception:
                pass

        status = f"OK ({elapsed:.0f}s)" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
        print(f"    {status} {resolution}")
        return RunResult(
            fixture=pair.name, status=status, elapsed_secs=elapsed, output_path=str(output_path), resolution=resolution
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        print(f"    TIMEOUT after {elapsed:.0f}s")
        return RunResult(fixture=pair.name, status="TIMEOUT", elapsed_secs=elapsed, output_path=str(output_path))


def generate_report(results: list[RunResult]) -> str:
    """Generate the Markdown report from test results."""
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    rows = []
    for r in results:
        rows.append(f"| {r.fixture} | {r.resolution} | {r.status} |")
    results_table = "\n".join(rows)

    return f"""# Keyframe Interpolation Test Matrix

_Generated: {timestamp}_

## Configuration

- **Model:** dgrauet/ltx-2.3-mlx-q8 (dev transformer + distilled LoRA)
- **CFG scale:** 3.0
- **Seed:** {SEED}
- **Frames:** {FRAMES}

## Results

| Fixture | Resolution | Status |
|---------|-----------|--------|
{results_table}

## Reproduction

```bash
# Generate fixtures from source photos
uv run python scripts/keyframe_tests/generate_fixtures.py

# Run full matrix
uv run python scripts/keyframe_tests/run_matrix.py

# Run specific pairs
uv run python scripts/keyframe_tests/run_matrix.py --pairs hedgehog,forest_zoom

# Dry run
uv run python scripts/keyframe_tests/run_matrix.py --dry-run
```
"""


def main() -> None:
    """Run the keyframe test matrix."""
    parser = argparse.ArgumentParser(description="Run keyframe interpolation test matrix")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated fixture pairs (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    all_pairs = get_fixture_pairs()
    if args.pairs:
        selected = [p.strip() for p in args.pairs.split(",")]
        pairs = {k: v for k, v in all_pairs.items() if k in selected}
    else:
        pairs = all_pairs

    if not pairs:
        print("No fixture pairs found. Run generate_fixtures.py first.")
        return

    print(f"Test matrix: {len(pairs)} pairs")
    print(f"Pairs: {list(pairs.keys())}\n")

    results: list[RunResult] = []
    for pair in pairs.values():
        result = run_single(pair, dry_run=args.dry_run)
        results.append(result)
        print()

    report = generate_report(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Report: {REPORT_PATH}")

    ok = sum(1 for r in results if r.status.startswith("OK"))
    print(f"Results: {ok}/{len(results)} passed")


if __name__ == "__main__":
    main()
