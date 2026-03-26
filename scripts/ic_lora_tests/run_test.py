"""Run IC-LoRA pipeline smoke tests.

Downloads the official IC-LoRA weights from HuggingFace and runs the pipeline
with synthetic control video fixtures.

Usage:
    # Generate fixtures first
    uv run python scripts/ic_lora_tests/generate_fixtures.py

    # Run tests (downloads LoRAs on first run)
    uv run python scripts/ic_lora_tests/run_test.py [--lora union|motion|all] [--dry-run]

    # Quick test at half resolution (skip stage 2)
    uv run python scripts/ic_lora_tests/run_test.py --skip-stage-2
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

SEED = 712577398
FRAMES = 33

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "ic_lora"
OUTPUT_DIR = ROOT / "tests" / "outputs" / "ic_lora"

LORA_CONFIGS = {
    "union": {
        "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "control_video": "depth_control.mp4",
        "prompt": "A bright sphere moving across a room with depth perspective",
    },
    "union_canny": {
        "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "control_video": "canny_control.mp4",
        "prompt": "A geometric shape moving with clear edges and outlines",
    },
    "motion": {
        "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        "control_video": "motion_track_control.mp4",
        "prompt": "Three colored particles moving in smooth trajectories",
    },
}


@dataclass
class RunResult:
    name: str
    status: str
    elapsed_secs: float = 0.0
    output_path: str = ""


def run_single(
    name: str,
    config: dict,
    dry_run: bool = False,
    skip_stage_2: bool = False,
) -> RunResult:
    """Run a single IC-LoRA test."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"ic_lora_{name}.mp4"
    control_video = FIXTURES_DIR / config["control_video"]

    if not control_video.exists():
        print(f"  SKIP: fixture {control_video} not found (run generate_fixtures.py first)")
        return RunResult(name=name, status="SKIP (no fixture)")

    cmd = [
        "uv",
        "run",
        "ltx-2-mlx",
        "ic-lora",
        "--prompt",
        config["prompt"],
        "--lora",
        config["repo"],
        "1.0",
        "--video-conditioning",
        str(control_video),
        "1.0",
        "-o",
        str(output_path),
        "--seed",
        str(SEED),
        "--frames",
        str(FRAMES),
    ]

    if skip_stage_2:
        cmd.append("--skip-stage-2")

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return RunResult(name=name, status="DRY-RUN", output_path=str(output_path))

    print(f"  Running {name} ...")
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            status = f"FAIL (exit {result.returncode})"
            print(f"    {status}")
            if result.stderr:
                # Print last 20 lines of stderr
                lines = result.stderr.strip().split("\n")
                for line in lines[-20:]:
                    print(f"      {line}")
        else:
            status = f"OK ({elapsed:.0f}s)"
            print(f"    {status}")

        return RunResult(name=name, status=status, elapsed_secs=elapsed, output_path=str(output_path))

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        print(f"    TIMEOUT after {elapsed:.0f}s")
        return RunResult(name=name, status="TIMEOUT", elapsed_secs=elapsed)


def main() -> None:
    """Run IC-LoRA tests."""
    parser = argparse.ArgumentParser(description="Run IC-LoRA pipeline tests")
    parser.add_argument(
        "--lora",
        choices=["union", "union_canny", "motion", "all"],
        default="all",
        help="Which LoRA test to run (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--skip-stage-2", action="store_true", help="Skip stage 2 (faster, half resolution)")
    args = parser.parse_args()

    if args.lora == "all":
        configs = LORA_CONFIGS
    else:
        configs = {args.lora: LORA_CONFIGS[args.lora]}

    print(f"IC-LoRA tests: {list(configs.keys())}")
    if args.skip_stage_2:
        print("  (skip-stage-2 enabled)")
    print()

    results = []
    for name, config in configs.items():
        result = run_single(name, config, dry_run=args.dry_run, skip_stage_2=args.skip_stage_2)
        results.append(result)
        print()

    # Summary
    print("=" * 50)
    print("Summary:")
    for r in results:
        print(f"  {r.name:<20s} {r.status}")

    ok = sum(1 for r in results if r.status.startswith("OK"))
    print(f"\n{ok}/{len(results)} passed")


if __name__ == "__main__":
    main()
