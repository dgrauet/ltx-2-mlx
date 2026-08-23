"""End-to-end tests for stepwise previews, driven through the CLI.

The unit tests in ``test_stepwise.py`` cover window resolution, naming and
failure isolation with stub decoders. They cannot answer the two questions that
actually matter once the feature ships:

1. Does a real generation produce previews that are valid, animated, and
   openable?
2. Does asking for previews change the generated video?

(2) is the load-bearing one. Previews tap the denoising loop and hold the VAE
decoder resident through it, so "opt-in and inert" is a claim about a code path
that unit tests with stub decoders never exercise. These tests assert it by
running the same seed twice through the real CLI and comparing the output MP4
byte for byte.

Marked ``slow``: needs the q8 weights and runs two full generations
(~4 minutes on an M2 Pro). Run with::

    uv run pytest tests/test_stepwise_e2e.py -m slow -v -s
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from tests.conftest import MODEL_DIR

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(MODEL_DIR is None, reason="q8 weights not found"),
]

# Small enough to run twice in a few minutes, large enough to be a real
# two-stage distilled generation with audio.
HEIGHT, WIDTH, FRAMES = 512, 512, 25
SEED = 81647281
PROMPT = "a heavy wooden door creaks slowly open in an old stone house, dust swirling in a shaft of afternoon light"

# Distilled two-stage: DISTILLED_SIGMAS yields 8 steps, STAGE_2_SIGMAS yields 3.
EXPECTED_STAGE_STEPS = {1: 8, 2: 3}

# (FRAMES - 1) // 8 + 1 latent frames; the default 8-frame window clamps to it.
EXPECTED_LATENT_FRAMES = (FRAMES - 1) // 8 + 1
EXPECTED_PREVIEW_PIXEL_FRAMES = 8 * EXPECTED_LATENT_FRAMES - 7

CHUNK_RE = re.compile(r"^seed_(?P<seed>-?\d+)_s(?P<stage>\d+)_step(?P<step>\d+)of(?P<total>\d+)\.webp$")


@dataclass
class RunResult:
    """One CLI generation: its output video, digest and captured stderr."""

    video: Path
    sha256: str
    stderr: str
    previews: list[Path]


def _run_cli(tmp_path: Path, *, preview_dir: Path | None) -> RunResult:
    """Run one generation through the installed CLI and collect its artefacts."""
    video = tmp_path / ("with_previews.mp4" if preview_dir else "baseline.mp4")
    cmd = [
        str(Path(sys.executable).parent / "ltx-2-mlx"),
        "generate",
        "--model",
        "dgrauet/ltx-2.3-mlx-q8",
        "--distilled",
        "--low-ram",
        "--seed",
        str(SEED),
        "-H",
        str(HEIGHT),
        "-W",
        str(WIDTH),
        "-f",
        str(FRAMES),
        "--frame-rate",
        "24",
        "-p",
        PROMPT,
        "-o",
        str(video),
    ]
    if preview_dir is not None:
        cmd += ["--stepwise-image-output-dir", str(preview_dir)]

    env = dict(os.environ)
    # The macOS GPU watchdog kills Gemma encoding on a contended machine (#75);
    # without this the test fails for reasons unrelated to what it asserts.
    env.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    assert proc.returncode == 0, f"CLI failed ({proc.returncode}):\n{proc.stderr[-3000:]}"
    assert video.exists(), f"no output video:\n{proc.stderr[-2000:]}"

    previews = sorted(preview_dir.glob("*.webp")) if preview_dir else []
    return RunResult(
        video=video,
        sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        stderr=proc.stderr,
        previews=previews,
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[RunResult, RunResult]:
    """Two generations with identical seeds: one with previews, one without."""
    root = tmp_path_factory.mktemp("stepwise_e2e")
    preview_dir = root / "previews"
    with_previews = _run_cli(root, preview_dir=preview_dir)
    baseline = _run_cli(root, preview_dir=None)
    return with_previews, baseline


def test_previews_do_not_change_the_generated_video(runs) -> None:
    """The whole safety claim: asking for previews must not alter the output.

    Same seed, same flags, previews the only difference. If the tap into the
    denoising loop perturbed state -- or the resident decoder changed
    allocation enough to move a result -- this is where it shows.
    """
    with_previews, baseline = runs
    assert with_previews.sha256 == baseline.sha256, (
        f"previews changed the output video\n"
        f"  with previews: {with_previews.sha256}\n"
        f"  baseline:      {baseline.sha256}"
    )


def test_one_preview_per_step_across_both_stages(runs) -> None:
    """Every denoising step of both stages is previewed, and nothing else is."""
    with_previews, _ = runs
    by_stage: dict[int, set[int]] = {}
    for path in with_previews.previews:
        match = CHUNK_RE.match(path.name)
        assert match, f"unexpected preview filename: {path.name}"
        assert int(match["seed"]) == SEED, f"wrong seed in {path.name}"
        stage, step, total = int(match["stage"]), int(match["step"]), int(match["total"])
        assert total == EXPECTED_STAGE_STEPS[stage], f"{path.name} claims {total} steps"
        by_stage.setdefault(stage, set()).add(step)

    assert by_stage == {stage: set(range(1, n + 1)) for stage, n in EXPECTED_STAGE_STEPS.items()}


def test_each_preview_is_a_valid_animation_of_the_right_length(runs) -> None:
    """Previews are real animated WebP, not a still or a truncated file.

    The VAE upsamples time 8x, so an N-latent-frame window decodes to 8N-7
    pixel frames -- the point of previewing a window rather than one frame.
    """
    with_previews, _ = runs
    assert with_previews.previews, "no previews were written"
    for path in with_previews.previews:
        with Image.open(path) as img:
            assert img.format == "WEBP", f"{path.name} is {img.format}"
            assert img.n_frames == EXPECTED_PREVIEW_PIXEL_FRAMES, (
                f"{path.name} has {img.n_frames} frames, expected {EXPECTED_PREVIEW_PIXEL_FRAMES}"
            )
            assert img.size == (WIDTH, HEIGHT) or img.size == (WIDTH // 2, HEIGHT // 2), (
                f"{path.name} is {img.size}, expected full or half resolution"
            )


def test_memory_warning_is_printed_whenever_previews_are_enabled(runs) -> None:
    """The jetsam warning fires on any preview run, not just under --low-ram.

    A jetsam kill is not an exception, so the handler's failure containment
    cannot cover it; this line is the only thing that points a user at
    previews afterwards.
    """
    with_previews, baseline = runs
    assert "[stepwise] warning:" in with_previews.stderr
    assert "killed outright" in with_previews.stderr
    # --low-ram adds its own sharper second line.
    assert "opposite of what --low-ram is for" in with_previews.stderr
    # And nothing at all when previews are off.
    assert "[stepwise]" not in baseline.stderr
