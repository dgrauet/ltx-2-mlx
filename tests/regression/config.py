"""Frozen configuration for regression goldens.

Do not modify the SEED / PROMPT / MODEL / resolution constants without
re-capturing the baseline (manifest.json becomes meaningless if inputs
drift).
"""

from __future__ import annotations

SEED = 9803402

PROMPT = (
    "A young woman with long auburn hair walks slowly along a sunlit "
    "cobblestone street in Lisbon at golden hour, her white linen dress "
    "gently swaying with each step. The camera tracks beside her at eye "
    "level, then slowly pulls back to reveal pastel-colored buildings and "
    "a yellow tram passing in the background. Warm afternoon light filters "
    "through jacaranda trees, casting dappled shadows across the ground. "
    "Shallow depth of field, soft 35mm film grain, cinematic natural "
    "color grading."
)

NEG_PROMPT = ""
MODEL = "dgrauet/ltx-2.3-mlx-q8"
FPS = 24

GEN_HEIGHT, GEN_WIDTH, GEN_FRAMES = 480, 704, 33

# Hedgehog keyframe golden — uses the previously-validated config from
# scripts/keyframe_tests/run_matrix.py (deleted in 71f123a). Restored
# fixtures live in tests/fixtures/keyframe_pairs/. Used while working on
# Fix 2 (num_pixel_frames) to evaluate impact on the keyframe regression.
KF_HEDGEHOG_SEED = 712577398
KF_HEDGEHOG_PROMPT = "A 3D animated hedgehog character in the rain, smooth camera transition"
KF_HEDGEHOG_FRAMES = 33
KF_HEDGEHOG_START = "tests/fixtures/keyframe_pairs/hedgehog_start.png"
KF_HEDGEHOG_END = "tests/fixtures/keyframe_pairs/hedgehog_end.png"

GOLDENS: dict[str, dict] = {
    "g1_two_stage": {
        "cli_args": [
            "generate",
            "--prompt",
            PROMPT,
            "--two-stage",
            "--seed",
            str(SEED),
            "-H",
            str(GEN_HEIGHT),
            "-W",
            str(GEN_WIDTH),
            "-f",
            str(GEN_FRAMES),
        ],
    },
    "g2_hq": {
        "cli_args": [
            "generate",
            "--prompt",
            PROMPT,
            "--hq",
            "--seed",
            str(SEED),
            "-H",
            str(GEN_HEIGHT),
            "-W",
            str(GEN_WIDTH),
            "-f",
            str(GEN_FRAMES),
        ],
    },
    "g3b_hedgehog": {
        # The removed `keyframe` command's upstream v1.5.0 replacement; compare against the
        # `v0_18_0` tag (older tags were captured with `keyframe`).
        "cli_args": [
            "generate",
            "--two-stage",
            "--prompt",
            KF_HEDGEHOG_PROMPT,
            "--image",
            KF_HEDGEHOG_START,
            "0",
            "1.0",
            "guide",
            "--image",
            KF_HEDGEHOG_END,
            "last",
            "1.0",
            "guide",
            "--refine-audio",
            "--seed",
            str(KF_HEDGEHOG_SEED),
            "-f",
            str(KF_HEDGEHOG_FRAMES),
            "--frame-rate",
            str(FPS),
            "--dev-transformer",
            "transformer-dev.safetensors",
            "--distilled-lora",
            "ltx-2.3-22b-distilled-lora-384.safetensors",
            "--cfg-scale",
            "3.0",
        ],
    },
}
