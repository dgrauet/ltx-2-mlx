"""Structural coverage for DistilledPipeline telemetry boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

_DISTILLED = (
    Path(__file__).parents[1]
    / "packages"
    / "ltx-pipelines-mlx"
    / "src"
    / "ltx_pipelines_mlx"
    / "distilled.py"
)

_EXPECTED_CALLS = {
    "Loading VAE encoder": {"_load_vae_encoder"},
    "Loading latent upsampler": {"_load_upsampler"},
    "Preparing Stage 1 conditioning/state": {"combined_image_conditionings", "create_noised_state"},
    "Stage 1 half-resolution denoise": {"_pre_denoise_flush", "denoise_loop", "aggressive_cleanup"},
    "Latent upscale": {"unpatchify", "denormalize_latent", "normalize_latent", "_materialize"},
    "Preparing Stage 2 conditioning/state": {"combined_image_conditionings", "patchify", "create_noised_state"},
    "Stage 2 full-resolution denoise": {"_pre_denoise_flush", "denoise_loop", "aggressive_cleanup"},
}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_distilled_profile_phases_cover_expensive_operations_without_human_output() -> None:
    tree = ast.parse(_DISTILLED.read_text())
    phases: dict[str, ast.With] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        context = node.items[0].context_expr
        if not isinstance(context, ast.Call) or _call_name(context) != "phase" or not context.args:
            continue
        label = ast.literal_eval(context.args[0])
        phases[label] = node
        if label in _EXPECTED_CALLS:
            verbose = next(keyword.value for keyword in context.keywords if keyword.arg == "verbose")
            assert isinstance(verbose, ast.Constant) and verbose.value is False

    assert _EXPECTED_CALLS.keys() <= phases.keys()
    for label, expected in _EXPECTED_CALLS.items():
        actual = {_call_name(node) for node in ast.walk(phases[label]) if isinstance(node, ast.Call)}
        assert expected <= actual, f"{label} no longer contains {expected - actual}"
