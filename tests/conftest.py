"""Shared test fixtures and helpers."""

import os
from pathlib import Path

# Exact float32 matmuls on GPUs with neural accelerators. MLX (0.32) sends float32 GEMM and attention to its NAX
# kernels on macOS 26.2+ with an applegpu_g17 or later GPU (M5) unless MLX_ENABLE_TF32=0; those keep ~10 mantissa
# bits (rel. error ~7e-4 on a plain ``a @ b``, against ~1e-7 on the CPU or with the variable at 0). The float32 tests
# compare against exact references at 1e-5 to catch logic errors (NA window masks, LoRA algebra, weight swaps), which
# a tolerance wide enough for TF32 would hide. MLX reads the variable once, at the first matmul, so it is set here,
# before any test runs. No effect on GPUs without NAX (the CI runners). Tests that need MLX's default path set the
# variable in a subprocess (``test_sparse_attention.py``).
os.environ["MLX_ENABLE_TF32"] = "0"

_HUB_DIR = Path.home() / ".cache/huggingface/hub"
_Q8_CANDIDATES = [
    _HUB_DIR / "models--dgrauet--ltx-2.3-mlx-q8" / "snapshots",
]

# A snapshot without this file cannot serve the weight-gated tests.
_REQUIRED_WEIGHT = "transformer-distilled.safetensors"


def find_q8_model_dir() -> Path | None:
    """Find a complete q8 model snapshot, or None if none is usable.

    The hub keeps one snapshot directory per revision, and a revision only
    symlinks the files that were fetched at it -- so a cache can hold a dozen
    partial snapshots (a lone ``config.json``, say) beside the complete ones.
    Any download that resolves a new revision adds another.

    Every candidate is therefore checked and the most recently modified
    *complete* one wins. Picking by name alone silently selects a partial
    snapshot as soon as one sorts last, which skips every weight-gated test
    while the suite still exits 0.
    """
    complete: list[Path] = []
    for candidate in _Q8_CANDIDATES:
        if not candidate.exists():
            continue
        complete += [d for d in candidate.iterdir() if (d / _REQUIRED_WEIGHT).exists()]
    if not complete:
        return None
    return max(complete, key=lambda d: d.stat().st_mtime)


MODEL_DIR = find_q8_model_dir()


def _local_pack(name: str) -> Path | None:
    """Local (non-hub) converted pack, used by the LTX-2.5 contract tests."""
    candidate = Path.home() / "Work/mlx/models" / name
    return candidate if (candidate / "embedded_config.json").exists() else None


LTX25_Q8_DIR = _local_pack("ltx-2.5-mlx-q8")
