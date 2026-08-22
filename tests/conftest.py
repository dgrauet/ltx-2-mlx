"""Shared test fixtures and helpers."""

from pathlib import Path

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
