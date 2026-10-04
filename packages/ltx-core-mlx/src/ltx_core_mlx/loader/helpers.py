"""Checkpoint metadata helpers — mirrors upstream ``ltx_core.loader.helpers``.

Only the pure-Python version parser is ported; upstream's meta-model
construction helpers are PyTorch-specific and have no MLX counterpart.
"""

from __future__ import annotations


def parse_model_version(version: str | None) -> tuple[int, ...]:
    """Parse a checkpoint's ``model_version`` into comparable numeric components.

    Mirrors upstream ``ltx_core.loader.helpers.parse_model_version`` verbatim.
    Parsing stops at the first dot-separated component that isn't a plain
    integer, so pre-release tags are dropped (``"2.3.rc1"`` -> ``(2, 3)``). A
    shorter numeric prefix already compares as "less than" a longer one sharing
    its leading components (``(2, 3) < (2, 4, 0)``), so dropping the tail is
    safe. An unset or non-numeric version parses to ``()``, which compares below
    every real version, so callers get their oldest fallback.

    Tags are not always dot-separated (``"2.4-rc2"`` parses to ``(2,)``). Callers
    that want such a build to compare equal to its own generation normalize the
    separator first (see ``detect_model_version`` in ltx-pipelines-mlx).

    Args:
        version: The raw ``model_version`` string, or ``None``.

    Returns:
        The leading integer components of ``version``.
    """
    if not version:
        return ()
    numeric_parts = []
    for part in version.split("."):
        if not part.isdigit():
            break
        numeric_parts.append(int(part))
    return tuple(numeric_parts)
