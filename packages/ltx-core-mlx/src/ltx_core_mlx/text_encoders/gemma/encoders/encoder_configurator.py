"""Evidence-based text-encoder selection for the LTX-2.3 / LTX-2.5 packs.

LTX-2.3 packs ship the Gemma 3 12B text encoder as a separate
``mlx-community`` download (no local weight files in the DiT pack
itself). LTX-2.5 packs bundle the Gemma 4 unified text tower directly
in the pack (``text_encoder.safetensors`` + ``text_encoder_config.json``).
:func:`select_text_encoder` keys off that evidence rather than a version
string, so any pack that ships the Gemma 4 files gets routed to
:class:`~ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder.Gemma4TextEncoder`,
and any pack that doesn't keeps using the existing
:class:`~ltx_core_mlx.text_encoders.gemma.encoders.base_encoder.GemmaLanguageModel`
path unchanged.

Mirrors upstream's ``_check_gemma_version`` (``configurator.py``) in
spirit, simplified to what this port's packs actually declare: the real
LTX-2.5 pack's ``embedded_config.json`` has no ``gemma_source_checkpoint``
key at all, so the full upstream fallback logic (requiring the key above
a version floor) has no evidence to key off here. This port only checks
the *mismatch* case -- when the DiT pack does declare an expected Gemma
version, it must agree with the tower it is about to select.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

_TEXT_ENCODER_WEIGHTS = "text_encoder.safetensors"
_TEXT_ENCODER_CONFIG = "text_encoder_config.json"
_EMBEDDED_CONFIG = "embedded_config.json"


def check_gemma_version(model_dir: str | Path) -> None:
    """Raise if the DiT pack's declared Gemma version disagrees with the tower's.

    Reads ``embedded_config.json``'s top-level ``gemma_source_checkpoint``
    key (if present) and compares its ``gemma_version`` against the
    ``gemma_version`` declared at the top level of
    ``text_encoder_config.json``. Absence of either the file or the key is
    not an error -- it means the pack makes no claim to check, matching
    the real LTX-2.5 pack (no ``gemma_source_checkpoint`` in its
    ``embedded_config.json``).

    Args:
        model_dir: Path to the pack directory.

    Raises:
        ValueError: If the pack declares a ``gemma_source_checkpoint``
            whose ``gemma_version`` disagrees with the tower config's.
    """
    model_dir = Path(model_dir)
    embedded_config_path = model_dir / _EMBEDDED_CONFIG
    if not embedded_config_path.exists():
        return

    with open(embedded_config_path) as f:
        embedded_config = json.load(f)

    gemma_source_checkpoint = embedded_config.get("gemma_source_checkpoint")
    if gemma_source_checkpoint is None:
        return

    expected_gemma_version = gemma_source_checkpoint.get("gemma_version")

    text_encoder_config_path = model_dir / _TEXT_ENCODER_CONFIG
    with open(text_encoder_config_path) as f:
        text_encoder_config = json.load(f)
    actual_gemma_version = text_encoder_config.get("gemma_version")

    if expected_gemma_version != actual_gemma_version:
        raise ValueError(
            f"Gemma version mismatch: {model_dir}/{_EMBEDDED_CONFIG}'s "
            f"gemma_source_checkpoint expects gemma_version={expected_gemma_version!r}, "
            f"but {model_dir}/{_TEXT_ENCODER_CONFIG} has gemma_version={actual_gemma_version!r}."
        )


def select_text_encoder(model_dir: str | Path) -> Literal["gemma3", "gemma4"]:
    """Select the text-encoder family from the pack's evidence on disk.

    Args:
        model_dir: Path to the pack directory (the same directory passed
            to the DiT loader).

    Returns:
        ``"gemma4"`` iff both ``text_encoder.safetensors`` and
        ``text_encoder_config.json`` exist directly under ``model_dir``;
        ``"gemma3"`` otherwise (the LTX-2.3 default: Gemma 3 is loaded
        from a separate ``mlx-community`` checkpoint, not the DiT pack).

    Raises:
        ValueError: See :func:`check_gemma_version` -- only reached when
            the evidence selects ``"gemma4"``.
    """
    model_dir = Path(model_dir)
    has_gemma4_weights = (model_dir / _TEXT_ENCODER_WEIGHTS).exists()
    has_gemma4_config = (model_dir / _TEXT_ENCODER_CONFIG).exists()

    if not (has_gemma4_weights and has_gemma4_config):
        return "gemma3"

    check_gemma_version(model_dir)
    return "gemma4"
