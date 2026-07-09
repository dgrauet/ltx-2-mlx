"""Unit tests for IC-LoRA dev-mode LoRA fusion + dev-transformer detection.

Covers the two behavioral guarantees added when running the Comfy IC-LoRA recipe
(dev checkpoint + task IC-LoRA @1.0 + distilled LoRA @0.5 fused together):

- ``ICLoraPipeline._effective_lora_paths`` — appends the distilled LoRA in dev
  mode (relative paths resolved against ``model_dir``, absolute passed through),
  leaves the IC-LoRA list untouched in distilled mode, and hard-fails on a
  missing distilled LoRA. This is the core of the fix: a future refactor that
  drops the distilled LoRA from the generation pass (the original bug) breaks
  these tests.
- ``ICLoraPipeline.__init__`` — a supplied ``--dev-transformer`` that does not
  exist raises immediately rather than silently reverting to distilled mode
  (which would also drop the distilled LoRA and reproduce the bad-output config).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltx_pipelines_mlx.ic_lora import ICLoraPipeline


def _bare_pipe(
    model_dir: Path,
    *,
    dev_mode: bool,
    distilled_lora_path: str | None,
    distilled_lora_strength: float = 0.5,
    lora_paths: list[tuple[str, float]] | None = None,
) -> ICLoraPipeline:
    """Construct just enough of ICLoraPipeline to exercise _effective_lora_paths.

    Bypasses the heavy __init__ (model dir resolve, block construction, LoRA
    download) via object.__new__ and sets only the attributes the method reads.
    """
    pipe = object.__new__(ICLoraPipeline)
    pipe.model_dir = Path(model_dir)
    pipe.dev_mode = dev_mode
    pipe.distilled_lora_path = distilled_lora_path
    pipe.distilled_lora_strength = distilled_lora_strength
    pipe._lora_paths = list(lora_paths or [])
    return pipe


# --- _effective_lora_paths ---------------------------------------------------


def test_distilled_mode_returns_ic_lora_only(tmp_path):
    """Non-dev mode ignores the distilled LoRA even if one is configured."""
    ic = [("ic.safetensors", 1.0)]
    pipe = _bare_pipe(tmp_path, dev_mode=False, distilled_lora_path="distilled.safetensors", lora_paths=ic)
    assert pipe._effective_lora_paths() == ic


def test_dev_mode_appends_distilled_relative(tmp_path):
    """Dev mode appends the distilled LoRA at its strength, resolved vs model_dir."""
    distilled = tmp_path / "distilled.safetensors"
    distilled.write_bytes(b"")
    ic = [("ic.safetensors", 1.0)]
    pipe = _bare_pipe(
        tmp_path,
        dev_mode=True,
        distilled_lora_path="distilled.safetensors",
        distilled_lora_strength=0.5,
        lora_paths=ic,
    )
    assert pipe._effective_lora_paths() == [*ic, (str(distilled), 0.5)]


def test_dev_mode_appends_distilled_absolute(tmp_path):
    """An absolute distilled path is passed through unchanged (not re-rooted)."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    distilled = other / "distilled.safetensors"
    distilled.write_bytes(b"")
    pipe = _bare_pipe(
        tmp_path,
        dev_mode=True,
        distilled_lora_path=str(distilled),
        distilled_lora_strength=0.25,
        lora_paths=[("ic.safetensors", 1.0)],
    )
    assert pipe._effective_lora_paths()[-1] == (str(distilled), 0.25)


def test_dev_mode_missing_distilled_raises(tmp_path):
    """A configured-but-absent distilled LoRA fails fast with a clear error."""
    pipe = _bare_pipe(tmp_path, dev_mode=True, distilled_lora_path="nope.safetensors", lora_paths=[])
    with pytest.raises(FileNotFoundError, match="Distilled LoRA not found"):
        pipe._effective_lora_paths()


def test_dev_mode_without_distilled_configured(tmp_path):
    """Dev mode with no distilled LoRA set leaves the IC-LoRA list untouched."""
    ic = [("ic.safetensors", 1.0)]
    pipe = _bare_pipe(tmp_path, dev_mode=True, distilled_lora_path=None, lora_paths=ic)
    assert pipe._effective_lora_paths() == ic


# --- dev-transformer detection (via real __init__) ---------------------------


def test_missing_dev_transformer_hard_fails(tmp_path):
    """A supplied --dev-transformer that doesn't exist raises, no silent fallback."""
    with pytest.raises(FileNotFoundError, match="dev-transformer"):
        ICLoraPipeline(model_dir=str(tmp_path), lora_paths=None, dev_transformer="missing.safetensors")


def test_present_dev_transformer_enables_dev_mode(tmp_path):
    (tmp_path / "transformer.safetensors").write_bytes(b"")
    pipe = ICLoraPipeline(model_dir=str(tmp_path), lora_paths=None, dev_transformer="transformer.safetensors")
    assert pipe.dev_mode is True
    assert pipe.dev_transformer_name == "transformer.safetensors"


def test_no_dev_transformer_stays_distilled_mode(tmp_path):
    pipe = ICLoraPipeline(model_dir=str(tmp_path), lora_paths=None)
    assert pipe.dev_mode is False
