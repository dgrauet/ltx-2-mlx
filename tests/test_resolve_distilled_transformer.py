"""``BasePipeline._resolve_distilled_transformer``: one rule for every distilled loader (#211)."""

from pathlib import Path

from ltx_pipelines_mlx._base import BasePipeline


def _touch(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def test_single_transformer_pack_wins(tmp_path: Path) -> None:
    single = _touch(tmp_path / "transformer.safetensors")
    _touch(tmp_path / "transformer-distilled-1.1.safetensors")
    assert BasePipeline._resolve_distilled_transformer(tmp_path) == single


def test_latest_versioned_distilled(tmp_path: Path) -> None:
    _touch(tmp_path / "transformer-distilled.safetensors")
    _touch(tmp_path / "transformer-distilled-1.0.safetensors")
    latest = _touch(tmp_path / "transformer-distilled-1.1.safetensors")
    assert BasePipeline._resolve_distilled_transformer(tmp_path) == latest


def test_unversioned_distilled(tmp_path: Path) -> None:
    plain = _touch(tmp_path / "transformer-distilled.safetensors")
    assert BasePipeline._resolve_distilled_transformer(tmp_path) == plain


def test_missing_returns_canonical_path(tmp_path: Path) -> None:
    """Nothing on disk: the canonical name comes back so the loader raises a clear FileNotFoundError."""
    assert BasePipeline._resolve_distilled_transformer(tmp_path) == tmp_path / "transformer-distilled.safetensors"
