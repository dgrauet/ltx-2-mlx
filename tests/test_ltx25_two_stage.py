"""LTX-2.5 pack detection + distilled LoRA resolution in TI2VidTwoStagesPipeline.

The tests exercise pack-evidence-based LoRA resolution (2.5 vs 2.3 defaults),
explicit-argument override, and upsampler path resolution via synthetic packs
shaped like 2.5 configs, 2.3 configs, and missing configs.
"""

import json

import pytest

from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline


def _write_pack_config(tmp_path, *, ltx25: bool) -> None:
    """Write a minimal transformer config marking the dir as a 2.5 / 2.3 pack."""
    transformer: dict = {"num_layers": 48}
    if ltx25:
        transformer["ff_bias"] = False
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))


def _make_two_stage(
    tmp_path,
    monkeypatch,
    *,
    ltx25: bool,
    distilled_lora: str | None = None,
    with_upscaler: str | None = None,
) -> TI2VidTwoStagesPipeline:
    """Build a TI2VidTwoStagesPipeline over a synthetic pack.

    Writes tmp_path/embedded_config.json shaped like a 2.5 or 2.3 config,
    creates an empty tmp_path/<with_upscaler>.safetensors if requested,
    stubs out network-dependent methods via monkeypatch, then constructs
    TI2VidTwoStagesPipeline(model_dir=str(tmp_path), distilled_lora=distilled_lora).

    Args:
        tmp_path: pytest temporary directory.
        monkeypatch: pytest monkeypatch fixture.
        ltx25: When True, write ff_bias=False (LTX-2.5 signal).
        distilled_lora: Optional override for the distilled LoRA filename.
        with_upscaler: Optional upsampler filename to create as an empty file.

    Returns:
        Constructed TI2VidTwoStagesPipeline instance.
    """
    _write_pack_config(tmp_path, ltx25=ltx25)

    if with_upscaler:
        (tmp_path / with_upscaler).touch()

    return TI2VidTwoStagesPipeline(
        model_dir=str(tmp_path),
        distilled_lora=distilled_lora,
    )


def _make_two_stage_hq(
    tmp_path,
    monkeypatch,
    *,
    ltx25: bool,
    distilled_lora: str | None = None,
    with_upscaler: str | None = None,
) -> TI2VidTwoStagesHQPipeline:
    """Build a TI2VidTwoStagesHQPipeline over a synthetic pack.

    Same as _make_two_stage but constructs TI2VidTwoStagesHQPipeline instead.
    """
    _write_pack_config(tmp_path, ltx25=ltx25)

    if with_upscaler:
        (tmp_path / with_upscaler).touch()

    return TI2VidTwoStagesHQPipeline(
        model_dir=str(tmp_path),
        distilled_lora=distilled_lora,
    )


def test_25_pack_sets_is_25_true(tmp_path, monkeypatch):
    """Verify that an LTX-2.5 pack sets _is_25 to True."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=True)
    assert pipe._is_25 is True


def test_23_pack_sets_is_25_false(tmp_path, monkeypatch):
    """Verify that an LTX-2.3 pack sets _is_25 to False."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=False)
    assert pipe._is_25 is False


def test_25_pack_resolves_450_lora_by_default(tmp_path, monkeypatch):
    """Verify that an LTX-2.5 pack resolves to the 2.5 distilled LoRA when unspecified."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=True)
    assert pipe._distilled_lora == "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"


def test_23_pack_resolves_384_lora_by_default(tmp_path, monkeypatch):
    """Verify that an LTX-2.3 pack resolves to the 2.3 distilled LoRA when unspecified."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=False)
    assert pipe._distilled_lora == "ltx-2.3-22b-distilled-lora-384.safetensors"


def test_explicit_lora_wins_on_25_pack(tmp_path, monkeypatch):
    """Verify that an explicit distilled_lora overrides the 2.5 default."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=True, distilled_lora="custom.safetensors")
    assert pipe._distilled_lora == "custom.safetensors"


def test_explicit_lora_wins_on_23_pack(tmp_path, monkeypatch):
    """Verify that an explicit distilled_lora overrides the 2.3 default."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=False, distilled_lora="custom.safetensors")
    assert pipe._distilled_lora == "custom.safetensors"


def test_25_pack_upsampler_resolves_v1_0(tmp_path, monkeypatch):
    """Verify that an LTX-2.5 pack picks the v1_0 upsampler when present."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=True, with_upscaler="spatial_upscaler_x2_v1_0.safetensors")
    assert pipe._resolve_upsampler_path().name == "spatial_upscaler_x2_v1_0.safetensors"


def test_23_pack_keeps_v1_1_upsampler(tmp_path, monkeypatch):
    """Verify that an LTX-2.3 pack keeps the v1_1 upsampler."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=False, with_upscaler="spatial_upscaler_x2_v1_1.safetensors")
    assert pipe._resolve_upsampler_path().name == "spatial_upscaler_x2_v1_1.safetensors"


def test_teacache_raises_on_25_pack(tmp_path, monkeypatch):
    """Verify that enable_teacache=True raises ValueError on an LTX-2.5 pack."""
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=True)
    with pytest.raises(ValueError, match="TeaCache"):
        pipe.generate_two_stage(prompt="a fox", frame_rate=24.0, enable_teacache=True)


class _AbortError(Exception):
    """Sentinel exception for testing that guard allows 2.3 path to proceed."""

    pass


def test_teacache_still_allowed_on_23_pack(tmp_path, monkeypatch):
    """Verify that the TeaCache guard does not raise on an LTX-2.3 pack.

    We plant a probe (_AbortError) right after the guard (in text encoding) and
    verify that it gets raised, not a ValueError from the guard.
    """
    pipe = _make_two_stage(tmp_path, monkeypatch, ltx25=False)

    def boom(*a, **k):
        raise _AbortError

    monkeypatch.setattr(pipe, "_encode_text_with_negative", boom, raising=False)
    with pytest.raises(_AbortError):
        pipe.generate_two_stage(prompt="a fox", frame_rate=24.0, enable_teacache=True)


def test_teacache_raises_on_25_pack_hq(tmp_path, monkeypatch):
    """Verify that enable_teacache=True raises ValueError on an LTX-2.5 pack (HQ pipeline)."""
    pipe = _make_two_stage_hq(tmp_path, monkeypatch, ltx25=True)
    with pytest.raises(ValueError, match="TeaCache"):
        pipe.generate_two_stage(prompt="a fox", frame_rate=24.0, enable_teacache=True)


def test_teacache_still_allowed_on_23_pack_hq(tmp_path, monkeypatch):
    """Verify that the TeaCache guard does not raise on an LTX-2.3 pack (HQ pipeline).

    We plant a probe (_AbortError) right after the guard (in text encoding) and
    verify that it gets raised, not a ValueError from the guard.
    """
    pipe = _make_two_stage_hq(tmp_path, monkeypatch, ltx25=False)

    def boom(*a, **k):
        raise _AbortError

    monkeypatch.setattr(pipe, "_encode_text_with_negative", boom, raising=False)
    with pytest.raises(_AbortError):
        pipe.generate_two_stage(prompt="a fox", frame_rate=24.0, enable_teacache=True)


def test_cli_generate_distilled_lora_defaults_to_none(monkeypatch):
    """Regression for the C1 review finding: the CLI must not reintroduce a
    literal ``--distilled-lora`` default, or the pack-evidence resolution in
    ``TI2VidTwoStagesPipeline.__init__`` (2.5 -> ...-450-bf16, 2.3 -> ...-384)
    is unreachable from ``generate --two-stage``. This test parses real argv
    through the real CLI parser (not a hand-built parser) so it fails the
    same way #C1 did if the default is ever reintroduced.
    """
    import ltx_pipelines_mlx.cli as cli

    captured: dict = {}

    def _capture(args):
        captured["distilled_lora"] = args.distilled_lora

    monkeypatch.setattr(cli, "_cmd_generate", _capture)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ltx-2-mlx",
            "generate",
            "--two-stage",
            "--prompt",
            "a fox",
            "--frame-rate",
            "24",
            "-o",
            "out.mp4",
        ],
    )

    cli.main()

    assert captured["distilled_lora"] is None
