"""LTX-2.5 pack detection + distilled LoRA resolution in TI2VidTwoStagesPipeline.

The tests exercise pack-evidence-based LoRA resolution (2.5 vs 2.3 defaults),
explicit-argument override, and upsampler path resolution via synthetic packs
shaped like 2.5 configs, 2.3 configs, and missing configs.
"""

import json

from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline


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

    # Stub out network-dependent methods
    def stub_load_text_encoder():
        pass

    def stub_encode_text(prompt):
        import mlx.core as mx

        return (
            mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
            mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
        )

    monkeypatch.setattr(
        "ltx_pipelines_mlx._base.BasePipeline._load_text_encoder",
        stub_load_text_encoder,
    )
    monkeypatch.setattr(
        "ltx_pipelines_mlx._base.BasePipeline._encode_text",
        stub_encode_text,
    )

    return TI2VidTwoStagesPipeline(
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
