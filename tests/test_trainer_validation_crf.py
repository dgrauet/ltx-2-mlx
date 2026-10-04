"""Trainer validation I2V re-compresses the conditioning image at the checkpoint's CRF (issue #185)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

import ltx_trainer_mlx.validation_sampler as vs
from ltx_pipelines_mlx.utils.constants import DEFAULT_IMAGE_CRF, LTX_2_4_IMAGE_CRF


def _write(path: Path, version: str | None) -> None:
    metadata = {"model_version": version} if version is not None else {}
    mx.save_safetensors(str(path), {"w": mx.zeros((2,))}, metadata=metadata)


def test_validation_image_crf_per_generation(tmp_path: Path) -> None:
    pack23 = tmp_path / "pack23"
    pack23.mkdir()
    _write(pack23 / "vae_encoder.safetensors", None)
    assert vs.validation_image_crf(pack23) == DEFAULT_IMAGE_CRF

    pack25 = tmp_path / "pack25"
    pack25.mkdir()
    _write(pack25 / "vae_decoder_conv.safetensors", "2.5.0")
    _write(pack25 / "vae_encoder_conv.safetensors", "2.5.0")
    assert vs.validation_image_crf(pack25) == LTX_2_4_IMAGE_CRF


class _Encoder:
    def __call__(self, image: mx.array) -> mx.array:
        return mx.zeros((1, 128, 1, 1, 1))


@pytest.mark.parametrize("crf", [LTX_2_4_IMAGE_CRF, DEFAULT_IMAGE_CRF])
def test_image_conditioning_uses_the_sampler_crf(monkeypatch: pytest.MonkeyPatch, crf: int) -> None:
    seen: list[int | None] = []

    def fake_prepare(image, height, width, crf=None):
        seen.append(crf)
        return mx.zeros((1, 3, height, width))

    monkeypatch.setattr(vs, "prepare_image_for_encoding", fake_prepare)
    sampler = vs.ValidationSampler(
        transformer=None,  # type: ignore[arg-type]
        vae_decoder=None,  # type: ignore[arg-type]
        vae_encoder=_Encoder(),  # type: ignore[arg-type]
        image_crf=crf,
    )
    sampler._build_image_conditioning(vs.GenerationConfig(prompt="p", condition_image="x.png", height=32, width=32))
    assert seen == [crf]
