"""Image conditioning CRF resolved from the model generation (upstream ``resolve_crf``, issue #185)."""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.loader import parse_model_version
from ltx_pipelines_mlx.utils.args import ImageAction, ImageConditioningInput
from ltx_pipelines_mlx.utils.blocks import ImageConditioner
from ltx_pipelines_mlx.utils.constants import (
    DEFAULT_IMAGE_CRF,
    LTX_2_3_PARAMS,
    LTX_2_4_IMAGE_CRF,
    detect_model_version,
    detect_params,
)
from ltx_pipelines_mlx.utils.media_io import load_image_and_preprocess, preprocess


def _write(path: Path, version: str | None) -> Path:
    metadata = {"model_version": version} if version is not None else {}
    mx.save_safetensors(str(path), {"w": mx.zeros((2,))}, metadata=metadata)
    return path


def _parse(*values: str) -> list[ImageConditioningInput]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action=ImageAction, nargs="+", dest="images", default=None)
    return parser.parse_args(["--image", *values]).images


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [("2.5.0", (2, 5, 0)), ("2.3.rc1", (2, 3)), ("2.4-rc2", (2,)), ("", ()), (None, ()), ("abc", ())],
)
def test_parse_model_version(raw: str | None, parsed: tuple[int, ...]) -> None:
    assert parse_model_version(raw) == parsed


@pytest.mark.parametrize(
    ("version", "parsed"),
    [("2.5.0", (2, 5, 0)), ("2.4-rc2", (2, 4)), ("2.3.0", (2, 3, 0)), (None, ()), ("", ())],
)
def test_detect_model_version(tmp_path: Path, version: str | None, parsed: tuple[int, ...]) -> None:
    assert detect_model_version(_write(tmp_path / "m.safetensors", version)) == parsed


def test_detect_model_version_unreadable(tmp_path: Path) -> None:
    assert detect_model_version(tmp_path / "missing.safetensors") == ()
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"not a safetensors file")
    assert detect_model_version(garbage) == ()


@pytest.mark.parametrize(
    ("version", "crf"),
    [
        (None, DEFAULT_IMAGE_CRF),
        ("2.0.0", DEFAULT_IMAGE_CRF),
        ("2.3.0", DEFAULT_IMAGE_CRF),
        ("2.4-rc2", LTX_2_4_IMAGE_CRF),
        ("2.4.0", LTX_2_4_IMAGE_CRF),
        ("2.5.0", LTX_2_4_IMAGE_CRF),
        ("3.0", LTX_2_4_IMAGE_CRF),
    ],
)
def test_crf_per_generation(tmp_path: Path, version: str | None, crf: int) -> None:
    assert detect_params(_write(tmp_path / "m.safetensors", version)).default_image_crf == crf


def test_crf_values_and_2_3_params() -> None:
    assert (DEFAULT_IMAGE_CRF, LTX_2_4_IMAGE_CRF) == (33, 18)
    assert LTX_2_3_PARAMS.default_image_crf == 33


def test_conditioner_reads_the_encoder_it_loads(tmp_path: Path) -> None:
    # 2.3 layout: vae_encoder.safetensors, no metadata -> 33.
    _write(tmp_path / "vae_encoder.safetensors", None)
    assert ImageConditioner(tmp_path).default_image_crf == 33

    # 2.5 layout: the conv pair is detected and the conv encoder's version is read.
    pack = tmp_path / "pack25"
    pack.mkdir()
    _write(pack / "vae_decoder_conv.safetensors", "2.5.0")
    _write(pack / "vae_encoder_conv.safetensors", "2.5.0")
    _write(pack / "vae_encoder.safetensors", None)  # a stray unversioned file must not be read
    conditioner = ImageConditioner(pack)
    assert conditioner.encoder_path.name == "vae_encoder_conv.safetensors"
    assert conditioner.default_image_crf == 18


def test_conditioner_without_encoder_file_falls_back(tmp_path: Path) -> None:
    assert ImageConditioner(tmp_path).default_image_crf == DEFAULT_IMAGE_CRF


def test_resolve_crf_keeps_explicit_values(tmp_path: Path) -> None:
    _write(tmp_path / "vae_encoder.safetensors", "2.5.0")
    images = [
        ImageConditioningInput("a.png", 0, 1.0),
        ImageConditioningInput("b.png", 8, 0.5, 0),
        ImageConditioningInput("c.png", 16, 1.0, 33),
    ]
    resolved = ImageConditioner(tmp_path).resolve_crf(images)
    assert [i.crf for i in resolved] == [18, 0, 33]
    assert [(i.path, i.frame_idx, i.strength) for i in resolved] == [(i.path, i.frame_idx, i.strength) for i in images]


def test_default_image_crf_is_lazy_and_cached(tmp_path: Path) -> None:
    conditioner = ImageConditioner(tmp_path)  # no file yet: construction must not read it
    _write(tmp_path / "vae_encoder.safetensors", "2.5.0")
    assert conditioner.default_image_crf == 18
    _write(tmp_path / "vae_encoder.safetensors", None)
    assert conditioner.default_image_crf == 18


def test_image_action_leaves_crf_unset() -> None:
    assert ImageConditioningInput("a.png", 0, 1.0).crf is None
    assert _parse("a.png")[0].crf is None
    assert _parse("a.png", "0", "1.0")[0].crf is None


def test_image_action_parses_explicit_crf() -> None:
    assert _parse("a.png", "0", "1.0", "23")[0].crf == 23
    assert _parse("a.png", "last", "1.0", "0")[0].crf == 0


def test_unresolved_crf_is_refused(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="crf=None"):
        preprocess(image, crf=None)
    assert preprocess(image, crf=0) is image

    from PIL import Image

    path = tmp_path / "a.png"
    Image.fromarray(image).save(path)
    with pytest.raises(ValueError, match="resolve_crf"):
        load_image_and_preprocess(path, 8, 8, crf=None)


def test_keyframe_encode_forwards_the_crf(monkeypatch: pytest.MonkeyPatch) -> None:
    import ltx_pipelines_mlx.keyframe_interpolation as kf

    seen = []

    def fake_prepare(image, height, width, crf):
        seen.append(crf)
        return mx.zeros((1, 3, height, width))

    class _Encoder:
        def encode(self, x: mx.array) -> mx.array:
            return mx.zeros((1, 128, 1, x.shape[3] // 32, x.shape[4] // 32))

    monkeypatch.setattr(kf, "prepare_image_for_encoding", fake_prepare)
    tokens = kf._encode_keyframe(_Encoder(), "a.png", 64, 64, crf=18)  # type: ignore[arg-type]
    assert seen == [18] and tokens.shape == (1, 4, 128)
