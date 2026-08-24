"""Tests for LTX-2.5 conv video VAE file-name resolution by pack evidence.

The 2.5 conv video VAE ships as ``vae_decoder_conv.safetensors`` /
``vae_encoder_conv.safetensors`` instead of the 2.3 names
(``vae_decoder.safetensors`` / ``vae_encoder.safetensors``). Both loader
call sites (``ImageConditioner.load`` for the encoder, ``VideoDecoder.load``
for the decoder) must pick the name+prefix pair by evidence of the decoder
conv file's presence in the pack directory, and must stay byte-identical
on a 2.3 pack (no conv file present).

The slow ``Test*LoadContract`` classes below pin the load contract of the
four LTX-2.5 media components (video decoder conv, video encoder conv,
audio_vae, vocoder) against the real ``ltx-2.5-mlx-q8`` pack: bidirectional
key-set comparison, header-only reads (mirrors
``tests/test_ltx25_load_contract.py``), constructing each module exactly
the way ``ltx_pipelines_mlx.utils.blocks`` does (same classes, same
remaps) so a silent-no-op — a pack tensor nobody loads, or a param the
pack never feeds — fails loudly instead of being a quiet skip.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import ClassVar

import pytest

from tests.conftest import LTX25_Q8_DIR

_slow_pack_gated = [
    pytest.mark.slow,
    pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found"),
]


def _header_keys(path: Path) -> set[str]:
    """Read a safetensors header without materializing any tensor data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {k for k in header if k != "__metadata__"}


def _strip_prefix(keys: set[str], prefix: str) -> set[str]:
    return {k[len(prefix) :] for k in keys if k.startswith(prefix)}


def _normalize_quant(keys: set[str]) -> set[str]:
    """Fold q8 ``.scales``/``.biases`` tensors onto the ``.weight`` param they quantize."""
    return {
        k.removesuffix(".scales").removesuffix(".biases") + ".weight" if k.endswith((".scales", ".biases")) else k
        for k in keys
    }


def _remap_audio_stats(keys: set[str]) -> set[str]:
    return {k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means") for k in keys}


def _assert_bidirectional(model_keys: set[str], pack_keys: set[str]) -> None:
    missing_in_model = sorted(pack_keys - model_keys)
    unfed_params = sorted(model_keys - pack_keys)
    assert not missing_in_model, f"pack tensors nobody would load: {missing_in_model[:10]}"
    assert not unfed_params, f"model params the pack does not feed: {unfed_params[:10]}"


def test_both_names_present_selects_conv_names(tmp_path):
    from ltx_pipelines_mlx.utils.blocks import _video_vae_names

    (tmp_path / "vae_decoder_conv.safetensors").touch()

    assert _video_vae_names(tmp_path) == ("vae_decoder_conv", "vae_encoder_conv")


def test_conv_file_absent_selects_23_names(tmp_path):
    from ltx_pipelines_mlx.utils.blocks import _video_vae_names

    assert _video_vae_names(tmp_path) == ("vae_decoder", "vae_encoder")


def test_only_23_files_present_selects_23_names(tmp_path):
    from ltx_pipelines_mlx.utils.blocks import _video_vae_names

    (tmp_path / "vae_decoder.safetensors").touch()
    (tmp_path / "vae_encoder.safetensors").touch()

    assert _video_vae_names(tmp_path) == ("vae_decoder", "vae_encoder")


class TestVideoVaeLoadSitesWiring:
    """Spy on ``load_split_safetensors`` to prove both load sites (decoder
    and encoder) resolve their file name + prefix from pack evidence, and
    that the 2.3 path (no conv file) stays byte-identical to before.
    """

    _WEIGHTS: ClassVar[dict[str, str]] = {"some.weight": "TENSOR"}

    def _stub_load_split_safetensors(self, monkeypatch):
        from ltx_pipelines_mlx.utils import blocks as blocks_mod

        load_calls: list[tuple[Path, str]] = []

        def _stub(path, prefix=""):
            load_calls.append((path, prefix))
            return dict(self._WEIGHTS)

        monkeypatch.setattr(blocks_mod, "load_split_safetensors", _stub)
        return blocks_mod, load_calls

    def test_decoder_site_uses_23_names_when_conv_absent(self, tmp_path, monkeypatch):
        from ltx_pipelines_mlx.utils.blocks import VideoDecoder

        blocks_mod, load_calls = self._stub_load_split_safetensors(monkeypatch)

        class _DummyDecoder:
            def load_weights(self, weights):
                self.load_weights_called_with = dict(weights)

        monkeypatch.setattr(blocks_mod, "_VideoVAEDecoder", _DummyDecoder)

        decoder_block = VideoDecoder(model_dir=tmp_path)
        decoder_block.load()

        assert load_calls == [(tmp_path / "vae_decoder.safetensors", "vae_decoder.")]

    def test_decoder_site_uses_conv_names_when_conv_present(self, tmp_path, monkeypatch):
        from ltx_pipelines_mlx.utils.blocks import VideoDecoder

        blocks_mod, load_calls = self._stub_load_split_safetensors(monkeypatch)

        class _DummyDecoder:
            def load_weights(self, weights):
                self.load_weights_called_with = dict(weights)

        monkeypatch.setattr(blocks_mod, "_VideoVAEDecoder", _DummyDecoder)

        (tmp_path / "vae_decoder_conv.safetensors").touch()

        decoder_block = VideoDecoder(model_dir=tmp_path)
        decoder_block.load()

        assert load_calls == [(tmp_path / "vae_decoder_conv.safetensors", "vae_decoder_conv.")]

    def test_encoder_site_uses_23_names_when_conv_absent(self, tmp_path, monkeypatch):
        from ltx_pipelines_mlx.utils.blocks import ImageConditioner

        blocks_mod, load_calls = self._stub_load_split_safetensors(monkeypatch)

        class _DummyEncoder:
            def load_weights(self, weights):
                self.load_weights_called_with = dict(weights)

        monkeypatch.setattr(blocks_mod, "_VideoVAEEncoder", _DummyEncoder)

        encoder_block = ImageConditioner(model_dir=tmp_path)
        encoder_block.load()

        assert load_calls == [(tmp_path / "vae_encoder.safetensors", "vae_encoder.")]

    def test_encoder_site_uses_conv_names_when_conv_present(self, tmp_path, monkeypatch):
        from ltx_pipelines_mlx.utils.blocks import ImageConditioner

        blocks_mod, load_calls = self._stub_load_split_safetensors(monkeypatch)

        class _DummyEncoder:
            def load_weights(self, weights):
                self.load_weights_called_with = dict(weights)

        monkeypatch.setattr(blocks_mod, "_VideoVAEEncoder", _DummyEncoder)

        # Only the decoder-conv file's presence is the documented evidence
        # signal; touch it so both sites pick the conv names.
        (tmp_path / "vae_decoder_conv.safetensors").touch()

        encoder_block = ImageConditioner(model_dir=tmp_path)
        encoder_block.load()

        assert load_calls == [(tmp_path / "vae_encoder_conv.safetensors", "vae_encoder_conv.")]


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
class TestVideoDecoderConvLoadContract:
    """``VideoDecoder`` block against ``vae_decoder_conv.safetensors``."""

    def test_decoder_conv_pack_and_model_fully_consume_each_other(self):
        from mlx.utils import tree_flatten

        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder as _VideoVAEDecoder

        model = _VideoVAEDecoder()
        model_keys = {k for k, _ in tree_flatten(model.parameters())}

        pack_path = LTX25_Q8_DIR / "vae_decoder_conv.safetensors"
        assert pack_path.exists()
        pack = _strip_prefix(_header_keys(pack_path), "vae_decoder_conv.")
        pack = _normalize_quant(pack)

        _assert_bidirectional(model_keys, pack)


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
class TestVideoEncoderConvLoadContract:
    """``ImageConditioner`` block against ``vae_encoder_conv.safetensors``.

    Mirrors ``ImageConditioner.load``'s ``._mean_of_means`` /
    ``._std_of_means`` -> ``.mean_of_means`` / ``.std_of_means`` remap
    applied before ``load_weights``.
    """

    def test_encoder_conv_pack_and_model_fully_consume_each_other(self):
        from mlx.utils import tree_flatten

        from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder as _VideoVAEEncoder

        model = _VideoVAEEncoder()
        model_keys = {k for k, _ in tree_flatten(model.parameters())}

        pack_path = LTX25_Q8_DIR / "vae_encoder_conv.safetensors"
        assert pack_path.exists()
        pack = _strip_prefix(_header_keys(pack_path), "vae_encoder_conv.")
        pack = _remap_audio_stats(pack)  # same replace() the block applies (video's mean/std keys too)
        pack = _normalize_quant(pack)

        _assert_bidirectional(model_keys, pack)


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
class TestAudioVaeLoadContract:
    """``AudioDecoder`` + ``AudioConditioner`` blocks against ``audio_vae.safetensors``.

    The 2.5 pack bundles audio VAE encoder + decoder + shared
    ``per_channel_statistics`` in one file. Production reads it via two
    separate blocks (``AudioDecoder`` for the decoder half,
    ``AudioConditioner`` for the encoder half), each merging in the shared
    stats and applying ``remap_audio_vae_keys``. Both halves plus the
    stats-only top-level group must be exactly {decoder, encoder,
    per_channel_statistics} -- any 4th group would silently go unloaded by
    either block.
    """

    def test_top_level_groups_are_exactly_decoder_encoder_stats(self):
        pack_path = LTX25_Q8_DIR / "audio_vae.safetensors"
        assert pack_path.exists()
        pack = _strip_prefix(_header_keys(pack_path), "audio_vae.")
        top_groups = {k.split(".", 1)[0] for k in pack}
        assert top_groups == {"decoder", "encoder", "per_channel_statistics"}

    def test_decoder_pack_and_model_fully_consume_each_other(self):
        from mlx.utils import tree_flatten

        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder

        model = AudioVAEDecoder()
        model_keys = {k for k, _ in tree_flatten(model.parameters())}

        pack_path = LTX25_Q8_DIR / "audio_vae.safetensors"
        pack = _strip_prefix(_header_keys(pack_path), "audio_vae.")
        decoder_keys = _strip_prefix(pack, "decoder.")
        stats_keys = _strip_prefix(pack, "per_channel_statistics.")
        pack_for_decoder = decoder_keys | {f"per_channel_statistics.{k}" for k in stats_keys}
        pack_for_decoder = _remap_audio_stats(pack_for_decoder)
        pack_for_decoder = _normalize_quant(pack_for_decoder)

        _assert_bidirectional(model_keys, pack_for_decoder)

    def test_encoder_pack_and_model_fully_consume_each_other(self):
        from mlx.utils import tree_flatten

        from ltx_core_mlx.model.audio_vae.encoder import AudioVAEEncoder

        model = AudioVAEEncoder()
        model_keys = {k for k, _ in tree_flatten(model.parameters())}

        pack_path = LTX25_Q8_DIR / "audio_vae.safetensors"
        pack = _strip_prefix(_header_keys(pack_path), "audio_vae.")
        encoder_keys = _strip_prefix(pack, "encoder.")
        stats_keys = _strip_prefix(pack, "per_channel_statistics.")
        pack_for_encoder = encoder_keys | {f"per_channel_statistics.{k}" for k in stats_keys}
        pack_for_encoder = _remap_audio_stats(pack_for_encoder)
        pack_for_encoder = _normalize_quant(pack_for_encoder)

        _assert_bidirectional(model_keys, pack_for_encoder)


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
class TestVocoderLoadContract:
    """``AudioDecoder`` block's ``VocoderWithBWE`` against ``vocoder.safetensors``.

    1227 tensors -- header-only key/shape reads, never materialized.
    """

    def test_vocoder_pack_and_model_fully_consume_each_other(self):
        from mlx.utils import tree_flatten

        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE

        model = VocoderWithBWE()
        model_keys = {k for k, _ in tree_flatten(model.parameters())}

        pack_path = LTX25_Q8_DIR / "vocoder.safetensors"
        assert pack_path.exists()
        pack = _strip_prefix(_header_keys(pack_path), "vocoder.")
        pack = _normalize_quant(pack)

        _assert_bidirectional(model_keys, pack)
