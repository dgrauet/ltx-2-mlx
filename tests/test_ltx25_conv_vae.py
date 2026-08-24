"""Tests for LTX-2.5 conv video VAE file-name resolution by pack evidence.

The 2.5 conv video VAE ships as ``vae_decoder_conv.safetensors`` /
``vae_encoder_conv.safetensors`` instead of the 2.3 names
(``vae_decoder.safetensors`` / ``vae_encoder.safetensors``). Both loader
call sites (``ImageConditioner.load`` for the encoder, ``VideoDecoder.load``
for the decoder) must pick the name+prefix pair by evidence of the decoder
conv file's presence in the pack directory, and must stay byte-identical
on a 2.3 pack (no conv file present).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar


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
