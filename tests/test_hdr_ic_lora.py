"""HDR IC-LoRA (upstream v1.4 single-stage ACEScct) — pipeline contracts without weights."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from ltx_pipelines_mlx.utils.blocks import ImageConditioner, VideoDecoder


def _pack(tmp_path):
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    return tmp_path


def test_vae_blocks_default_to_bf16_and_accept_fp32(tmp_path):
    _pack(tmp_path)
    assert VideoDecoder(tmp_path, video_decoder="diffusion").dtype == mx.bfloat16
    assert VideoDecoder(tmp_path, video_decoder="diffusion", dtype=mx.float32).dtype == mx.float32
    assert ImageConditioner(tmp_path).dtype is None
    assert ImageConditioner(tmp_path, dtype=mx.float32).dtype == mx.float32


class _StubInner:
    def tiled_decode(self, video_latent, tiling, *, seed, keyframes):
        assert seed == 7 and keyframes is None
        yield mx.full((1, 3, 2, 4, 6), -1.0)
        yield mx.full((1, 3, 3, 4, 6), 1.0)


class _StubDiffusion:
    _decoder = _StubInner()

    def resolve_tiling(self, latent_shape, *, keyframe_planes=0):
        return None


def test_iter_frames_yields_float32_unit_range(tmp_path, monkeypatch):
    _pack(tmp_path)
    dec = VideoDecoder(tmp_path, video_decoder="diffusion")
    monkeypatch.setattr(VideoDecoder, "load", lambda self: _StubDiffusion())
    chunks = list(dec.iter_frames(mx.zeros((1, 128, 1, 1, 1)), seed=7, keyframes=None))
    assert [c.shape for c in chunks] == [(2, 4, 6, 3), (3, 4, 6, 3)]
    assert all(c.dtype == np.float32 for c in chunks)
    assert chunks[0].max() == 0.0 and chunks[1].min() == 1.0


def test_iter_frames_rejects_conv_decoder(tmp_path):
    _pack(tmp_path)
    dec = VideoDecoder(tmp_path)
    with pytest.raises(ValueError):
        next(dec.iter_frames(mx.zeros((1, 128, 1, 1, 1)), seed=0, keyframes=None))
