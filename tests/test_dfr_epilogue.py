"""DFR spatial epilogue building blocks: Lanczos x2, tile clamping, single-frame decode."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig
from ltx_pipelines_mlx.dfr import (
    EPILOGUE_KEYFRAME_STRENGTH,
    EPILOGUE_NOISE_SEED_OFFSET,
    EPILOGUE_SPATIAL_OVERLAP,
    KEYFRAME_PLANE_DECODE_SEED_OFFSET,
    clamp_tile_counts,
    floor_to_multiple,
    lanczos_x2,
)


def test_constants():
    assert EPILOGUE_SPATIAL_OVERLAP == 12
    assert EPILOGUE_KEYFRAME_STRENGTH == 1.0
    assert KEYFRAME_PLANE_DECODE_SEED_OFFSET == 4000
    assert EPILOGUE_NOISE_SEED_OFFSET == 2000


def _pil_lanczos_x2(frame: np.ndarray) -> np.ndarray:
    """Reference: uint8-round -> PIL resize x2 LANCZOS -> float32 [0, 1]."""
    height, width, channels = frame.shape
    array = np.clip(frame, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    if channels == 1:
        image = Image.fromarray(array[..., 0], mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    image = image.resize((width * 2, height * 2), resample=Image.Resampling.LANCZOS)
    resized = np.asarray(image, dtype=np.float32) / 255.0
    if resized.ndim == 2:
        resized = resized[..., None]
    return resized


class TestLanczosX2:
    def test_matches_pil_reference_rgb(self):
        rng = np.random.default_rng(0)
        frames = rng.random((3, 8, 10, 3)).astype(np.float32)
        out = lanczos_x2(frames)
        assert out.shape == (3, 16, 20, 3)
        assert out.dtype == np.float32
        expected = np.stack([_pil_lanczos_x2(f) for f in frames], axis=0)
        np.testing.assert_array_equal(out, expected)

    def test_matches_pil_reference_single_channel(self):
        rng = np.random.default_rng(1)
        frames = rng.random((2, 6, 6, 1)).astype(np.float32)
        out = lanczos_x2(frames)
        assert out.shape == (2, 12, 12, 1)
        expected = np.stack([_pil_lanczos_x2(f) for f in frames], axis=0)
        np.testing.assert_array_equal(out, expected)

    def test_output_range_clamped(self):
        frames = np.full((1, 4, 4, 3), 2.0, dtype=np.float32)  # out of [0, 1]
        out = lanczos_x2(frames)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_requires_at_least_one_frame(self):
        with pytest.raises(ValueError):
            lanczos_x2(np.zeros((0, 4, 4, 3), dtype=np.float32))

    def test_requires_fhwc(self):
        with pytest.raises(ValueError):
            lanczos_x2(np.zeros((4, 4, 3), dtype=np.float32))

    def test_rejects_bad_channel_count(self):
        with pytest.raises(ValueError):
            lanczos_x2(np.zeros((1, 4, 4, 2), dtype=np.float32))


class TestClampTileCounts:
    def test_untouched_when_it_fits(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(1, 0),
            height=DimensionTilingConfig(2, 12),
            width=DimensionTilingConfig(2, 12),
        )
        clamped = clamp_tile_counts(tiling, (9, 32, 48))
        assert clamped == tiling

    def test_dim_smaller_than_num_tiles_falls_back_to_one_tile(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(1, 0),
            height=DimensionTilingConfig(2, 12),
            width=DimensionTilingConfig(2, 12),
        )
        clamped = clamp_tile_counts(tiling, (9, 1, 48))
        assert clamped.height == DimensionTilingConfig(1, 0)
        assert clamped.width == DimensionTilingConfig(2, 12)

    def test_overlap_clamped_to_dim_minus_num_tiles(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(1, 0),
            height=DimensionTilingConfig(2, 12),
            width=DimensionTilingConfig(2, 12),
        )
        clamped = clamp_tile_counts(tiling, (9, 4, 48))
        assert clamped.height == DimensionTilingConfig(2, 2)  # dim_size - n = 4 - 2 = 2
        assert clamped.width == DimensionTilingConfig(2, 12)

    def test_single_tile_dimension_untouched(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(1, 0),
            height=DimensionTilingConfig(1, 0),
            width=DimensionTilingConfig(2, 12),
        )
        clamped = clamp_tile_counts(tiling, (9, 1, 4))
        assert clamped.height == DimensionTilingConfig(1, 0)
        assert clamped.width == DimensionTilingConfig(2, 2)


def test_floor_to_multiple():
    assert floor_to_multiple(1080, 128) == 1024
    assert floor_to_multiple(1024, 128) == 1024
    assert floor_to_multiple(127, 128) == 0


class _StubConvDecoder:
    def __init__(self):
        self.calls = []

    def decode(self, latent):
        self.calls.append(latent)
        import mlx.core as mx

        b, _c, f, h, w = latent.shape
        return mx.full((b, 3, f, h * 32, w * 32), 1.0, dtype=latent.dtype)  # mimics the real 32x spatial upsample


class _StubInnerDiffusionDecoder:
    def __init__(self):
        self.calls = []

    def decode(self, latent, *, seed=0):
        self.calls.append((latent, seed))
        b, _c, f, h, w = latent.shape
        import mlx.core as mx

        return mx.full((b, 3, f, h, w), 0.5, dtype=latent.dtype)


class _StubDiffusionBlock:
    """Mimics ``_DiffusionVideoDecoder``'s public surface for decode_single_frame."""

    def __init__(self):
        self._decoder = _StubInnerDiffusionDecoder()


class TestDecodeSingleFrame:
    def test_conv_path_calls_decoder_and_normalizes(self, monkeypatch):
        import mlx.core as mx

        from ltx_pipelines_mlx.utils.blocks import VideoDecoder

        block = VideoDecoder.__new__(VideoDecoder)
        block.video_decoder = "conv"
        block._decoder = _StubConvDecoder()
        monkeypatch.setattr(block, "load", lambda: block._decoder)

        latent = mx.zeros((1, 128, 1, 2, 3))
        out = block.decode_single_frame(latent, seed=7)

        assert len(block._decoder.calls) == 1
        assert out.shape == (1, 2 * 32, 3 * 32, 3)
        assert bool(mx.all((out >= 0) & (out <= 1)).item())

    def test_diffusion_path_passes_seed(self, monkeypatch):
        import mlx.core as mx

        from ltx_pipelines_mlx.utils.blocks import VideoDecoder

        block = VideoDecoder.__new__(VideoDecoder)
        block.video_decoder = "diffusion"
        stub = _StubDiffusionBlock()
        block._decoder = stub
        monkeypatch.setattr(block, "load", lambda: block._decoder)

        latent = mx.zeros((1, 128, 1, 2, 3))
        out = block.decode_single_frame(latent, seed=42)

        assert len(stub._decoder.calls) == 1
        _, seed = stub._decoder.calls[0]
        assert seed == 42
        assert out.shape[0] == 1
        assert out.shape[-1] == 3

    def test_rejects_multi_frame_latent(self):
        import mlx.core as mx

        from ltx_pipelines_mlx.utils.blocks import VideoDecoder

        block = VideoDecoder.__new__(VideoDecoder)
        block.video_decoder = "conv"

        latent = mx.zeros((1, 128, 2, 2, 3))
        with pytest.raises(ValueError):
            block.decode_single_frame(latent, seed=0)
