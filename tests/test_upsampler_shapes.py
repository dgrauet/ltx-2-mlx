"""Shape tests for the latent upsampler."""

import mlx.core as mx

from ltx_core_mlx.model.upsampler.model import LatentUpsampler


class TestLatentUpsampler:
    def test_2x_spatial_upscale(self):
        upsampler = LatentUpsampler(
            in_channels=8, mid_channels=32, num_blocks_per_stage=1, spatial_upsample=True, spatial_scale=2.0
        )
        latent = mx.zeros((1, 8, 2, 3, 4))  # BCFHW
        out = upsampler(latent)
        assert out.shape == (1, 8, 2, 6, 8)  # F same, H*2, W*2

    def test_1_5x_spatial_upscale(self):
        upsampler = LatentUpsampler(
            in_channels=8,
            mid_channels=32,
            num_blocks_per_stage=1,
            spatial_upsample=True,
            spatial_scale=1.5,
            rational_resampler=True,
        )
        latent = mx.zeros((1, 8, 2, 4, 6))  # BCFHW
        out = upsampler(latent)
        # 4*3/2=6, 6*3/2=9
        assert out.shape == (1, 8, 2, 6, 9)

    def test_2x_temporal_upscale(self):
        upsampler = LatentUpsampler(
            in_channels=8,
            mid_channels=32,
            num_blocks_per_stage=1,
            spatial_upsample=False,
            temporal_upsample=True,
        )
        latent = mx.zeros((1, 8, 4, 3, 4))  # BCFHW
        out = upsampler(latent)
        # 4*2 - 1 = 7 (first frame removed)
        assert out.shape == (1, 8, 7, 3, 4)

    def test_from_config(self):
        config = {"in_channels": 8, "mid_channels": 32, "num_blocks_per_stage": 1, "spatial_scale": 2.0}
        upsampler = LatentUpsampler.from_config(config)
        latent = mx.zeros((1, 8, 2, 3, 4))
        out = upsampler(latent)
        assert out.shape == (1, 8, 2, 6, 8)
