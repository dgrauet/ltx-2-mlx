"""Unit tests for the conv VAE decode memory model and the automatic tiling choice (issue #142).

All tests are pure arithmetic — no model weights, no GPU work, sub-second.
"""

import mlx.core as mx
import pytest

from ltx_core_mlx.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core_mlx.model.video_vae.video_vae import (
    CONV_DECODE_BYTES_PER_PIXEL_FRAME,
    _compute_decode_tiling,
    decode_budget_bytes,
    decode_cache_limit,
    estimate_decode_peak_bytes,
)

# 512x768x49 output: latent (1, 128, 7, 16, 24) -> 49 x 512 x 768 pixel-frames.
_LATENT = (1, 128, 7, 16, 24)
_PX = 49 * 512 * 768
_ACC = 4 * 3 * 4  # four fp32 accumulation tensors (chunk + weights, current + previous) per pixel


def _temporal(frames: int, overlap: int) -> TilingConfig:
    return TilingConfig(temporal_config=TemporalTilingConfig(frames, overlap))


class TestEstimate:
    def test_untiled_is_bytes_per_pixel_frame(self):
        assert CONV_DECODE_BYTES_PER_PIXEL_FRAME == 750
        assert estimate_decode_peak_bytes(_LATENT, None) == 750 * _PX

    def test_temporal_tile_counts_tile_pixels_plus_accumulators(self):
        est = estimate_decode_peak_bytes(_LATENT, _temporal(40, 8))
        assert est == 750 * 40 * 512 * 768 + 40 * 512 * 768 * _ACC

    def test_temporal_tile_longer_than_clip_is_clamped(self):
        assert estimate_decode_peak_bytes(_LATENT, _temporal(80, 24)) == 750 * _PX + _PX * _ACC

    def test_spatial_tile_follows_prepare_tiles_scaling(self):
        # 512 px on the long side (24 latent cells): W tile = 16 cells = 512 px, H tile = round(16*16/24) = 11 cells
        cfg = TilingConfig(spatial_config=SpatialTilingConfig(512, 32), temporal_config=TemporalTilingConfig(40, 8))
        est = estimate_decode_peak_bytes(_LATENT, cfg)
        assert est == 750 * 40 * (11 * 32) * 512 + 40 * 512 * 768 * _ACC

    def test_spatial_only_uses_the_whole_clip_temporally(self):
        cfg = TilingConfig(spatial_config=SpatialTilingConfig(256, 32))
        # 256 px -> 8 cells on W; H = round(8*16/24) = 5 cells (>= overlap+1 = 2)
        assert estimate_decode_peak_bytes(_LATENT, cfg) == 750 * 49 * (5 * 32) * 256 + _PX * _ACC

    def test_measured_cases_within_25_percent(self):
        # M2 Pro 32 GB, LTX-2.5 q8 pack, peak Metal memory minus weights (scratch run 2026-09-20)
        measured = [
            ((1, 128, 7, 16, 24), None, 13.27),
            ((1, 128, 3, 24, 36), None, 10.20),
            ((1, 128, 2, 32, 48), None, 9.80),
            ((1, 128, 7, 16, 24), _temporal(40, 8), 8.42),
        ]
        for shape, cfg, gb in measured:
            est = estimate_decode_peak_bytes(shape, cfg) / 2**30
            assert 0.75 <= est / gb <= 1.5, (shape, cfg, est, gb)


class TestBudget:
    def test_env_or_half_of_unified_memory(self, monkeypatch):
        monkeypatch.delenv("LTX2_VAE_DECODE_BUDGET_GB", raising=False)
        assert decode_budget_bytes() == mx.device_info()["memory_size"] // 2
        monkeypatch.setenv("LTX2_VAE_DECODE_BUDGET_GB", "2.5")
        assert decode_budget_bytes() == int(2.5 * 1024**3)


class TestAutoTiling:
    def _pick(self, budget_gb: float, shape=_LATENT, frame_rate: float = 24.0):
        return _compute_decode_tiling(shape, frame_rate=frame_rate, budget_bytes=int(budget_gb * 2**30))

    def test_none_when_untiled_fits(self):
        assert self._pick(64.0) is None
        assert _compute_decode_tiling(_LATENT, budget_bytes=750 * _PX) is None

    def test_tiles_just_above_budget(self):
        assert _compute_decode_tiling(_LATENT, budget_bytes=750 * _PX - 1) is not None

    def test_temporal_first_largest_tile_that_fits(self):
        # 40-frame tiles need 12.5 GB, 48-frame tiles 15.1 GB: a 13 GB budget picks 40 frames
        cfg = self._pick(13.0)
        assert cfg.spatial_config is None
        assert cfg.temporal_config == TemporalTilingConfig(40, 8)

    def test_temporal_ladder_prefers_upstream_default(self):
        # a 300-frame clip at a generous budget: 80-frame tiles with the upstream 24-frame overlap
        cfg = self._pick(30.0, shape=(1, 128, 38, 16, 24))
        assert cfg.temporal_config == TemporalTilingConfig(80, 24)

    def test_spatial_ladder_when_40_frames_still_too_big(self):
        # 40-frame temporal tiles need 12.5 GB: at 8 GB add spatial tiles, largest that fits
        cfg = self._pick(8.0)
        assert cfg.temporal_config == TemporalTilingConfig(40, 8)
        assert cfg.spatial_config == SpatialTilingConfig(512, 32)
        cfg = self._pick(4.0)
        assert cfg.spatial_config == SpatialTilingConfig(256, 32)

    def test_last_resort_shrinks_frames_below_40(self):
        cfg = self._pick(1.0)
        assert cfg.spatial_config == SpatialTilingConfig(256, 32)
        assert cfg.temporal_config is not None and cfg.temporal_config.tile_size_in_frames < 40
        assert cfg.temporal_config.tile_size_in_frames >= 16

    def test_overlap_scales_with_frame_rate_and_caps_at_30_percent(self):
        for fps, overlap in ((24.0, 24), (30.0, 24), (48.0, 24), (60.0, 24)):
            cfg = self._pick(30.0, shape=(1, 128, 38, 16, 24), frame_rate=fps)
            assert cfg.temporal_config.tile_overlap_in_frames == overlap
        cfg = self._pick(13.0)  # 40-frame tile: 30 % = 12 -> rounded down to 8
        assert cfg.temporal_config.tile_overlap_in_frames == 8

    def test_short_clip_never_gets_temporal_tiles(self):
        # 9 frames: no temporal split possible; spatial only
        cfg = self._pick(1.0, shape=(1, 128, 2, 32, 48))
        assert cfg.temporal_config is None and cfg.spatial_config is not None

    def test_configs_pass_validation(self):
        for budget in (13.0, 8.0, 4.0, 2.0, 1.0, 0.5):
            cfg = self._pick(budget)
            if cfg.temporal_config is not None:
                TemporalTilingConfig(
                    cfg.temporal_config.tile_size_in_frames, cfg.temporal_config.tile_overlap_in_frames
                )
            if cfg.spatial_config is not None:
                SpatialTilingConfig(cfg.spatial_config.tile_size_in_pixels, cfg.spatial_config.tile_overlap_in_pixels)

    def test_env_budget_is_the_default(self, monkeypatch):
        monkeypatch.setenv("LTX2_VAE_DECODE_BUDGET_GB", "13.0")
        assert _compute_decode_tiling(_LATENT) == self._pick(13.0)


class TestCacheLimit:
    def test_cache_limit_is_zero_inside_and_restored_after(self):
        before = mx.set_cache_limit(3 * 2**30)
        try:
            with decode_cache_limit():
                assert mx.set_cache_limit(0) == 0
            assert mx.set_cache_limit(before) == 3 * 2**30
        finally:
            mx.set_cache_limit(before)

    def test_opt_out_env(self, monkeypatch):
        monkeypatch.setenv("LTX2_VAE_DECODE_KEEP_CACHE", "1")
        before = mx.set_cache_limit(3 * 2**30)
        try:
            with decode_cache_limit():
                assert mx.set_cache_limit(3 * 2**30) == 3 * 2**30
        finally:
            mx.set_cache_limit(before)


@pytest.mark.parametrize("shape", [(1, 128, 13, 16, 24), (1, 128, 16, 22, 40)])
def test_reporter_shapes_tile_within_a_16_gb_budget(shape):
    """Issue #142's 768x512x121 and 704x1280x121 must get a config whose estimate fits 16 GB."""
    cfg = _compute_decode_tiling(shape, budget_bytes=16 * 2**30)
    assert cfg is not None
    assert estimate_decode_peak_bytes(shape, cfg) <= 16 * 2**30
