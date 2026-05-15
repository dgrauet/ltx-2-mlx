"""Unit tests for _compute_decode_tiling.

All tests are pure arithmetic — no model weights, no GPU work, sub-second.
"""

import pytest

from ltx_core_mlx.model.video_vae.tiling import TemporalTilingConfig
from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling


# Latent shape (B, C, F_lat, H_lat, W_lat) used across tests.
# 4×4 spatial keeps block-3 bytes small so we can control the budget precisely.
# block3_bytes_per_lat_frame = 512 * 4 * (4*4) * (4*4) * 4 = 2,097,152 (~2 MB)
_SMALL_LATENT = (1, 128, 100, 4, 4)  # 100 latent frames → ~200 MB total, triggers at budget < 200 MB
_BYTES_PER_LAT_FRAME = 512 * 4 * (4 * 4) * (4 * 4) * 4  # 2 MB


class TestNoTilingNeeded:
    def test_returns_none_when_video_fits(self):
        # 2 latent frames × 2 MB = 4 MB << 1 GB budget
        shape = (1, 128, 2, 4, 4)
        result = _compute_decode_tiling(shape, peak_budget_gb=1.0)
        assert result is None

    def test_returns_none_exactly_at_budget(self):
        # budget == total bytes → should NOT tile (≤ check)
        budget_bytes = _BYTES_PER_LAT_FRAME * 100
        budget_gb = budget_bytes / 1024**3
        result = _compute_decode_tiling(_SMALL_LATENT, peak_budget_gb=budget_gb)
        assert result is None

    def test_triggers_just_above_budget(self):
        # One byte under exact fit → should tile
        budget_bytes = _BYTES_PER_LAT_FRAME * 100 - 1
        budget_gb = budget_bytes / 1024**3
        result = _compute_decode_tiling(_SMALL_LATENT, peak_budget_gb=budget_gb)
        assert result is not None


class TestTilingConfig:
    """When tiling triggers, the returned config must satisfy TemporalTilingConfig constraints."""

    # budget=0.1 GB gives tile_frames=424 (well above 16), large enough that the
    # fps-scaling overlap formula is exercised rather than the 25% cap.
    _BUDGET = 0.1

    def _cfg(self, fps: float) -> TemporalTilingConfig:
        result = _compute_decode_tiling(_SMALL_LATENT, peak_budget_gb=self._BUDGET, frame_rate=fps)
        assert result is not None
        assert result.temporal_config is not None
        return result.temporal_config

    def test_tile_size_gte_16(self):
        cfg = self._cfg(24.0)
        assert cfg.tile_size_in_frames >= 16

    def test_tile_size_divisible_by_8(self):
        cfg = self._cfg(24.0)
        assert cfg.tile_size_in_frames % 8 == 0

    def test_overlap_divisible_by_8(self):
        for fps in (24.0, 30.0, 48.0, 60.0):
            cfg = self._cfg(fps)
            assert cfg.tile_overlap_in_frames % 8 == 0, f"overlap not multiple of 8 at {fps} fps"

    def test_overlap_less_than_tile(self):
        for fps in (24.0, 30.0, 48.0, 60.0):
            cfg = self._cfg(fps)
            assert cfg.tile_overlap_in_frames < cfg.tile_size_in_frames

    def test_config_passes_tiling_validation(self):
        # TemporalTilingConfig.__post_init__ raises on invalid values
        for fps in (24.0, 30.0, 48.0, 60.0):
            cfg = self._cfg(fps)
            TemporalTilingConfig(
                tile_size_in_frames=cfg.tile_size_in_frames,
                tile_overlap_in_frames=cfg.tile_overlap_in_frames,
            )  # must not raise


class TestFrameRateScaling:
    """Overlap should grow with frame rate to keep the blend window ~1 second."""

    _BUDGET = 0.1  # large enough tiles that fps scaling, not 25% cap, dominates

    def _overlap(self, fps: float) -> int:
        result = _compute_decode_tiling(_SMALL_LATENT, peak_budget_gb=self._BUDGET, frame_rate=fps)
        assert result is not None and result.temporal_config is not None
        return result.temporal_config.tile_overlap_in_frames

    def test_24fps_overlap(self):
        # (int(24) // 8) * 8 = 24
        assert self._overlap(24.0) == 24

    def test_30fps_overlap(self):
        # (int(30) // 8) * 8 = 24  (rounds down to same as 24fps)
        assert self._overlap(30.0) == 24

    def test_48fps_overlap(self):
        # (int(48) // 8) * 8 = 48
        assert self._overlap(48.0) == 48

    def test_60fps_overlap(self):
        # (int(60) // 8) * 8 = 56
        assert self._overlap(60.0) == 56

    def test_overlap_nondecreasing_with_fps(self):
        fps_values = [24.0, 30.0, 48.0, 60.0]
        overlaps = [self._overlap(fps) for fps in fps_values]
        assert overlaps == sorted(overlaps), f"overlaps not monotone: {list(zip(fps_values, overlaps))}"


class TestEdgeCases:
    def test_minimum_tile_size_enforced(self):
        # Tiny budget forces max_lat_frames=2 → tile_frames=max(16,16)=16
        result = _compute_decode_tiling(_SMALL_LATENT, peak_budget_gb=1e-6)
        assert result is not None
        assert result.temporal_config is not None
        assert result.temporal_config.tile_size_in_frames >= 16

    def test_overlap_zero_when_tile_too_small(self):
        # With tile_frames=16: (16//32)*8 = 0 → overlap=0
        result = _compute_decode_tiling(_SMALL_LATENT, peak_budget_gb=1e-6)
        assert result is not None
        assert result.temporal_config is not None
        assert result.temporal_config.tile_overlap_in_frames == 0

    def test_default_fps_is_24(self):
        r1 = _compute_decode_tiling(_SMALL_LATENT, peak_budget_gb=0.1)
        r2 = _compute_decode_tiling(_SMALL_LATENT, peak_budget_gb=0.1, frame_rate=24.0)
        assert r1 is not None and r2 is not None
        assert r1.temporal_config == r2.temporal_config
