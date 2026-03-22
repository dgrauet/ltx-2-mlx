"""Tests for ltx_2_mlx.utils."""

import shutil

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg, find_ffprobe
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup, get_memory_stats


class TestMemory:
    def test_aggressive_cleanup_runs(self):
        aggressive_cleanup()

    def test_get_memory_stats_keys(self):
        stats = get_memory_stats()
        assert "active_gb" in stats
        assert "peak_gb" in stats
        assert "cache_gb" in stats
        assert all(isinstance(v, float) for v in stats.values())


class TestFfmpeg:
    def test_find_ffmpeg(self):
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")
        path = find_ffmpeg()
        assert "ffmpeg" in path

    def test_find_ffprobe(self):
        if shutil.which("ffprobe") is None:
            pytest.skip("ffprobe not installed")
        path = find_ffprobe()
        assert "ffprobe" in path


class TestImagePrep:
    def test_shape_and_range(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
        tensor = prepare_image_for_encoding(img, height=64, width=96)
        assert tensor.shape == (1, 3, 64, 96)
        assert tensor.dtype == mx.bfloat16
        vals = tensor.astype(mx.float32)
        assert float(mx.min(vals)) >= -1.0 - 1e-3
        assert float(mx.max(vals)) <= 1.0 + 1e-3

    def test_resize(self):
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        tensor = prepare_image_for_encoding(img, height=32, width=48)
        assert tensor.shape == (1, 3, 32, 48)
