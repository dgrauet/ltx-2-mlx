"""Regression tests for IC-LoRA reference-video frame alignment (issue #27).

The video VAE encoder requires a (1 + 8k)-frame input. Source files produced
by LTX itself are 8k-trimmed on save, so passing them through unchanged hits
``space_to_depth`` with an unreshapeable frame count. The helper must round
the loader's frame count down to (1 + 8k) before invoking the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import pytest

from ltx_pipelines_mlx import iclora_utils


@dataclass
class _FakeVideoInfo:
    """Stand-in for ltx_core_mlx.utils.ffmpeg.VideoInfo."""

    width: int = 1280
    height: int = 704
    num_frames: int = 72
    fps: float = 24.0
    duration: float = 3.0
    has_audio: bool = False


class _FakeEncoder:
    """Capture-only encoder; returns a tensor with the right rank but the
    actual values are not exercised — we only assert what got loaded."""

    def encode(self, video: mx.array) -> mx.array:
        b = video.shape[0]
        f_lat = max(1, (video.shape[2] - 1) // 8 + 1)
        return mx.zeros((b, 128, f_lat, 1, 1))


@pytest.fixture
def captured_load(monkeypatch):
    """Patch probe_video_info + loader; capture the frame count passed to the loader."""
    calls: list[int] = []

    def fake_probe(_path: str) -> _FakeVideoInfo:
        return _FakeVideoInfo(num_frames=72)

    def fake_load(_path, height: int, width: int, max_frames: int) -> mx.array:
        calls.append(max_frames)
        return mx.zeros((1, 3, max_frames, height, width), dtype=mx.bfloat16)

    monkeypatch.setattr(iclora_utils, "probe_video_info", fake_probe)
    monkeypatch.setattr(iclora_utils, "load_video_frames_normalized", fake_load)
    return calls


def _invoke(num_frames: int, scale: int = 2) -> None:
    """Helper: invoke the helper with a single conditioning at the given target."""
    iclora_utils.append_ic_lora_reference_video_conditionings(
        conditionings=[],
        video_conditioning=[("/fake/path.mp4", 1.0)],
        height=704,
        width=1280,
        num_frames=num_frames,
        video_encoder=_FakeEncoder(),
        reference_downscale_factor=scale,
    )


class TestFrameAlignment:
    """The bug from issue #27: 8k-frame source + naive load → unreshapeable."""

    def test_short_source_aligns_to_1_plus_8k(self, captured_load):
        # File has 72 frames; caller asks for 121. Loader must get 65 (= 1 + 8*8).
        _invoke(num_frames=121)
        assert captured_load == [65], f"expected loader to receive 65 frames, got {captured_load}"

    def test_aligned_source_passes_through(self, captured_load, monkeypatch):
        # File has exactly 73 frames (1 + 8*9); target 121. min=73, k=9, vae=73.
        monkeypatch.setattr(iclora_utils, "probe_video_info", lambda _p: _FakeVideoInfo(num_frames=73))
        _invoke(num_frames=121)
        assert captured_load == [73]

    def test_long_source_capped_at_target(self, captured_load, monkeypatch):
        # File has 200 frames; target 73 (= 1 + 8*9). min=73, k=9, vae=73.
        monkeypatch.setattr(iclora_utils, "probe_video_info", lambda _p: _FakeVideoInfo(num_frames=200))
        _invoke(num_frames=73)
        assert captured_load == [73]

    def test_long_source_with_misaligned_target_rounds_down(self, captured_load, monkeypatch):
        # File has 200 frames; target 121 (= 1 + 8*15). min=121, k=15, vae=121. ✓
        # And a non-aligned target (90): min=90, k=(90-1)//8=11, vae=89.
        monkeypatch.setattr(iclora_utils, "probe_video_info", lambda _p: _FakeVideoInfo(num_frames=200))
        _invoke(num_frames=90)
        assert captured_load == [89]

    def test_multiple_conditionings_each_aligned(self, captured_load, monkeypatch):
        # Two refs of different lengths; each one independently aligned.
        infos = iter([_FakeVideoInfo(num_frames=72), _FakeVideoInfo(num_frames=49)])
        monkeypatch.setattr(iclora_utils, "probe_video_info", lambda _p: next(infos))
        iclora_utils.append_ic_lora_reference_video_conditionings(
            conditionings=[],
            video_conditioning=[("/a.mp4", 1.0), ("/b.mp4", 1.0)],
            height=704,
            width=1280,
            num_frames=121,
            video_encoder=_FakeEncoder(),
            reference_downscale_factor=2,
        )
        # 72 → k=8 → 65; 49 → k=6 → 49
        assert captured_load == [65, 49]

    def test_one_frame_minimum(self, captured_load, monkeypatch):
        # Pathological: source has 1 frame. k = max(1, 0) = 1, vae = 9.
        # The loader will return 1 frame anyway (file capacity), but we send 9.
        monkeypatch.setattr(iclora_utils, "probe_video_info", lambda _p: _FakeVideoInfo(num_frames=1))
        _invoke(num_frames=121)
        assert captured_load == [9]
