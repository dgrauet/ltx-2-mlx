"""Tests for video and audio position computation."""

import mlx.core as mx

from ltx_core_mlx.utils.positions import (
    AUDIO_DOWNSAMPLE_FACTOR,
    AUDIO_HOP_LENGTH,
    AUDIO_LATENTS_PER_SECOND,
    AUDIO_SAMPLE_RATE,
    VIDEO_SPATIAL_SCALE,
    VIDEO_TEMPORAL_SCALE,
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class TestConstants:
    def test_video_temporal_scale(self):
        assert VIDEO_TEMPORAL_SCALE == 8

    def test_video_spatial_scale(self):
        assert VIDEO_SPATIAL_SCALE == 32

    def test_audio_downsample_factor(self):
        assert AUDIO_DOWNSAMPLE_FACTOR == 4

    def test_audio_hop_length(self):
        assert AUDIO_HOP_LENGTH == 160

    def test_audio_sample_rate(self):
        assert AUDIO_SAMPLE_RATE == 16000

    def test_audio_latents_per_second(self):
        assert AUDIO_LATENTS_PER_SECOND == 25.0


# ---------------------------------------------------------------------------
# compute_audio_token_count
# ---------------------------------------------------------------------------
class TestComputeAudioTokenCount:
    def test_24fps_24frames(self):
        """24 frames at 24 fps = 1 second = 25 tokens."""
        count = compute_audio_token_count(num_video_frames=24, fps=24.0)
        assert count == 25

    def test_48fps_48frames(self):
        """48 frames at 48 fps = 1 second = 25 tokens."""
        count = compute_audio_token_count(num_video_frames=48, fps=48.0)
        assert count == 25

    def test_24fps_48frames(self):
        """48 frames at 24 fps = 2 seconds = 50 tokens."""
        count = compute_audio_token_count(num_video_frames=48, fps=24.0)
        assert count == 50

    def test_24fps_1frame(self):
        """1 frame at 24 fps ~ 0.0417 seconds ~ 1 token."""
        count = compute_audio_token_count(num_video_frames=1, fps=24.0)
        assert count == round(1.0 / 24.0 * 25.0)
        assert count == 1

    def test_rounding(self):
        """Check rounding behavior for non-integer results."""
        # 10 frames at 24 fps = 0.4167s * 25 = 10.4167 -> round to 10
        count = compute_audio_token_count(num_video_frames=10, fps=24.0)
        assert count == 10

    def test_zero_frames(self):
        count = compute_audio_token_count(num_video_frames=0, fps=24.0)
        assert count == 0

    def test_common_generation_length(self):
        """97 frames at 24 fps -> ~4.04s * 25 = ~101 tokens."""
        count = compute_audio_token_count(num_video_frames=97, fps=24.0)
        expected = round(97 / 24.0 * 25.0)
        assert count == expected


# ---------------------------------------------------------------------------
# compute_video_positions
# ---------------------------------------------------------------------------
class TestComputeVideoPositions:
    def test_shape(self):
        positions = compute_video_positions(num_frames=3, height=2, width=4)
        assert positions.shape == (1, 3 * 2 * 4, 3)

    def test_dtype(self):
        positions = compute_video_positions(num_frames=2, height=2, width=2)
        assert positions.dtype == mx.float32

    def test_single_element(self):
        positions = compute_video_positions(num_frames=1, height=1, width=1)
        assert positions.shape == (1, 1, 3)

    def test_spatial_midpoints(self):
        """Spatial positions should be pixel-space midpoints: h*32+16, w*32+16."""
        positions = compute_video_positions(num_frames=1, height=3, width=2)
        # Shape: (1, 3*2, 3)
        # Token ordering: (h=0,w=0), (h=0,w=1), (h=1,w=0), (h=1,w=1), (h=2,w=0), (h=2,w=1)
        h_values = positions[0, :, 1]  # height axis
        w_values = positions[0, :, 2]  # width axis

        # h=0: midpoint = 0*32+16 = 16
        assert abs(float(h_values[0]) - 16.0) < 1e-5
        # h=1: midpoint = 1*32+16 = 48
        assert abs(float(h_values[2]) - 48.0) < 1e-5
        # h=2: midpoint = 2*32+16 = 80
        assert abs(float(h_values[4]) - 80.0) < 1e-5

        # w=0: midpoint = 0*32+16 = 16
        assert abs(float(w_values[0]) - 16.0) < 1e-5
        # w=1: midpoint = 1*32+16 = 48
        assert abs(float(w_values[1]) - 48.0) < 1e-5

    def test_temporal_causal_fix_first_frame(self):
        """First latent frame: start=max(0, 0*8+1-8)=0, end=0*8+1=1. Mid=0.5/fps."""
        positions = compute_video_positions(num_frames=2, height=1, width=1, fps=24.0)
        t0 = float(positions[0, 0, 0])
        # start = max(0, 0*8 + 1 - 8) = 0
        # end = max(0, (0+1)*8 + 1 - 8) = 1
        # mid = (0 + 1) / 2 / 24 = 0.5 / 24
        expected = 0.5 / 24.0
        assert abs(t0 - expected) < 1e-5

    def test_temporal_causal_fix_second_frame(self):
        """Second latent frame: start=max(0, 1*8+1-8)=1, end=(1+1)*8+1-8=9. Mid=5/fps."""
        positions = compute_video_positions(num_frames=2, height=1, width=1, fps=24.0)
        t1 = float(positions[0, 1, 0])
        # start = max(0, 1*8 + 1 - 8) = 1
        # end = max(0, 2*8 + 1 - 8) = 9
        # mid = (1 + 9) / 2 / 24 = 5 / 24
        expected = 5.0 / 24.0
        assert abs(t1 - expected) < 1e-5

    def test_temporal_monotonic(self):
        """Temporal positions should be strictly increasing across frames."""
        positions = compute_video_positions(num_frames=10, height=1, width=1)
        times = positions[0, :, 0]
        for i in range(9):
            assert float(times[i]) < float(times[i + 1])

    def test_spatial_constant_across_frames(self):
        """Spatial positions should be the same for all frames."""
        positions = compute_video_positions(num_frames=3, height=2, width=2)
        # Token layout: frame 0 has indices 0-3, frame 1 has 4-7, frame 2 has 8-11
        for f in range(3):
            for hw in range(4):
                idx = f * 4 + hw
                h_val = float(positions[0, idx, 1])
                w_val = float(positions[0, idx, 2])
                # Same spatial position as frame 0
                h_ref = float(positions[0, hw, 1])
                w_ref = float(positions[0, hw, 2])
                assert abs(h_val - h_ref) < 1e-5
                assert abs(w_val - w_ref) < 1e-5

    def test_fps_scaling(self):
        """Different fps should scale temporal positions proportionally."""
        pos_24 = compute_video_positions(num_frames=2, height=1, width=1, fps=24.0)
        pos_48 = compute_video_positions(num_frames=2, height=1, width=1, fps=48.0)
        # At double fps, temporal positions should be half
        ratio = float(pos_24[0, 1, 0]) / float(pos_48[0, 1, 0])
        assert abs(ratio - 2.0) < 1e-5

    def test_all_positions_non_negative(self):
        positions = compute_video_positions(num_frames=5, height=3, width=4)
        assert float(mx.min(positions)) >= 0.0

    def test_token_count_matches_fhw(self):
        F, H, W = 4, 3, 5
        positions = compute_video_positions(num_frames=F, height=H, width=W)
        assert positions.shape[1] == F * H * W


# ---------------------------------------------------------------------------
# compute_audio_positions
# ---------------------------------------------------------------------------
class TestComputeAudioPositions:
    def test_shape(self):
        positions = compute_audio_positions(num_tokens=10)
        assert positions.shape == (1, 10, 1)

    def test_single_token(self):
        positions = compute_audio_positions(num_tokens=1)
        assert positions.shape == (1, 1, 1)

    def test_monotonic(self):
        """Audio positions should be monotonically non-decreasing."""
        positions = compute_audio_positions(num_tokens=50)
        for i in range(49):
            assert float(positions[0, i, 0]) <= float(positions[0, i + 1, 0])

    def test_first_tokens_causal(self):
        """First few tokens should have causal positions (clamped at 0)."""
        positions = compute_audio_positions(num_tokens=10)
        # Token 0: start = max(0, (0+1-4)*4) * 160/16000 = max(0, -12) * 0.01 = 0
        #          end   = max(0, (0+2-4)*4) * 160/16000 = max(0, -8)  * 0.01 = 0
        # mid = 0
        assert float(positions[0, 0, 0]) == 0.0

    def test_later_tokens_positive(self):
        """Later tokens should have positive position values."""
        positions = compute_audio_positions(num_tokens=20)
        # By token ~4, positions should be positive
        assert float(positions[0, -1, 0]) > 0.0

    def test_values_in_seconds(self):
        """Position values should be in real-time seconds."""
        positions = compute_audio_positions(num_tokens=100)
        # 100 tokens at 25 tokens/sec = 4 seconds
        last_pos = float(positions[0, -1, 0])
        # Should be roughly around 3-4 seconds
        assert 2.0 < last_pos < 5.0

    def test_dtype_float32(self):
        """Audio positions should be float32 (from arange)."""
        positions = compute_audio_positions(num_tokens=5)
        assert positions.dtype == mx.float32

    def test_zero_tokens(self):
        """Zero tokens should produce empty array."""
        positions = compute_audio_positions(num_tokens=0)
        assert positions.shape == (1, 0, 1)
