"""Tests for sigma schedules and dynamic schedulers."""

import pytest

from ltx_pipelines_mlx.scheduler import (
    DISTILLED_SIGMAS,
    STAGE_2_SIGMAS,
    get_sigma_schedule,
    ltx2_schedule,
    sigma_to_timestep,
)


# ---------------------------------------------------------------------------
# Predefined schedules
# ---------------------------------------------------------------------------
class TestDistilledSigmas:
    def test_has_9_values_for_8_steps(self):
        assert len(DISTILLED_SIGMAS) == 9

    def test_starts_at_1(self):
        assert DISTILLED_SIGMAS[0] == 1.0

    def test_ends_at_0(self):
        assert DISTILLED_SIGMAS[-1] == 0.0

    def test_strictly_decreasing(self):
        for i in range(len(DISTILLED_SIGMAS) - 1):
            assert DISTILLED_SIGMAS[i] > DISTILLED_SIGMAS[i + 1]

    def test_all_in_range_0_1(self):
        for s in DISTILLED_SIGMAS:
            assert 0.0 <= s <= 1.0

    def test_known_values(self):
        """Check some specific values from the predefined schedule."""
        assert DISTILLED_SIGMAS[1] == 0.99375
        assert DISTILLED_SIGMAS[5] == 0.909375
        assert DISTILLED_SIGMAS[7] == 0.421875


class TestStage2Sigmas:
    def test_has_4_values_for_3_steps(self):
        assert len(STAGE_2_SIGMAS) == 4

    def test_ends_at_0(self):
        assert STAGE_2_SIGMAS[-1] == 0.0

    def test_strictly_decreasing(self):
        for i in range(len(STAGE_2_SIGMAS) - 1):
            assert STAGE_2_SIGMAS[i] > STAGE_2_SIGMAS[i + 1]

    def test_all_in_range_0_1(self):
        for s in STAGE_2_SIGMAS:
            assert 0.0 <= s <= 1.0

    def test_starts_below_1(self):
        """Stage 2 starts at a lower sigma (not from full noise)."""
        assert STAGE_2_SIGMAS[0] < 1.0

    def test_stage2_values_subset_of_distilled(self):
        """Stage 2 sigmas should be a subset of distilled sigmas."""
        for s in STAGE_2_SIGMAS:
            assert s in DISTILLED_SIGMAS


# ---------------------------------------------------------------------------
# get_sigma_schedule
# ---------------------------------------------------------------------------
class TestGetSigmaSchedule:
    def test_distilled_full(self):
        sigmas = get_sigma_schedule("distilled")
        assert sigmas == DISTILLED_SIGMAS

    def test_stage_2_full(self):
        sigmas = get_sigma_schedule("stage_2")
        assert sigmas == STAGE_2_SIGMAS

    def test_truncate(self):
        sigmas = get_sigma_schedule("distilled", num_steps=4)
        assert len(sigmas) == 4
        assert sigmas == DISTILLED_SIGMAS[:4]

    def test_truncate_1(self):
        sigmas = get_sigma_schedule("distilled", num_steps=1)
        assert len(sigmas) == 1
        assert sigmas[0] == 1.0

    def test_truncate_none_returns_full(self):
        sigmas = get_sigma_schedule("distilled", num_steps=None)
        assert sigmas == DISTILLED_SIGMAS

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            get_sigma_schedule("unknown_schedule")

    def test_truncate_beyond_length(self):
        """Truncating beyond the schedule length returns the full schedule."""
        sigmas = get_sigma_schedule("distilled", num_steps=100)
        assert sigmas == DISTILLED_SIGMAS


# ---------------------------------------------------------------------------
# sigma_to_timestep
# ---------------------------------------------------------------------------
class TestSigmaToTimestep:
    def test_returns_1d_array(self):
        import mlx.core as mx

        t = sigma_to_timestep(0.5)
        assert t.shape == (1,)
        assert t.dtype == mx.bfloat16

    def test_value_matches(self):
        t = sigma_to_timestep(0.75)
        assert abs(float(t[0]) - 0.75) < 0.01  # bfloat16 precision

    def test_zero(self):
        t = sigma_to_timestep(0.0)
        assert float(t[0]) == 0.0

    def test_one(self):
        t = sigma_to_timestep(1.0)
        assert float(t[0]) == 1.0


# ---------------------------------------------------------------------------
# ltx2_schedule
# ---------------------------------------------------------------------------
class TestLtx2Schedule:
    def test_length(self):
        """Should return steps+1 values."""
        sigmas = ltx2_schedule(steps=10)
        assert len(sigmas) == 11

    def test_starts_near_1(self):
        """First sigma should be close to 1 (shifted)."""
        sigmas = ltx2_schedule(steps=20)
        assert sigmas[0] > 0.5

    def test_ends_at_0_with_stretch(self):
        """With stretch=True (default), last value should be 0."""
        sigmas = ltx2_schedule(steps=10, stretch=True)
        assert sigmas[-1] == 0.0

    def test_decreasing(self):
        """Sigmas should be monotonically decreasing."""
        sigmas = ltx2_schedule(steps=20)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1]

    def test_all_non_negative(self):
        sigmas = ltx2_schedule(steps=50)
        for s in sigmas:
            assert s >= 0.0

    def test_no_stretch(self):
        """Without stretch, last non-zero sigma may not reach terminal."""
        sigmas = ltx2_schedule(steps=10, stretch=False)
        assert len(sigmas) == 11
        assert sigmas[-1] == 0.0

    def test_single_step(self):
        sigmas = ltx2_schedule(steps=1)
        assert len(sigmas) == 2
        assert sigmas[-1] == 0.0

    def test_token_count_affects_shift(self):
        """More tokens should shift sigmas differently."""
        sigmas_small = ltx2_schedule(steps=10, num_tokens=1024)
        sigmas_large = ltx2_schedule(steps=10, num_tokens=4096)
        # The schedules should differ
        assert sigmas_small != sigmas_large

    def test_max_shift_param(self):
        """Different max_shift should produce different schedules."""
        s1 = ltx2_schedule(steps=10, max_shift=1.5)
        s2 = ltx2_schedule(steps=10, max_shift=3.0)
        assert s1 != s2

    def test_values_in_0_1(self):
        """All sigma values should be in [0, 1]."""
        sigmas = ltx2_schedule(steps=50)
        for s in sigmas:
            assert 0.0 <= s <= 1.0
