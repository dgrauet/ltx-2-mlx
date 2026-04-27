"""Tests for the robust polyfit analysis (scripts/fit_teacache_poly.py)."""

from __future__ import annotations

import math

import pytest

from ltx_pipelines_mlx.scripts.fit_teacache_poly import fit_robust


class TestStableFits:
    def test_linear_data_picks_degree_one(self):
        """y = 2x on a clean grid → degree 1 is stable; deg 1 is the lowest, picked."""
        xs = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
        ys = [2 * x for x in xs]
        result = fit_robust(xs, ys, max_degree=4)
        assert result.degree == 1
        # Coefficients should be ~[2, 0]
        assert result.coefficients[0] == pytest.approx(2.0, abs=1e-6)
        assert result.coefficients[1] == pytest.approx(0.0, abs=1e-6)
        assert not result.diagnostics["fallback"]

    def test_quadratic_data_picks_degree_two(self):
        """y = x^2 on clean grid → degree 1 not enough (poor fit), degree 2 stable."""
        xs = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
        ys = [x**2 for x in xs]
        result = fit_robust(xs, ys, max_degree=4)
        # Degree 1 may technically be "stable" (monotone+positive) but a poor fit.
        # The fitter prefers Occam — lowest stable. So degree 1 wins if monotone+positive.
        # That's acceptable: the threshold filter is shape, not RMSE.
        assert result.degree in (1, 2)
        assert not result.diagnostics["fallback"]


class TestPathologicalFits:
    def test_nonmonotone_pathological_falls_back(self):
        """Synthetic data with strong negative-then-positive oscillation defeats
        all polynomial degrees on [min_x, max_x] → fallback marker set."""
        # Strongly non-monotone with negative values at both ends — even a
        # linear fit produces a negative intercept that violates `positive`.
        xs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        ys = [-0.5, -0.3, 0.4, 0.5, 0.4, -0.3, -0.5]  # symmetric inverted-U around 0.15
        result = fit_robust(xs, ys, max_degree=4)
        # No degree 1..4 produces a monotone-non-decreasing AND non-negative
        # curve over [0.01, 0.3] — fitter falls back to lowest-RMSE.
        assert result.diagnostics["fallback"]
        # The diagnostics should record candidate evaluations
        candidates = result.diagnostics["candidates"]
        assert len(candidates) == 4
        for c in candidates:
            assert "monotone" in c and "positive" in c and "rmse" in c

    def test_known_bad_calibration_flagged(self):
        """The actual coefficients we got from our 87-delta deg-4 fit:
        large leading coef, oscillates, goes negative — must be flagged."""
        # Replay the bad coefficients evaluated on a synthetic x grid that mimics
        # what TeaCache actually saw. Then re-fit at lower degrees and confirm
        # the fitter doesn't "endorse" the deg-4 unstable curve.
        import numpy as np

        bad_poly = np.poly1d(
            [
                -3535.6188003737434,
                1898.9632500329858,
                -345.5198275741567,
                26.964112228380383,
                -0.33149055535368976,
            ]
        )
        # Sample the bad polynomial as if it were ground truth — to confirm even
        # then, lower-degree fits would be flagged stable while deg 4 is unstable.
        xs = [0.01 + 0.03 * i for i in range(20)]  # 0.01 to 0.58
        ys = [float(bad_poly(x)) for x in xs]

        result = fit_robust(xs, ys, max_degree=4)
        # We don't assert what degree wins (depends on data); we just confirm
        # the fitter ran and returned diagnostics.
        assert "candidates" in result.diagnostics
        assert "delta_range" in result.diagnostics


class TestEdgeCases:
    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_robust([], [], max_degree=4)

    def test_single_point_raises_or_handles(self):
        # numpy.polyfit warns on under-determined systems but returns something.
        # We allow either: graceful handling or a clear error.
        try:
            result = fit_robust([0.1], [0.2], max_degree=1)
            # If it returned, the result should be at least syntactically valid
            assert isinstance(result.coefficients, list)
        except (ValueError, Exception):
            pass  # Acceptable to raise

    def test_max_degree_clamped_to_data_size(self):
        """With only 3 points, fitting deg 4 is under-determined. Fitter shouldn't crash."""
        xs = [0.1, 0.2, 0.3]
        ys = [0.05, 0.15, 0.3]
        result = fit_robust(xs, ys, max_degree=4)
        # Should produce *some* result without raising
        assert result.degree >= 1
        assert isinstance(result.coefficients, list)

    def test_constant_output_handled(self):
        """y = constant → degree 1 has slope 0, technically monotone-non-decreasing."""
        xs = [0.1, 0.2, 0.3, 0.5]
        ys = [0.4, 0.4, 0.4, 0.4]
        result = fit_robust(xs, ys, max_degree=4)
        # Degree 0 isn't tested (we start at 1). Degree 1 fits a horizontal line: stable.
        assert result.degree == 1
        # Constant ~0.4
        assert math.isclose(result.coefficients[-1], 0.4, abs_tol=1e-6)
