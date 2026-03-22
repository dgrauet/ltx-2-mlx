"""Tests for the Euler denoising math and res2s helpers."""

import math

import mlx.core as mx

from ltx_pipelines_mlx.utils.samplers import (
    _compute_per_token_timesteps,
    _is_uniform_mask,
    _res2s_coefficients,
    _res2s_phi,
    _res2s_sde_coeff,
    euler_step,
)


# ---------------------------------------------------------------------------
# euler_step
# ---------------------------------------------------------------------------
class TestEulerStep:
    def test_zero_sigma_returns_x0(self):
        x = mx.ones((1, 4, 8))
        x0 = mx.zeros((1, 4, 8))
        result = euler_step(x, x0, sigma=0.0, sigma_next=0.0)
        assert mx.allclose(result, x0).item()

    def test_step_towards_x0(self):
        x = mx.ones((1, 4, 8))
        x0 = mx.zeros((1, 4, 8))
        result = euler_step(x, x0, sigma=1.0, sigma_next=0.0)
        assert mx.allclose(result, x0, atol=1e-5).item()

    def test_partial_step(self):
        x = mx.ones((1, 4, 8)) * 2.0
        x0 = mx.zeros((1, 4, 8))
        result = euler_step(x, x0, sigma=1.0, sigma_next=0.5)
        expected = mx.ones((1, 4, 8))
        assert mx.allclose(result, expected, atol=1e-5).item()

    def test_identity_when_sigma_equals_sigma_next(self):
        """When sigma == sigma_next, no step is taken."""
        x = mx.ones((1, 4, 8)) * 3.0
        x0 = mx.zeros((1, 4, 8))
        result = euler_step(x, x0, sigma=0.5, sigma_next=0.5)
        assert mx.allclose(result, x, atol=1e-5).item()

    def test_euler_step_math(self):
        """Verify the formula: x + (sigma_next - sigma) * (x - x0) / sigma."""
        x = mx.array([[[4.0, 6.0]]])
        x0 = mx.array([[[2.0, 3.0]]])
        sigma = 2.0
        sigma_next = 1.0
        # d = (x - x0) / sigma = [1, 1.5]
        # result = x + (1 - 2) * d = [4, 6] + [-1, -1.5] = [3, 4.5]
        result = euler_step(x, x0, sigma, sigma_next)
        expected = mx.array([[[3.0, 4.5]]])
        assert mx.allclose(result, expected, atol=1e-5).item()

    def test_larger_sigma_next(self):
        """sigma_next > sigma should move away from x0 (reverse step)."""
        x = mx.zeros((1, 2, 2))
        x0 = mx.ones((1, 2, 2))
        # d = (0 - 1) / 0.5 = -2
        # result = 0 + (1 - 0.5) * (-2) = -1
        result = euler_step(x, x0, sigma=0.5, sigma_next=1.0)
        expected = mx.full((1, 2, 2), -1.0)
        assert mx.allclose(result, expected, atol=1e-5).item()

    def test_batch_dimension(self):
        x = mx.ones((3, 4, 8))
        x0 = mx.zeros((3, 4, 8))
        result = euler_step(x, x0, sigma=1.0, sigma_next=0.0)
        assert result.shape == (3, 4, 8)
        assert mx.allclose(result, x0, atol=1e-5).item()

    def test_x_equals_x0_no_change(self):
        """When x == x0, d = 0, so result should equal x0 regardless of sigmas."""
        x = mx.ones((1, 4, 8)) * 5.0
        x0 = mx.ones((1, 4, 8)) * 5.0
        result = euler_step(x, x0, sigma=0.8, sigma_next=0.3)
        assert mx.allclose(result, x0, atol=1e-5).item()


# ---------------------------------------------------------------------------
# _is_uniform_mask
# ---------------------------------------------------------------------------
class TestIsUniformMask:
    def test_all_ones(self):
        mask = mx.ones((1, 8, 1))
        assert _is_uniform_mask(mask) is True

    def test_has_zeros(self):
        mask = mx.array([[[1.0], [0.0], [1.0], [1.0]]])
        assert _is_uniform_mask(mask) is False

    def test_all_zeros(self):
        mask = mx.zeros((1, 4, 1))
        assert _is_uniform_mask(mask) is False

    def test_partial_values(self):
        mask = mx.full((1, 4, 1), 0.5)
        assert _is_uniform_mask(mask) is False

    def test_single_token(self):
        mask = mx.ones((1, 1, 1))
        assert _is_uniform_mask(mask) is True

    def test_batch_all_ones(self):
        mask = mx.ones((2, 8, 1))
        assert _is_uniform_mask(mask) is True

    def test_batch_mixed(self):
        """If any element is not 1.0, should return False."""
        mask = mx.ones((2, 4, 1))
        mask = mask.at[1, 0, 0].add(mx.array(-1.0))  # set to 0.0
        assert _is_uniform_mask(mask) is False


# ---------------------------------------------------------------------------
# _compute_per_token_timesteps
# ---------------------------------------------------------------------------
class TestComputePerTokenTimesteps:
    def test_uniform_mask(self):
        mask = mx.ones((1, 4, 1))
        result = _compute_per_token_timesteps(sigma=0.5, denoise_mask=mask)
        assert result.shape == (1, 4)
        assert mx.allclose(result, mx.full((1, 4), 0.5), atol=1e-5).item()

    def test_zero_mask(self):
        mask = mx.zeros((1, 4, 1))
        result = _compute_per_token_timesteps(sigma=0.8, denoise_mask=mask)
        assert mx.allclose(result, mx.zeros((1, 4)), atol=1e-5).item()

    def test_mixed_mask(self):
        mask = mx.array([[[1.0], [0.0], [0.5], [1.0]]])
        result = _compute_per_token_timesteps(sigma=0.6, denoise_mask=mask)
        expected = mx.array([[0.6, 0.0, 0.3, 0.6]])
        assert mx.allclose(result, expected, atol=1e-5).item()

    def test_sigma_zero(self):
        mask = mx.ones((1, 4, 1))
        result = _compute_per_token_timesteps(sigma=0.0, denoise_mask=mask)
        assert mx.allclose(result, mx.zeros((1, 4)), atol=1e-5).item()

    def test_sigma_one(self):
        mask = mx.array([[[1.0], [0.0]]])
        result = _compute_per_token_timesteps(sigma=1.0, denoise_mask=mask)
        expected = mx.array([[1.0, 0.0]])
        assert mx.allclose(result, expected, atol=1e-5).item()

    def test_output_shape_squeeze(self):
        """Last dimension should be squeezed: (B, N, 1) -> (B, N)."""
        mask = mx.ones((2, 10, 1))
        result = _compute_per_token_timesteps(sigma=0.3, denoise_mask=mask)
        assert result.shape == (2, 10)


# ---------------------------------------------------------------------------
# _res2s_phi
# ---------------------------------------------------------------------------
class TestRes2sPhi:
    def test_phi_0_near_zero(self):
        """phi_0(z) near z=0 should be 1/0! = 1."""
        result = _res2s_phi(0, 1e-15)
        assert abs(result - 1.0) < 1e-6

    def test_phi_1_near_zero(self):
        """phi_1(z) near z=0 should be 1/1! = 1."""
        result = _res2s_phi(1, 1e-15)
        assert abs(result - 1.0) < 1e-6

    def test_phi_2_near_zero(self):
        """phi_2(z) near z=0 should be 1/2! = 0.5."""
        result = _res2s_phi(2, 1e-15)
        assert abs(result - 0.5) < 1e-6

    def test_phi_0_nonzero(self):
        """phi_0(neg_h) = exp(neg_h)."""
        neg_h = -0.5
        result = _res2s_phi(0, neg_h)
        expected = math.exp(neg_h)
        assert abs(result - expected) < 1e-6

    def test_phi_1_nonzero(self):
        """phi_1(neg_h) = (exp(neg_h) - 1) / neg_h."""
        neg_h = -0.5
        result = _res2s_phi(1, neg_h)
        expected = (math.exp(neg_h) - 1.0) / neg_h
        assert abs(result - expected) < 1e-6

    def test_phi_2_nonzero(self):
        """phi_2(neg_h) = (exp(neg_h) - 1 - neg_h) / neg_h^2."""
        neg_h = -0.3
        result = _res2s_phi(2, neg_h)
        expected = (math.exp(neg_h) - 1.0 - neg_h) / (neg_h**2)
        assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# _res2s_coefficients
# ---------------------------------------------------------------------------
class TestRes2sCoefficients:
    def test_returns_three_floats(self):
        a21, b1, b2 = _res2s_coefficients(h=0.5)
        assert isinstance(a21, float)
        assert isinstance(b1, float)
        assert isinstance(b2, float)

    def test_near_zero_h(self):
        """Near-zero step size should give well-defined coefficients."""
        a21, b1, b2 = _res2s_coefficients(h=1e-12)
        assert math.isfinite(a21)
        assert math.isfinite(b1)
        assert math.isfinite(b2)

    def test_positive_h(self):
        a21, b1, b2 = _res2s_coefficients(h=1.0)
        assert math.isfinite(a21)
        assert math.isfinite(b1)
        assert math.isfinite(b2)

    def test_custom_c2(self):
        """c2=1.0 should give different coefficients than c2=0.5."""
        coeffs_half = _res2s_coefficients(h=0.5, c2=0.5)
        coeffs_one = _res2s_coefficients(h=0.5, c2=1.0)
        assert coeffs_half != coeffs_one


# ---------------------------------------------------------------------------
# _res2s_sde_coeff
# ---------------------------------------------------------------------------
class TestRes2sSdeCoeff:
    def test_returns_three_floats(self):
        alpha, sigma_down, sigma_up = _res2s_sde_coeff(sigma_next=0.5)
        assert isinstance(alpha, float)
        assert isinstance(sigma_down, float)
        assert isinstance(sigma_up, float)

    def test_sigma_up_bounded(self):
        """sigma_up should be <= sigma_next * sigma_up_fraction."""
        _, _, sigma_up = _res2s_sde_coeff(sigma_next=0.5, sigma_up_fraction=0.5)
        assert sigma_up <= 0.5 * 0.5 + 1e-8

    def test_sigma_up_bounded_by_0_9999(self):
        """sigma_up should also be bounded by sigma_next * 0.9999."""
        _, _, sigma_up = _res2s_sde_coeff(sigma_next=0.1, sigma_up_fraction=10.0)
        assert sigma_up <= 0.1 * 0.9999 + 1e-8

    def test_all_non_negative(self):
        alpha, sigma_down, sigma_up = _res2s_sde_coeff(sigma_next=0.3)
        assert alpha >= 0.0
        assert sigma_down >= 0.0
        assert sigma_up >= 0.0

    def test_zero_sigma_next(self):
        """sigma_next=0 should give sigma_up=0."""
        alpha, sigma_down, sigma_up = _res2s_sde_coeff(sigma_next=0.0)
        assert sigma_up == 0.0

    def test_sigma_up_fraction_zero(self):
        """sigma_up_fraction=0 should give sigma_up=0."""
        alpha, sigma_down, sigma_up = _res2s_sde_coeff(sigma_next=0.5, sigma_up_fraction=0.0)
        assert sigma_up == 0.0
