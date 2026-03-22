"""Comprehensive tests for the conditioning system."""

import mlx.core as mx
import pytest

from ltx_core_mlx.conditioning.mask_utils import (
    build_attention_mask,
    resolve_cross_mask,
    update_attention_mask,
)
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    TemporalRegionMask,
    VideoConditionByLatentIndex,
    add_noise_with_state,
    apply_conditioning,
    apply_denoise_mask,
    create_initial_state,
    noise_latent_state,
)
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent


# ---------------------------------------------------------------------------
# LatentState
# ---------------------------------------------------------------------------
class TestLatentState:
    def test_create_initial_state_shape(self):
        state = create_initial_state((1, 16, 8), seed=42)
        assert state.latent.shape == (1, 16, 8)
        assert state.clean_latent.shape == (1, 16, 8)
        assert state.denoise_mask.shape == (1, 16, 1)

    def test_create_initial_state_mask_all_ones(self):
        state = create_initial_state((1, 16, 8), seed=42)
        assert float(mx.min(state.denoise_mask)) == 1.0
        assert float(mx.max(state.denoise_mask)) == 1.0

    def test_create_initial_state_dtype_bfloat16(self):
        state = create_initial_state((1, 4, 8), seed=0)
        assert state.latent.dtype == mx.bfloat16
        assert state.clean_latent.dtype == mx.bfloat16
        assert state.denoise_mask.dtype == mx.bfloat16

    def test_create_with_clean_latent(self):
        clean = mx.ones((1, 16, 8))
        state = create_initial_state((1, 16, 8), seed=42, clean_latent=clean)
        assert mx.allclose(state.clean_latent, clean).item()

    def test_default_clean_latent_is_zeros(self):
        state = create_initial_state((1, 4, 8), seed=42)
        assert float(mx.max(mx.abs(state.clean_latent))) == 0.0

    def test_default_optional_fields(self):
        state = create_initial_state((1, 16, 8), seed=42)
        assert state.positions is None
        assert state.attention_mask is None

    def test_create_with_positions(self):
        positions = mx.zeros((1, 16, 3))
        state = create_initial_state((1, 16, 8), seed=42, positions=positions)
        assert state.positions is not None
        assert state.positions.shape == (1, 16, 3)

    def test_reproducible_with_same_seed(self):
        s1 = create_initial_state((1, 4, 8), seed=123)
        s2 = create_initial_state((1, 4, 8), seed=123)
        assert mx.allclose(s1.latent.astype(mx.float32), s2.latent.astype(mx.float32), atol=1e-5).item()

    def test_different_seeds_produce_different_noise(self):
        s1 = create_initial_state((1, 4, 8), seed=1)
        s2 = create_initial_state((1, 4, 8), seed=2)
        assert not mx.allclose(s1.latent.astype(mx.float32), s2.latent.astype(mx.float32), atol=1e-3).item()

    def test_batch_size_greater_than_one(self):
        state = create_initial_state((2, 8, 4), seed=42)
        assert state.latent.shape == (2, 8, 4)
        assert state.denoise_mask.shape == (2, 8, 1)

    def test_single_token(self):
        state = create_initial_state((1, 1, 4), seed=42)
        assert state.latent.shape == (1, 1, 4)
        assert state.denoise_mask.shape == (1, 1, 1)


# ---------------------------------------------------------------------------
# apply_denoise_mask
# ---------------------------------------------------------------------------
class TestApplyDenoiseMask:
    def test_full_denoise(self):
        x0 = mx.ones((1, 4, 8))
        clean = mx.zeros((1, 4, 8))
        mask = mx.ones((1, 4, 1))
        result = apply_denoise_mask(x0, clean, mask)
        assert mx.allclose(result, x0).item()

    def test_full_preserve(self):
        x0 = mx.ones((1, 4, 8))
        clean = mx.zeros((1, 4, 8))
        mask = mx.zeros((1, 4, 1))
        result = apply_denoise_mask(x0, clean, mask)
        assert mx.allclose(result, clean).item()

    def test_half_mask(self):
        x0 = mx.ones((1, 4, 8))
        clean = mx.zeros((1, 4, 8))
        mask = mx.array([[[1.0], [1.0], [0.0], [0.0]]])
        result = apply_denoise_mask(x0, clean, mask)
        assert float(mx.sum(result[:, :2, :])) > 0
        assert float(mx.sum(result[:, 2:, :])) == 0

    def test_partial_strength_blends(self):
        x0 = mx.ones((1, 2, 4)) * 10.0
        clean = mx.zeros((1, 2, 4))
        mask = mx.full((1, 2, 1), 0.3)
        result = apply_denoise_mask(x0, clean, mask)
        expected = x0 * 0.3 + clean * 0.7
        assert mx.allclose(result, expected, atol=1e-5).item()

    def test_broadcasting_over_channels(self):
        """Mask (B,N,1) should broadcast over C dimension."""
        x0 = mx.ones((1, 3, 8)) * 5.0
        clean = mx.ones((1, 3, 8)) * 2.0
        mask = mx.array([[[1.0], [0.5], [0.0]]])
        result = apply_denoise_mask(x0, clean, mask)
        assert mx.allclose(result[0, 0, :], mx.full((8,), 5.0), atol=1e-5).item()
        assert mx.allclose(result[0, 1, :], mx.full((8,), 3.5), atol=1e-5).item()
        assert mx.allclose(result[0, 2, :], mx.full((8,), 2.0), atol=1e-5).item()

    def test_batch_dimension(self):
        x0 = mx.ones((2, 4, 8))
        clean = mx.zeros((2, 4, 8))
        mask = mx.ones((2, 4, 1))
        result = apply_denoise_mask(x0, clean, mask)
        assert result.shape == (2, 4, 8)


# ---------------------------------------------------------------------------
# noise_latent_state
# ---------------------------------------------------------------------------
class TestNoiseLatentState:
    def test_sigma_zero_preserves_clean(self):
        state = create_initial_state((1, 4, 8), seed=42)
        state = LatentState(
            latent=state.latent,
            clean_latent=mx.ones((1, 4, 8), dtype=mx.bfloat16),
            denoise_mask=mx.ones((1, 4, 1), dtype=mx.bfloat16),
        )
        result = noise_latent_state(state, sigma=0.0, seed=99)
        assert mx.allclose(result.latent.astype(mx.float32), mx.ones((1, 4, 8)), atol=1e-2).item()

    def test_sigma_one_full_noise(self):
        clean = mx.zeros((1, 4, 8), dtype=mx.bfloat16)
        state = LatentState(
            latent=mx.zeros((1, 4, 8), dtype=mx.bfloat16),
            clean_latent=clean,
            denoise_mask=mx.ones((1, 4, 1), dtype=mx.bfloat16),
        )
        result = noise_latent_state(state, sigma=1.0, seed=42)
        # With sigma=1.0 and mask=1.0: latent = noise * 1.0 + clean * 0.0 = noise
        # Result should not equal clean (zeros)
        assert float(mx.sum(mx.abs(result.latent))) > 0

    def test_preserved_region_stays_clean(self):
        clean = mx.ones((1, 4, 8), dtype=mx.bfloat16) * 5.0
        mask = mx.array([[[0.0], [0.0], [1.0], [1.0]]], dtype=mx.bfloat16)
        state = LatentState(
            latent=mx.zeros((1, 4, 8), dtype=mx.bfloat16),
            clean_latent=clean,
            denoise_mask=mask,
        )
        result = noise_latent_state(state, sigma=1.0, seed=42)
        # Preserved tokens (mask=0): latent = noise*0 + clean*1 = clean
        preserved = result.latent[:, :2, :].astype(mx.float32)
        assert mx.allclose(preserved, mx.full((1, 2, 8), 5.0), atol=0.1).item()

    def test_preserves_metadata(self):
        positions = mx.zeros((1, 4, 3))
        attn = mx.ones((1, 4, 4))
        state = LatentState(
            latent=mx.zeros((1, 4, 8), dtype=mx.bfloat16),
            clean_latent=mx.zeros((1, 4, 8), dtype=mx.bfloat16),
            denoise_mask=mx.ones((1, 4, 1), dtype=mx.bfloat16),
            positions=positions,
            attention_mask=attn,
        )
        result = noise_latent_state(state, sigma=0.5, seed=42)
        assert result.positions is not None
        assert result.attention_mask is not None
        assert result.positions.shape == (1, 4, 3)


# ---------------------------------------------------------------------------
# add_noise_with_state
# ---------------------------------------------------------------------------
class TestAddNoiseWithState:
    def test_zero_sigma_returns_clean(self):
        clean = mx.ones((1, 4, 8), dtype=mx.bfloat16)
        latent = mx.ones((1, 4, 8), dtype=mx.bfloat16) * 2.0
        state = LatentState(
            latent=latent,
            clean_latent=clean,
            denoise_mask=mx.ones((1, 4, 1), dtype=mx.bfloat16),
        )
        result = add_noise_with_state(state, mx.array([0.0]))
        assert mx.allclose(result.astype(mx.float32), clean.astype(mx.float32), atol=1e-2).item()

    def test_preserved_tokens_unaffected(self):
        clean = mx.ones((1, 4, 8), dtype=mx.bfloat16) * 3.0
        noise_dir = mx.ones((1, 4, 8), dtype=mx.bfloat16) * 10.0
        latent = clean + noise_dir  # latent - clean = noise_dir
        mask = mx.array([[[0.0], [0.0], [1.0], [1.0]]], dtype=mx.bfloat16)
        state = LatentState(latent=latent, clean_latent=clean, denoise_mask=mask)
        result = add_noise_with_state(state, mx.array([0.5]))
        # Preserved tokens (mask=0): effective_sigma=0, result = clean
        preserved = result[:, :2, :].astype(mx.float32)
        assert mx.allclose(preserved, mx.full((1, 2, 8), 3.0), atol=0.1).item()


# ---------------------------------------------------------------------------
# apply_conditioning
# ---------------------------------------------------------------------------
class TestApplyConditioning:
    def test_empty_list_no_change(self):
        state = create_initial_state((1, 8, 4), seed=42)
        result = apply_conditioning(state, [], spatial_dims=(4, 1, 2))
        assert mx.allclose(result.latent.astype(mx.float32), state.latent.astype(mx.float32), atol=1e-5).item()

    def test_chain_multiple_conditions(self):
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        cond1 = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        cond2 = VideoConditionByLatentIndex(frame_indices=[3], clean_latent=clean)
        result = apply_conditioning(state, [cond1, cond2], spatial_dims=(4, 1, 2))
        # Both frame 0 and frame 3 should be preserved
        assert float(result.denoise_mask[0, 0, 0]) == 0.0
        assert float(result.denoise_mask[0, 1, 0]) == 0.0
        assert float(result.denoise_mask[0, 6, 0]) == 0.0
        assert float(result.denoise_mask[0, 7, 0]) == 0.0
        # Middle frames should stay as 1.0
        assert float(result.denoise_mask[0, 2, 0]) == 1.0


# ---------------------------------------------------------------------------
# TemporalRegionMask
# ---------------------------------------------------------------------------
class TestTemporalRegionMask:
    def test_mask_shape(self):
        region = TemporalRegionMask(start_frame=1, end_frame=3)
        mask = region.create_mask(num_frames=4, tokens_per_frame=2)
        assert mask.shape == (1, 8, 1)

    def test_region_values(self):
        region = TemporalRegionMask(start_frame=1, end_frame=3)
        mask = region.create_mask(num_frames=4, tokens_per_frame=2)
        # Frame 0 tokens (idx 0,1): preserved (0)
        assert float(mask[0, 0, 0]) == 0.0
        assert float(mask[0, 1, 0]) == 0.0
        # Frame 1-2 tokens (idx 2-5): denoise (1)
        assert float(mask[0, 2, 0]) == 1.0
        assert float(mask[0, 3, 0]) == 1.0
        assert float(mask[0, 4, 0]) == 1.0
        assert float(mask[0, 5, 0]) == 1.0
        # Frame 3 tokens (idx 6,7): preserved (0)
        assert float(mask[0, 6, 0]) == 0.0
        assert float(mask[0, 7, 0]) == 0.0

    def test_full_range(self):
        region = TemporalRegionMask(start_frame=0, end_frame=4)
        mask = region.create_mask(num_frames=4, tokens_per_frame=2)
        assert float(mx.min(mask)) == 1.0

    def test_single_frame(self):
        region = TemporalRegionMask(start_frame=2, end_frame=3)
        mask = region.create_mask(num_frames=5, tokens_per_frame=1)
        for i in range(5):
            expected = 1.0 if i == 2 else 0.0
            assert float(mask[0, i, 0]) == expected

    def test_end_clamped_to_total(self):
        """end_frame beyond num_frames should be clamped."""
        region = TemporalRegionMask(start_frame=3, end_frame=100)
        mask = region.create_mask(num_frames=5, tokens_per_frame=1)
        assert mask.shape == (1, 5, 1)
        assert float(mask[0, 3, 0]) == 1.0
        assert float(mask[0, 4, 0]) == 1.0

    def test_empty_region(self):
        """start_frame == end_frame produces no denoise region."""
        region = TemporalRegionMask(start_frame=2, end_frame=2)
        mask = region.create_mask(num_frames=4, tokens_per_frame=2)
        assert float(mx.max(mask)) == 0.0

    def test_multiple_tokens_per_frame(self):
        region = TemporalRegionMask(start_frame=0, end_frame=1)
        mask = region.create_mask(num_frames=2, tokens_per_frame=4)
        assert mask.shape == (1, 8, 1)
        # First 4 tokens (frame 0): denoise
        assert float(mx.sum(mask[:, :4, :])) == 4.0
        # Last 4 tokens (frame 1): preserve
        assert float(mx.sum(mask[:, 4:, :])) == 0.0


# ---------------------------------------------------------------------------
# VideoConditionByLatentIndex
# ---------------------------------------------------------------------------
class TestVideoConditionByLatentIndex:
    def test_preserves_frame(self):
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert float(new_state.denoise_mask[0, 0, 0]) == 0.0
        assert float(new_state.denoise_mask[0, 1, 0]) == 0.0

    def test_updates_latent_field(self):
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert mx.allclose(new_state.latent[0, :2, :], clean[0], atol=1e-6).item()

    def test_updates_clean_latent_field(self):
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4)) * 7.0
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert mx.allclose(new_state.clean_latent[0, :2, :], clean[0], atol=1e-6).item()

    def test_strength_partial(self):
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean, strength=0.5)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert abs(float(new_state.denoise_mask[0, 0, 0]) - 0.5) < 1e-6
        assert abs(float(new_state.denoise_mask[0, 1, 0]) - 0.5) < 1e-6
        assert float(new_state.denoise_mask[0, 2, 0]) == 1.0

    def test_strength_zero_means_full_denoise(self):
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean, strength=0.0)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        # mask_value = 1.0 - 0.0 = 1.0 (full denoise, no conditioning effect)
        assert float(new_state.denoise_mask[0, 0, 0]) == 1.0

    def test_positions_preserved(self):
        positions = mx.zeros((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.positions is not None
        assert new_state.positions.shape == (1, 8, 3)

    def test_multiple_frames(self):
        """Condition frames 0 and 2 in a 4-frame sequence with 2 tokens/frame."""
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 4, 4))  # 2 frames * 2 tokens/frame
        condition = VideoConditionByLatentIndex(frame_indices=[0, 2], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        # Frame 0: preserved
        assert float(new_state.denoise_mask[0, 0, 0]) == 0.0
        assert float(new_state.denoise_mask[0, 1, 0]) == 0.0
        # Frame 1: generated
        assert float(new_state.denoise_mask[0, 2, 0]) == 1.0
        # Frame 2: preserved
        assert float(new_state.denoise_mask[0, 4, 0]) == 0.0
        assert float(new_state.denoise_mask[0, 5, 0]) == 0.0

    def test_last_frame(self):
        """Condition the last frame."""
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[3], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert float(new_state.denoise_mask[0, 6, 0]) == 0.0
        assert float(new_state.denoise_mask[0, 7, 0]) == 0.0

    def test_shape_preserved(self):
        """Output shape should be the same as input."""
        state = create_initial_state((1, 8, 4), seed=42)
        clean = mx.ones((1, 2, 4))
        condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=clean)
        new_state = condition.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.latent.shape == (1, 8, 4)
        assert new_state.clean_latent.shape == (1, 8, 4)
        assert new_state.denoise_mask.shape == (1, 8, 1)


# ---------------------------------------------------------------------------
# resolve_cross_mask
# ---------------------------------------------------------------------------
class TestResolveCrossMask:
    def test_scalar_float(self):
        mask = resolve_cross_mask(1.0, num_new_tokens=3, batch_size=2)
        assert mask.shape == (2, 3)
        assert float(mx.min(mask)) == 1.0

    def test_scalar_int(self):
        mask = resolve_cross_mask(0, num_new_tokens=4, batch_size=1)
        assert mask.shape == (1, 4)
        assert float(mx.max(mask)) == 0.0

    def test_0d_array(self):
        mask = resolve_cross_mask(mx.array(0.5), num_new_tokens=3, batch_size=2)
        assert mask.shape == (2, 3)
        assert abs(float(mask[0, 0]) - 0.5) < 1e-5

    def test_1d_array(self):
        arr = mx.array([1.0, 0.0, 0.5])
        mask = resolve_cross_mask(arr, num_new_tokens=3, batch_size=2)
        assert mask.shape == (2, 3)
        assert abs(float(mask[1, 2]) - 0.5) < 1e-5

    def test_1d_wrong_length_raises(self):
        with pytest.raises(ValueError, match="num_new_tokens"):
            resolve_cross_mask(mx.array([1.0, 0.0]), num_new_tokens=3, batch_size=1)

    def test_2d_array(self):
        arr = mx.ones((1, 4))
        mask = resolve_cross_mask(arr, num_new_tokens=4, batch_size=2)
        assert mask.shape == (2, 4)

    def test_2d_wrong_dim1_raises(self):
        with pytest.raises(ValueError, match="num_new_tokens"):
            resolve_cross_mask(mx.ones((1, 3)), num_new_tokens=4, batch_size=1)

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="0-D, 1-D, or 2-D"):
            resolve_cross_mask(mx.ones((1, 2, 3)), num_new_tokens=2, batch_size=1)


# ---------------------------------------------------------------------------
# build_attention_mask
# ---------------------------------------------------------------------------
class TestBuildAttentionMask:
    def test_first_conditioning_item(self):
        cross_mask = mx.ones((1, 4))
        mask = build_attention_mask(
            existing_mask=None,
            num_noisy_tokens=8,
            num_new_tokens=4,
            num_existing_tokens=8,
            cross_mask=cross_mask,
        )
        assert mask.shape == (1, 12, 12)
        assert float(mx.min(mask[0, :8, :8])) == 1.0
        assert float(mx.min(mask[0, 8:, 8:])) == 1.0
        assert float(mx.min(mask[0, :8, 8:])) == 1.0
        assert float(mx.min(mask[0, 8:, :8])) == 1.0

    def test_cross_mask_zeros(self):
        cross_mask = mx.zeros((1, 4))
        mask = build_attention_mask(
            existing_mask=None,
            num_noisy_tokens=8,
            num_new_tokens=4,
            num_existing_tokens=8,
            cross_mask=cross_mask,
        )
        assert float(mx.max(mask[0, :8, 8:])) == 0.0
        assert float(mx.max(mask[0, 8:, :8])) == 0.0

    def test_second_conditioning_preserves_existing(self):
        existing = mx.ones((1, 12, 12))
        existing = existing.at[:, 8:, :8].add(mx.full((1, 4, 8), -1.0))
        cross_mask = mx.ones((1, 3))
        mask = build_attention_mask(
            existing_mask=existing,
            num_noisy_tokens=8,
            num_new_tokens=3,
            num_existing_tokens=12,
            cross_mask=cross_mask,
        )
        assert mask.shape == (1, 15, 15)
        assert mx.allclose(mask[:, :12, :12], existing, atol=1e-6).item()
        assert float(mx.max(mask[0, 8:12, 12:])) == 0.0

    def test_batch_size_two(self):
        cross_mask = mx.ones((2, 3))
        mask = build_attention_mask(
            existing_mask=None,
            num_noisy_tokens=4,
            num_new_tokens=3,
            num_existing_tokens=4,
            cross_mask=cross_mask,
        )
        assert mask.shape == (2, 7, 7)

    def test_new_ref_self_attend(self):
        """New reference tokens should always self-attend (bottom-right block = 1)."""
        cross_mask = mx.zeros((1, 5))  # No cross-attention
        mask = build_attention_mask(
            existing_mask=None,
            num_noisy_tokens=4,
            num_new_tokens=5,
            num_existing_tokens=4,
            cross_mask=cross_mask,
        )
        # Bottom-right 5x5 should be all 1s
        assert float(mx.min(mask[0, 4:, 4:])) == 1.0


# ---------------------------------------------------------------------------
# update_attention_mask
# ---------------------------------------------------------------------------
class TestUpdateAttentionMask:
    def test_none_mask_none_existing(self):
        state = LatentState(
            latent=mx.zeros((1, 8, 4)),
            clean_latent=mx.zeros((1, 8, 4)),
            denoise_mask=mx.ones((1, 8, 1)),
        )
        result = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=8,
            num_new_tokens=4,
            batch_size=1,
        )
        assert result is None

    def test_scalar_mask(self):
        state = LatentState(
            latent=mx.zeros((1, 8, 4)),
            clean_latent=mx.zeros((1, 8, 4)),
            denoise_mask=mx.ones((1, 8, 1)),
        )
        result = update_attention_mask(
            latent_state=state,
            attention_mask=1.0,
            num_noisy_tokens=8,
            num_new_tokens=4,
            batch_size=1,
        )
        assert result is not None
        assert result.shape == (1, 12, 12)

    def test_none_mask_with_existing_expands(self):
        """When attention_mask=None but state has existing mask, expand with 1s."""
        existing = mx.ones((1, 8, 8))
        state = LatentState(
            latent=mx.zeros((1, 8, 4)),
            clean_latent=mx.zeros((1, 8, 4)),
            denoise_mask=mx.ones((1, 8, 1)),
            attention_mask=existing,
        )
        result = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=8,
            num_new_tokens=3,
            batch_size=1,
        )
        assert result is not None
        assert result.shape == (1, 11, 11)

    def test_scalar_zero_blocks_cross(self):
        state = LatentState(
            latent=mx.zeros((1, 4, 4)),
            clean_latent=mx.zeros((1, 4, 4)),
            denoise_mask=mx.ones((1, 4, 1)),
        )
        result = update_attention_mask(
            latent_state=state,
            attention_mask=0.0,
            num_noisy_tokens=4,
            num_new_tokens=2,
            batch_size=1,
        )
        assert result is not None
        # Cross attention blocks should be zero
        assert float(mx.max(result[0, :4, 4:])) == 0.0
        assert float(mx.max(result[0, 4:, :4])) == 0.0


# ---------------------------------------------------------------------------
# VideoConditionByKeyframeIndex
# ---------------------------------------------------------------------------
class TestVideoConditionByKeyframeIndex:
    """Tests for the per-keyframe conditioning (new API: one instance per keyframe)."""

    def _make_cond(self, frame_idx: int = 0, num_tokens: int = 2, channels: int = 4, strength: float = 1.0):
        """Helper to create a VideoConditionByKeyframeIndex."""
        # spatial_dims: F=4 frames, H=1, W=2 -> 2 tokens per frame
        latent = mx.ones((1, num_tokens, channels))
        return VideoConditionByKeyframeIndex(
            frame_idx=frame_idx,
            keyframe_latent=latent,
            spatial_dims=(4, 1, 2),
            strength=strength,
        )

    def test_appends_tokens(self):
        state = create_initial_state((1, 8, 4), seed=42)
        cond = self._make_cond(frame_idx=0, num_tokens=2)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.latent.shape == (1, 10, 4)
        assert new_state.clean_latent.shape == (1, 10, 4)
        assert new_state.denoise_mask.shape == (1, 10, 1)

    def test_strength_default(self):
        state = create_initial_state((1, 8, 4), seed=42)
        cond = self._make_cond(frame_idx=0, num_tokens=2)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        # strength=1.0 -> mask_value=0.0 (preserved)
        assert float(new_state.denoise_mask[0, 8, 0]) == 0.0
        assert float(new_state.denoise_mask[0, 9, 0]) == 0.0

    def test_strength_partial(self):
        state = create_initial_state((1, 8, 4), seed=42)
        cond = self._make_cond(frame_idx=0, num_tokens=2, strength=0.5)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert abs(float(new_state.denoise_mask[0, 8, 0]) - 0.5) < 1e-6

    def test_positions_extended(self):
        positions = mx.zeros((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        cond = self._make_cond(frame_idx=0, num_tokens=2)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.positions is not None
        assert new_state.positions.shape == (1, 10, 3)

    def test_no_attention_mask_when_none(self):
        state = create_initial_state((1, 8, 4), seed=42)
        cond = self._make_cond(frame_idx=0, num_tokens=2)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.attention_mask is None

    def test_attention_mask_padded_when_existing(self):
        existing_mask = mx.ones((1, 8, 8))
        state = LatentState(
            latent=mx.zeros((1, 8, 4)),
            clean_latent=mx.zeros((1, 8, 4)),
            denoise_mask=mx.ones((1, 8, 1)),
            attention_mask=existing_mask,
        )
        cond = self._make_cond(frame_idx=0, num_tokens=2)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.attention_mask is not None
        assert new_state.attention_mask.shape == (1, 10, 10)

    def test_original_mask_unchanged(self):
        """Original state's denoise mask should be unchanged in output."""
        state = create_initial_state((1, 8, 4), seed=42)
        cond = self._make_cond(frame_idx=0, num_tokens=2)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert float(mx.min(new_state.denoise_mask[:, :8, :])) == 1.0

    def test_keyframe_values_in_latent(self):
        """Appended latent tokens should match keyframe_latent."""
        state = create_initial_state((1, 4, 4), seed=42)
        latent = mx.ones((1, 2, 4)) * 42.0
        cond = VideoConditionByKeyframeIndex(
            frame_idx=0,
            keyframe_latent=latent,
            spatial_dims=(2, 1, 2),
        )
        new_state = cond.apply(state, spatial_dims=(2, 1, 2))
        appended = new_state.latent[:, 4:, :].astype(mx.float32)
        assert mx.allclose(appended, mx.full((1, 2, 4), 42.0), atol=1e-5).item()

    def test_sequential_application(self):
        """Two keyframes applied sequentially should append independently."""
        state = create_initial_state((1, 8, 4), seed=42)
        cond0 = self._make_cond(frame_idx=0, num_tokens=2)
        cond1 = self._make_cond(frame_idx=3, num_tokens=2)
        state = cond0.apply(state, spatial_dims=(4, 1, 2))
        state = cond1.apply(state, spatial_dims=(4, 1, 2))
        # 8 original + 2 kf0 + 2 kf1 = 12
        assert state.latent.shape == (1, 12, 4)


# ---------------------------------------------------------------------------
# VideoConditionByReferenceLatent
# ---------------------------------------------------------------------------
class TestVideoConditionByReferenceLatent:
    def test_appends_tokens(self):
        state = create_initial_state((1, 8, 4), seed=42)
        ref = mx.ones((1, 4, 4))
        cond = VideoConditionByReferenceLatent(reference_latent=ref)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert new_state.latent.shape == (1, 12, 4)
        assert new_state.denoise_mask.shape == (1, 12, 1)

    def test_strength_default(self):
        state = create_initial_state((1, 8, 4), seed=42)
        ref = mx.ones((1, 4, 4))
        cond = VideoConditionByReferenceLatent(reference_latent=ref)
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        assert float(new_state.denoise_mask[0, 8, 0]) == 0.0

    def test_downscale_factor_scales_spatial_only(self):
        positions = mx.ones((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        ref = mx.ones((1, 4, 4))
        ref_positions = mx.ones((1, 4, 3))
        cond = VideoConditionByReferenceLatent(
            reference_latent=ref,
            reference_positions=ref_positions,
            downscale_factor=2,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        ref_pos = new_state.positions[:, 8:, :]
        assert abs(float(ref_pos[0, 0, 0]) - 1.0) < 1e-6  # time unchanged
        assert abs(float(ref_pos[0, 0, 1]) - 2.0) < 1e-6  # h scaled by 2
        assert abs(float(ref_pos[0, 0, 2]) - 2.0) < 1e-6  # w scaled by 2

    def test_downscale_factor_1_no_scaling(self):
        positions = mx.ones((1, 8, 3))
        state = create_initial_state((1, 8, 4), seed=42, positions=positions)
        ref = mx.ones((1, 4, 4))
        ref_positions = mx.ones((1, 4, 3))
        cond = VideoConditionByReferenceLatent(
            reference_latent=ref,
            reference_positions=ref_positions,
            downscale_factor=1,
        )
        new_state = cond.apply(state, spatial_dims=(4, 1, 2))
        ref_pos = new_state.positions[:, 8:, :]
        assert mx.allclose(ref_pos, ref_positions, atol=1e-6).item()

    def test_downscale_factor_4(self):
        """Test with a larger downscale factor."""
        positions = mx.ones((1, 4, 3))
        state = create_initial_state((1, 4, 4), seed=42, positions=positions)
        ref = mx.ones((1, 2, 4))
        ref_positions = mx.ones((1, 2, 3)) * 10.0
        cond = VideoConditionByReferenceLatent(
            reference_latent=ref,
            reference_positions=ref_positions,
            downscale_factor=4,
        )
        new_state = cond.apply(state, spatial_dims=(2, 1, 2))
        ref_pos = new_state.positions[:, 4:, :]
        assert abs(float(ref_pos[0, 0, 0]) - 10.0) < 1e-4  # time unchanged
        assert abs(float(ref_pos[0, 0, 1]) - 40.0) < 1e-4  # h * 4
        assert abs(float(ref_pos[0, 0, 2]) - 40.0) < 1e-4  # w * 4

    def test_no_positions_no_crash(self):
        """No positions in state or ref should work fine."""
        state = create_initial_state((1, 4, 4), seed=42)
        ref = mx.ones((1, 2, 4))
        cond = VideoConditionByReferenceLatent(reference_latent=ref)
        new_state = cond.apply(state, spatial_dims=(2, 1, 2))
        assert new_state.positions is None

    def test_ref_clean_latent_matches(self):
        """Reference tokens in clean_latent should match reference_latent."""
        state = create_initial_state((1, 4, 4), seed=42)
        ref = mx.ones((1, 3, 4)) * 99.0
        cond = VideoConditionByReferenceLatent(reference_latent=ref)
        new_state = cond.apply(state, spatial_dims=(2, 1, 2))
        appended_clean = new_state.clean_latent[:, 4:, :].astype(mx.float32)
        assert mx.allclose(appended_clean, mx.full((1, 3, 4), 99.0), atol=1e-4).item()
