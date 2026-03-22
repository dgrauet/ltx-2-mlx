"""Tests for the STG perturbation system."""

import mlx.core as mx

from ltx_core_mlx.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)


# ---------------------------------------------------------------------------
# PerturbationType
# ---------------------------------------------------------------------------
class TestPerturbationType:
    def test_enum_values(self):
        assert PerturbationType.SKIP_A2V_CROSS_ATTN.value == "skip_a2v_cross_attn"
        assert PerturbationType.SKIP_V2A_CROSS_ATTN.value == "skip_v2a_cross_attn"
        assert PerturbationType.SKIP_VIDEO_SELF_ATTN.value == "skip_video_self_attn"
        assert PerturbationType.SKIP_AUDIO_SELF_ATTN.value == "skip_audio_self_attn"

    def test_has_four_members(self):
        assert len(PerturbationType) == 4

    def test_from_value(self):
        assert PerturbationType("skip_video_self_attn") == PerturbationType.SKIP_VIDEO_SELF_ATTN


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------
class TestPerturbation:
    def test_is_perturbed_matching_type_all_blocks(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        assert p.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is True
        assert p.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=47) is True

    def test_is_perturbed_wrong_type(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        assert p.is_perturbed(PerturbationType.SKIP_AUDIO_SELF_ATTN, block=0) is False

    def test_is_perturbed_specific_blocks(self):
        p = Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=[0, 5, 10])
        assert p.is_perturbed(PerturbationType.SKIP_A2V_CROSS_ATTN, block=0) is True
        assert p.is_perturbed(PerturbationType.SKIP_A2V_CROSS_ATTN, block=5) is True
        assert p.is_perturbed(PerturbationType.SKIP_A2V_CROSS_ATTN, block=3) is False

    def test_is_perturbed_empty_blocks(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=[])
        assert p.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is False

    def test_frozen_dataclass(self):
        """Perturbation should be immutable."""
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        try:
            p.type = PerturbationType.SKIP_AUDIO_SELF_ATTN
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# PerturbationConfig
# ---------------------------------------------------------------------------
class TestPerturbationConfig:
    def test_is_perturbed_with_matching(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        config = PerturbationConfig(perturbations=[p])
        assert config.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is True

    def test_is_perturbed_no_matching(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        config = PerturbationConfig(perturbations=[p])
        assert config.is_perturbed(PerturbationType.SKIP_AUDIO_SELF_ATTN, block=0) is False

    def test_is_perturbed_none_perturbations(self):
        config = PerturbationConfig(perturbations=None)
        assert config.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is False

    def test_is_perturbed_empty_perturbations(self):
        config = PerturbationConfig(perturbations=[])
        assert config.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is False

    def test_empty_factory(self):
        config = PerturbationConfig.empty()
        assert config.perturbations == []
        assert config.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is False

    def test_multiple_perturbations(self):
        p1 = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=[0, 1])
        p2 = Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=None)
        config = PerturbationConfig(perturbations=[p1, p2])
        assert config.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is True
        assert config.is_perturbed(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=5) is False
        assert config.is_perturbed(PerturbationType.SKIP_AUDIO_SELF_ATTN, block=99) is True
        assert config.is_perturbed(PerturbationType.SKIP_A2V_CROSS_ATTN, block=0) is False

    def test_frozen_dataclass(self):
        config = PerturbationConfig.empty()
        try:
            config.perturbations = None
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# BatchedPerturbationConfig
# ---------------------------------------------------------------------------
class TestBatchedPerturbationConfig:
    def test_mask_no_perturbation(self):
        batch = BatchedPerturbationConfig.empty(batch_size=3)
        mask = batch.mask(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0)
        assert mask.shape == (3,)
        assert mask.dtype == mx.bfloat16
        # All unperturbed -> all 1.0
        assert float(mx.min(mask)) == 1.0

    def test_mask_with_perturbation(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        configs = [
            PerturbationConfig(perturbations=[p]),
            PerturbationConfig.empty(),
            PerturbationConfig(perturbations=[p]),
        ]
        batch = BatchedPerturbationConfig(perturbations=configs)
        mask = batch.mask(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0)
        assert mask.shape == (3,)
        expected = mx.array([0.0, 1.0, 0.0], dtype=mx.bfloat16)
        assert mx.allclose(mask, expected, atol=1e-3).item()

    def test_mask_different_type(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        configs = [PerturbationConfig(perturbations=[p])]
        batch = BatchedPerturbationConfig(perturbations=configs)
        mask = batch.mask(PerturbationType.SKIP_AUDIO_SELF_ATTN, block=0)
        assert float(mask[0]) == 1.0  # Not perturbed for audio

    def test_mask_like_shape(self):
        batch = BatchedPerturbationConfig.empty(batch_size=2)
        values = mx.zeros((2, 10, 4, 8))
        mask = batch.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0, values=values)
        assert mask.shape == (2, 1, 1, 1)

    def test_mask_like_broadcastable(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        configs = [
            PerturbationConfig(perturbations=[p]),
            PerturbationConfig.empty(),
        ]
        batch = BatchedPerturbationConfig(perturbations=configs)
        values = mx.ones((2, 4, 8))
        mask = batch.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0, values=values)
        result = values * mask
        # Batch 0 perturbed -> 0, batch 1 unperturbed -> 1
        assert float(mx.max(mx.abs(result[0]))) == 0.0
        assert float(mx.min(result[1])) == 1.0

    def test_any_in_batch(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        configs = [
            PerturbationConfig.empty(),
            PerturbationConfig(perturbations=[p]),
        ]
        batch = BatchedPerturbationConfig(perturbations=configs)
        assert batch.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is True
        assert batch.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, block=0) is False

    def test_all_in_batch(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        configs = [
            PerturbationConfig(perturbations=[p]),
            PerturbationConfig(perturbations=[p]),
        ]
        batch = BatchedPerturbationConfig(perturbations=configs)
        assert batch.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is True

    def test_all_in_batch_false(self):
        p = Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None)
        configs = [
            PerturbationConfig(perturbations=[p]),
            PerturbationConfig.empty(),
        ]
        batch = BatchedPerturbationConfig(perturbations=configs)
        assert batch.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0) is False

    def test_empty_factory(self):
        batch = BatchedPerturbationConfig.empty(batch_size=4)
        assert len(batch.perturbations) == 4
        for cfg in batch.perturbations:
            assert cfg.perturbations == []

    def test_block_specific_perturbation(self):
        """Only specific blocks should be perturbed."""
        p = Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=[5, 10, 15])
        configs = [PerturbationConfig(perturbations=[p])]
        batch = BatchedPerturbationConfig(perturbations=configs)
        # Block 5: perturbed
        assert float(batch.mask(PerturbationType.SKIP_V2A_CROSS_ATTN, block=5)[0]) == 0.0
        # Block 6: not perturbed
        assert float(batch.mask(PerturbationType.SKIP_V2A_CROSS_ATTN, block=6)[0]) == 1.0

    def test_single_batch(self):
        batch = BatchedPerturbationConfig.empty(batch_size=1)
        mask = batch.mask(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0)
        assert mask.shape == (1,)

    def test_mask_like_2d(self):
        """mask_like with 2D values should give (B, 1) shape."""
        batch = BatchedPerturbationConfig.empty(batch_size=3)
        values = mx.zeros((3, 10))
        mask = batch.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, block=0, values=values)
        assert mask.shape == (3, 1)
