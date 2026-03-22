"""Tests for the loader module: sd_ops, primitives, and fuse_loras."""

import mlx.core as mx

from ltx_core_mlx.loader.fuse_loras import _fuse_delta_with_float, _prepare_deltas, apply_loras
from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core_mlx.loader.sd_ops import (
    LTXV_LORA_COMFY_RENAMING_MAP,
    ContentMatching,
    ContentReplacement,
    SDOps,
)


# ---------------------------------------------------------------------------
# SDOps
# ---------------------------------------------------------------------------
class TestSDOps:
    def test_with_replacement(self):
        ops = SDOps("test").with_matching().with_replacement("old", "new")
        assert ops.apply_to_key("old.weight") == "new.weight"

    def test_with_matching_prefix(self):
        ops = SDOps("test").with_matching(prefix="model.")
        assert ops.apply_to_key("model.layer.weight") == "model.layer.weight"
        assert ops.apply_to_key("other.layer.weight") is None

    def test_with_matching_suffix(self):
        ops = SDOps("test").with_matching(suffix=".weight")
        assert ops.apply_to_key("layer.weight") == "layer.weight"
        assert ops.apply_to_key("layer.bias") is None

    def test_chained_replacements(self):
        ops = (
            SDOps("test")
            .with_matching()
            .with_replacement("diffusion_model.", "")
            .with_replacement(".lora_A.weight", ".weight")
        )
        result = ops.apply_to_key("diffusion_model.layer.lora_A.weight")
        assert result == "layer.weight"

    def test_no_match_returns_none(self):
        ops = SDOps("test").with_matching(prefix="prefix.")
        assert ops.apply_to_key("no_match") is None

    def test_comfy_renaming_map(self):
        result = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key("diffusion_model.blocks.0.weight")
        assert result == "blocks.0.weight"

    def test_empty_mapping(self):
        ops = SDOps("empty")
        assert ops.apply_to_key("any.key") is None

    def test_apply_to_key_value(self):
        ops = SDOps("test").with_matching()
        results = ops.apply_to_key_value("key", mx.array([1.0]))
        assert len(results) == 1
        assert results[0].new_key == "key"

    def test_immutability(self):
        ops1 = SDOps("test").with_matching()
        ops2 = ops1.with_replacement("a", "b")
        assert len(ops1.mapping) == 1
        assert len(ops2.mapping) == 2


# ---------------------------------------------------------------------------
# ContentReplacement / ContentMatching
# ---------------------------------------------------------------------------
class TestContentTypes:
    def test_content_replacement_frozen(self):
        cr = ContentReplacement("old", "new")
        assert cr.content == "old"
        assert cr.replacement == "new"

    def test_content_matching_defaults(self):
        cm = ContentMatching()
        assert cm.prefix == ""
        assert cm.suffix == ""

    def test_content_matching_with_values(self):
        cm = ContentMatching(prefix="model.", suffix=".weight")
        assert cm.prefix == "model."
        assert cm.suffix == ".weight"


# ---------------------------------------------------------------------------
# StateDict / Primitives
# ---------------------------------------------------------------------------
class TestStateDict:
    def test_creation(self):
        sd = StateDict(
            sd={"w": mx.ones((4, 4))},
            size=64,
            dtype={mx.float32},
        )
        assert "w" in sd.sd
        assert sd.size == 64
        assert sd.footprint() == 64

    def test_empty(self):
        sd = StateDict(sd={}, size=0, dtype=set())
        assert len(sd.sd) == 0


class TestLoraTypes:
    def test_lora_path_strength(self):
        from ltx_core_mlx.loader.primitives import LoraPathStrengthAndSDOps

        ops = SDOps("test").with_matching()
        lora = LoraPathStrengthAndSDOps("path/to/lora.safetensors", 0.8, ops)
        assert lora.path == "path/to/lora.safetensors"
        assert lora.strength == 0.8

    def test_lora_state_dict_with_strength(self):
        sd = StateDict(sd={"w": mx.ones((2, 2))}, size=16, dtype={mx.float32})
        lsd = LoraStateDictWithStrength(sd, 0.5)
        assert lsd.strength == 0.5
        assert "w" in lsd.state_dict.sd


# ---------------------------------------------------------------------------
# LoRA Fusion
# ---------------------------------------------------------------------------
class TestPrepareDelta:
    def test_no_matching_lora(self):
        sd = StateDict(sd={"other.lora_A.weight": mx.ones((2, 4))}, size=0, dtype=set())
        result = _prepare_deltas([LoraStateDictWithStrength(sd, 1.0)], "layer.weight")
        assert result is None

    def test_single_lora_delta(self):
        a = mx.ones((2, 4))  # rank 2, in_features 4
        b = mx.ones((8, 2))  # out_features 8, rank 2
        sd = StateDict(
            sd={"layer.lora_A.weight": a, "layer.lora_B.weight": b},
            size=0,
            dtype=set(),
        )
        result = _prepare_deltas([LoraStateDictWithStrength(sd, 1.0)], "layer.weight")
        assert result is not None
        assert result.shape == (8, 4)  # B @ A = (8, 2) @ (2, 4) = (8, 4)

    def test_strength_scaling(self):
        a = mx.ones((1, 4))
        b = mx.ones((4, 1))
        sd = StateDict(sd={"l.lora_A.weight": a, "l.lora_B.weight": b}, size=0, dtype=set())
        result_1 = _prepare_deltas([LoraStateDictWithStrength(sd, 1.0)], "l.weight")
        result_half = _prepare_deltas([LoraStateDictWithStrength(sd, 0.5)], "l.weight")
        # strength=0.5 should give half the delta
        assert mx.allclose(result_half, result_1 * 0.5).item()

    def test_multiple_loras_sum(self):
        a = mx.ones((1, 4))
        b = mx.ones((4, 1))
        sd1 = StateDict(sd={"l.lora_A.weight": a, "l.lora_B.weight": b}, size=0, dtype=set())
        sd2 = StateDict(sd={"l.lora_A.weight": a * 2, "l.lora_B.weight": b}, size=0, dtype=set())
        result = _prepare_deltas(
            [LoraStateDictWithStrength(sd1, 1.0), LoraStateDictWithStrength(sd2, 1.0)],
            "l.weight",
        )
        # Should be sum of both deltas
        single1 = _prepare_deltas([LoraStateDictWithStrength(sd1, 1.0)], "l.weight")
        single2 = _prepare_deltas([LoraStateDictWithStrength(sd2, 1.0)], "l.weight")
        assert mx.allclose(result, single1 + single2).item()


class TestFuseDeltaWithFloat:
    def test_basic_fusion(self):
        weight = mx.ones((4, 4)) * 2.0
        delta = mx.ones((4, 4)) * 0.5
        result = _fuse_delta_with_float(delta, weight, "w", mx.bfloat16)
        expected = 2.0 + 0.5
        assert result["w"].dtype == mx.bfloat16
        assert mx.allclose(result["w"].astype(mx.float32), mx.full((4, 4), expected)).item()


class TestApplyLoras:
    def test_no_loras_copies_weights(self):
        model_sd = StateDict(
            sd={"layer.weight": mx.ones((4, 4))},
            size=64,
            dtype={mx.float32},
        )
        result = apply_loras(model_sd, [])
        assert "layer.weight" in result.sd
        assert mx.array_equal(result.sd["layer.weight"], model_sd.sd["layer.weight"]).item()

    def test_single_lora_fusion(self):
        weight = mx.zeros((4, 4))
        model_sd = StateDict(sd={"layer.weight": weight}, size=64, dtype={mx.float32})

        a = mx.eye(4)  # identity rank-4 LoRA
        b = mx.eye(4)
        lora_sd = StateDict(
            sd={"layer.lora_A.weight": a, "layer.lora_B.weight": b},
            size=0,
            dtype=set(),
        )

        result = apply_loras(model_sd, [LoraStateDictWithStrength(lora_sd, 1.0)])
        # zero weight + identity delta = identity
        assert mx.allclose(result.sd["layer.weight"], mx.eye(4)).item()

    def test_skips_scales_biases_keys(self):
        model_sd = StateDict(
            sd={
                "layer.weight": mx.ones((4, 4)),
                "layer.scales": mx.ones((4,)),
                "layer.biases": mx.zeros((4,)),
            },
            size=0,
            dtype=set(),
        )
        result = apply_loras(model_sd, [])
        # scales and biases should not appear as standalone entries
        # (they're handled with their weight key)
        assert "layer.weight" in result.sd
