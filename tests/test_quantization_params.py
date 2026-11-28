"""Unit tests for quantization parameter derivation and application.

Self-contained (no model weights on disk): builds tiny models, quantizes them
at various (bits, group_size) combinations, and verifies the loader rebuilds
matching QuantizedLinear modules.

Regression coverage for the group_size!=64 bug: the loader previously assumed
group_size=64, which mis-derived bits for any other group size (e.g. an int4/
g32 transformer was read as int2/g64) and failed at load_weights with a shape
mismatch. See ltx_core_mlx.utils.weights._derive_quant_params.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_core_mlx.loader.fuse_loras import apply_loras
from ltx_core_mlx.loader.lora_adapters import attach_loras
from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core_mlx.utils.weights import (
    _derive_quant_params,
    apply_quantization,
    derive_quant_params,
    infer_quant_mode,
)


class TinyModel(nn.Module):
    """Two Linear layers with different in_features to exercise per-shape math."""

    # in_features are divisible by every group size under test (32/64/128).
    def __init__(self, in1: int = 128, out1: int = 64, in2: int = 256, out2: int = 128):
        super().__init__()
        self.a = nn.Linear(in1, out1, bias=False)
        self.b = nn.Linear(in2, out2, bias=False)


def _quantize_model(model: TinyModel, bits: int, group_size: int) -> dict[str, mx.array]:
    """Produce a weights dict (packed weight + scales + biases) for each Linear."""
    weights: dict[str, mx.array] = {}
    for name, lin in (("a", model.a), ("b", model.b)):
        q, scales, biases = mx.quantize(lin.weight, bits=bits, group_size=group_size)
        weights[f"{name}.weight"] = q
        weights[f"{name}.scales"] = scales
        weights[f"{name}.biases"] = biases
    return weights


_PARAMS = [
    pytest.param(4, 32, id="int4-g32"),
    pytest.param(4, 64, id="int4-g64"),
    pytest.param(4, 128, id="int4-g128"),
    pytest.param(5, 32, id="int5-g32"),
    pytest.param(5, 64, id="int5-g64"),
    pytest.param(5, 128, id="int5-g128"),
    pytest.param(6, 32, id="int6-g32"),
    pytest.param(6, 64, id="int6-g64"),
    pytest.param(6, 128, id="int6-g128"),
    pytest.param(8, 32, id="int8-g32"),
    pytest.param(8, 64, id="int8-g64"),
    pytest.param(8, 128, id="int8-g128"),
]


class TestDerivePureFunction:
    """The shared, model-free ``derive_quant_params`` helper."""

    @pytest.mark.parametrize(("bits", "group_size"), _PARAMS)
    def test_recovers_exact_params(self, bits: int, group_size: int) -> None:
        in_features = 128
        packed_cols = in_features * bits // 32
        scales_cols = in_features // group_size
        assert derive_quant_params(in_features, packed_cols, scales_cols) == (bits, group_size)

    def test_rejects_inexact_bit_packing(self) -> None:
        # packed_cols that rounds toward bits=4 but is not an exact 32-bit packing
        # of in_features would previously be coerced by the min/max clamp.
        with pytest.raises(ValueError, match="exact bit-packing"):
            derive_quant_params(in_features=128, packed_cols=17, scales_cols=4)

    def test_rejects_indivisible_group_size(self) -> None:
        # int4/g32 of in_features=128 -> packed_cols=16, but scales_cols=5 does not
        # divide 128 evenly, so no uniform group_size exists.
        with pytest.raises(ValueError, match="not divisible"):
            derive_quant_params(in_features=128, packed_cols=16, scales_cols=5)

    def test_rejects_nonpositive_shapes(self) -> None:
        with pytest.raises(ValueError, match="positive shapes"):
            derive_quant_params(in_features=0, packed_cols=16, scales_cols=4)


class TestDeriveQuantParams:
    @pytest.mark.parametrize(("bits", "group_size"), _PARAMS)
    def test_recovers_exact_params(self, bits: int, group_size: int) -> None:
        source = TinyModel()
        weights = _quantize_model(source, bits=bits, group_size=group_size)
        # Fresh (float) model still carries true in_features, which disambiguates.
        model = TinyModel()
        derived = _derive_quant_params(model, weights, {"a", "b"})
        assert derived == (bits, group_size)

    def test_returns_none_when_no_layer_matches(self) -> None:
        model = TinyModel()
        # scales present but no corresponding model weight key
        weights = {"ghost.scales": mx.ones((4, 2))}
        assert _derive_quant_params(model, weights, {"ghost"}) is None

    def test_skips_already_quantized_model_layer(self) -> None:
        """A model layer that is already QuantizedLinear must be skipped.

        Its ``.weight`` last dim is the packed column count, not in_features, so
        deriving from it would yield garbage. With only such a layer available,
        derivation finds no float reference and returns None.
        """
        model = TinyModel()
        # Quantize the model in place so model.a is a QuantizedLinear.
        weights = _quantize_model(TinyModel(), bits=4, group_size=32)
        apply_quantization(model, weights)
        assert isinstance(model.a, nn.QuantizedLinear)

        assert _derive_quant_params(model, weights, {"a", "b"}) is None


class TestApplyQuantization:
    @pytest.mark.parametrize(("bits", "group_size"), _PARAMS)
    def test_load_weights_strict_roundtrip(self, bits: int, group_size: int) -> None:
        """apply_quantization must build modules that strictly load the weights.

        With the old hardcoded g64 assumption this raised a shape mismatch for
        every non-g64 model (the reported int4/g32 failure).
        """
        source = TinyModel()
        weights = _quantize_model(source, bits=bits, group_size=group_size)

        model = TinyModel()
        apply_quantization(model, weights)
        # Would raise ValueError("Expected shape ... but received ...") on the bug.
        model.load_weights(list(weights.items()), strict=True)

        assert isinstance(model.a, nn.QuantizedLinear)
        assert model.a.bits == bits
        assert model.a.group_size == group_size

    def test_no_scales_is_noop(self) -> None:
        model = TinyModel()
        float_weights = {
            "a.weight": model.a.weight,
            "b.weight": model.b.weight,
        }
        apply_quantization(model, float_weights)
        assert isinstance(model.a, nn.Linear)
        assert not isinstance(model.a, nn.QuantizedLinear)


class MockMixedDiTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(128, 64, bias=False)
        self.to_k = nn.Linear(128, 64, bias=False)
        self.to_v = nn.Linear(256, 64, bias=False)
        self.ff_in = nn.Linear(256, 128, bias=False)
        self.ff_out = nn.Linear(128, 64, bias=False)
        self.proj_out = nn.Linear(64, 64, bias=False)


def _pack_layer(weight: mx.array, *, bits: int, group_size: int, mode: str) -> dict[str, mx.array]:
    out = mx.quantize(weight, bits=bits, group_size=group_size, mode=mode)
    res = {"weight": out[0], "scales": out[1]}
    if len(out) == 3:
        res["biases"] = out[2]
    return res


class TestMixedPrecisionAndModes:
    @pytest.mark.parametrize(
        ("bits", "group_size", "has_biases", "scales_dtype", "expected"),
        [
            (6, 64, True, mx.bfloat16, "affine"),
            (4, 32, False, mx.uint8, "mxfp4"),
            (8, 32, False, mx.uint8, "mxfp8"),
            (4, 16, False, mx.uint8, "nvfp4"),
        ],
    )
    def test_infer_quant_mode(
        self,
        bits: int,
        group_size: int,
        has_biases: bool,
        scales_dtype: mx.Dtype,
        expected: str,
    ) -> None:
        assert infer_quant_mode(bits, group_size, has_biases=has_biases, scales_dtype=scales_dtype) == expected

    def test_infer_quant_mode_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="cannot infer quantization mode"):
            infer_quant_mode(6, 64, has_biases=False, scales_dtype=mx.uint8)

    def test_mixed_precision_block_loading_and_forward_parity(self) -> None:
        mx.random.seed(0)
        src = MockMixedDiTBlock()
        plan = {
            "to_q": (6, 64, "affine"),
            "to_k": (8, 64, "affine"),
            "to_v": (4, 64, "affine"),
            "ff_in": (4, 32, "mxfp4"),
            "ff_out": (8, 32, "mxfp8"),
        }
        weights: dict[str, mx.array] = {}
        for name, (bits, gs, mode) in plan.items():
            packed = _pack_layer(getattr(src, name).weight.astype(mx.bfloat16), bits=bits, group_size=gs, mode=mode)
            for suffix, arr in packed.items():
                weights[f"{name}.{suffix}"] = arr
        weights["proj_out.weight"] = src.proj_out.weight.astype(mx.bfloat16)

        dst = MockMixedDiTBlock()
        apply_quantization(dst, weights)
        dst.load_weights(list(weights.items()), strict=True)

        for name, (bits, gs, mode) in plan.items():
            layer = getattr(dst, name)
            assert isinstance(layer, nn.QuantizedLinear)
            assert layer.bits == bits
            assert layer.group_size == gs
            assert layer.mode == mode

        assert isinstance(dst.proj_out, nn.Linear)
        assert not isinstance(dst.proj_out, nn.QuantizedLinear)

        for name, (bits, gs, mode) in plan.items():
            layer = getattr(dst, name)
            in_dim = getattr(src, name).weight.shape[1]
            x = mx.random.normal((2, in_dim)).astype(mx.bfloat16)
            y = layer(x)
            ref = mx.quantized_matmul(
                x,
                layer.weight,
                scales=layer.scales,
                biases=getattr(layer, "biases", None),
                transpose=True,
                group_size=gs,
                bits=bits,
                mode=mode,
            )
            mx.eval(y, ref)
            assert mx.allclose(y, ref, atol=1e-4).item()

    @pytest.mark.parametrize(
        ("bits", "group_size", "mode"),
        [
            (6, 64, "affine"),
            (4, 32, "mxfp4"),
            (8, 32, "mxfp8"),
        ],
    )
    def test_lora_fusion_and_unfused_adapter(self, bits: int, group_size: int, mode: str) -> None:
        mx.random.seed(7)
        src = TinyModel()
        base_sd: dict[str, mx.array] = {}
        for name, lin in (("a", src.a), ("b", src.b)):
            packed = _pack_layer(lin.weight.astype(mx.bfloat16), bits=bits, group_size=group_size, mode=mode)
            for suffix, arr in packed.items():
                base_sd[f"{name}.{suffix}"] = arr

        rank = 8
        lora_a = mx.random.normal((rank, 128)).astype(mx.bfloat16) * 0.05
        lora_b = mx.random.normal((64, rank)).astype(mx.bfloat16) * 0.05
        lora_sd = StateDict(sd={"a.lora_A.weight": lora_a, "a.lora_B.weight": lora_b}, size=0, dtype={mx.bfloat16})

        fused = apply_loras(
            StateDict(sd=base_sd, size=0, dtype={mx.bfloat16}),
            [LoraStateDictWithStrength(state_dict=lora_sd, strength=0.75)],
        )
        fused_model = TinyModel()
        apply_quantization(fused_model, fused.sd)
        fused_model.load_weights(list(fused.sd.items()), strict=True)
        assert fused_model.a.bits == bits
        assert fused_model.a.group_size == group_size
        assert fused_model.a.mode == mode

        unfused_model = TinyModel()
        apply_quantization(unfused_model, base_sd)
        unfused_model.load_weights(list(base_sd.items()), strict=True)
        handle = attach_loras(unfused_model, [LoraStateDictWithStrength(state_dict=lora_sd, strength=0.75)])

        x = mx.random.normal((2, 128)).astype(mx.bfloat16)
        y_fused = fused_model.a(x).astype(mx.float32)
        y_unfused = unfused_model.a(x).astype(mx.float32)
        mx.eval(y_fused, y_unfused)
        cos = (mx.sum(y_fused * y_unfused) / (mx.linalg.norm(y_fused) * mx.linalg.norm(y_unfused))).item()
        assert cos > 0.98
        handle.detach()

    def test_block_streamer_reconciles_mixed_precision_blocks(self) -> None:
        from ltx_core_mlx.loader.block_streaming import BlockStreamer

        mx.random.seed(99)
        src0 = TinyModel()
        src1 = TinyModel()

        w0: dict[str, mx.array] = {}
        for name, lin, (bits, gs, mode) in (
            ("a", src0.a, (6, 64, "affine")),
            ("b", src0.b, (8, 64, "affine")),
        ):
            for suffix, arr in _pack_layer(lin.weight.astype(mx.bfloat16), bits=bits, group_size=gs, mode=mode).items():
                w0[f"{name}.{suffix}"] = arr

        w1: dict[str, mx.array] = {}
        for name, lin, (bits, gs, mode) in (
            ("a", src1.a, (4, 32, "mxfp4")),
            ("b", src1.b, (4, 64, "affine")),
        ):
            for suffix, arr in _pack_layer(lin.weight.astype(mx.bfloat16), bits=bits, group_size=gs, mode=mode).items():
                w1[f"{name}.{suffix}"] = arr

        blk = TinyModel()
        apply_quantization(blk, w0)
        blk.load_weights(list(w0.items()), strict=True)
        assert (blk.a.bits, blk.a.group_size, blk.a.mode) == (6, 64, "affine")

        BlockStreamer._reconcile_quant_layers(blk, list(w1.items()))
        blk.load_weights(list(w1.items()), strict=True)
        assert (blk.a.bits, blk.a.group_size, blk.a.mode) == (4, 32, "mxfp4")
        assert (blk.b.bits, blk.b.group_size, blk.b.mode) == (4, 64, "affine")

        x = mx.random.normal((2, 128)).astype(mx.bfloat16)
        y = blk.a(x)
        mx.eval(y)
        assert y.shape == (2, 64)

        BlockStreamer._reconcile_quant_layers(blk, list(w0.items()))
        blk.load_weights(list(w0.items()), strict=True)
        assert (blk.a.bits, blk.a.group_size, blk.a.mode) == (6, 64, "affine")
        assert (blk.b.bits, blk.b.group_size, blk.b.mode) == (8, 64, "affine")
