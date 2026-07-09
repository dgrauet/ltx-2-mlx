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

from ltx_core_mlx.utils.weights import (
    _derive_quant_params,
    apply_quantization,
    derive_quant_params,
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


# group sizes and bit widths that must all round-trip, not just the old g64 default
_PARAMS = [
    pytest.param(4, 32, id="int4-g32"),
    pytest.param(8, 64, id="int8-g64"),
    pytest.param(4, 64, id="int4-g64"),
    pytest.param(8, 32, id="int8-g32"),
    pytest.param(4, 128, id="int4-g128"),
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
