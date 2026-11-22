"""Weight loading utilities for pre-converted MLX safetensors."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_split_safetensors(
    path: str | Path,
    prefix: str | None = None,
) -> dict[str, mx.array]:
    """Load weights from a safetensors file, optionally stripping a prefix.

    Uses mx.load() which natively handles bfloat16 and all MLX dtypes.

    Args:
        path: Path to the .safetensors file.
        prefix: If provided, only keys starting with this prefix are loaded
            and the prefix is stripped from the key names.

    Returns:
        Dict mapping parameter names to MLX arrays.
    """
    path = Path(path)
    raw = mx.load(str(path))

    if not prefix:
        return raw

    weights: dict[str, mx.array] = {}
    for key, tensor in raw.items():
        if key.startswith(prefix):
            weights[key[len(prefix) :]] = tensor

    return weights


def _detect_quantization_bits(
    weights: dict[str, mx.array],
    group_size: int = 64,
) -> int:
    """Auto-detect quantization bit width from weight shapes.

    For a quantized Linear(I, O):
      - scales shape: (O, I / group_size)
      - weight shape: (O, I * bits / 32)
    So bits = weight_cols * 32 / (scales_cols * group_size).

    Args:
        weights: Weight dict containing .weight and .scales keys.
        group_size: Quantization group size.

    Returns:
        Detected bit width (typically 4 or 8).
    """
    for key in weights:
        if key.endswith(".scales"):
            weight_key = key.rsplit(".scales", 1)[0] + ".weight"
            if weight_key in weights:
                weight_cols = weights[weight_key].shape[-1]
                scales_cols = weights[key].shape[-1]
                bits = round(weight_cols * 32 / (scales_cols * group_size))
                return bits
    return 8  # default fallback


def derive_quant_params(
    in_features: int,
    packed_cols: int,
    scales_cols: int,
) -> tuple[int, int]:
    """Derive ``(bits, group_size)`` from a quantized layer's true in_features.

    Given the layer's real ``in_features`` (from the float weight or the LoRA
    delta, both of which carry the unpacked shape) and the saved packed weight /
    scales column counts:

        packed_cols = in_features * bits / 32   -> bits = 32 * packed_cols / in_features
        scales_cols = in_features / group_size  -> group_size = in_features / scales_cols

    The result is validated for *exact* consistency: a stale or mis-shaped tensor
    whose ratio merely rounds to a plausible ``bits`` is rejected here instead of
    silently misinterpreting the weights downstream. Single home for the
    derivation shared by load-time quantization and LoRA fusion.

    Raises:
        ValueError: if the three counts are not exactly consistent with a valid
            MLX (bits, group_size) split.
    """
    if in_features <= 0 or packed_cols <= 0 or scales_cols <= 0:
        raise ValueError(
            "quant param derivation needs positive shapes, got "
            f"in_features={in_features}, packed_cols={packed_cols}, scales_cols={scales_cols}"
        )
    bits = round(32 * packed_cols / in_features)
    if not 2 <= bits <= 8 or packed_cols * 32 != in_features * bits:
        raise ValueError(
            f"packed weight cols ({packed_cols}) are not an exact bit-packing of "
            f"in_features={in_features}: 32*{packed_cols} != {in_features}*bits for any "
            f"bits in [2, 8] (nearest bits={bits})"
        )
    if in_features % scales_cols != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by scales cols ({scales_cols}); "
            "cannot derive a uniform group_size"
        )
    group_size = in_features // scales_cols
    return bits, group_size


def infer_quant_mode(
    bits: int,
    group_size: int,
    *,
    has_biases: bool,
    scales_dtype: mx.Dtype | None = None,
) -> str:
    if has_biases:
        return "affine"
    if scales_dtype == mx.uint8:
        if bits == 4 and group_size == 32:
            return "mxfp4"
        if bits == 8 and group_size == 32:
            return "mxfp8"
        if bits == 4 and group_size == 16:
            return "nvfp4"
    raise ValueError(
        f"cannot infer quantization mode for bits={bits}, group_size={group_size}, "
        f"has_biases={has_biases}, scales_dtype={scales_dtype}"
    )


def validate_config_matches_weights(transformer_path: Path, config) -> None:
    """Refuse a 2.5-shaped checkpoint driven by a 2.3 config.

    A pack whose ``ff.proj_in`` has no bias was exported from LTX-2.5
    (``ff_bias=false``). If the config in hand still says ``ff_bias=True``,
    the checkpoint directory is missing its ``embedded_config.json`` and
    ``from_checkpoint_dir`` fell back to 2.3 defaults — the silent-wrong
    class of #37. Fail here, with the cause, instead of deep inside
    ``load_weights``.

    Reads only the safetensors header (a few KB), never the tensors.
    """
    import json
    import struct

    with open(transformer_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        keys = json.loads(f.read(header_len)).keys()

    probe = "transformer.transformer_blocks.0.ff.proj_in"
    if f"{probe}.weight" not in keys and f"{probe}.scales" not in keys:
        return  # not a transformer file we understand; loading will complain
    has_bias = f"{probe}.bias" in keys
    if not has_bias and config.ff_bias:
        raise ValueError(
            f"{transformer_path.name} is a 2.5-shaped checkpoint (no ff biases) "
            "but the config in use says ff_bias=True — embedded_config.json is "
            "missing or unreadable in the checkpoint directory. Refusing to "
            "load with 2.3 defaults (see issue #37 for what silent config "
            "fallbacks cost)."
        )


def _derive_layer_quant_params(
    model: nn.Module,
    weights: dict[str, mx.array],
    quantized_layers: set[str],
) -> dict[str, tuple[int, int, str]]:
    from mlx.utils import tree_flatten

    model_params = dict(tree_flatten(model.parameters()))
    per_layer: dict[str, tuple[int, int, str]] = {}
    for layer in quantized_layers:
        wkey = f"{layer}.weight"
        skey = f"{layer}.scales"
        if wkey not in weights or skey not in weights or wkey not in model_params:
            continue
        if skey in model_params:
            continue
        in_features = int(model_params[wkey].shape[-1])
        packed_cols = int(weights[wkey].shape[-1])
        scales_cols = int(weights[skey].shape[-1])
        bits, group_size = derive_quant_params(in_features, packed_cols, scales_cols)
        mode = infer_quant_mode(
            bits,
            group_size,
            has_biases=f"{layer}.biases" in weights,
            scales_dtype=weights[skey].dtype,
        )
        per_layer[layer] = (bits, group_size, mode)
    return per_layer


def _derive_quant_params(
    model: nn.Module,
    weights: dict[str, mx.array],
    quantized_layers: set[str],
) -> tuple[int, int] | None:
    """Derive (bits, group_size) from a representative quantized layer."""
    per_layer = _derive_layer_quant_params(model, weights, quantized_layers)
    if not per_layer:
        return None
    bits, group_size, _mode = next(iter(per_layer.values()))
    return bits, group_size


def apply_quantization(
    model: nn.Module,
    weights: dict[str, mx.array],
    group_size: int = 64,
    bits: int | None = None,
) -> None:
    """Apply quantization to Linear layers that have quantized weights.

    Detects quantized layers by checking for ``.scales`` keys in ``weights``
    and converts matching ``nn.Linear`` modules to ``nn.QuantizedLinear``
    using per-layer ``(bits, group_size, mode)`` derived from each layer's
    unpacked ``in_features`` and saved tensor metadata. Supports uniform and
    mixed-precision checkpoints across affine (2..8-bit), MXFP4, MXFP8, and
    NVFP4 formats.
    """
    quantized_layers: set[str] = set()

    for key in weights:
        if key.endswith(".scales"):
            layer_name = key.rsplit(".scales", 1)[0]
            quantized_layers.add(layer_name)

    if not quantized_layers:
        return

    per_layer = _derive_layer_quant_params(model, weights, quantized_layers)
    if per_layer:
        def _layer_predicate(path: str, module: nn.Module) -> bool | dict:
            if not isinstance(module, nn.Linear):
                return False
            params = per_layer.get(path)
            if params is None:
                return False
            layer_bits, layer_group_size, layer_mode = params
            return {
                "bits": bits if bits is not None else layer_bits,
                "group_size": layer_group_size,
                "mode": layer_mode,
            }

        nn.quantize(model, class_predicate=_layer_predicate)
        return

    if bits is None:
        bits = _detect_quantization_bits(weights, group_size)

    def _should_quantize(path: str, _module: nn.Module) -> bool:
        return path in quantized_layers and isinstance(_module, nn.Linear)

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_should_quantize)


def remap_audio_vae_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap underscore-prefixed per-channel stats keys for audio VAE.

    MLX treats ``_``-prefixed attributes as private, so safetensors keys
    ``_mean_of_means`` / ``_std_of_means`` must be loaded as
    ``mean_of_means`` / ``std_of_means``.
    """
    return {
        k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
        for k, v in weights.items()
    }
