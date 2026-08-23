"""Contrat de chargement du DiT 2.5 : pack réel <-> arbre de paramètres.

`mx.load` mmap-e sans matérialiser, donc comparer les clés est peu coûteux
même sur 20 GB. Ce test attrape la classe silent-no-op (#52) : un poids
présent que personne ne lit, ou un paramètre que le pack ne sert pas.
"""

import json
import struct

import pytest

from tests.conftest import LTX25_Q8_DIR

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found"),
]


def _pack_keys() -> set[str]:
    path = LTX25_Q8_DIR / "transformer-dev.safetensors"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {k for k in header if k != "__metadata__"}


def test_25_config_covers_every_pack_tensor():
    from mlx.utils import tree_flatten

    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

    config = LTXModelConfig.from_checkpoint_dir(LTX25_Q8_DIR)
    assert config.ff_bias is False, "embedded_config.json not read"
    model = LTXModel(config)

    model_keys = {f"transformer.{k}" for k, _ in tree_flatten(model.parameters())}
    pack = _pack_keys()

    # Quantization scales/biases du q8 correspondent aux .weight du modèle.
    pack_normalized = {
        k.removesuffix(".scales").removesuffix(".biases") + ".weight" if k.endswith((".scales", ".biases")) else k
        for k in pack
    }
    missing_in_model = sorted(pack_normalized - model_keys)
    unfed_params = sorted(model_keys - pack_normalized)
    assert not missing_in_model, f"pack tensors nobody would load: {missing_in_model[:10]}"
    assert not unfed_params, f"model params the pack does not feed: {unfed_params[:10]}"
