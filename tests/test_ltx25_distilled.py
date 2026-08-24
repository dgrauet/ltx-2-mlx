"""is_ltx25_pack detection + ancestral sampler constants.

The real-pack tests are gated behind LTX25_Q8_DIR / MODEL_DIR (see
tests/conftest.py) since they require locally converted weight packs.
The synthetic tests exercise the same code path (LTXModelConfig.from_checkpoint_dir)
against tmp_path directories shaped like a 2.5 config, a 2.3 config, and no
config at all, so the contract is pinned even on machines without the real
packs.
"""

import json

import pytest

from ltx_pipelines_mlx.distilled import (
    ANCESTRAL_ETA,
    ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_S_NOISE,
)
from ltx_pipelines_mlx.utils.generation import is_ltx25_pack
from tests.conftest import LTX25_Q8_DIR, MODEL_DIR

skip_no_25_weights = pytest.mark.skipif(LTX25_Q8_DIR is None, reason="ltx-2.5-mlx-q8 pack not found")
skip_no_23_weights = pytest.mark.skipif(MODEL_DIR is None, reason="q8 weights not found")


def test_ancestral_constants_exact_values():
    assert ANCESTRAL_ETA == 1.0
    assert ANCESTRAL_S_NOISE == 1.0
    assert ANCESTRAL_NOISE_SEED_OFFSET == 10000


@pytest.mark.slow
@skip_no_25_weights
def test_is_ltx25_pack_true_on_real_25_pack():
    assert is_ltx25_pack(LTX25_Q8_DIR) is True


@pytest.mark.slow
@skip_no_23_weights
def test_is_ltx25_pack_false_on_real_23_pack():
    assert is_ltx25_pack(MODEL_DIR) is False


def test_is_ltx25_pack_true_on_synthetic_25_config(tmp_path):
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    assert is_ltx25_pack(tmp_path) is True


def test_is_ltx25_pack_false_on_synthetic_23_config(tmp_path):
    (tmp_path / "embedded_config.json").write_text(
        json.dumps({"transformer": {"num_layers": 48, "av_ca_timestep_scale_multiplier": 1000.0}})
    )
    assert is_ltx25_pack(tmp_path) is False


def test_is_ltx25_pack_false_when_no_config_present(tmp_path):
    # LTXModelConfig.from_checkpoint_dir finds neither embedded_config.json
    # nor config.json: it warns on stderr and returns the hardcoded
    # defaults, where ff_bias=True (2.3-shaped) -> is_ltx25_pack is False.
    # Must not raise.
    assert is_ltx25_pack(tmp_path) is False
