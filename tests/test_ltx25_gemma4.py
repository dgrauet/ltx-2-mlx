"""Tests for Gemma4TextConfig parsing (LTX-2.5 text tower config).

Covers a truncated-but-representative literal dict mirroring the shape of
the real ``text_encoder_config.json`` pack file, plus a slow test that
parses the real pack when available (``LTX25_Q8_DIR``).
"""

import pytest

from tests.conftest import LTX25_Q8_DIR

# Truncated literal mirroring the real pack's text_encoder_config.json
# shape (48 layers -> collapsed to a smaller layer_types list here, since
# Gemma4TextConfig must work off whatever layer_types length the config
# provides -- num_hidden_layers is read separately).
_FULL_CONFIG = {
    "gemma_version": "gemma4-12b-ltx-v1",
    "model_type": "gemma4_unified",
    "text_config": {
        "model_type": "gemma4_unified_text",
        "attention_k_eq_v": True,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_attention_heads": 16,
        "num_global_key_value_heads": 1,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "num_kv_shared_layers": 0,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "sliding_window": 1024,
        "use_bidirectional_attention": "vision",
        "vocab_size": 262144,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        },
    },
}

_FULL_ATTENTION_INDICES = [5, 11, 17, 23, 29, 35, 41, 47]


def _config_with(**text_config_overrides):
    import copy

    cfg = copy.deepcopy(_FULL_CONFIG)
    cfg["text_config"].update(text_config_overrides)
    return cfg


def test_parses_basic_fields():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    config = Gemma4TextConfig.from_text_encoder_config(_FULL_CONFIG)

    assert config.gemma_version == "gemma4-12b-ltx-v1"
    assert config.hidden_size == 3840
    assert config.num_hidden_layers == 48
    assert config.num_attention_heads == 16
    assert config.head_dim == 256
    assert config.global_head_dim == 512
    assert config.num_key_value_heads == 8
    assert config.num_global_key_value_heads == 1
    assert config.intermediate_size == 15360
    assert config.vocab_size == 262144
    assert config.rms_norm_eps == 1e-06
    assert config.sliding_window == 1024
    assert config.attention_k_eq_v is True
    assert config.pad_token_id == 0
    assert len(config.layer_types) == 48
    assert isinstance(config.layer_types, tuple)


def test_parses_rope_params_per_attention_type():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    config = Gemma4TextConfig.from_text_encoder_config(_FULL_CONFIG)

    assert config.rope_local_theta == 10000.0
    assert config.rope_global_theta == 1000000.0
    assert config.partial_rotary_factor == 0.25


def test_full_attention_layer_indices():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    config = Gemma4TextConfig.from_text_encoder_config(_FULL_CONFIG)

    full_indices = [i for i in range(config.num_hidden_layers) if not config.layer_is_sliding(i)]
    assert full_indices == _FULL_ATTENTION_INDICES


def test_layer_is_sliding():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    config = Gemma4TextConfig.from_text_encoder_config(_FULL_CONFIG)

    assert config.layer_is_sliding(0) is True
    assert config.layer_is_sliding(5) is False


def test_layer_head_dim_by_type():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    config = Gemma4TextConfig.from_text_encoder_config(_FULL_CONFIG)

    assert config.layer_head_dim(0) == 256  # sliding
    assert config.layer_head_dim(5) == 512  # full


def test_layer_num_kv_by_type():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    config = Gemma4TextConfig.from_text_encoder_config(_FULL_CONFIG)

    assert config.layer_num_kv(0) == 8  # sliding
    assert config.layer_num_kv(5) == 1  # full


def test_rejects_num_kv_shared_layers_nonzero():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    bad = _config_with(num_kv_shared_layers=1)
    with pytest.raises(ValueError, match="num_kv_shared_layers"):
        Gemma4TextConfig.from_text_encoder_config(bad)


def test_rejects_bidirectional_attention_all():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    bad = _config_with(use_bidirectional_attention="all")
    with pytest.raises(ValueError, match="use_bidirectional_attention"):
        Gemma4TextConfig.from_text_encoder_config(bad)


def test_rejects_wrong_model_type():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    bad = _config_with(model_type="gemma3_text")
    with pytest.raises(ValueError, match="model_type"):
        Gemma4TextConfig.from_text_encoder_config(bad)


def test_defaults_when_fields_absent():
    """use_bidirectional_attention, attention_k_eq_v, num_kv_shared_layers
    default per the HF configuration_gemma4_unified.py reference."""
    import copy

    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    cfg = copy.deepcopy(_FULL_CONFIG)
    del cfg["text_config"]["use_bidirectional_attention"]
    del cfg["text_config"]["attention_k_eq_v"]
    del cfg["text_config"]["num_kv_shared_layers"]

    config = Gemma4TextConfig.from_text_encoder_config(cfg)

    assert config.attention_k_eq_v is False


def test_gemma_version_missing_is_none():
    import copy

    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    cfg = copy.deepcopy(_FULL_CONFIG)
    del cfg["gemma_version"]

    config = Gemma4TextConfig.from_text_encoder_config(cfg)
    assert config.gemma_version is None


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_parses_real_pack_text_encoder_config():
    import json

    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    with open(LTX25_Q8_DIR / "text_encoder_config.json") as f:
        raw = json.load(f)

    config = Gemma4TextConfig.from_text_encoder_config(raw)

    assert config.gemma_version == "gemma4-12b-ltx-v1"
    assert config.num_hidden_layers == 48
    full_indices = [i for i in range(config.num_hidden_layers) if not config.layer_is_sliding(i)]
    assert full_indices == _FULL_ATTENTION_INDICES

    assert config.layer_head_dim(0) == 256
    assert config.layer_head_dim(5) == 512
    assert config.layer_num_kv(0) == 8
    assert config.layer_num_kv(5) == 1

    assert config.rope_local_theta == 10000.0
    assert config.rope_global_theta == 1000000.0
    assert config.partial_rotary_factor == 0.25
    assert config.attention_k_eq_v is True
