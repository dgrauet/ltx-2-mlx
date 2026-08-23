"""Parsing de la config 2.5 — assertions contre le mapping upstream.

Le dict CONFIG_25 ci-dessous est la copie verbatim du bloc "transformer" de
embedded_config.json du pack 2.5 (sha a59ff335…), tronquée aux champs que le
mapping consomme + quelques champs 2.3 pour vérifier la non-régression.
"""

import mlx.core as mx
import pytest

from ltx_core_mlx.model.transformer.feed_forward import FeedForward
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

CONFIG_25 = {
    "transformer": {
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "audio_cross_attention_dim": 2048,
        "av_ca_timestep_scale_multiplier": 1000.0,
        "rope_type": "split",
        "ff_bias": False,
        "share_ff": False,
        "use_keyframes_abs_pos_embedding": True,
        "frequencies_precision": "float64",
    }
}

CONFIG_23 = {
    "transformer": {
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "av_ca_timestep_scale_multiplier": 1000.0,
        "rope_type": "split",
    }
}


def test_25_fields_parsed_from_checkpoint_config():
    c = LTXModelConfig.from_checkpoint_config(CONFIG_25)
    assert c.ff_bias is False
    assert c.audio_ff_bias is True  # absent du checkpoint -> défaut upstream
    assert c.use_prompt_adaln_single is True  # absent -> défaut upstream
    assert c.use_keyframes_abs_pos_embedding is True
    assert c.double_precision_rope is True  # frequencies_precision == float64


def test_23_config_yields_23_defaults():
    c = LTXModelConfig.from_checkpoint_config(CONFIG_23)
    assert c.ff_bias is True
    assert c.audio_ff_bias is True
    assert c.use_keyframes_abs_pos_embedding is False
    assert c.double_precision_rope is False


def test_dataclass_defaults_are_23_behavior():
    c = LTXModelConfig()
    assert c.ff_bias is True
    assert c.audio_ff_bias is True
    assert c.use_prompt_adaln_single is True
    assert c.use_keyframes_abs_pos_embedding is False
    assert c.double_precision_rope is False


def test_share_ff_true_is_rejected():
    # Upstream: check_config_value(config, "share_ff", False) — assertion dure.
    bad = {"transformer": {**CONFIG_25["transformer"], "share_ff": True}}
    with pytest.raises(ValueError, match="share_ff"):
        LTXModelConfig.from_checkpoint_config(bad)


def test_feed_forward_bias_toggle():
    with_bias = FeedForward(64, mult=2.0)
    without = FeedForward(64, mult=2.0, bias=False)
    assert "bias" in with_bias.proj_in
    assert "bias" in with_bias.proj_out
    assert "bias" not in without.proj_in
    assert "bias" not in without.proj_out


def _tiny(cfg_kwargs: dict) -> LTXModel:
    return LTXModel(
        LTXModelConfig(
            num_layers=1,
            video_dim=64,
            audio_dim=32,
            video_num_heads=2,
            audio_num_heads=2,
            video_head_dim=32,
            audio_head_dim=16,
            av_cross_num_heads=2,
            av_cross_head_dim=16,
            **cfg_kwargs,
        )
    )


def test_model_threads_ff_bias_to_blocks():
    m25 = _tiny({"ff_bias": False, "audio_ff_bias": True})
    block = m25.transformer_blocks[0]
    assert "bias" not in block.ff.proj_in
    assert "bias" in block.audio_ff.proj_in
    # Défauts 2.3 : tout le monde a des biais.
    m23 = _tiny({})
    assert "bias" in m23.transformer_blocks[0].ff.proj_in


def test_keyframes_pos_embedding_created_when_configured():
    m25 = _tiny({"use_keyframes_abs_pos_embedding": True})
    assert m25.keyframes_abs_pos_embedding.shape == (1, 64)
    assert bool(mx.all(m25.keyframes_abs_pos_embedding == 0).item())
    m23 = _tiny({})
    assert not hasattr(m23, "keyframes_abs_pos_embedding")
