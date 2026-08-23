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


def test_rope_double_precision_flag():
    from ltx_core_mlx.model.transformer.rope import compute_freqs, generate_freq_grid

    grid32 = generate_freq_grid(theta=10000.0, num_pos_dims=3, inner_dim=128)
    grid64 = generate_freq_grid(theta=10000.0, num_pos_dims=3, inner_dim=128, double_precision=True)
    assert grid64.dtype == mx.float32  # toujours casté en sortie

    positions = mx.array([[[0.5, 100.0, 200.0]]])
    f32 = compute_freqs(grid32, positions, max_pos=[20, 2048, 2048])
    f64 = compute_freqs(grid64, positions, max_pos=[20, 2048, 2048], double_precision=True)
    assert f64.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(f64)).item())

    # Flag off == comportement actuel, bit-identique.
    again = compute_freqs(grid32, positions, max_pos=[20, 2048, 2048])
    assert bool(mx.array_equal(f32, again).item())


def test_embeddings_connector_double_precision_rope():
    from ltx_core_mlx.text_encoders.gemma.embeddings_connector import Embeddings1DConnector

    mx.random.seed(0)
    hidden_states = mx.random.normal((1, 16, 32))

    connector_default = Embeddings1DConnector(
        dim=32,
        num_heads=2,
        head_dim=16,
        num_layers=1,
        num_registers=0,
        max_pos=64,
    )
    connector_f64 = Embeddings1DConnector(
        dim=32,
        num_heads=2,
        head_dim=16,
        num_layers=1,
        num_registers=0,
        max_pos=64,
        double_precision_rope=True,
    )
    # Same weights for both instances so the only difference is the flag.
    connector_f64.update(connector_default.parameters())

    assert connector_default.double_precision_rope is False
    assert connector_f64.double_precision_rope is True

    out_default = connector_default(hidden_states)
    out_f64 = connector_f64(hidden_states)

    assert out_default.dtype == mx.float32
    assert out_f64.dtype == mx.float32

    # double_precision_rope=True must not change the default (flag off) path.
    out_default_again = connector_default(hidden_states)
    assert bool(mx.array_equal(out_default, out_default_again).item())


def test_25_shaped_weights_without_config_raise(tmp_path):
    """Un transformer sans biais de ff + une config 2.3 -> erreur explicite."""
    from ltx_core_mlx.utils.weights import validate_config_matches_weights

    # Un header minimal de forme 2.5 : le poids existe, le bias non.
    path = tmp_path / "transformer-dev.safetensors"
    mx.save_safetensors(
        str(path),
        {"transformer.transformer_blocks.0.ff.proj_in.weight": mx.zeros((4, 4))},
    )
    with pytest.raises(ValueError, match="ff_bias"):
        validate_config_matches_weights(path, LTXModelConfig())  # défauts 2.3
    # La config 2.5 correspondante passe.
    validate_config_matches_weights(path, LTXModelConfig(ff_bias=False))


def test_23_pack_requests_double_precision_rope():
    """The shipped 2.3 packs declare frequencies_precision=float64 — the field
    our port left inert until the versioned config landed, which is why the
    2.3 SHA baseline moved (re-baselined 2026-08-24, upstream-faithful)."""
    from tests.conftest import MODEL_DIR

    if MODEL_DIR is None:
        pytest.skip("q8 2.3 pack not found")
    c = LTXModelConfig.from_checkpoint_dir(MODEL_DIR)
    assert c.double_precision_rope is True
