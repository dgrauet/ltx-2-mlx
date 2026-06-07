"""Regression tests for the av_ca_timestep_scale_multiplier fix (issue #37).

Every LTX-2.3 checkpoint ships ``av_ca_timestep_scale_multiplier = 1000.0`` in
both ``config.json`` and ``embedded_config.json``, but the MLX
``LTXModelConfig`` dataclass default was ``1.0`` and the loaders never read the
checkpoint config — so the AV cross-attention gate AdaLN received
``sigma * 1`` instead of ``sigma * 1000``, mis-gating cross-modal
(speech / lip-sync) information.

The fix mirrors upstream ``LTXModelConfigurator.from_config``: read the
hyperparameters from the checkpoint config. These tests pin that behaviour and
guard against architecture drift (every *other* mapped field must still equal
the dataclass default for the shipped checkpoints).
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.transformer.model import LTXModelConfig
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding

# A minimal but faithful slice of the real LTX-2.3 transformer config.
_CKPT_TRANSFORMER = {
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "num_layers": 48,
    "cross_attention_dim": 4096,
    "timestep_scale_multiplier": 1000,
    "av_ca_timestep_scale_multiplier": 1000.0,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "in_channels": 128,
    "audio_in_channels": 128,
    "rope_type": "split",
    "positional_embedding_theta": 10000.0,
    "norm_eps": 1e-06,
}


def test_av_ca_multiplier_read_from_checkpoint():
    """The checkpoint value (1000.0) overrides the dataclass default (1.0)."""
    cfg = LTXModelConfig.from_checkpoint_config({"transformer": _CKPT_TRANSFORMER})
    assert cfg.av_ca_timestep_scale_multiplier == 1000.0
    assert LTXModelConfig().av_ca_timestep_scale_multiplier == 1.0  # unchanged default


def test_missing_key_falls_back_to_default():
    """A config without the key keeps the dataclass default (no KeyError)."""
    t = {k: v for k, v in _CKPT_TRANSFORMER.items() if k != "av_ca_timestep_scale_multiplier"}
    cfg = LTXModelConfig.from_checkpoint_config({"transformer": t})
    assert cfg.av_ca_timestep_scale_multiplier == LTXModelConfig().av_ca_timestep_scale_multiplier


def test_no_architecture_drift_on_shipped_checkpoint():
    """Only av_ca_timestep_scale_multiplier differs from defaults for the shipped config.

    Reading the checkpoint must not silently change any architectural field
    (dims/heads/layers), which would corrupt weight loading.
    """
    cfg = LTXModelConfig.from_checkpoint_config({"transformer": _CKPT_TRANSFORMER})
    default = LTXModelConfig()
    differing = {f.name for f in fields(LTXModelConfig) if getattr(cfg, f.name) != getattr(default, f.name)}
    assert differing == {"av_ca_timestep_scale_multiplier"}


def test_accepts_bare_transformer_dict():
    """``from_checkpoint_config`` accepts either the full dict or the sub-dict."""
    a = LTXModelConfig.from_checkpoint_config({"transformer": _CKPT_TRANSFORMER})
    b = LTXModelConfig.from_checkpoint_config(_CKPT_TRANSFORMER)
    assert a == b


def test_from_checkpoint_dir_prefers_embedded(tmp_path: Path):
    """``embedded_config.json`` wins over ``config.json`` when both exist."""
    (tmp_path / "config.json").write_text(
        json.dumps({"transformer": {**_CKPT_TRANSFORMER, "av_ca_timestep_scale_multiplier": 7.0}})
    )
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": _CKPT_TRANSFORMER}))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.av_ca_timestep_scale_multiplier == 1000.0  # from embedded, not config


def test_from_checkpoint_dir_falls_back_to_config_json(tmp_path: Path):
    """``config.json`` is used when ``embedded_config.json`` is absent."""
    (tmp_path / "config.json").write_text(json.dumps({"transformer": _CKPT_TRANSFORMER}))
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg.av_ca_timestep_scale_multiplier == 1000.0


def test_from_checkpoint_dir_missing_returns_defaults(tmp_path: Path, capsys):
    """No config files → defaults (with a loud stderr warning), no crash."""
    cfg = LTXModelConfig.from_checkpoint_dir(tmp_path)
    assert cfg == LTXModelConfig()
    assert "issue #37" in capsys.readouterr().err


def test_gate_timestep_embedding_is_load_bearing():
    """av_ca 1.0 vs 1000.0 yields a materially different gate timestep embedding.

    The gate AdaLN input is ``sigma * av_ca_timestep_scale_multiplier`` (see
    ``LTXModel.__call__``). A different multiplier => a different point on the
    sinusoidal embedding => a different gate.
    """
    sigma = mx.array([0.5], dtype=mx.float32)
    dim = LTXModelConfig().timestep_embedding_dim
    emb_1 = get_timestep_embedding(sigma * 1.0, dim)
    emb_1000 = get_timestep_embedding(sigma * 1000.0, dim)
    mx.eval(emb_1, emb_1000)
    assert not mx.allclose(emb_1, emb_1000, atol=1e-3)
