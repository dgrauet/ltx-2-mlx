"""Tests for Gemma4TextConfig parsing (LTX-2.5 text tower config).

Covers a truncated-but-representative literal dict mirroring the shape of
the real ``text_encoder_config.json`` pack file, plus a slow test that
parses the real pack when available (``LTX25_Q8_DIR``).
"""

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Parity against the torch reference (tests/parity_gemma4_reference.py).
#
# Generate the npz first (disposable env, torch + transformers):
#
#   uv run --no-project --with "transformers>=5.10,<6" --with torch --with numpy \
#       python tests/parity_gemma4_reference.py --out /tmp/gemma4_parity.npz
#
# These tests skip when the npz is absent.
# ---------------------------------------------------------------------------

PARITY_NPZ = Path("/tmp/gemma4_parity.npz")

_needs_parity_npz = pytest.mark.skipif(
    not PARITY_NPZ.exists(),
    reason=f"reference npz not generated at {PARITY_NPZ} (see tests/parity_gemma4_reference.py)",
)

# Mirrors tests/parity_gemma4_reference.py::TINY_CONFIG, in the pack's
# JSON shape so Gemma4TextConfig parses it.
_TINY_PACK_CONFIG = {
    "gemma_version": "tiny-parity",
    "text_config": {
        "model_type": "gemma4_unified_text",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 16,
        "global_head_dim": 32,
        "sliding_window": 8,
        "layer_types": ["sliding_attention", "full_attention"],
        "attention_k_eq_v": True,
        "num_kv_shared_layers": 0,
        "rms_norm_eps": 1e-6,
        "pad_token_id": 0,
        "use_bidirectional_attention": "vision",
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    },
}

PARITY_ATOL = 2e-5


@pytest.fixture(scope="module")
def ref():
    """The torch reference arrays."""
    import numpy as np

    return dict(np.load(PARITY_NPZ))


@pytest.fixture(scope="module")
def tiny_config():
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    return Gemma4TextConfig.from_text_encoder_config(_TINY_PACK_CONFIG)


def _max_abs_diff(a, b):
    import mlx.core as mx
    import numpy as np

    return float(np.abs(np.array(mx.array(a).astype(mx.float32)) - np.asarray(b)).max())


@pytest.mark.slow
@_needs_parity_npz
def test_rmsnorm_matches_reference(ref):
    """Gemma4RMSNorm reproduces the reference input_layernorm exactly."""
    import mlx.core as mx

    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4RMSNorm

    norm = Gemma4RMSNorm(64, eps=1e-6)
    norm.update({"weight": mx.array(ref["w.layers.0.input_layernorm.weight"])})

    got = norm(mx.array(ref["hs.0"]))

    assert _max_abs_diff(got, ref["L0.attn_in"]) < PARITY_ATOL


@pytest.mark.slow
@_needs_parity_npz
def test_rmsnorm_without_scale_is_scale_free():
    """with_scale=False registers no weight and skips the multiply."""
    import mlx.core as mx

    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4RMSNorm

    norm = Gemma4RMSNorm(4, eps=1e-6, with_scale=False)
    assert "weight" not in norm.parameters()

    x = mx.array([[3.0, 0.0, 0.0, 0.0]])
    got = norm(x)
    assert abs(float(got[0, 0]) - 2.0) < 1e-5


@pytest.mark.slow
@_needs_parity_npz
@pytest.mark.parametrize("layer_type,layer_idx", [("sliding_attention", 0), ("full_attention", 1)])
def test_rotary_matches_reference(ref, tiny_config, layer_type, layer_idx):
    """Both rope parametrizations (default / proportional+partial) match."""
    import mlx.core as mx

    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4RotaryEmbedding

    rope = Gemma4RotaryEmbedding.from_config(tiny_config, layer_idx)

    # 1 float32 ULP apart from torch: same formula, different rounding order.
    assert _max_abs_diff(rope.inv_freq, ref[f"rope.{layer_type}.inv_freq"]) < 1e-7

    positions = mx.arange(ref["input_ids"].shape[1])[None]
    cos, sin = rope(positions)

    assert _max_abs_diff(cos, ref[f"rope.{layer_type}.cos"]) < PARITY_ATOL
    assert _max_abs_diff(sin, ref[f"rope.{layer_type}.sin"]) < PARITY_ATOL


@pytest.mark.slow
@_needs_parity_npz
@pytest.mark.parametrize("layer_type,window", [("full_attention", None), ("sliding_attention", 8)])
def test_attention_mask_matches_reference(ref, layer_type, window):
    """Causal (and causal+window) masks match the reference mask pattern."""
    import mlx.core as mx
    import numpy as np

    from ltx_core_mlx.text_encoders.gemma.gemma4 import build_attention_mask

    seq_len = ref["input_ids"].shape[1]
    got = np.array(build_attention_mask(seq_len, sliding_window=window))
    want = ref[f"mask.{layer_type}"]

    assert (got > -1).squeeze().tolist() == (want > -1).squeeze().tolist()
    assert mx.array(got).dtype == mx.float32


@pytest.mark.slow
@_needs_parity_npz
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_attention_matches_reference(ref, tiny_config, layer_idx):
    """Both attention flavors match the reference block output."""
    import mlx.core as mx

    from ltx_core_mlx.text_encoders.gemma.gemma4 import (
        Gemma4Attention,
        Gemma4RotaryEmbedding,
        build_attention_mask,
    )

    attn = Gemma4Attention(tiny_config, layer_idx)
    prefix = f"w.layers.{layer_idx}.self_attn."
    weights = {key[len(prefix) :]: mx.array(value) for key, value in ref.items() if key.startswith(prefix)}
    assert ("v_proj.weight" in weights) is (layer_idx == 0)
    attn.load_weights(list(weights.items()))

    seq_len = ref["input_ids"].shape[1]
    rope = Gemma4RotaryEmbedding.from_config(tiny_config, layer_idx)
    cos, sin = rope(mx.arange(seq_len)[None])
    window = tiny_config.sliding_window if tiny_config.layer_is_sliding(layer_idx) else None
    mask = build_attention_mask(seq_len, sliding_window=window)

    got = attn(mx.array(ref[f"L{layer_idx}.attn_in"]), cos, sin, mask)

    assert _max_abs_diff(got, ref[f"L{layer_idx}.attn_out"]) < PARITY_ATOL


@pytest.mark.slow
@_needs_parity_npz
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_mlp_matches_reference(ref, tiny_config, layer_idx):
    """The gated gelu_pytorch_tanh MLP matches the reference block output."""
    import mlx.core as mx

    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4MLP

    mlp = Gemma4MLP(tiny_config)
    prefix = f"w.layers.{layer_idx}.mlp."
    mlp.load_weights([(key[len(prefix) :], mx.array(value)) for key, value in ref.items() if key.startswith(prefix)])

    got = mlp(mx.array(ref[f"L{layer_idx}.mlp_in"]))

    assert _max_abs_diff(got, ref[f"L{layer_idx}.mlp_out"]) < PARITY_ATOL


# ---------------------------------------------------------------------------
# Full-tower assembly (Gemma4TextModel): embeddings + N layers + final norm.
# ---------------------------------------------------------------------------


def _load_tiny_tower(ref, tiny_config):
    """Build a Gemma4TextModel matching the tiny reference config and load
    its weights from the reference npz's ``w.*`` state dict."""
    import mlx.core as mx

    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextModel

    tower = Gemma4TextModel(tiny_config)
    weights = [(key[2:], mx.array(value)) for key, value in ref.items() if key.startswith("w.")]
    tower.load_weights(weights)
    return tower


@pytest.mark.slow
@_needs_parity_npz
def test_tower_matches_reference_hidden_states(ref, tiny_config):
    """embed_tokens (x sqrt(hidden)) + both layers reproduce hs.0/hs.1/hs.2."""
    import mlx.core as mx

    tower = _load_tiny_tower(ref, tiny_config)

    got = tower(mx.array(ref["input_ids"]))

    assert len(got) == tiny_config.num_hidden_layers + 1 == 3
    for i, hs in enumerate(got):
        assert _max_abs_diff(hs, ref[f"hs.{i}"]) < PARITY_ATOL


@pytest.mark.slow
@_needs_parity_npz
def test_tower_final_norm_matches_reference(ref, tiny_config):
    """self.norm applied to the last hidden state reproduces hs.final."""
    import mlx.core as mx

    tower = _load_tiny_tower(ref, tiny_config)

    got = tower(mx.array(ref["input_ids"]))
    final = tower.norm(got[-1])

    assert _max_abs_diff(final, ref["hs.final"]) < PARITY_ATOL


@pytest.mark.slow
@_needs_parity_npz
def test_tower_matches_reference_padded_batch(ref, tiny_config):
    """Left-padded batch (row 0 padded, row 1 not): padding_mask path.

    Padded query positions can legitimately diverge (an all-masked softmax
    row is undefined in both frameworks), so only real (non-padded) token
    positions are compared.
    """
    import mlx.core as mx
    import numpy as np

    tower = _load_tiny_tower(ref, tiny_config)

    input_ids = mx.array(ref["padded.input_ids"])
    attention_mask = mx.array(ref["padded.attention_mask"])
    valid = np.asarray(ref["padded.attention_mask"]) != 0  # (B, T)

    got = tower(input_ids, attention_mask=attention_mask)

    assert len(got) == tiny_config.num_hidden_layers + 1 == 3
    for i, hs in enumerate(got):
        got_np = np.array(mx.array(hs).astype(mx.float32))
        want_np = ref[f"padded.hs.{i}"]
        diff = np.abs(got_np - want_np)[valid]
        assert float(diff.max()) < PARITY_ATOL

    final = tower.norm(got[-1])
    final_np = np.array(mx.array(final).astype(mx.float32))
    want_final = ref["padded.hs.final"]
    diff = np.abs(final_np - want_final)[valid]
    assert float(diff.max()) < PARITY_ATOL


# ---------------------------------------------------------------------------
# Load contract: the fail-loud allowlist logic, tested against synthetic
# key sets (fast -- no pack access needed).
# ---------------------------------------------------------------------------


def test_pack_ignore_allowlist_has_exactly_15_entries():
    from ltx_core_mlx.text_encoders.gemma.gemma4 import GEMMA4_PACK_IGNORE_ALLOWLIST

    assert len(GEMMA4_PACK_IGNORE_ALLOWLIST) == 15
    assert all(
        k.startswith("text_encoder.") and not k.startswith("text_encoder.model.") for k in GEMMA4_PACK_IGNORE_ALLOWLIST
    )


def test_load_from_pack_rejects_stray_keys(tmp_path):
    """A key neither under the model prefix nor in the allowlist must fail loudly."""
    import json
    import struct

    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextModel

    text_config = _TINY_PACK_CONFIG["text_config"]
    (tmp_path / "text_encoder_config.json").write_text(json.dumps(_TINY_PACK_CONFIG))

    header = {
        "text_encoder.model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [text_config["vocab_size"], text_config["hidden_size"]],
            "data_offsets": [0, 0],
        },
        "text_encoder.bogus_new_component.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 0],
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with open(tmp_path / "text_encoder.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)

    with pytest.raises(ValueError, match=r"text_encoder\.bogus_new_component\.weight"):
        Gemma4TextModel.load_from_pack(tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_25_gemma4_config_covers_every_pack_tensor():
    """Mirrors tests/test_ltx25_load_contract.py: two-direction key check
    against the real q8 pack, without materializing the 16 GB of weights.

    ``mx.load`` mmaps without materializing, and ``Gemma4TextModel(config)``
    only allocates a lazy parameter tree -- shapes are known without
    evaluating, so this stays cheap even for the real 12B-param tower.
    """
    import json

    from mlx.utils import tree_flatten

    from ltx_core_mlx.text_encoders.gemma.gemma4 import (
        GEMMA4_PACK_IGNORE_ALLOWLIST,
        Gemma4TextModel,
        _safetensors_header_keys,
    )
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    with open(LTX25_Q8_DIR / "text_encoder_config.json") as f:
        raw_config = json.load(f)
    config = Gemma4TextConfig.from_text_encoder_config(raw_config)
    model = Gemma4TextModel(config)

    model_keys = {f"text_encoder.model.{k}" for k, _ in tree_flatten(model.parameters())}

    pack = _safetensors_header_keys(LTX25_Q8_DIR / "text_encoder.safetensors")
    pack_model = {k for k in pack if k.startswith("text_encoder.model.")}
    pack_other = pack - pack_model

    assert pack_other == GEMMA4_PACK_IGNORE_ALLOWLIST

    # Quantized layers save weight/scales/biases separately; normalize back
    # onto a single ".weight" entry, same convention as the DiT contract.
    pack_normalized = {
        k.removesuffix(".scales").removesuffix(".biases") + ".weight" if k.endswith((".scales", ".biases")) else k
        for k in pack_model
    }
    missing_in_model = sorted(pack_normalized - model_keys)
    unfed_params = sorted(model_keys - pack_normalized)
    assert not missing_in_model, f"pack tensors nobody would load: {missing_in_model[:10]}"
    assert not unfed_params, f"model params the pack does not feed: {unfed_params[:10]}"
    assert len(pack_normalized) == 666
