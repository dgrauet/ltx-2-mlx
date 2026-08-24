"""Tests for Gemma4TextConfig parsing (LTX-2.5 text tower config).

Covers a truncated-but-representative literal dict mirroring the shape of
the real ``text_encoder_config.json`` pack file, plus a slow test that
parses the real pack when available (``LTX25_Q8_DIR``).
"""

from pathlib import Path
from typing import ClassVar

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

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


def test_rejects_wrong_root_model_type():
    """Root-level model_type (config["model_type"], sibling of text_config) must
    also be gemma4_unified -- distinct from text_config.model_type checked above."""
    import copy

    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    bad = copy.deepcopy(_FULL_CONFIG)
    bad["model_type"] = "gemma3"
    with pytest.raises(ValueError, match="model_type"):
        Gemma4TextConfig.from_text_encoder_config(bad)


def test_rejects_flipped_sliding_rope_type():
    """A pack with sliding_attention.rope_type != "default" parses fine under
    every other check (same keys, same types) and would silently compute wrong
    rotary tables -- must be caught explicitly."""
    import copy

    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    bad = copy.deepcopy(_FULL_CONFIG)
    bad["text_config"]["rope_parameters"]["sliding_attention"]["rope_type"] = "proportional"
    with pytest.raises(ValueError, match=r"rope_parameters\.sliding_attention\.rope_type"):
        Gemma4TextConfig.from_text_encoder_config(bad)


def test_rejects_flipped_full_attention_rope_type():
    """Mirror of the sliding-side guard, for full_attention.rope_type != "proportional"."""
    import copy

    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    bad = copy.deepcopy(_FULL_CONFIG)
    bad["text_config"]["rope_parameters"]["full_attention"]["rope_type"] = "default"
    with pytest.raises(ValueError, match=r"rope_parameters\.full_attention\.rope_type"):
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
    "model_type": "gemma4_unified",
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


# ---------------------------------------------------------------------------
# Gemma4TextEncoder: system prompts, the post-final-norm stacking contract,
# and tokenizer LEFT-pad + real-tokenizer parity.
# ---------------------------------------------------------------------------

_PROMPTS_DIR = (
    Path(__file__).resolve().parents[1] / "packages/ltx-core-mlx/src/ltx_core_mlx/text_encoders/gemma/encoders/prompts"
)

# Pinned at port time (2026-08-24) against upstream Lightricks/LTX-2 @ main:
#   packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_{t2v,i2v}_system_prompt.txt
_PROMPT_INTEGRITY = {
    "gemma4_t2v_system_prompt.txt": {
        "length": 3769,
        "sha256": "0cddf69456bcd51e65430f848386295d9ac4d17d5df3ea65d5f3d8a9ad842f3c",
    },
    "gemma4_i2v_system_prompt.txt": {
        "length": 4708,
        "sha256": "15992bfb757d3bbd83f2d27ad86e450fc4caffa0f7cb7523772a60e346ef3fee",
    },
}


@pytest.mark.parametrize("prompt_name", sorted(_PROMPT_INTEGRITY))
def test_gemma4_system_prompt_matches_upstream_verbatim(prompt_name):
    """The prompt files must be byte-identical to what was fetched from upstream.

    Any future edit (deliberate or accidental) changes the sha and must
    update this pin explicitly -- these prompts are consumed verbatim as
    the Gemma-4 system message and are not meant to drift silently.
    """
    import hashlib

    path = _PROMPTS_DIR / prompt_name
    data = path.read_bytes()

    assert len(data) == _PROMPT_INTEGRITY[prompt_name]["length"]
    assert hashlib.sha256(data).hexdigest() == _PROMPT_INTEGRITY[prompt_name]["sha256"]


def test_gemma4_text_encoder_loads_default_system_prompts():
    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4TextEncoder

    encoder = Gemma4TextEncoder()
    t2v = encoder.default_gemma4_t2v_system_prompt
    i2v = encoder.default_gemma4_i2v_system_prompt
    assert len(t2v.encode("utf-8")) == _PROMPT_INTEGRITY["gemma4_t2v_system_prompt.txt"]["length"]
    assert len(i2v.encode("utf-8")) == _PROMPT_INTEGRITY["gemma4_i2v_system_prompt.txt"]["length"]


@pytest.mark.slow
def test_gemma4_get_all_hidden_states_last_entry_is_post_final_norm():
    """HARD REQUIREMENT (Task 3 ledger): entry -1 of the 49-stack must be
    ``tower.norm(raw[-1])``, not the raw tower output -- HF's
    ``output_hidden_states=True`` tuple ties its last entry to the
    post-final-norm ``last_hidden_state``
    (``transformers/output_capturing.py:261-263``), and the upstream
    encoder consumes that tuple as-is (see
    ``upstream_gemma_base_encoder.py:68-71``).

    Uses the tiny 2-layer config (no real pack needed) with randomly
    initialized weights, so the norm's effect is guaranteed non-trivial:
    a bug that forgets to apply ``tower.norm`` would make this test fail,
    not silently pass.
    """
    import numpy as np

    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4TextEncoder
    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextModel
    from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

    config = Gemma4TextConfig.from_text_encoder_config(_TINY_PACK_CONFIG)
    tower = Gemma4TextModel(config)

    # Randomize weights: default init leaves norm weights at zero (Gemma
    # RMSNorm has no "+1" -- see Task 2 report), which would make
    # norm(x) == x * rsqrt(mean(x^2)+eps) still differ from raw x, but
    # randomizing everything makes the mismatch unmistakably large rather
    # than relying on that one coincidence.
    mx.random.seed(0)
    flat = dict(tree_flatten(tower.parameters()))
    randomized = {k: mx.random.normal(v.shape) * 0.1 for k, v in flat.items()}
    tower.update(tree_unflatten(list(randomized.items())))

    encoder = Gemma4TextEncoder(tower=tower)
    input_ids = mx.random.randint(0, config.vocab_size, (1, 6))

    raw_states = tower(input_ids, attention_mask=None)
    got_states = encoder.get_all_hidden_states(input_ids, attention_mask=None)

    assert len(got_states) == len(raw_states) == config.num_hidden_layers + 1

    # Entries 0..-2: identical to the raw tower output.
    for i in range(len(raw_states) - 1):
        diff = float(np.abs(np.array(got_states[i]) - np.array(raw_states[i])).max())
        assert diff == 0.0, f"entry {i} must be byte-identical to the raw tower output, got diff={diff}"

    # Entry -1: must equal tower.norm(raw[-1]) ...
    want_final = tower.norm(raw_states[-1])
    diff = float(np.abs(np.array(got_states[-1]) - np.array(want_final)).max())
    assert diff == 0.0

    # ... and must NOT equal the raw (pre-norm) last entry -- catches the
    # #37-class bug of silently forgetting to apply the final norm.
    raw_vs_got = float(np.abs(np.array(got_states[-1]) - np.array(raw_states[-1])).max())
    assert raw_vs_got > 1e-3, "entry -1 must differ from the raw pre-norm tower output"


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_gemma4_tokenize_left_pads_with_binary_mask():
    """Our convention: LEFT padding + binary attention mask (divergence
    from upstream's right-pad-before-connector, documented at the top of
    ``base_encoder.py`` / the module docstring of ``gemma4_encoder.py``).
    """
    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4TextEncoder

    encoder = Gemma4TextEncoder()
    encoder.load_tokenizer(LTX25_Q8_DIR)

    max_length = 32
    token_ids, attention_mask = encoder.tokenize("hello world", max_length=max_length)

    assert token_ids.shape == (1, max_length)
    assert attention_mask.shape == (1, max_length)

    mask_np = np.array(attention_mask)[0]
    ids_np = np.array(token_ids)[0]

    # Real tokens are a contiguous suffix (left padding).
    num_real = int(mask_np.sum())
    assert num_real > 0
    assert (mask_np[: max_length - num_real] == 0).all()
    assert (mask_np[max_length - num_real :] == 1).all()

    # Padding uses the tokenizer's native pad id.
    pad_id = encoder.tokenizer.pad_token_id
    assert (ids_np[: max_length - num_real] == pad_id).all()


@pytest.mark.slow
@_needs_parity_npz
def test_gemma4_tokenizer_matches_pack_reference_ids(ref):
    """Our tokenize() must produce the same ids as transformers.AutoTokenizer
    loaded straight from the pack, for the same control sentence -- this is
    the "tokenizer parity" leg of the harness (Task 4 brief), independent of
    the tiny-config attention/rotary parity above.
    """
    if "tokenizer.control_ids" not in ref:
        pytest.skip("npz was regenerated without the real pack available (see parity_gemma4_reference.PACK_DIR)")
    if LTX25_Q8_DIR is None:
        pytest.skip("local ltx-2.5-mlx-q8 pack not found")

    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4TextEncoder

    # Must stay byte-identical to parity_gemma4_reference.CONTROL_SENTENCE.
    # Not imported directly: that module imports torch at module scope
    # (a --no-project-only dependency), which would break collection of
    # this test file in the normal project env.
    CONTROL_SENTENCE = "A lone lighthouse keeper watches the storm roll in over the grey harbor."

    encoder = Gemma4TextEncoder()
    encoder.load_tokenizer(LTX25_Q8_DIR)

    want_ids = ref["tokenizer.control_ids"].tolist()
    # Pad wide enough that the control sentence is never truncated, then
    # strip the left padding back off using the mask -- exercises the real
    # tokenize() codepath end-to-end rather than calling the raw tokenizer.
    max_length = len(want_ids) + 64
    token_ids, attention_mask = encoder.tokenize(CONTROL_SENTENCE, max_length=max_length)

    mask_np = np.array(attention_mask)[0]
    ids_np = np.array(token_ids)[0]
    num_real = int(mask_np.sum())
    got_ids = ids_np[max_length - num_real :].tolist()

    assert got_ids == want_ids


# --- select_text_encoder / check_gemma_version (Task 5: evidence-based configurator) ---


def _write_json(path: Path, data: dict) -> None:
    import json

    with open(path, "w") as f:
        json.dump(data, f)


class TestSelectTextEncoder:
    """``select_text_encoder`` keys purely off file presence in ``model_dir``."""

    def test_both_files_present_selects_gemma4(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        (tmp_path / "text_encoder.safetensors").touch()
        _write_json(tmp_path / "text_encoder_config.json", {"gemma_version": "gemma4-12b-ltx-v1"})

        assert select_text_encoder(tmp_path) == "gemma4"

    def test_only_weights_present_selects_gemma3(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        (tmp_path / "text_encoder.safetensors").touch()

        assert select_text_encoder(tmp_path) == "gemma3"

    def test_only_config_present_selects_gemma3(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        _write_json(tmp_path / "text_encoder_config.json", {"gemma_version": "gemma4-12b-ltx-v1"})

        assert select_text_encoder(tmp_path) == "gemma3"

    def test_neither_present_selects_gemma3(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        assert select_text_encoder(tmp_path) == "gemma3"

    @pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
    def test_real_pack_selects_gemma4_without_raising(self):
        """The real pack has no ``gemma_source_checkpoint`` in embedded_config.json,
        so selection must succeed without the version guard raising."""
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        assert select_text_encoder(LTX25_Q8_DIR) == "gemma4"


class TestCheckGemmaVersion:
    """Mirrors upstream ``_check_gemma_version``, scoped to the mismatch case only."""

    def _make_pack(self, tmp_path: Path, *, embedded_config: dict | None, tower_gemma_version: str | None) -> Path:
        (tmp_path / "text_encoder.safetensors").touch()
        _write_json(tmp_path / "text_encoder_config.json", {"gemma_version": tower_gemma_version})
        if embedded_config is not None:
            _write_json(tmp_path / "embedded_config.json", embedded_config)
        return tmp_path

    def test_mismatched_version_raises(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        pack = self._make_pack(
            tmp_path,
            embedded_config={"gemma_source_checkpoint": {"gemma_version": "gemma4-12b-ltx-v1"}},
            tower_gemma_version="some-other-version",
        )

        with pytest.raises(ValueError, match="Gemma version mismatch"):
            select_text_encoder(pack)

    def test_matching_version_passes(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        pack = self._make_pack(
            tmp_path,
            embedded_config={"gemma_source_checkpoint": {"gemma_version": "gemma4-12b-ltx-v1"}},
            tower_gemma_version="gemma4-12b-ltx-v1",
        )

        assert select_text_encoder(pack) == "gemma4"

    def test_no_declaration_passes_unchecked(self, tmp_path):
        """embedded_config.json present but with no gemma_source_checkpoint key at all
        (the real LTX-2.5 pack's shape) -- no check performed, no raise."""
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        pack = self._make_pack(
            tmp_path,
            embedded_config={"transformer": {}, "scheduler": {}},
            tower_gemma_version="gemma4-12b-ltx-v1",
        )

        assert select_text_encoder(pack) == "gemma4"

    def test_no_embedded_config_file_passes_unchecked(self, tmp_path):
        """No embedded_config.json at all -- no check performed, no raise."""
        from ltx_core_mlx.text_encoders.gemma.encoders.encoder_configurator import select_text_encoder

        pack = self._make_pack(tmp_path, embedded_config=None, tower_gemma_version="gemma4-12b-ltx-v1")

        assert select_text_encoder(pack) == "gemma4"


class TestPromptEncoderWiring:
    """Proves the 2.3 (gemma3) construction site is untouched and the selection is real.

    ``_stub_feature_extractor_deps`` fakes ``load_split_safetensors`` with a
    prefix-keyed recorder (rather than a blanket ``{}``) so these tests stay
    load-bearing against the gemma4 ``text_embedding_projection`` merge
    (commit 8bc5640): a stub that ignores ``prefix`` would go on returning
    the same ``{}`` even if that merge were reverted, and both spy tests
    below would stay green while the real load path silently regressed.
    """

    _CONNECTOR_WEIGHTS: ClassVar[dict[str, str]] = {
        "connector.video_embeddings_connector.some.weight": "CONNECTOR_TENSOR"
    }
    _PROJECTION_WEIGHTS: ClassVar[dict[str, str]] = {
        "video_aggregate_embed.weight": "VIDEO_AGG_W",
        "video_aggregate_embed.bias": "VIDEO_AGG_B",
        "audio_aggregate_embed.weight": "AUDIO_AGG_W",
        "audio_aggregate_embed.bias": "AUDIO_AGG_B",
    }

    def _stub_feature_extractor_deps(self, monkeypatch):
        """Neutralize the connector-loading half of PromptEncoder.load() so these
        tests exercise only the text-encoder selection/construction branch --
        while still recording every ``load_split_safetensors`` call (path, prefix)
        and returning prefix-appropriate fake weights, so callers can assert on
        exactly which safetensors files got merged into the connector.

        Returns ``(blocks_mod, load_calls)``: ``load_calls`` accumulates
        ``(path, prefix)`` tuples in call order.
        """
        import types

        from ltx_pipelines_mlx.utils import blocks as blocks_mod

        monkeypatch.setattr(
            blocks_mod.LTXModelConfig,
            "from_checkpoint_dir",
            classmethod(lambda cls, model_dir: types.SimpleNamespace(double_precision_rope=False)),
        )

        load_calls: list[tuple[Path, str]] = []

        def _stub_load_split_safetensors(path, prefix=""):
            load_calls.append((path, prefix))
            if prefix == "connector.":
                return dict(self._CONNECTOR_WEIGHTS)
            if prefix == "text_encoder.text_embedding_projection.":
                return dict(self._PROJECTION_WEIGHTS)
            raise AssertionError(f"unexpected load_split_safetensors call: path={path!r} prefix={prefix!r}")

        monkeypatch.setattr(blocks_mod, "load_split_safetensors", _stub_load_split_safetensors)

        class _DummyConnector:
            def __init__(self) -> None:
                self.load_weights_called_with: dict | None = None

            def load_weights(self, weights):
                self.load_weights_called_with = dict(weights)

        class _DummyFeatureExtractor:
            def __init__(self, double_precision_rope: bool) -> None:
                self.double_precision_rope = double_precision_rope
                self.connector = _DummyConnector()

        monkeypatch.setattr(blocks_mod, "GemmaFeaturesExtractorV2", _DummyFeatureExtractor)
        return blocks_mod, load_calls

    def test_gemma3_path_never_constructs_gemma4_text_encoder(self, tmp_path, monkeypatch):
        from ltx_core_mlx.text_encoders.gemma.encoders import gemma4_encoder as gemma4_encoder_mod
        from ltx_pipelines_mlx.utils.blocks import PromptEncoder

        blocks_mod, load_calls = self._stub_feature_extractor_deps(monkeypatch)

        # tmp_path has neither text_encoder.safetensors nor text_encoder_config.json
        # -> select_text_encoder(tmp_path) == "gemma3".
        select_calls = []
        real_select = blocks_mod.select_text_encoder

        def _spy_select(model_dir):
            select_calls.append(model_dir)
            return real_select(model_dir)

        monkeypatch.setattr(blocks_mod, "select_text_encoder", _spy_select)

        class _BlowUpIfConstructed:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Gemma4TextEncoder must not be constructed on the gemma3 path")

        monkeypatch.setattr(gemma4_encoder_mod, "Gemma4TextEncoder", _BlowUpIfConstructed)

        class _DummyGemma3:
            load_called_with = None

            def load(self, gemma_model_id):
                type(self).load_called_with = gemma_model_id

        monkeypatch.setattr(blocks_mod, "GemmaLanguageModel", _DummyGemma3)

        encoder = PromptEncoder(model_dir=tmp_path, gemma_model_id="mlx-community/gemma-3-12b-it-4bit")
        encoder.load()

        # (d) select_text_encoder is actually called by the construction site.
        assert select_calls == [encoder.model_dir]
        # (c) the gemma3 branch ran -- existing GemmaLanguageModel path used.
        assert isinstance(encoder._text_encoder, _DummyGemma3)
        assert _DummyGemma3.load_called_with == "mlx-community/gemma-3-12b-it-4bit"

        # F1: the gemma3 path must call load_split_safetensors exactly once,
        # for connector.safetensors only -- no text_encoder.safetensors
        # projection merge (that file doesn't even exist on this path).
        assert load_calls == [(encoder.model_dir / "connector.safetensors", "connector.")]
        assert encoder._feature_extractor.connector.load_weights_called_with == self._CONNECTOR_WEIGHTS

    def test_gemma4_path_constructs_gemma4_text_encoder(self, tmp_path, monkeypatch):
        from ltx_pipelines_mlx.utils.blocks import PromptEncoder

        blocks_mod, load_calls = self._stub_feature_extractor_deps(monkeypatch)

        (tmp_path / "text_encoder.safetensors").touch()
        _write_json(tmp_path / "text_encoder_config.json", {"gemma_version": "gemma4-12b-ltx-v1"})

        class _DummyGemma4:
            load_called_with = None

            def load(self, model_dir):
                type(self).load_called_with = model_dir

        import ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder as gemma4_encoder_mod

        monkeypatch.setattr(gemma4_encoder_mod, "Gemma4TextEncoder", _DummyGemma4)

        class _BlowUpIfConstructed:
            def __init__(self, *args, **kwargs):
                raise AssertionError("GemmaLanguageModel must not be constructed on the gemma4 path")

        monkeypatch.setattr(blocks_mod, "GemmaLanguageModel", _BlowUpIfConstructed)

        encoder = PromptEncoder(model_dir=tmp_path, gemma_model_id="mlx-community/gemma-3-12b-it-4bit")
        encoder.load()

        assert isinstance(encoder._text_encoder, _DummyGemma4)
        assert _DummyGemma4.load_called_with == encoder.model_dir

        # F1 (commit 8bc5640 pin): the gemma4 path must merge weights from
        # TWO files -- connector.safetensors AND the tower's
        # text_encoder.safetensors under the projection prefix -- in that
        # order, and hand the connector all of it re-prefixed correctly.
        assert load_calls == [
            (encoder.model_dir / "connector.safetensors", "connector."),
            (encoder.model_dir / "text_encoder.safetensors", "text_encoder.text_embedding_projection."),
        ]

        merged = encoder._feature_extractor.connector.load_weights_called_with
        assert merged is not None
        expected_projection_keys = {f"text_embedding_projection.{k}" for k in self._PROJECTION_WEIGHTS}
        assert expected_projection_keys <= merged.keys()
        for k, v in self._PROJECTION_WEIGHTS.items():
            assert merged[f"text_embedding_projection.{k}"] == v
        # Original connector.safetensors weights must survive the merge too.
        assert self._CONNECTOR_WEIGHTS.keys() <= merged.keys()


class TestEnhanceGemma4Guard:
    """``enhance`` must fail clearly, not obscurely, when pointed at a Gemma 4 pack."""

    def test_guard_raises_on_local_gemma4_pack(self, tmp_path):
        from ltx_pipelines_mlx.cli import _guard_enhance_not_gemma4

        (tmp_path / "text_encoder.safetensors").touch()
        _write_json(tmp_path / "text_encoder_config.json", {"gemma_version": "gemma4-12b-ltx-v1"})

        with pytest.raises(NotImplementedError, match=r"not supported for LTX-2\.5"):
            _guard_enhance_not_gemma4(str(tmp_path))

    def test_guard_passes_on_local_gemma3_pack(self, tmp_path):
        from ltx_pipelines_mlx.cli import _guard_enhance_not_gemma4

        _guard_enhance_not_gemma4(str(tmp_path))  # no raise

    def test_guard_passes_on_nonexistent_path_hf_id(self):
        from ltx_pipelines_mlx.cli import _guard_enhance_not_gemma4

        _guard_enhance_not_gemma4("mlx-community/gemma-3-12b-it-4bit")  # no raise, path doesn't exist locally
