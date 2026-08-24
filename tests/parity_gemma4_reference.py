"""Torch reference harness for the Gemma-4 attention/rotary/norm/MLP parity test.

This is a standalone script, NOT a pytest module: it needs torch +
transformers, which are not dependencies of this project. Run it in a
disposable environment to produce the reference npz that
``tests/test_ltx25_gemma4.py`` compares the MLX blocks against::

    uv run --no-project --with "transformers>=5.10,<6" --with torch --with numpy \
        python tests/parity_gemma4_reference.py --out /tmp/gemma4_parity.npz

The npz is deliberately NOT committed (it is regenerable and version
dependent); the parity tests skip when it is absent.

It builds a tiny fixed-seed ``Gemma4UnifiedTextModel`` (2 layers:
``[sliding_attention, full_attention]``) on CPU in float32, runs one
forward pass, and dumps:

* ``w.<state_dict key>`` -- every model weight;
* ``input_ids`` / ``attention_mask`` -- the inputs used;
* ``hs.<i>`` -- hidden states entering layer ``i`` (``hs.0`` are the
  scaled token embeddings), ``hs.<n_layers>`` the last layer's output and
  ``hs.final`` the output of the trailing RMSNorm;
* ``rope.<layer_type>.{cos,sin}`` -- the per-layer-type rotary tables;
* ``mask.<layer_type>`` -- the additive attention masks;
* ``L<i>.{attn_in,attn_out,mlp_in,mlp_out}`` -- per-layer submodule
  boundaries, so the MLX blocks can be checked in isolation.
* ``padded.hs.<i>`` / ``padded.hs.final`` / ``padded.input_ids`` /
  ``padded.attention_mask`` -- a second forward pass over a batch of 2,
  row 0 left-padded by 3 tokens, exercising the ``padding_mask`` path
  through ``build_attention_mask`` (unexercised by the all-ones batch=1
  case above).
* ``tokenizer.control_ids`` -- token ids for ``CONTROL_SENTENCE``,
  produced by ``transformers.AutoTokenizer`` loaded from the real
  LTX-2.5 pack's ``tokenizer.json`` (not the tiny reference model above --
  this exercises the *real* Gemma tokenizer, independent of the tiny
  attention/rotary config). Only written when ``PACK_DIR`` exists locally;
  ``tests/test_ltx25_gemma4.py`` skips the tokenizer parity test when this
  key is absent from the npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextModel

SEED = 1234

# Local (non-hub) pack, same convention as tests/conftest.py::_local_pack.
# Only used for the tokenizer control-sentence dump below -- unrelated to
# the tiny reference model's config/weights.
PACK_DIR = Path.home() / "Work/mlx/models/ltx-2.5-mlx-q8"
CONTROL_SENTENCE = "A lone lighthouse keeper watches the storm roll in over the grey harbor."

# Fixed tiny config, mirroring the real pack's *shape* (two attention
# flavors, k_eq_v on the full layers, proportional partial rope on the
# global layers) at a size that runs in a second on CPU.
TINY_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "global_head_dim": 32,
    "num_global_key_value_heads": 1,
    "sliding_window": 8,
    "layer_types": ["sliding_attention", "full_attention"],
    "attention_k_eq_v": True,
    "num_kv_shared_layers": 0,
    "use_double_wide_mlp": False,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 512,
    "pad_token_id": 0,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    },
}

SEQ_LEN = 12
BATCH = 1


def build_config() -> Gemma4UnifiedTextConfig:
    """Instantiate the fixed tiny reference config."""
    return Gemma4UnifiedTextConfig(attn_implementation="eager", **TINY_CONFIG)


def main() -> None:
    """Run the reference forward pass and save the npz."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Destination .npz path")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    config = build_config()
    model = Gemma4UnifiedTextModel(config).to(torch.float32).eval()

    # Randomize every parameter: the default init leaves norms at ones and
    # would hide scale bugs.
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=torch.float32) * 0.1)

    input_ids = torch.randint(0, TINY_CONFIG["vocab_size"], (BATCH, SEQ_LEN), generator=generator)
    attention_mask = torch.ones((BATCH, SEQ_LEN), dtype=torch.long)

    out: dict[str, np.ndarray] = {}

    def store(name: str, tensor: torch.Tensor) -> None:
        out[name] = tensor.detach().to(torch.float32).cpu().numpy()

    handles = []

    for idx, layer in enumerate(model.layers):
        handles.append(
            layer.register_forward_pre_hook(
                lambda _m, inputs, i=idx: store(f"hs.{i}", inputs[0]),
            )
        )
        handles.append(
            layer.register_forward_hook(
                lambda _m, _inputs, output, i=idx: store(
                    f"hs.{i + 1}", output[0] if isinstance(output, tuple) else output
                ),
            )
        )
        handles.append(
            layer.self_attn.register_forward_pre_hook(
                lambda _m, _inputs, kwargs, i=idx: store(f"L{i}.attn_in", kwargs["hidden_states"]),
                with_kwargs=True,
            )
        )
        handles.append(
            layer.self_attn.register_forward_hook(
                lambda _m, _inputs, output, i=idx: store(f"L{i}.attn_out", output[0]),
            )
        )
        handles.append(
            layer.mlp.register_forward_pre_hook(
                lambda _m, inputs, i=idx: store(f"L{i}.mlp_in", inputs[0]),
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                lambda _m, _inputs, output, i=idx: store(f"L{i}.mlp_out", output),
            )
        )

    with torch.no_grad():
        result = model(input_ids=input_ids, attention_mask=attention_mask)

    for handle in handles:
        handle.remove()

    store("hs.final", result.last_hidden_state)

    # Padded-batch case (batch=2, row 0 left-padded by PAD_LEN tokens): the
    # padding_mask path through build_attention_mask is unexercised by the
    # batch=1 case above, since attention_mask there is all-ones. Reuses the
    # same per-layer hook mechanism, under a "padded." prefix.
    PAD_LEN = 3
    padded_input_ids = torch.randint(0, TINY_CONFIG["vocab_size"], (2, SEQ_LEN), generator=generator)
    padded_input_ids[0, :PAD_LEN] = TINY_CONFIG["pad_token_id"]
    padded_attention_mask = torch.ones((2, SEQ_LEN), dtype=torch.long)
    padded_attention_mask[0, :PAD_LEN] = 0

    padded_handles = []
    for idx, layer in enumerate(model.layers):
        padded_handles.append(
            layer.register_forward_pre_hook(
                lambda _m, inputs, i=idx: store(f"padded.hs.{i}", inputs[0]),
            )
        )
        padded_handles.append(
            layer.register_forward_hook(
                lambda _m, _inputs, output, i=idx: store(
                    f"padded.hs.{i + 1}", output[0] if isinstance(output, tuple) else output
                ),
            )
        )

    with torch.no_grad():
        padded_result = model(input_ids=padded_input_ids, attention_mask=padded_attention_mask)

    for handle in padded_handles:
        handle.remove()

    store("padded.hs.final", padded_result.last_hidden_state)
    out["padded.input_ids"] = padded_input_ids.numpy()
    out["padded.attention_mask"] = padded_attention_mask.numpy()

    # Rotary tables + masks, recomputed the same way the model does.
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    embeds = model.embed_tokens(input_ids)
    for layer_type in sorted(set(TINY_CONFIG["layer_types"])):
        cos, sin = model.rotary_emb(embeds, position_ids, layer_type)
        store(f"rope.{layer_type}.cos", cos)
        store(f"rope.{layer_type}.sin", sin)
        store(f"rope.{layer_type}.inv_freq", getattr(model.rotary_emb, f"{layer_type}_inv_freq"))

    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    mask_kwargs = {
        "config": config,
        "inputs_embeds": embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    store("mask.full_attention", create_causal_mask(**mask_kwargs))
    store("mask.sliding_attention", create_sliding_window_causal_mask(**mask_kwargs))

    for key, value in model.state_dict().items():
        store(f"w.{key}", value)

    out["input_ids"] = input_ids.numpy()
    out["attention_mask"] = attention_mask.numpy()

    # Real-pack tokenizer control sentence (independent of the tiny
    # reference model above): proves our Gemma4TextEncoder.tokenize()
    # produces identical ids to transformers.AutoTokenizer loaded from the
    # same pack. Skipped when the pack is not present on this machine --
    # the pytest side skips the comparison when this key is absent.
    if PACK_DIR.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(PACK_DIR))
        control_ids = tokenizer.encode(CONTROL_SENTENCE.strip())
        out["tokenizer.control_ids"] = np.array(control_ids, dtype=np.int64)
        print(f"tokenizer control ids: {len(control_ids)} tokens (pack at {PACK_DIR})")
    else:
        print(f"pack dir {PACK_DIR} not found -- skipping tokenizer control-sentence dump")

    np.savez(args.out, **out)
    print(f"wrote {args.out} with {len(out)} arrays")
    print(
        f"last_hidden_state: shape={tuple(result.last_hidden_state.shape)} mean={result.last_hidden_state.mean():.6f}"
    )


if __name__ == "__main__":
    main()
