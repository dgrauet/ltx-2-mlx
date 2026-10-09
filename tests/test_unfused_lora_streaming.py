"""Unfused LoRAs under ``--low-ram`` block streaming (#192).

A ``BlockLoraSource(fuse=False)`` hands each block's ``A`` / ``B`` factors to run-time adapters on the
streamed shared block instead of fusing (and re-quantizing) the delta at bind. The streamed model must
then compute what a resident model with :func:`attach_loras` computes.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_core_mlx.loader.block_streaming import BlockLoraSource, BlockStreamer, StreamingLTXModel
from ltx_core_mlx.loader.lora_adapters import LoRAAdapter, attach_loras
from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

PREFIX = "transformer_blocks."


def _config(num_layers: int = 3) -> LTXModelConfig:
    return LTXModelConfig(
        num_layers=num_layers,
        video_dim=64,
        audio_dim=64,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=16,
        audio_head_dim=16,
        av_cross_num_heads=4,
        av_cross_head_dim=16,
        video_patch_channels=8,
        audio_patch_channels=8,
        ff_mult=2.0,
        timestep_embedding_dim=32,
    )


def _flat(tree, prefix: str = "") -> list[tuple[str, mx.array]]:
    out: list[tuple[str, mx.array]] = []
    if isinstance(tree, mx.array):
        out.append((prefix, tree))
    elif isinstance(tree, dict):
        for k, v in tree.items():
            out += _flat(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            out += _flat(v, f"{prefix}.{i}" if prefix else str(i))
    return out


def _model(cfg: LTXModelConfig, quantize: bool, seed: int = 0) -> LTXModel:
    mx.random.seed(seed)
    model = LTXModel(cfg)
    if quantize:
        for block in model.transformer_blocks:
            nn.quantize(block, group_size=32, bits=8, class_predicate=lambda _p, m: isinstance(m, nn.Linear))
    mx.eval(model.parameters())
    return model


# Layer paths (block-relative) the synthetic LoRA targets, with the blocks that carry each one:
# attn1.to_v is absent from block 0 and block 2 uses a smaller rank on attn1.to_q, so the streamed
# adapters must pad with zeros.
_TARGETS = {
    "attn1.to_q": {0: 4, 1: 4, 2: 2},
    "attn1.to_v": {1: 4, 2: 4},
    "ff.proj_in": {0: 4, 1: 4, 2: 4},
    "audio_attn1.to_out": {0: 4, 2: 4},
}


def _lora(model: LTXModel, path: Path, seed: int, scale: float = 0.2) -> dict[str, mx.array]:
    from ltx_core_mlx.loader.lora_adapters import _in_out_features, _resolve

    mx.random.seed(seed)
    tensors: dict[str, mx.array] = {}
    for layer_path, blocks in _TARGETS.items():
        for idx, rank in blocks.items():
            layer = _resolve(model.transformer_blocks[idx], layer_path)[2]
            in_f, out_f = _in_out_features(layer)
            tensors[f"{PREFIX}{idx}.{layer_path}.lora_A.weight"] = (mx.random.normal((rank, in_f)) * scale).astype(
                mx.bfloat16
            )
            tensors[f"{PREFIX}{idx}.{layer_path}.lora_B.weight"] = (mx.random.normal((out_f, rank)) * scale).astype(
                mx.bfloat16
            )
    mx.save_safetensors(str(path), tensors)
    return tensors


def _streamed(model: LTXModel, blocks_path: Path, sources: list[BlockLoraSource]) -> StreamingLTXModel:
    cfg = model.config
    stream_model = LTXModel(cfg)
    stream_model.transformer_blocks = [stream_model.transformer_blocks[0]]
    if isinstance(model.transformer_blocks[0].attn1.to_q, nn.QuantizedLinear):
        nn.quantize(
            stream_model.transformer_blocks[0],
            group_size=32,
            bits=8,
            class_predicate=lambda _p, m: isinstance(m, nn.Linear),
        )
    non_block = [(k, v) for k, v in _flat(model.parameters()) if not k.startswith("transformer_blocks.")]
    stream_model.load_weights(non_block, strict=False)
    streamer = BlockStreamer(blocks_path, block_prefix=PREFIX)
    streamer.bind(stream_model.transformer_blocks[0], 0)
    return StreamingLTXModel(stream_model, streamer, lora_sources=sources)


def _save_blocks(model: LTXModel, path: Path) -> None:
    out = {
        f"{PREFIX}{k[len('transformer_blocks.') :]}": v
        for k, v in _flat(model.parameters())
        if k.startswith("transformer_blocks.")
    }
    mx.save_safetensors(str(path), out)


def _inputs(cfg: LTXModelConfig) -> dict:
    mx.random.seed(7)
    return dict(
        video_latent=mx.random.normal((1, 24, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, 8, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([0.5]),
        video_text_embeds=mx.random.normal((1, 6, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, 6, cfg.audio_dim)).astype(mx.bfloat16),
    )


def _resident_unfused(model: LTXModel, tensors: dict[str, mx.array], strength: float):
    sd = StateDict(sd=dict(tensors), size=0, dtype=set())
    return attach_loras(model, [LoraStateDictWithStrength(sd, strength)])


@pytest.mark.parametrize("quantize", [False, True])
def test_streamed_unfused_matches_resident_unfused(tmp_path: Path, quantize: bool) -> None:
    cfg = _config()
    model = _model(cfg, quantize)
    blocks_path = tmp_path / "blocks.safetensors"
    _save_blocks(model, blocks_path)
    tensors = _lora(model, tmp_path / "lora.safetensors", seed=3)

    source = BlockLoraSource(tmp_path / "lora.safetensors", block_prefix=PREFIX, strength=0.7, fuse=False)
    streamed = _streamed(model, blocks_path, [source])
    inputs = _inputs(cfg)
    v_stream, a_stream = streamed(**inputs)

    _resident_unfused(model, tensors, 0.7)
    v_ref, a_ref = model(**inputs)
    mx.eval(v_stream, a_stream, v_ref, a_ref)
    # Compiled vs eager block: a few float32 ULPs.
    assert mx.allclose(v_stream, v_ref, atol=1e-4, rtol=1e-4).item()
    assert mx.allclose(a_stream, a_ref, atol=1e-4, rtol=1e-4).item()


def test_unfused_source_changes_the_output_and_clearing_it_restores_the_base(tmp_path: Path) -> None:
    cfg = _config()
    model = _model(cfg, quantize=True)
    blocks_path = tmp_path / "blocks.safetensors"
    _save_blocks(model, blocks_path)
    _lora(model, tmp_path / "lora.safetensors", seed=5)
    inputs = _inputs(cfg)

    streamed = _streamed(model, blocks_path, [])
    v_base, _ = streamed(**inputs)

    source = BlockLoraSource(tmp_path / "lora.safetensors", block_prefix=PREFIX, strength=1.0, fuse=False)
    object.__setattr__(streamed, "_lora_sources", [source])
    v_lora, _ = streamed(**inputs)
    shared = object.__getattribute__(streamed, "_shared_block")
    assert isinstance(shared.attn1.to_q, LoRAAdapter)
    assert shared.attn1.to_q["lora_a"].shape[0] == 4  # the source's largest rank for that layer

    object.__setattr__(streamed, "_lora_sources", [])
    v_cleared, _ = streamed(**inputs)
    mx.eval(v_base, v_lora, v_cleared)
    assert not isinstance(shared.attn1.to_q, LoRAAdapter)
    assert "lora_a" not in shared.attn1.to_q
    assert not mx.allclose(v_base, v_lora, atol=1e-3).item()
    assert mx.array_equal(v_base, v_cleared).item()


def test_two_unfused_sources_stack_like_resident_attach(tmp_path: Path) -> None:
    cfg = _config()
    model = _model(cfg, quantize=False)
    blocks_path = tmp_path / "blocks.safetensors"
    _save_blocks(model, blocks_path)
    t1 = _lora(model, tmp_path / "l1.safetensors", seed=11)
    t2 = _lora(model, tmp_path / "l2.safetensors", seed=12)

    sources = [
        BlockLoraSource(tmp_path / "l1.safetensors", block_prefix=PREFIX, strength=1.0, fuse=False),
        BlockLoraSource(tmp_path / "l2.safetensors", block_prefix=PREFIX, strength=0.5, fuse=False),
    ]
    streamed = _streamed(model, blocks_path, sources)
    inputs = _inputs(cfg)
    v_stream, _ = streamed(**inputs)
    shared = object.__getattribute__(streamed, "_shared_block")
    assert shared.attn1.to_q["lora_a"].shape[0] == 8  # 4 + 4, stacked

    tensors = {f"transformer_blocks.{k[len(PREFIX) :]}": v for k, v in t1.items()}
    tensors2 = {f"transformer_blocks.{k[len(PREFIX) :]}": v for k, v in t2.items()}
    attach_loras(
        model,
        [
            LoraStateDictWithStrength(StateDict(sd=tensors, size=0, dtype=set()), 1.0),
            LoraStateDictWithStrength(StateDict(sd=tensors2, size=0, dtype=set()), 0.5),
        ],
    )
    v_ref, _ = model(**inputs)
    mx.eval(v_stream, v_ref)
    assert mx.allclose(v_stream, v_ref, atol=1e-4, rtol=1e-4).item()


def test_fused_and_unfused_sources_mix(tmp_path: Path) -> None:
    """A fused source (the distilled LoRA) keeps fusing at bind next to an unfused one."""
    cfg = _config()
    model = _model(cfg, quantize=False)
    blocks_path = tmp_path / "blocks.safetensors"
    _save_blocks(model, blocks_path)
    t_fused = _lora(model, tmp_path / "fused.safetensors", seed=21)
    t_unfused = _lora(model, tmp_path / "unfused.safetensors", seed=22)
    sources = [
        BlockLoraSource(tmp_path / "fused.safetensors", block_prefix=PREFIX, strength=1.0, fuse=True),
        BlockLoraSource(tmp_path / "unfused.safetensors", block_prefix=PREFIX, strength=1.0, fuse=False),
    ]
    streamed = _streamed(model, blocks_path, sources)
    inputs = _inputs(cfg)
    v_stream, _ = streamed(**inputs)

    from ltx_core_mlx.loader.fuse_loras import apply_loras

    flat = dict(_flat(model.parameters()))
    fused_sd = apply_loras(
        StateDict(sd=flat, size=0, dtype=set()),
        [
            LoraStateDictWithStrength(
                StateDict(
                    sd={f"transformer_blocks.{k[len(PREFIX) :]}": v for k, v in t_fused.items()}, size=0, dtype=set()
                ),
                1.0,
            )
        ],
    )
    model.load_weights(list(fused_sd.sd.items()), strict=False)
    _resident_unfused(model, {f"transformer_blocks.{k[len(PREFIX) :]}": v for k, v in t_unfused.items()}, 1.0)
    v_ref, _ = model(**inputs)
    mx.eval(v_stream, v_ref)
    assert mx.allclose(v_stream, v_ref, atol=1e-3, rtol=1e-3).item()


def test_compute_dtype_casts_the_streamed_factors(tmp_path: Path) -> None:
    cfg = _config()
    model = _model(cfg, quantize=True)
    blocks_path = tmp_path / "blocks.safetensors"
    _save_blocks(model, blocks_path)
    _lora(model, tmp_path / "lora.safetensors", seed=31)
    source = BlockLoraSource(tmp_path / "lora.safetensors", block_prefix=PREFIX, strength=1.0, fuse=False)
    streamed = _streamed(model, blocks_path, [source])
    streamed.set_compute_dtype(mx.float16)
    v, _ = streamed(**_inputs(cfg))
    mx.eval(v)
    shared = object.__getattribute__(streamed, "_shared_block")
    assert shared.attn1.to_q["lora_a"].dtype == mx.float16
    assert shared.ff.proj_in["lora_b"].dtype == mx.float16
    assert mx.all(mx.isfinite(v)).item()


def test_source_ranks_and_factors(tmp_path: Path) -> None:
    cfg = _config()
    model = _model(cfg, quantize=False)
    _lora(model, tmp_path / "lora.safetensors", seed=41)
    source = BlockLoraSource(tmp_path / "lora.safetensors", block_prefix=PREFIX, fuse=False)
    assert source.ranks() == {"attn1.to_q": 4, "attn1.to_v": 4, "ff.proj_in": 4, "audio_attn1.to_out": 4}
    assert set(source.get_block_factors(0)) == {"attn1.to_q", "ff.proj_in", "audio_attn1.to_out"}
    assert source.get_block_factors(2)["attn1.to_q"][0].shape[0] == 2
    assert source.factor_dtype() == mx.bfloat16
    assert BlockLoraSource(tmp_path / "lora.safetensors", block_prefix=PREFIX).fuse is True
