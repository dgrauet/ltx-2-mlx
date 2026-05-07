"""Block streaming smoke + correctness tests.

Verifies BlockStreamer can rebind a shared block to weights pulled from
a memory-mapped safetensors, and that LTXModel's ``block_provider``
hook gives the same forward output as the regular ``transformer_blocks``
iteration when fed identical weights.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.loader.block_streaming import BlockStreamer
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock


def _tiny_config(num_layers: int = 2) -> LTXModelConfig:
    return LTXModelConfig(
        num_layers=num_layers,
        video_dim=32,
        audio_dim=16,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=4,
        av_cross_head_dim=4,
        video_patch_channels=8,
        audio_patch_channels=8,
        ff_mult=2.0,
        timestep_embedding_dim=32,
    )


def _flatten_params(module) -> list[tuple[str, mx.array]]:
    flat: list[tuple[str, mx.array]] = []

    def _rec(node, p: str) -> None:
        if isinstance(node, mx.array):
            flat.append((p, node))
        elif isinstance(node, dict):
            for k, v in node.items():
                _rec(v, f"{p}.{k}" if p else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _rec(v, f"{p}.{i}" if p else str(i))

    _rec(module.parameters(), "")
    return flat


def _save_block_weights_to_safetensors(model: LTXModel, path: Path, prefix: str) -> None:
    out: dict[str, mx.array] = {}
    for i, block in enumerate(model.transformer_blocks):
        for k, v in _flatten_params(block):
            out[f"{prefix}{i}.{k}"] = v
    mx.save_safetensors(str(path), out)


def _make_block(cfg: LTXModelConfig) -> BasicAVTransformerBlock:
    return BasicAVTransformerBlock(
        video_dim=cfg.video_dim,
        audio_dim=cfg.audio_dim,
        video_num_heads=cfg.video_num_heads,
        audio_num_heads=cfg.audio_num_heads,
        video_head_dim=cfg.video_head_dim,
        audio_head_dim=cfg.audio_head_dim,
        av_cross_num_heads=cfg.av_cross_num_heads,
        av_cross_head_dim=cfg.av_cross_head_dim,
        ff_mult=cfg.ff_mult,
        norm_eps=cfg.norm_eps,
    )


class TestBlockStreamer:
    def test_constructor_and_block_count(self):
        cfg = _tiny_config(num_layers=3)
        model = LTXModel(cfg)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blocks.safetensors"
            _save_block_weights_to_safetensors(model, path, prefix="transformer_blocks.")
            streamer = BlockStreamer(path, block_prefix="transformer_blocks.")
            assert streamer.block_count == 3
            sample_keys = streamer.block_keys(0)
            assert any("attn1" in k for k in sample_keys)
            streamer.close()

    def test_bind_loads_weights(self):
        cfg = _tiny_config(num_layers=2)
        ref_model = LTXModel(cfg)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blocks.safetensors"
            _save_block_weights_to_safetensors(ref_model, path, prefix="transformer_blocks.")
            streamer = BlockStreamer(path, block_prefix="transformer_blocks.")

            shared = _make_block(cfg)
            streamer.bind(shared, idx=1)

            ref_w = ref_model.transformer_blocks[1].attn1.to_q.weight
            shared_w = shared.attn1.to_q.weight
            mx.eval(ref_w, shared_w)
            assert mx.array_equal(ref_w, shared_w).item()
            streamer.close()


class TestBlockProvider:
    def _build_inputs(self, cfg: LTXModelConfig):
        B, Nv, Na, Nt = 1, 16, 8, 4
        return dict(
            video_latent=mx.random.normal((B, Nv, cfg.video_patch_channels)).astype(mx.bfloat16),
            audio_latent=mx.random.normal((B, Na, cfg.audio_patch_channels)).astype(mx.bfloat16),
            timestep=mx.array([0.5]),
            video_text_embeds=mx.random.normal((B, Nt, cfg.video_dim)).astype(mx.bfloat16),
            audio_text_embeds=mx.random.normal((B, Nt, cfg.audio_dim)).astype(mx.bfloat16),
        )

    def test_block_provider_matches_baseline(self):
        cfg = _tiny_config(num_layers=2)
        mx.random.seed(42)
        model = LTXModel(cfg)
        mx.eval(model.parameters())
        common = self._build_inputs(cfg)
        baseline_v, baseline_a = model(**common)
        provided_v, provided_a = model(**common, block_provider=lambda i: model.transformer_blocks[i])
        mx.eval(baseline_v, baseline_a, provided_v, provided_a)
        assert mx.array_equal(baseline_v, provided_v).item()
        assert mx.array_equal(baseline_a, provided_a).item()

    def test_streaming_matches_baseline(self):
        cfg = _tiny_config(num_layers=2)
        mx.random.seed(42)
        model = LTXModel(cfg)
        mx.eval(model.parameters())
        common = self._build_inputs(cfg)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blocks.safetensors"
            _save_block_weights_to_safetensors(model, path, prefix="transformer_blocks.")
            streamer = BlockStreamer(path, block_prefix="transformer_blocks.")
            shared = _make_block(cfg)

            def provider(idx: int):
                streamer.bind(shared, idx)
                return shared

            baseline_v, baseline_a = model(**common)
            streamed_v, streamed_a = model(**common, block_provider=provider)
            mx.eval(baseline_v, baseline_a, streamed_v, streamed_a)
            assert mx.array_equal(baseline_v, streamed_v).item()
            assert mx.array_equal(baseline_a, streamed_a).item()
            streamer.close()

    def test_eviction_then_reload_keeps_outputs_bit_exact(self):
        """Evicting all blocks across one forward + auto-reload on the
        next forward must yield the same outputs (proves multi-step
        inference works without leaking memory between calls)."""
        cfg = _tiny_config(num_layers=3)
        mx.random.seed(42)
        model = LTXModel(cfg)
        mx.eval(model.parameters())
        common = self._build_inputs(cfg)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blocks.safetensors"
            _save_block_weights_to_safetensors(model, path, prefix="transformer_blocks.")
            streamer = BlockStreamer(path, block_prefix="transformer_blocks.")
            shared = _make_block(cfg)

            def make_evicting_provider():
                prev = [None]

                def provider(idx: int):
                    streamer.bind(shared, idx, evict_previous=prev[0])
                    prev[0] = idx
                    return shared

                return provider

            run1_v, run1_a = model(**common, block_provider=make_evicting_provider())
            run2_v, run2_a = model(**common, block_provider=make_evicting_provider())
            mx.eval(run1_v, run1_a, run2_v, run2_a)
            assert mx.array_equal(run1_v, run2_v).item()
            assert mx.array_equal(run1_a, run2_a).item()
            streamer.close()
