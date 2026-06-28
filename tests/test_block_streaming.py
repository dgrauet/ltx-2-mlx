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


class TestStreamingLTXModel:
    def test_wrapper_matches_baseline(self):
        """StreamingLTXModel(model, streamer) produces same output as model(...)."""
        from ltx_core_mlx.loader.block_streaming import StreamingLTXModel

        cfg = _tiny_config(num_layers=3)
        mx.random.seed(42)
        model = LTXModel(cfg)
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blocks.safetensors"
            _save_block_weights_to_safetensors(model, path, prefix="transformer_blocks.")
            streamer = BlockStreamer(path, block_prefix="transformer_blocks.")

            # Build the streaming model:
            # truncate transformer_blocks to only the first one,
            # then wrap. The wrapper's __call__ feeds the block_provider.
            stream_model = LTXModel(cfg)
            mx.eval(stream_model.parameters())  # init weights
            # Drop blocks 1+ from the streaming model, then bind block 0 from disk.
            stream_model.transformer_blocks = [stream_model.transformer_blocks[0]]
            streamer.bind(stream_model.transformer_blocks[0], idx=0)
            wrapped = StreamingLTXModel(stream_model, streamer)

            # Copy non-block weights (input proj, AdaLN, output, etc.) from the
            # reference model into the streaming model so non-block forward
            # paths match. We use load_weights with strict=False on the
            # whole reference parameter tree minus block keys.
            non_block_params = []

            def _walk(node, path: str):
                if isinstance(node, mx.array):
                    if not path.startswith("transformer_blocks."):
                        non_block_params.append((path, node))
                elif isinstance(node, dict):
                    for k, v in node.items():
                        _walk(v, f"{path}.{k}" if path else k)
                elif isinstance(node, list):
                    for i, v in enumerate(node):
                        _walk(v, f"{path}.{i}" if path else str(i))

            _walk(model.parameters(), "")
            stream_model.load_weights(non_block_params, strict=False)

            # Inputs
            B, Nv, Na, Nt = 1, 16, 8, 4
            common = dict(
                video_latent=mx.random.normal((B, Nv, cfg.video_patch_channels)).astype(mx.bfloat16),
                audio_latent=mx.random.normal((B, Na, cfg.audio_patch_channels)).astype(mx.bfloat16),
                timestep=mx.array([0.5]),
                video_text_embeds=mx.random.normal((B, Nt, cfg.video_dim)).astype(mx.bfloat16),
                audio_text_embeds=mx.random.normal((B, Nt, cfg.audio_dim)).astype(mx.bfloat16),
            )

            baseline_v, baseline_a = model(**common)
            streamed_v, streamed_a = wrapped(**common)
            mx.eval(baseline_v, baseline_a, streamed_v, streamed_a)
            # mx.compile fuses kernels, so the streamed output differs
            # from the eager baseline by ~1 fp32 ULP.
            assert mx.allclose(baseline_v, streamed_v, atol=1e-5, rtol=1e-5).item()
            assert mx.allclose(baseline_a, streamed_a, atol=1e-5, rtol=1e-5).item()
            streamer.close()


class TestBlockLoraSource:
    def _build_synthetic_lora(self, model: LTXModel, path: Path, prefix: str) -> None:
        """Save a minimal A+B pair targeting block 0's attn1.to_q.weight."""
        block0 = model.transformer_blocks[0]
        out_dim, in_dim = block0.attn1.to_q.weight.shape
        rank = 4
        a = mx.random.normal((rank, in_dim)).astype(mx.bfloat16)
        b = mx.random.normal((out_dim, rank)).astype(mx.bfloat16) * 0.1
        out: dict[str, mx.array] = {
            f"{prefix}0.attn1.to_q.lora_A.weight": a,
            f"{prefix}0.attn1.to_q.lora_B.weight": b,
        }
        mx.save_safetensors(str(path), out)

    def test_bind_with_lora_fuses_delta(self):
        from ltx_core_mlx.loader.block_streaming import BlockLoraSource

        cfg = _tiny_config(num_layers=2)
        ref_model = LTXModel(cfg)
        mx.eval(ref_model.parameters())

        with tempfile.TemporaryDirectory() as td:
            blocks_path = Path(td) / "blocks.safetensors"
            lora_path = Path(td) / "lora.safetensors"
            _save_block_weights_to_safetensors(ref_model, blocks_path, prefix="transformer_blocks.")
            self._build_synthetic_lora(ref_model, lora_path, prefix="transformer_blocks.")

            streamer = BlockStreamer(blocks_path, block_prefix="transformer_blocks.")
            lora = BlockLoraSource(lora_path, block_prefix="transformer_blocks.", strength=1.0)
            assert lora.has_block(0)
            assert not lora.has_block(1)

            shared = _make_block(cfg)

            # Expected: original + delta = (B @ A) at attn1.to_q.weight.
            expected_q = (
                ref_model.transformer_blocks[0].attn1.to_q.weight.astype(mx.float32)
                + (
                    lora.get_block_lora_dict(0)["attn1.to_q.lora_B.weight"].astype(mx.float32)
                    @ lora.get_block_lora_dict(0)["attn1.to_q.lora_A.weight"].astype(mx.float32)
                )
            ).astype(ref_model.transformer_blocks[0].attn1.to_q.weight.dtype)

            streamer.bind(shared, 0, lora_sources=[lora])
            mx.eval(shared.attn1.to_q.weight, expected_q)
            assert mx.allclose(shared.attn1.to_q.weight, expected_q, atol=1e-5, rtol=1e-5).item()

            # Block 1 has no LoRA → bind should leave weights unchanged.
            streamer.bind(shared, 1, evict_previous=0, lora_sources=[lora])
            ref_q1 = ref_model.transformer_blocks[1].attn1.to_q.weight
            mx.eval(shared.attn1.to_q.weight, ref_q1)
            assert mx.array_equal(shared.attn1.to_q.weight, ref_q1).item()
            streamer.close()
            lora.close()

    def test_comfy_renamed_lora_matches_pipeline_prefix(self):
        """Regression for #52: a ComfyUI/diffusers-named LoRA remapped via
        ``LTXV_LORA_COMFY_RENAMING_MAP`` must match blocks under the prefix
        the pipelines hand to ``BlockLoraSource``.

        The map strips ``diffusion_model.`` but never adds ``transformer.``,
        so the streaming call sites must use ``LTXV_LORA_BLOCK_PREFIX``
        (``"transformer_blocks."``). Passing ``"transformer.transformer_blocks."``
        silently drops every delta — output is byte-identical to no-LoRA.
        """
        from ltx_core_mlx.loader.block_streaming import BlockLoraSource
        from ltx_core_mlx.loader.sd_ops import (
            LTXV_LORA_BLOCK_PREFIX,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )

        cfg = _tiny_config(num_layers=2)
        ref_model = LTXModel(cfg)
        mx.eval(ref_model.parameters())

        with tempfile.TemporaryDirectory() as td:
            lora_path = Path(td) / "comfy_lora.safetensors"
            # ComfyUI/diffusers naming, as a trainer- or Lightricks-produced
            # LoRA actually ships it on disk.
            self._build_synthetic_lora(ref_model, lora_path, prefix="diffusion_model.transformer_blocks.")

            # The map output namespace must equal the prefix constant the
            # pipelines pass. This is the single source of truth the three
            # call sites (_base, ic_lora, ti2vid_two_stages) reference.
            renamed = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(
                "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"
            )
            assert renamed.startswith(LTXV_LORA_BLOCK_PREFIX)

            lora = BlockLoraSource(
                lora_path,
                block_prefix=LTXV_LORA_BLOCK_PREFIX,
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
            assert lora.has_block(0), (
                "ComfyUI-renamed LoRA produced zero matched deltas — the streaming LoRA path is silently a no-op (#52)."
            )
            lora.close()
