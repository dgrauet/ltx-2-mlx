"""Shape tests for text encoder components."""

import mlx.core as mx

from ltx_core_mlx.text_encoders.gemma.embeddings_connector import (
    ConnectorAttention,
    ConnectorFeedForward,
    ConnectorTransformerBlock,
    Embeddings1DConnector,
)
from ltx_core_mlx.text_encoders.gemma.feature_extractor import (
    GemmaFeaturesExtractorV2,
    TextEmbeddingProjection,
    TextEncoderConnector,
)


class TestConnectorAttention:
    def test_shape(self):
        attn = ConnectorAttention(dim=64, num_heads=4, head_dim=16)
        x = mx.zeros((1, 10, 64))
        out = attn(x)
        assert out.shape == (1, 10, 64)

    def test_with_rope(self):
        attn = ConnectorAttention(dim=64, num_heads=4, head_dim=16)
        x = mx.zeros((1, 10, 64))
        # Connector RoPE is a (cos, sin, type) tuple from precompute_rope_freqs
        cos_f = mx.ones((1, 4, 10, 8))  # (B, H, N, head_dim//2)
        sin_f = mx.zeros((1, 4, 10, 8))
        out = attn(x, rope_freqs=(cos_f, sin_f, "split"))
        assert out.shape == (1, 10, 64)

    def test_weight_keys(self):
        """Verify weight key structure matches connector.safetensors format."""
        import mlx.nn as nn

        attn = ConnectorAttention(dim=64, num_heads=4, head_dim=16)
        keys = {k for k, _ in nn.utils.tree_flatten(attn.parameters())}
        expected = {
            "to_q.weight",
            "to_q.bias",
            "to_k.weight",
            "to_k.bias",
            "to_v.weight",
            "to_v.bias",
            "to_out.0.weight",
            "to_out.0.bias",  # list-wrapped
            "to_gate_logits.weight",
            "to_gate_logits.bias",
            "q_norm.weight",
            "k_norm.weight",
        }
        assert keys == expected, f"Key mismatch: {keys ^ expected}"


class TestConnectorFeedForward:
    def test_shape(self):
        ff = ConnectorFeedForward(dim=64, mult=4.0)
        x = mx.zeros((1, 10, 64))
        out = ff(x)
        assert out.shape == (1, 10, 64)

    def test_weight_keys(self):
        """Verify weight key structure: net.0.proj + net.2."""
        import mlx.nn as nn

        ff = ConnectorFeedForward(dim=64, mult=4.0)
        keys = {k for k, _ in nn.utils.tree_flatten(ff.parameters())}
        expected = {
            "net.0.proj.weight",
            "net.0.proj.bias",
            "net.2.weight",
            "net.2.bias",
        }
        assert keys == expected, f"Key mismatch: {keys ^ expected}"


class TestConnectorTransformerBlock:
    def test_shape(self):
        block = ConnectorTransformerBlock(dim=64, num_heads=4, head_dim=16)
        x = mx.zeros((1, 10, 64))
        out = block(x)
        assert out.shape == (1, 10, 64)

    def test_weight_keys(self):
        """Verify block has attn1.* and ff.* keys, no norm weights."""
        import mlx.nn as nn

        block = ConnectorTransformerBlock(dim=64, num_heads=4, head_dim=16)
        keys = {k for k, _ in nn.utils.tree_flatten(block.parameters())}
        # Should have attn1.* and ff.* but no norm.* (affine-free)
        attn_keys = {k for k in keys if k.startswith("attn1.")}
        ff_keys = {k for k in keys if k.startswith("ff.")}
        norm_keys = {k for k in keys if "norm" in k and not k.startswith("attn1.")}
        assert len(attn_keys) == 12  # 6 linear params + 2 gate + 2 qk_norm
        assert len(ff_keys) == 4  # net.0.proj.{w,b} + net.2.{w,b}
        assert len(norm_keys) == 0  # no standalone norm weights


class TestEmbeddings1DConnector:
    def test_shape_with_registers_no_mask(self):
        connector = Embeddings1DConnector(dim=64, num_heads=4, head_dim=16, num_layers=2, num_registers=8)
        hidden = mx.zeros((1, 16, 64))
        out = connector(hidden)
        # Without mask, registers are appended
        assert out.shape == (1, 16 + 8, 64)

    def test_shape_no_registers(self):
        connector = Embeddings1DConnector(dim=64, num_heads=4, head_dim=16, num_layers=2, num_registers=0)
        hidden = mx.zeros((1, 16, 64))
        out = connector(hidden)
        assert out.shape == (1, 16, 64)

    def test_weight_keys(self):
        """Verify top-level key structure: learnable_registers + transformer_1d_blocks.*"""
        import mlx.nn as nn

        connector = Embeddings1DConnector(dim=64, num_heads=4, head_dim=16, num_layers=2, num_registers=8)
        keys = {k for k, _ in nn.utils.tree_flatten(connector.parameters())}
        assert "learnable_registers" in keys
        block_keys = {k for k in keys if k.startswith("transformer_1d_blocks.")}
        assert len(block_keys) == 2 * 16  # 2 blocks * 16 params each

    def test_full_key_structure(self):
        """Verify all keys match connector.safetensors format exactly."""
        import mlx.nn as nn

        connector = Embeddings1DConnector(dim=64, num_heads=4, head_dim=16, num_layers=8, num_registers=128)
        keys = {k for k, _ in nn.utils.tree_flatten(connector.parameters())}

        # Should be 8 blocks * 16 params + 1 learnable_registers = 129 keys
        assert len(keys) == 129, f"Expected 129 keys, got {len(keys)}: {sorted(keys)}"

        # Verify structure for block 0
        block0_expected = {
            "transformer_1d_blocks.0.attn1.to_q.weight",
            "transformer_1d_blocks.0.attn1.to_q.bias",
            "transformer_1d_blocks.0.attn1.to_k.weight",
            "transformer_1d_blocks.0.attn1.to_k.bias",
            "transformer_1d_blocks.0.attn1.to_v.weight",
            "transformer_1d_blocks.0.attn1.to_v.bias",
            "transformer_1d_blocks.0.attn1.to_out.0.weight",
            "transformer_1d_blocks.0.attn1.to_out.0.bias",
            "transformer_1d_blocks.0.attn1.to_gate_logits.weight",
            "transformer_1d_blocks.0.attn1.to_gate_logits.bias",
            "transformer_1d_blocks.0.attn1.q_norm.weight",
            "transformer_1d_blocks.0.attn1.k_norm.weight",
            "transformer_1d_blocks.0.ff.net.0.proj.weight",
            "transformer_1d_blocks.0.ff.net.0.proj.bias",
            "transformer_1d_blocks.0.ff.net.2.weight",
            "transformer_1d_blocks.0.ff.net.2.bias",
        }
        assert block0_expected.issubset(keys), f"Missing block 0 keys: {block0_expected - keys}"


class TestTextEmbeddingProjection:
    def test_shape(self):
        proj = TextEmbeddingProjection(input_dim=256, video_dim=64, audio_dim=32)
        x = mx.zeros((1, 10, 256))
        video, audio = proj(x)
        assert video.shape == (1, 10, 64)
        assert audio.shape == (1, 10, 32)

    def test_weight_keys(self):
        import mlx.nn as nn

        proj = TextEmbeddingProjection(input_dim=256, video_dim=64, audio_dim=32)
        keys = {k for k, _ in nn.utils.tree_flatten(proj.parameters())}
        expected = {
            "video_aggregate_embed.weight",
            "video_aggregate_embed.bias",
            "audio_aggregate_embed.weight",
            "audio_aggregate_embed.bias",
        }
        assert keys == expected


class TestTextEncoderConnector:
    def test_weight_key_count(self):
        """Verify total key count matches connector.safetensors (262 keys)."""
        import mlx.nn as nn

        connector = TextEncoderConnector()
        keys = {k for k, _ in nn.utils.tree_flatten(connector.parameters())}
        # 4 (projection) + 129 (video) + 129 (audio) = 262
        assert len(keys) == 262, f"Expected 262 keys, got {len(keys)}"

    def test_weight_key_prefixes(self):
        """Verify top-level key prefixes match safetensors structure."""
        import mlx.nn as nn

        connector = TextEncoderConnector()
        keys = {k for k, _ in nn.utils.tree_flatten(connector.parameters())}
        prefixes = {k.split(".")[0] for k in keys}
        assert prefixes == {
            "text_embedding_projection",
            "video_embeddings_connector",
            "audio_embeddings_connector",
        }


class TestGemmaFeaturesExtractorV2:
    def test_dual_output(self):
        extractor = GemmaFeaturesExtractorV2(
            caption_channels=64,
            num_gemma_layers=4,
            video_dim=32,
            audio_dim=16,
            num_heads=4,
            video_head_dim=8,
            audio_head_dim=4,
            num_connector_layers=2,
            num_registers=8,
        )
        # 4 layers of hidden states, each (1, 10, 64)
        all_hidden = [mx.zeros((1, 10, 64)) for _ in range(4)]
        video_embeds, audio_embeds = extractor(all_hidden)
        # With registers appended (no mask), seq_len = 10 + 8 = 18
        assert video_embeds.shape == (1, 18, 32)
        assert audio_embeds.shape == (1, 18, 16)
