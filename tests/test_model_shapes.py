"""Shape tests for the DiT model with small config."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.feed_forward import FeedForward
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_core_mlx.model.transformer.rope import apply_rope_split, get_frequencies, get_positional_embedding
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock


class TestRoPE:
    def test_get_frequencies_shape(self):
        positions = mx.arange(16)
        freqs = get_frequencies(positions, 32)
        assert freqs.shape == (16, 16)

    def test_apply_rope_split(self):
        x = mx.ones((2, 4, 8, 32))
        cos_f = mx.ones((2, 1, 8, 16))
        sin_f = mx.zeros((2, 1, 8, 16))
        out = apply_rope_split(x, cos_f, sin_f)
        assert out.shape == x.shape

    def test_positional_embedding(self):
        positions = mx.zeros((4, 3))  # 4 positions, 3 axes
        emb = get_positional_embedding(positions, 96)
        assert emb.shape == (4, 96)


class TestFeedForward:
    def test_shape(self):
        ff = FeedForward(dim=32, mult=4.0)
        x = mx.zeros((2, 8, 32))
        out = ff(x)
        mx.synchronize()
        assert out.shape == (2, 8, 32)

    def test_key_names(self):
        ff = FeedForward(dim=32, mult=4.0)
        keys = {k for k, _ in ff.parameters().items()}
        assert "proj_in" in keys
        assert "proj_out" in keys


class TestTimestepEmbedding:
    def test_sinusoidal(self):
        t = mx.array([0.5, 1.0])
        emb = get_timestep_embedding(t, 64)
        assert emb.shape == (2, 64)

    def test_adaln_single(self):
        adaln = AdaLayerNormSingle(32, num_params=9)
        t = mx.zeros((2, 32))
        params, embedded = adaln(t)
        mx.synchronize()
        assert params.shape == (2, 9 * 32)
        assert embedded.shape == (2, 32)

    def test_adaln_key_names(self):
        adaln = AdaLayerNormSingle(32, num_params=9)
        leaf_keys = set()
        for k, _ in nn.utils.tree_flatten(adaln.trainable_parameters()):
            leaf_keys.add(k)
        # Must match: emb.timestep_embedder.linear1.weight, ...linear2.weight, linear.weight
        assert any("emb.timestep_embedder.linear1.weight" in k for k in leaf_keys)
        assert any("emb.timestep_embedder.linear2.weight" in k for k in leaf_keys)
        assert any("linear.weight" in k for k in leaf_keys)


class TestAttention:
    def test_self_attention(self):
        attn = Attention(query_dim=32, num_heads=4, head_dim=8)
        x = mx.zeros((2, 8, 32))
        out = attn(x)
        mx.synchronize()
        assert out.shape == (2, 8, 32)

    def test_cross_attention(self):
        attn = Attention(query_dim=32, kv_dim=16, num_heads=4, head_dim=8, use_rope=False)
        x = mx.zeros((2, 8, 32))
        ctx = mx.zeros((2, 4, 16))
        out = attn(x, encoder_hidden_states=ctx)
        mx.synchronize()
        assert out.shape == (2, 8, 32)

    def test_cross_attention_different_out_dim(self):
        attn = Attention(query_dim=32, kv_dim=16, out_dim=64, num_heads=4, head_dim=8, use_rope=False)
        x = mx.zeros((2, 8, 32))
        ctx = mx.zeros((2, 4, 16))
        out = attn(x, encoder_hidden_states=ctx)
        mx.synchronize()
        assert out.shape == (2, 8, 64)

    def test_key_names(self):
        attn = Attention(query_dim=32, num_heads=4, head_dim=8)
        leaf_keys = {k for k, _ in nn.utils.tree_flatten(attn.parameters())}
        assert any("to_q.weight" in k for k in leaf_keys)
        assert any("to_k.weight" in k for k in leaf_keys)
        assert any("to_v.weight" in k for k in leaf_keys)
        assert any("to_out.weight" in k for k in leaf_keys)
        assert any("to_gate_logits.weight" in k for k in leaf_keys)
        assert any("q_norm.weight" in k for k in leaf_keys)
        assert any("k_norm.weight" in k for k in leaf_keys)


class TestTransformerBlock:
    def test_shape(self):
        block = BasicAVTransformerBlock(
            video_dim=32,
            audio_dim=16,
            video_num_heads=4,
            audio_num_heads=4,
            video_head_dim=8,
            audio_head_dim=4,
            av_cross_num_heads=4,
            av_cross_head_dim=4,
        )
        B = 1
        video = mx.zeros((B, 8, 32))
        audio = mx.zeros((B, 4, 16))
        v_adaln = mx.zeros((B, 9 * 32))
        a_adaln = mx.zeros((B, 9 * 16))
        v_prompt = mx.zeros((B, 2 * 32))
        a_prompt = mx.zeros((B, 2 * 16))
        av_v = mx.zeros((B, 4 * 32))
        av_a = mx.zeros((B, 4 * 16))
        a2v_gate = mx.zeros((B, 32))
        v2a_gate = mx.zeros((B, 16))
        v_text = mx.zeros((B, 3, 32))
        a_text = mx.zeros((B, 3, 16))

        v_out, a_out = block(
            video,
            audio,
            v_adaln,
            a_adaln,
            v_prompt,
            a_prompt,
            av_v,
            av_a,
            a2v_gate,
            v2a_gate,
            video_text_embeds=v_text,
            audio_text_embeds=a_text,
        )
        mx.synchronize()
        assert v_out.shape == (1, 8, 32)
        assert a_out.shape == (1, 4, 16)

    def test_key_names(self):
        block = BasicAVTransformerBlock(
            video_dim=32,
            audio_dim=16,
            video_num_heads=4,
            audio_num_heads=4,
            video_head_dim=8,
            audio_head_dim=4,
            av_cross_num_heads=4,
            av_cross_head_dim=4,
        )
        leaf_keys = {k for k, _ in nn.utils.tree_flatten(block.parameters())}
        # Check key sub-module names
        assert any(k.startswith("attn1.") for k in leaf_keys)
        assert any(k.startswith("audio_attn1.") for k in leaf_keys)
        assert any(k.startswith("attn2.") for k in leaf_keys)
        assert any(k.startswith("audio_attn2.") for k in leaf_keys)
        assert any(k.startswith("audio_to_video_attn.") for k in leaf_keys)
        assert any(k.startswith("video_to_audio_attn.") for k in leaf_keys)
        assert any(k.startswith("ff.") for k in leaf_keys)
        assert any(k.startswith("audio_ff.") for k in leaf_keys)
        assert "scale_shift_table" in leaf_keys
        assert "audio_scale_shift_table" in leaf_keys
        assert "prompt_scale_shift_table" in leaf_keys
        assert "audio_prompt_scale_shift_table" in leaf_keys
        assert "scale_shift_table_a2v_ca_video" in leaf_keys
        assert "scale_shift_table_a2v_ca_audio" in leaf_keys


class TestLTXModel:
    @pytest.fixture()
    def small_config(self):
        return LTXModelConfig(
            num_layers=2,
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

    def test_forward_shape(self, small_config):
        model = LTXModel(small_config)
        B, Nv, Na, Nt = 1, 16, 8, 4

        video_out, audio_out = model(
            video_latent=mx.zeros((B, Nv, 8)),
            audio_latent=mx.zeros((B, Na, 8)),
            timestep=mx.array([0.5]),
            video_text_embeds=mx.zeros((B, Nt, 32)),
            audio_text_embeds=mx.zeros((B, Nt, 16)),
        )
        mx.synchronize()
        assert video_out.shape == (B, Nv, 8)
        assert audio_out.shape == (B, Na, 8)

    def test_x0_model(self, small_config):
        model = X0Model(LTXModel(small_config))
        B, Nv, Na, Nt = 1, 16, 8, 4

        v_x0, a_x0 = model(
            video_latent=mx.zeros((B, Nv, 8)),
            audio_latent=mx.zeros((B, Na, 8)),
            sigma=mx.array([0.5]),
            video_text_embeds=mx.zeros((B, Nt, 32)),
            audio_text_embeds=mx.zeros((B, Nt, 16)),
        )
        mx.synchronize()
        assert v_x0.shape == (B, Nv, 8)
        assert a_x0.shape == (B, Na, 8)

    def test_top_level_key_names(self, small_config):
        model = LTXModel(small_config)
        leaf_keys = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
        # Check top-level module names match weight file
        assert any(k.startswith("adaln_single.") for k in leaf_keys)
        assert any(k.startswith("audio_adaln_single.") for k in leaf_keys)
        assert any(k.startswith("prompt_adaln_single.") for k in leaf_keys)
        assert any(k.startswith("audio_prompt_adaln_single.") for k in leaf_keys)
        assert any(k.startswith("av_ca_video_scale_shift_adaln_single.") for k in leaf_keys)
        assert any(k.startswith("av_ca_audio_scale_shift_adaln_single.") for k in leaf_keys)
        assert any(k.startswith("av_ca_a2v_gate_adaln_single.") for k in leaf_keys)
        assert any(k.startswith("av_ca_v2a_gate_adaln_single.") for k in leaf_keys)
        assert any(k.startswith("patchify_proj.") for k in leaf_keys)
        assert any(k.startswith("audio_patchify_proj.") for k in leaf_keys)
        assert any(k.startswith("proj_out.") for k in leaf_keys)
        assert any(k.startswith("audio_proj_out.") for k in leaf_keys)
        assert "scale_shift_table" in leaf_keys
        assert "audio_scale_shift_table" in leaf_keys
        assert any(k.startswith("transformer_blocks.") for k in leaf_keys)
