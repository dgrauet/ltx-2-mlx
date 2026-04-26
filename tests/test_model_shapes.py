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


class TestBlockGateSignal:
    def test_compute_video_normed_sa_matches_inline_modulation(self):
        """The helper must match the inline computation used at the start of
        the block's self-attention (transformer.py line 266)."""
        block = BasicAVTransformerBlock(
            video_dim=32,
            audio_dim=16,
            video_num_heads=4,
            audio_num_heads=4,
            video_head_dim=8,
            audio_head_dim=4,
            av_cross_num_heads=4,
            av_cross_head_dim=4,
            ff_mult=2.0,
        )
        B, Nv = 2, 6
        video_hidden = mx.random.normal((B, Nv, 32))
        # 9-param video AdaLN: (B, 9 * video_dim)
        video_adaln_params = mx.random.normal((B, 9 * 32))

        normed = block.compute_video_normed_sa(video_hidden, video_adaln_params)

        # Reference: replicate the transformer.py:266 computation
        v_shift_sa, v_scale_sa, *_ = block._unpack_adaln(video_adaln_params, block.scale_shift_table, 9, 32)
        ref = block._rms_norm(video_hidden) * (1.0 + v_scale_sa) + v_shift_sa

        mx.synchronize()
        assert normed.shape == (B, Nv, 32)
        assert mx.allclose(normed, ref, atol=1e-6).item()

    def test_compute_video_normed_sa_per_token_adaln(self):
        """Per-token AdaLN (B, N, 9*dim) must also work."""
        block = BasicAVTransformerBlock(
            video_dim=32,
            audio_dim=16,
            video_num_heads=4,
            audio_num_heads=4,
            video_head_dim=8,
            audio_head_dim=4,
            av_cross_num_heads=4,
            av_cross_head_dim=4,
            ff_mult=2.0,
        )
        B, Nv = 2, 6
        video_hidden = mx.random.normal((B, Nv, 32))
        video_adaln_params = mx.random.normal((B, Nv, 9 * 32))

        normed = block.compute_video_normed_sa(video_hidden, video_adaln_params)
        mx.synchronize()
        assert normed.shape == (B, Nv, 32)


class TestLTXModelGateSignal:
    @staticmethod
    def _tiny_config():
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
        )

    def test_compute_gate_signal_shape(self):
        model = LTXModel(self._tiny_config())
        B, Nv, Na = 2, 8, 4
        video_latent = mx.random.normal((B, Nv, 128))
        audio_latent = mx.random.normal((B, Na, 128))
        timestep = mx.array([0.5, 0.5], dtype=mx.bfloat16)

        gate = model.compute_gate_signal(video_latent, audio_latent, timestep)
        mx.synchronize()
        assert gate.shape == (B, Nv, 32)

    def test_gate_signal_matches_inline_block0_modulation(self):
        """The gate must equal block 0's video_normed during a real forward."""
        model = LTXModel(self._tiny_config())
        B, Nv, Na = 1, 4, 2
        video_latent = mx.random.normal((B, Nv, 128))
        audio_latent = mx.random.normal((B, Na, 128))
        timestep = mx.array([0.7], dtype=mx.bfloat16)

        gate = model.compute_gate_signal(video_latent, audio_latent, timestep)

        # Replicate the prelude manually and call block 0's helper.
        v_hidden = model.patchify_proj(video_latent.astype(mx.bfloat16))
        t_emb = model._embed_timestep_scalar(timestep.astype(mx.bfloat16))
        video_adaln_emb, _ = model.adaln_single(t_emb)
        ref = model.transformer_blocks[0].compute_video_normed_sa(v_hidden, video_adaln_emb)

        mx.synchronize()
        assert mx.allclose(gate, ref, atol=1e-5).item()


class TestLTXModelHooks:
    @staticmethod
    def _tiny():
        return LTXModel(
            LTXModelConfig(
                num_layers=2,
                video_dim=32,
                audio_dim=16,
                video_num_heads=4,
                audio_num_heads=4,
                video_head_dim=8,
                audio_head_dim=4,
                av_cross_num_heads=4,
                av_cross_head_dim=4,
            )
        )

    def _inputs(self):
        B, Nv, Na = 1, 4, 2
        return {
            "video_latent": mx.random.normal((B, Nv, 128)),
            "audio_latent": mx.random.normal((B, Na, 128)),
            "timestep": mx.array([0.5], dtype=mx.bfloat16),
        }

    def test_tap_called_once_with_block_residuals(self):
        model = self._tiny()
        recorded = []

        def tap(video_block_residual, audio_block_residual):
            recorded.append((video_block_residual.shape, audio_block_residual.shape))

        out_v, out_a = model(**self._inputs(), tap=tap)
        mx.synchronize()
        assert len(recorded) == 1
        assert recorded[0] == ((1, 4, 32), (1, 2, 16))
        assert out_v.shape == (1, 4, 128)
        assert out_a.shape == (1, 2, 128)

    def test_no_tap_no_overhead_no_failure(self):
        model = self._tiny()
        out_v, out_a = model(**self._inputs())
        mx.synchronize()
        assert out_v.shape == (1, 4, 128)
        assert out_a.shape == (1, 2, 128)

    def test_block_stack_override_replaces_block_iteration(self):
        model = self._tiny()
        called = []

        def override(v_hidden, a_hidden):
            called.append(True)
            # Return a fixed deterministic transformation
            return v_hidden + 1.0, a_hidden - 1.0

        out_v, out_a = model(**self._inputs(), block_stack_override=override)
        mx.synchronize()
        assert called == [True]
        assert out_v.shape == (1, 4, 128)
        # Sanity: the head still ran (output channels are patch_channels=128, not video_dim=32)

    def test_override_and_tap_coexist(self):
        model = self._tiny()
        recorded = []

        def tap(v_res, a_res):
            recorded.append((v_res.shape, a_res.shape))

        def override(v_hidden, a_hidden):
            return v_hidden * 2.0, a_hidden * 2.0

        model(**self._inputs(), tap=tap, block_stack_override=override)
        mx.synchronize()
        # Tap fires once; residual is computed against the override output.
        assert len(recorded) == 1
        # Residuals are (override_output - block_input) — non-zero by construction.
