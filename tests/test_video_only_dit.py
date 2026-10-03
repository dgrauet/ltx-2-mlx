"""Video-only DiT path (upstream: ``BasicAVTransformerBlock.forward`` with ``audio=None``)."""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock


def _block() -> BasicAVTransformerBlock:
    mx.random.seed(0)
    block = BasicAVTransformerBlock(
        video_dim=64,
        audio_dim=32,
        video_num_heads=2,
        video_head_dim=32,
        audio_num_heads=2,
        audio_head_dim=16,
        av_cross_num_heads=2,
        av_cross_head_dim=16,
    )
    mx.eval(block.parameters())
    return block


def _video_kwargs(nv=6, nt=3):
    return dict(
        video_hidden=mx.random.normal((1, nv, 64)),
        video_adaln_params=mx.random.normal((1, 9 * 64)) * 0.1,
        video_prompt_adaln_params=mx.random.normal((1, 2 * 64)) * 0.1,
        video_text_embeds=mx.random.normal((1, nt, 64)),
    )


_NO_AUDIO = dict(
    audio_hidden=None,
    audio_adaln_params=None,
    audio_prompt_adaln_params=None,
    av_ca_video_params=None,
    av_ca_audio_params=None,
    av_ca_a2v_gate_params=None,
    av_ca_v2a_gate_params=None,
)


def test_block_runs_without_audio():
    block = _block()
    v, a = block(**_NO_AUDIO, **_video_kwargs())
    assert a is None and v.shape == (1, 6, 64)
    assert bool(mx.all(mx.isfinite(v)).item())


def test_video_only_equals_joint_with_a2v_gated_off():
    """Audio only reaches video through A->V; with that gate at zero the joint block's video output
    must equal the video-only one (proves the video path is untouched)."""
    block = _block()
    block.scale_shift_table_a2v_ca_video = mx.zeros_like(block.scale_shift_table_a2v_ca_video)
    kw = _video_kwargs()
    joint_v, _ = block(
        audio_hidden=mx.random.normal((1, 4, 32)),
        audio_adaln_params=mx.random.normal((1, 9 * 32)) * 0.1,
        audio_prompt_adaln_params=mx.random.normal((1, 2 * 32)) * 0.1,
        av_ca_video_params=mx.zeros((1, 4 * 64)),
        av_ca_audio_params=mx.zeros((1, 4 * 32)),
        av_ca_a2v_gate_params=mx.zeros((1, 64)),
        av_ca_v2a_gate_params=mx.zeros((1, 32)),
        audio_text_embeds=mx.random.normal((1, 3, 32)),
        **kw,
    )
    solo_v, _ = block(**_NO_AUDIO, **kw)
    assert mx.allclose(joint_v, solo_v, atol=1e-5).item()
