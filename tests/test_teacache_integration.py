"""Integration tests for the sampler-level TeaCache hooks (tap + teacache)."""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop


def _tiny_x0_model():
    return X0Model(
        LTXModel(
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
    )


def _make_state(batch: int, n_tokens: int) -> LatentState:
    latent = mx.random.normal((batch, n_tokens, 128)).astype(mx.bfloat16)
    return LatentState(
        latent=latent,
        clean_latent=mx.zeros_like(latent),
        denoise_mask=mx.ones((batch, n_tokens, 1), dtype=mx.bfloat16),
    )


def _cfg_factory(batch: int, video_dim: int) -> MultiModalGuiderFactory:
    """CFG-only guider factory (no STG, no modality)."""
    params = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=0.0,
        rescale_scale=0.7,
        modality_scale=1.0,
        stg_blocks=[],
    )
    neg = mx.zeros((batch, 4, video_dim), dtype=mx.bfloat16)
    return create_multimodal_guider_factory(params, negative_context=neg)


class TestTapHook:
    def test_tap_fires_once_per_step(self):
        model = _tiny_x0_model()
        B, Nv, Na = 1, 4, 2
        video_state = _make_state(B, Nv)
        audio_state = _make_state(B, Na)

        # 4 steps total → 3 sigma pairs (denoising)
        sigmas = [1.0, 0.7, 0.4, 0.1, 0.0]
        video_text_embeds = mx.zeros((B, 4, 32), dtype=mx.bfloat16)
        audio_text_embeds = mx.zeros((B, 4, 16), dtype=mx.bfloat16)

        recorded: list[tuple] = []

        def tap(step_idx, gate, video_residual, audio_residual):
            recorded.append((step_idx, gate.shape, video_residual.shape, audio_residual.shape))

        guided_denoise_loop(
            model=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_guider_factory=_cfg_factory(B, 32),
            audio_guider_factory=_cfg_factory(B, 16),
            sigmas=sigmas,
            show_progress=False,
            tap=tap,
        )
        assert [r[0] for r in recorded] == [0, 1, 2, 3]
        for _, gate_shape, v_res_shape, a_res_shape in recorded:
            assert gate_shape == (B, Nv, 32)
            assert v_res_shape == (B, Nv, 32)
            assert a_res_shape == (B, Na, 16)
