"""Integration tests for the sampler-level TeaCache hooks (tap + teacache)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop, res2s_denoise_loop


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


class _StubController:
    """Records calls and returns canned should_compute decisions."""

    def __init__(self, decisions: list[bool]):
        self.decisions = list(decisions)
        self.cached = None
        self.gate_calls: list = []
        self.cache_calls: list = []
        self.reset_calls = 0

    def should_compute(self, step_idx, gate_signal):
        self.gate_calls.append((step_idx, gate_signal.shape))
        return self.decisions.pop(0)

    def cache_residual(self, payload):
        self.cache_calls.append(payload)
        self.cached = payload

    @property
    def previous_residual(self):
        if self.cached is None:
            raise RuntimeError("nothing cached")
        return self.cached

    def reset(self):
        self.reset_calls += 1


class TestTeaCacheHook:
    def _run(self, decisions: list[bool]):
        model = _tiny_x0_model()
        B, Nv, Na = 1, 4, 2
        video_state = _make_state(B, Nv)
        audio_state = _make_state(B, Na)
        sigmas = [1.0, 0.7, 0.4, 0.1, 0.0]
        controller = _StubController(decisions)
        guided_denoise_loop(
            model=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=mx.zeros((B, 4, 32), dtype=mx.bfloat16),
            audio_text_embeds=mx.zeros((B, 4, 16), dtype=mx.bfloat16),
            video_guider_factory=_cfg_factory(B, 32),
            audio_guider_factory=_cfg_factory(B, 16),
            sigmas=sigmas,
            show_progress=False,
            teacache=controller,
        )
        return controller

    def test_should_compute_called_once_per_step(self):
        c = self._run(decisions=[True, True, True, True])
        assert len(c.gate_calls) == 4
        assert all(shape == (1, 4, 32) for _, shape in c.gate_calls)

    def test_compute_path_caches_per_pass_dict(self):
        c = self._run(decisions=[True, True, True, True])
        # 4 cache_residual calls (one per step). Each is a dict with 'cond' and 'uncond' keys.
        assert len(c.cache_calls) == 4
        for payload in c.cache_calls:
            assert isinstance(payload, dict)
            assert set(payload.keys()) == {"cond", "uncond"}
            v_res, a_res = payload["cond"]
            assert v_res.shape == (1, 4, 32)
            assert a_res.shape == (1, 2, 16)

    def test_skip_path_does_not_cache(self):
        """When should_compute returns False at step 1, no cache is stored
        for that step (the previous residual is reused via override). 4 steps
        with one skip → 3 cache_residual calls."""
        c = self._run(decisions=[True, False, True, True])
        assert len(c.gate_calls) == 4
        assert len(c.cache_calls) == 3


class TestRes2sTeaCacheHook:
    """Same contract as TestTeaCacheHook but exercising res2s_denoise_loop.

    res2s does TWO model evaluations per outer step (stage 1 at sigma, stage 2
    at sub_sigma). The TeaCache decision is per outer step, and the cache
    payload contains residuals for BOTH stages keyed by ``"stage1"`` / ``"stage2"``.
    """

    def _run(self, decisions: list[bool]):
        model = _tiny_x0_model()
        B, Nv, Na = 1, 4, 2
        video_state = _make_state(B, Nv)
        audio_state = _make_state(B, Na)
        sigmas = [1.0, 0.7, 0.4, 0.1, 0.05]  # 4 outer steps, no terminal 0 to keep the test simple
        controller = _StubController(decisions)
        res2s_denoise_loop(
            model=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=mx.zeros((B, 4, 32), dtype=mx.bfloat16),
            audio_text_embeds=mx.zeros((B, 4, 16), dtype=mx.bfloat16),
            video_guider_factory=_cfg_factory(B, 32),
            audio_guider_factory=_cfg_factory(B, 16),
            sigmas=sigmas,
            show_progress=False,
            bongmath=False,  # bong iteration is irrelevant for the cache contract
            teacache=controller,
        )
        return controller

    def test_should_compute_called_once_per_outer_step(self):
        c = self._run(decisions=[True, True, True, True])
        assert len(c.gate_calls) == 4
        assert all(shape == (1, 4, 32) for _, shape in c.gate_calls)

    def test_compute_path_caches_stage1_and_stage2(self):
        c = self._run(decisions=[True, True, True, True])
        assert len(c.cache_calls) == 4
        for payload in c.cache_calls:
            assert isinstance(payload, dict)
            assert set(payload.keys()) == {"stage1", "stage2"}
            for stage in ("stage1", "stage2"):
                assert set(payload[stage].keys()) == {"cond", "uncond"}
                v_res, a_res = payload[stage]["cond"]
                assert v_res.shape == (1, 4, 32)
                assert a_res.shape == (1, 2, 16)

    def test_skip_path_does_not_cache(self):
        c = self._run(decisions=[True, False, True, True])
        assert len(c.gate_calls) == 4
        assert len(c.cache_calls) == 3


class TestPipelineFlag:
    def test_enable_teacache_raises_clear_error_before_calibration(self):
        """Until calibration is run, LTX2_TEACACHE_COEFFICIENTS is empty. The
        pipeline must raise a clear error pointing at the calibration script."""
        from ltx_pipelines_mlx import ti2vid_two_stages

        # Fail fast if calibration has already been dropped in.
        if ti2vid_two_stages.LTX2_TEACACHE_COEFFICIENTS:
            pytest.skip("LTX2 coefficients already populated; this guard no longer applies")

        with pytest.raises(RuntimeError, match="calibrate_teacache"):
            ti2vid_two_stages._build_teacache_controller(num_steps=30, thresh=None)

    def test_build_teacache_controller_uses_module_constants(self):
        """When calibration is populated, the controller picks up the module-level
        constants (coefficients + default threshold)."""
        from ltx_pipelines_mlx import ti2vid_two_stages

        # Stub the constants for the duration of the test.
        original_coeffs = ti2vid_two_stages.LTX2_TEACACHE_COEFFICIENTS
        original_thresh = ti2vid_two_stages.LTX2_TEACACHE_THRESH
        try:
            ti2vid_two_stages.LTX2_TEACACHE_COEFFICIENTS = [1.0, 2.0, 3.0]
            ti2vid_two_stages.LTX2_TEACACHE_THRESH = 0.2

            ctrl = ti2vid_two_stages._build_teacache_controller(num_steps=30, thresh=None)
            assert ctrl.num_steps == 30
            assert ctrl.rel_l1_thresh == 0.2
            assert list(ctrl.coefficients) == [1.0, 2.0, 3.0]

            # thresh override
            ctrl = ti2vid_two_stages._build_teacache_controller(num_steps=30, thresh=0.5)
            assert ctrl.rel_l1_thresh == 0.5
        finally:
            ti2vid_two_stages.LTX2_TEACACHE_COEFFICIENTS = original_coeffs
            ti2vid_two_stages.LTX2_TEACACHE_THRESH = original_thresh
