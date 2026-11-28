"""Guidance rescale over denoised tokens only (upstream LTX-2 v1.6.0).

The rescale factor ``std(cond) / std(pred)`` is estimated over the tokens being denoised, so fixed
conditioning and reference tokens (denoise mask 0) do not skew it.
"""

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.components.guiders import MultiModalGuider, MultiModalGuiderParams

PARAMS = MultiModalGuiderParams(cfg_scale=3.0, stg_scale=0.0, rescale_scale=0.7, modality_scale=1.0)


def _predictions(n_tokens: int = 6) -> tuple[mx.array, mx.array]:
    mx.random.seed(0)
    cond = mx.random.normal((1, n_tokens, 4))
    uncond = mx.random.normal((1, n_tokens, 4))
    return cond, uncond


def _expected(cond: mx.array, uncond: mx.array, rows: slice) -> np.ndarray:
    """Guided prediction with the factor computed from ``rows`` only, in float64."""
    c, u = np.array(cond, dtype=np.float64), np.array(uncond, dtype=np.float64)
    pred = c + (PARAMS.cfg_scale - 1) * (c - u)
    factor = c[:, rows].std() / pred[:, rows].std()
    return pred * (PARAMS.rescale_scale * factor + (1 - PARAMS.rescale_scale))


def test_factor_ignores_preserved_tokens() -> None:
    cond, uncond = _predictions()
    # Two preserved tokens whose predictions are far off scale: they dominate an unmasked std.
    cond = mx.concatenate([cond[:, :4], cond[:, 4:] * 50.0], axis=1)
    mask = mx.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]).reshape(1, 6, 1)
    out = MultiModalGuider(PARAMS).calculate(cond, uncond, 0.0, 0.0, denoise_mask=mask)
    np.testing.assert_allclose(np.array(out), _expected(cond, uncond, slice(0, 4)), rtol=1e-4, atol=1e-4)
    unmasked = MultiModalGuider(PARAMS).calculate(cond, uncond, 0.0, 0.0)
    assert not mx.allclose(out, unmasked, rtol=1e-2, atol=1e-2).item()


def test_partial_mask_values_count_as_denoised() -> None:
    cond, uncond = _predictions()
    mask = mx.array([0.5, 1.0, 0.25, 1.0, 0.0, 0.0]).reshape(1, 6, 1)
    out = MultiModalGuider(PARAMS).calculate(cond, uncond, 0.0, 0.0, denoise_mask=mask)
    np.testing.assert_allclose(np.array(out), _expected(cond, uncond, slice(0, 4)), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mask", [None, mx.ones((1, 6, 1))])
def test_all_tokens_denoised_is_bit_identical_to_no_mask(mask: mx.array | None) -> None:
    cond, uncond = _predictions()
    reference = MultiModalGuider(PARAMS).calculate(cond, uncond, 0.0, 0.0)
    out = MultiModalGuider(PARAMS).calculate(cond, uncond, 0.0, 0.0, denoise_mask=mask)
    assert mx.array_equal(out, reference).item()


def test_frozen_modality_is_not_rescaled() -> None:
    cond, uncond = _predictions()
    out = MultiModalGuider(PARAMS).calculate(cond, uncond, 0.0, 0.0, denoise_mask=mx.zeros((1, 6, 1)))
    assert mx.allclose(out, cond + (PARAMS.cfg_scale - 1) * (cond - uncond), atol=1e-6).item()


@pytest.mark.parametrize("loop_name", ["guided_denoise_loop", "res2s_denoise_loop"])
def test_guided_loops_pass_each_modality_its_denoise_mask(monkeypatch: pytest.MonkeyPatch, loop_name: str) -> None:
    """Both guided loops hand the guider the state's mask, for video and for audio."""
    from ltx_pipelines_mlx.utils import samplers
    from tests.test_res2s_anchor_mask import PINNED, _run

    seen: list[mx.array | None] = []
    original = MultiModalGuider.calculate

    def spy(self, *args, denoise_mask=None, **kwargs):
        seen.append(denoise_mask)
        return original(self, *args, denoise_mask=denoise_mask, **kwargs)

    monkeypatch.setattr(MultiModalGuider, "calculate", spy)
    _, video_state, audio_state = _run(getattr(samplers, loop_name))
    assert seen and all(mask is not None for mask in seen)
    token_counts = {mask.shape[1] for mask in seen}
    assert token_counts == {video_state.denoise_mask.shape[1], audio_state.denoise_mask.shape[1]}
    assert all(float(mask[:, :PINNED].sum().item()) == 0.0 for mask in seen)
