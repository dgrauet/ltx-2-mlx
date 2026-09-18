"""A2VidPipelineTwoStage stage-1 must thread its TeaCache controller through.

``TI2VidTwoStagesPipeline`` has carried calibrated TeaCache Euler coefficients and a
``_build_teacache_controller`` helper for a while, and ``_add_generation_args`` already
exposes ``--enable-teacache`` on every generate subcommand — but the a2v pipeline never
passed a controller down to ``guided_denoise_loop``, so the flag was accepted and silently
ignored on ``a2v``. These pin the wiring.

The assertions are deliberately at ``_denoise_stage1`` rather than through
``generate_and_save``: the wiring under test is one keyword argument's journey into
``guided_denoise_loop``, and standing up the full pipeline (VAE, text encoder, audio
encoder, upsampler) to observe it would test the stubs more than the code.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_pipelines_mlx import a2vid_two_stage as a2v_mod
from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage


class _LoopSpy:
    """Stand-in for guided_denoise_loop that records the kwargs it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return object()


@pytest.fixture
def stage1(monkeypatch):
    """A pipeline whose _denoise_stage1 can be called without loading any weights."""
    pipe = A2VidPipelineTwoStage.__new__(A2VidPipelineTwoStage)
    # _pre_denoise_flush only exists to release memory between stages; there is nothing
    # allocated here to release.
    pipe._pre_denoise_flush = lambda *a, **k: None  # type: ignore[method-assign]

    spy = _LoopSpy()
    monkeypatch.setattr(a2v_mod, "guided_denoise_loop", spy)

    def run(**overrides):
        kwargs = dict(
            x0_model=object(),
            video_state=object(),
            audio_state=object(),
            video_embeds=mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
            audio_embeds=mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
            neg_video_embeds=mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
            neg_audio_embeds=mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
            sigmas=[1.0, 0.5, 0.0],
        )
        kwargs.update(overrides)
        pipe._denoise_stage1(**kwargs)
        return spy

    return run


def test_teacache_controller_reaches_the_denoise_loop(stage1):
    """The regression: a controller passed in must arrive as `teacache=`."""
    controller = object()

    spy = stage1(teacache_controller=controller)

    assert len(spy.calls) == 1
    assert spy.calls[0]["teacache"] is controller, (
        "identity, not truthiness: the controller carries calibrated per-model "
        "coefficients, so the loop must receive the caller's object rather than "
        "any controller"
    )


def test_teacache_defaults_to_none(stage1):
    """Omitting it must disable TeaCache, not enable a default one."""
    spy = stage1()

    assert spy.calls[0]["teacache"] is None


def test_on_step_hook_still_threads_through(stage1):
    """The stepwise hook shares these call sites; neither may displace the other."""
    hook = object()

    spy = stage1(on_step=hook, teacache_controller=object())

    assert spy.calls[0]["on_step"] is hook
    assert spy.calls[0]["teacache"] is not None
