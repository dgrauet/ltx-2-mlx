"""STG prefix reuse: the STG pass resumes from the conditioned pass at the first perturbed block.

The STG pass has the conditioned pass's inputs and perturbs only the self-attention of
``stg_blocks``, so the blocks before the first perturbed one compute the same hidden
states in both passes. ``BlockPrefix`` carries them from one pass to the other. These
tests check that resuming gives exactly the recomputed output (model and both guided
loops), that the skipped blocks really are skipped, and the cases that must recompute.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_core_mlx.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core_mlx.model.transformer.model import BlockPrefix, LTXModel, LTXModelConfig, X0Model
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop, res2s_denoise_loop, stg_block_prefix

NUM_LAYERS = 4
STG_BLOCK = 2
SIGMAS = [1.0, 0.6, 0.25, 0.0]


def _config() -> LTXModelConfig:
    return LTXModelConfig(
        num_layers=NUM_LAYERS,
        video_dim=128,
        audio_dim=64,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=32,
        audio_head_dim=16,
        av_cross_num_heads=4,
        av_cross_head_dim=16,
        video_patch_channels=64,
        audio_patch_channels=64,
        ff_mult=2.0,
        timestep_embedding_dim=64,
    )


def _model() -> LTXModel:
    mx.random.seed(0)
    model = LTXModel(_config())
    model.set_dtype(mx.bfloat16)
    for block in model.transformer_blocks:
        for name in ("scale_shift_table", "audio_scale_shift_table", "prompt_scale_shift_table"):
            block[name] = 0.05 * mx.random.normal(block[name].shape)
    mx.eval(model.parameters())
    return model


def _forward_inputs(cfg: LTXModelConfig) -> dict:
    mx.random.seed(1)
    return dict(
        video_latent=mx.random.normal((1, 48, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, 8, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([0.7]),
        video_text_embeds=mx.random.normal((1, 6, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, 6, cfg.audio_dim)).astype(mx.bfloat16),
    )


def _stg(blocks: list[int] | None) -> BatchedPerturbationConfig:
    return BatchedPerturbationConfig(
        perturbations=[
            PerturbationConfig(
                perturbations=[
                    Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=blocks),
                    Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=blocks),
                ]
            )
        ]
    )


@pytest.fixture
def block_calls(monkeypatch):
    """Count BasicAVTransformerBlock forwards."""
    calls = {"n": 0}
    original = BasicAVTransformerBlock.__call__

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BasicAVTransformerBlock, "__call__", counting)
    return calls


def test_resume_matches_full_perturbed_forward(block_calls):
    model = _model()
    inputs = _forward_inputs(model.config)

    reference = model(**inputs, perturbations=_stg([STG_BLOCK]))
    assert block_calls["n"] == NUM_LAYERS

    prefix = BlockPrefix(STG_BLOCK)
    model(**inputs, capture_prefix=prefix)
    assert prefix.captured
    block_calls["n"] = 0
    resumed = model(**inputs, perturbations=_stg([STG_BLOCK]), resume_prefix=prefix)

    assert block_calls["n"] == NUM_LAYERS - STG_BLOCK
    for got, ref in zip(resumed, reference, strict=True):
        assert mx.array_equal(got, ref)


def test_capture_does_not_change_the_forward():
    model = _model()
    inputs = _forward_inputs(model.config)
    plain = model(**inputs)
    captured = model(**inputs, capture_prefix=BlockPrefix(STG_BLOCK))
    for got, ref in zip(captured, plain, strict=True):
        assert mx.array_equal(got, ref)


def test_resume_guards():
    model = _model()
    inputs = _forward_inputs(model.config)
    with pytest.raises(ValueError, match="never captured"):
        model(**inputs, resume_prefix=BlockPrefix(STG_BLOCK))
    prefix = BlockPrefix(STG_BLOCK)
    model(**inputs, capture_prefix=prefix)
    with pytest.raises(ValueError, match="tap"):
        model(**inputs, resume_prefix=prefix, tap=lambda v, a: None)


def _perturbations(blocks: list[int] | None) -> list[Perturbation]:
    return [Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=blocks)]


def test_stg_block_prefix_rules(monkeypatch):
    monkeypatch.delenv("LTX2_STG_PREFIX_REUSE", raising=False)
    x0 = X0Model(_model())
    assert stg_block_prefix(x0, _perturbations([28, 30])).index == 28
    two = _perturbations([5]) + [Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=[3])]
    assert stg_block_prefix(x0, two).index == 3
    assert stg_block_prefix(x0, []) is None
    assert stg_block_prefix(x0, _perturbations(None)) is None  # every block
    assert stg_block_prefix(x0, _perturbations([])) is None
    assert stg_block_prefix(x0, _perturbations([0, 4])) is None  # nothing before block 0

    monkeypatch.setenv("LTX2_STG_PREFIX_REUSE", "0")
    assert stg_block_prefix(x0, _perturbations([28])) is None


def test_stg_block_prefix_refuses_modality_tiling(monkeypatch):
    from ltx_core_mlx.components.modality_tiling import TiledLTXModel

    monkeypatch.delenv("LTX2_STG_PREFIX_REUSE", raising=False)
    tiled = TiledLTXModel(_model(), tiler=None)
    assert stg_block_prefix(X0Model(tiled), _perturbations([28])) is None


def test_stg_block_prefix_through_streaming_wrapper(monkeypatch):
    from ltx_core_mlx.loader.block_streaming import StreamingLTXModel

    monkeypatch.delenv("LTX2_STG_PREFIX_REUSE", raising=False)
    wrapper = StreamingLTXModel.__new__(StreamingLTXModel)
    nn.Module.__init__(wrapper)
    wrapper["inner"] = _model()
    assert wrapper.supports_block_prefix is True


def _state(num_tokens: int, seed: int) -> LatentState:
    mx.random.seed(seed)
    latent = mx.random.normal((1, num_tokens, 64)).astype(mx.bfloat16)
    return LatentState(
        latent=latent,
        clean_latent=mx.zeros_like(latent),
        denoise_mask=mx.ones((1, num_tokens, 1), dtype=mx.bfloat16),
    )


def _factory(cfg: LTXModelConfig, stg_blocks: list[int]) -> MultiModalGuiderFactory:
    return MultiModalGuiderFactory.constant(
        MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0, stg_blocks=stg_blocks, rescale_scale=0.7),
        negative_context=mx.zeros((1, 6, cfg.video_dim), dtype=mx.bfloat16),
    )


def _run(loop: str, model: X0Model, stg_blocks: list[int]):
    cfg = model.model.config
    mx.random.seed(2)
    text = dict(
        video_text_embeds=mx.random.normal((1, 6, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, 6, cfg.audio_dim)).astype(mx.bfloat16),
    )
    video, audio = _state(48, 3), _state(8, 4)
    if loop == "guided":
        out = guided_denoise_loop(
            model,
            video,
            audio,
            video_guider_factory=_factory(cfg, stg_blocks),
            sigmas=SIGMAS,
            show_progress=False,
            **text,
        )
    else:
        out = res2s_denoise_loop(
            model,
            video,
            audio,
            sigmas=SIGMAS,
            bongmath=False,
            video_guider_factory=_factory(cfg, stg_blocks),
            show_progress=False,
            **text,
        )
    mx.eval(out.video_latent, out.audio_latent)
    return out


@pytest.mark.parametrize("loop", ["guided", "res2s"])
def test_loops_reuse_prefix_with_identical_output(loop, monkeypatch, block_calls):
    model = X0Model(_model())

    monkeypatch.setenv("LTX2_STG_PREFIX_REUSE", "0")
    reference = _run(loop, model, [STG_BLOCK])
    calls_off = block_calls["n"]

    monkeypatch.delenv("LTX2_STG_PREFIX_REUSE")
    block_calls["n"] = 0
    reused = _run(loop, model, [STG_BLOCK])
    calls_on = block_calls["n"]

    assert mx.array_equal(reused.video_latent, reference.video_latent)
    assert mx.array_equal(reused.audio_latent, reference.audio_latent)
    # three passes per model evaluation (cond, uncond, ptb); the ptb pass skips STG_BLOCK blocks
    evaluations = calls_off // (3 * NUM_LAYERS)
    assert calls_off == evaluations * 3 * NUM_LAYERS
    assert calls_on == calls_off - evaluations * STG_BLOCK


def test_loop_recomputes_when_block_zero_is_perturbed(monkeypatch, block_calls):
    monkeypatch.delenv("LTX2_STG_PREFIX_REUSE", raising=False)
    _run("guided", X0Model(_model()), [0])
    assert block_calls["n"] == (len(SIGMAS) - 1) * 3 * NUM_LAYERS
