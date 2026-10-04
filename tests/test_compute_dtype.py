"""Opt-in compute dtype for the DiT attention / feed-forward internals.

``LTXModel.set_compute_dtype`` (and ``LTX2_COMPUTE_DTYPE`` in the pipelines)
runs the inside of every attention and feed-forward module in another dtype
while the residual stream and the AdaLN modulation keep theirs. These tests use
a tiny int8-quantized model with F32 AdaLN tables and per-token timesteps, the
same dtype layout as the shipped q8 packs, so the default path really runs in
float32 here as it does in production.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from ltx_core_mlx.loader.block_streaming import BlockStreamer, StreamingLTXModel
from ltx_core_mlx.model.transformer.attention import Attention, _cast_inputs
from ltx_core_mlx.model.transformer.feed_forward import FeedForward
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, compute_dtype_from_env

_TABLES = (
    "scale_shift_table",
    "audio_scale_shift_table",
    "prompt_scale_shift_table",
    "audio_prompt_scale_shift_table",
    "scale_shift_table_a2v_ca_video",
    "scale_shift_table_a2v_ca_audio",
)


def _config(num_layers: int = 2) -> LTXModelConfig:
    # Smallest dims the int8 / group-64 quantization accepts.
    return LTXModelConfig(
        num_layers=num_layers,
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


def _model(seed: int = 0, num_layers: int = 2) -> LTXModel:
    """bf16 weights, int8 block linears with bf16 scales, F32 AdaLN tables -- as in the packs."""
    mx.random.seed(seed)
    model = LTXModel(_config(num_layers))
    model.set_dtype(mx.bfloat16)
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda path, m: (
            path.startswith("transformer_blocks.") and isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0
        ),
    )
    for block in model.transformer_blocks:
        for name in _TABLES:
            block[name] = 0.05 * mx.random.normal(block[name].shape)  # float32
    mx.eval(model.parameters())
    return model


def _inputs(cfg: LTXModelConfig, seed: int = 1) -> dict:
    """I2V-shaped call: the first frame's tokens sit at sigma 0 (per-token timesteps)."""
    mx.random.seed(seed)
    b, nv, na, nt = 1, 48, 8, 6
    sigma = 0.7
    video_timesteps = mx.concatenate([mx.zeros((b, 16)), mx.full((b, nv - 16), sigma)], axis=1)
    return dict(
        video_latent=mx.random.normal((b, nv, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((b, na, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([sigma]),
        video_text_embeds=mx.random.normal((b, nt, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((b, nt, cfg.audio_dim)).astype(mx.bfloat16),
        video_timesteps=video_timesteps,
    )


def _rel_err(a: mx.array, ref: mx.array) -> float:
    a, ref = a.astype(mx.float32), ref.astype(mx.float32)
    return (mx.sqrt(mx.sum((a - ref) ** 2)) / mx.sqrt(mx.sum(ref**2))).item()


def _compute_modules(model: LTXModel) -> list:
    return [m for b in model.transformer_blocks for _, m in b.compute_modules()]


def test_default_is_untouched():
    model = _model()
    assert model.compute_dtype is None
    assert all(m.compute_dtype is None for m in _compute_modules(model))
    assert model.transformer_blocks[0].attn1.to_q.scales.dtype == mx.bfloat16
    # the default path computes in float32 because of the F32 tables (the reason for this option)
    video, _ = model(**_inputs(model.config))
    assert video.dtype == mx.float32


def test_set_compute_dtype_casts_only_attention_and_ff_params():
    model = _model()
    model.set_compute_dtype(mx.float16)
    assert model.compute_dtype == mx.float16
    for module in _compute_modules(model):
        assert isinstance(module, (Attention, FeedForward))
        assert module.compute_dtype == mx.float16
    block = model.transformer_blocks[0]
    assert block.attn1.to_q.weight.dtype == mx.uint32  # packed int8 untouched
    assert block.attn1.to_q.scales.dtype == mx.float16
    assert block.attn1.to_q.biases.dtype == mx.float16
    assert block.attn1.q_norm.weight.dtype == mx.float16
    for name in _TABLES:
        assert block[name].dtype == mx.float32  # AdaLN modulation stays float32
    assert model.adaln_single.linear.weight.dtype == mx.bfloat16  # outside the blocks


def test_attention_and_ff_compute_in_the_set_dtype():
    model = _model()
    model.set_compute_dtype(mx.float16)
    block = model.transformer_blocks[0]
    x = mx.random.normal((1, 8, model.config.video_dim))  # float32, like the modulated activations
    text = mx.random.normal((1, 4, model.config.video_dim))
    assert block.attn1(x).dtype == mx.float16
    assert block.attn2(x, encoder_hidden_states=text).dtype == mx.float16
    assert block.ff(x).dtype == mx.float16
    model.set_compute_dtype(None)
    assert block.attn1(x).dtype == mx.float32  # back to the input dtype
    assert block.ff(x).dtype == mx.float32


@pytest.mark.parametrize("dtype, tol", [(mx.float16, 5e-3), (mx.bfloat16, 3e-2)])
def test_output_close_to_default(dtype, tol):
    ref_model, model = _model(), _model()
    model.set_compute_dtype(dtype)
    inputs = _inputs(ref_model.config)
    ref_v, ref_a = ref_model(**inputs)
    v, a = model(**inputs)
    assert v.dtype == mx.float32 and a.dtype == mx.float32  # residual stream stays float32
    assert 1e-6 < _rel_err(v, ref_v) < tol  # lower bound: the internals really ran in `dtype`
    assert _rel_err(a, ref_a) < tol


def test_float16_is_closer_to_default_than_bfloat16():
    ref_model, m16, mbf = _model(), _model(), _model()
    m16.set_compute_dtype(mx.float16)
    mbf.set_compute_dtype(mx.bfloat16)
    inputs = _inputs(ref_model.config)
    ref_v, _ = ref_model(**inputs)
    assert _rel_err(m16(**inputs)[0], ref_v) < _rel_err(mbf(**inputs)[0], ref_v)


def test_overflow_guard_recomputes_without_compute_dtype(capsys):
    ref_model, model = _model(), _model()
    for m in (ref_model, model):
        # Finite in float32, past float16's 65504 inside the first block's feed-forward.
        ff = m.transformer_blocks[0].ff.proj_out
        ff.scales = ff.scales * 5e5
        mx.eval(m.parameters())
    model.set_compute_dtype(mx.float16)
    inputs = _inputs(ref_model.config)
    ref_v, _ = ref_model(**inputs)
    assert bool(mx.all(mx.isfinite(ref_v)).item())
    v, a = model(**inputs)
    assert "not finite" in capsys.readouterr().err
    assert model.compute_dtype is None
    assert all(m.compute_dtype is None for m in _compute_modules(model))
    assert bool(mx.all(mx.isfinite(v)).item()) and bool(mx.all(mx.isfinite(a)).item())
    assert _rel_err(v, ref_v) < 1e-3  # same float32 activations; only the scales went through float16


def test_additive_mask_is_clamped_not_infinite():
    x = mx.zeros((1, 4, 8))
    mask = mx.array([[[[0.0, -1e9], [-1e9, -1e9]]]])
    _, _, cast_mask, _ = _cast_inputs(mx.float16, x, None, mask, None)
    assert cast_mask.dtype == mx.float16
    assert bool(mx.all(mx.isfinite(cast_mask)).item())
    assert cast_mask[0, 0, 1, 1].item() == mx.finfo(mx.float16).min


def test_inplace_lora_fusion_is_recast_to_the_compute_dtype():
    """LoRA fusion re-quantizes from float32 but keeps the scales' dtype; the recast after it is a no-op here."""
    from types import SimpleNamespace

    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
    from ltx_core_mlx.utils.weights import apply_quantization
    from ltx_pipelines_mlx._base import BasePipeline

    model = _model()
    model.set_compute_dtype(mx.float16)
    dim = model.config.video_dim
    key = "transformer_blocks.0.attn1.to_q"
    lora = {
        f"{key}.lora_A.weight": 0.01 * mx.random.normal((4, dim)),
        f"{key}.lora_B.weight": 0.01 * mx.random.normal((dim, 4)),
    }
    model_sd = StateDict(sd=dict(tree_flatten(model.parameters())), size=0, dtype=set())
    fused = apply_loras(
        model_sd=model_sd,
        lora_sd_and_strengths=[
            LoraStateDictWithStrength(state_dict=StateDict(sd=lora, size=0, dtype=set()), strength=1.0)
        ],
    )
    apply_quantization(model, fused.sd)
    model.load_weights(list(fused.sd.items()))
    assert model.transformer_blocks[0].attn1.to_q.scales.dtype == mx.float16  # fuse_loras keeps the scales' dtype

    BasePipeline._recast_after_inplace_fusion(SimpleNamespace(dit=model))  # what dfr / ic_lora now call
    assert model.transformer_blocks[0].attn1.to_q.scales.dtype == mx.float16
    assert model.transformer_blocks[0].attn1(mx.random.normal((1, 8, dim))).dtype == mx.float16


def test_two_stage_distilled_lora_fusion_is_recast(tmp_path):
    """Stage 2 of the dev-model pipelines fuses the distilled LoRA in place: it must stay float16."""
    from types import SimpleNamespace

    from ltx_pipelines_mlx._base import BasePipeline
    from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline

    model = _model()
    model.set_compute_dtype(mx.float16)
    dim = model.config.video_dim
    key = "transformer_blocks.0.attn1.to_q"
    mx.save_safetensors(
        str(tmp_path / "distilled-lora.safetensors"),
        {
            f"{key}.lora_A.weight": 0.01 * mx.random.normal((4, dim)),
            f"{key}.lora_B.weight": 0.01 * mx.random.normal((dim, 4)),
        },
    )
    pipe = SimpleNamespace(
        low_ram_streaming=False,
        model_dir=tmp_path,
        dit=model,
        _distilled_lora="distilled-lora.safetensors",
        _distilled_lora_strength=1.0,
        _resolve_safetensors=BasePipeline._resolve_safetensors,
    )
    pipe._recast_after_inplace_fusion = lambda dit=None: BasePipeline._recast_after_inplace_fusion(pipe, dit)

    TI2VidTwoStagesPipeline._fuse_distilled_lora(pipe, model)  # the call site used by two-stage, hq, a2v, keyframe
    assert model.transformer_blocks[0].attn1.to_q.scales.dtype == mx.float16
    assert model.transformer_blocks[0].attn1(mx.random.normal((1, 8, dim))).dtype == mx.float16


def _save_blocks(model: LTXModel, path: Path) -> None:
    flat = {}
    for i, block in enumerate(model.transformer_blocks):
        for k, v in tree_flatten(block.parameters()):
            flat[f"transformer_blocks.{i}.{k}"] = v
    mx.save_safetensors(str(path), flat)


def test_streamed_model_matches_resident_model():
    resident = _model()
    resident.set_compute_dtype(mx.float16)
    inputs = _inputs(resident.config)
    ref_v, ref_a = resident(**inputs)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "blocks.safetensors"
        _save_blocks(_model(), path)  # same seed -> same bf16 weights as `resident` before the cast
        inner = _model()
        inner.transformer_blocks = [inner.transformer_blocks[0]]
        streamed = StreamingLTXModel(inner, BlockStreamer(path, block_prefix="transformer_blocks."))
        streamed.set_compute_dtype(mx.float16)
        v, a = streamed(**inputs)
        assert streamed.compute_dtype == mx.float16
        assert _rel_err(v, ref_v) < 1e-5
        assert _rel_err(a, ref_a) < 1e-5


def test_streamed_overflow_guard_drops_the_setting_on_the_wrapper(capsys):
    def overflowing(seed=0):
        m = _model(seed)
        ff = m.transformer_blocks[0].ff.proj_out
        ff.scales = ff.scales * 5e5  # finite in float32, past float16's 65504 inside the feed-forward
        mx.eval(m.parameters())
        return m

    reference = overflowing()  # no compute dtype: the default path
    inputs = _inputs(reference.config)
    ref_v, ref_a = reference(**inputs)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "blocks.safetensors"
        _save_blocks(overflowing(), path)
        inner = _model()
        inner.transformer_blocks = [inner.transformer_blocks[0]]
        streamer = BlockStreamer(path, block_prefix="transformer_blocks.")
        bound_dtypes = []
        bind = streamer.bind

        def spy(block, idx, **kw):
            bound_dtypes.append(kw.get("cast_dtype"))
            return bind(block, idx, **kw)

        streamer.bind = spy
        streamed = StreamingLTXModel(inner, streamer)
        streamed.set_compute_dtype(mx.float16)

        v, a = streamed(**inputs)
        assert "not finite" in capsys.readouterr().err
        n = reference.config.num_layers
        assert bound_dtypes == [mx.float16] * n + [None] * n  # the recompute binds uncast weights
        # Dropped on the wrapper too: later binds stop casting and the compiled block is used again.
        assert object.__getattribute__(streamed, "_cast_dtype") is None
        assert inner.compute_dtype is None
        # The recompute rebinds the stored weights, so it is the default path, not float16-rounded.
        assert _rel_err(v, ref_v) < 1e-5 and _rel_err(a, ref_a) < 1e-5

        v2, _ = streamed(**inputs)
        assert "not finite" not in capsys.readouterr().err
        assert _rel_err(v2, ref_v) < 1e-5


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", None),
        ("float32", None),
        ("fp32", None),
        ("float16", mx.float16),
        ("FP16", mx.float16),
        ("bf16", mx.bfloat16),
    ],
)
def test_env_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("LTX2_COMPUTE_DTYPE", value)
    assert compute_dtype_from_env() is expected


def test_env_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("LTX2_COMPUTE_DTYPE", "int8")
    with pytest.raises(ValueError, match="LTX2_COMPUTE_DTYPE"):
        compute_dtype_from_env()


def test_env_unset_leaves_pipeline_dit_untouched(monkeypatch):
    from ltx_pipelines_mlx._base import apply_compute_dtype_from_env

    monkeypatch.delenv("LTX2_COMPUTE_DTYPE", raising=False)
    model = _model()
    assert apply_compute_dtype_from_env(model) is model
    assert model.compute_dtype is None
    assert model.transformer_blocks[0].attn1.to_q.scales.dtype == mx.bfloat16

    monkeypatch.setenv("LTX2_COMPUTE_DTYPE", "float16")
    apply_compute_dtype_from_env(model)
    assert model.compute_dtype == mx.float16
