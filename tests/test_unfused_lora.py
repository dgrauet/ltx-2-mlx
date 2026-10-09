"""Unfused LoRAs (``LTX2_LORA_MODE=unfused``): run-time adapters instead of fusion into the weights.

``apply_loras`` dequantizes a q8 weight, adds ``strength * B @ A`` and re-quantizes; for a LoRA
whose delta is small next to the weight (the IC-LoRAs) the re-quantization error is of the same
order as the delta. ``attach_loras`` keeps the weights and adds ``(x @ A^T) @ B^T`` at run time.
These tests use synthetic layers and the tiny int8 model of ``test_compute_dtype`` (bf16 weights,
int8 / group-64 block linears with bf16 scales, as in the packs).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from ltx_core_mlx.loader import (
    AttachedLoras,
    LoRAAdapter,
    LoraStateDictWithStrength,
    StateDict,
    apply_loras,
    attach_loras,
    lora_mode_from_env,
)
from ltx_core_mlx.utils.weights import apply_quantization
from tests.test_compute_dtype import _inputs, _model

_ADAPTED = (
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn2.to_v",
    "transformer_blocks.1.attn1.to_out",
    "transformer_blocks.1.ff.proj_in",
)


@pytest.fixture(autouse=True)
def _fused_by_default(monkeypatch):
    monkeypatch.delenv("LTX2_LORA_MODE", raising=False)
    monkeypatch.delenv("LTX2_COMPUTE_DTYPE", raising=False)


def _sd(tensors: dict[str, mx.array], strength: float = 1.0) -> LoraStateDictWithStrength:
    return LoraStateDictWithStrength(state_dict=StateDict(sd=tensors, size=0, dtype=set()), strength=strength)


def _factors(out_f: int, in_f: int, rank: int, seed: int, scale: float = 0.05) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    a = (scale * mx.random.normal((rank, in_f))).astype(mx.bfloat16)
    b = (scale * mx.random.normal((out_f, rank))).astype(mx.bfloat16)
    return a, b


def _lora_for(model: nn.Module, paths, rank: int = 4, seed: int = 3) -> dict[str, mx.array]:
    """A LoRA state dict targeting ``paths`` of ``model``, with factors shaped like the layers."""
    tensors = {}
    for i, path in enumerate(paths):
        layer = model
        for part in path.split("."):
            layer = layer[int(part)] if part.isdigit() else layer[part]
        out_f = layer.weight.shape[0]
        in_f = (
            layer.weight.shape[1] * 32 // layer.bits if isinstance(layer, nn.QuantizedLinear) else layer.weight.shape[1]
        )
        a, b = _factors(out_f, in_f, rank, seed + i)
        tensors[f"{path}.lora_A.weight"] = a
        tensors[f"{path}.lora_B.weight"] = b
    return tensors


def _rel(a: mx.array, ref: mx.array) -> float:
    a, ref = a.astype(mx.float32), ref.astype(mx.float32)
    return (mx.linalg.norm(a - ref) / mx.linalg.norm(ref)).item()


def _quantized_layer(out_f: int = 256, in_f: int = 512, seed: int = 0) -> nn.Module:
    """A q8 / group-64 linear with bf16 weights and scales, weights at a realistic 0.02 scale."""
    mx.random.seed(seed)
    holder = nn.Module()
    holder.layer = nn.Linear(in_f, out_f)
    holder.layer.weight = 0.02 * mx.random.normal((out_f, in_f))
    holder.set_dtype(mx.bfloat16)
    nn.quantize(holder, group_size=64, bits=8)
    mx.eval(holder.parameters())
    return holder


# ---- the adapter itself --------------------------------------------------------------------------


def test_adapter_equals_the_dense_weight_plus_the_lora():
    mx.random.seed(0)
    holder = nn.Module()
    holder.layer = nn.Linear(96, 64)
    w, bias = holder.layer.weight, holder.layer.bias
    a, b = _factors(64, 96, rank=8, seed=1)
    lora = [_sd({"layer.lora_A.weight": a, "layer.lora_B.weight": b}, strength=0.75)]
    x = mx.random.normal((5, 96))
    dense = w + 0.75 * b.astype(mx.float32) @ a.astype(mx.float32)

    handle = attach_loras(holder, lora, dtype=mx.float32)
    assert isinstance(holder.layer, LoRAAdapter) and isinstance(holder.layer, nn.Linear)
    assert len(handle) == 1
    assert _rel(holder.layer(x), x @ dense.T + bias) < 1e-5
    handle.detach()

    # Default storage: the LoRA's bf16, so 0.75 * B is rounded once (2^-9 of the delta).
    attach_loras(holder, lora)
    assert holder.layer.lora_b.dtype == mx.bfloat16
    assert _rel(holder.layer(x), x @ dense.T + bias) < 1e-3


def test_unfused_keeps_a_small_lora_that_fusion_loses_to_requantization():
    """The point of the change: on a q8 layer, fuse + re-quantize loses most of a 1 % delta."""
    holder = _quantized_layer()
    base = holder.layer
    in_f, out_f = 512, 256
    a, b = _factors(out_f, in_f, rank=16, seed=2)
    strength = 0.5
    delta = strength * b.astype(mx.float32) @ a.astype(mx.float32)
    w = mx.dequantize(base.weight, base.scales, base.biases, group_size=64, bits=8).astype(mx.float32)
    # A small steering LoRA, as the IC-LoRAs are: 1 % of the weight's norm.
    a = (a.astype(mx.float32) * (0.01 * mx.linalg.norm(w) / mx.linalg.norm(delta))).astype(mx.bfloat16)
    delta = strength * b.astype(mx.float32) @ a.astype(mx.float32)
    lora = {"layer.lora_A.weight": a, "layer.lora_B.weight": b}
    x = mx.random.normal((64, in_f))
    wanted = x @ delta.T  # what the LoRA should add to the output
    base_out = base(x)

    fused_holder = _quantized_layer()
    fused = apply_loras(
        StateDict(sd=dict(tree_flatten(fused_holder.parameters())), size=0, dtype=set()), [_sd(lora, strength)]
    )
    fused_holder.load_weights(list(fused.sd.items()))
    fused_err = _rel(fused_holder.layer(x) - base_out, wanted)

    attach_loras(holder, [_sd(lora, strength)])
    unfused_err = _rel(holder.layer(x) - base_out, wanted)

    assert fused_err > 0.3  # re-quantization error of the order of the delta itself (0.78 here)
    assert unfused_err < 1e-3  # the LoRA as shipped (2.5e-6 here)


def test_several_loras_on_one_layer_stack_and_detach_in_reverse():
    mx.random.seed(0)
    holder = nn.Module()
    holder.layer = nn.Linear(64, 32)
    w, bias = holder.layer.weight, holder.layer.bias
    loras = [_factors(32, 64, rank=r, seed=s) for r, s in ((4, 1), (2, 2), (3, 3))]
    keys = ("layer.lora_A.weight", "layer.lora_B.weight")
    x = mx.random.normal((7, 64))

    def ref(pairs):
        dense = w + sum(s * b.astype(mx.float32) @ a.astype(mx.float32) for (a, b), s in pairs)
        return x @ dense.T + bias

    first = attach_loras(
        holder, [_sd(dict(zip(keys, loras[0], strict=True)), 1.0), _sd(dict(zip(keys, loras[1], strict=True)), 0.5)]
    )
    after_first = holder.layer
    assert after_first.lora_a.shape[0] == 6  # ranks stacked: 4 + 2
    assert _rel(holder.layer(x), ref([(loras[0], 1.0), (loras[1], 0.5)])) < 1e-5

    second = attach_loras(holder, [_sd(dict(zip(keys, loras[2], strict=True)), 2.0)])
    assert holder.layer.lora_a.shape[0] == 9
    assert _rel(holder.layer(x), ref([(loras[0], 1.0), (loras[1], 0.5), (loras[2], 2.0)])) < 1e-5

    with pytest.raises(RuntimeError, match="reverse order"):
        first.detach()
    second.detach()
    assert holder.layer is after_first
    assert _rel(holder.layer(x), ref([(loras[0], 1.0), (loras[1], 0.5)])) < 1e-5
    first.detach()
    assert type(holder.layer) is nn.Linear
    assert mx.array_equal(holder.layer(x), x @ w.T + bias)


def test_unknown_keys_are_ignored_and_bad_shapes_raise():
    mx.random.seed(0)
    holder = nn.Module()
    holder.layer = nn.Linear(16, 8)
    a, b = _factors(8, 16, rank=2, seed=1)
    handle = attach_loras(holder, [_sd({"missing.lora_A.weight": a, "missing.lora_B.weight": b})])
    assert len(handle) == 0 and handle.skipped == ["missing"]
    assert type(holder.layer) is nn.Linear
    with pytest.raises(ValueError, match="layer"):
        attach_loras(holder, [_sd({"layer.lora_A.weight": b.T, "layer.lora_B.weight": a.T})])


# ---- on the DiT ----------------------------------------------------------------------------------


def test_detach_restores_the_exact_original_modules():
    model = _model()
    inputs = _inputs(model.config)
    before_out, _ = model(**inputs)
    modules = {p: _resolve(model, p) for p in _ADAPTED}
    params = dict(tree_flatten(model.parameters()))

    handle = attach_loras(model, [_sd(_lora_for(model, _ADAPTED))])
    assert len(handle) == len(_ADAPTED)
    assert all(isinstance(_resolve(model, p), LoRAAdapter) for p in _ADAPTED)
    adapted_out, _ = model(**inputs)
    assert not mx.array_equal(adapted_out, before_out)

    handle.detach()
    handle.detach()  # idempotent
    assert all(_resolve(model, p) is modules[p] for p in _ADAPTED)
    restored = dict(tree_flatten(model.parameters()))
    assert restored.keys() == params.keys()
    assert all(restored[k] is params[k] for k in params)
    after_out, _ = model(**inputs)
    assert mx.array_equal(after_out, before_out)


def test_the_handle_does_not_keep_a_freed_model_alive():
    """Pipelines free the DiT (self.dit = None) with the handle still around: nothing must leak."""
    import gc
    import weakref

    model = _model()
    to_q = model.transformer_blocks[0].attn1.to_q
    handle = attach_loras(model, [_sd(_lora_for(model, _ADAPTED))])
    assert "weight" not in to_q  # the replaced layer holds no parameters while the adapter has them
    model_ref = weakref.ref(model)
    weight_ref = weakref.ref(model.transformer_blocks[0].attn1.to_q.weight)
    del model, to_q
    gc.collect()
    assert model_ref() is None
    assert weight_ref() is None
    assert handle.detach() == 0


def test_adapted_model_keeps_its_weight_names():
    """Only lora_a / lora_b are added, so state-dict code (fusion, load_weights) still sees every weight."""
    model = _model()
    before = {k for k, _ in tree_flatten(model.parameters())}
    attach_loras(model, [_sd(_lora_for(model, _ADAPTED))])
    after = {k for k, _ in tree_flatten(model.parameters())}
    assert after - before == {f"{p}.{n}" for p in _ADAPTED for n in ("lora_a", "lora_b")}


def test_adapters_follow_the_compute_dtype():
    # Attached after the compute dtype is set: created in it, base parameters untouched (no recast needed).
    model = _model()
    model.set_compute_dtype(mx.float16)
    to_q = model.transformer_blocks[0].attn1.to_q
    scales = to_q.scales
    attach_loras(model, [_sd(_lora_for(model, _ADAPTED))])
    adapted = model.transformer_blocks[0].attn1.to_q
    assert adapted.lora_a.dtype == adapted.lora_b.dtype == mx.float16
    assert adapted.scales is scales
    assert model.transformer_blocks[0].attn1(mx.random.normal((1, 8, model.config.video_dim))).dtype == mx.float16

    # Attached first (the --lora load path): set_compute_dtype casts the factors with the rest.
    model = _model()
    attach_loras(model, [_sd(_lora_for(model, _ADAPTED))])
    assert model.transformer_blocks[0].attn1.to_q.lora_a.dtype == mx.bfloat16
    model.set_compute_dtype(mx.float16)
    assert model.transformer_blocks[0].attn1.to_q.lora_a.dtype == mx.float16
    assert model.transformer_blocks[1].ff.proj_in.lora_b.dtype == mx.float16


def _dequantized(model: nn.Module, deltas: dict[str, mx.array]) -> nn.Module:
    """Replace every quantized block linear by its float32 dequantization, plus ``deltas[path]`` if given."""
    for path, layer in [(p, m) for p, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)]:
        w = mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=64, bits=8).astype(mx.float32)
        dense = nn.Linear(w.shape[1], w.shape[0], bias="bias" in layer)
        dense.weight = w + deltas[path] if path in deltas else w
        if "bias" in layer:
            dense.bias = layer.bias.astype(mx.float32)
        parent, key = path.rsplit(".", 1)
        _resolve(model, parent)[key] = dense
    return model


def test_unfused_dit_output_is_closer_to_the_dense_reference_than_fused():
    """Whole-model version of the re-quantization test, with a small LoRA on every adapted layer."""
    lora = _lora_for(_model(), _ADAPTED, rank=4, seed=11)
    for path in _ADAPTED:  # shrink every delta to well under the weight's norm
        lora[f"{path}.lora_B.weight"] = (0.05 * lora[f"{path}.lora_B.weight"].astype(mx.float32)).astype(mx.bfloat16)
    deltas = {
        p: lora[f"{p}.lora_B.weight"].astype(mx.float32) @ lora[f"{p}.lora_A.weight"].astype(mx.float32)
        for p in _ADAPTED
    }
    inputs = _inputs(_model().config)

    # What the LoRA should change, measured in float32 on the dequantized weights.
    wanted = _dequantized(_model(), deltas)(**inputs)[0] - _dequantized(_model(), {})(**inputs)[0]
    base_out, _ = _model()(**inputs)

    fused = _model()
    fused_sd = apply_loras(StateDict(sd=dict(tree_flatten(fused.parameters())), size=0, dtype=set()), [_sd(lora)])
    apply_quantization(fused, fused_sd.sd)
    fused.load_weights(list(fused_sd.sd.items()))
    fused_out, _ = fused(**inputs)

    unfused = _model()
    attach_loras(unfused, [_sd(lora)])
    unfused_out, _ = unfused(**inputs)

    unfused_err = _rel(unfused_out - base_out, wanted)
    fused_err = _rel(fused_out - base_out, wanted)
    assert fused_err > 0.3, fused_err  # 1.39 here
    assert unfused_err < 0.02, unfused_err  # 3.5e-3 here: float32 vs quantized kernels downstream


def test_inplace_fusion_still_reaches_an_adapted_layer(tmp_path):
    """Stage 2 of --two-stage fuses the distilled LoRA into a DiT that may carry --lora adapters."""
    from ltx_pipelines_mlx._base import BasePipeline
    from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline

    path = "transformer_blocks.0.attn1.to_q"
    distilled = _lora_for(_model(), [path], rank=8, seed=21)
    mx.save_safetensors(str(tmp_path / "distilled-lora.safetensors"), distilled)
    user_lora = _lora_for(_model(), _ADAPTED, seed=31)

    def fuse_distilled(model):
        pipe = SimpleNamespace(
            low_ram_streaming=False,
            model_dir=tmp_path,
            dit=model,
            _distilled_lora="distilled-lora.safetensors",
            _distilled_lora_strength=1.0,
            _resolve_safetensors=BasePipeline._resolve_safetensors,
        )
        pipe._recast_after_inplace_fusion = lambda dit=None: BasePipeline._recast_after_inplace_fusion(pipe, dit)
        TI2VidTwoStagesPipeline._fuse_distilled_lora(pipe, model)

    adapted_first = _model()
    attach_loras(adapted_first, [_sd(user_lora)])
    fuse_distilled(adapted_first)

    fused_first = _model()
    fuse_distilled(fused_first)
    attach_loras(fused_first, [_sd(user_lora)])

    x = mx.random.normal((1, 8, adapted_first.config.video_dim))
    layer_a = adapted_first.transformer_blocks[0].attn1.to_q
    layer_b = fused_first.transformer_blocks[0].attn1.to_q
    assert isinstance(layer_a, LoRAAdapter)
    assert mx.array_equal(layer_a.weight, layer_b.weight)  # the distilled LoRA was fused under the adapter
    assert mx.array_equal(layer_a(x), layer_b(x))


def test_lora_file_with_comfy_keys(tmp_path):
    """The pipelines' loading path: a Comfy-named file through LTXV_LORA_COMFY_RENAMING_MAP."""
    from ltx_core_mlx.loader import LTXV_LORA_COMFY_RENAMING_MAP, SafetensorsStateDictLoader

    model = _model()
    native = _lora_for(model, ("transformer_blocks.1.attn1.to_out", "transformer_blocks.1.ff.proj_in"))
    comfy = {
        "diffusion_model." + k.replace(".to_out.", ".to_out.0.").replace(".ff.proj_in.", ".ff.net.0.proj."): v
        for k, v in native.items()
    }
    mx.save_safetensors(str(tmp_path / "lora.safetensors"), comfy)
    sd = SafetensorsStateDictLoader().load(str(tmp_path / "lora.safetensors"), sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
    handle = attach_loras(model, [LoraStateDictWithStrength(state_dict=sd, strength=1.0)])
    assert len(handle) == 2 and not handle.skipped
    assert isinstance(model.transformer_blocks[1].attn1.to_out, LoRAAdapter)
    assert isinstance(model.transformer_blocks[1].ff.proj_in, LoRAAdapter)


# ---- pipeline call sites -------------------------------------------------------------------------


def _save_lora(model, tmp_path, paths=_ADAPTED, name="lora.safetensors", seed=3):
    path = tmp_path / name
    mx.save_safetensors(str(path), _lora_for(model, paths, seed=seed))
    return str(path)


def _no_fusion(*args, **kwargs):
    raise AssertionError("unfused mode must not fuse")


def test_dfr_unfused_attach_and_in_place_detach(tmp_path, monkeypatch):
    import ltx_pipelines_mlx.dfr as dfr_mod
    from tests.test_dfr import _write_25_pack

    _write_25_pack(tmp_path)
    model = _model()
    lora_path = _save_lora(model, tmp_path)
    monkeypatch.setenv("LTX2_LORA_MODE", "unfused")  # read when the pipeline is built
    pipe = dfr_mod.DFRPipeline(str(tmp_path), low_memory=False, detailing_lora=lora_path)
    pipe.dit = model  # type: ignore[assignment]
    pipe._detailing_lora_path = lora_path
    originals = {p: _resolve(model, p) for p in _ADAPTED}
    weights = {p: originals[p].weight for p in _ADAPTED}
    monkeypatch.setattr(dfr_mod, "apply_loras", _no_fusion)
    monkeypatch.setattr(pipe, "_load_transformer_with_optional_streaming", _no_fusion)

    pipe._attach_detailing_lora()
    pipe._attach_detailing_lora()  # the epilogue re-attaches: replaced, never stacked
    assert pipe.dit is model
    for p in _ADAPTED:
        layer = _resolve(model, p)
        assert isinstance(layer, LoRAAdapter) and layer.lora_a.shape[0] == 4  # rank 4, once
        assert layer.weight is weights[p]  # base weights untouched
    b = _lora_for(model, _ADAPTED)[f"{_ADAPTED[0]}.lora_B.weight"]
    expected_b = (b.astype(mx.float32) * dfr_mod.DETAILING_LORA_STRENGTH).astype(mx.bfloat16)
    assert mx.array_equal(_resolve(model, _ADAPTED[0]).lora_b, expected_b)

    pipe._detach_detailing_lora()  # what the temporal rounds call: in place, no reload
    assert pipe.dit is model
    assert all(_resolve(model, p) is originals[p] and originals[p].weight is weights[p] for p in _ADAPTED)
    assert pipe._detailing_adapters is None


@pytest.mark.parametrize(("mode", "fuse"), [("unfused", False), ("fused", True)])
def test_dfr_detailing_lora_under_low_ram_follows_the_mode(tmp_path, monkeypatch, capsys, mode, fuse):
    """Under --low-ram the detailing LoRA streams as a BlockLoraSource; unfused makes it fuse=False (#192)."""
    import ltx_pipelines_mlx.dfr as dfr_mod
    from tests.test_dfr import _write_25_pack

    _write_25_pack(tmp_path)
    monkeypatch.setenv("LTX2_LORA_MODE", mode)
    pipe = dfr_mod.DFRPipeline(
        str(tmp_path), low_memory=False, low_ram_streaming=True, detailing_lora=str(tmp_path / "detail.safetensors")
    )
    pipe.dit = SimpleNamespace(_lora_sources=[])  # type: ignore[assignment]
    made = []
    monkeypatch.setattr(dfr_mod, "BlockLoraSource", lambda path, **kw: made.append((path, kw["fuse"])) or ("src", path))
    pipe._attach_detailing_lora()
    assert made == [(str(tmp_path / "detail.safetensors"), fuse)]
    assert pipe._detailing_adapters is None
    assert "does not apply" not in capsys.readouterr().err


def _ic_pipe(tmp_path, model, lora_paths, *, dev_mode=False, distilled=None, lora_mode="fused"):
    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline

    pipe = object.__new__(ICLoraPipeline)
    pipe.lora_mode = lora_mode  # what BasePipeline.__init__ parses from LTX2_LORA_MODE
    pipe.model_dir = tmp_path
    pipe.dev_mode = dev_mode
    pipe.distilled_lora_path = distilled
    pipe.distilled_lora_strength = 0.5
    pipe._lora_paths = list(lora_paths)
    pipe.low_ram_streaming = False
    pipe.dit = model
    return pipe


def test_ic_lora_default_mode_is_the_fusion_unchanged(tmp_path):
    """Env unset: _fuse_loras gives exactly what fusing with apply_loras gives (the pre-change code path)."""
    from ltx_core_mlx.loader import LTXV_LORA_COMFY_RENAMING_MAP, SafetensorsStateDictLoader

    lora_path = _save_lora(_model(), tmp_path)
    pipe = _ic_pipe(tmp_path, _model(), [(lora_path, 0.8)])
    pipe._fuse_loras()

    reference = _model()
    sd = SafetensorsStateDictLoader().load(lora_path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
    fused = apply_loras(
        StateDict(sd=dict(tree_flatten(reference.parameters())), size=0, dtype=set()),
        [LoraStateDictWithStrength(state_dict=sd, strength=0.8)],
    )
    apply_quantization(reference, fused.sd)
    reference.load_weights(list(fused.sd.items()))

    got = dict(tree_flatten(pipe.dit.parameters()))
    want = dict(tree_flatten(reference.parameters()))
    assert got.keys() == want.keys()
    assert all(mx.array_equal(got[k], want[k]) and got[k].dtype == want[k].dtype for k in want)
    assert not any(isinstance(_resolve(pipe.dit, p), LoRAAdapter) for p in _ADAPTED)


def test_ic_lora_unfused_attaches_and_reload_detaches_in_place(tmp_path, monkeypatch):
    import ltx_pipelines_mlx.ic_lora as ic_mod

    model = _model()
    originals = {p: _resolve(model, p) for p in _ADAPTED}
    pipe = _ic_pipe(tmp_path, model, [(_save_lora(model, tmp_path), 1.0)], lora_mode="unfused")
    monkeypatch.setattr(ic_mod, "apply_loras", _no_fusion)
    monkeypatch.setattr(pipe, "_load_transformer_with_optional_streaming", _no_fusion, raising=False)

    pipe._fuse_loras()
    pipe._fuse_loras()  # a second generate call: replaced, never stacked
    assert all(isinstance(_resolve(model, p), LoRAAdapter) for p in _ADAPTED)
    assert _resolve(model, _ADAPTED[0]).lora_a.shape[0] == 4
    assert isinstance(pipe._lora_adapters, AttachedLoras)

    pipe._reload_clean_transformer()  # stage 2 of the distilled path
    assert pipe.dit is model
    assert all(_resolve(model, p) is originals[p] for p in _ADAPTED)


def test_ic_lora_dev_mode_still_fuses_the_distilled_lora(tmp_path, monkeypatch):
    import ltx_pipelines_mlx.ic_lora as ic_mod

    model = _model()
    task = _save_lora(model, tmp_path, paths=_ADAPTED[:2], name="task.safetensors")
    distilled = _save_lora(model, tmp_path, paths=_ADAPTED[2:], name="distilled.safetensors", seed=9)
    pipe = _ic_pipe(tmp_path, model, [(task, 1.0)], dev_mode=True, distilled=distilled, lora_mode="unfused")
    pipe._recast_after_inplace_fusion = lambda dit=None: None
    fused_with: list = []
    real_apply = ic_mod.apply_loras

    def spy(**kwargs):
        fused_with.append([lsd.strength for lsd in kwargs["lora_sd_and_strengths"]])
        return real_apply(**kwargs)

    monkeypatch.setattr(ic_mod, "apply_loras", spy)
    pipe._fuse_loras()
    assert fused_with == [[0.5]]  # only the distilled LoRA, at its strength
    assert all(isinstance(_resolve(model, p), LoRAAdapter) for p in _ADAPTED[:2])
    assert not any(isinstance(_resolve(model, p), LoRAAdapter) for p in _ADAPTED[2:])


def test_pending_loras_unfused_attach_after_a_plain_load(tmp_path, monkeypatch):
    """generate --lora on a resident model: load as usual, then attach; no weight-dict fusion."""
    from pathlib import Path
    from unittest.mock import patch

    from ltx_pipelines_mlx._base import BasePipeline

    model = _model()
    lora_path = _save_lora(model, tmp_path)
    stub = SimpleNamespace(
        verbose=False, low_ram_streaming=False, lora_mode="unfused", _pending_loras=[(lora_path, 1.0)]
    )
    stub._fuse_pending_loras = _no_fusion
    monkeypatch.setenv("LTX2_COMPUTE_DTYPE", "float16")
    with patch("ltx_pipelines_mlx.utils._orchestration.load_transformer", return_value=model) as load:
        dit = BasePipeline._load_transformer_with_optional_streaming(stub, Path("/fake/transformer.safetensors"))
    load.assert_called_once()
    assert dit is model
    assert all(isinstance(_resolve(model, p), LoRAAdapter) for p in _ADAPTED)
    assert model.compute_dtype == mx.float16
    assert _resolve(model, _ADAPTED[0]).lora_a.dtype == mx.float16  # cast with the module, after the attach


@pytest.mark.parametrize(("value", "expected"), [("", "fused"), ("fused", "fused"), ("UNFUSED", "unfused")])
def test_env_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("LTX2_LORA_MODE", value)
    assert lora_mode_from_env() == expected


def test_env_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("LTX2_LORA_MODE", "adapter")
    with pytest.raises(ValueError, match="LTX2_LORA_MODE"):
        lora_mode_from_env()


def _resolve(model, path):
    node = model
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else node[part]
    return node


@pytest.mark.parametrize("low_ram", [False, True])
def test_a_bad_mode_fails_when_the_pipeline_is_built(tmp_path, monkeypatch, low_ram):
    """A typo stops the run before any work, also under --low-ram, where no LoRA path reads the mode."""
    import ltx_pipelines_mlx.dfr as dfr_mod
    from tests.test_dfr import _write_25_pack

    _write_25_pack(tmp_path)
    monkeypatch.setenv("LTX2_LORA_MODE", "unfuse")
    with pytest.raises(ValueError, match="LTX2_LORA_MODE"):
        dfr_mod.DFRPipeline(
            str(tmp_path),
            low_memory=False,
            low_ram_streaming=low_ram,
            detailing_lora=str(tmp_path / "detail.safetensors"),
        )


@pytest.mark.parametrize(("mode", "fuse"), [("unfused", False), ("fused", True)])
def test_pending_loras_under_low_ram_stream_with_the_mode(tmp_path, monkeypatch, capsys, mode, fuse):
    """generate --lora --low-ram: a BlockLoraSource per LoRA, fuse=False when unfused (#192)."""
    from pathlib import Path
    from unittest.mock import patch

    from ltx_pipelines_mlx._base import BasePipeline

    streamed = SimpleNamespace(_lora_sources=[])
    stub = SimpleNamespace(verbose=False, low_ram_streaming=True, lora_mode=mode, _pending_loras=[("l.st", 1.0)])
    stub._fuse_pending_loras = _no_fusion
    with (
        patch("ltx_pipelines_mlx.utils._orchestration.load_transformer", return_value=streamed),
        patch("ltx_pipelines_mlx.utils._orchestration.resolve_lora_path", side_effect=lambda p: p),
        patch(
            "ltx_core_mlx.loader.block_streaming.BlockLoraSource", side_effect=lambda p, **kw: ("src", p, kw["fuse"])
        ),
    ):
        dit = BasePipeline._load_transformer_with_optional_streaming(stub, Path("/fake/transformer.safetensors"))
    assert dit is streamed
    assert streamed._lora_sources == [("src", "l.st", fuse)]
    assert "does not apply" not in capsys.readouterr().err


def test_ic_lora_low_ram_unfused_streams_task_loras_unfused_and_the_distilled_lora_fused(tmp_path, monkeypatch):
    import ltx_pipelines_mlx.ic_lora as ic_mod
    from ltx_core_mlx.loader import block_streaming

    made = []
    monkeypatch.setattr(
        block_streaming, "BlockLoraSource", lambda path, **kw: made.append((path, kw["fuse"])) or ("src", path)
    )
    pipe = ic_mod.ICLoraPipeline.__new__(ic_mod.ICLoraPipeline)
    pipe.lora_mode = "unfused"
    pipe.low_ram_streaming = True
    pipe._lora_paths = [("task.safetensors", 1.0)]
    pipe._effective_lora_paths = lambda: [("task.safetensors", 1.0), ("distilled.safetensors", 0.5)]
    pipe.dit = SimpleNamespace(_lora_sources=[])
    pipe._fuse_loras()
    assert made == [("task.safetensors", False), ("distilled.safetensors", True)]
