"""Block-sparse video self-attention (Sol routing) for the distilled stage 2, opt-in through ``LTX2_SOL_TAU``.

The kernel is checked against MLX's dense attention (``tau = -inf`` routes every block, so the two must agree bit for
bit at head_dim 128 wherever both run the same steel loop; on GPUs where MLX sends dense attention to its NAX kernels
they agree to one rounding of the dtype, see ``_dense_attention_is_steel``) and against a plain-MLX reference of the Sol formula at finite
tau. The module, model, streaming and pipeline tests pin where it applies: the video self-attention of blocks 1 and up,
plain calls only (no mask, no STG perturbation), stage 2 only, one tau per step picked by sigma.
"""

from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from ltx_core_mlx.loader.block_streaming import BlockStreamer, StreamingLTXModel
from ltx_core_mlx.model.transformer import sparse_attention as sa
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.model.transformer.sparse_attention import (
    BLOCK,
    SparseAttentionState,
    block_summaries,
    build_route,
    kernel_available,
    sol_taus_from_env,
    sparse_self_attention,
)

pytestmark = pytest.mark.skipif(not kernel_available(), reason="sparse attention kernel unavailable on this MLX")

STAGE_2_SIGMAS = (0.909375, 0.725, 0.421875)
NVIDIA_TAUS = (1.0, 1.25, 1.5)


def _qkv(n: int, dtype: mx.Dtype, *, b: int = 2, h: int = 2, d: int = 128, seed: int = 0):
    """q, k, v as (B, H, N, D) views of (B, N, H, D) arrays: token-strided, like the projections in ``Attention``."""
    keys = mx.random.split(mx.random.key(seed), 3)
    return tuple(mx.random.normal((b, n, h, d), key=k).astype(dtype).transpose(0, 2, 1, 3) for k in keys)


def _reference(q, k, v, scale: float, tau: float, tokens_per_frame: int = 0, temporal: bool = True) -> mx.array:
    """Sol in plain MLX, float32, with the same routing: exact routed keys plus one centroid key per skipped block.

    Returns (B, N, H, D), like the kernel. Centroid keys / value means are rounded to the input dtype as the kernel
    receives them.
    """
    n = q.shape[2]
    qc, kc, vmean, log2_len = block_summaries(q, k, v)
    route = build_route(qc, kc, scale, tau, n, tokens_per_frame, temporal).astype(mx.bool_)
    blk = mx.arange(n) // BLOCK
    token_route = route[:, :, blk][:, :, :, blk]
    qf, kf, vf = (a.astype(mx.float32) for a in (q, k, v))
    exact = mx.where(token_route, (qf @ kf.swapaxes(-1, -2)) * scale, -mx.inf)
    centroid = (qf @ kc.astype(k.dtype).astype(mx.float32).swapaxes(-1, -2)) * scale + log2_len * math.log(2.0)
    centroid = mx.where(route[:, :, blk], -mx.inf, centroid)
    p = mx.softmax(mx.concatenate([exact, centroid], axis=-1), axis=-1)
    values = mx.concatenate([vf, vmean.astype(mx.float32)], axis=2)
    return (p @ values).transpose(0, 2, 1, 3)


def _dense_attention_is_steel(dtype: mx.Dtype) -> bool:
    """Whether ``mx.fast.scaled_dot_product_attention`` at head_dim 128 runs the steel loop the Sol kernel is built on.

    Mirrors MLX 0.32's dispatch (``is_nax_available`` in ``backend/metal/device.cpp``, the NAX branch of
    ``sdpa_full_self_attention_metal``): on macOS 26.2+ with an ``applegpu_g17`` or later GPU (``g18`` for the ``p``
    variants) dense attention runs ``steel_attention_nax`` in float16 / bfloat16, and in float32 too unless
    ``MLX_ENABLE_TF32=0``, which ``conftest.py`` sets for the suite. That kernel tiles and accumulates differently, so
    the results can differ by one rounding.
    """
    mac = tuple(int(x) for x in (platform.mac_ver()[0] or "0").split(".")[:2])
    match = re.fullmatch(r"applegpu_g(\d+)([a-z])", str(mx.device_info().get("architecture", "")))
    if mac < (26, 2) or match is None or int(match[1]) < (18 if match[2] == "p" else 17):
        return True
    assert os.environ.get("MLX_ENABLE_TF32") == "0", "conftest.py pins exact float32 attention"
    return dtype == mx.float32


def _assert_matches_dense(out: mx.array, dense: mx.array) -> None:
    """Bit for bit when dense attention runs the same steel loop; otherwise within one rounding of the dtype."""
    if _dense_attention_is_steel(out.dtype):
        assert mx.array_equal(out, dense).item()
    else:
        eps = {mx.float16: 2.0**-10, mx.bfloat16: 2.0**-7}[out.dtype]  # one rounding at |x| < 1
        assert mx.allclose(out.astype(mx.float32), dense.astype(mx.float32), rtol=eps, atol=eps).item()


def _rel_err(a: mx.array, ref: mx.array) -> float:
    a, ref = a.astype(mx.float32), ref.astype(mx.float32)
    return (mx.sqrt(mx.sum((a - ref) ** 2)) / mx.sqrt(mx.sum(ref**2))).item()


# --- LTX2_SOL_TAU -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "off", " OFF "])
def test_env_unset_or_off_is_dense(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LTX2_SOL_TAU", raising=False)
    else:
        monkeypatch.setenv("LTX2_SOL_TAU", value)
    assert sol_taus_from_env() is None


def test_env_parses_one_tau_per_step(monkeypatch):
    monkeypatch.setenv("LTX2_SOL_TAU", "1.0,1.25, 1.5")
    assert sol_taus_from_env() == NVIDIA_TAUS


@pytest.mark.parametrize("value", ["fast", "1.0,,1.5", "1.0;1.25", "nan", "1.0,inf"])
def test_env_rejects_malformed_values(monkeypatch, value):
    monkeypatch.setenv("LTX2_SOL_TAU", value)
    with pytest.raises(ValueError, match="LTX2_SOL_TAU"):
        sol_taus_from_env()


# --- kernel -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("n", [130, 1000, 4100])
def test_every_block_routed_is_dense_attention_bit_for_bit(dtype, n):
    """``tau = -inf``: the same steel loop over every key, i.e. ``mx.fast.scaled_dot_product_attention``."""
    q, k, v = _qkv(n, dtype)
    scale = 128**-0.5
    out = sparse_self_attention(q, k, v, scale, tau=-math.inf)
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale).transpose(0, 2, 1, 3)
    assert out.shape == (2, n, 2, 128) and out.dtype == dtype
    _assert_matches_dense(out, dense)


@pytest.mark.parametrize(("dtype", "tol"), [(mx.float32, 1e-5), (mx.float16, 1e-3), (mx.bfloat16, 5e-3)])
@pytest.mark.parametrize("tau", [0.0, 1.0])
@pytest.mark.parametrize("n", [700, 4100])
def test_matches_the_sol_reference(dtype, tol, tau, n):
    q, k, v = _qkv(n, dtype, seed=1)
    scale = 128**-0.5
    qc, kc, _, _ = block_summaries(q, k, v)
    density = build_route(qc, kc, scale, tau, n, 256, True).astype(mx.float32).mean().item()
    assert 0.05 < density < 0.95, "the test must skip blocks to exercise the centroid path"
    out = sparse_self_attention(q, k, v, scale, tau=tau, tokens_per_frame=256)
    assert _rel_err(out, _reference(q, k, v, scale, tau, 256)) < tol


@pytest.mark.parametrize("d", [64, 128])
def test_other_head_dims_and_contiguous_inputs(d):
    """Contiguous (B, H, N, D) inputs and head_dim 64 go through the same kernel."""
    q, k, v = (mx.random.normal((1, 3, 333, d), key=mx.random.key(i)).astype(mx.float16) for i in range(3))
    scale = d**-0.5
    out = sparse_self_attention(q, k, v, scale, tau=0.5)
    assert _rel_err(out, _reference(q, k, v, scale, 0.5)) < 1e-3


def test_block_summaries_are_block_means():
    q, k, v = _qkv(200, mx.float16)  # 4 blocks, the last one 8 tokens long
    qc, kc, vmean, log2_len = block_summaries(q, k, v)
    for i, (start, stop) in enumerate([(0, 64), (64, 128), (128, 192), (192, 200)]):
        assert mx.allclose(qc[:, :, i], q[:, :, start:stop].astype(mx.float32).mean(axis=2), atol=1e-5).item()
        assert mx.allclose(kc[:, :, i], k[:, :, start:stop].astype(mx.float32).mean(axis=2), atol=1e-5).item()
        mean_v = v[:, :, start:stop].astype(mx.float32).mean(axis=2)
        assert mx.allclose(vmean[:, :, i].astype(mx.float32), mean_v, atol=1e-3).item()
    assert vmean.dtype == mx.float16
    assert log2_len.tolist() == [6.0, 6.0, 6.0, 3.0]


def test_route_threshold_neighbours_and_temporal_blocks():
    n, tpf, tau = 1280, 256, 1.0  # 20 blocks, a latent frame = 4 blocks
    q, k, v = _qkv(n, mx.float32, b=1, h=1, seed=3)
    scale = 128**-0.5
    qc, kc, _, _ = block_summaries(q, k, v)
    route = build_route(qc, kc, scale, tau, n, tpf, temporal=True)[0, 0]
    plain = build_route(qc, kc, scale, tau, n, tpf, temporal=False)[0, 0]

    # Sol's "diag" threshold, written out: mean + tau * std of the query block's centroid scores, in log2 units.
    log2e = scale * math.log2(math.e)
    qb, kb = qc[0, 0], kc[0, 0]
    scores = (qb @ kb.T) * log2e
    k_mean = kb.mean(axis=0)
    k_var = ((kb - k_mean) ** 2).mean(axis=0)
    thr = (qb * k_mean).sum(-1) * log2e + tau * mx.sqrt((qb * qb * k_var).sum(-1) * log2e**2 + 1e-6)
    i = mx.arange(route.shape[0])
    expected = (scores > thr[:, None]) | (mx.abs(i[:, None] - i[None, :]) <= 1)
    assert mx.array_equal(plain.astype(mx.bool_), expected).item()

    for b in range(20):
        for other in (b - 4, b + 4):  # same (h, w) positions one latent frame before / after
            if 0 <= other < 20:
                assert route[b, other] == 1
    assert mx.all(route >= plain).item()  # temporal neighbours only add blocks
    assert mx.all(build_route(qc, kc, scale, -math.inf, n)).item()


# --- state --------------------------------------------------------------------------------------------------------


def test_state_picks_the_tau_of_the_step_by_sigma():
    state = SparseAttentionState(taus=NVIDIA_TAUS, sigmas=STAGE_2_SIGMAS)
    assert [state.tau_for_sigma(s) for s in STAGE_2_SIGMAS] == list(NVIDIA_TAUS)
    # the sampler's sigma may arrive bfloat16-rounded
    rounded = mx.array(STAGE_2_SIGMAS).astype(mx.bfloat16).astype(mx.float32).tolist()
    assert [state.tau_for_sigma(s) for s in rounded] == list(NVIDIA_TAUS)
    assert state.tau_for_sigma(1.0) is None  # not a step of this table: dense
    assert state.tau_for_sigma(0.5) is None
    short = SparseAttentionState(taus=(2.0,), sigmas=STAGE_2_SIGMAS)
    assert [short.tau_for_sigma(s) for s in STAGE_2_SIGMAS] == [2.0, 2.0, 2.0]  # the last tau repeats


def test_state_reads_the_latent_frame_from_the_positions():
    from ltx_core_mlx.utils.positions import compute_video_positions

    state = SparseAttentionState(taus=NVIDIA_TAUS, sigmas=STAGE_2_SIGMAS)
    positions = compute_video_positions(3, 4, 5, frame_rate=24.0)  # (1, 60, 3), frame-major
    state.prepare(mx.array([0.725]), positions)
    assert state.tau == 1.25 and state.tokens_per_frame == 20
    # appended conditioning tokens at time 0 (e.g. an IC-LoRA reference) do not change it
    with_reference = mx.concatenate([positions, positions[:, :20]], axis=1)
    state.prepare(mx.array([0.725]), with_reference)
    assert state.tokens_per_frame == 20
    state.prepare(mx.array([0.725]), positions[:, :20])  # one latent frame
    assert state.tokens_per_frame == 20
    state.prepare(mx.array([0.3]), None)
    assert state.tau is None and state.tokens_per_frame == 0


# --- Attention module ---------------------------------------------------------------------------------------------


def _attention(seed: int = 0) -> Attention:
    mx.random.seed(seed)
    attn = Attention(query_dim=256, num_heads=2, head_dim=128)
    attn.set_dtype(mx.float16)
    mx.eval(attn.parameters())
    return attn


def _state(tau: float, min_tokens: int = 64) -> SparseAttentionState:
    state = SparseAttentionState(taus=(tau,), sigmas=(0.5,), min_tokens=min_tokens)
    state.prepare(mx.array([0.5]), None)
    return state


def test_attention_runs_sparse_only_on_plain_self_attention_calls():
    attn = _attention()
    x = mx.random.normal((1, 300, 256)).astype(mx.float16)
    dense = attn(x)

    attn.sparse_attention = _state(-math.inf)
    _assert_matches_dense(attn(x), dense)  # every block routed: the dense result
    assert attn.sparse_attention.calls == 1

    attn.sparse_attention = state = _state(1.0)
    sparse = attn(x)
    assert state.calls == 1
    # random weights leave the routing no structure to keep: correctness is pinned by the kernel tests
    assert 0 < _rel_err(sparse, dense) < 1.0

    text = mx.random.normal((1, 7, 256)).astype(mx.float16)
    mask = mx.ones((1, 1, 300, 300), dtype=mx.bool_)
    perturbation = mx.ones((1, 1, 1, 1))
    for kwargs in ({"encoder_hidden_states": text}, {"attention_mask": mask}, {"perturbation_mask": perturbation}):
        attn.sparse_attention = None
        expected = attn(x, **kwargs)
        attn.sparse_attention = state
        assert mx.array_equal(attn(x, **kwargs), expected).item(), kwargs
    assert state.calls == 1  # none of those ran sparse

    attn.sparse_attention = short = _state(1.0, min_tokens=301)
    assert mx.array_equal(attn(x), dense).item()
    assert short.calls == 0

    attn.sparse_attention = idle = _state(1.0)
    idle.tau = None  # a sigma outside the table
    assert mx.array_equal(attn(x), dense).item()


# --- LTXModel -----------------------------------------------------------------------------------------------------


def _config(num_layers: int = 3) -> LTXModelConfig:
    return LTXModelConfig(
        num_layers=num_layers,
        video_dim=128,
        audio_dim=64,
        video_num_heads=1,
        audio_num_heads=4,
        video_head_dim=128,  # the kernel's production head dim
        audio_head_dim=16,
        av_cross_num_heads=4,
        av_cross_head_dim=16,
        video_patch_channels=64,
        audio_patch_channels=64,
        ff_mult=2.0,
        timestep_embedding_dim=64,
    )


def _model(seed: int = 0, num_layers: int = 3) -> LTXModel:
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
        block.scale_shift_table = 0.05 * mx.random.normal(block.scale_shift_table.shape)
    mx.eval(model.parameters())
    return model


def _inputs(cfg: LTXModelConfig, sigma: float = 0.725, seed: int = 1) -> dict:
    from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions

    mx.random.seed(seed)
    f, h, w = 4, 8, 16  # 512 video tokens: 8 blocks, a latent frame = 2 blocks
    nv, na, nt = f * h * w, 8, 6
    return dict(
        video_latent=mx.random.normal((1, nv, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, na, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([sigma]),
        video_text_embeds=mx.random.normal((1, nt, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, nt, cfg.audio_dim)).astype(mx.bfloat16),
        video_positions=compute_video_positions(f, h, w, frame_rate=24.0),
        audio_positions=compute_audio_positions(na),
    )


def _model_state(taus=NVIDIA_TAUS) -> SparseAttentionState:
    return SparseAttentionState(taus=taus, sigmas=STAGE_2_SIGMAS, min_tokens=64)


def test_model_attaches_the_state_to_blocks_one_and_up():
    model = _model()
    assert model.sparse_attention is None
    state = _model_state()
    model.set_sparse_attention(state)
    assert model.sparse_attention is state
    assert model.transformer_blocks[0].attn1.sparse_attention is None  # block 0 dense, as NVIDIA's config
    for block in model.transformer_blocks[1:]:
        assert block.attn1.sparse_attention is state
        for name in ("audio_attn1", "attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"):
            assert block[name].sparse_attention is None
    model.set_sparse_attention(None)
    assert model.sparse_attention is None
    assert all(block.attn1.sparse_attention is None for block in model.transformer_blocks)


def test_model_forward_routes_per_step_and_stays_exact_when_every_block_is_routed():
    model = _model()
    inputs = _inputs(model.config)
    ref_v, ref_a = model(**inputs)

    model.set_sparse_attention(exact := _model_state(taus=(-math.inf,)))
    v, a = model(**inputs)
    assert exact.calls == model.config.num_layers - 1
    assert mx.array_equal(v, ref_v).item() and mx.array_equal(a, ref_a).item()

    model.set_sparse_attention(state := _model_state())
    v, _ = model(**inputs)
    assert state.tau == 1.25 and state.tokens_per_frame == 128  # sigma 0.725 is the second step
    assert state.calls == model.config.num_layers - 1
    assert 0 < _rel_err(v, ref_v) < 0.1

    off_table = _inputs(model.config, sigma=0.6)
    dense_v, _ = _model()(**off_table)
    v, _ = model(**off_table)
    assert state.tau is None and state.calls == model.config.num_layers - 1  # unchanged: that forward was dense
    assert mx.array_equal(v, dense_v).item()


def test_model_recompute_after_overflow_keeps_the_step(capsys):
    """The float16 overflow guard re-enters ``__call__`` with the same arguments: same sigma, same tau."""
    model = _model()
    ff = model.transformer_blocks[0].ff.proj_out
    ff.scales = ff.scales * 5e5  # finite in float32, past float16's range inside the feed-forward
    mx.eval(model.parameters())
    model.set_compute_dtype(mx.float16)
    model.set_sparse_attention(state := _model_state())
    model(**_inputs(model.config, sigma=STAGE_2_SIGMAS[2]))
    assert "not finite" in capsys.readouterr().err
    assert state.tau == 1.5
    assert state.calls == 2 * (model.config.num_layers - 1)  # the first pass and the recompute both ran sparse


def test_tiled_model_reads_each_tiles_latent_frame():
    """``--tile-*``: each tile is its own forward over its own tokens, so the frame size is the tile's."""
    from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler
    from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig

    model = _model()
    tiler = VideoModalityTiler(
        TileCountConfig(width=DimensionTilingConfig(num_tiles=2, overlap=0)), latent_shape=(4, 8, 16)
    )
    state = _model_state()
    seen = []
    prepare = state.prepare

    def spy(timestep, video_positions):
        prepare(timestep, video_positions)
        seen.append((state.tau, state.tokens_per_frame, video_positions.shape[1]))

    state.prepare = spy  # type: ignore[method-assign]
    model.set_sparse_attention(state)
    TiledLTXModel(model, tiler)(**_inputs(model.config))
    assert seen == [(1.25, 64, 256), (1.25, 64, 256)]  # two 4 x 8 x 8 tiles
    assert state.calls == 2 * (model.config.num_layers - 1)


def _save_blocks(model: LTXModel, path: Path) -> None:
    flat = {}
    for i, block in enumerate(model.transformer_blocks):
        for k, v in tree_flatten(block.parameters()):
            flat[f"transformer_blocks.{i}.{k}"] = v
    mx.save_safetensors(str(path), flat)


def test_streamed_model_follows_the_bound_block():
    resident = _model()
    inputs = _inputs(resident.config)
    resident.set_sparse_attention(resident_state := _model_state())
    ref_v, ref_a = resident(**inputs)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "blocks.safetensors"
        _save_blocks(_model(), path)
        inner = _model()
        inner.transformer_blocks = [inner.transformer_blocks[0]]
        streamed = StreamingLTXModel(inner, BlockStreamer(path, block_prefix="transformer_blocks."))
        streamed.set_sparse_attention(state := _model_state())
        assert streamed.sparse_attention is state
        v, a = streamed(**inputs)
        assert state.calls == resident_state.calls == resident.config.num_layers - 1
        assert _rel_err(v, ref_v) < 1e-5 and _rel_err(a, ref_a) < 1e-5

        streamed.set_sparse_attention(None)
        assert inner.transformer_blocks[0].attn1.sparse_attention is None
        dense_v, _ = _model()(**inputs)
        v, _ = streamed(**inputs)
        assert _rel_err(v, dense_v) < 1e-5


# --- pipeline -----------------------------------------------------------------------------------------------------


class _RecordingDit:
    """Stand-in DiT: records the sparse-attention state it is given."""

    def __init__(self):
        self.sparse_attention: SparseAttentionState | None = None
        self.history: list[SparseAttentionState | None] = []

    def set_sparse_attention(self, state):
        self.sparse_attention = state
        self.history.append(state)


def _stubbed_distilled(tmp_path, monkeypatch, taus: str | None):
    import json

    from ltx_pipelines_mlx import distilled as distilled_mod
    from ltx_pipelines_mlx.distilled import DistilledPipeline
    from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

    if taus is None:
        monkeypatch.delenv("LTX2_SOL_TAU", raising=False)
    else:
        monkeypatch.setenv("LTX2_SOL_TAU", taus)
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    pipe = DistilledPipeline(str(tmp_path), low_memory=False)
    pipe._load_text_encoder = lambda: None  # type: ignore[method-assign]
    pipe._encode_text = lambda prompt: (  # type: ignore[method-assign]
        mx.zeros((1, 8, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 8, 2048), dtype=mx.bfloat16),
    )
    pipe.load = lambda: None  # type: ignore[method-assign]
    pipe.dit = dit = _RecordingDit()  # type: ignore[assignment]

    class _Vae:
        def denormalize_latent(self, x):
            return x

        def normalize_latent(self, x):
            return x

    pipe.vae_encoder = _Vae()  # type: ignore[assignment]
    pipe.upsampler = lambda x: mx.repeat(mx.repeat(x, 2, axis=3), 2, axis=4)  # type: ignore[assignment]
    monkeypatch.setattr(distilled_mod, "X0Model", lambda m: m)

    seen: list[SparseAttentionState | None] = []

    def loop(**kwargs):
        seen.append(dit.sparse_attention)
        return DenoiseOutput(video_latent=kwargs["video_state"].latent, audio_latent=kwargs["audio_state"].latent)

    monkeypatch.setattr(distilled_mod, "euler_ancestral_denoising_loop", loop)
    monkeypatch.setattr(distilled_mod, "denoise_loop", loop)
    return pipe, dit, seen


def test_distilled_pipeline_turns_it_on_for_stage_2_only(tmp_path, monkeypatch, capsys):
    from ltx_pipelines_mlx.scheduler import LTX_2_5_STAGE_2_DISTILLED_SIGMAS

    pipe, dit, seen = _stubbed_distilled(tmp_path, monkeypatch, "1.0,1.25,1.5")
    assert pipe.sol_taus == NVIDIA_TAUS
    pipe.generate_two_stage(prompt="a fox", height=128, width=128, num_frames=9, frame_rate=24.0, seed=7)

    stage_1, stage_2 = seen
    assert stage_1 is None
    assert stage_2 is not None
    assert stage_2.taus == NVIDIA_TAUS
    assert stage_2.sigmas == tuple(LTX_2_5_STAGE_2_DISTILLED_SIGMAS[:-1])
    assert dit.history == [stage_2, None]  # off again after stage 2
    err = capsys.readouterr().err
    assert "[sparse-attention] stage 2" in err and "tau 1.5" in err


def test_distilled_pipeline_default_never_touches_the_dit(tmp_path, monkeypatch):
    pipe, dit, seen = _stubbed_distilled(tmp_path, monkeypatch, None)
    assert pipe.sol_taus is None
    pipe.generate_two_stage(prompt="a fox", height=128, width=128, num_frames=9, frame_rate=24.0, seed=7)
    assert seen == [None, None]
    assert dit.history == []


def test_stage_2_turns_it_off_when_the_loop_raises(tmp_path, monkeypatch):
    from ltx_pipelines_mlx import distilled as distilled_mod

    pipe, dit, seen = _stubbed_distilled(tmp_path, monkeypatch, "1.0")
    calls = {"n": 0}

    def failing_loop(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("stage 2 failed")
        from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

        return DenoiseOutput(video_latent=kwargs["video_state"].latent, audio_latent=kwargs["audio_state"].latent)

    monkeypatch.setattr(distilled_mod, "euler_ancestral_denoising_loop", failing_loop)
    with pytest.raises(RuntimeError, match="stage 2 failed"):
        pipe.generate_two_stage(prompt="a fox", height=128, width=128, num_frames=9, frame_rate=24.0, seed=7)
    assert dit.sparse_attention is None
    assert len(dit.history) == 2 and dit.history[-1] is None


def test_malformed_env_fails_when_the_pipeline_is_built(tmp_path, monkeypatch):
    from ltx_pipelines_mlx.distilled import DistilledPipeline

    monkeypatch.setenv("LTX2_SOL_TAU", "fast")
    with pytest.raises(ValueError, match="LTX2_SOL_TAU"):
        DistilledPipeline(str(tmp_path), low_memory=False)


def test_kernel_unavailable_falls_back_to_dense(monkeypatch, capsys):
    monkeypatch.setattr(sa, "_AVAILABLE", {mx.float16: False})
    attn = _attention()
    x = mx.random.normal((1, 300, 256)).astype(mx.float16)
    dense = attn(x)
    attn.sparse_attention = state = _state(1.0)
    assert mx.array_equal(attn(x), dense).item()
    assert state.calls == 0
    q, k, v = _qkv(130, mx.float16)
    with pytest.raises(RuntimeError, match="unavailable"):
        sparse_self_attention(q, k, v, 128**-0.5, tau=1.0)


def _break_attention_kernel(monkeypatch, broken: mx.Dtype | None) -> list[dict]:
    """Make the attention kernel fail to build for ``broken`` only (a future MLX breaking one steel instantiation;
    ``None`` breaks nothing) and clear the probe cache. Returns the list every attention-kernel call is recorded in."""
    real = sa._kernels()
    calls: list[dict] = []

    def attn(**kwargs):
        calls.append(kwargs)
        if broken is not None and dict(kwargs["template"])["T"] == broken:
            raise RuntimeError(f"steel attention failed to compile for {broken}")
        return real["attn"](**kwargs)

    monkeypatch.setattr(sa, "_KERNELS", {"attn": attn, "sum": real["sum"]})
    monkeypatch.setattr(sa, "_AVAILABLE", {})
    return calls


def test_probe_runs_the_kernel_variant_of_a_real_call(monkeypatch):
    # mx.fast.metal_kernel passes inputs with fewer than 8 elements in constant memory, which builds another kernel:
    # the probe must put every array input but the scalars (ip, sc) in device memory, as a real call (nb >= 64) does.
    calls = _break_attention_kernel(monkeypatch, broken=None)
    assert kernel_available(mx.float32)
    (call,) = calls
    names = ["q", "k", "v", "kc", "vc", "cb", "route", "ip", "sc"]
    sizes = {name: a.size for name, a in zip(names, call["inputs"], strict=True)}
    assert all(sizes[name] >= 8 for name in names[:7]), sizes
    assert dict(call["template"])["T"] == mx.float32
    assert call["inputs"][0].shape == (1, 2, sa._PROBE_TOKENS, 128)


def test_float32_probe_passes_when_mlx_runs_dense_attention_in_tf32():
    """With MLX's default ``MLX_ENABLE_TF32=1`` a NAX GPU runs float32 dense attention at ~10 mantissa bits; the probe
    must still accept the (exact) kernel. A fresh process, since MLX reads the variable once."""
    code = (
        "import mlx.core as mx\n"
        "from ltx_core_mlx.model.transformer.sparse_attention import kernel_available\n"
        "assert kernel_available(mx.float32)\n"
    )
    env = {**os.environ, "MLX_ENABLE_TF32": "1"}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "staying on dense attention" not in result.stderr


@pytest.mark.parametrize("broken", [mx.float32, mx.bfloat16])
def test_kernel_broken_for_one_dtype_falls_back_to_dense_for_that_dtype_only(monkeypatch, capsys, broken):
    _break_attention_kernel(monkeypatch, broken)

    # The broken dtype stays dense for the whole render, with one warning, instead of failing mid stage 2.
    attn = _attention()
    attn.set_dtype(broken)
    x = mx.random.normal((1, 300, 256)).astype(broken)
    dense = attn(x)
    attn.sparse_attention = state = _state(1.0)
    assert mx.array_equal(attn(x), dense).item()
    assert mx.array_equal(attn(x), dense).item()
    assert state.calls == 0
    err = capsys.readouterr().err
    assert err.count("staying on dense attention") == 1 and str(broken) in err

    # The other dtypes are probed on their own and still run sparse.
    assert kernel_available(mx.float16)
    attn16 = _attention()
    attn16.sparse_attention = state16 = _state(1.0)
    attn16(mx.random.normal((1, 300, 256)).astype(mx.float16))
    assert state16.calls == 1
    assert {broken: False, mx.float16: True} == sa._AVAILABLE
