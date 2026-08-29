"""Bit-exactness tests for the AdaLN per-token dedupe and deferred gather.

The dedupe replaces a ``B*N``-row AdaLN GEMM with a handful-of-rows GEMM plus
a gather. The deferred gather then keeps the deduplicated form all the way into
the transformer blocks, so the full ``(B*N, num_params*dim)`` float32 tensor is
never materialised. Neither is allowed to be used unless it reproduces the
original *bit for bit*, so every test here asserts ``mx.array_equal`` (exact),
never ``allclose``.

Runnable either under pytest or directly::

    python tests/test_adaln_dedupe.py
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.model.transformer import model as model_mod
from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle, PerTokenAdaLNParams
from ltx_core_mlx.model.transformer.model import LTXModel
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

TIMESTEP_DIM = 256
SCALE = 1000.0


def _embed(per_token_timesteps: mx.array) -> mx.array:
    """Mirror ``LTXModel._embed_timestep_per_token`` without building a model."""
    b, n = per_token_timesteps.shape
    flat = (per_token_timesteps * SCALE).reshape(-1)
    return get_timestep_embedding(flat, TIMESTEP_DIM).reshape(b, n, -1)


def _module(dim: int, num_params: int, seed: int = 0) -> AdaLayerNormSingle:
    mx.random.seed(seed)
    mod = AdaLayerNormSingle(dim, num_params=num_params, timestep_dim=TIMESTEP_DIM)
    mod.set_dtype(mx.bfloat16)  # matches the shipped checkpoints: AdaLN stays bf16
    mx.eval(mod.parameters())
    return mod


def _reference(mod: AdaLayerNormSingle, t_emb: mx.array) -> tuple[mx.array, mx.array]:
    """The pre-optimisation implementation, verbatim."""
    b, n, d = t_emb.shape
    params, embedded = mod(t_emb.reshape(b * n, d))
    return params.reshape(b, n, -1), embedded.reshape(b, n, -1)


def _materialise(x):
    """Expand a deferred-gather carrier; pass plain arrays through."""
    return x.gather() if isinstance(x, PerTokenAdaLNParams) else x


def _deduped(mod: AdaLayerNormSingle, t_emb: mx.array) -> tuple[mx.array, mx.array]:
    params, embedded = LTXModel._adaln_per_token(None, mod, t_emb)  # method never touches `self`
    return _materialise(params), _materialise(embedded)


def _assert_identical(mod: AdaLayerNormSingle, t_emb: mx.array, label: str) -> None:
    ref_p, ref_e = _reference(mod, t_emb)
    got_p, got_e = _deduped(mod, t_emb)
    mx.eval(ref_p, ref_e, got_p, got_e)
    assert got_p.shape == ref_p.shape, f"{label}: params shape {got_p.shape} != {ref_p.shape}"
    assert got_e.shape == ref_e.shape, f"{label}: embedded shape {got_e.shape} != {ref_e.shape}"
    assert bool(mx.array_equal(got_p, ref_p).item()), f"{label}: params not bit-identical"
    assert bool(mx.array_equal(got_e, ref_e).item()), f"{label}: embedded not bit-identical"
    del ref_p, ref_e, got_p, got_e
    mx.clear_cache()


# --- sigma patterns ---------------------------------------------------------


def _pattern(name: str, b: int, n: int, sigma: float = 0.7, seed: int = 3) -> mx.array:
    """Build a (B, N) per-token timestep array for a given conditioning shape."""
    rng = np.random.default_rng(seed)
    if name == "uniform":  # no conditioning: every token at the same sigma
        vals = np.full((b, n), sigma, dtype=np.float32)
    elif name == "two":  # i2v / extend: conditioning at 0, target at sigma
        vals = np.full((b, n), sigma, dtype=np.float32)
        vals[:, : n // 8] = 0.0
    elif name == "four":  # several conditioning groups at different strengths
        vals = np.full((b, n), sigma, dtype=np.float32)
        vals[:, : n // 16] = 0.0
        vals[:, n // 16 : n // 8] = sigma * 0.25
        vals[:, n // 8 : n // 6] = sigma * 0.5
    elif name == "ramp":  # worst case: a continuous per-token sigma ramp
        vals = np.linspace(0.0, sigma, n, dtype=np.float32)[None, :].repeat(b, axis=0)
        vals = vals + rng.standard_normal((b, n)).astype(np.float32) * 1e-6
    else:  # pragma: no cover
        raise ValueError(name)
    return mx.array(vals).astype(mx.bfloat16)


PATTERNS = ("uniform", "two", "four", "ramp")
SHAPES = ((1, 8192), (1, 20000), (1, 30000), (2, 8192))


def test_dedupe_is_bit_identical_video() -> None:
    """9-param video AdaLN at 4096, every sigma pattern x realistic shapes."""
    mod = _module(4096, 9)
    for pattern in PATTERNS:
        for batch, tokens in SHAPES:
            _assert_identical(mod, _embed(_pattern(pattern, batch, tokens)), f"video/{pattern}/{batch}x{tokens}")


def test_dedupe_is_bit_identical_other_heads() -> None:
    """Audio (2048) and the 4-param AV cross-attention heads."""
    for dim, num_params in ((2048, 9), (4096, 4), (2048, 4)):
        mod = _module(dim, num_params, seed=dim + num_params)
        for pattern in PATTERNS:
            _assert_identical(mod, _embed(_pattern(pattern, 1, 20000)), f"{dim}/{num_params}/{pattern}")


def test_dedupe_repeats_are_stable_across_calls() -> None:
    """After calibration the fast path must keep matching, step after step."""
    mod = _module(4096, 9, seed=11)
    for sigma in (1.0, 0.8, 0.55, 0.3, 0.05):
        _assert_identical(mod, _embed(_pattern("two", 1, 20000, sigma=sigma)), f"sigma={sigma}")


def test_small_inputs_take_the_original_path() -> None:
    """Below the row threshold nothing is deduped, and results still match."""
    mod = _module(2048, 9, seed=5)
    _assert_identical(mod, _embed(_pattern("two", 1, 512)), "small")


def test_kill_switch_disables_dedupe() -> None:
    prev = model_mod._ADALN_DEDUPE
    model_mod._ADALN_DEDUPE = False
    try:
        mod = _module(2048, 9, seed=7)
        _assert_identical(mod, _embed(_pattern("two", 1, 8192)), "kill-switch")
    finally:
        model_mod._ADALN_DEDUPE = prev


def test_unique_rows_grouping_is_exact() -> None:
    """The grouping key must never merge rows that are not bitwise equal."""
    rng = np.random.default_rng(0)
    reps = rng.standard_normal((5, TIMESTEP_DIM)).astype(np.float32)
    idx = rng.integers(0, 5, size=8192)
    flat = mx.array(reps[idx])
    grouped = model_mod._unique_rows(flat)
    assert grouped is not None
    got_reps, inv, u = grouped
    assert u == 5, f"expected 5 unique rows, got {u}"
    assert bool(mx.array_equal(mx.take(got_reps, inv, axis=0), flat).item())


def test_unique_rows_declines_on_continuous_ramp() -> None:
    """A continuous per-token sigma ramp has no duplication to exploit."""
    vals = np.linspace(0.0, 1.0, 8192, dtype=np.float32)
    flat = get_timestep_embedding(mx.array(vals) * SCALE, TIMESTEP_DIM)
    assert model_mod._unique_rows(flat) is None


# --- deferred per-block gather ----------------------------------------------


def _eager_unpack(params: mx.array, table: mx.array, num_params: int, dim: int) -> list[mx.array]:
    """The pre-optimisation ``_unpack_adaln`` per-token branch, verbatim."""
    b, n, _ = params.shape
    p = params.reshape(b, n, num_params, dim)
    p = p + table[None, None, :num_params, :]
    return [p[:, :, i, :] for i in range(num_params)]


def _assert_unpack_identical(dim: int, num_params: int, pattern: str, batch: int, tokens: int) -> None:
    mod = _module(dim, num_params, seed=dim + num_params)
    t_emb = _embed(_pattern(pattern, batch, tokens))
    label = f"{dim}/{num_params}/{pattern}/{batch}x{tokens}"

    # The plan lives on the module and the calibrating (first) call returns
    # the exact reference, not a carrier -- warm it up, then exercise dedupe.
    LTXModel._adaln_per_token(None, mod, t_emb)
    lazy_params, _ = LTXModel._adaln_per_token(None, mod, t_emb)
    assert isinstance(lazy_params, PerTokenAdaLNParams), f"{label}: expected a deferred-gather carrier"
    assert lazy_params.shape == (batch, tokens, num_params * dim), f"{label}: carrier reports {lazy_params.shape}"
    assert lazy_params.unique_rows < tokens, f"{label}: carrier kept {lazy_params.unique_rows} rows"

    table = mx.random.normal((num_params, dim)).astype(mx.bfloat16)
    mx.eval(table)

    eager = _eager_unpack(lazy_params.gather(), table, num_params, dim)
    lazy = BasicAVTransformerBlock._unpack_adaln(lazy_params, table, num_params, dim)
    mx.eval(eager, lazy)
    assert len(lazy) == len(eager)
    for i, (got, ref) in enumerate(zip(lazy, eager, strict=True)):
        assert got.shape == ref.shape, f"{label}: param {i} shape {got.shape} != {ref.shape}"
        assert got.dtype == ref.dtype, f"{label}: param {i} dtype {got.dtype} != {ref.dtype}"
        assert bool(mx.array_equal(got, ref).item()), f"{label}: param {i} not bit-identical"
    del eager, lazy
    mx.clear_cache()


def test_lazy_unpack_is_bit_identical() -> None:
    """Deferred per-block expansion == eager unpack of the gathered tensor."""
    for pattern in ("uniform", "two", "four"):
        _assert_unpack_identical(4096, 9, pattern, 1, 20000)
    for dim, num_params in ((2048, 9), (4096, 4), (2048, 4)):
        _assert_unpack_identical(dim, num_params, "two", 1, 20000)
    _assert_unpack_identical(4096, 9, "two", 2, 8192)  # batch > 1
    _assert_unpack_identical(4096, 9, "ramp", 1, 20000)  # worst case, still deduped


def test_lazy_carrier_holds_only_the_distinct_rows() -> None:
    """The whole point: what survives the call is a handful of rows, not B*N."""
    mod = _module(4096, 9, seed=21)
    t_emb = _embed(_pattern("two", 1, 20000))
    LTXModel._adaln_per_token(None, mod, t_emb)  # calibrating call returns the reference
    params, embedded = LTXModel._adaln_per_token(None, mod, t_emb)
    assert isinstance(params, PerTokenAdaLNParams)
    assert isinstance(embedded, PerTokenAdaLNParams)
    assert params.unique_rows == 2, f"expected 2 distinct rows, got {params.unique_rows}"
    assert embedded.unique_rows == 2, f"expected 2 distinct rows, got {embedded.unique_rows}"
    assert params.shape == (1, 20000, 9 * 4096)
    assert embedded.shape == (1, 20000, 4096)


def test_lazy_kill_switch_materialises_eagerly() -> None:
    """LTX2_ADALN_LAZY=0 restores the eager gather, bit for bit."""
    mod = _module(2048, 9, seed=13)
    t_emb = _embed(_pattern("two", 1, 8192))
    ref_p, ref_e = _reference(mod, t_emb)

    prev = model_mod._ADALN_LAZY
    model_mod._ADALN_LAZY = False
    try:
        got_p, got_e = LTXModel._adaln_per_token(None, mod, t_emb)
    finally:
        model_mod._ADALN_LAZY = prev
    assert isinstance(got_p, mx.array), "kill switch must return a plain array"
    assert isinstance(got_e, mx.array), "kill switch must return a plain array"
    mx.eval(ref_p, ref_e, got_p, got_e)
    assert bool(mx.array_equal(got_p, ref_p).item()), "kill-switch params not bit-identical"
    assert bool(mx.array_equal(got_e, ref_e).item()), "kill-switch embedded not bit-identical"


def test_lazy_declines_when_dedupe_declines() -> None:
    """Nothing to defer without the dedupe -- both fallbacks return arrays."""
    mod = _module(2048, 9, seed=17)
    small = _embed(_pattern("two", 1, 512))  # below the dedupe row threshold
    params, embedded = LTXModel._adaln_per_token(None, mod, small)
    assert isinstance(params, mx.array) and isinstance(embedded, mx.array)

    prev = model_mod._ADALN_DEDUPE
    model_mod._ADALN_DEDUPE = False
    try:
        params, embedded = LTXModel._adaln_per_token(None, mod, _embed(_pattern("two", 1, 8192)))
    finally:
        model_mod._ADALN_DEDUPE = prev
    assert isinstance(params, mx.array) and isinstance(embedded, mx.array)


def test_scalar_and_per_token_unpack_paths_still_work() -> None:
    """Plain arrays keep both original ``_unpack_adaln`` branches."""
    table = mx.zeros((9, 32))
    scalar = mx.random.normal((2, 9 * 32))
    out = BasicAVTransformerBlock._unpack_adaln(scalar, table, 9, 32)
    assert len(out) == 9 and out[0].shape == (2, 1, 32)
    per_token = mx.random.normal((2, 7, 9 * 32))
    out = BasicAVTransformerBlock._unpack_adaln(per_token, table, 9, 32)
    assert len(out) == 9 and out[0].shape == (2, 7, 32)


def test_verdict_does_not_transport_between_modules() -> None:
    """The M1-CI bug class (#86 review): a verdict earned by one module's
    weights must never be reused by a same-shaped module with other weights.

    Both modules share every signature component; only the weights differ.
    Each must go through its own calibrating call (reference output, no
    carrier) before its own dedupe kicks in.
    """
    t_emb = _embed(_pattern("two", 1, 4096))
    mod_a = _module(4096, 9, seed=1)
    mod_b = _module(4096, 9, seed=2)

    LTXModel._adaln_per_token(None, mod_a, t_emb)  # calibrates A
    probe_a, _ = LTXModel._adaln_per_token(None, mod_a, t_emb)
    if not isinstance(probe_a, PerTokenAdaLNParams):
        pytest.skip("calibration rejects the shrunken GEMM on this hardware; transport is moot")
    params_b, _ = LTXModel._adaln_per_token(None, mod_b, t_emb)
    assert not isinstance(params_b, PerTokenAdaLNParams), (
        "module B skipped its own calibration: verdict transported from module A"
    )
    params_b2, _ = LTXModel._adaln_per_token(None, mod_b, t_emb)
    assert isinstance(params_b2, PerTokenAdaLNParams), "module B never calibrated"


def test_plan_never_reaches_module_parameters() -> None:
    """The per-module plan is bookkeeping, not state: parameters() and the
    weight-loading contracts must not see it."""
    from mlx.utils import tree_flatten

    mod = _module(2048, 4, seed=3)
    t_emb = _embed(_pattern("two", 1, 4096))
    LTXModel._adaln_per_token(None, mod, t_emb)
    LTXModel._adaln_per_token(None, mod, t_emb)
    assert "_adaln_dedupe_plan" in mod.__dict__, "plan was never created"
    leaked = [k for k, _ in tree_flatten(mod.parameters()) if "dedupe" in k]
    assert leaked == [], f"plan leaked into parameters(): {leaked}"


if __name__ == "__main__":
    import time
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except Exception:
            failures += 1
            print(f"FAIL  {name}  ({time.time() - t0:.1f}s)")
            traceback.print_exc()
    raise SystemExit(1 if failures else 0)
