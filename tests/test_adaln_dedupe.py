"""Bit-exactness tests for the AdaLN per-token dedupe.

The dedupe replaces a ``B*N``-row AdaLN GEMM with a handful-of-rows GEMM plus
a gather. It is only allowed to be used when it reproduces the full-size GEMM
*bit for bit*, so every test here asserts ``mx.array_equal`` (exact), never
``allclose``.

Runnable either under pytest or directly::

    python tests/test_adaln_dedupe.py
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.transformer import model as model_mod
from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle
from ltx_core_mlx.model.transformer.model import LTXModel
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding

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


def _deduped(mod: AdaLayerNormSingle, t_emb: mx.array) -> tuple[mx.array, mx.array]:
    return LTXModel._adaln_per_token(None, mod, t_emb)  # method never touches `self`


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
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}  ({time.time() - t0:.1f}s)")
            traceback.print_exc()
    raise SystemExit(1 if failures else 0)
