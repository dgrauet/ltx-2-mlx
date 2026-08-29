"""LTXModel -- top-level DiT model for joint audio+video diffusion.

Ported from ltx-core/src/ltx_core/model/model.py

Top-level weight keys (after stripping ``transformer.`` prefix):
    adaln_single, audio_adaln_single         -- 9-param timestep AdaLN
    prompt_adaln_single, audio_prompt_adaln_single  -- 2-param text cross-attn
    av_ca_video_scale_shift_adaln_single     -- 4-param AV cross-attn video
    av_ca_audio_scale_shift_adaln_single     -- 4-param AV cross-attn audio
    av_ca_a2v_gate_adaln_single              -- 1-param A->V gate
    av_ca_v2a_gate_adaln_single              -- 1-param V->A gate
    patchify_proj, audio_patchify_proj       -- patch embed projections
    proj_out, audio_proj_out                 -- output projections
    scale_shift_table, audio_scale_shift_table  -- (2, dim) output AdaLN
    transformer_blocks.N.*                   -- per-block weights
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx
import mlx.nn as nn
import numpy as _np

from ltx_core_mlx.guidance.perturbations import BatchedPerturbationConfig
from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle, PerTokenAdaLNParams
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

# LTX2_DIT_EVAL_EVERY: insert mx.eval every N blocks to keep each Metal
# command buffer below the macOS GPU watchdog (~10 s deadline).
# The 48-layer DiT lazy graph can exceed the deadline at production
# resolutions (640x480+, 33+ frames) even on 64 GB Macs - validated
# by MTLCommandBufferErrorInternal (code 14) crashes on M2 Max 64 GB.
# Default 8: splits 48 blocks into 6 command buffers of ~6 blocks
# each (~1-2 s/buffer), well within the watchdog window.
# Set to 0 to disable (full lazy graph, original behaviour).
_DIT_EVAL_EVERY = int(_os.environ.get("LTX2_DIT_EVAL_EVERY", "8"))
_mx_eval = getattr(mx, "eval")  # noqa: B009

# ---------------------------------------------------------------------------
# AdaLN per-token dedupe + deferred per-block gather
# ---------------------------------------------------------------------------
# When per-token timesteps are in play (any conditioning that puts some tokens
# at a different sigma than the target tokens) the AdaLN MLP is evaluated on
# every one of the B*N tokens, every step -- even though the per-token
# conditioning vector only ever takes a handful of distinct values (the target
# sigma, plus one per conditioning group). Deduplicating the rows turns a
# B*N-row GEMM into a handful-of-rows GEMM plus a gather.
#
# Bit-exactness caveat (measured, not assumed): MLX dispatches different Metal
# GEMM kernels depending on the row count, so a naively shrunk GEMM is NOT
# guaranteed to reproduce the full-size one bit-for-bit -- the K-reduction
# order can change. This implementation therefore
#   (a) pads the unique rows up to a row count that lands on the same kernel
#       as the full-size GEMM, and
#   (b) verifies -- once per shape signature, against the real full-size
#       result, with mx.array_equal -- that the deduped path is bitwise
#       identical before it is ever used for a returned value.
# If no candidate row count reproduces the full-size result exactly, the
# signature is permanently marked not-deduplicable and the original code path
# runs. The optimisation can therefore never change a single output bit.
#
# LTX2_ADALN_DEDUPE=0 disables it entirely.
_ADALN_DEDUPE = _os.environ.get("LTX2_ADALN_DEDUPE", "1") != "0"
# Keep the deduplicated form all the way into the blocks instead of gathering
# the full (B*N, num_params*dim) float32 tensor here. See PerTokenAdaLNParams:
# that tensor is 2.95 GB for the 9-parameter video head at 20 000 tokens, plus
# 1.31 GB for the 4-parameter AV cross-attention head, and it stays resident
# across all 48 blocks. Deferring the gather into
# BasicAVTransformerBlock._unpack_adaln expands one (B, N, dim) slice at a time
# instead, which the block frees as it goes.
#
# Requires the dedupe (there is nothing to defer without it); falls back to the
# eager gather whenever the dedupe declines.
# LTX2_ADALN_LAZY=0 disables it independently of the dedupe.
_ADALN_LAZY = _os.environ.get("LTX2_ADALN_LAZY", "1") != "0"
# Below this many rows the GEMM is cheap and the bookkeeping is not worth it.
_ADALN_DEDUPE_MIN_ROWS = int(_os.environ.get("LTX2_ADALN_DEDUPE_MIN_ROWS", "1024"))
# Only worth it when the unique rows are a small fraction of the total.
_ADALN_DEDUPE_MAX_FRAC = float(_os.environ.get("LTX2_ADALN_DEDUPE_MAX_FRAC", "0.5"))
# Candidate padded row counts, cheapest first. 0 = "use the unique rows as-is".
_ADALN_DEDUPE_PADS = (0, 256, 1024, 4096)
# Chunk size (rows) for the one-time bitwise verification, to bound peak memory.
_ADALN_VERIFY_CHUNK = 2048

# The dedupe plan (signature -> padded row count, or None once proven not
# bit-identical) lives ON each AdaLayerNormSingle instance, never in a module
# global. Bit-equality between the shrunken and full GEMM is only evidence
# about the kernel pair exercised with *this module's weights*; a verdict
# earned by one module must not be transported to another whose weights (or
# very identity) differ. A global plan did exactly that and broke bit-identity
# on M1 CI while passing on newer chips (#86 review). Plain dict of
# primitives, stored via ``module.__dict__`` on purpose: that BYPASSES
# ``nn.Module.__setattr__`` (which would route a dict into the module tree and
# expose it to tree traversals). Never assign the plan with plain attribute
# syntax.


def _dedupe_plan_of(adaln_module: AdaLayerNormSingle) -> dict[tuple, int | None]:
    """Return the module's own calibration plan, creating it on first use."""
    plan = adaln_module.__dict__.get("_adaln_dedupe_plan")
    if plan is None:
        plan = {}
        adaln_module.__dict__["_adaln_dedupe_plan"] = plan
    return plan


def _adaln_signature(adaln_module: AdaLayerNormSingle, flat: mx.array, padded_rows: int) -> tuple:
    """Per-module shape/dtype signature for the calibration verdict.

    A verdict certifies bit-equality between the shrunken and the full GEMM,
    established empirically for this module's weights at these exact shapes.
    That evidence does not transport: not to another module, not to other
    weights, not to other shapes -- the M1-CI failures that motivated the
    per-module plan are the proof that kernel-pair agreement observed on one
    GEMM says nothing about another. Hence the plan lives on the module (see
    ``_dedupe_plan_of``), ``padded_rows`` -- the row count of the shrunken
    GEMM being validated -- is part of the key, and so is the identity of the
    weight array itself, so a fine-tune step that swaps the weight buffer
    invalidates every verdict earned under the old values.
    """
    lin = adaln_module.linear
    w = lin.weight
    return (
        int(flat.shape[0]),
        int(flat.shape[1]),
        int(padded_rows),
        int(adaln_module.num_params),
        id(w),
        tuple(int(s) for s in w.shape),
        str(w.dtype),
        type(lin).__name__,
        str(flat.dtype),
    )


def _unique_rows(flat: mx.array) -> tuple[mx.array, mx.array, int] | None:
    """Exact unique-row decomposition of ``flat`` (M, D).

    Uses the raw bit patterns of two columns of the sinusoidal timestep
    embedding as a cheap grouping key (cos and sin of the fundamental
    frequency -- jointly injective in the timestep over the principal period),
    then *verifies the grouping exactly* against the full rows. A key collision
    can therefore only ever cost the optimisation, never correctness.

    Returns:
        ``(reps, inverse, U)`` with ``reps`` (U, D) and ``inverse`` (M,) such
        that ``mx.take(reps, inverse, axis=0)`` is bitwise ``flat``; or None
        when the rows are not usefully duplicated.
    """
    m, d = int(flat.shape[0]), int(flat.shape[1])
    half = d // 2
    key = mx.stack([flat[:, 0], flat[:, half]], axis=1).astype(mx.float32)
    key_bits = _np.ascontiguousarray(_np.asarray(key)).view(_np.uint32)
    packed = (key_bits[:, 0].astype(_np.uint64) << _np.uint64(32)) | key_bits[:, 1].astype(_np.uint64)
    uniq, first_idx, inverse = _np.unique(packed, return_index=True, return_inverse=True)
    u = int(uniq.shape[0])
    if u >= m or u > m * _ADALN_DEDUPE_MAX_FRAC:
        return None
    idx = mx.array(first_idx.astype(_np.int32))
    inv = mx.array(_np.ascontiguousarray(inverse.reshape(-1)).astype(_np.int32))
    reps = mx.take(flat, idx, axis=0)
    # Exact grouping check: every row must equal its representative, bitwise.
    if not bool(mx.array_equal(mx.take(reps, inv, axis=0), flat).item()):
        return None
    return reps, inv, u


def _pad_rows(reps: mx.array, target: int) -> mx.array:
    """Repeat the last row until ``reps`` has at least ``target`` rows.

    Padding rows are pure filler: each GEMM output row is independent, so the
    extra rows only influence which Metal kernel MLX selects.
    """
    u = int(reps.shape[0])
    if target <= u:
        return reps
    filler = mx.broadcast_to(reps[-1:], (target - u, int(reps.shape[1])))
    return mx.concatenate([reps, filler], axis=0)


def _gathered_equals(out_u: mx.array, inv: mx.array, ref: mx.array) -> bool:
    """Bitwise ``mx.take(out_u, inv, axis=0) == ref``, chunked to bound memory."""
    if out_u.shape[-1] != ref.shape[-1]:
        return False
    m = int(ref.shape[0])
    for start in range(0, m, _ADALN_VERIFY_CHUNK):
        stop = min(start + _ADALN_VERIFY_CHUNK, m)
        gathered = mx.take(out_u, inv[start:stop], axis=0)
        equal = bool(mx.array_equal(gathered, ref[start:stop]).item())
        del gathered
        if not equal:
            return False
    return True


def _trim_rows(x: mx.array, u: int) -> mx.array:
    """Drop the padding rows added by ``_pad_rows``, copying into a new buffer.

    ``x[:u]`` would keep the padded buffer alive behind a view; a gather makes
    the small result independent so the padded GEMM output can be released.
    Row selection is a pure copy, so this is exact.
    """
    if int(x.shape[0]) == u:
        return x
    return mx.take(x, mx.arange(u, dtype=mx.int32), axis=0)


def _dedupe_adaln(adaln_module: AdaLayerNormSingle, flat: mx.array) -> tuple[mx.array, mx.array, mx.array | None]:
    """Evaluate ``adaln_module`` on ``flat`` (M, D), deduplicating equal rows.

    Returns ``(params, embedded, inverse)``. When ``inverse`` is None the two
    arrays are already the full ``(M, ·)`` result; otherwise they hold only the
    distinct rows and ``mx.take(x, inverse, axis=0)`` reconstructs the full
    result exactly. Keeping the two forms distinct is what lets the caller
    defer the gather (see ``PerTokenAdaLNParams``).

    Bitwise identical to ``adaln_module(flat)`` for every input: whenever the
    deduped path has not been *proven* equal for this shape signature, the
    original full-size call is what is returned.
    """
    m = int(flat.shape[0])
    if not _ADALN_DEDUPE or m < _ADALN_DEDUPE_MIN_ROWS:
        params, embedded = adaln_module(flat)
        return params, embedded, None

    grouped = _unique_rows(flat)
    if grouped is None:  # no useful duplication in this input
        params, embedded = adaln_module(flat)
        return params, embedded, None
    reps, inv, u = grouped

    # Verdicts are keyed on the *shrunken* row count actually used, so a change
    # in the number of unique rows re-validates instead of silently reusing a
    # verdict established for a different GEMM shape.
    pending: list[tuple[int, tuple]] = []
    for rows in _candidate_rows(u, m):
        sig = _adaln_signature(adaln_module, flat, rows)
        verdict = _dedupe_plan_of(adaln_module).get(sig, -1)
        if verdict is None:
            continue  # already proven not bit-identical -- try a larger padding
        if verdict == -1:
            pending.append((rows, sig))
            continue
        params_u, embedded_u = adaln_module(_pad_rows(reps, rows))
        return _trim_rows(params_u, u), _trim_rows(embedded_u, u), inv

    if pending:
        # First time for these shapes: compute the reference once, test the
        # candidates against it, and return the reference itself so the
        # calibrating call is exact by construction.
        ref_params, ref_embedded = _calibrate_adaln_dedupe(adaln_module, flat, reps, inv, u, pending)
        return ref_params, ref_embedded, None
    params, embedded = adaln_module(flat)
    return params, embedded, None


def _candidate_rows(u: int, m: int) -> list[int]:
    """Shrunken row counts worth trying, cheapest first, de-duplicated."""
    out: list[int] = []
    for pad in _ADALN_DEDUPE_PADS:
        rows = max(u, pad)
        if rows >= m:
            break  # no saving left at this padding or beyond
        if rows not in out:
            out.append(rows)
    return out


def _calibrate_adaln_dedupe(
    adaln_module: AdaLayerNormSingle,
    flat: mx.array,
    reps: mx.array,
    inv: mx.array,
    u: int,
    pending: list[tuple[int, tuple]],
) -> tuple[mx.array, mx.array]:
    """Find a shrunken row count that reproduces the full-size GEMM exactly.

    Returns the *reference* (full-size) result, so the calibrating call is
    exact by construction. Records a verdict per candidate in
    the module's own dedupe plan (see ``_dedupe_plan_of``).
    """
    debug = bool(_os.environ.get("LTX2_ADALN_DEDUPE_DEBUG"))
    ref_params, ref_embedded = adaln_module(flat)
    _mx_eval(ref_params, ref_embedded)

    for rows, sig in pending:
        params_u, embedded_u = adaln_module(_pad_rows(reps, rows))
        _mx_eval(params_u, embedded_u)
        ok = _gathered_equals(params_u, inv, ref_params) and _gathered_equals(embedded_u, inv, ref_embedded)
        del params_u, embedded_u
        mx.clear_cache()
        _dedupe_plan_of(adaln_module)[sig] = rows if ok else None
        if debug:
            print(
                f"[adaln-dedupe] tokens={sig[0]} nparams={sig[3]} unique={u} gemm_rows={rows} -> "
                f"{'ACCEPTED' if ok else 'rejected (not bit-identical)'}",
                flush=True,
            )
        if ok:
            break

    return ref_params, ref_embedded


class Modality(Enum):
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class LTXModelConfig:
    """Configuration for LTXModel."""

    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_num_heads: int = 32
    audio_num_heads: int = 32
    video_head_dim: int = 128
    audio_head_dim: int = 64
    av_cross_num_heads: int = 32
    av_cross_head_dim: int = 64
    video_patch_channels: int = 128
    audio_patch_channels: int = 128
    ff_mult: float = 4.0
    timestep_embedding_dim: int = 256
    timestep_scale_multiplier: float = 1000.0
    av_ca_timestep_scale_multiplier: int = 1  # upstream-iso default; checkpoint config supplies 1000
    rope_theta: float = 10000.0
    rope_type: str = "split"
    positional_embedding_max_pos: tuple[int, ...] = (20, 2048, 2048)
    audio_positional_embedding_max_pos: tuple[int, ...] = (20,)
    norm_eps: float = 1e-6
    # 2.5 fields, read from the checkpoint config. Defaults reproduce 2.3
    # exactly — the mapping and defaults mirror upstream model_configurator.py.
    ff_bias: bool = True
    audio_ff_bias: bool = True
    use_prompt_adaln_single: bool = True
    use_keyframes_abs_pos_embedding: bool = False
    double_precision_rope: bool = False

    @classmethod
    def from_checkpoint_config(cls, config: dict) -> LTXModelConfig:
        """Build a config from a checkpoint config dict (``config.json`` /
        ``embedded_config.json``).

        Mirrors upstream ``LTXModelConfigurator.from_config`` field mapping so the
        runtime hyperparameters track the checkpoint metadata instead of the
        hardcoded dataclass defaults. The dataclass defaults serve as the
        per-key fallback, matching upstream's ``config.get(key, default)``.

        This is the fix for the ``av_ca_timestep_scale_multiplier`` divergence
        (issue #37): every LTX-2.3 checkpoint ships ``1000.0`` but the dataclass
        default is ``1.0``. The root cause was copying upstream's *dataclass*
        default (``1``) without wiring upstream's *configurator* (which reads the
        value from the checkpoint → ``1000``).

        Args:
            config: Parsed checkpoint config. May be the full dict (with a
                ``"transformer"`` sub-dict) or the transformer sub-dict itself.

        Returns:
            An :class:`LTXModelConfig` populated from the checkpoint.
        """
        t = config.get("transformer", config)
        d = cls()
        if t.get("share_ff", False):
            raise ValueError(
                "share_ff=true is not supported (upstream asserts it False; no shipped checkpoint sets it)"
            )
        return cls(
            num_layers=t.get("num_layers", d.num_layers),
            video_dim=t.get("cross_attention_dim", d.video_dim),
            audio_dim=t.get("audio_cross_attention_dim", d.audio_dim),
            video_num_heads=t.get("num_attention_heads", d.video_num_heads),
            audio_num_heads=t.get("audio_num_attention_heads", d.audio_num_heads),
            video_head_dim=t.get("attention_head_dim", d.video_head_dim),
            audio_head_dim=t.get("audio_attention_head_dim", d.audio_head_dim),
            av_cross_num_heads=t.get("audio_num_attention_heads", d.av_cross_num_heads),
            av_cross_head_dim=t.get("audio_attention_head_dim", d.av_cross_head_dim),
            video_patch_channels=t.get("in_channels", d.video_patch_channels),
            audio_patch_channels=t.get("audio_in_channels", d.audio_patch_channels),
            timestep_scale_multiplier=t.get("timestep_scale_multiplier", d.timestep_scale_multiplier),
            av_ca_timestep_scale_multiplier=t.get("av_ca_timestep_scale_multiplier", d.av_ca_timestep_scale_multiplier),
            rope_theta=t.get("positional_embedding_theta", d.rope_theta),
            rope_type=t.get("rope_type", d.rope_type),
            positional_embedding_max_pos=tuple(t.get("positional_embedding_max_pos", d.positional_embedding_max_pos)),
            audio_positional_embedding_max_pos=tuple(
                t.get("audio_positional_embedding_max_pos", d.audio_positional_embedding_max_pos)
            ),
            norm_eps=t.get("norm_eps", d.norm_eps),
            ff_bias=t.get("ff_bias", d.ff_bias),
            audio_ff_bias=t.get("audio_ff_bias", d.audio_ff_bias),
            use_prompt_adaln_single=t.get("use_prompt_adaln_single", d.use_prompt_adaln_single),
            use_keyframes_abs_pos_embedding=t.get("use_keyframes_abs_pos_embedding", d.use_keyframes_abs_pos_embedding),
            double_precision_rope=t.get("frequencies_precision", "") == "float64",
        )

    @classmethod
    def from_checkpoint_dir(cls, model_dir) -> LTXModelConfig:
        """Read the transformer config from a checkpoint directory.

        Prefers ``embedded_config.json`` (the richer config, includes
        ``rope_type``) and falls back to ``config.json``. If neither is present
        or parseable, warns on stderr and returns the hardcoded defaults — which
        would reintroduce the ``av_ca_timestep_scale_multiplier`` bug, so the
        warning is loud.

        Args:
            model_dir: Directory containing the checkpoint config files.

        Returns:
            An :class:`LTXModelConfig` read from the checkpoint, or defaults.
        """
        import json
        import sys
        from pathlib import Path

        model_dir = Path(model_dir)
        for name in ("embedded_config.json", "config.json"):
            path = model_dir / name
            if not path.exists():
                continue
            try:
                return cls.from_checkpoint_config(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"warning: failed to read {path}: {exc}; using defaults", file=sys.stderr)
                return cls()
        print(
            f"warning: no transformer config (embedded_config.json / config.json) found in "
            f"{model_dir}; using hardcoded defaults "
            f"(av_ca_timestep_scale_multiplier={cls().av_ca_timestep_scale_multiplier}). "
            "Audio cross-modal gating may be wrong — see issue #37.",
            file=sys.stderr,
        )
        return cls()


class LTXModel(nn.Module):
    """LTX-2.3 Diffusion Transformer for joint audio+video generation.

    19B parameter DiT with 48 transformer blocks, joint audio+video
    attention, and adaptive layer norm conditioning.
    """

    def __init__(self, config: LTXModelConfig | None = None):
        super().__init__()
        if config is None:
            config = LTXModelConfig()
        self.config = config

        vd = config.video_dim
        ad = config.audio_dim
        t_dim = config.timestep_embedding_dim

        # --- Patch embedding projections ---
        self.patchify_proj = nn.Linear(config.video_patch_channels, vd)
        self.audio_patchify_proj = nn.Linear(config.audio_patch_channels, ad)

        # --- Output projections ---
        self.proj_out = nn.Linear(vd, config.video_patch_channels)
        self.audio_proj_out = nn.Linear(ad, config.audio_patch_channels)

        # --- Output scale/shift tables (raw parameters, shape (2, dim)) ---
        self.scale_shift_table = mx.zeros((2, vd))
        self.audio_scale_shift_table = mx.zeros((2, ad))

        # Marks tokens whose latent encodes a single standalone pixel frame.
        # Zero-initialized, so a checkpoint that predates it behaves
        # identically until the parameter is trained. Applied by the keyframe
        # conditioning path (not in this forward) — mirror of upstream
        # model.py. Only created when the checkpoint config asks for it, so
        # 2.3 checkpoints keep their exact parameter tree.
        if config.use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = mx.zeros((1, vd))

        # --- Timestep AdaLN (9-param: self-attn shift/scale/gate x3) ---
        self.adaln_single = AdaLayerNormSingle(vd, num_params=9, timestep_dim=t_dim)
        self.audio_adaln_single = AdaLayerNormSingle(ad, num_params=9, timestep_dim=t_dim)

        # --- Prompt (text cross-attn) AdaLN (2-param: shift, scale) ---
        self.prompt_adaln_single = AdaLayerNormSingle(vd, num_params=2, timestep_dim=t_dim)
        self.audio_prompt_adaln_single = AdaLayerNormSingle(ad, num_params=2, timestep_dim=t_dim)

        # --- AV cross-attention AdaLN ---
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(vd, num_params=4, timestep_dim=t_dim)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(ad, num_params=4, timestep_dim=t_dim)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(vd, num_params=1, timestep_dim=t_dim)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(ad, num_params=1, timestep_dim=t_dim)

        # --- Transformer blocks ---
        self.transformer_blocks = [
            BasicAVTransformerBlock(
                video_dim=vd,
                audio_dim=ad,
                video_num_heads=config.video_num_heads,
                audio_num_heads=config.audio_num_heads,
                video_head_dim=config.video_head_dim,
                audio_head_dim=config.audio_head_dim,
                av_cross_num_heads=config.av_cross_num_heads,
                av_cross_head_dim=config.av_cross_head_dim,
                ff_mult=config.ff_mult,
                norm_eps=config.norm_eps,
                ff_bias=config.ff_bias,
                audio_ff_bias=config.audio_ff_bias,
            )
            for _ in range(config.num_layers)
        ]

        # Training-only: recompute each block in the backward pass instead of
        # storing all 48 blocks' activations. Caps activation memory at ~1 block
        # so backprop through the dev model fits on 64 GB. No effect on inference.
        self.gradient_checkpointing = False

    def _embed_timestep_scalar(
        self,
        timestep: mx.array,
    ) -> mx.array:
        """Compute timestep embedding from scalar (B,) timesteps.

        Returns:
            Timestep embedding (B, timestep_embedding_dim).
        """
        t_scaled = timestep * self.config.timestep_scale_multiplier
        return get_timestep_embedding(t_scaled, self.config.timestep_embedding_dim)

    def _embed_timestep_per_token(
        self,
        per_token_timesteps: mx.array,
    ) -> mx.array:
        """Compute timestep embedding from per-token (B, N) timesteps.

        Flattens to (B*N,), passes through sinusoidal embedding,
        then reshapes back to (B, N, timestep_embedding_dim).

        Returns:
            Timestep embedding (B, N, timestep_embedding_dim).
        """
        B, N = per_token_timesteps.shape
        flat = (per_token_timesteps * self.config.timestep_scale_multiplier).reshape(-1)
        emb = get_timestep_embedding(flat, self.config.timestep_embedding_dim)  # (B*N, D)
        return emb.reshape(B, N, -1)

    def _adaln_per_token(
        self,
        adaln_module: AdaLayerNormSingle,
        t_emb_per_token: mx.array,
    ) -> tuple[mx.array | PerTokenAdaLNParams, mx.array | PerTokenAdaLNParams]:
        """Apply AdaLN with per-token timestep embeddings.

        Args:
            adaln_module: AdaLayerNormSingle module.
            t_emb_per_token: (B, N, timestep_embedding_dim).

        Returns:
            Tuple of (params, embedded_timestep), logically shaped
            (B, N, num_params * dim) and (B, N, dim). Both are either plain
            ``mx.array`` or, when the rows deduplicated and ``LTX2_ADALN_LAZY``
            is on, ``PerTokenAdaLNParams`` -- the distinct rows plus an index,
            expanded later by whoever consumes them.

        The AdaLN MLP is row-independent and the per-token conditioning vector
        takes only a few distinct values in practice, so the flattened rows are
        deduplicated before the GEMM. The deduped path is only ever used after
        it has been proven bitwise identical to the full-size path for that
        shape signature (see the module header); otherwise this falls back to
        the original full-size GEMM.
        """
        B, N, D = t_emb_per_token.shape
        flat = t_emb_per_token.reshape(B * N, D)
        params, embedded, inv = _dedupe_adaln(adaln_module, flat)
        if inv is None:
            return params.reshape(B, N, -1), embedded.reshape(B, N, -1)
        if _ADALN_LAZY:
            return (
                PerTokenAdaLNParams(params, inv, B, N, adaln_module.num_params),
                PerTokenAdaLNParams(embedded, inv, B, N, 1),
            )
        return (
            mx.take(params, inv, axis=0).reshape(B, N, -1),
            mx.take(embedded, inv, axis=0).reshape(B, N, -1),
        )

    def compute_gate_signal(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        timestep: mx.array,
        video_timesteps: mx.array | None = None,
    ) -> mx.array:
        """Cheap probe: block 0's modulated video input (TeaCache gate signal).

        Runs the prelude (patchify_proj + video AdaLN) but no transformer
        blocks. The output is bit-equivalent to ``video_normed`` as it would
        be computed inside block 0 during a full forward.

        Args:
            video_latent: (B, Nv, video_patch_channels).
            audio_latent: (B, Na, audio_patch_channels). Unused for the
                gate signal itself but accepted for API symmetry; ignored.
            timestep: (B,) sigma value.
            video_timesteps: Optional (B, Nv) per-token timesteps; matches
                ``__call__`` semantics for conditioning masks.

        Returns:
            Gate signal ``(B, Nv, video_dim)``.
        """
        del audio_latent  # signature parity with __call__; not needed for gate
        video_latent = video_latent.astype(mx.bfloat16)
        timestep = timestep.astype(mx.bfloat16)

        video_hidden = self.patchify_proj(video_latent)
        t_emb = self._embed_timestep_scalar(timestep)

        if video_timesteps is not None:
            vt_emb = self._embed_timestep_per_token(video_timesteps)
            video_adaln_emb, _ = self._adaln_per_token(self.adaln_single, vt_emb)
        else:
            video_adaln_emb, _ = self.adaln_single(t_emb)

        return self.transformer_blocks[0].compute_video_normed_sa(video_hidden, video_adaln_emb)

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        timestep: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_positions: mx.array | None = None,
        audio_positions: mx.array | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        video_cross_attention_mask: mx.array | None = None,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        tap: callable | None = None,
        block_stack_override: callable | None = None,
        block_provider: callable | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            video_latent: (B, Nv, patch_channels) -- patchified video latent tokens.
            audio_latent: (B, Na, patch_channels) -- patchified audio latent tokens.
            timestep: (B,) -- diffusion timestep (sigma value).
            video_text_embeds: (B, Nt, video_dim) -- projected text embeddings.
            audio_text_embeds: (B, Nt, audio_dim) -- projected text embeddings.
            video_positions: (B, Nv, num_axes) -- positions for RoPE.
            audio_positions: (B, Na, num_axes) -- positions for RoPE.
            video_attention_mask: Optional mask for video attention.
            audio_attention_mask: Optional mask for audio attention.
            video_cross_attention_mask: Optional additive bias for the video→text
                cross-attention (Prompt Relay temporal prompt gating). Same array
                is broadcast to every block.
            video_timesteps: Optional (B, Nv) per-token timesteps for video.
                When provided, AdaLN parameters are computed per-token instead
                of per-batch, enabling preserved tokens (mask=0) to receive
                timestep=0 (no modulation).
            audio_timesteps: Optional (B, Na) per-token timesteps for audio.
            perturbations: Optional perturbation config for STG guidance.
            tap: Optional callback ``tap(video_block_residual,
                audio_block_residual)`` invoked after the block stack with
                the residuals ``block_output - block_input``. Used for
                calibration; does not affect control flow.
            block_stack_override: Optional callable
                ``(video_hidden, audio_hidden) -> (video_hidden_out,
                audio_hidden_out)`` that replaces the block iteration. Used
                by TeaCache on skip steps to reconstruct outputs from a
                cached residual without running the blocks. The model's
                head still runs.
            block_provider: Optional callable ``(int) -> nn.Module`` that
                returns the block to invoke at each iteration. When
                provided, ``self.transformer_blocks`` is bypassed and
                the provider is called once per ``block_idx`` to fetch
                the active block instance. Used by block streaming to
                bind weights from a memory-mapped safetensors file into
                a single shared block module, capping resident memory
                at ~1 block instead of ~num_layers blocks. Mutually
                exclusive with ``block_stack_override``.

        Returns:
            Tuple of (video_velocity, audio_velocity), same shapes as inputs.
        """
        # Cast inputs to bfloat16 to match weight dtype and avoid mixed-precision
        # accumulation errors over 48 transformer blocks
        video_latent = video_latent.astype(mx.bfloat16)
        audio_latent = audio_latent.astype(mx.bfloat16)
        if video_text_embeds is not None:
            video_text_embeds = video_text_embeds.astype(mx.bfloat16)
        if audio_text_embeds is not None:
            audio_text_embeds = audio_text_embeds.astype(mx.bfloat16)

        # Embed patches
        video_hidden = self.patchify_proj(video_latent)
        audio_hidden = self.audio_patchify_proj(audio_latent)

        # --- Timestep embeddings ---
        timestep = timestep.astype(mx.bfloat16)
        t_emb = self._embed_timestep_scalar(timestep)

        # AV cross-attention gate uses a different timestep scale (default 1, not 1000).
        # Reference: gate_adaln receives sigma * av_ca_timestep_scale_multiplier.
        av_ca_factor = self.config.av_ca_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        t_emb_av_gate = get_timestep_embedding(
            timestep * self.config.timestep_scale_multiplier * av_ca_factor,
            self.config.timestep_embedding_dim,
        )

        # Video AdaLN: per-token or scalar
        # Note: prompt AdaLN always uses scalar timestep — text embeddings
        # don't correspond to individual latent tokens, so per-token
        # modulation would cause a shape mismatch in cross-attention.
        if video_timesteps is not None:
            vt_emb = self._embed_timestep_per_token(video_timesteps)
            video_adaln_emb, video_embedded_ts = self._adaln_per_token(self.adaln_single, vt_emb)
            av_ca_video_emb, _ = self._adaln_per_token(self.av_ca_video_scale_shift_adaln_single, vt_emb)
        else:
            video_adaln_emb, video_embedded_ts = self.adaln_single(t_emb)
            av_ca_video_emb, _ = self.av_ca_video_scale_shift_adaln_single(t_emb)
        # AV cross-attention gate always uses scalar timestep at av_ca scale,
        # even in per-token mode. Reference: gate_adaln receives sigma * av_ca_factor (scalar).
        av_ca_a2v_gate_emb, _ = self.av_ca_a2v_gate_adaln_single(t_emb_av_gate)
        # Prompt AdaLN: always scalar (from global timestep)
        video_prompt_emb, _ = self.prompt_adaln_single(t_emb)

        # Audio AdaLN: per-token or scalar
        if audio_timesteps is not None:
            at_emb = self._embed_timestep_per_token(audio_timesteps)
            audio_adaln_emb, audio_embedded_ts = self._adaln_per_token(self.audio_adaln_single, at_emb)
            av_ca_audio_emb, _ = self._adaln_per_token(self.av_ca_audio_scale_shift_adaln_single, at_emb)
        else:
            audio_adaln_emb, audio_embedded_ts = self.audio_adaln_single(t_emb)
            av_ca_audio_emb, _ = self.av_ca_audio_scale_shift_adaln_single(t_emb)
        # AV cross-attention gate always uses scalar timestep at av_ca scale
        av_ca_v2a_gate_emb, _ = self.av_ca_v2a_gate_adaln_single(t_emb_av_gate)
        # Audio prompt AdaLN: always scalar (from global timestep)
        audio_prompt_emb, _ = self.audio_prompt_adaln_single(t_emb)

        # RoPE frequencies (per-head, using reference log-spaced grid)
        video_rope_freqs = None
        audio_rope_freqs = None
        if video_positions is not None:
            video_rope_freqs = self._compute_rope_freqs(
                video_positions,
                self.config.video_num_heads,
                self.config.video_head_dim,
            )
        if audio_positions is not None:
            audio_rope_freqs = self._compute_rope_freqs(
                audio_positions,
                self.config.audio_num_heads,
                self.config.audio_head_dim,
                max_pos_override=list(self.config.audio_positional_embedding_max_pos),
            )

        # Cross-modal RoPE: 1D temporal positions, av_cross inner dim.
        # Reference computes from modality.positions[:, 0:1, :] (temporal only)
        # with inner_dim=audio_cross_attention_dim and max_pos=[cross_pe_max_pos].
        video_cross_rope_freqs = None
        audio_cross_rope_freqs = None
        cross_pe_max_pos = max(
            self.config.positional_embedding_max_pos[0],
            self.config.audio_positional_embedding_max_pos[0],
        )
        if video_positions is not None:
            video_cross_rope_freqs = self._compute_rope_freqs(
                video_positions[:, :, 0:1],  # temporal dimension only
                self.config.av_cross_num_heads,
                self.config.av_cross_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )
        if audio_positions is not None:
            audio_cross_rope_freqs = self._compute_rope_freqs(
                audio_positions[:, :, 0:1],  # temporal dimension only
                self.config.av_cross_num_heads,
                self.config.av_cross_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )

        # --- Block stack (optionally overridden) ---
        block_input_v = video_hidden
        block_input_a = audio_hidden

        if block_stack_override is not None:
            video_hidden, audio_hidden = block_stack_override(video_hidden, audio_hidden)
        else:
            num_layers = self.config.num_layers if block_provider is not None else len(self.transformer_blocks)
            for block_idx in range(num_layers):
                block = block_provider(block_idx) if block_provider is not None else self.transformer_blocks[block_idx]

                if self.gradient_checkpointing:
                    # Recompute this block in the backward pass to cap activation
                    # memory. The block's trainable params MUST be passed as an
                    # explicit mx.checkpoint input (and rebound via update inside),
                    # otherwise mx.checkpoint — which only tracks gradients w.r.t.
                    # its inputs — gives the LoRA params zero gradient (they stay
                    # at init and the adapter is a no-op). Pattern from mlx_lm's
                    # tuner.trainer.grad_checkpoint. Conditioning is captured as
                    # constants; mx.eval guards are skipped (illegal under autodiff).
                    def _run_block(params, vh, ah, _block=block, _bidx=block_idx):
                        _block.update(params)
                        return _block(
                            video_hidden=vh,
                            audio_hidden=ah,
                            video_adaln_params=video_adaln_emb,
                            audio_adaln_params=audio_adaln_emb,
                            video_prompt_adaln_params=video_prompt_emb,
                            audio_prompt_adaln_params=audio_prompt_emb,
                            av_ca_video_params=av_ca_video_emb,
                            av_ca_audio_params=av_ca_audio_emb,
                            av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                            av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                            video_text_embeds=video_text_embeds,
                            audio_text_embeds=audio_text_embeds,
                            video_rope_freqs=video_rope_freqs,
                            audio_rope_freqs=audio_rope_freqs,
                            video_cross_rope_freqs=video_cross_rope_freqs,
                            audio_cross_rope_freqs=audio_cross_rope_freqs,
                            video_attention_mask=video_attention_mask,
                            audio_attention_mask=audio_attention_mask,
                            video_cross_attention_mask=video_cross_attention_mask,
                            perturbations=perturbations,
                            block_idx=_bidx,
                        )

                    video_hidden, audio_hidden = mx.checkpoint(_run_block)(
                        block.trainable_parameters(), video_hidden, audio_hidden
                    )
                    continue

                video_hidden, audio_hidden = block(
                    video_hidden=video_hidden,
                    audio_hidden=audio_hidden,
                    video_adaln_params=video_adaln_emb,
                    audio_adaln_params=audio_adaln_emb,
                    video_prompt_adaln_params=video_prompt_emb,
                    audio_prompt_adaln_params=audio_prompt_emb,
                    av_ca_video_params=av_ca_video_emb,
                    av_ca_audio_params=av_ca_audio_emb,
                    av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                    av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                    video_text_embeds=video_text_embeds,
                    audio_text_embeds=audio_text_embeds,
                    video_rope_freqs=video_rope_freqs,
                    audio_rope_freqs=audio_rope_freqs,
                    video_cross_rope_freqs=video_cross_rope_freqs,
                    audio_cross_rope_freqs=audio_cross_rope_freqs,
                    video_attention_mask=video_attention_mask,
                    audio_attention_mask=audio_attention_mask,
                    video_cross_attention_mask=video_cross_attention_mask,
                    perturbations=perturbations,
                    block_idx=block_idx,
                )
                if block_provider is not None:
                    # Streaming: force MLX graph materialization between
                    # blocks so the previous block's weights become
                    # evictable.
                    _mx_eval(video_hidden, audio_hidden)
                elif _DIT_EVAL_EVERY > 0 and (block_idx + 1) % _DIT_EVAL_EVERY == 0:
                    # Watchdog guard: flush accumulated lazy graph every N blocks
                    # so no single Metal command buffer exceeds the ~10 s deadline.
                    _mx_eval(video_hidden, audio_hidden)

        if tap is not None:
            tap(video_hidden - block_input_v, audio_hidden - block_input_a)

        # Output: AdaLN with scale_shift_table + embedded_timestep + proj
        video_out = self._output_block(video_hidden, video_embedded_ts, self.scale_shift_table, self.proj_out)
        audio_out = self._output_block(
            audio_hidden, audio_embedded_ts, self.audio_scale_shift_table, self.audio_proj_out
        )

        return video_out, audio_out

    def _output_block(
        self,
        x: mx.array,
        embedded_timestep: mx.array | PerTokenAdaLNParams,
        scale_shift_table: mx.array,
        proj: nn.Linear,
    ) -> mx.array:
        """Apply output norm + adaptive scale/shift + projection.

        Reference: scale_shift_values = scale_shift_table + embedded_timestep
        The table (2, dim) provides learnable base values; the embedded_timestep
        provides per-sample (or per-token) conditioning.
        """
        # Per-token embedded timesteps arrive deduplicated and are only needed
        # here, after the block stack -- expand at the point of use so the
        # (B, N, dim) float32 tensor never spans the 48-block loop.
        if isinstance(embedded_timestep, PerTokenAdaLNParams):
            embedded_timestep = embedded_timestep.gather()
        # embedded_timestep: (B, dim) for scalar or (B, N, dim) for per-token
        if embedded_timestep.ndim == 2:
            embedded_timestep = embedded_timestep[:, None, :]  # (B, 1, dim)
        # scale_shift_table: (2, dim) -> broadcast
        scale_shift_values = scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        # scale_shift_values: (B, N_or_1, 2, dim)
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        # Reference uses LayerNorm(elementwise_affine=False), not RMSNorm
        x = mx.fast.layer_norm(x, weight=None, bias=None, eps=self.config.norm_eps)
        x = x * (1.0 + scale) + shift
        return proj(x)

    def _compute_rope_freqs(
        self,
        positions: mx.array,
        num_heads: int,
        head_dim: int,
        max_pos_override: list[int] | None = None,
    ) -> mx.array:
        """Compute per-head RoPE frequencies using reference log-spaced grid.

        Args:
            positions: (B, N, num_axes) integer position indices.
            num_heads: Number of attention heads.
            head_dim: Per-head dimension.
            max_pos_override: Override max_pos (e.g. for cross-modal 1D temporal RoPE).

        Returns:
            Per-head frequencies (B, num_heads, N, head_dim // 2).
        """
        from ltx_core_mlx.model.transformer.rope import precompute_rope_freqs

        inner_dim = num_heads * head_dim
        if max_pos_override is not None:
            max_pos = max_pos_override
        else:
            max_pos = list(self.config.positional_embedding_max_pos[: positions.shape[-1]])
        return precompute_rope_freqs(
            positions,
            inner_dim=inner_dim,
            num_heads=num_heads,
            theta=self.config.rope_theta,
            max_pos=max_pos,
            rope_type=self.config.rope_type,
            double_precision=self.config.double_precision_rope,
        )


class X0Model(nn.Module):
    """Wrapper that converts velocity prediction to x0 prediction.

    Given the diffusion equation: x_t = x_0 + sigma * v,
    x_0 = x_t - sigma * v
    """

    def __init__(self, model: LTXModel):
        super().__init__()
        self.model = model

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        sigma: mx.array,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        tap: callable | None = None,
        block_stack_override: callable | None = None,
        **kwargs,
    ) -> tuple[mx.array, mx.array]:
        """Predict x0 from noisy input.

        Uses per-token timesteps when available so preserved tokens (timestep=0)
        are kept unchanged: x0 = x_t - 0 * v = x_t.

        Args:
            video_latent: Noisy video latent.
            audio_latent: Noisy audio latent.
            sigma: Current noise level (B,).
            video_timesteps: Optional per-token timesteps (B, Nv).
            audio_timesteps: Optional per-token timesteps (B, Na).
            perturbations: Optional perturbation config for STG guidance.
            tap: Optional callback passed through to the inner model.
            block_stack_override: Optional callable passed through to the inner model.
            **kwargs: Passed to inner model.

        Returns:
            Tuple of (video_x0, audio_x0).
        """
        video_v, audio_v = self.model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=sigma,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            perturbations=perturbations,
            tap=tap,
            block_stack_override=block_stack_override,
            **kwargs,
        )

        # x0 = x_t - sigma * v
        # Use per-token timesteps when available (preserved tokens get timestep=0)
        # Cast to float32 for precision (reference does this too)
        if video_timesteps is not None:
            video_sigma = video_timesteps[:, :, None].astype(mx.float32)
        else:
            video_sigma = sigma[:, None, None].astype(mx.float32)

        if audio_timesteps is not None:
            audio_sigma = audio_timesteps[:, :, None].astype(mx.float32)
        else:
            audio_sigma = sigma[:, None, None].astype(mx.float32)

        video_x0 = (video_latent.astype(mx.float32) - video_sigma * video_v.astype(mx.float32)).astype(
            video_latent.dtype
        )
        audio_x0 = (audio_latent.astype(mx.float32) - audio_sigma * audio_v.astype(mx.float32)).astype(
            audio_latent.dtype
        )

        return video_x0, audio_x0
