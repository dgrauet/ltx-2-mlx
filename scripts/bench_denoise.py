#!/usr/bin/env python3
"""Time the denoising loops on real weights, without the text encoder or the VAEs.

Builds random latent states and text embeddings at a fixed shape, loads one DiT
from a pack, and times the real sampler loops the pipelines run:

* ``distilled``: ``denoise_loop`` (1 forward per step) at the given (full) size,
  i.e. the work of a ``--distilled`` stage 2.
* ``guided``: ``guided_denoise_loop`` with the dev defaults (CFG 3, STG 1 on block
  28, modality 3: up to 4 passes per step) at the given size, i.e. the work of a
  ``--two-stage`` stage 1 (pass the half-resolution size).

``--i2v`` pins the first latent frame (per-token timesteps, the I2V / keyframe
path). The first step is a warm-up and is not counted. ``--profile-ops`` runs one
extra forward with every attention / feed-forward call synchronised and timed, to
split a forward into attention, feed-forward and the rest.

Examples::

    uv run python scripts/bench_denoise.py --mode distilled -H 512 -W 768 -f 97 --i2v
    uv run python scripts/bench_denoise.py --mode guided -H 192 -W 320 -f 97 --steps 3
    LTX2_COMPUTE_DTYPE=float32 uv run python scripts/bench_denoise.py --mode distilled --profile-ops

Compare two runs (e.g. before/after a change) with ``--json out.json`` on each.
Weights are never converted or downloaded beyond ``huggingface_hub``'s cache lookup
(``local_files_only``); a pack directory works too.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

DEFAULT_MODEL = "dgrauet/ltx-2.5-mlx-q8"
_mx_eval = getattr(mx, "eval")  # noqa: B009 -- mx.eval is the MLX graph materialiser


def _resolve_model_dir(model: str) -> Path:
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True))


def _transformer_path(model_dir: Path, which: str) -> Path:
    from ltx_pipelines_mlx._base import BasePipeline

    if which == "distilled":
        return BasePipeline._resolve_distilled_transformer(model_dir)
    candidates = sorted(model_dir.glob("transformer-dev*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no transformer-dev*.safetensors in {model_dir}")
    return candidates[-1]


def _chip() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"


def _build_states(height: int, width: int, frames: int, frame_rate: float, i2v: bool, seed: int):
    from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
    from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core_mlx.utils.positions import (
        compute_audio_positions,
        compute_audio_token_count,
        compute_video_positions,
    )
    from ltx_pipelines_mlx.utils.helpers import create_noised_state

    F, H, W = compute_video_latent_shape(frames, height, width)
    audio_t = compute_audio_token_count(frames, frame_rate=frame_rate)
    conditionings = []
    if i2v:
        clean = mx.random.normal((1, H * W, 128), key=mx.random.key(seed + 7)).astype(mx.bfloat16)
        conditionings = [VideoConditionByLatentIndex([0], clean, 1.0)]
    video = create_noised_state(
        base_shape=(1, F * H * W, 128),
        conditionings=conditionings,
        spatial_dims=(F, H, W),
        positions=compute_video_positions(F, H, W, frame_rate=frame_rate),
        seed=seed,
        sigma=1.0,
        legacy_scalar_blend=True,
    )
    audio = create_noised_state(
        base_shape=(1, audio_t, 128),
        conditionings=[],
        spatial_dims=(F, H, W),
        positions=compute_audio_positions(audio_t),
        seed=seed + 1,
        sigma=1.0,
        legacy_scalar_blend=True,
    )
    return video, audio, (F, H, W)


def _embeds(seed: int, length: int = 1024) -> tuple[mx.array, mx.array]:
    k1, k2 = mx.random.split(mx.random.key(seed))
    return (
        mx.random.normal((1, length, 4096), key=k1).astype(mx.bfloat16),
        mx.random.normal((1, length, 2048), key=k2).astype(mx.bfloat16),
    )


class _StepClock:
    """``on_step`` hook: wall time per step, measured after the step's outputs are evaluated."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self._last = time.perf_counter()

    def __call__(self, *args, **kwargs) -> None:
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, mx.array):
                _mx_eval(a)
        now = time.perf_counter()
        self.times.append(now - self._last)
        self._last = now


def _run_loop(args, dit, video, audio, sigmas: list[float]):
    from ltx_core_mlx.model.transformer.model import X0Model
    from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop

    video_embeds, audio_embeds = _embeds(args.seed + 100)
    clock = _StepClock()
    model = X0Model(dit)
    if args.mode == "distilled":
        out = denoise_loop(
            model=model,
            video_state=video,
            audio_state=audio,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
            show_progress=False,
            on_step=clock,
        )
    else:
        from ltx_core_mlx.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory

        neg_video, neg_audio = _embeds(args.seed + 200)
        vparams = MultiModalGuiderParams(
            cfg_scale=3.0, stg_scale=args.stg_scale, rescale_scale=0.7, modality_scale=3.0, stg_blocks=[28]
        )
        aparams = MultiModalGuiderParams(
            cfg_scale=7.0, stg_scale=args.stg_scale, rescale_scale=0.7, modality_scale=3.0, stg_blocks=[28]
        )
        out = guided_denoise_loop(
            model=model,
            video_state=video,
            audio_state=audio,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=create_multimodal_guider_factory(vparams, negative_context=neg_video),
            audio_guider_factory=create_multimodal_guider_factory(aparams, negative_context=neg_audio),
            sigmas=sigmas,
            show_progress=False,
            on_step=clock,
        )
    _mx_eval(out.video_latent)
    return out, clock.times


def _profile_ops(dit, video, audio) -> dict[str, float]:
    """One synchronised forward; seconds per op family (attention by role, feed-forward, rest)."""
    from ltx_core_mlx.model.transformer.attention import Attention
    from ltx_core_mlx.model.transformer.feed_forward import FeedForward

    inner = getattr(dit, "_inner", dit)
    labels: dict[int, str] = {}
    for name, module in inner.named_modules():
        if isinstance(module, (Attention, FeedForward)):
            labels[id(module)] = name.rsplit(".", 1)[-1]
    totals: dict[str, float] = defaultdict(float)
    originals = {cls: cls.__call__ for cls in (Attention, FeedForward)}

    def wrap(cls):
        orig = originals[cls]

        def timed(self, *a, **kw):
            _mx_eval([x for x in list(a) + list(kw.values()) if isinstance(x, mx.array)])
            t0 = time.perf_counter()
            out = orig(self, *a, **kw)
            _mx_eval(out)
            totals[labels.get(id(self), cls.__name__)] += time.perf_counter() - t0
            return out

        return timed

    video_embeds, audio_embeds = _embeds(0)
    sigma = mx.array([0.9])
    kwargs = dict(
        video_latent=video.latent,
        audio_latent=audio.latent,
        timestep=sigma,
        video_text_embeds=video_embeds,
        audio_text_embeds=audio_embeds,
        video_positions=video.positions,
        audio_positions=audio.positions,
        video_timesteps=(sigma * video.denoise_mask[..., 0]),
        audio_timesteps=(sigma * audio.denoise_mask[..., 0]),
    )
    try:
        for cls in originals:
            cls.__call__ = wrap(cls)
        _mx_eval(video.latent, audio.latent)
        t0 = time.perf_counter()
        out = dit(**kwargs)
        _mx_eval(out)
        totals["forward_total"] = time.perf_counter() - t0
    finally:
        for cls, orig in originals.items():
            cls.__call__ = orig
    timed = sum(v for k, v in totals.items() if k != "forward_total")
    totals["rest (norms, AdaLN, residuals, RoPE, head)"] = totals["forward_total"] - timed
    return dict(totals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id (cached) or pack directory")
    ap.add_argument("--mode", choices=("distilled", "guided"), default="distilled")
    ap.add_argument("--transformer", choices=("distilled", "dev"), help="default: distilled / dev by mode")
    ap.add_argument("-H", "--height", type=int, default=512)
    ap.add_argument("-W", "--width", type=int, default=768)
    ap.add_argument("-f", "--frames", type=int, default=97)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=3, help="timed steps (after one warm-up step)")
    ap.add_argument("--stg-scale", type=float, default=1.0, help="guided mode only")
    ap.add_argument("--i2v", action="store_true", help="pin the first latent frame (per-token timesteps)")
    ap.add_argument("--low-ram", action="store_true")
    ap.add_argument("--profile-ops", action="store_true")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--json", type=Path, help="write the report here")
    args = ap.parse_args()

    from ltx_pipelines_mlx._base import apply_compute_dtype_from_env
    from ltx_pipelines_mlx.utils._orchestration import load_transformer

    model_dir = _resolve_model_dir(args.model)
    which = args.transformer or ("distilled" if args.mode == "distilled" else "dev")
    path = _transformer_path(model_dir, which)
    t0 = time.perf_counter()
    dit = apply_compute_dtype_from_env(load_transformer(path, low_ram_streaming=args.low_ram))
    load_s = time.perf_counter() - t0

    video, audio, (F, H, W) = _build_states(args.height, args.width, args.frames, args.frame_rate, args.i2v, args.seed)
    _mx_eval(video.latent, audio.latent)
    # One warm-up step plus the timed ones, on a descending ladder so every step has a new sigma.
    n = args.steps + 1
    sigmas = [1.0 - i * (0.95 / n) for i in range(n)] + [0.0]
    mx.reset_peak_memory()
    _, times = _run_loop(args, dit, video, audio, sigmas)
    timed = times[1:]
    s_per_step = sum(timed) / len(timed) if timed else float("nan")

    report = {
        "chip": _chip(),
        "mlx": md.version("mlx"),
        "compute_dtype": os.environ.get("LTX2_COMPUTE_DTYPE", "") or "float16 (default)",
        "sol_tau": os.environ.get("LTX2_SOL_TAU", ""),
        "stg_prefix_reuse": os.environ.get("LTX2_STG_PREFIX_REUSE", "1"),
        "model": str(model_dir),
        "transformer": path.name,
        "mode": args.mode,
        "low_ram": args.low_ram,
        "i2v": args.i2v,
        "shape_px": [args.frames, args.height, args.width],
        "latent": [F, H, W],
        "video_tokens": int(video.latent.shape[1]),
        "audio_tokens": int(audio.latent.shape[1]),
        "load_s": round(load_s, 2),
        "warmup_step_s": round(times[0], 3) if times else None,
        "step_s": [round(t, 3) for t in timed],
        "mean_step_s": round(s_per_step, 3),
        "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2),
    }
    if args.profile_ops:
        if args.low_ram:
            print("--profile-ops ignores --low-ram's compiled block; profile a resident model", file=sys.stderr)
        report["profile_ops_s"] = {k: round(v, 4) for k, v in _profile_ops(dit, video, audio).items()}

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
