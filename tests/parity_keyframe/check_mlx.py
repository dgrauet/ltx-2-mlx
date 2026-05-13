"""Compare ours vs upstream-PT keyframe conditioning outputs.

Run from this repo:
    uv run python tests/parity_keyframe/check_mlx.py

Loads /tmp/keyframe_parity_pt.npz (produced by dump_pt.py in upstream venv)
and prints max-abs diffs against our MLX implementation per case.
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import LatentState

PT_DUMP = "/tmp/keyframe_parity_pt.npz"


def make_state(seed: int, n_tokens: int, channels: int = 128) -> LatentState:
    rng = np.random.default_rng(seed)
    latent_np = rng.standard_normal((1, n_tokens, channels)).astype(np.float32)
    latent = mx.array(latent_np)
    return LatentState(
        latent=latent,
        clean_latent=latent,
        denoise_mask=mx.ones((1, n_tokens, 1), dtype=mx.float32),
        positions=mx.zeros((1, n_tokens, 3), dtype=mx.float32),
        attention_mask=None,
    )


def run_case(
    label: str,
    frame_idx: int,
    h: int,
    w: int,
    fps: float = 24.0,
    num_pixel_frames: int = 1,
    seed: int = 42,
    channels: int = 128,
    n_existing: int = 5,
) -> dict:
    rng = np.random.default_rng(seed)
    keyframe_np = rng.standard_normal((1, channels, 1, h, w)).astype(np.float32)
    # MLX expects (B, H*W, C) tokens after manual patchify
    keyframe_tokens = mx.array(keyframe_np.transpose(0, 2, 3, 4, 1).reshape(1, -1, channels))

    state = make_state(seed=seed + 1, n_tokens=n_existing, channels=channels)

    cond = VideoConditionByKeyframeIndex(
        frame_idx=frame_idx,
        keyframe_latent=keyframe_tokens,
        spatial_dims=(5, h, w),
        frame_rate=fps,
        strength=1.0,
        num_pixel_frames=num_pixel_frames,
    )
    out = cond.apply(state, (5, h, w))
    return {
        "positions": np.array(out.positions),
        "latent": np.array(out.latent),
        "clean_latent": np.array(out.clean_latent),
        "denoise_mask": np.array(out.denoise_mask),
        "attention_mask": (np.array(out.attention_mask) if out.attention_mask is not None else None),
    }


def diff_summary(label: str, name: str, mlx_arr, pt_arr) -> tuple[str, float]:
    if mlx_arr is None and (pt_arr is None or pt_arr.shape == ()):
        return ("OK (both None)", 0.0)
    if mlx_arr is None or pt_arr is None or pt_arr.shape == ():
        return ("FAIL: shape mismatch", float("nan"))
    if mlx_arr.shape != pt_arr.shape:
        return (f"FAIL: mlx={mlx_arr.shape} pt={pt_arr.shape}", float("nan"))
    delta = np.max(np.abs(mlx_arr.astype(np.float32) - pt_arr.astype(np.float32)))
    return (f"max_abs={delta:.4e}", float(delta))


def main() -> None:
    pt = dict(np.load(PT_DUMP))

    cases = [
        ("case0_start_halfres", dict(frame_idx=0, h=14, w=22)),
        ("case1_end_halfres", dict(frame_idx=32, h=14, w=22)),
        ("case2_end_fullres", dict(frame_idx=32, h=28, w=44)),
    ]

    fail = False
    for label, kwargs in cases:
        print(f"\n=== {label} ===")
        mlx_out = run_case(label, **kwargs)
        comparisons = [
            ("positions", mlx_out["positions"], pt[f"{label}_positions_mid"]),
            ("latent", mlx_out["latent"], pt[f"{label}_latent"]),
            ("clean_latent", mlx_out["clean_latent"], pt[f"{label}_clean_latent"]),
            ("denoise_mask", mlx_out["denoise_mask"], pt[f"{label}_denoise_mask"]),
            ("attention_mask", mlx_out["attention_mask"], pt[f"{label}_attention_mask"]),
        ]
        for name, mlx_arr, pt_arr in comparisons:
            verdict, delta = diff_summary(label, name, mlx_arr, pt_arr)
            print(f"  {name:18s}: {verdict}")
            if delta != delta or delta > 1e-3:
                fail = True

    print()
    print("FAIL" if fail else "ALL OK (max_abs < 1e-3)")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
