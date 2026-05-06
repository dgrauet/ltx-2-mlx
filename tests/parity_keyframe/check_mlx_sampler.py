"""Compare MLX sampler outputs vs PT dump (apply_denoise_mask + Euler step)."""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from ltx_core_mlx.conditioning.types.latent_cond import apply_denoise_mask

# euler_step lives in mlx-arsenal
try:
    from mlx_arsenal.diffusion import euler_step
except ImportError:
    # Older arsenal version
    from ltx_pipelines_mlx.utils.samplers import euler_step  # type: ignore


PT = "/tmp/keyframe_sampler_parity_pt.npz"


def diff(name: str, mlx_arr, pt_arr) -> tuple[str, float]:
    delta = float(np.max(np.abs(mlx_arr.astype(np.float32) - pt_arr.astype(np.float32))))
    return f"{name:18s}: max_abs={delta:.4e}", delta


def main() -> None:
    pt = dict(np.load(PT))
    x = mx.array(pt["x"])
    clean = mx.array(pt["clean_latent"])
    mask = mx.array(pt["mask"])
    denoised = mx.array(pt["denoised"])

    # 1) post-process (apply_denoise_mask)
    post = np.array(apply_denoise_mask(denoised, clean, mask))
    msg, d_post = diff("post", post, pt["post"])
    print(msg)

    # 2) timesteps_from_mask: mask * sigma -> our equivalent is just mask*sigma
    sigma_scalar = 0.7
    timesteps_mlx = np.array(mask * sigma_scalar)
    msg, d_ts = diff("timesteps", timesteps_mlx, pt["timesteps"])
    print(msg)

    # 3) Euler step from x with mock-denoised post
    sigmas = mx.array([0.7, 0.5, 0.3, 0.0])
    next_x = np.array(euler_step(x, mx.array(post), sigmas[0], sigmas[1]))
    msg, d_step0 = diff("next_x_step0", next_x, pt["next_x_step0"])
    print(msg)

    # 4) Iterate 3 steps
    state = x
    for step in range(3):
        post_step = apply_denoise_mask(denoised, clean, mask)
        state = euler_step(state, post_step, sigmas[step], sigmas[step + 1])
    state_after_3 = np.array(state)
    msg, d_after3 = diff("state_after_3", state_after_3, pt["state_after_3"])
    print(msg)

    # Critical: keyframe preservation in MLX
    n_total, n_kf = 313, 308
    kf_init = pt["x"][:, n_total - n_kf :, :]
    kf_after = state_after_3[:, n_total - n_kf :, :]
    kf_drift = float(np.max(np.abs(kf_init - kf_after)))
    print(f"kf preservation MLX (should be 0): {kf_drift:.4e}")

    fail = max(d_post, d_ts, d_step0, d_after3, kf_drift) > 1e-3
    print()
    print("FAIL" if fail else "ALL OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
