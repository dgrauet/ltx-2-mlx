"""Dump upstream PT sampler step outputs (apply_denoise_mask + euler_step).

Tests one Euler step starting from a keyframe-conditioned state, with
a controlled (mock) denoised output. Verifies that:
  - post_process_latent (mask blend) produces expected output
  - EulerDiffusionStep.step preserves clean tokens (mask=0) bit-exactly

Run from upstream venv:
    cd /Users/dgrauet/sandbox/ltx-reference
    uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/dump_pt_sampler.py
"""

from __future__ import annotations

import numpy as np
import torch
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_pipelines.utils.helpers import post_process_latent, timesteps_from_mask


def make_inputs(seed: int, n_total: int, n_kf: int, channels: int = 128):
    """Build (latent, clean_latent, denoise_mask, denoised_x0) with controlled values.

    Layout: first (n_total - n_kf) tokens are noisy/generation (mask=1),
    last n_kf tokens are keyframe (mask=0, clean values preserved).
    """
    rng = np.random.default_rng(seed)
    # Generation tokens: noisy at sigma=1.0 (pure noise + zero clean)
    n_gen = n_total - n_kf
    noise_gen = rng.standard_normal((1, n_gen, channels)).astype(np.float32)
    # Keyframe tokens: clean values
    kf_clean = rng.standard_normal((1, n_kf, channels)).astype(np.float32)
    # x: actual current latent — gen tokens noisy, kf tokens clean
    x = np.concatenate([noise_gen, kf_clean], axis=1)
    # clean_latent: zeros for gen (we have no clean for them yet) + kf values
    clean_latent = np.concatenate([np.zeros_like(noise_gen), kf_clean], axis=1)
    # denoise_mask: 1 for gen, 0 for kf
    mask = np.concatenate(
        [
            np.ones((1, n_gen, 1), dtype=np.float32),
            np.zeros((1, n_kf, 1), dtype=np.float32),
        ],
        axis=1,
    )
    # Mock denoised model output — should be ANYTHING since mask blend
    # forces the kf portion back to clean. Make it intentionally noisy
    # in the kf region so we can see if preservation works.
    denoised = rng.standard_normal((1, n_total, channels)).astype(np.float32)
    return x, clean_latent, mask, denoised


def main() -> None:
    n_total, n_kf = 313, 308  # half-res hedgehog: 5*14*22 + 308 (one keyframe of 14*22)
    x, clean_latent, mask, denoised = make_inputs(seed=42, n_total=n_total, n_kf=n_kf)

    x_t = torch.from_numpy(x)
    clean_t = torch.from_numpy(clean_latent)
    mask_t = torch.from_numpy(mask)
    denoised_t = torch.from_numpy(denoised)

    # 1) post-process (mask blend)
    post_pt = post_process_latent(denoised_t, mask_t, clean_t).numpy()

    # 2) timesteps from mask + sigma=0.7
    sigma_scalar = 0.7
    timesteps_pt = timesteps_from_mask(mask_t, sigma_scalar).numpy()

    # 3) Euler step
    sigmas = torch.tensor([0.7, 0.5, 0.3, 0.0])  # 3-step schedule
    stepper = EulerDiffusionStep()
    next_x_pt = stepper.step(x_t, torch.from_numpy(post_pt), sigmas, step_index=0).numpy()

    # 4) Iterate 3 steps (the kf tokens MUST stay bit-exact across all)
    state = x_t.clone()
    for step in range(3):
        # Same mock denoised at each step (deterministic test)
        post = post_process_latent(denoised_t, mask_t, clean_t)
        state = stepper.step(state, post, sigmas, step_index=step)

    state_after_3_pt = state.numpy()

    np.savez(
        "/tmp/keyframe_sampler_parity_pt.npz",
        x=x,
        clean_latent=clean_latent,
        mask=mask,
        denoised=denoised,
        post=post_pt,
        timesteps=timesteps_pt,
        next_x_step0=next_x_pt,
        state_after_3=state_after_3_pt,
    )
    print("wrote /tmp/keyframe_sampler_parity_pt.npz")
    # Sanity: kf region of state_after_3 must equal kf region of x (preserved)
    kf_init = x[:, n_total - n_kf :, :]
    kf_after = state_after_3_pt[:, n_total - n_kf :, :]
    print(f"kf preservation max_abs after 3 steps (should be 0): {np.max(np.abs(kf_init - kf_after)):.4e}")


if __name__ == "__main__":
    main()
