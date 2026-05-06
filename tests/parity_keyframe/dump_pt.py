"""Dump upstream PT VideoConditionByKeyframeIndex outputs for 3 test cases.

Run from inside the upstream venv:
    cd /Users/dgrauet/sandbox/ltx-reference
    uv run python /Users/dgrauet/Work/mlx/ports/ltx-2-mlx/tests/parity_keyframe/dump_pt.py

Saves /tmp/keyframe_parity_pt.npz with:
    case{i}_positions       (B, 3, N, 2)  — upstream raw [start, end]
    case{i}_positions_mid   (B, N, 3)     — collapsed midpoint, equivalent to ours
    case{i}_denoise_mask    (B, N+M, 1)
    case{i}_attention_mask  (B, N+M, N+M) or None
    case{i}_clean_latent    (B, N+M, C)
    case{i}_latent          (B, N+M, C)
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core.types import LatentState, SpatioTemporalScaleFactors, VideoLatentShape


@dataclasses.dataclass
class FakeLatentTools:
    """Minimal VideoLatentTools surrogate for keyframe.apply_to."""

    target_shape: VideoLatentShape
    fps: float
    causal_fix: bool
    scale_factors: SpatioTemporalScaleFactors
    patchifier: VideoLatentPatchifier


def make_state(seed: int, n_tokens: int, channels: int = 128) -> LatentState:
    """Build an existing latent state with seeded random latent + ones mask + no positions."""
    rng = np.random.default_rng(seed)
    latent = torch.from_numpy(rng.standard_normal((1, n_tokens, channels)).astype(np.float32))
    return LatentState(
        latent=latent,
        clean_latent=latent.clone(),
        denoise_mask=torch.ones((1, n_tokens, 1), dtype=torch.float32),
        positions=torch.zeros((1, 3, n_tokens, 2), dtype=torch.float32),
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
    """Apply VideoConditionByKeyframeIndex.apply_to and capture outputs."""
    rng = np.random.default_rng(seed)
    keyframe = torch.from_numpy(rng.standard_normal((1, channels, 1, h, w)).astype(np.float32))

    target = VideoLatentShape(batch=1, channels=channels, frames=5, height=h, width=w)
    tools = FakeLatentTools(
        target_shape=target,
        fps=fps,
        causal_fix=True,
        scale_factors=SpatioTemporalScaleFactors(8, 32, 32),
        patchifier=VideoLatentPatchifier(patch_size=1),
    )

    state = make_state(seed=seed + 1, n_tokens=n_existing, channels=channels)

    cond = VideoConditionByKeyframeIndex(
        keyframes=keyframe,
        frame_idx=frame_idx,
        strength=1.0,
        num_pixel_frames=num_pixel_frames,
    )
    out = cond.apply_to(state, tools)

    positions = out.positions.detach().cpu().numpy()
    pos_start = positions[..., 0]
    pos_end = positions[..., 1]
    positions_mid = ((pos_start + pos_end) / 2.0).transpose(0, 2, 1)

    return {
        f"{label}_positions": positions,
        f"{label}_positions_mid": positions_mid.astype(np.float32),
        f"{label}_latent": out.latent.detach().cpu().numpy().astype(np.float32),
        f"{label}_clean_latent": out.clean_latent.detach().cpu().numpy().astype(np.float32),
        f"{label}_denoise_mask": out.denoise_mask.detach().cpu().numpy().astype(np.float32),
        f"{label}_attention_mask": (
            out.attention_mask.detach().cpu().numpy().astype(np.float32)
            if out.attention_mask is not None
            else np.array(0.0, dtype=np.float32)
        ),
        f"{label}_keyframe": keyframe.detach().cpu().numpy().astype(np.float32),
    }


def main() -> None:
    out: dict[str, np.ndarray] = {}
    out.update(run_case("case0_start_halfres", frame_idx=0, h=14, w=22))
    out.update(run_case("case1_end_halfres", frame_idx=32, h=14, w=22))
    out.update(run_case("case2_end_fullres", frame_idx=32, h=28, w=44))
    np.savez("/tmp/keyframe_parity_pt.npz", **out)
    print(f"wrote /tmp/keyframe_parity_pt.npz with {len(out)} arrays")
    for k, v in out.items():
        print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else 'scalar'} dtype={v.dtype}")


if __name__ == "__main__":
    main()
