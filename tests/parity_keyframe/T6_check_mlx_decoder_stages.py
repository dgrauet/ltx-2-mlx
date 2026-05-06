"""T6 (MLX side): Per-stage decoder forward, compare each stage to PT."""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ltx_core_mlx.model.video_vae.normalization import pixel_norm
from ltx_core_mlx.model.video_vae.sampling import pixel_shuffle_3d
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder, unpatchify_spatial
from ltx_core_mlx.utils.weights import load_split_safetensors

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/vae_decoder.safetensors"
)
PT = "/tmp/T6_decoder_stages_pt.npz"


def instrumented_decode(decoder: VideoDecoder, latent: mx.array) -> dict:
    """Replicate VideoDecoder.decode but capture intermediates in BCFHW layout."""
    out: dict[str, np.ndarray] = {}

    # MLX decoder works in BFHWC; transpose input
    x = latent.transpose(0, 2, 3, 4, 1)
    x = decoder.denormalize_latent(x)
    out["after_denorm"] = np.asarray(x.transpose(0, 4, 1, 2, 3)).astype(np.float32)

    x = decoder.conv_in(x)
    out["after_conv_in"] = np.asarray(x.transpose(0, 4, 1, 2, 3)).astype(np.float32)

    upsample_idx = 0
    for i, block in enumerate(decoder.up_blocks):
        x = block(x)
        if i % 2 == 1:
            sf, tf = decoder._upsample_config[upsample_idx]
            x = pixel_shuffle_3d(x, spatial_factor=sf, temporal_factor=tf)
            if tf > 1:
                x = x[:, 1:, :, :, :]
            upsample_idx += 1
        out[f"after_up_block_{i}"] = np.asarray(x.transpose(0, 4, 1, 2, 3)).astype(np.float32)

    pn = pixel_norm(x)
    out["after_conv_norm_out"] = np.asarray(pn.transpose(0, 4, 1, 2, 3)).astype(np.float32)
    act = nn.silu(pn)
    out["after_conv_act"] = np.asarray(act.transpose(0, 4, 1, 2, 3)).astype(np.float32)

    co = decoder.conv_out(act)
    out["after_conv_out"] = np.asarray(co.transpose(0, 4, 1, 2, 3)).astype(np.float32)

    final = unpatchify_spatial(co, patch_size=4)
    out["final"] = np.asarray(final.transpose(0, 4, 1, 2, 3)).astype(np.float32)
    return out


def diff(name: str, mlx_arr: np.ndarray, pt_arr: np.ndarray) -> tuple[str, float]:
    if mlx_arr.shape != pt_arr.shape:
        return (f"shape mlx={mlx_arr.shape} pt={pt_arr.shape}", float("nan"))
    delta = float(np.max(np.abs(mlx_arr - pt_arr)))
    rel = delta / max(float(np.max(np.abs(pt_arr))), 1e-9)
    return (f"max_abs={delta:.4e}  rel={rel:.2%}", delta)


def main() -> None:
    pt = dict(np.load(PT))
    decoder = VideoDecoder()
    weights = load_split_safetensors(WEIGHTS, prefix="vae_decoder.")
    decoder.load_weights(list(weights.items()))

    latent_np = pt["latent"]
    latent_mlx = mx.array(latent_np)
    out = instrumented_decode(decoder, latent_mlx)

    stages = [
        "after_denorm",
        "after_conv_in",
        "after_up_block_0",
        "after_up_block_1",
        "after_up_block_2",
        "after_up_block_3",
        "after_up_block_4",
        "after_up_block_5",
        "after_up_block_6",
        "after_up_block_7",
        "after_up_block_8",
        "after_conv_norm_out",
        "after_conv_act",
        "after_conv_out",
        "final",
    ]
    print(f"{'stage':<22} {'verdict':<48}")
    print("-" * 70)
    first_div = None
    for stage in stages:
        if stage not in out:
            continue
        verdict, delta = diff(stage, out[stage], pt[stage])
        marker = "OK" if delta != delta or delta < 1e-3 else "DIVERGE"
        if marker == "DIVERGE" and first_div is None:
            first_div = stage
        print(f"  {stage:<22} {verdict:<40} {marker}")

    print()
    if first_div:
        print(f"FIRST DIVERGENCE: {first_div}")
        sys.exit(1)
    print("ALL STAGES OK")


if __name__ == "__main__":
    main()
