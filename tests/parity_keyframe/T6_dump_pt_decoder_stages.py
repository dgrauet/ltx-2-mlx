"""T6 (PT side): Per-stage VAE decoder forward, dumps intermediates at each block."""

from __future__ import annotations

import json
import os

import numpy as np
import safetensors
import torch
from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType
from ltx_core.model.video_vae.video_vae import VideoDecoder

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/vae_decoder.safetensors"
)
EMBEDDED_CFG = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q8/"
    "snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d/embedded_config.json"
)


def load_pt_state_dict() -> dict:
    pt_state = {}
    with safetensors.safe_open(WEIGHTS, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open not a dict)
            t = f.get_tensor(k).to(torch.float32)
            new_k = k.replace("vae_decoder.", "", 1)
            new_k = new_k.replace("per_channel_statistics.mean", "per_channel_statistics.mean-of-means")
            new_k = new_k.replace("per_channel_statistics.std", "per_channel_statistics.std-of-means")
            if new_k.endswith(".conv.weight") and t.ndim == 5:
                t = t.permute(0, 4, 1, 2, 3).contiguous()
            pt_state[new_k] = t
    return pt_state


def instrumented_forward(decoder: VideoDecoder, sample: torch.Tensor, timestep: torch.Tensor) -> dict:
    """Replicate VideoDecoder.forward but capture intermediates."""
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        # Replicate decoder.forward bypass (no noise, timestep_conditioning=False)
        sample = decoder.per_channel_statistics.un_normalize(sample)
        out["after_denorm"] = sample.detach().cpu().numpy()

        sample = decoder.conv_in(sample, causal=decoder.causal)
        out["after_conv_in"] = sample.detach().cpu().numpy()

        for i, up_block in enumerate(decoder.up_blocks):
            sample = up_block(sample, causal=decoder.causal)
            out[f"after_up_block_{i}"] = sample.detach().cpu().numpy()

        sample = decoder.conv_norm_out(sample)
        out["after_conv_norm_out"] = sample.detach().cpu().numpy()

        sample = decoder.conv_act(sample)
        out["after_conv_act"] = sample.detach().cpu().numpy()

        sample = decoder.conv_out(sample, causal=decoder.causal)
        out["after_conv_out"] = sample.detach().cpu().numpy()

        from ltx_core.model.video_vae.ops import unpatchify

        sample = unpatchify(sample, patch_size_hw=decoder.patch_size, patch_size_t=1)
        out["final"] = sample.detach().cpu().numpy()

    return out


def main() -> None:
    with open(EMBEDDED_CFG) as f:
        cfg = json.load(f)["vae"]

    decoder = VideoDecoder(
        convolution_dimensions=cfg.get("dims", 3),
        in_channels=cfg.get("latent_channels", 128),
        out_channels=cfg.get("out_channels", 3),
        decoder_blocks=cfg.get("decoder_blocks", []),
        patch_size=cfg.get("patch_size", 4),
        norm_layer=NormLayerType(cfg.get("norm_layer", "pixel_norm")),
        causal=cfg.get("causal_decoder", False),
        timestep_conditioning=cfg.get("timestep_conditioning", False),
        decoder_spatial_padding_mode=PaddingModeType(cfg.get("spatial_padding_mode", "reflect")),
        base_channels=cfg.get("decoder_base_channels", 128),
    )

    state = load_pt_state_dict()
    decoder.load_state_dict(state, strict=False)
    decoder.train(False)
    decoder = decoder.to(torch.float32)

    rng = np.random.default_rng(42)
    latent_np = rng.standard_normal((1, 128, 2, 4, 4)).astype(np.float32)
    timestep = torch.zeros((1,), dtype=torch.float32)

    latent = torch.from_numpy(latent_np)
    out = instrumented_forward(decoder, latent, timestep)

    print(f"latent shape: {latent.shape}")
    print(f"PT up_blocks count: {len(decoder.up_blocks)}")
    for k, v in out.items():
        print(f"  {k}: shape={tuple(v.shape)} mean={v.mean():.4f} std={v.std():.4f}")

    np.savez("/tmp/T6_decoder_stages_pt.npz", latent=latent_np, **out)
    print("\nwrote /tmp/T6_decoder_stages_pt.npz")


if __name__ == "__main__":
    main()
