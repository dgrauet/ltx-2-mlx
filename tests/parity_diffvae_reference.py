"""Torch reference harness for the diffusion video decoder parity tests.

Standalone, NOT a pytest module: needs torch + upstream ltx-core. Run it from the repository
root in a disposable environment::

    uv run --no-project --with torch --with numpy --with safetensors --with einops \
      --with "ltx-core @ git+https://github.com/Lightricks/LTX-2@a95ab856bf29407b6b066ede0abe1846050db56c#subdirectory=packages/ltx-core" \
      python tests/parity_diffvae_reference.py --pack /path/to/ltx-2.5-mlx-q8 \
      --out /tmp/diffvae_parity.npz

Builds the upstream ``DiffusionVideoDecoder`` (via
``ltx_core.model.video_vae.model_configurator._build_diffusion_video_decoder``) from the pack's
metadata config, loads our ``vae_decoder_av.safetensors`` (reverse key mapping), applies
``DiffVAEMode.CHUNKED_EAGER`` (the upstream default the MLX port reproduces; on a host without
natten ``apply_diffvae_config`` resolves its attention to the eager tiled SDPA fallback, which
has the same exact-NA semantics), runs on CPU in fp32 on tiny latents with a captured noise
tensor, and dumps every stage boundary plus the final pixels.
"""

from __future__ import annotations

import argparse
import json
import struct

import numpy as np
import torch
from safetensors.torch import load_file


def read_config(path: str) -> dict:
    """The whole ``vae`` config dict from the pack metadata (decoder fields under ``decoder``)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return json.loads(header["__metadata__"]["config"])["vae"]


def to_upstream_state_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reverse the pack layout: strip prefix, split qkv rows, rename t_embedder / stats.

    Weights are bf16 on disk; they are widened to fp32 here because the parity run is fp32.
    """
    sd: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        k = key.removeprefix("vae_decoder_av.")
        t = value.float()
        if k.endswith(("attn.qkv.weight", "attn.qkv.bias")):
            suffix = "weight" if k.endswith("weight") else "bias"
            base = k[: -len(f"qkv.{suffix}")]
            q, kk, vv = torch.chunk(t, 3, dim=0)
            sd[f"{base}qkv.to_q.{suffix}"] = q
            sd[f"{base}qkv.to_k.{suffix}"] = kk
            sd[f"{base}qkv.to_v.{suffix}"] = vv
        elif k.startswith("t_embedder.mlp.0."):
            sd["t_embedder.timestep_embedder.linear_1." + k.split(".")[-1]] = t
        elif k.startswith("t_embedder.mlp.2."):
            sd["t_embedder.timestep_embedder.linear_2." + k.split(".")[-1]] = t
        elif k == "per_channel_statistics.mean":
            sd["per_channel_statistics.mean-of-means"] = t
        elif k == "per_channel_statistics.std":
            sd["per_channel_statistics.std-of-means"] = t
        else:
            sd[k] = t
    return sd


def build_decoder(pack: str) -> torch.nn.Module:
    """Upstream ``DiffusionVideoDecoder`` in fp32 eval mode, chunked-eager, our weights loaded."""
    from ltx_core.model.video_vae.model_configurator import _build_diffusion_video_decoder
    from ltx_core.model.video_vae.transformer.apply import apply_diffvae_config
    from ltx_core.model.video_vae.transformer.config import DiffVAEMode

    weights_path = f"{pack}/vae_decoder_av.safetensors"
    dec = _build_diffusion_video_decoder(read_config(weights_path))
    missing, unexpected = dec.load_state_dict(to_upstream_state_dict(load_file(weights_path)), strict=False)
    # ``rope_inv_*`` and ``default_inference_timesteps`` are non-persistent buffers: never in a
    # checkpoint, always rebuilt by ``__init__``. Anything else missing is a real mismatch.
    assert not [m for m in missing if "rope_inv" not in m and "default_inference_timesteps" not in m], missing
    assert not unexpected, unexpected
    dec = dec.float().eval()
    apply_diffvae_config(dec, DiffVAEMode.CHUNKED_EAGER.resolve())
    return dec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dec = build_decoder(args.pack)
    out: dict[str, np.ndarray] = {}
    hooks = []

    # ``.clone()`` is load-bearing: the chunked stage-5 path recycles one activation buffer
    # across the diffusion blocks, so a plain view would leave every slot holding the LAST
    # block's values.
    def dump(name):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            out[name] = tensor.detach().float().clone().numpy()

        return _hook

    # Stage hooks fire on the LAST block of each det stage, i.e. before that stage's upsample.
    for s, stage in enumerate(dec.det_stages):
        hooks.append(stage[-1].register_forward_hook(dump(f"s{s + 1}.out")))

    # The chunked stage-5 path calls ``block.forward_x_ctx(...)`` directly rather than the
    # module's ``__call__``, so a registered forward hook would never fire. Wrap the bound
    # method instead.
    def wrap_block(block, name):
        original = block.forward_x_ctx

        def wrapped(*a, _orig=original, _name=name, **kw):
            result = _orig(*a, **kw)
            out[_name] = result.detach().float().clone().numpy()
            return result

        block.forward_x_ctx = wrapped

    for i, block in enumerate(dec.diff_blocks):
        wrap_block(block, f"s5.b{i}.out")

    torch.manual_seed(0)
    for tag, shape in (("a", (1, 128, 3, 7, 7)), ("b", (1, 128, 2, 5, 9))):
        latent = torch.randn(shape)
        out[f"{tag}.in.latent"] = latent.numpy()

        # The decode draws its stage-5 noise with a single ``torch.randn`` call; capture it so
        # the MLX side can be fed the exact same tensor (the two RNGs cannot agree otherwise).
        real_randn = torch.randn
        captured: dict[str, torch.Tensor] = {}

        def fake_randn(*size, _real=real_randn, _captured=captured, **kwargs):
            drawn = _real(*size, **kwargs)
            _captured.setdefault("noise", drawn.clone())
            return drawn

        torch.randn = fake_randn
        try:
            with torch.no_grad():
                pixels = dec.forward(latent, generator=torch.Generator().manual_seed(1))
        finally:
            torch.randn = real_randn

        out[f"{tag}.in.noise"] = captured["noise"].float().numpy()
        out[f"{tag}.out.pixels"] = pixels.float().numpy()
        for key in list(out):
            if key.startswith("s") and not key.startswith(f"{tag}."):
                out[f"{tag}.{key}"] = out.pop(key)

    for hook in hooks:
        hook.remove()

    from ltx_core.model.video_vae.keyframes import DecodeKeyframes as TorchDecodeKeyframes

    def wrap_kf_block(block, name):
        original = block.forward_with_keyframes

        def wrapped(*a, _orig=original, _name=name, **kw):
            x, stream = _orig(*a, **kw)
            out[_name + ".out"] = x.detach().float().clone().numpy()
            out[_name + ".kf"] = stream.x.detach().float().clone().numpy()
            return x, stream

        block.forward_with_keyframes = wrapped

    def wrap_kf_diff(block, name):
        original = block.forward_x_ctx_with_keyframes

        def wrapped(*a, _orig=original, _name=name, **kw):
            x, kx = _orig(*a, **kw)
            out[_name + ".out"] = x.detach().float().clone().numpy()
            out[_name + ".kf"] = kx.detach().float().clone().numpy()
            return x, kx

        block.forward_x_ctx_with_keyframes = wrapped

    for s, stage in enumerate(dec.det_stages):
        wrap_kf_block(stage[-1], f"s{s + 1}")
    for i, block in enumerate(dec.diff_blocks):
        wrap_kf_diff(block, f"s5.b{i}")

    torch.manual_seed(1)
    for tag, shape, indices in (("a", (1, 128, 3, 7, 7), (5, 12)), ("b", (1, 128, 2, 5, 9), (3, 7))):
        latent = torch.randn(shape)
        kf_latents = torch.randn((1, 128, len(indices), shape[3], shape[4]))
        kf = TorchDecodeKeyframes(latents=kf_latents, pixel_frame_indices=torch.tensor(indices, dtype=torch.int64))
        real_randn = torch.randn
        draws: list[torch.Tensor] = []

        def fake_randn(*size, _real=real_randn, _draws=draws, **kwargs):
            drawn = _real(*size, **kwargs)
            _draws.append(drawn.clone())
            return drawn

        torch.randn = fake_randn
        try:
            with torch.no_grad():
                pixels = next(dec._decode_pixels_with_keyframes(latent, kf, None, torch.Generator().manual_seed(1)))
        finally:
            torch.randn = real_randn
        assert len(draws) == 2, [d.shape for d in draws]  # video canvas noise, then the plane noise
        out[f"{tag}.k.in.latent"] = latent.numpy()
        out[f"{tag}.k.in.kf_latents"] = kf_latents.numpy()
        out[f"{tag}.k.in.kf_indices"] = np.array(indices, dtype=np.int64)
        out[f"{tag}.k.in.noise"] = draws[0].float().numpy()
        out[f"{tag}.k.in.kf_noise"] = draws[1].float().numpy()
        out[f"{tag}.k.out.pixels"] = pixels.float().numpy()
        for key in list(out):
            if key.startswith("s") and not key.startswith(f"{tag}."):
                out[f"{tag}.k.{key}"] = out.pop(key)

    np.savez(args.out, **out)
    print("wrote", args.out, len(out), "arrays")
    for key in sorted(out):
        print(f"  {key:24s} {out[key].shape}")


if __name__ == "__main__":
    main()
