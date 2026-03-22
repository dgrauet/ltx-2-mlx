"""
Local reference test: run ltx-core DiT on CPU with minimal size.
Uses the already-cloned ltx-reference repo.
Run with: source /tmp/ltx-ref-venv/bin/activate && python scripts/local_reference_test.py
"""

from pathlib import Path

import numpy as np
import torch
from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.rope import LTXRopeType, precompute_freqs_cis
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm
from safetensors.torch import load_file

MODEL_DIR = sorted((Path.home() / ".cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-distilled/snapshots").iterdir())[
    -1
]
OUT = Path("/tmp/ltx_ref_outputs")
weights = load_file(str(MODEL_DIR / "transformer.safetensors"))


def load_adaln(prefix, dim, n):
    adaln = AdaLayerNormSingle(embedding_dim=dim, embedding_coefficient=n)
    state = {
        k.replace(prefix, "").replace("linear1", "linear_1").replace("linear2", "linear_2"): v
        for k, v in weights.items()
        if k.startswith(prefix)
    }
    adaln.load_state_dict(state)
    return adaln


def load_block(idx, video_cfg, audio_cfg):
    block = BasicAVTransformerBlock(idx=idx, video=video_cfg, audio=audio_cfg, rope_type=LTXRopeType.INTERLEAVED)
    bk = {}
    prefix = f"transformer.transformer_blocks.{idx}."
    for k, v in weights.items():
        if k.startswith(prefix):
            nk = k[len(prefix) :]
            if ".to_out." in nk and ".to_out.0." not in nk:
                nk = nk.replace(".to_out.", ".to_out.0.")
            nk = nk.replace("ff.proj_in.", "ff.net.0.proj.")
            nk = nk.replace("ff.proj_out.", "ff.net.2.")
            nk = nk.replace("audio_ff.proj_in.", "audio_ff.net.0.proj.")
            nk = nk.replace("audio_ff.proj_out.", "audio_ff.net.2.")
            bk[nk] = v
    block.load_state_dict(bk)
    return block


# Configs
video_cfg = TransformerConfig(
    dim=4096,
    heads=32,
    d_head=128,
    context_dim=4096,
    apply_gated_attention=True,
    cross_attention_adaln=True,
)
audio_cfg = TransformerConfig(
    dim=2048,
    heads=32,
    d_head=64,
    context_dim=2048,
    apply_gated_attention=True,
    cross_attention_adaln=True,
)

# Small test: 2x4x5 = 40 video tokens, 5 audio tokens
N_v, N_a = 40, 5
torch.manual_seed(42)
video_latent = torch.randn(1, N_v, 128, dtype=torch.bfloat16) * 0.5
audio_latent = torch.randn(1, N_a, 128, dtype=torch.bfloat16) * 0.5
video_text = torch.randn(1, 16, 4096, dtype=torch.bfloat16) * 0.1
audio_text = torch.randn(1, 16, 2048, dtype=torch.bfloat16) * 0.1

# Patchify
vx = torch.nn.functional.linear(
    video_latent,
    weights["transformer.patchify_proj.weight"],
    weights["transformer.patchify_proj.bias"],
)
ax = torch.nn.functional.linear(
    audio_latent,
    weights["transformer.audio_patchify_proj.weight"],
    weights["transformer.audio_patchify_proj.bias"],
)

# AdaLN
timestep = torch.tensor([1000.0])
av = load_adaln("transformer.adaln_single.", 4096, 9)
aa = load_adaln("transformer.audio_adaln_single.", 2048, 9)
pv = load_adaln("transformer.prompt_adaln_single.", 4096, 2)
pa = load_adaln("transformer.audio_prompt_adaln_single.", 2048, 2)
cvv = load_adaln("transformer.av_ca_video_scale_shift_adaln_single.", 4096, 4)
cva = load_adaln("transformer.av_ca_audio_scale_shift_adaln_single.", 2048, 4)
ga2v = load_adaln("transformer.av_ca_a2v_gate_adaln_single.", 4096, 1)
gv2a = load_adaln("transformer.av_ca_v2a_gate_adaln_single.", 2048, 1)

with torch.no_grad():
    vp, ve = av(timestep)
    ap_, ae = aa(timestep)
    pp, _ = pv(timestep)
    ppa, _ = pa(timestep)
    cvvp, _ = cvv(timestep)
    cvap, _ = cva(timestep)
    ga2vp, _ = ga2v(timestep)
    gv2ap, _ = gv2a(timestep)

# RoPE
vpos = torch.tensor([[[f + 0.5, h + 0.5, w + 0.5] for f in range(2) for h in range(4) for w in range(5)]])
apos = torch.tensor([[[t + 0.5] for t in range(5)]])
vpg = vpos.permute(0, 2, 1).unsqueeze(-1)
vpb = torch.cat([vpg - 0.5, vpg + 0.5], dim=-1)
v_rope = precompute_freqs_cis(
    vpb,
    dim=4096,
    out_dtype=torch.bfloat16,
    theta=10000.0,
    max_pos=[20, 2048, 2048],
    use_middle_indices_grid=True,
    num_attention_heads=32,
    rope_type=LTXRopeType.INTERLEAVED,
)
apg = apos.permute(0, 2, 1).unsqueeze(-1)
apb = torch.cat([apg - 0.5, apg + 0.5], dim=-1)
a_rope = precompute_freqs_cis(
    apb,
    dim=2048,
    out_dtype=torch.bfloat16,
    theta=10000.0,
    max_pos=[20],
    use_middle_indices_grid=True,
    num_attention_heads=32,
    rope_type=LTXRopeType.INTERLEAVED,
)


def mk(x, params, emb, ctx, prompt_p, cross_p, gate_p, rope):
    n = params.shape[-1] // x.shape[-1]
    return TransformerArgs(
        x=x,
        timesteps=params.reshape(1, 1, n, -1),
        positional_embeddings=rope,
        context=ctx,
        context_mask=None,
        self_attention_mask=None,
        enabled=True,
        prompt_timestep=prompt_p.reshape(1, 1, 2, -1),
        cross_scale_shift_timestep=cross_p.reshape(1, 1, 4, -1),
        cross_gate_timestep=gate_p.reshape(1, 1, 1, -1),
        cross_positional_embeddings=None,
        embedded_timestep=emb.unsqueeze(1),
    )


# Run ALL 48 blocks
print("Running 48 blocks on CPU (this takes a few minutes)...")
with torch.no_grad():
    for i in range(48):
        block = load_block(i, video_cfg, audio_cfg)
        va = mk(vx, vp, ve, video_text, pp, cvvp, ga2vp, v_rope)
        aargs = mk(ax, ap_, ae, audio_text, ppa, cvap, gv2ap, a_rope)
        vo, ao = block(va, aargs)
        vx, ax = vo.x, ao.x
        del block
        if i % 12 == 0 or i == 47:
            print(f"  Block {i}: video=[{vx.float().min():.1f}, {vx.float().max():.1f}]")

    # Output block
    sst = weights["transformer.scale_shift_table"]
    ss = sst.unsqueeze(0).unsqueeze(0) + ve.unsqueeze(1).unsqueeze(2)
    shift, scale = ss[:, :, 0], ss[:, :, 1]
    vx_n = rms_norm(vx) * (1 + scale) + shift
    vel = torch.nn.functional.linear(
        vx_n.to(weights["transformer.proj_out.weight"].dtype),
        weights["transformer.proj_out.weight"],
        weights["transformer.proj_out.bias"],
    )

print(f"\nVelocity: [{vel.float().min():.4f}, {vel.float().max():.4f}]")
np.save(OUT / "cpu_full_velocity_with_rope.npy", vel.float().numpy())

# Now do Euler step: x0 = x_input - sigma * velocity
x_input = torch.tensor(np.load(str(OUT / "full_video_latent.npy")), dtype=torch.bfloat16)
x0 = x_input - 1.0 * vel.to(x_input.dtype)
print(f"x0: [{x0.float().min():.4f}, {x0.float().max():.4f}], std={x0.float().std():.4f}")
print(f"x0 spatial std: {x0[0].float().std(dim=0).mean():.4f}")

np.save(OUT / "cpu_full_x0.npy", x0.float().numpy())
print("\nSaved! Compare x0 spatial std with MLX to check if model produces structured output.")
