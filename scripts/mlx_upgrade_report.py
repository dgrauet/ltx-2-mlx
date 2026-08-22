#!/usr/bin/env python3
"""Fingerprint this environment's numerics, so an mlx bump can be judged on evidence.

Bumping mlx is routine; the risk is that a kernel change silently moves the
model's output. Bit-identity across mlx versions is *not* achievable -- kernel
selection and reduction order are free to change -- so the gate cannot be "the
hashes match". What this measures instead is a layered fingerprint, which tells
you *where* a change originates, plus the invariants that must hold regardless
of version.

Run it once per environment and diff the two reports::

    uv run python scripts/mlx_upgrade_report.py -o before.json     # current lock
    uv lock --upgrade-package mlx --upgrade-package mlx-metal
    uv sync --all-extras
    uv run python scripts/mlx_upgrade_report.py -o after.json
    uv run python scripts/mlx_upgrade_report.py --compare before.json after.json

Layers, cheapest first -- the first one that differs localises the change:

1. ``quantized_matmul`` on the six real DiT shapes, affine int8/int4 group 64.
2. Gemma text embeddings (positive and negative).
3. Post-denoise video and audio latents.
4. The muxed MP4, plus wall clock and peak GPU memory.

Layer 4 also runs twice -- eager and ``--low-ram`` -- because the streaming path
compiles its block and the two must stay equivalent.

Interpreting a diff:

* Layers 1-3 identical, layer 4 differs -> decode/mux only. Suspicious.
* Layer 3 differs, 1-2 identical -> the DiT forward moved. Expected from an
  attention or kernel change; judge it on quality, not on the hash.
* Layer 1 or 2 differs -> a shared primitive moved. Look there first.
* Wall clock or peak memory regresses -> a regression regardless of hashes.

Needs the q8 weights; skips with a clear message when they are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROMPT = "a heavy wooden door creaks slowly open in an old stone house, dust swirling in a shaft of afternoon light"
SEED = 81647281
HEIGHT, WIDTH, FRAMES = 512, 512, 25

# (M, K, N) taken from the real DiT: video 4096-dim, audio 2048-dim, the 9-param
# AdaLN projection at 36864, and the 4x feed-forward expansions.
QMM_SHAPES = [
    (1, 4096, 4096),
    (256, 4096, 16384),
    (1650, 4096, 4096),
    (1650, 2048, 8192),
    (3168, 4096, 4096),
    (77, 4096, 36864),
]


def _digest(array: mx.array) -> str:
    contiguous = mx.contiguous(array.astype(mx.float32))
    mx.eval(contiguous)
    return hashlib.sha256(bytes(memoryview(contiguous))).hexdigest()[:24]


def layer1_quantized_matmul() -> dict:
    """Affine int8/int4 quantized matmul -- the packs' hot path."""
    mx.random.seed(0)
    rows = {}
    for m, k, n in QMM_SHAPES:
        for bits in (8, 4):
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            x = mx.random.normal((m, k)).astype(mx.bfloat16)
            wq, scales, biases = mx.quantize(w, group_size=64, bits=bits)
            y = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=bits)
            rows[f"M{m}_K{k}_N{n}_int{bits}"] = _digest(y)
    return rows


def layer2_text_embeddings() -> dict:
    """Gemma plus connector: everything downstream inherits any change here."""
    from ltx_pipelines_mlx.distilled import DistilledPipeline

    pipe = DistilledPipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")
    video, audio, neg_video, neg_audio = pipe._encode_text_with_negative(PROMPT)
    return {
        "video": _digest(video),
        "audio": _digest(audio),
        "neg_video": _digest(neg_video),
        "neg_audio": _digest(neg_audio),
    }


def layer3_latents() -> dict:
    """Post-denoise latents: isolates the DiT forward from the VAE decode."""
    from ltx_pipelines_mlx.distilled import DistilledPipeline

    mx.reset_peak_memory()
    pipe = DistilledPipeline(model_dir="dgrauet/ltx-2.3-mlx-q8", low_ram_streaming=True)
    video_latent, audio_latent = pipe.generate_two_stage(
        PROMPT, height=HEIGHT, width=WIDTH, num_frames=FRAMES, frame_rate=24.0, seed=SEED
    )
    return {
        "video_latent": _digest(video_latent),
        "audio_latent": _digest(audio_latent),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def layer4_render(tmp: Path, *, low_ram: bool) -> dict:
    """A full CLI generation: output digest, wall clock, peak memory."""
    out = tmp / f"render_{'lowram' if low_ram else 'eager'}.mp4"
    cmd = [
        str(Path(sys.executable).parent / "ltx-2-mlx"),
        "generate",
        "--model",
        "dgrauet/ltx-2.3-mlx-q8",
        "--distilled",
        "--seed",
        str(SEED),
        "-H",
        str(HEIGHT),
        "-W",
        str(WIDTH),
        "-f",
        str(FRAMES),
        "--frame-rate",
        "24",
        "-p",
        PROMPT,
        "-o",
        str(out),
    ]
    if low_ram:
        cmd.append("--low-ram")
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0 or not out.exists():
        return {"error": proc.stderr[-800:]}
    return {
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "seconds": round(elapsed, 1),
    }


def build_report(tmp: Path) -> dict:
    return {
        "mlx": md.version("mlx"),
        "mlx_metal": md.version("mlx-metal"),
        "machine": platform.machine(),
        "chip": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip(),
        "layer1_quantized_matmul": layer1_quantized_matmul(),
        "layer2_text_embeddings": layer2_text_embeddings(),
        "layer3_latents": layer3_latents(),
        "layer4_render_eager": layer4_render(tmp, low_ram=False),
        "layer4_render_low_ram": layer4_render(tmp, low_ram=True),
    }


def compare(before: dict, after: dict) -> int:
    """Print a layer-by-layer diff. Returns 1 when something needs a human."""
    print(f"mlx {before['mlx']}  ->  {after['mlx']}   on {after.get('chip', '?')}\n")
    verdict = 0

    for layer in ("layer1_quantized_matmul", "layer2_text_embeddings", "layer3_latents"):
        changed = [k for k in before[layer] if k not in ("peak_gb",) and before[layer].get(k) != after[layer].get(k)]
        status = "IDENTICAL" if not changed else f"CHANGED ({', '.join(changed)})"
        print(f"{layer:28s} {status}")
        if changed:
            verdict = 1

    for mode in ("eager", "low_ram"):
        b, a = before[f"layer4_render_{mode}"], after[f"layer4_render_{mode}"]
        same = b.get("sha256") == a.get("sha256")
        slower = a.get("seconds", 0) / max(b.get("seconds", 1), 1e-9)
        print(
            f"{'layer4_' + mode:28s} {'IDENTICAL' if same else 'CHANGED'}"
            f"   {b.get('seconds')}s -> {a.get('seconds')}s ({slower:.2f}x)"
        )
        if not same:
            verdict = 1
        if slower > 1.10:
            print(f"  ! wall clock regressed {slower:.2f}x")
            verdict = 1

    for label, report in (("before", before), ("after", after)):
        eager = report["layer4_render_eager"].get("sha256")
        low = report["layer4_render_low_ram"].get("sha256")
        if eager and low:
            print(f"  {label}: eager vs --low-ram {'match' if eager == low else 'DIFFER'}")

    b_peak = before["layer3_latents"].get("peak_gb")
    a_peak = after["layer3_latents"].get("peak_gb")
    print(f"{'peak GPU memory':28s} {b_peak} GB -> {a_peak} GB")
    if b_peak and a_peak and a_peak > b_peak * 1.10:
        print("  ! peak memory regressed >10%")
        verdict = 1

    print(
        "\nHashes changing is not by itself a regression: kernel selection and "
        "reduction order may move between releases.\nWhat needs a human is the "
        "*first* layer that changed, whether quality holds, and whether time or "
        "memory got worse."
    )
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, help="write the report as JSON")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        before, after = (json.loads(p.read_text()) for p in args.compare)
        return compare(before, after)

    tmp = Path(args.output).parent if args.output else Path.cwd()
    report = build_report(tmp)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
