# ltx-2-mlx

Pure MLX port of [LTX-2](https://github.com/Lightricks/LTX-2) for Apple Silicon. Three-package monorepo mirroring the reference structure — inference, pipelines, and training — running natively on Metal.

## Features

- **Text-to-Video** — generate video + stereo 48kHz audio from a text prompt
- **Image-to-Video** — animate a reference image
- **Audio-to-Video** — generate video conditioned on an audio track
- **Retake / Extend** — edit existing videos (regenerate segments, add frames)
- **Keyframe interpolation** — smooth transition between reference images
- **IC-LoRA** — reference video conditioning (depth/pose/edges)
- **HDR IC-LoRA** — LogC3-compressed HDR generation (V2V upgrade or pure T2V) producing linear HDR `.npz` + SDR mp4 preview
- **LipDub** *(experimental)* — lip-dub a reference video by re-syncing visuals to the source audio
- **Two-stage generation** — half-res → neural upscale → refine
- **HQ generation** — res_2s second-order sampler + CFG/STG guidance
- **Prompt enhancement** — Gemma 3 12B rewrites short prompts into detailed descriptions
- **Training** — LoRA fine-tuning with flow matching (T2V and V2V strategies)
- **Block streaming (`--low-ram`)** — stream transformer blocks from disk so q8 fits 16 GB Macs and bf16 fits 32 GB Macs (covers generate / `--two-stage` / `--two-stages-hq` / a2v / keyframe / ic-lora; bind-time LoRA fusion supports custom distilled-lora-strength)
- **Modality tiling (`--tile-frames N --tile-spatial M`)** — split video tokens into spatial+temporal tiles to cap O(N²) attention activations. Combined with `--low-ram`, unblocks long / HD / 4K generations on Mac Studio (64-128 GB) that would otherwise OOM.
- **3 model variants** — bf16, int8, int4 (fits 16GB–64GB Macs)
- **3 upsamplers** — spatial 2x, spatial 1.5x, temporal 2x

> **Production readiness**: pipelines are classified as Stable / Beta /
> Experimental. See [docs/PIPELINE_MATURITY.md](docs/PIPELINE_MATURITY.md)
> before relying on a pipeline in production code. CLI subcommands flagged
> `[beta]` or `[experimental]` in `--help` may have known quality
> limitations or pre-1.0 LoRA dependencies.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 32GB+ RAM recommended (int8) or 16GB+ with `--low-ram`. 16GB minimum (int4 without streaming)
- ffmpeg (for video encoding)

## Installation

```bash
git clone https://github.com/dgrauet/ltx-2-mlx.git
cd ltx-2-mlx
uv sync --all-extras
```

## Quick Start

### CLI

```bash
# Text-to-Video — pick a pipeline mode (one of `--two-stage`, `--two-stages-hq`, `--one-stage`, `--distilled`).
# Two-stage is the upstream-recommended production default.
ltx-2-mlx generate --prompt "A sunset over the ocean" --two-stage -o sunset.mp4

# Image-to-Video (any mode supports --image)
ltx-2-mlx generate --prompt "Animate this" --image photo.jpg --two-stage -o animated.mp4

# HQ (res_2s sampler, highest quality)
ltx-2-mlx generate --prompt "A scene" --two-stages-hq --stage1-steps 20 -o hq.mp4

# Distilled two-stage (fastest, mirrors upstream DistilledPipeline)
ltx-2-mlx generate --prompt "A scene" --distilled -H 720 -W 1280 -o distilled.mp4

# One-stage dev + CFG (full target res, mirrors upstream TI2VidOneStagePipeline)
ltx-2-mlx generate --prompt "A scene" --one-stage -o one_stage.mp4

# Audio-to-Video
ltx-2-mlx a2v --prompt "Music video" --audio music.wav -o a2v.mp4

# Retake (regenerate frames 1-3 of a video)
ltx-2-mlx retake --prompt "New action" --video source.mp4 --start 1 --end 3 -o retake.mp4

# Extend (add 2 latent frames after)
ltx-2-mlx extend --prompt "Continue the scene" --video source.mp4 --extend-frames 2 -o extended.mp4

# Keyframe interpolation
ltx-2-mlx keyframe --prompt "Smooth transition" --start frame1.png --end frame2.png -o transition.mp4

# Prompt enhancement
ltx-2-mlx enhance --prompt "a cat" --mode t2v

# Use int4 model (fits 16GB)
ltx-2-mlx generate -p "A cat" --distilled -o cat.mp4 --model dgrauet/ltx-2.3-mlx-q4

# Block streaming: bf16 model on 32 GB Mac
ltx-2-mlx generate -p "A cat" --two-stage -o cat.mp4 --model dgrauet/ltx-2.3-mlx --low-ram

# Block streaming: q8 model on 16 GB Mac
ltx-2-mlx generate -p "A cat" --distilled -o cat.mp4 --model dgrauet/ltx-2.3-mlx-q8 --low-ram

# Block streaming works on every generate mode + a2v / keyframe / ic-lora
ltx-2-mlx generate -p "A cat" -o cat.mp4 --two-stage --low-ram
ltx-2-mlx generate -p "A cat" -o cat.mp4 --two-stages-hq --low-ram
ltx-2-mlx a2v -p "music video" --audio music.wav -o a2v.mp4 --low-ram
ltx-2-mlx keyframe -p "transition" --start a.png --end b.png -o kf.mp4 --low-ram
ltx-2-mlx ic-lora -p "scene" --lora lora.safetensors 1.0 --video-conditioning depth.mp4 1.0 --low-ram -o out.mp4

# HDR IC-LoRA — V2V upgrade an SDR video to linear HDR (saves out.mp4 + out.hdr.npz)
ltx-2-mlx hdr-ic-lora -p "cinematic golden hour" \
    --lora Lightricks/LTX-2.3-22b-IC-LoRA-HDR 1.0 \
    --video-conditioning source_sdr.mp4 1.0 --low-ram -o out.mp4

# HDR IC-LoRA — pure T2V (no conditioning video)
ltx-2-mlx hdr-ic-lora -p "a sunset over the ocean, vivid HDR" \
    --lora Lightricks/LTX-2.3-22b-IC-LoRA-HDR 1.0 --low-ram -o out.mp4

# Modality tiling: split video tokens for long/HD scenarios that exceed attention memory.
# Stack with --low-ram for max memory savings on big targets.
ltx-2-mlx generate -p "long scene" --two-stage --low-ram \
    --tile-frames 2 --tile-overlap 4 -o long.mp4
ltx-2-mlx generate -p "1080p scene" --two-stages-hq --low-ram \
    --tile-spatial 2 --tile-overlap 4 -H 1080 -W 1920 -o hd.mp4

# Model info
ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-q8
```

### Python API

Pick the pipeline class matching your target — every public class
mirrors an upstream Lightricks/LTX-2 pipeline.

Two-stage (recommended for most use cases — dev model + CFG + upscale):

```python
from ltx_pipelines_mlx import TI2VidTwoStagesPipeline

pipe = TI2VidTwoStagesPipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")
pipe.generate_and_save(
    prompt="A sunset over the ocean with waves crashing",
    output_path="sunset.mp4",
    height=480,
    width=704,
    num_frames=97,
    seed=42,
    image="photo.jpg",  # optional I2V
)
```

For other modes:

- `DistilledPipeline` — fastest (distilled half-res + upscale).
- `TI2VidTwoStagesHQPipeline` — highest quality (res_2s + CFG + upscale).
- `TI2VidOneStagePipeline` — full-res CFG, no upscaler dependency.

Audio-to-Video:

```python
from ltx_pipelines_mlx import A2VidPipelineTwoStage

pipe = A2VidPipelineTwoStage(model_dir="dgrauet/ltx-2.3-mlx-q8")
pipe.generate_and_save(
    prompt="A musician performing",
    output_path="a2v.mp4",
    audio_path="music.wav",
)
```

Retake / Extend (single class — extend is folded into `RetakePipeline`):

```python
from ltx_pipelines_mlx import RetakePipeline

pipe = RetakePipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")

# Retake: regenerate latent frames 1-3
video_lat, audio_lat = pipe.retake_from_video(
    prompt="A different scene",
    video_path="source.mp4",
    start_frame=1,
    end_frame=3,
)

# Extend: add 2 latent frames after
video_lat, audio_lat = pipe.extend_from_video(
    prompt="Continue the motion",
    video_path="source.mp4",
    extend_frames=2,
    direction="after",
)
```

## CLI Reference

> **Full pipeline + flag matrix**: see [docs/PIPELINES.md](docs/PIPELINES.md) for a complete matrix of every CLI subcommand, the pipeline class behind it, supported sampler / model defaults, and which memory / perf flags apply where.

```
ltx-2-mlx generate   T2V / I2V / two-stage / HQ generation
  --prompt, -p        Text prompt (required)
  --output, -o        Output .mp4 path (required)
  --model, -m         Model weights (default: dgrauet/ltx-2.3-mlx-q8)
  --height, -H        Video height (default: 480)
  --width, -W         Video width (default: 704)
  --frames, -f        Number of frames (default: 97)
  --seed, -s          Random seed (-1 = random)
  --image, -i         Reference image for I2V
  --steps             Denoising steps for one-stage (default: 8)
  --two-stage         Enable two-stage pipeline (dev model + CFG)
  --two-stages-hq                Enable HQ pipeline (res_2s sampler)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)
  --stage1-steps      Stage 1 steps (default: 30 standard, 15 HQ)
  --stage2-steps      Stage 2 steps (default: 3)
  --enhance-prompt    Enhance prompt with Gemma before generation
  --quiet, -q         Suppress progress output

ltx-2-mlx a2v        Audio-to-Video (two-stage, dev model + CFG)
  --audio, -a         Input audio file (required)
  --frame-rate        Output frame rate (required; LTX-2.3 trained at 24)
  --image, -i         Reference image for I2V (optional)
  --two-stages-hq                HQ mode (res_2s sampler for stage 1)
  --audio-start       Audio start time in seconds (default: 0)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)
  --stage1-steps      Stage 1 steps (default: 30 standard, 15 HQ)
  --stage2-steps      Stage 2 steps (default: 3)

ltx-2-mlx retake     Regenerate a time segment (dev model + CFG)
  --video, -v         Source video file (required)
  --start             Start latent frame index (required)
  --end               End latent frame index (required)
  --steps             Denoising steps (default: 30)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)
  --no-regen-audio    Preserve original audio

ltx-2-mlx extend     Add frames before/after (dev model + CFG)
  --video, -v         Source video file (required)
  --extend-frames     Number of latent frames to add (required)
  --direction         "before" or "after" (default: after)
  --steps             Denoising steps (default: 30)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)

ltx-2-mlx keyframe   Keyframe interpolation (two-stage, dev model + CFG)
  --start             Start keyframe image (required)
  --end               End keyframe image (required)
  --frame-rate        Output frame rate (required; LTX-2.3 trained at 24)
  --cfg-scale         CFG scale (default: 3.0)
  --stg-scale         STG scale (default: 0.0)
  --stage1-steps      Stage 1 steps (default: 30)
  --stage2-steps      Stage 2 steps (default: 3)

ltx-2-mlx hdr-ic-lora HDR IC-LoRA (two-stage, LogC3 → linear HDR)
  --lora PATH STRENGTH       HDR LoRA (e.g. Lightricks/LTX-2.3-22b-IC-LoRA-HDR), repeatable
  --video-conditioning P S   Optional SDR ref video for V2V upgrade (omit for pure T2V)
  --image, -i                Optional I2V reference image
  --stage1-steps             Stage 1 steps (default: 8)
  --stage2-steps             Stage 2 steps (default: 3)
  --conditioning-strength    IC-LoRA attention strength (default: 1.0)
  --skip-stage-2             Skip upscale stage (half-res HDR output)
                  → saves <output>.mp4 + <output>.hdr.npz (fp32 (F,H,W,3) linear HDR)

ltx-2-mlx lipdub     [experimental] Lip-dub a reference video → audio
  --reference-video          Reference video providing visuals + target audio (required)
  --lora PATH STRENGTH       LipDub IC-LoRA (e.g. Lightricks/LTX-2.3-22b-IC-LoRA-LipDub), exactly one
  --reference-strength       Reference video conditioning strength (default: 1.0)
  --stage1-steps / --stage2-steps
                  Frame count auto-derived from the reference video (snapped to 8k+1)

ltx-2-mlx enhance    Prompt enhancement (no generation)
  --mode              "t2v" or "i2v" (default: t2v)

ltx-2-mlx info       Model info and memory estimate
```

### Environment variables

- `LTX2_GEMMA_EVAL_EVERY=N` — per-layer `mx.eval` cadence in the Gemma forward (default: `1`, i.e. eval every layer). Keeps each Metal command buffer below the macOS GPU watchdog (~10 s) deadline. Set to `0` on Mac Studio / M-series Ultra owners who never see the watchdog crash to recover full lazy-graph throughput.
- `LTX2_DIT_EVAL_EVERY=N` — flush the DiT block loop every N blocks (default: `8`, splits 48 blocks into 6 command buffers). Same trade-off as above; set to `0` on machines that don't crash to maximise throughput.
- `LTX2_GEMMA_MAX_LENGTH=N` — cap Gemma padded sequence length (default: `1024`). Last-resort knob; quality risk on lower values because left-padded RoPE positions drift outside the LTX training distribution.
- `LTX2_GEMMA_MAX_LENGTH=N` — cap padded Gemma sequence length (default 1024). Reducing to 512/256 speeds Gemma forward proportionally but **shifts left-padded RoPE positions** away from the LTX training distribution (quality risk). Last-resort knob.

## Frame Count Reference

The number of frames must be `8k + 1` (due to VAE temporal compression 8x). Common values at 24 fps:

| Frames | Duration | Latent frames | Notes |
|--------|----------|---------------|-------|
| 9 | 0.4s | 2 | Minimal, for quick tests |
| 25 | 1.0s | 4 | Short clip |
| 41 | 1.7s | 6 | |
| 49 | 2.0s | 7 | |
| 65 | 2.7s | 9 | |
| 81 | 3.4s | 11 | |
| 97 | 4.0s | 13 | **Default** |
| 121 | 5.0s | 16 | |
| 145 | 6.0s | 19 | |
| 161 | 6.7s | 21 | |
| 193 | 8.0s | 25 | Requires 64GB+ RAM |

Higher frame counts require more RAM. With int4 on 32GB, 97 frames at 512x320 is comfortable. Reduce resolution for longer videos.

## Pre-converted Weights

| Variant | HuggingFace | Size | RAM |
|---------|-------------|------|-----|
| bf16 | [dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx) | ~42 GB | 64 GB+ |
| int8 | [dgrauet/ltx-2.3-mlx-q8](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8) | ~21 GB | 32 GB+ |
| int4 | [dgrauet/ltx-2.3-mlx-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) | ~12 GB | 16 GB+ |

Weights are pre-converted to MLX format by [mlx-forge](https://github.com/dgrauet/mlx-forge).

## Packages

| Package | Description |
|---------|-------------|
| `ltx-core-mlx` | Model library: DiT, VAE, audio, text encoder, conditioning, guidance |
| `ltx-pipelines-mlx` | Generation pipelines: T2V, I2V, A2V, retake, extend, keyframe, two-stage |
| `ltx-trainer-mlx` | Training: LoRA fine-tuning with flow matching |

## Resources

- [LTX-2](https://github.com/Lightricks/LTX-2) — Lightricks reference (ltx-core + ltx-pipelines + ltx-trainer)
- [mlx-forge](https://github.com/dgrauet/mlx-forge) — weight conversion tool
- [Pre-converted weights](https://huggingface.co/collections/dgrauet/ltx-23) — HuggingFace collection
- [MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework

## License

MIT
