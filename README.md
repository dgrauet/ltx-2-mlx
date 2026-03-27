# ltx-2-mlx

Pure MLX port of [LTX-2](https://github.com/Lightricks/LTX-2) for Apple Silicon. Three-package monorepo mirroring the reference structure — inference, pipelines, and training — running natively on Metal.

## Features

- **Text-to-Video** — generate video + stereo 48kHz audio from a text prompt
- **Image-to-Video** — animate a reference image
- **Audio-to-Video** — generate video conditioned on an audio track
- **Retake / Extend** — edit existing videos (regenerate segments, add frames)
- **Keyframe interpolation** — smooth transition between reference images
- **IC-LoRA** — reference video conditioning (depth/pose/edges)
- **Two-stage generation** — half-res → neural upscale → refine
- **HQ generation** — res_2s second-order sampler + CFG/STG guidance
- **Prompt enhancement** — Gemma 3 12B rewrites short prompts into detailed descriptions
- **Training** — LoRA fine-tuning with flow matching (T2V and V2V strategies)
- **3 model variants** — bf16, int8, int4 (fits 16GB–64GB Macs)
- **3 upsamplers** — spatial 2x, spatial 1.5x, temporal 2x

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 32GB+ RAM recommended (int8), 16GB minimum (int4)
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
# Text-to-Video
ltx-2-mlx generate --prompt "A sunset over the ocean" --output sunset.mp4

# Image-to-Video
ltx-2-mlx generate --prompt "Animate this" --image photo.jpg -o animated.mp4

# Two-stage (higher quality)
ltx-2-mlx generate --prompt "A scene" --two-stage -o hires.mp4

# HQ (res_2s sampler, best quality)
ltx-2-mlx generate --prompt "A scene" --hq --stage1-steps 20 -o hq.mp4

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
ltx-2-mlx generate -p "A cat" -o cat.mp4 --model dgrauet/ltx-2.3-mlx-q4

# Model info
ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-q8
```

### Python API

```python
from ltx_pipelines_mlx import TextToVideoPipeline

pipe = TextToVideoPipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")
pipe.generate_and_save(
    prompt="A sunset over the ocean with waves crashing",
    output_path="sunset.mp4",
    height=480,
    width=704,
    num_frames=97,
    seed=42,
)
```

Image-to-Video:

```python
from ltx_pipelines_mlx import ImageToVideoPipeline

pipe = ImageToVideoPipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")
pipe.generate_and_save(
    prompt="Animate this scene with gentle motion",
    output_path="animated.mp4",
    image="photo.jpg",
)
```

Audio-to-Video:

```python
from ltx_pipelines_mlx import AudioToVideoPipeline

pipe = AudioToVideoPipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")
pipe.generate_and_save(
    prompt="A musician performing",
    output_path="a2v.mp4",
    audio_path="music.wav",
)
```

Retake / Extend:

```python
from ltx_pipelines_mlx import RetakePipeline, ExtendPipeline

# Retake: regenerate latent frames 1-3
pipe = RetakePipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")
video_lat, audio_lat = pipe.retake_from_video(
    prompt="A different scene",
    video_path="source.mp4",
    start_frame=1,
    end_frame=3,
)

# Extend: add 2 latent frames after
pipe = ExtendPipeline(model_dir="dgrauet/ltx-2.3-mlx-q8")
video_lat, audio_lat = pipe.extend_from_video(
    prompt="Continue the motion",
    video_path="source.mp4",
    extend_frames=2,
    direction="after",
)
```

## CLI Reference

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
  --hq                Enable HQ pipeline (res_2s sampler)
  --cfg-scale         CFG guidance scale (default: 3.0)
  --stg-scale         STG guidance scale (default: 0.0)
  --stage1-steps      Stage 1 steps (default: 30 standard, 15 HQ)
  --stage2-steps      Stage 2 steps (default: 3)
  --enhance-prompt    Enhance prompt with Gemma before generation
  --quiet, -q         Suppress progress output

ltx-2-mlx a2v        Audio-to-Video (two-stage, dev model + CFG)
  --audio, -a         Input audio file (required)
  --image, -i         Reference image for I2V (optional)
  --hq                HQ mode (res_2s sampler for stage 1)
  --fps               Frame rate (default: 24)
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
  --fps               Frame rate (default: 24)
  --cfg-scale         CFG scale (default: 3.0)
  --stg-scale         STG scale (default: 0.0)
  --stage1-steps      Stage 1 steps (default: 30)
  --stage2-steps      Stage 2 steps (default: 3)

ltx-2-mlx enhance    Prompt enhancement (no generation)
  --mode              "t2v" or "i2v" (default: t2v)

ltx-2-mlx info       Model info and memory estimate
```

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
