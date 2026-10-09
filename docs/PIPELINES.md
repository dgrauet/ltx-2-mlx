# Pipelines guide — which pipeline, which flags

Current as of **v0.16.3**. Architecture and internals: [CLAUDE.md](../CLAUDE.md).
Stability tiers: [PIPELINE_MATURITY.md](PIPELINE_MATURITY.md).
User-facing overview: [README.md](../README.md).

Resolutions are written width × height in prose. On the command line the parser
takes `--height, -H` first and `--width, -W` second.

## Which pipeline?

Start from what you have:

- **Text only** → `generate --distilled` (fastest, 2.5 packs recommended) · `generate --two-stage` (dev + CFG, best default quality) · `generate --two-stages-hq` (res_2s sampler, slower) · `generate --one-stage` (native resolution ≤ 704 × 480, no upsampler).
- **Text + one or more images** → the same `generate` modes (all five, `--dfr` included) with `--image PATH FRAME STRENGTH` (repeatable).
- **Two images to interpolate between** → `keyframe` (dev + CFG) · on 2.5 packs also `generate --distilled` with a start and an end `--image` (about 4–6× faster, hardware per number on the `keyframe` card).
- **A control video** (depth / canny / pose / motion tracks) → `ic-lora`; **an SDR clip to upgrade to HDR (2.5 packs)** → `hdr-ic-lora`.
- **An audio track to drive the video** → `a2v` (dev + CFG) · `a2v --distilled` (distilled, no CFG, experimental).
- **An existing video to change** → `retake` (a time range; `retake --distilled` runs the distilled model, experimental) · `extend` (add frames; `extend --distilled` continues it on the distilled path, experimental) · `lipdub` (re-sync lips to audio, experimental).
- **A prompt to improve first** → the `enhance` subcommand.

Then pick by constraint: 16–32 GB machines add `--low-ram`; 1080p or clips over 8 s
add `--tile-spatial 2`; a sharper decode on 2.5 packs adds `--video-decoder diffusion`;
**maximum detail on 2.5 packs** → `generate` `--dfr` (experimental).

`generate` has no implicit mode. One of `--one-stage`, `--two-stage`, `--two-stages-hq`,
`--distilled` or `--dfr` is mandatory, because each maps 1:1 to an upstream pipeline class.

## Pipelines

One card per subcommand or mode. "Own flags" are the flags specific to that card.
Everything else is in [Common flags](#common-flags).

### `generate --distilled`

- **Produces:** T2V / I2V mp4 with audio. Half-res distilled pass, 2× latent upsample, 3-step distilled refine.
- **Packs:** 2.3 and 2.5. On 2.5 both stages use the ancestral sampler (stage-2 noise seeded from `seed + 20000`); 2.3 stays deterministic Euler. **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`. `--frames` is required on 2.3 packs and auto-predicted on 2.5.
- **Own flags:** `--stage1-steps` (8), `--stage2-steps` (3), `--nag` with `--negative-prompt` (experimental, see [NAG](#negative-prompt-without-cfg-nag)). No CFG, so `--cfg-scale`, `--stg-scale` and TeaCache do not apply, and `--negative-prompt` alone is rejected.
- **Example:** `ltx-2-mlx generate --distilled -p "a fox in the forest" -H 512 -W 768 -f 49 --frame-rate 24 -o fox.mp4`
- **Notes:** fastest mode. Quality sits slightly below the dev + CFG variants.

### `generate --two-stage`

- **Produces:** T2V / I2V mp4 with audio. Dev model + CFG at half resolution, 2× upsample, distilled 3-step refine of the video. Stage 2 keeps stage 1's audio as frozen conditioning (upstream v1.4.0 `freeze_audio=True`).
- **Packs:** 2.3 and 2.5. **Tier:** Stable. Best default quality per minute.
- **Required:** `--prompt`, `--output`, `--frame-rate`; `--frames` on 2.3 packs.
- **Own flags:** `--stage1-steps` (30), `--stage2-steps` (3), `--cfg-scale` (3.0), `--negative-prompt`, `--stg-scale` (1.0; each unit above 0 adds one extra forward pass per step — pass `--stg-scale 0` on 32 GB Macs for long clips), `--dev-transformer` (`transformer-dev.safetensors`), `--distilled-lora`, `--distilled-lora-strength` (1.0), `--enable-teacache`, `--teacache-thresh`.
- **Example:** `ltx-2-mlx generate --two-stage -p "a fox in the forest" -H 480 -W 704 -f 97 --frame-rate 24 --low-ram -o fox.mp4`
- **Cost (M2 Pro 32 GB, q8):** 704 × 480, 97 frames ≈ 1374 s, or ≈ 942 s with `--enable-teacache`. [Details](../CLAUDE.md#teacache-opt-in-stage-1-acceleration).

### `generate --two-stages-hq`

- **Produces:** the same two-stage output with the second-order res_2s sampler on both stages (stage 2 without guidance, as upstream). Unlike `--two-stage`, stage 2 re-noises the stage-1 audio and refines it with the video, as upstream does.
- **Packs:** 2.3 and 2.5. **Tier:** Stable. Roughly twice the stage-1 cost of `--two-stage`.
- **Required:** identical to `--two-stage`.
- **Own flags:** identical to `--two-stage`, except `--stage1-steps` defaults to 15 and `--stg-scale` to 0.0. Each step runs two model evaluations.
- **Example:** `ltx-2-mlx generate --two-stages-hq -p "a fox in the forest" -H 480 -W 704 -f 97 --frame-rate 24 --low-ram -o fox.mp4`
- **Cost (M2 Pro 32 GB, `--low-ram`):** 704 × 480, 97 frames ≈ 44 min at q8, ≈ 50 min at bf16. TeaCache at `--teacache-thresh 1.0` cuts a 576 × 384, 65-frame run from 1370 s to 768 s.

### `generate --one-stage`

- **Produces:** T2V / I2V mp4 with audio in a single dev + CFG pass at the target resolution. No upsampler, no stage 2.
- **Packs:** 2.3 and 2.5. **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`; `--frames` on 2.3 packs.
- **Own flags:** `--steps` (30), `--cfg-scale` (3.0), `--negative-prompt`, `--stg-scale` (1.0), `--dev-transformer`. No stage-2 or TeaCache flags.
- **Example:** `ltx-2-mlx generate --one-stage -p "a fox in the forest" -H 480 -W 704 -f 33 --frame-rate 24 --low-ram -o fox.mp4`
- **Cost (M2 Pro 32 GB, q8, `--low-ram`):** 704 × 480, 33 frames ≈ 2 min 31 s.
- **Notes:** pick it for native resolutions up to 704 × 480, or when you would rather not depend on the neural upsampler. `--two-stage` is faster at larger targets.

### `generate` `--dfr` *(experimental, LTX 2.5 packs only)*

- **Produces:** T2V / I2V mp4 (+ audio) — the DFR base path: half-res distilled stage with keyframe slots on a segment-aligned canvas, 2× latent upsample, then a full-res detailing stage with the official detailing IC-LoRA (`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`, strength 0.5) guided by the stage-1 latent. With `--video-decoder diffusion` the stage-2 keyframe slots are decoded as a keyframe-aware second stream (`--video-decoder conv`, the default, ignores them with a warning). `--temporal-upscalings T` adds T temporal x2 refine rounds on top: `(N-1)*2**T+1` frames at `frame_rate*2**T` fps, audio carried over unchanged from stage 1 (frozen, not re-denoised). `--spatial-upscalings 2` runs stage 1 at H/4 and stage 2 + the temporal rounds at H/2 (dims floored to multiples of 128 px, with a warning), then adds a full-res spatial epilogue: the carry keyframes are decoded one plane at a time, Lanczos-upsampled x2 and re-encoded as strength-1.0 keyframes, and the H/2 latent is spatially upsampled and re-denoised window by window (one window per last-round temporal tile, each pinned to the previous one; detailing LoRA, the H/2 latent as an IC-LoRA reference, frozen stage-1 audio, an opening frame anchoring frame 0 when no image does): one ancestral step on a 2×2 spatial tile grid, then the remaining steps on 4×4.
- **Packs:** 2.5 only. **Tier:** Experimental.
- **Required:** `--prompt`, `--output`, `--frame-rate` (`-f` optional: auto-predicted).
- **Own flags:** `--detailing-lora PATH_OR_REPO` (official LoRA, downloaded on first use — **gated repo**: accept the licence once at [huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) with the account `huggingface-cli login` uses, or the run stops before any model load; a local `.safetensors` path skips the download), `--stage1-steps` (8), `--stage2-steps` (3), `--image PATH FRAME STRENGTH` (repeatable), `--spatial-upscalings {1,2}` (1), `--temporal-upscalings {0,1,2}` (0), `--temporal-upsampler-path PATH` (default: the pack's `temporal_upscaler_x2_v1_0.safetensors`). `--nag` with `--negative-prompt` (experimental, see [NAG](#negative-prompt-without-cfg-nag); every pass, temporal rounds and epilogue included). Not accepted: `--num-generated-keyframes` (slots come from the canvas), `--enable-teacache`, `--cfg-scale`, `--stg-scale`, `--negative-prompt` without `--nag`; with `--temporal-upscalings` set or `--spatial-upscalings 2`, also not accepted: `--segment` (Prompt Relay), `--tile-frames` / `--tile-spatial`.
- **Example:** `ltx-2-mlx generate --dfr --model /path/to/ltx-2.5-mlx-q8 -p "a fox in the forest" -H 512 -W 768 -f 49 --frame-rate 24 --low-ram -o fox.mp4`
- **Cost (M2 Pro 32 GB, q8, `--low-ram`):** 768 × 512, 49 frames ≈ 277 s (8-step stage 1 at ~10.7 s/forward over 864 video tokens, 3-step stage 2 at ~53.5 s/forward over 4128 tokens — target + 2 keyframe slots + the half-res reference), peak Metal 14.0 GB, max RSS 10.8 GB. Frame 24 is visibly sharper (tree crowns, haze texture) than the plain `--distilled` render at the same seed. `--image` (I2V, frame 0 anchor) adds ~7 s. A 137-frame request pads to a 145-frame canvas (6 slots) and costs ≈ 783 s. `--video-decoder diffusion` at 1152 × 768, 25 frames runs untiled at ≈ 465 s, 12 GB peak Metal (the decoder now runs in its own bf16 whatever dtype the pipeline hands it; the first measurement, 22 GB, was the decode promoted to fp32 by an fp32 stage-2 latent). The keyframe-aware diffusion decode costs +58 % on the decode phase at 768 × 512 × 49 (157 s vs 100 s, 2 planes), same peak Metal. Temporal rounds at 768 × 512, 121 frames, `--no-audio`: `--temporal-upscalings 1` (241 frames @ 48 fps) ≈ 27 min (1599 s), `--temporal-upscalings 2` (481 @ 96) ≈ 67 min (4028 s). Spatial epilogue at 1536 × 1024, 49 frames, `--no-audio`: `--spatial-upscalings 2` ≈ 38 min (2272 s; epilogue 1941 s: one step on 2×2 tiles, then 4×4), peak Metal 11.2 GB; + `--temporal-upscalings 1` (97 frames @ 48 fps) ≈ 84 min (5015 s; epilogue 4251 s), peak Metal 16.8 GB.
- **Notes:** on I2V, the first-frame keyframe marker is now applied consistently (see [Details](../CLAUDE.md#dfr-base-path-generate---dfr-25-packs-experimental)). `--segment` (Prompt Relay) auto-distributes over the padded canvas, not the requested duration (e.g. 145 frames for `-f 137`), so segment boundaries shift by the padding before the tail is trimmed — pass explicit segment lengths for exact boundaries. [Details](../CLAUDE.md#dfr-base-path-generate---dfr-25-packs-experimental).

### `keyframe`

- **Produces:** an interpolation between a start image and an end image. Dev model + CFG at half res, then a distilled refine.
- **Packs:** 2.3 and 2.5 (2.5 needs `--dev-transformer transformer-dev.safetensors`). **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`, `--start`, `--end` (image paths).
- **Own flags:** `--frames` (97), `--start-strength` (1.0), `--end-strength` (1.0), `--stage1-steps`, `--stage2-steps`, `--cfg-scale` (3.0 video, 7.0 audio), `--negative-prompt`, `--stg-scale` (1.0), `--dev-transformer`, `--distilled-lora`, `--lora-strength` (1.0).
- **Example:** `ltx-2-mlx keyframe -p "the camera pushes in" --start a.jpg --end b.jpg -f 97 --frame-rate 24 -o out.mp4`
- **Notes:** the distilled model hallucinates on interpolation, so this pipeline always uses the dev model. Lower `--end-strength` gives the prompt more freedom to diverge from the end fixture.
- **Distilled alternative (2.5 packs):** `generate --distilled --image start.png 0 1.0 --image end.png last 1.0` interpolated cleanly in a comparison against `keyframe` at 768 × 512, 97 frames (a front-to-three-quarter turn and a front-to-profile turn, seeds 5 and 24, `LTX2_COMPUTE_DTYPE=float16`). On one machine (M1 Max, three-quarter turn, seed 5) `keyframe` took 951 s (cold start included) and the distilled run 165 s, 5.8×; `keyframe` took 725 s and 712 s on the other two M1 Max runs, and the remaining distilled runs (168–177 s) ran on an M4 Pro, which is slower than the M1 Max on this workload, so the speed-up on one machine is at least ~4×. The first and last frames match their images about as closely as `keyframe`'s do (PSNR against the image, first / last frame: 35.7–36.0 / 31.5–32.8 dB for the distilled runs with the end anchor at 1.0, 34.5–34.9 / 31.3–33.6 dB for `keyframe`; anchors at 0.7 give 35.8–36.5 / 30.9–32.6 dB), and an end anchor at strength 0.7 instead of 1.0 changed little. The two paths take different routes between the poses: on the three-quarter turn the distilled run turned later in the clip, and on the profile turn it briefly went past profile before settling. The anchors at `frame_idx > 0` are soft, so a pose the motion cannot plausibly reach is blended rather than reached; keep `keyframe` for 2.3 packs and when the motion between the images matters more than the time. [Details](../CLAUDE.md#keyframe-interpolation-pipeline).

### `ic-lora`

- **Produces:** control-conditioned video from a control clip (canny, depth, pose, motion tracks) plus an official Lightricks IC-LoRA.
- **Packs:** 2.3 only. No official 2.5 task IC-LoRAs exist yet. **Tier:** Stable.
- **Required:** `--prompt`, `--output`, `--frame-rate`, `--lora PATH STRENGTH`, `--video-conditioning PATH STRENGTH`.
- **Own flags:** `--frames` (97), `--conditioning-strength` (1.0), `--skip-stage-2`, `--upsample-only`, `--refine-steps`, `--single-stage`, `--stage1-steps`, `--stage2-steps`, `--dev-transformer`, `--distilled-lora`, `--distilled-lora-strength` (0.5 in dev mode), `--image`.
- **Example:**
  ```
  ltx-2-mlx ic-lora -p "a person walking" \
    --lora Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control 1.0 \
    --video-conditioning depth.mp4 1.0 \
    -H 704 -W 1280 -f 97 --frame-rate 24 --low-ram -o out.mp4
  ```
- **Cost (M2 Pro 32 GB, q8, `--low-ram`):** 1280 × 704, 97 frames ≈ 15 min.
- **Topologies:** the default is half-res generation then a control-blind stage-2 refine. `--single-stage` generates at full resolution in one pass and tracks the control most tightly. `--upsample-only` skips the refine for a fast draft, and adding `--refine-steps N` runs a control-aware refine instead. `--skip-stage-2` stops at half resolution.
- **Notes:** `--dev-transformer` switches on dev mode, which fuses the distilled LoRA alongside the task LoRA. This is also the recipe that preserves a source image's identity across a whole clip. [Details](../CLAUDE.md#ic-lora-pipeline) · [static-scene recipe](../CLAUDE.md#static-scene-i2v-recipe-preserve-identity).

### `hdr-ic-lora`

- **Produces:** `<output>.mp4`, a BT.2020/HLG 10-bit HEVC master, plus `<stem>_<exr-colorspace>_exr/frame_*.exr` beside it (ACEScg by default).
- **Packs:** 2.5 only (2.3 packs are refused). **Tier:** Experimental.
- **Install:** the EXR writer needs the optional `hdr` extra (`pip install 'ltx-pipelines-mlx[hdr]'`, or `uv sync --extra hdr` in this repo) and the HLG master needs an ffmpeg with the `libx265` encoder (Homebrew's has it). Both are checked when the pipeline is built, before any download.
- **Required:** `--input` (MP4/MOV or a folder of `*.exr`), `--output-path` / `--output` / `-o`, `--hdr-lora` (the gated `Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR`, as a local file), `--text-embeddings` (the scene-emb file from the same repo).
- **Own flags:** `--input-colorspace {srgb_gamma,srgb,acescg,acescct}` (default `srgb_gamma`), `--exr-colorspace {srgb_linear,acescg,acescct}` (default `acescg`), `--frame-rate` (EXR folders only, forbidden for MP4/MOV), `--high-quality` (2x frames internally, ~2x slower), `--no-keyframes`, `--keyframe-strength` (0.95). There is no prompt, size, frame-count or stage flag: the output follows the source (frame count must be 8k+1).
- **Example:**
  ```
  ltx-2-mlx hdr-ic-lora --model dgrauet/ltx-2.5-mlx-q8 --input source_sdr.mp4 \
    --hdr-lora ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors \
    --text-embeddings ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors \
    --low-ram -o out.mp4
  ```
- **Notes:** single stage on the distilled transformer with the LoRA fused at strength 1.0; both VAEs run in fp32 and the keyframe-aware diffusion decoder produces the pixels. Replaces the old LogC3 / `.hdr.npz` command. [Details](../CLAUDE.md#hdr-ic-lora-pipeline).

### `a2v`

- **Produces:** video driven by an existing audio track, optionally anchored on an image. Dev model + CFG, two stages.
- **Packs:** 2.3 and 2.5. **Tier:** Beta.
- **Required:** `--prompt`, `--output`, `--frame-rate`, `--audio`.
- **Own flags:** `--frames` (97), `--audio-start` (0 s), `--stage1-steps` (30), `--stage2-steps` (3), `--cfg-scale` (3.0), `--negative-prompt`, `--stg-scale` (1.0), `--image`, `--enable-teacache` / `--teacache-thresh` (LTX-2.3 packs; about 1.30× at 30 steps).
- **Example:** `ltx-2-mlx a2v -p "a singer performing" --audio music.wav -i photo.jpg -f 97 --frame-rate 24 -o out.mp4`
- **Notes:** sync quality depends on how well the prompt matches the audio. Audio CFG runs at 7.0.

### `a2v --distilled`

- **Produces:** video driven by an existing audio track, optionally anchored on images, on the distilled two-stage path of `generate --distilled`: half-res pass, 2× latent upsample, 3-step refine. The input audio's latent is frozen in both stages, and the output carries the input waveform, cut to the clip.
- **Packs:** 2.3 and 2.5, validated on 2.5 q8 only (on 2.5 both stages use the ancestral sampler, as in `generate --distilled`). **Tier:** Experimental.
- **Required:** `--prompt`, `--output`, `--frame-rate`, `--audio`.
- **Own flags:** `--frames` (97), `--audio-start` (0 s), `--stage1-steps` (8), `--stage2-steps` (3), `--image`. No CFG, so `--cfg-scale`, `--stg-scale`, `--negative-prompt` and `--enable-teacache` are rejected before anything loads.
- **Example:** `ltx-2-mlx a2v --distilled -p "a woman talks to the camera" --audio vocals.wav -i photo.jpg -H 1280 -W 704 -f 121 --frame-rate 24 -o out.mp4`
- **Notes:** mirrors Lightricks' `LTX-2.5_A2V_Two_Stage_Distilled` ComfyUI workflow. With music or loud ambience, pass the isolated vocals (Lightricks' LTX-2.0 A2V template separates them before encoding); on a front-facing test clip, a voice over street ambience synced about as well as the vocals alone (SyncNet LSE-C 4.52 vs 4.57). Costs a `generate --distilled` render: 512 × 768, 49 frames, takes 109 s on an M1 Max (2.5 q8) against 631 s for `a2v`.

### `retake`

- **Produces:** the source video with one latent-frame range regenerated. Dev model + CFG, single stage.
- **Packs:** 2.3 and 2.5. **Tier:** Beta.
- **Required:** `--prompt`, `--output`, `--video`, `--start`, `--end` (latent frame indices, end exclusive).
- **Own flags:** `--steps` (30), `--cfg-scale` (3.0), `--negative-prompt`, `--stg-scale` (1.0), `--no-regen-audio`. Resolution and frame count come from the source clip, so `--height`, `--width`, `--frames` and `--frame-rate` are absent.
- **Example:** `ltx-2-mlx retake -p "a different action" -v source.mp4 --start 2 --end 5 --low-ram -o retake.mp4`
- **Cost:** follows the **total** clip length, not the size of the regenerated window. Preserved frames are still computed and attended over on every pass.

### `retake --distilled`

- **Produces:** the same as `retake`, with the distilled transformer: the fixed 8-step distilled sigma table, one forward per step, no CFG. This is upstream `RetakePipeline`'s default mode (`distilled=True`: `DISTILLED_SIGMAS`, `SimpleDenoiser`, deterministic Euler).
- **Packs:** 2.3 and 2.5, validated on 2.5 q8 only (deterministic Euler on both, as upstream's retake; the ancestral sampler of `generate --distilled` on 2.5 is not used). **Tier:** Experimental.
- **Required:** `--prompt`, `--output`, `--video`, `--start`, `--end`.
- **Own flags:** `--steps` (8; fewer keeps σ=1.0 and the last N sigmas of the table), `--no-regen-audio`. No CFG, so `--cfg-scale`, `--stg-scale` and `--negative-prompt` are rejected before anything loads.
- **Example:** `ltx-2-mlx retake --distilled -p "a different action" -v source.mp4 --start 2 --end 5 -o retake.mp4`
- **Notes:** costs 8 forwards over the whole clip instead of `retake`'s 30 steps × 4 passes. On an M1 Max (2.5 q8), a 512 × 768, 49-frame retake takes 148 s (120 s with `LTX2_COMPUTE_DTYPE=float16`) against 1793 s for `retake`; 704 × 1280 × 121 takes 695 s in float16 (716 s with `--low-ram`). Tokens outside the window stay the source's latents, as with `retake`. Like `retake`, it re-rolls a segment and is not a semantic editor: on a 49-frame clip of a woman talking, a 1 s window with "laughs and waves" kept her pose (dev and distilled alike); over latent frames 1–5, the same prompt gave a short two-hand gesture and no wave, and "covers her mouth with both hands and bursts out laughing" brought both hands up to her mouth for a few frames. `LTX2_SOL_TAU` does not apply: retake is one stage and has no stage 2.

### `extend`

- **Produces:** the source video with latent frames appended before or after it. Same pipeline class as `retake`.
- **Packs:** 2.3 and 2.5. **Tier:** Beta.
- **Required:** `--prompt`, `--output`, `--video`, `--extend-frames`.
- **Own flags:** `--direction` (`after`), `--steps` (30), `--cfg-scale` (3.0), `--negative-prompt`, `--stg-scale` (1.0).
- **Example:** `ltx-2-mlx extend -p "continue the scene" -v source.mp4 --extend-frames 4 --low-ram -o extended.mp4`
- **Cost:** same rule as `retake` — the whole clip is denoised each step.

### `extend --distilled`

- **Produces:** the source video with `8 × N` frames appended after it (`--extend-frames N`, latent frames). The source's last 25 frames of video and audio latent are pinned at the start of a new window of `4 + N` latent frames, which is generated like `generate --distilled`: half resolution, 2× latent upsample, 3-step refine, no CFG (ancestral sampler on 2.5). This is how upstream continues a video chunk by chunk (`DistilledPipeline` chunks, `next_video_carry_frames=25`). The new frames are appended to the source's latent and the whole clip is decoded once.
- **Packs:** 2.3 and 2.5, validated on 2.5 q8 only. **Tier:** Experimental.
- **Required:** `--prompt`, `--output`, `--video`, `--extend-frames`. The source size must be a multiple of 64 on both sides (the two-stage grid); the output keeps it.
- **Own flags:** `--image PATH [FRAME STRENGTH [CRF]]` (repeatable, as on `generate --distilled`): an anchor for the new frames, as upstream passes images to each chunk. FRAME counts the appended frames only: `0` is the first new frame, `last` or `-1` the last one, so the valid range is `0 … 8 × N − 1` whatever the source length. Every anchor is a guide (it sits after the 25 carried frames), applied in both stages and re-encoded at full resolution for stage 2. Otherwise `extend`'s flags; `--direction` must stay `after`, and `--steps`, `--cfg-scale`, `--stg-scale` and `--negative-prompt` are rejected before anything loads. `LTX2_SOL_TAU` applies to its stage 2.
- **Example:** `ltx-2-mlx extend --distilled -p "she nods and tucks her hair behind her ear" -v clip.mp4 --extend-frames 12 -o longer.mp4`
- **Cost:** denoising follows the new window (`4 + N` latent frames), not the source length; encoding the source and decoding the result still grow with it. On an M1 Max (2.5 q8, `LTX2_COMPUTE_DTYPE=float16`), adding 4 s to a 10 s 704 × 1280 clip took 589 s, and to a 5 s clip 502 s; `extend` on the 5 s clip would denoise for about 5 h 50 min (176 s per forward × 120). At 512 × 768 × 49 + 16 frames, default environment: 103 s against 2398 s for `extend`.
- **Notes:** the prompt describes the new window; the model sees only the last 25 source frames, so a face that is turned away or hidden there can come back as a different face, and details (earrings, hair colour, framing) drift over a few seconds. An anchor brings the face back: a front-facing frame of the source at the last new frame, strength 0.7 (Lightricks' first/last-frame value), took the new part's ArcFace from 0.42 to 0.55 on the 5 s case and from 0.61 to 0.67 on the 10 s one, with one larger frame change in the last third of a second as the clip settles on the anchor. On the 10 s clip the frame change at the join is 6.3 (mean absolute luma difference), against the clip's 95th percentile of 6.8. At 704 × 1280 on a 48 GB Mac, use `--low-ram` (31.7 GB peak footprint against 56.6 GB resident, same output bytes).

### `lipdub`

- **Produces:** a reference clip re-synced so the lips follow its own audio track.
- **Packs:** 2.3 only. **Tier:** Experimental, on a pre-1.0 LipDub IC-LoRA.
- **Required:** `--prompt`, `--output`, `--reference-video`, `--lora PATH STRENGTH` (exactly one).
- **Own flags:** `--reference-strength` (1.0), `--stage1-steps`, `--stage2-steps`. The frame count and frame rate come from the reference video, so `--frames` and `--frame-rate` are absent.
- **Example:** `ltx-2-mlx lipdub -p "a person speaking" --reference-video clip.mp4 --lora <lipdub-lora> 1.0 -o out.mp4`
- **Notes:** the output audio is a VAE and vocoder reconstruction, audibly degraded on rich music. Remux the original audio when fidelity matters. Stage 2 uses stage 1's generated audio as its audio reference (upstream Dub-It), and the stage-1 reference is the source audio sliced or zero-padded to the clip window. `--low-ram` is accepted and routed through the `ic-lora` streaming path (the LipDub LoRA is attached as a `BlockLoraSource`), but it has not been validated end to end on this pipeline.

### Utilities

These subcommands do not generate video and have no column in the matrix below.

- **`enhance`** rewrites a prompt with Gemma and prints it. Takes `--prompt`, `--gemma`, `--seed`, and `--mode`. Gemma 3 only, so it raises on 2.5 packs, which ship Gemma 4.
- **`info`** lists a pack's safetensors files and a naive RAM estimate (total weight size × 1.3). The estimate ignores `--low-ram` and which transformer a mode loads, so it heavily overstates 2.5 packs, which carry dev + distilled + Gemma 4. Takes `--model`.
- **`train`** trains a LoRA or a full model from a YAML config file. Takes a config path and `--low-ram`.
- **`preprocess`** encodes raw videos into latents and conditions for training. Takes a video directory, a caption directory and extension, an output directory, `--model`, `--gemma`, `--height`, `--width`, `--frame-rate`, a maximum frame count, and an audio switch.
- **`slice`** cuts long source videos into training clips. Takes an output directory plus clip-selection options (timecodes, interval, sampling, minimum length, maximum clip count, head and tail trims, resolution, fit mode, frame rate, encoder quality, caption template).

Run `ltx-2-mlx <subcommand> --help` for the exact spelling of these options.

## Common flags

### Inputs and outputs

| Flag | Default | Effect | Applies to |
|---|---|---|---|
| `--prompt`, `-p` | required | Text prompt. | all except `hdr-ic-lora` |
| `--output`, `-o` | required | Output video path (`.mp4`). | all (on `hdr-ic-lora` both are aliases of `--output-path`) |
| `--model`, `-m` | `dgrauet/ltx-2.3-mlx-q8` (`hdr-ic-lora`: `dgrauet/ltx-2.5-mlx-q8`) | Weights, as a HuggingFace repo id or a local pack directory. 2.5 support is auto-detected from the pack. | all |
| `--gemma` | `mlx-community/gemma-3-12b-it-4bit` | Text encoder for 2.3 packs. 2.5 packs carry their own Gemma 4 tower and ignore it. | all except `hdr-ic-lora` |
| `--seed`, `-s` | -1 (random) | Random seed. Pass a fixed value for reproducible runs. | all |
| `--quiet`, `-q` | off | Suppress the progress output described below. | all |
| `--height`, `-H` | 480 | Output height in pixels. Non-multiples of 64 round down on two-stage paths. | all except `retake` / `extend` / `hdr-ic-lora` |
| `--width`, `-W` | 704 | Output width in pixels. Same rounding rule. | all except `retake` / `extend` / `hdr-ic-lora` |
| `--frame-rate` | required | Output frame rate. LTX-2.3 was trained at 24; values far from that drift out of distribution. | all except `retake` / `extend` / `lipdub` |
| `--frames`, `-f` | 97, or auto on `generate` with a 2.5 pack | Frame count, on the 8k+1 grid; an off-grid value is floored to it with a warning (e.g. 87 → 81). On `generate` with a 2.3 pack, omitting it fails immediately. | all except `retake` / `extend` / `lipdub` / `hdr-ic-lora` |
| `--auto-duration MIN:MAX` | 1:20 | Clamp, in seconds, for the duration predicted by the 2.5 DurationHead. Ignored with a warning when `--frames` is given. [Details](../CLAUDE.md#auto-duration-durationhead--f-optional-on-25). | `generate` on 2.5 packs |
| `--image`, `-i` | — | Reference image: `PATH [FRAME_IDX STRENGTH [CRF]]`. Repeatable, so you can anchor several pixel frames. Frame 0 replaces the first latent frame; later indices act as soft keyframes. `FRAME_IDX` may be `last` or negative (counted from the end), which keeps an end anchor on the final frame under `--auto-duration`. On 2.5 packs, a frame-0 anchor now also carries the learned keyframe marker through, which shifts 2.5 I2V output slightly (2.3 unaffected). `CRF` (H.264 re-compression of the image, `0` = none) defaults to the model generation's value, as upstream: 33 on 2.3 packs, 18 on 2.5 packs (LTX-2.4 and later). [Details](../CLAUDE.md#multi-anchor-i2v---image-repeatable). | `generate` modes, `ic-lora`, `a2v`, `extend --distilled` (FRAME counts the appended frames) |
| `--num-generated-keyframes N` | 0 | Add N generated keyframe slots at evenly spaced interior frames in stage 1, which relaxes the temporal compression where motion is fast. Each slot costs a latent frame of tokens. Refused up front on 2.3 packs. [Details](../CLAUDE.md#generated-keyframe-slots---num-generated-keyframes-n-25-packs) | `generate` modes on 2.5 packs |
| `--no-audio` | off | Skip the audio decode and mux. Video is unchanged; the DiT still produces audio latents jointly. | `generate` modes |
| `--video-decoder {conv,diffusion}` | `conv` | Video VAE decoder. `diffusion` is sharper and slower and needs a 2.5 pack. [Details](../CLAUDE.md#diffusion-video-decoder---video-decoder-diffusion-25-packs-experimental). | `generate` modes |
| `--diffvae-tile FRAMES HEIGHT WIDTH` | auto | Diffusion-decoder tile size, in pixel frames and pixels (multiples of 2 and 8; `0` leaves an axis untiled, `0 0 0` forces a single tile). | with `--video-decoder diffusion` |
| `--lora PATH STRENGTH` | — | Extra LoRA weights, repeatable. A local `.safetensors` file or a HuggingFace repo id. Required on the IC-LoRA family. | `generate` modes, `ic-lora`, `lipdub` |
| `--enhance-prompt` | off | Rewrite the prompt with Gemma before generating. Gemma 3 only, so it raises on 2.5 packs. | `generate` modes |
| `--negative-prompt` | upstream `DEFAULT_NEGATIVE_PROMPT` | Negative prompt for CFG: what the video should avoid. One global prompt, even with `--segment`. `""` is encoded as an empty prompt, not replaced by the default. Rejected by `generate --distilled` / `--dfr` (unless `--nag` is set; there it has no default) and by `a2v --distilled`, `retake --distilled` and `extend --distilled` (no CFG, no `--nag`). | CFG modes: `generate --one-stage` / `--two-stage` / `--two-stages-hq`, `keyframe`, `a2v`, `retake`, `extend`; `--distilled` / `--dfr` with `--nag` |
| `--dev-transformer` | `transformer-dev.safetensors` on `generate`, unset elsewhere | Filename of the dev (non-distilled) transformer inside the pack. On `keyframe` and `ic-lora` there is no default, and on `ic-lora` passing it switches dev mode on. | `generate` modes, `keyframe`, `ic-lora` |
| `--distilled-lora` | resolved from the pack; `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` on `ic-lora` | Filename of the distilled LoRA used by the refine stage. | `generate` modes, `keyframe`, `ic-lora` |
| `--distilled-lora-strength` | 1.0 (0.5 on `ic-lora` dev mode) | Strength of that LoRA. Any value other than 1.0 forces bind-time fusion under `--low-ram`. | `generate` modes, `ic-lora` |

### Negative prompt without CFG (NAG)

Experimental. Normalized Attention Guidance applies `--negative-prompt` on the distilled paths, which have no
unconditional pass: inside every text cross-attention (video `attn2`, audio `audio_attn2`) the queries also attend to the
negative prompt, and the two outputs are extrapolated, L1-norm clipped and blended. One extra attention per
cross-attention call instead of a second model pass. The settings and their defaults are those of kijai's `LTX2_NAG`
ComfyUI node. Measured on the 2.5 q8 pack (M4 Pro, `--distilled` 768×512×97): +14 % per stage-1 step and +7 % per
stage-2 step, peak memory unchanged; a negative naming a wide-open mouth made the mouth open less in 4 of 4 runs with the
speech unchanged, while a negative naming distorted hands did not fix motion-blurred fingers.
[Details](../CLAUDE.md#negative-prompt-without-cfg-nag---nag).

| Flag | Default | Effect | Applies to |
|---|---|---|---|
| `--nag` | off | Turn NAG on. Needs `--negative-prompt`. | `generate --distilled`, `generate --dfr` |
| `--nag-scale` | 11.0 | Extrapolation away from the negative prompt, `>= 1` (`1` = no effect). | with `--nag` |
| `--nag-alpha` | 0.25 | Blend of the guided output with the plain one, in [0, 1]. | with `--nag` |
| `--nag-tau` | 2.5 | Clip on how far the guided output's L1 norm may grow over the plain one's, per token. | with `--nag` |
| `--nag-video-only` | off | Leave the audio cross-attention unguided. | with `--nag` |

### Memory

| Flag or variable | Default | Effect | Applies to |
|---|---|---|---|
| `--low-ram` | off | Stream transformer blocks from mmap'd safetensors. Cuts transformer peak Metal memory by about 75%, at roughly 5% more time per step. Targets 16 GB Macs at q8 and 32 GB Macs at bf16. [Details](../CLAUDE.md#block-streaming---low-ram). | every generating pipeline |
| `--tile-frames N` | 1 | Split the video tokens into N temporal tiles, denoised independently and blended back. Caps the quadratic attention activation. [Details](../CLAUDE.md#modality-tiling---tile-frames-n---tile-spatial-m). | `generate` modes |
| `--tile-spatial M` | 1 | Split into M × M spatial tiles. Total tiles are `tile-frames × M²`. | same as above |
| `--tile-overlap K` | 2 | Token-grid overlap between adjacent tiles. More overlap means a smoother blend and more redundant compute. | when tiling is active |
| `LTX2_VAE_DECODE_BUDGET_GB` | half of unified memory | Peak-memory budget for the VAE decode. Drives the conv decoder's automatic tiling and the diffusion decoder's tile sizing. Raise it on 64–256 GB machines for fewer, larger tiles. [Details](../CLAUDE.md#conv-vae-decode-budget-and-auto-tiling). | video decode |
| `LTX2_VAE_DECODE_KEEP_CACHE` | unset | Keep the MLX allocator cache during the decode. It is disabled by default for the decode only; pixels are identical either way and the process footprint is tens of GB lower on long clips. | video decode |

Tiling costs wall-clock time and only pays for itself when attention activations
would otherwise not fit. On a 32 GB Mac at typical token counts, prefer `--low-ram` alone.

### Speed

| Flag or variable | Default | Effect | Applies to |
|---|---|---|---|
| `--enable-teacache` | off | Timestep-aware residual caching in stage 1. About 1.46× on the Euler sampler and 1.78× on res_2s. The same seed gives a different render, often a different composition, so it cannot preview a plain render. [Details](../CLAUDE.md#teacache-opt-in-stage-1-acceleration). LTX-2.3 packs only (refused on 2.5). | `generate --two-stage`, `generate --two-stages-hq`, `a2v` (not `--distilled`) |
| `--teacache-thresh F` | 0.5 Euler, 1.0 res_2s | How aggressively steps are skipped. Higher is faster and lossier. Ignored without `--enable-teacache`. | with `--enable-teacache` |
| `--stage2-steps N` | full table (3) | Fewer stage-2 refine steps. Takes the **last** N sigmas of the stage-2 table, so the refine re-noises less and always finishes at σ=0 (`1` → `[0.421875, 0.0]`, `2` → `[0.725, 0.421875, 0.0]`). A one-step refine gives a usable preview of the shot, not a softer copy of it. | every two-stage pipeline except `extend --distilled` (full 8 + 3 tables) |
| `--stage1-steps N` | full table (8) | On a pipeline whose stage 1 uses the fixed distilled table, keeps σ=1.0 and then the **last** N sigmas of the table (`3` → `[1.0, 0.725, 0.421875, 0.0]`). On `--two-stage`, `--two-stages-hq`, `a2v` (without `--distilled`) and dev-mode `keyframe`, stage 1 uses the dynamic schedule instead and N is simply its step count. | every two-stage pipeline except `extend --distilled` (full 8 + 3 tables) |
| `LTX2_GEMMA_EVAL_EVERY` | 1 | Per-layer flush cadence in the Gemma forward, which keeps each Metal command buffer under the macOS GPU watchdog deadline. Set to `0` only if you have never seen a watchdog crash. [Details](../CLAUDE.md#metal-watchdog-mitigation). | all pipelines |
| `LTX2_DIT_EVAL_EVERY` | 8 | Same guard for the DiT block loop: flush every N of the 48 blocks. `0` disables it. | all pipelines |
| `LTX2_COMPUTE_DTYPE` | unset | Dtype for the inside of the DiT's attention and feed-forward modules. Unset, the blocks run in float32 (the F32 AdaLN tables promote them). `float16` keeps the residual stream and AdaLN modulation in float32 and runs the projections and attention in float16: on an M1 Max with the 2.5 q8 pack, a 576×1024×241 `--distilled` I2V render went from 701 s to 563 s. `bfloat16` is the precision upstream PyTorch uses. Off by default; a step whose output is not finite is recomputed without it. [Details](../CLAUDE.md#dit-compute-dtype-ltx2_compute_dtype). | all pipelines |
| `LTX2_LORA_MODE` | `fused` | `unfused` applies LoRAs at run time (`base(x) + (x @ A^T) @ B^T`) instead of fusing them into the quantized weights, where re-quantizing loses most of a small LoRA's update. Used for the IC-LoRAs of `ic-lora` / `hdr-ic-lora` / `lipdub`, the `--dfr` detailing LoRA (detached in place, no DiT reload) and `generate --lora`; the distilled LoRA is always fused. Under `--low-ram` the same LoRAs become run-time adapters on the streamed block (each block's factors are bound with its weights). [Details](../CLAUDE.md#lora-mode-ltx2_lora_mode). | pipelines with LoRAs, resident or `--low-ram` |
| `LTX2_SOL_TAU` | unset | Block-sparse video self-attention in the distilled stage 2 (NVIDIA's Sol-Attn routing): one threshold per stage-2 step, e.g. `1.0,1.25,1.5` (NVIDIA's LTX-2.5 values; the last one repeats). Key blocks below the threshold enter the softmax as one centroid each instead of 64 exact keys; block 0 and calls with a mask stay dense. On an M1 Max with the 2.5 q8 pack (float16 compute), stage-2 steps were 21 % faster at 17,856 tokens and 36 % at 32,640, whole renders 12 % and 22 %; the gain grows with resolution and length. The output changes slightly (the skipped blocks are approximated): about 30–32 dB PSNR against the dense render on I2V clips, same speech and lip sync, identity to the start image a little lower. Unset or `off` is the dense path, unchanged. [Details](../CLAUDE.md#block-sparse-stage-2-attention-ltx2_sol_tau). | `generate --distilled`, `generate --dfr`, `a2v --distilled`, `extend --distilled` (stage 2; not `retake --distilled`, which has no stage 2) |
| `LTX2_GEMMA_MAX_LENGTH` | 1024 | Cap on the padded Gemma sequence length. Lowering it halves the encode time but shifts the text positions away from what the model was trained with (quality risk). Last resort. | all pipelines |
| `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` | unset | An AGX driver knob, not an LTX-2 variable, and never set automatically. It relaxes the watchdog eviction timeout for the process that sets it, working around a macOS 26.x and MLX 0.31.x regression. The UI may stutter during the run, and on some machines running with the display off is the only reliable workaround. | all pipelines |

### Previews and prompt gating

| Flag | Default | Effect | Applies to |
|---|---|---|---|
| `--stepwise-image-output-dir DIR` | off | Experimental. Every N steps, decode a short window of the in-progress prediction and write it to this directory as an animated WebP. One file per step, written once. It keeps the VAE decoder resident, which raises peak memory and works against `--low-ram`. | all generating pipelines |
| `--stepwise-interval N` | 1 | Preview every N denoising steps. The final step is always previewed. | with `--stepwise-image-output-dir` |
| `--stepwise-frames N` | 8 | Latent frames per preview. The VAE upsamples time 8×, so N latent frames give 8N−7 pixel frames, about 2.3 s at the default. Cost is independent of clip length. `1` gives a single still. | with `--stepwise-image-output-dir` |
| `--stepwise-frame I` | middle | Latent frame the preview window is centred on. Negative values count from the end. The middle is the default because frame 0 is the clean conditioning image on I2V runs. | with `--stepwise-image-output-dir` |
| `--segment "TEXT" [LEN]` | — | Prompt Relay: a local prompt gated to a slice of the timeline, repeatable in timeline order. `LEN` is in latent frames. The global `--prompt` still applies everywhere. Validated on 2.3 and 2.5 packs. Not compatible with modality tiling. [Details](../CLAUDE.md#prompt-relay---segment). | `generate` modes |
| `--relay-epsilon` | 1e-3 | Prompt Relay falloff. Smaller is sharper temporal gating. | with `--segment` |
| `--relay-strength` | 1.0 | Prompt Relay penalty multiplier. Higher isolates segments more strictly. | with `--segment` |

## Flags by pipeline

Columns are the cards above. `enhance`, `info`, `train`, `preprocess` and `slice`
are utilities and have no column.

| Flag | distilled | two-stage | hq | one-stage | dfr | keyframe | ic-lora | hdr-ic-lora | a2v | retake | extend | lipdub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--prompt` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `--output` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--model` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--gemma` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `--seed` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--quiet` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--height` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| `--width` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| `--frame-rate` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `--frames` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| `--auto-duration` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--image` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| `--num-generated-keyframes` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--no-audio` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--video-decoder` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--diffvae-tile` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--lora` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `--enhance-prompt` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--negative-prompt` | with `--nag` | ✅ | ✅ | ✅ | with `--nag` | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| `--nag` | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--nag-scale` | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--nag-alpha` | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--nag-tau` | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--nag-video-only` | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--dev-transformer` | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--distilled-lora` | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--distilled-lora-strength` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--detailing-lora` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--spatial-upscalings` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--temporal-upscalings` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--temporal-upsampler-path` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--low-ram` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--tile-frames` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--tile-spatial` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--tile-overlap` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--enable-teacache` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| `--teacache-thresh` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| `--stepwise-image-output-dir` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `--stepwise-interval` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `--stepwise-frames` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `--stepwise-frame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `--segment` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--relay-epsilon` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--relay-strength` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `--input` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--output-path` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--hdr-lora` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--text-embeddings` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--input-colorspace` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--exr-colorspace` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--high-quality` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--no-keyframes` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--keyframe-strength` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |

A ❌ means the flag is either rejected by the parser or accepted and inert for that
mode. The five `generate` modes share one parser: a ❌ flag on one of them is either
rejected up front with an explicit error (CFG, NAG, TeaCache and DFR-only flags) or accepted
and ignored.

`a2v --distilled` takes the `a2v` column except the CFG and TeaCache flags: `--cfg-scale`,
`--stg-scale`, `--negative-prompt` and `--enable-teacache` are rejected up front.

`retake --distilled` takes the `retake` column except the CFG flags: `--cfg-scale`,
`--stg-scale` and `--negative-prompt` are rejected up front.

`extend --distilled` takes the `extend` column except `--steps`, `--cfg-scale`, `--stg-scale`
and `--negative-prompt` (rejected up front); `--direction` must be `after`. `--image` on `extend`
needs `--distilled` (the dev `extend` rejects it up front).

## Progress output (stderr)

All CLI progress goes to **stderr** so stdout stays clean for callers that pipe it:

- `[phase] ...` / `[phase] done in X.Ys` brackets the silent stages (Gemma load, prompt encode, DiT load, decoders, decode). Silenced by `--quiet`.
- Each denoising stage prints `[estimate] <stage>: N steps x P passes over V video + A audio tokens = F forwards` **before** its first step (tqdm cannot say anything until iteration 1 completes, which at tens of seconds per step is exactly when a run looks hung), then `[estimate] <stage>: ~T remaining (S s/forward)` after the first computed step (tagged `first step includes warm-up` — kernel compilation and cache warm-up make it an upper bound) and once more after the second, timed on that step alone (tagged `refined`, the number to trust). Nothing after that. Passes per step come from the guider schedule (`cond` always, `uncond` under CFG, `ptb` under STG, `mod` under modality isolation; res_2s doubles them). Multi-stage pipelines print one pair per stage; there is no cross-stage total.
- `retake` / `extend`: the estimate carries `cost follows total clip length, not the regenerated window` — preserved frames are still computed and attended over on every pass, so retaking 1 latent frame of a 10 s clip costs the same as retaking all of it.

## Multishot on LTX-2.5

"Native multishot" is a capability of the 2.5 model, not a pipeline feature: write the shots in order in a single prompt, following [Lightricks' prompting guide](https://docs.ltx.video/open-source-model/usage-guides/prompting-guide). Nothing to enable on this runtime. If the model does not cut where you want, `--segment` (Prompt Relay) gates local prompts to time ranges on top of the global prompt.

## Compatibility notes

- `generate --lora` works with `--low-ram`: each LoRA is attached as a `BlockLoraSource` and fused at block bind time (slower per step than an in-place fuse).
- `--low-ram` + custom `--distilled-lora-strength` (≠1.0) on two-stage uses bind-time LoRA fusion (slower per step but supports any strength). At strength=1.0, swaps to pre-fused `transformer-distilled.safetensors`.
- TeaCache calibration is sampler-specific (Euler vs res_2s). Don't reuse coefficients across `--two-stage` and `--two-stages-hq`.
- Modality tiling overhead dominates over memory benefit at default Nv (1650-3168). Use only when targeting 1080p / 8s+ on Mac Studio 64-128 GB; on 32 GB Mac, prefer `--low-ram` alone.
- `generate` requires a mode flag (`--one-stage`, `--two-stage`, `--two-stages-hq`, `--distilled`, or `--dfr` on 2.5 packs, experimental). There is **no implicit default** — each `generate` mode maps 1:1 to an upstream Lightricks/LTX-2 class (local additions such as `a2v --distilled` and `--nag` follow Lightricks / ComfyUI workflows instead).
- `generate --one-stage` vs `generate --two-stage`: same dev model + CFG, but `--one-stage` runs **once at the target resolution** (no upscaler dependency, simpler latents for downstream). `--two-stage` runs at half-res then upscales 2× and refines (typically faster overall and better at large targets). Pick `--one-stage` for native res ≤ 704 × 480 or if you don't trust the upsampler; pick `--two-stage` for everything else.
- `generate --distilled` vs `generate --two-stage`: same half-res + upscale structure, but `--distilled` skips CFG entirely (8 stage 1 steps × 1 forward instead of 30 × 2-4). Fastest mode; quality slightly below the dev+CFG variants.
- `a2v --distilled` vs `a2v`: the same audio conditioning (the input track frozen in both stages, its waveform muxed into the output), but the distilled transformer without CFG: 8 + 3 single forwards instead of 30 guided steps and a LoRA-fused refine. On an M1 Max (2.5 q8, default environment), 512 × 768, 49 frames: 109 s against 631 s.
- `extend --distilled` vs `extend`: the same request (frames appended after the source), but the distilled transformer on a window of the source's last 25 frames plus the new ones, instead of 30 guided steps × 4 passes over the whole extended clip. On an M1 Max (2.5 q8, default environment), 512 × 768 × 49 + 16 frames: 103 s against 2398 s.
- `retake --distilled` vs `retake`: the same window mask and audio handling, but the distilled transformer without CFG: 8 forwards instead of 30 steps × 4 passes over the whole clip. On an M1 Max (2.5 q8, default environment), 512 × 768, 49 frames: 148 s against 1793 s.
- `--video-decoder diffusion` (LTX 2.5 packs only, experimental) reproduces upstream's **default** `chunked_eager` stage-5 mode exactly: the neighborhood-attention stage runs on four width slabs with a halo, so the first/last ~20 px of each row are edge-replicated rather than attending over the full volume — identical to upstream's own default-mode output, not a port shortfall. Decodes above the memory budget are tiled automatically (`--diffvae-tile` to override); conv remains the default decoder.
