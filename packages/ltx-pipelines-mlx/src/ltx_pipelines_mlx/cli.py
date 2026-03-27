"""Command-line interface for ltx-2-mlx.

Usage:
    ltx-2-mlx generate --prompt "a cat walking" --output out.mp4
    ltx-2-mlx generate --prompt "animate this" --image photo.jpg -o anim.mp4
    ltx-2-mlx generate --prompt "a scene" --two-stage -o hires.mp4
    ltx-2-mlx generate --prompt "a scene" --hq --stage1-steps 20 -o hq.mp4
    ltx-2-mlx a2v --prompt "music video" --audio music.wav -o a2v.mp4
    ltx-2-mlx retake --prompt "new scene" --video source.mp4 --start 1 --end 3 -o retake.mp4
    ltx-2-mlx extend --prompt "continue" --video source.mp4 --extend-frames 2 -o extended.mp4
    ltx-2-mlx keyframe --prompt "transition" --start img1.png --end img2.png -o kf.mp4
    ltx-2-mlx ic-lora --prompt "scene" --lora lora.safetensors 1.0 --video-conditioning depth.mp4 1.0 -o out.mp4
    ltx-2-mlx enhance --prompt "a cat walking" --mode t2v
    ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-q8
"""

from __future__ import annotations

import argparse
import sys
import time

DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q8"
DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"


def _add_base_args(parser: argparse.ArgumentParser) -> None:
    """Add base arguments shared by all subcommands (prompt, output, model, seed)."""
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    parser.add_argument("--output", "-o", required=True, help="Output video path (.mp4)")
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL, help=f"Model weights (HF repo or path, default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--gemma", default=DEFAULT_GEMMA, help=f"Gemma model for text encoding (default: {DEFAULT_GEMMA})"
    )
    parser.add_argument("--seed", "-s", type=int, default=-1, help="Random seed (-1 = random)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    """Add generation-specific arguments (dimensions, steps) on top of base args."""
    _add_base_args(parser)
    parser.add_argument("--height", "-H", type=int, default=480, help="Video height (default: 480)")
    parser.add_argument("--width", "-W", type=int, default=704, help="Video width (default: 704)")
    parser.add_argument("--frames", "-f", type=int, default=97, help="Number of frames (default: 97)")


def main() -> None:
    """Entry point for the ltx-2-mlx CLI."""
    parser = argparse.ArgumentParser(
        prog="ltx-2-mlx",
        description="LTX-2.3 video generation on Apple Silicon (MLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  ltx-2-mlx generate --prompt "a sunset" --output sunset.mp4
  ltx-2-mlx generate --prompt "animate" --image photo.jpg -o anim.mp4
  ltx-2-mlx generate --prompt "a scene" --two-stage -o hires.mp4
  ltx-2-mlx a2v --prompt "music video" --audio music.wav -o a2v.mp4
  ltx-2-mlx retake --prompt "new scene" --video source.mp4 --start 1 --end 3 -o out.mp4
  ltx-2-mlx extend --prompt "continue" --video source.mp4 --extend-frames 2 -o out.mp4
  ltx-2-mlx keyframe --prompt "transition" --start img1.png --end img2.png -o out.mp4
  ltx-2-mlx ic-lora --prompt "scene" --lora lora.safetensors 1.0 --video-conditioning depth.mp4 1.0 -o out.mp4
  ltx-2-mlx enhance --prompt "a cat walking" --mode t2v
  ltx-2-mlx info --model dgrauet/ltx-2.3-mlx-q4
""",
    )
    sub = parser.add_subparsers(dest="command")

    # --- generate (T2V / I2V / two-stage / HQ) ---
    gen = sub.add_parser("generate", help="Generate video from text (T2V) or image (I2V)")
    _add_generation_args(gen)
    gen.add_argument("--image", "-i", default=None, help="Reference image for I2V (optional)")
    gen.add_argument("--steps", type=int, default=None, help="Denoising steps for one-stage (default: 8)")
    gen.add_argument(
        "--two-stage",
        action="store_true",
        help="Two-stage pipeline: dev model + CFG at half-res, upscale, distilled LoRA refine (requires q8 model)",
    )
    gen.add_argument("--hq", action="store_true", help="HQ two-stage pipeline (res_2s sampler for stage 1)")
    gen.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 steps (default: 30 standard, 15 HQ)")
    gen.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 steps (default: 3)")
    gen.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (default: 3.0)")
    gen.add_argument("--stg-scale", type=float, default=None, help="STG guidance scale (default: 1.0 standard, 0.0 HQ)")
    gen.add_argument(
        "--dev-transformer",
        default="transformer-dev.safetensors",
        help="Dev transformer filename (default: transformer-dev.safetensors)",
    )
    gen.add_argument(
        "--distilled-lora",
        default="ltx-2.3-22b-distilled-lora-384.safetensors",
        help="Distilled LoRA filename for stage 2 (default: ltx-2.3-22b-distilled-lora-384.safetensors)",
    )
    gen.add_argument("--lora-strength", type=float, default=1.0, help="Distilled LoRA strength (default: 1.0)")
    gen.add_argument("--enhance-prompt", action="store_true", help="Enhance prompt using Gemma before generation")

    # --- a2v (Audio-to-Video) ---
    a2v = sub.add_parser("a2v", help="Generate video from audio + text prompt")
    _add_generation_args(a2v)
    a2v.add_argument("--audio", "-a", required=True, help="Input audio file (WAV/MP3/etc.)")
    a2v.add_argument("--fps", type=float, default=24.0, help="Frame rate (default: 24)")
    a2v.add_argument("--audio-start", type=float, default=0.0, help="Audio start time in seconds (default: 0)")
    a2v.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 steps (default: 30 standard, 15 HQ)")
    a2v.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 steps (default: 3)")
    a2v.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (default: 3.0)")
    a2v.add_argument("--stg-scale", type=float, default=None, help="STG guidance scale (default: 1.0 standard, 0.0 HQ)")
    a2v.add_argument("--image", "-i", default=None, help="Reference image for I2V conditioning (optional)")
    a2v.add_argument("--hq", action="store_true", help="HQ mode: use res_2s sampler for stage 1")

    # --- retake ---
    ret = sub.add_parser("retake", help="Regenerate a time segment of an existing video")
    _add_base_args(ret)
    ret.add_argument("--video", "-v", required=True, help="Source video file")
    ret.add_argument("--start", type=int, required=True, help="Start latent frame index (inclusive)")
    ret.add_argument("--end", type=int, required=True, help="End latent frame index (exclusive)")
    ret.add_argument("--steps", type=int, default=None, help="Denoising steps (default: 30)")
    ret.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (default: 3.0)")
    ret.add_argument("--stg-scale", type=float, default=None, help="STG guidance scale (default: 0.0)")
    ret.add_argument("--no-regen-audio", action="store_true", help="Preserve original audio (don't regenerate)")

    # --- extend ---
    ext = sub.add_parser("extend", help="Add frames before or after an existing video")
    _add_base_args(ext)
    ext.add_argument("--video", "-v", required=True, help="Source video file")
    ext.add_argument("--extend-frames", type=int, required=True, help="Number of latent frames to add")
    ext.add_argument("--direction", choices=["before", "after"], default="after", help="Direction (default: after)")
    ext.add_argument("--steps", type=int, default=None, help="Denoising steps (default: 30)")
    ext.add_argument("--cfg-scale", type=float, default=None, help="CFG guidance scale (default: 3.0)")
    ext.add_argument("--stg-scale", type=float, default=None, help="STG guidance scale (default: 0.0)")

    # --- keyframe ---
    kf = sub.add_parser("keyframe", help="Interpolate between keyframe images")
    _add_generation_args(kf)
    kf.add_argument("--start", required=True, help="Start keyframe image path")
    kf.add_argument("--end", required=True, help="End keyframe image path")
    kf.add_argument("--fps", type=float, default=24.0, help="Frame rate (default: 24)")
    kf.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 denoising steps")
    kf.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 denoising steps")
    kf.add_argument("--cfg-scale", type=float, default=None, help="Override CFG scale (default: 3.0 video, 7.0 audio)")
    kf.add_argument("--stg-scale", type=float, default=None, help="Override STG scale (default: 1.0)")
    kf.add_argument(
        "--dev-transformer",
        default=None,
        help="Dev (non-distilled) transformer filename for higher quality stage 1 (e.g. transformer-dev.safetensors)",
    )
    kf.add_argument(
        "--distilled-lora",
        default=None,
        help="Distilled LoRA filename for stage 2 refinement (e.g. ltx-2.3-22b-distilled-lora-384.safetensors)",
    )
    kf.add_argument("--lora-strength", type=float, default=1.0, help="Distilled LoRA strength (default: 1.0)")

    # --- ic-lora ---
    ic = sub.add_parser("ic-lora", help="Generate video with IC-LoRA control conditioning")
    _add_generation_args(ic)
    ic.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help=(
            "IC-LoRA weights and strength (repeatable). PATH can be a local .safetensors file "
            "or a HuggingFace repo ID (e.g. Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control). "
            "Example: --lora Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control 1.0"
        ),
    )
    ic.add_argument(
        "--video-conditioning",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
        help="Reference control video and strength (repeatable). Example: --video-conditioning depth.mp4 1.0",
    )
    ic.add_argument("--image", "-i", default=None, help="Optional reference image for I2V conditioning")
    ic.add_argument("--stage1-steps", type=int, default=None, help="Stage 1 denoising steps")
    ic.add_argument("--stage2-steps", type=int, default=None, help="Stage 2 denoising steps")
    ic.add_argument(
        "--conditioning-strength",
        type=float,
        default=1.0,
        help="IC-LoRA conditioning attention strength 0.0-1.0 (default: 1.0)",
    )
    ic.add_argument("--skip-stage-2", action="store_true", help="Skip stage 2 upsampling (half resolution output)")

    # --- enhance ---
    enh = sub.add_parser("enhance", help="Enhance a prompt using Gemma (no video generation)")
    enh.add_argument("--prompt", "-p", required=True, help="Prompt to enhance")
    enh.add_argument("--mode", choices=["t2v", "i2v"], default="t2v", help="Prompt mode (default: t2v)")
    enh.add_argument("--gemma", default=DEFAULT_GEMMA, help=f"Gemma model (default: {DEFAULT_GEMMA})")
    enh.add_argument("--seed", "-s", type=int, default=10, help="Random seed (default: 10)")

    # --- info ---
    info = sub.add_parser("info", help="Show model info and memory estimate")
    info.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model weights (HF repo or path)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Resolve seed=-1 to a random value
    if hasattr(args, "seed") and args.seed < 0:
        import random

        args.seed = random.randint(0, 2**31 - 1)

    commands = {
        "generate": _cmd_generate,
        "a2v": _cmd_a2v,
        "retake": _cmd_retake,
        "extend": _cmd_extend,
        "keyframe": _cmd_keyframe,
        "ic-lora": _cmd_ic_lora,
        "enhance": _cmd_enhance,
        "info": _cmd_info,
    }
    commands[args.command](args)


# =============================================================================
# Generate (T2V / I2V / Two-stage / HQ)
# =============================================================================


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate a video from a text prompt (and optionally a reference image)."""
    t0 = time.time()

    prompt = _maybe_enhance_prompt(args)

    if args.hq or args.two_stage:
        if args.hq:
            from ltx_pipelines_mlx.ti2vid_two_stages_hq import TwoStageHQPipeline as PipeClass

            mode_name = "HQ Two-Stage (res_2s + CFG + distilled LoRA)"
        else:
            from ltx_pipelines_mlx.ti2vid_two_stages import TwoStagePipeline as PipeClass

            mode_name = "Two-Stage (Euler + CFG + distilled LoRA)"

        if not args.quiet:
            print(f"Mode: {mode_name}")
            print(f"  Model: {args.model}")

        pipe = PipeClass(
            model_dir=args.model,
            gemma_model_id=args.gemma,
            low_memory=True,
            dev_transformer=args.dev_transformer,
            distilled_lora=args.distilled_lora,
            distilled_lora_strength=args.lora_strength,
        )
        # Only pass non-None overrides; pipeline defaults take over otherwise
        kwargs: dict = dict(
            prompt=prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            image=args.image,
        )
        if args.stage1_steps is not None:
            kwargs["stage1_steps"] = args.stage1_steps
        if args.stage2_steps is not None:
            kwargs["stage2_steps"] = args.stage2_steps
        if args.cfg_scale is not None:
            kwargs["cfg_scale"] = args.cfg_scale
        if args.stg_scale is not None:
            kwargs["stg_scale"] = args.stg_scale
        pipe.generate_and_save(**kwargs)

    elif args.image:
        from ltx_pipelines_mlx.ti2vid_one_stage import ImageToVideoPipeline

        if not args.quiet:
            print("Mode: Image-to-Video")
            print(f"Image: {args.image}")

        pipe = ImageToVideoPipeline(model_dir=args.model, gemma_model_id=args.gemma)
        pipe.generate_and_save(
            prompt=prompt,
            output_path=args.output,
            image=args.image,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            num_steps=args.steps,
        )

    else:
        from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline

        if not args.quiet:
            print("Mode: Text-to-Video")

        pipe = TextToVideoPipeline(model_dir=args.model, gemma_model_id=args.gemma)
        pipe.generate_and_save(
            prompt=prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            seed=args.seed,
            num_steps=args.steps,
        )

    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Audio-to-Video
# =============================================================================


def _cmd_a2v(args: argparse.Namespace) -> None:
    """Generate video from audio + text prompt."""
    t0 = time.time()

    if args.hq:
        from ltx_pipelines_mlx.a2vid_two_stage_hq import AudioToVideoHQPipeline as PipeClass

        mode_name = "Audio-to-Video HQ (res_2s + CFG)"
    else:
        from ltx_pipelines_mlx.a2vid_two_stage import AudioToVideoPipeline as PipeClass

        mode_name = "Audio-to-Video (Euler + CFG)"

    if not args.quiet:
        print(f"Mode: {mode_name}")
        print(f"Audio: {args.audio}")
        print(f"  Model: {args.model}")

    pipe = PipeClass(model_dir=args.model, gemma_model_id=args.gemma)
    kwargs: dict = dict(
        prompt=args.prompt,
        output_path=args.output,
        audio_path=args.audio,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        fps=args.fps,
        seed=args.seed,
        image=args.image,
        audio_start_time=args.audio_start,
    )
    if args.stage1_steps is not None:
        kwargs["stage1_steps"] = args.stage1_steps
    if args.stage2_steps is not None:
        kwargs["stage2_steps"] = args.stage2_steps
    if args.cfg_scale is not None:
        kwargs["cfg_scale"] = args.cfg_scale
    if args.stg_scale is not None:
        kwargs["stg_scale"] = args.stg_scale
    pipe.generate_and_save(**kwargs)

    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Retake
# =============================================================================


def _cmd_retake(args: argparse.Namespace) -> None:
    """Regenerate a time segment of an existing video."""
    t0 = time.time()

    from ltx_pipelines_mlx.retake import RetakePipeline

    if not args.quiet:
        print("Mode: Retake")
        print(f"Video: {args.video}, frames {args.start}-{args.end}")

    pipe = RetakePipeline(model_dir=args.model, gemma_model_id=args.gemma)
    kwargs: dict = dict(
        prompt=args.prompt,
        video_path=args.video,
        start_frame=args.start,
        end_frame=args.end,
        seed=args.seed,
        regenerate_audio=not args.no_regen_audio,
    )
    if args.steps is not None:
        kwargs["num_steps"] = args.steps
    if args.cfg_scale is not None:
        kwargs["cfg_scale"] = args.cfg_scale
    if args.stg_scale is not None:
        kwargs["stg_scale"] = args.stg_scale
    video_latent, audio_latent = pipe.retake_from_video(**kwargs)

    _decode_and_save(pipe, video_latent, audio_latent, args)
    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Extend
# =============================================================================


def _cmd_extend(args: argparse.Namespace) -> None:
    """Add frames before or after an existing video."""
    t0 = time.time()

    from ltx_pipelines_mlx.extend import ExtendPipeline

    if not args.quiet:
        print(f"Mode: Extend ({args.direction})")
        print(f"Video: {args.video}, +{args.extend_frames} latent frames")

    pipe = ExtendPipeline(model_dir=args.model, gemma_model_id=args.gemma)
    kwargs: dict = dict(
        prompt=args.prompt,
        video_path=args.video,
        extend_frames=args.extend_frames,
        direction=args.direction,
        seed=args.seed,
    )
    if args.steps is not None:
        kwargs["num_steps"] = args.steps
    if args.cfg_scale is not None:
        kwargs["cfg_scale"] = args.cfg_scale
    if args.stg_scale is not None:
        kwargs["stg_scale"] = args.stg_scale
    video_latent, audio_latent = pipe.extend_from_video(**kwargs)

    _decode_and_save(pipe, video_latent, audio_latent, args)
    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Keyframe interpolation
# =============================================================================


def _cmd_keyframe(args: argparse.Namespace) -> None:
    """Interpolate between two keyframe images."""
    t0 = time.time()

    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    if not args.quiet:
        print("Mode: Keyframe Interpolation (two-stage)")
        print(f"Start: {args.start}, End: {args.end}")

    last_pixel_frame = args.frames - 1

    pipe = KeyframeInterpolationPipeline(
        model_dir=args.model,
        gemma_model_id=args.gemma,
        dev_transformer=args.dev_transformer,
        distilled_lora=args.distilled_lora,
        distilled_lora_strength=args.lora_strength,
    )
    # Build guider params (defaults match reference LTX_2_3_PARAMS)
    from ltx_core_mlx.components.guiders import MultiModalGuiderParams

    cfg = args.cfg_scale if args.cfg_scale is not None else 3.0
    stg = args.stg_scale if args.stg_scale is not None else 1.0
    video_gp = MultiModalGuiderParams(
        cfg_scale=cfg,
        stg_scale=stg,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[28],
    )
    audio_gp = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=stg,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[28],
    )

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        keyframe_images=[args.start, args.end],
        keyframe_indices=[0, last_pixel_frame],
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        fps=args.fps,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        cfg_scale=cfg,
        video_guider_params=video_gp,
        audio_guider_params=audio_gp,
    )
    _print_result(args.output, t0, args.quiet)


# =============================================================================
# IC-LoRA
# =============================================================================


def _cmd_ic_lora(args: argparse.Namespace) -> None:
    """Generate video with IC-LoRA control conditioning."""
    t0 = time.time()

    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline

    # Parse --lora pairs into (path, strength) tuples
    lora_paths = [(path, float(strength)) for path, strength in args.lora]

    # Parse --video-conditioning pairs
    video_conditioning = [(path, float(strength)) for path, strength in args.video_conditioning]

    if not args.quiet:
        print("Mode: IC-LoRA (two-stage)")
        for path, strength in lora_paths:
            print(f"  LoRA: {path} (strength={strength})")
        for path, strength in video_conditioning:
            print(f"  Control: {path} (strength={strength})")

    pipe = ICLoraPipeline(
        model_dir=args.model,
        lora_paths=lora_paths,
        gemma_model_id=args.gemma,
        low_memory=True,
    )

    # Build image conditioning if provided
    images = None
    if args.image:
        images = [(args.image, 0, 1.0)]

    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        video_conditioning=video_conditioning,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        images=images,
        conditioning_attention_strength=args.conditioning_strength,
        skip_stage_2=args.skip_stage_2,
    )
    _print_result(args.output, t0, args.quiet)


# =============================================================================
# Shared helpers
# =============================================================================


def _decode_and_save(
    pipe: object,
    video_latent: object,
    audio_latent: object,
    args: argparse.Namespace,
) -> None:
    """Decode latents and save to file."""
    import tempfile
    from pathlib import Path

    from ltx_core_mlx.utils.memory import aggressive_cleanup

    if hasattr(pipe, "low_memory") and pipe.low_memory:
        pipe.dit = None
        pipe.text_encoder = None
        pipe.feature_extractor = None
        pipe._loaded = False
        aggressive_cleanup()

    assert pipe.audio_decoder is not None
    assert pipe.vocoder is not None
    mel = pipe.audio_decoder.decode(audio_latent)
    waveform = pipe.vocoder(mel)
    aggressive_cleanup()

    audio_path = tempfile.mktemp(suffix=".wav")
    pipe._save_waveform(waveform, audio_path, sample_rate=48000)

    assert pipe.vae_decoder is not None
    pipe.vae_decoder.decode_and_stream(video_latent, args.output, fps=24.0, audio_path=audio_path)
    Path(audio_path).unlink(missing_ok=True)
    aggressive_cleanup()


def _maybe_enhance_prompt(args: argparse.Namespace) -> str:
    """Enhance prompt if --enhance-prompt is set."""
    prompt = args.prompt
    if not getattr(args, "enhance_prompt", False):
        return prompt

    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    if not args.quiet:
        print("Enhancing prompt...")
    gemma = GemmaLanguageModel()
    gemma.load(args.gemma)
    if getattr(args, "image", None):
        prompt = gemma.enhance_i2v(prompt, seed=args.seed)
    else:
        prompt = gemma.enhance_t2v(prompt, seed=args.seed)
    if not args.quiet:
        print(f"Enhanced: {prompt[:200]}...")
    del gemma
    aggressive_cleanup()
    return prompt


def _print_result(output: str, t0: float, quiet: bool) -> None:
    """Print generation result."""
    elapsed = time.time() - t0
    if not quiet:
        print(f"\nSaved to: {output}")
        print(f"Time: {elapsed:.1f}s")


def _cmd_enhance(args: argparse.Namespace) -> None:
    """Enhance a prompt using Gemma."""
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

    print("Loading Gemma...")
    gemma = GemmaLanguageModel()
    gemma.load(args.gemma)

    if args.mode == "t2v":
        enhanced = gemma.enhance_t2v(args.prompt, seed=args.seed)
    else:
        enhanced = gemma.enhance_i2v(args.prompt, seed=args.seed)

    print(f"\nOriginal: {args.prompt}")
    print(f"\nEnhanced: {enhanced}")


def _cmd_info(args: argparse.Namespace) -> None:
    """Show model info and memory estimate."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    model_dir = Path(args.model)
    if not model_dir.exists():
        try:
            model_dir = Path(snapshot_download(args.model))
        except Exception as e:
            print(f"Could not find or download model: {args.model}")
            print(f"  {e}")
            sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Path:  {model_dir}")
    print()

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print("  No .safetensors files found.")
        return

    total_bytes = 0
    for f in safetensor_files:
        size = f.stat().st_size
        total_bytes += size
        print(f"  {f.name:<45s} {size / 1024**2:>8.1f} MB")

    total_mb = total_bytes / 1024**2
    total_gb = total_mb / 1024
    print(f"  {'─' * 55}")
    print(f"  {'Total':<45s} {total_mb:>8.1f} MB ({total_gb:.1f} GB)")
    print(f"  Estimated RAM: ~{total_gb * 1.3:.0f} GB (model + inference overhead)")

    json_files = sorted(model_dir.glob("*.json"))
    if json_files:
        print(f"\n  Config files: {', '.join(f.name for f in json_files)}")

    upsampler_files = [f for f in safetensor_files if "upscaler" in f.name or "upsampler" in f.name]
    if upsampler_files:
        print(f"\n  Upsamplers: {', '.join(f.stem for f in upsampler_files)}")


if __name__ == "__main__":
    main()
