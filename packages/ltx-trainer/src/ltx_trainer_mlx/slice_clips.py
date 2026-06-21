"""Slice long videos into fixed-length, resolution-normalized training clips.

ffmpeg-based helper for preparing ASMR / talking-head footage for joint
audio-video LoRA training. Each source video is sliced into ``--interval``-second
clips (or clips from an explicit timecode list), scaled+cropped to the target
resolution **without distortion**, and re-encoded with a clean 16 kHz stereo audio
stream so the downstream audio VAE encoder always finds a usable track.

Because source videos are often long (many clips each), output is organized into a
**per-source subfolder**::

    out_dir/
      <source_stem_a>/
        <source_stem_a>_000.mp4
        <source_stem_a>_000.txt   # optional caption template
        <source_stem_a>_001.mp4
        ...
      <source_stem_b>/
        ...

Aspect handling (``--fit``):
- ``crop`` (default): scale to *cover* the target, then center-crop. No distortion,
  no bars; edges are cropped.
- ``pad``: scale to *fit* inside the target, then letterbox-pad. No distortion, full
  frame preserved with black bars.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg, probe_video_info

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_SAMPLE_RATE = 16000


def _build_filter(width: int, height: int, fit: str) -> str:
    """Build the ffmpeg ``-vf`` filter for aspect-safe resizing.

    Args:
        width: Target width (must be divisible by 32 for the VAE).
        height: Target height (must be divisible by 32 for the VAE).
        fit: ``"crop"`` (cover + center-crop) or ``"pad"`` (fit + letterbox).

    Returns:
        Filter string for ffmpeg ``-vf``.
    """
    if fit == "crop":
        return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    if fit == "pad":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
    raise ValueError(f"Unknown fit mode: {fit!r} (expected 'crop' or 'pad')")


def _parse_timecodes(timecodes_file: Path) -> list[tuple[float, float]]:
    """Parse a timecode list file (``start,end`` seconds per line).

    Blank lines and lines starting with ``#`` are ignored.

    Args:
        timecodes_file: Path to the timecode list.

    Returns:
        List of ``(start, end)`` second pairs.
    """
    spans: list[tuple[float, float]] = []
    for raw in timecodes_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        start_s, end_s = line.split(",")
        spans.append((float(start_s), float(end_s)))
    return spans


def _spans_for_source(
    duration: float,
    interval: float,
    timecodes_file: Path | None,
    min_length: float,
    max_clips: int | None,
    sample: str,
    skip_start: float,
    skip_end: float,
) -> list[tuple[float, float]]:
    """Compute the (start, end) spans to cut from a single source.

    Args:
        duration: Source duration in seconds.
        interval: Fixed clip length in seconds (used when no timecodes given).
        timecodes_file: Optional explicit timecode list (overrides everything else).
        min_length: Drop clips shorter than this many seconds.
        max_clips: Cap the number of clips taken from this source (None = all).
        sample: When capping, ``"even"`` spreads clips across the whole source,
            ``"sequential"`` takes the first ``max_clips``.
        skip_start: Skip this many seconds at the start (intro).
        skip_end: Skip this many seconds at the end (outro).

    Returns:
        List of ``(start, end)`` spans.
    """
    if timecodes_file is not None:
        return _parse_timecodes(timecodes_file)

    range_end = duration - max(skip_end, 0.0)
    candidates: list[tuple[float, float]] = []
    start = max(skip_start, 0.0)
    while start < range_end:
        end = min(start + interval, range_end)
        if end - start >= min_length:
            candidates.append((start, end))
        start += interval

    if max_clips is not None and len(candidates) > max_clips:
        if sample == "even":
            if max_clips == 1:
                idxs = [0]
            else:
                idxs = sorted({round(i * (len(candidates) - 1) / (max_clips - 1)) for i in range(max_clips)})
            candidates = [candidates[i] for i in idxs]
        else:  # sequential
            candidates = candidates[:max_clips]

    return candidates


def _slice_one_source(
    video_file: Path,
    out_dir: Path,
    width: int,
    height: int,
    fps: float,
    fit: str,
    spans: list[tuple[float, float]],
    caption_template: str | None,
    crf: int,
) -> int:
    """Slice a single source video into clips inside its own subfolder.

    Returns:
        Number of clips written.
    """
    ffmpeg = find_ffmpeg()
    vf = _build_filter(width, height, fit)
    source_out = out_dir / video_file.stem
    source_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx, (start, end) in enumerate(spans):
        clip_path = source_out / f"{video_file.stem}_{idx:03d}.mp4"
        if clip_path.exists():
            logger.info("Skipping (exists): %s", clip_path.name)
            written += 1
            continue

        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_file),
            "-t",
            str(end - start),
            "-r",
            str(fps),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-ac",
            "2",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            logger.error("ffmpeg failed on %s [%s-%s]: %s", video_file.name, start, end, result.stderr.decode()[-500:])
            continue

        if caption_template is not None:
            clip_path.with_suffix(".txt").write_text(caption_template.strip() + "\n")

        written += 1

    return written


def slice_videos(
    sources: list[str | Path],
    out_dir: str | Path,
    *,
    interval: float = 4.0,
    timecodes_file: str | Path | None = None,
    res: str = "384x384",
    fps: float = 24.0,
    fit: str = "crop",
    min_length: float = 2.0,
    max_clips: int | None = None,
    sample: str = "even",
    skip_start: float = 0.0,
    skip_end: float = 0.0,
    caption_template: str | None = None,
    crf: int = 18,
) -> int:
    """Slice one or more long videos into normalized training clips.

    Args:
        sources: Video file paths (or directories — scanned for video files).
        out_dir: Root output directory; each source gets its own subfolder.
        interval: Fixed clip length in seconds (ignored if ``timecodes_file`` set).
        timecodes_file: Optional ``start,end`` list applied to every source.
        res: Target resolution ``"WxH"`` (both divisible by 32).
        fps: Output frame rate.
        fit: ``"crop"`` (center-crop) or ``"pad"`` (letterbox).
        min_length: Drop clips shorter than this (seconds).
        max_clips: Cap clips per source (None = all). Useful for building a
            browsable pool from very long footage that you then cull by hand.
        sample: When capping, ``"even"`` (spread across the source) or
            ``"sequential"`` (first N).
        skip_start: Skip this many seconds at the start of each source (intro).
        skip_end: Skip this many seconds at the end of each source (outro).
        caption_template: If set, write a ``.txt`` next to each clip with this text.
        crf: x264 quality (lower = higher quality / larger files).

    Returns:
        Total number of clips written across all sources.
    """
    width, height = (int(x) for x in res.lower().split("x"))
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(f"Resolution {res} must be divisible by 32 on both axes (VAE requirement).")

    video_files = _discover_sources(sources)
    if not video_files:
        raise ValueError("No video files found in the given sources.")

    out_dir = Path(out_dir)
    tc_path = Path(timecodes_file) if timecodes_file is not None else None

    total = 0
    for video_file in video_files:
        info = probe_video_info(str(video_file))
        if not info.has_audio:
            logger.warning(
                "Source %s has no audio stream — its clips will be unusable for audio training.", video_file.name
            )

        spans = _spans_for_source(info.duration, interval, tc_path, min_length, max_clips, sample, skip_start, skip_end)
        print(f"{video_file.name}: {info.duration:.0f}s → {len(spans)} clips @ {res}/{fps}fps ({fit})")
        total += _slice_one_source(
            video_file=video_file,
            out_dir=out_dir,
            width=width,
            height=height,
            fps=fps,
            fit=fit,
            spans=spans,
            caption_template=caption_template,
            crf=crf,
        )

    print(f"\nDone. {total} clips written under {out_dir}")
    return total


def _discover_sources(sources: list[str | Path]) -> list[Path]:
    """Expand source paths (files or directories) into a sorted list of video files."""
    found: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            found.extend(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file())
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            found.append(p)
        else:
            logger.warning("Skipping (not a video file/dir): %s", p)
    return sorted(set(found))
