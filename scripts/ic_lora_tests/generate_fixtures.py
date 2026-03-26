"""Generate synthetic control video fixtures for IC-LoRA testing.

Creates:
  - Canny edge video (white edges on black background)
  - Depth map video (animated grayscale gradient)
  - Motion track video (colored circles on black, BGR channel order)

Usage:
    uv run python scripts/ic_lora_tests/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ic_lora"
WIDTH = 352  # Half of 704 (reference at 0.5x)
HEIGHT = 240  # Half of 480
NUM_FRAMES = 33  # Short video for testing
FPS = 24


def _save_video(frames: np.ndarray, path: Path, fps: int = FPS) -> None:
    """Save frames to MP4 using ffmpeg.

    Args:
        frames: (F, H, W, 3) uint8 RGB array.
        path: Output path.
        fps: Frame rate.
    """
    import subprocess

    from ltx_core_mlx.utils.ffmpeg import find_ffmpeg

    ffmpeg = find_ffmpeg()
    h, w = frames.shape[1], frames.shape[2]

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")


def generate_canny_video() -> Path:
    """Generate a Canny-style control video: white edges on black background.

    Draws a rectangle that moves and resizes across frames, simulating
    detected edges from a moving object.
    """
    frames = np.zeros((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)

    for i in range(NUM_FRAMES):
        t = i / max(NUM_FRAMES - 1, 1)

        # Moving rectangle edges
        cx = int(WIDTH * (0.3 + 0.4 * t))
        cy = int(HEIGHT * (0.3 + 0.2 * np.sin(t * np.pi)))
        rw = int(WIDTH * (0.15 + 0.1 * t))
        rh = int(HEIGHT * (0.2 + 0.05 * np.cos(t * np.pi)))

        x1, x2 = max(0, cx - rw), min(WIDTH - 1, cx + rw)
        y1, y2 = max(0, cy - rh), min(HEIGHT - 1, cy + rh)

        # Draw rectangle edges (2px thick)
        for d in range(2):
            if y1 + d < HEIGHT:
                frames[i, y1 + d, x1:x2, :] = 255  # top edge
            if y2 - d >= 0:
                frames[i, y2 - d, x1:x2, :] = 255  # bottom edge
            if x1 + d < WIDTH:
                frames[i, y1:y2, x1 + d, :] = 255  # left edge
            if x2 - d >= 0:
                frames[i, y1:y2, x2 - d, :] = 255  # right edge

        # Add a diagonal line (simulating another edge)
        for j in range(min(HEIGHT, WIDTH)):
            px = int(j * WIDTH / HEIGHT + 20 * np.sin(t * 2 * np.pi))
            py = j
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                frames[i, py, px, :] = 255

    path = FIXTURES_DIR / "canny_control.mp4"
    _save_video(frames, path)
    print(f"  Canny: {path}")
    return path


def generate_depth_video() -> Path:
    """Generate a depth map control video: animated grayscale gradient.

    Simulates a scene with a foreground object moving closer/farther,
    represented as a bright circle on a gradient background.
    """
    frames = np.zeros((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Create coordinate grids
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]

    for i in range(NUM_FRAMES):
        t = i / max(NUM_FRAMES - 1, 1)

        # Background: horizontal gradient (left=far/dark, right=close/bright)
        bg = (xx / WIDTH * 128).astype(np.uint8)

        # Foreground: bright circle (close object) that moves
        cx = int(WIDTH * (0.3 + 0.4 * t))
        cy = int(HEIGHT * 0.5)
        radius = int(min(HEIGHT, WIDTH) * 0.15)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        fg_mask = dist < radius
        fg_value = np.clip(200 + 55 * (1.0 - dist / radius), 0, 255).astype(np.uint8)

        depth = np.where(fg_mask, fg_value, bg)
        frames[i, :, :, 0] = depth
        frames[i, :, :, 1] = depth
        frames[i, :, :, 2] = depth

    path = FIXTURES_DIR / "depth_control.mp4"
    _save_video(frames, path)
    print(f"  Depth: {path}")
    return path


def generate_motion_track_video() -> Path:
    """Generate a motion track control video: colored circles on black, BGR order.

    Follows the reference sparse_tracks.py format:
    - Background: pure black
    - Color gradient per track: blue -> green -> yellow -> red (age-based)
    - Circle sizes: small (oldest) to large (newest)
    - CRITICAL: BGR channel order (matches IC-LoRA training data)
    """
    frames = np.zeros((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    max_trail = 50
    min_radius = 2
    max_radius = 8

    # Define 3 track trajectories (smooth curves)
    tracks = []
    for track_idx in range(3):
        points = []
        for i in range(NUM_FRAMES):
            t = i / max(NUM_FRAMES - 1, 1)
            if track_idx == 0:
                # Horizontal sweep
                x = int(WIDTH * (0.2 + 0.6 * t))
                y = int(HEIGHT * (0.3 + 0.1 * np.sin(t * 2 * np.pi)))
            elif track_idx == 1:
                # Circular motion
                x = int(WIDTH * (0.5 + 0.2 * np.cos(t * 2 * np.pi)))
                y = int(HEIGHT * (0.5 + 0.2 * np.sin(t * 2 * np.pi)))
            else:
                # Diagonal with oscillation
                x = int(WIDTH * (0.1 + 0.7 * t))
                y = int(HEIGHT * (0.7 - 0.4 * t + 0.05 * np.sin(t * 4 * np.pi)))
            points.append((x, y))
        tracks.append(points)

    for i in range(NUM_FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        for track in tracks:
            # Draw trail: oldest to newest
            trail_start = max(0, i - max_trail + 1)
            trail_len = i - trail_start + 1

            for j in range(trail_start, i + 1):
                age = (j - trail_start) / max(trail_len - 1, 1)  # 0=oldest, 1=newest

                # Color gradient: blue -> green -> yellow -> red
                if age < 1 / 3:
                    r = 0.0
                    g = age * 3
                    b = 1.0 - age * 3
                elif age < 2 / 3:
                    r = (age - 1 / 3) * 3
                    g = 1.0
                    b = 0.0
                else:
                    r = 1.0
                    g = 1.0 - (age - 2 / 3) * 3
                    b = 0.0

                # Radius: small (oldest) to large (newest)
                radius = int(min_radius + (max_radius - min_radius) * age)

                x, y = track[j]
                # Draw filled circle
                yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
                circle_mask = xx**2 + yy**2 <= radius**2
                y1 = max(0, y - radius)
                y2 = min(HEIGHT, y + radius + 1)
                x1 = max(0, x - radius)
                x2 = min(WIDTH, x + radius + 1)

                cy1 = max(0, radius - y)
                cy2 = circle_mask.shape[0] - max(0, (y + radius + 1) - HEIGHT)
                cx1 = max(0, radius - x)
                cx2 = circle_mask.shape[1] - max(0, (x + radius + 1) - WIDTH)

                if y1 < y2 and x1 < x2 and cy1 < cy2 and cx1 < cx2:
                    mask = circle_mask[cy1:cy2, cx1:cx2]
                    # BGR order (matches IC-LoRA training format)
                    frame[y1:y2, x1:x2, 0][mask] = int(b * 255)  # B
                    frame[y1:y2, x1:x2, 1][mask] = int(g * 255)  # G
                    frame[y1:y2, x1:x2, 2][mask] = int(r * 255)  # R

        frames[i] = frame

    path = FIXTURES_DIR / "motion_track_control.mp4"
    _save_video(frames, path)
    print(f"  Motion track: {path}")
    return path


def main() -> None:
    """Generate all IC-LoRA test fixtures."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating IC-LoRA test fixtures ({NUM_FRAMES} frames, {WIDTH}x{HEIGHT})")
    print()

    generate_canny_video()
    generate_depth_video()
    generate_motion_track_video()

    print(f"\nFixtures saved to: {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
