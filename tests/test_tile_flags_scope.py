"""--tile-* exists only where a pipeline honours it.

Only ``generate`` builds a TileCountConfig and hands it to its pipeline. a2v,
keyframe, ic-lora and hdr-ic-lora used to accept the flags and silently ignore
them (their pipelines never read ``tile_count``); they now refuse them.
"""

from __future__ import annotations

import pytest

from ltx_pipelines_mlx.cli import _build_parser

_BASE = ["-p", "x", "-o", "o.mp4", "--frame-rate", "24"]


@pytest.mark.parametrize(
    "argv",
    [
        ["a2v", *_BASE, "--audio", "a.wav"],
        ["keyframe", *_BASE, "--start", "a.png", "--end", "b.png"],
        ["ic-lora", *_BASE, "--lora", "l.safetensors", "1.0", "--video-conditioning", "c.mp4", "1.0"],
        ["hdr-ic-lora", *_BASE, "--lora", "l.safetensors", "1.0"],
    ],
    ids=lambda a: a[0],
)
@pytest.mark.parametrize("flag", ["--tile-frames", "--tile-spatial", "--tile-overlap"])
def test_pipelines_without_tiling_reject_the_flags(argv, flag):
    _build_parser().parse_args(argv)  # the base command line itself is valid
    with pytest.raises(SystemExit):
        _build_parser().parse_args([*argv, flag, "2"])


def test_generate_keeps_the_flags():
    args = _build_parser().parse_args(["generate", "--distilled", *_BASE, "--tile-frames", "2", "--tile-spatial", "2"])
    assert (args.tile_frames, args.tile_spatial) == (2, 2)
