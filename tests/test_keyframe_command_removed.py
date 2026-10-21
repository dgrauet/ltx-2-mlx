"""The ``keyframe`` command and ``KeyframeInterpolationPipeline`` are gone (upstream v1.5.0).

Interpolating between images is ``generate --two-stage`` with each image bound as a guide and
``--refine-audio``.
"""

from __future__ import annotations

import pytest

import ltx_pipelines_mlx
from ltx_pipelines_mlx.cli import _build_parser


def test_keyframe_subcommand_is_gone():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["keyframe", "-p", "x", "-o", "o.mp4", "--frame-rate", "24", "--start", "a.png", "--end", "b.png"]
        )


def test_keyframe_pipeline_is_not_exported():
    assert not hasattr(ltx_pipelines_mlx, "KeyframeInterpolationPipeline")
    assert "KeyframeInterpolationPipeline" not in ltx_pipelines_mlx.__all__


def test_replacement_command_parses():
    args = _build_parser().parse_args(
        [
            "generate",
            "--two-stage",
            *("-p", "x", "-o", "o.mp4", "--frame-rate", "24", "-f", "97"),
            *("--image", "a.png", "0", "1.0", "guide", "--image", "b.png", "last", "1.0", "guide"),
            "--refine-audio",
        ]
    )
    assert args.two_stage and args.refine_audio and len(args.images) == 2
