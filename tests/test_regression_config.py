"""Every regression golden's CLI args must parse against the current CLI."""

from __future__ import annotations

import pytest

from ltx_pipelines_mlx.cli import _build_parser
from tests.regression.config import FPS, GOLDENS


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_golden_args_parse(name: str) -> None:
    args = _build_parser().parse_args([*GOLDENS[name]["cli_args"], "-o", "out.mp4"])
    assert args.frame_rate == FPS
