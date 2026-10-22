"""The local distilled pipelines are importable from the package root (#211)."""

from __future__ import annotations

import ltx_pipelines_mlx
from ltx_pipelines_mlx.a2vid_dfr import A2VidDFRPipeline
from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline
from ltx_pipelines_mlx.extend_distilled import ExtendDistilledPipeline


def test_local_distilled_pipelines_are_exported():
    assert ltx_pipelines_mlx.A2VidDistilledPipeline is A2VidDistilledPipeline
    assert ltx_pipelines_mlx.A2VidDFRPipeline is A2VidDFRPipeline
    assert ltx_pipelines_mlx.ExtendDistilledPipeline is ExtendDistilledPipeline
    assert {"A2VidDistilledPipeline", "A2VidDFRPipeline", "ExtendDistilledPipeline"} <= set(ltx_pipelines_mlx.__all__)
