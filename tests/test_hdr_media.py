"""HDR media: HLG writer, EXR I/O, SDR->ACEScct loader (upstream ``utils/media_io``)."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from ltx_core_mlx.utils.ffmpeg import find_ffprobe
from ltx_pipelines_mlx.utils.hdr_media import HlgFfmpegWriter


def test_hlg_writer_tags_bt2020_hlg_10bit(tmp_path):
    out = tmp_path / "m.mp4"
    with HlgFfmpegWriter(str(out), width=64, height=32, fps=24.0) as writer:
        writer.write(np.full((3, 32, 64, 3), 0.5, dtype=np.float32))
        writer.write(np.full((2, 32, 64, 3), 2.0, dtype=np.float32))
    probe = json.loads(
        subprocess.run(
            [
                find_ffprobe(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,pix_fmt,color_primaries,color_transfer,color_space,color_range,nb_read_frames",
                "-of",
                "json",
                str(out),
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    )["streams"][0]
    assert probe["codec_name"] == "hevc" and probe["pix_fmt"] == "yuv420p10le"
    assert (probe["color_primaries"], probe["color_transfer"], probe["color_space"]) == (
        "bt2020",
        "arib-std-b67",
        "bt2020nc",
    )
    assert probe["color_range"] == "tv" and probe["nb_read_frames"] == "5"


def test_writer_rejects_odd_dims(tmp_path):
    with pytest.raises(ValueError, match="even"):
        HlgFfmpegWriter(str(tmp_path / "x.mp4"), width=63, height=32, fps=24.0)
