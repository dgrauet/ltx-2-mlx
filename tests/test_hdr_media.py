"""HDR media: HLG writer, EXR I/O, SDR->ACEScct loader (upstream ``utils/media_io``)."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from ltx_core_mlx.color.primaries import Primaries
from ltx_core_mlx.utils.ffmpeg import find_ffmpeg, find_ffprobe
from ltx_pipelines_mlx.utils.hdr_media import (
    EXRColorSpace,
    HlgFfmpegWriter,
    align_resolution,
    encode_hdr_outputs,
    load_video_as_hdr_conditioning,
    read_exr,
    resize_and_reflect_pad,
    save_exr_frame,
)

openexr = pytest.importorskip("OpenEXR")


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


def test_writer_cleans_up_on_write_error(tmp_path):
    out = tmp_path / "m.mp4"
    with (
        pytest.raises(ValueError, match="expected"),
        HlgFfmpegWriter(str(out), width=64, height=32, fps=24.0) as writer,
    ):
        writer.write(np.full((1, 32, 64, 3), 0.5, dtype=np.float32))
        writer.write(np.full((2, 32, 32, 3), 0.5, dtype=np.float32))  # wrong width
    assert not out.exists()


def test_align_resolution_rounds_up_to_32():
    assert align_resolution(1920, 1080) == (1920, 1088, 1920, 1080)
    assert align_resolution(704, 384) == (704, 384, 704, 384)


def test_reflect_pad_then_crop_round_trips_1080():
    src = np.random.default_rng(0).random((1080, 1920, 3)).astype(np.float32)
    padded = resize_and_reflect_pad(src, 1088, 1920)
    assert padded.shape == (1088, 1920, 3)
    np.testing.assert_array_equal(padded[:1080], src)
    np.testing.assert_array_equal(padded[1080:1088], src[1078:1070:-1])  # reflect, edge excluded


def test_exr_round_trip_with_tags(tmp_path):
    rgb = np.random.default_rng(1).random((4, 6, 3)).astype(np.float32) * 8
    p = tmp_path / "f.exr"
    save_exr_frame(rgb, p, Primaries.AP1, "ACEScg")
    np.testing.assert_allclose(read_exr(p), rgb.astype(np.float16).astype(np.float32))
    with openexr.File(str(p)) as f:
        assert f.header()["colorSpace"] == "ACEScg"
        assert tuple(round(c, 5) for c in f.header()["chromaticities"]) == Primaries.AP1.exr_chromaticities


def test_sdr_video_loads_as_acescct_vae_range(tmp_path):
    src = tmp_path / "s.mp4"
    subprocess.run(
        [
            find_ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=64x32:d=0.5",
            "-r",
            "24",
            "-pix_fmt",
            "yuv444p",
            "-crf",
            "0",
            "-c:v",
            "libx264",
            str(src),
        ],
        check=True,
    )
    frames = list(load_video_as_hdr_conditioning(src, 32, 64, 9, gamma_encoded=True))
    assert len(frames) == 9 and frames[0].shape == (32, 64, 3)
    white_acescct = 0.5547  # ACEScct(1.0)
    np.testing.assert_allclose(frames[0], white_acescct * 2 - 1, atol=1e-2)


def test_encode_hdr_outputs_writes_exr_and_hlg(tmp_path):
    chunks = [np.full((2, 32, 64, 3), 0.4, dtype=np.float32), np.full((1, 32, 64, 3), 0.6, dtype=np.float32)]
    exr_dir = encode_hdr_outputs(iter(chunks), str(tmp_path / "out.mp4"), 24.0, EXRColorSpace.ACESCG)
    assert exr_dir.name == "out_acescg_exr" and len(list(exr_dir.glob("frame_*.exr"))) == 3
    assert (tmp_path / "out.mp4").stat().st_size > 0
