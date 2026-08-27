"""Regression: the retake/extend CLI decode helper forwards the source frame rate.

The v0.14.0 audit made ``_decode_and_save_video``'s ``frame_rate`` keyword-only;
the shared CLI helper ``_decode_and_save`` was missed (wrapper-audit-gap class,
see the v0.14.1 memo) and every ``ltx-2-mlx retake``/``extend`` invocation died
with a TypeError at decode time. These tests pin the forwarding.
"""

import argparse

import pytest

from ltx_pipelines_mlx.cli import _decode_and_save


class _SpyPipe:
    """Minimal pipeline double for the CLI decode helper."""

    low_memory = False
    source_frame_rate: float | None = 24.0

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def _load_decoders(self) -> None:
        self.calls.append({"load_decoders": True})

    def _decode_and_save_video(self, video_latent, audio_latent, output_path, *, frame_rate: float):
        self.calls.append({"output": output_path, "frame_rate": frame_rate})


def test_decode_and_save_forwards_source_frame_rate(tmp_path):
    pipe = _SpyPipe()
    pipe.source_frame_rate = 30.0
    args = argparse.Namespace(output=str(tmp_path / "out.mp4"))

    _decode_and_save(pipe, object(), object(), args)

    decode_call = pipe.calls[-1]
    assert decode_call["frame_rate"] == 30.0
    assert decode_call["output"] == args.output


def test_decode_and_save_requires_recorded_frame_rate(tmp_path):
    pipe = _SpyPipe()
    pipe.source_frame_rate = None
    args = argparse.Namespace(output=str(tmp_path / "out.mp4"))

    with pytest.raises(RuntimeError, match="source_frame_rate"):
        _decode_and_save(pipe, object(), object(), args)


def test_retake_cli_forwards_low_ram(monkeypatch):
    """The retake/extend subcommands construct RetakePipeline with low_ram_streaming.

    Upstream's RetakePipeline threads ``offload_mode`` into its stages; our
    ``--low-ram`` is the MLX equivalent, so dropping the pass-through would
    silently regress to full-weight loading (the 32 GB OOM class).
    """
    from ltx_pipelines_mlx.cli import _build_parser

    parser = _build_parser()
    for argv in (
        ["retake", "-p", "x", "-o", "o.mp4", "--video", "v.mp4", "--start", "1", "--end", "2", "--low-ram"],
        ["extend", "-p", "x", "-o", "o.mp4", "--video", "v.mp4", "--extend-frames", "4", "--low-ram"],
    ):
        args = parser.parse_args(argv)
        assert args.low_ram is True

    args = parser.parse_args(["retake", "-p", "x", "-o", "o.mp4", "--video", "v.mp4", "--start", "1", "--end", "2"])
    assert args.low_ram is False


def test_retake_pipeline_accepts_low_ram_streaming(tmp_path, monkeypatch):
    from ltx_pipelines_mlx.retake import RetakePipeline

    (tmp_path / "embedded_config.json").write_text('{"transformer": {}}')
    pipe = RetakePipeline(model_dir=str(tmp_path), low_ram_streaming=True)
    assert pipe.low_ram_streaming is True
