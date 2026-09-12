"""``--no-audio`` on ``generate`` skips audio decode + mux; video generation is untouched (#126).

The DiT still produces audio latents jointly (there is no video-only forward
upstream); the flag only stops the audio decoders from being loaded and run,
and writes an mp4 without an audio track.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_pipelines_mlx._base import BasePipeline
from ltx_pipelines_mlx.utils._orchestration import decode_and_save_video


class _AudioDecoderMustNotRun:
    def load(self) -> None:
        raise AssertionError("audio decoder loaded despite generate_audio=False")

    def __call__(self, audio_latent):
        raise AssertionError("audio decoder ran despite generate_audio=False")


class _RecordingVideoDecoder:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def decode_and_stream(self, latent, output_path, *, frame_rate, audio_path=None):
        self.calls.append({"output_path": output_path, "frame_rate": frame_rate, "audio_path": audio_path})


def test_orchestration_skips_audio_decoder_and_muxes_no_track(tmp_path):
    video = _RecordingVideoDecoder()
    out = str(tmp_path / "out.mp4")

    result = decode_and_save_video(
        video,
        _AudioDecoderMustNotRun(),
        mx.zeros((1, 128, 1, 2, 2)),
        mx.zeros((1, 8, 4, 16)),
        out,
        frame_rate=24.0,
        generate_audio=False,
    )

    assert result == out
    assert video.calls == [{"output_path": out, "frame_rate": 24.0, "audio_path": None}]


def test_pipeline_generate_audio_defaults_on():
    p = BasePipeline.__new__(BasePipeline)
    assert p.generate_audio is True


def test_load_decoders_skips_audio_block_when_off():
    p = BasePipeline.__new__(BasePipeline)
    p.verbose = False
    p.generate_audio = False
    p.video_decoder_block = _RecordingVideoDecoder()
    p.audio_decoder_block = _AudioDecoderMustNotRun()

    p._load_decoders()

    assert p.video_decoder_block.loaded


def test_decode_and_save_video_forwards_generate_audio(monkeypatch):
    seen = {}

    def _stub(video_block, audio_block, video_latent, audio_latent, output_path, **kwargs):
        seen.update(kwargs)
        return output_path

    import ltx_pipelines_mlx.utils._orchestration as orch

    monkeypatch.setattr(orch, "decode_and_save_video", _stub)

    p = BasePipeline.__new__(BasePipeline)
    p.low_memory = False
    p.verbose = False
    p.dit = None
    p._loaded = False
    p.video_decoder_block = object()
    p.audio_decoder_block = object()
    p.generate_audio = False

    p._decode_and_save_video(mx.zeros((1,)), mx.zeros((1,)), "out.mp4", frame_rate=24.0)

    assert seen["generate_audio"] is False


def _parse_generate_args(*extra: str):
    from ltx_pipelines_mlx.cli import _build_parser

    return _build_parser().parse_args(
        ["generate", "-p", "a fox", "-o", "out.mp4", "--frame-rate", "24", "-f", "9", *extra]
    )


def test_cli_no_audio_flag_defaults_off_and_parses():
    assert _parse_generate_args("--distilled").no_audio is False
    assert _parse_generate_args("--distilled", "--no-audio").no_audio is True


@pytest.mark.parametrize("mode", ["--distilled", "--one-stage", "--two-stage", "--two-stages-hq"])
def test_cli_no_audio_reaches_every_generate_mode(monkeypatch, mode):
    """Each generate mode must set ``pipe.generate_audio`` from the flag before generating."""
    seen = {}

    class _FakePipe:
        def __init__(self, *args, **kwargs):
            pass

        def generate_and_save(self, **kwargs):
            seen["generate_audio"] = getattr(self, "generate_audio", "UNSET")

    import ltx_pipelines_mlx.distilled as distilled
    import ltx_pipelines_mlx.ti2vid_one_stage as one_stage
    import ltx_pipelines_mlx.ti2vid_two_stages as two_stages
    import ltx_pipelines_mlx.ti2vid_two_stages_hq as two_stages_hq

    monkeypatch.setattr(distilled, "DistilledPipeline", _FakePipe)
    monkeypatch.setattr(one_stage, "TI2VidOneStagePipeline", _FakePipe)
    monkeypatch.setattr(two_stages, "TI2VidTwoStagesPipeline", _FakePipe)
    monkeypatch.setattr(two_stages_hq, "TI2VidTwoStagesHQPipeline", _FakePipe)

    from ltx_pipelines_mlx.cli import _cmd_generate

    _cmd_generate(_parse_generate_args(mode, "--no-audio", "--quiet"))

    assert seen["generate_audio"] is False
