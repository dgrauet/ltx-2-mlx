"""No-copy media serialization preserves exact bytes and short-write safety."""

from __future__ import annotations

import struct
import wave

import mlx.core as mx
import pytest

from ltx_core_mlx.model.video_vae.video_vae import _OrderedFrameWriter, _write_all
from ltx_pipelines_mlx.utils._orchestration import save_waveform


class _ShortWriter:
    def __init__(self, limit: int = 3):
        self.limit = limit
        self.data = bytearray()
        self.calls = 0

    def write(self, value: memoryview) -> int:
        self.calls += 1
        count = min(self.limit, len(value))
        self.data.extend(value[:count])
        return count


def test_write_all_handles_partial_writes_without_copying() -> None:
    writer = _ShortWriter(limit=3)
    source = memoryview(bytearray(range(17)))
    _write_all(source, writer)
    assert writer.data == bytearray(range(17))
    assert writer.calls > 1


@pytest.mark.parametrize("overlap", [False, True])
def test_ordered_frame_writer_preserves_exact_bytes(overlap: bool) -> None:
    sink = _ShortWriter(limit=2)
    writer = _OrderedFrameWriter(sink, overlap=overlap)
    try:
        writer.submit(bytearray([1, 2, 3]))
        writer.submit(bytearray([4, 5]))
        writer.submit(bytearray([6, 7, 8, 9]))
        writer.finish()
    finally:
        writer.shutdown()

    assert sink.data == bytearray(range(1, 10))
    assert writer.completed == 3


def test_ordered_frame_writer_propagates_pipe_failure() -> None:
    sink = _ShortWriter(limit=0)
    writer = _OrderedFrameWriter(sink, overlap=True)
    try:
        writer.submit(bytearray([1, 2, 3]))
        with pytest.raises(BrokenPipeError):
            writer.finish()
    finally:
        writer.shutdown()


def test_save_waveform_writes_reference_pcm_bytes(tmp_path) -> None:
    waveform = mx.array([[[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]]], dtype=mx.float32)
    path = tmp_path / "sample.wav"
    save_waveform(waveform, str(path), sample_rate=24000)

    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getframerate() == 24000
        payload = stream.readframes(stream.getnframes())

    assert payload == struct.pack("<7h", -32767, -32767, -16383, 0, 16383, 32767, 32767)


def test_save_waveform_interleaves_stereo_channels(tmp_path) -> None:
    waveform = mx.array([[[1.0, 0.5], [-1.0, -0.5]]], dtype=mx.float32)
    path = tmp_path / "stereo.wav"
    save_waveform(waveform, str(path))

    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 2
        payload = stream.readframes(stream.getnframes())

    assert payload == struct.pack("<4h", 32767, -32767, 16383, -16383)
