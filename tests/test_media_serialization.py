"""No-copy media serialization preserves exact bytes and short-write safety."""

from __future__ import annotations

import struct
import wave

import mlx.core as mx
import pytest

from ltx_core_mlx.model.video_vae.video_vae import (
    _media_write_overlap_enabled,
    _OrderedFrameWriter,
    _write_all,
)
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("maybe", False),
    ],
)
def test_media_write_overlap_is_opt_in(monkeypatch, value, expected) -> None:
    # Pins the default: the zero-copy write path is unconditional, the worker
    # thread is not. Without this, reverting the default would pass the suite.
    monkeypatch.delenv("LTX2_MEDIA_WRITE_OVERLAP", raising=False)
    if value is not None:
        monkeypatch.setenv("LTX2_MEDIA_WRITE_OVERLAP", value)
    assert _media_write_overlap_enabled() is expected


def test_ordered_frame_writer_propagates_stalled_write() -> None:
    # A sink that accepts zero bytes is a stalled stream, not a closed pipe;
    # the error surfaces as OSError naming the byte counts.
    sink = _ShortWriter(limit=0)
    writer = _OrderedFrameWriter(sink, overlap=True)
    try:
        writer.submit(bytearray([1, 2, 3]))
        with pytest.raises(OSError, match="reported 0 bytes written") as excinfo:
            writer.finish()
        # Exactly OSError: BrokenPipeError subclasses it, so a bare raises()
        # would still pass if the stalled-write path regressed to a pipe error.
        assert type(excinfo.value) is OSError
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
