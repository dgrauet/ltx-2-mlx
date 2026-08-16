"""The no-copy media path depends on MLX honouring exported buffers.

``VideoDecoder.decode_and_stream`` hands a ``memoryview`` of an MLX array to a
writer thread, then immediately drops its own reference and, every eight
frames, runs ``aggressive_cleanup()`` -- ``gc.collect()`` plus
``mx.clear_cache()`` -- while that write is still in flight.

That is only safe because MLX's buffer protocol keeps the exporting array (and
therefore its unified-memory buffer) alive for as long as the ``memoryview``
lives, so the buffer is never reclaimed or handed to a later allocation. That
is an implementation property of MLX, not something the calling code can
enforce. If a future MLX release stopped honouring it, the failure would be
silently corrupted video frames rather than an exception -- the same shape as
the near-silent audio of #34 and the M2/M3 mosaic of #40.

These tests pin the property so that such a change fails loudly here instead of
in someone's render.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import pytest

from ltx_core_mlx.model.video_vae.video_vae import _OrderedFrameWriter
from ltx_core_mlx.utils.memory import aggressive_cleanup

# Small enough to stay quick, large enough to span many pages so a reclaimed
# buffer would almost certainly be overwritten by the churn below.
FRAME_SHAPE = (128, 224, 3)


class _CollectingStream:
    """Minimal stdin stand-in that records everything written to it."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, view: memoryview) -> int:
        # Copy immediately: the point under test is what the buffer held at
        # write time, not whether it still holds it afterwards.
        data = bytes(view)
        self.chunks.append(data)
        return len(data)

    def digest(self) -> str:
        return hashlib.sha256(b"".join(self.chunks)).hexdigest()


def _frame(seed: int) -> mx.array:
    """Build one evaluated, contiguous HWC uint8 frame with a known pattern."""
    count = FRAME_SHAPE[0] * FRAME_SHAPE[1] * FRAME_SHAPE[2]
    values = (mx.arange(count, dtype=mx.int32) + seed) % 251
    frame = mx.contiguous(values.astype(mx.uint8).reshape(FRAME_SHAPE))
    mx.eval(frame)
    return frame


def _churn() -> None:
    """Allocate and evaluate enough to reuse any buffer that was reclaimed."""
    for _ in range(4):
        junk = mx.zeros(FRAME_SHAPE, dtype=mx.uint8) + 7
        mx.eval(junk)
        del junk


def test_exported_buffer_survives_owner_release_and_cache_clear() -> None:
    """A memoryview outlives ``del`` + ``aggressive_cleanup()`` + reallocation."""
    frame = _frame(seed=0)
    expected = hashlib.sha256(bytes(memoryview(frame))).hexdigest()

    view = memoryview(frame)
    del frame  # the decode loop drops its reference right after submitting
    aggressive_cleanup()  # gc.collect() + mx.clear_cache()
    _churn()
    aggressive_cleanup()

    try:
        assert hashlib.sha256(bytes(view.cast("B"))).hexdigest() == expected
    finally:
        view.release()


def test_exported_buffer_survives_concurrent_cleanup() -> None:
    """The buffer stays intact while a worker reads it and the owner cleans up.

    This is the real decode-loop shape: submit, release, clean up, decode the
    next frame -- all while the previous write is still running.
    """
    verdicts: list[bool] = []

    def consume(view: memoryview, expected: str) -> None:
        buffer = memoryview(view).cast("B")
        try:
            verdicts.append(hashlib.sha256(bytes(buffer)).hexdigest() == expected)
        finally:
            buffer.release()

    executor = ThreadPoolExecutor(max_workers=1)
    pending = None
    try:
        for index in range(12):
            frame = _frame(seed=index)
            expected = hashlib.sha256(bytes(memoryview(frame))).hexdigest()

            if pending is not None:
                pending.result()
            pending = executor.submit(consume, memoryview(frame), expected)

            del frame
            if index % 8 == 0:
                aggressive_cleanup()
            _churn()

        if pending is not None:
            pending.result()
    finally:
        executor.shutdown(wait=True)

    assert verdicts == [True] * 12


@pytest.mark.parametrize("overlap", [False, True])
def test_ordered_frame_writer_survives_cleanup_in_flight(overlap: bool) -> None:
    """``_OrderedFrameWriter`` writes correct bytes despite cleanup between submits.

    The PR's own tests exercise the writer in isolation; this one reproduces the
    decode loop's interleaving, where ``aggressive_cleanup()`` runs between
    submissions with a write potentially still in flight.
    """
    stream = _CollectingStream()
    writer = _OrderedFrameWriter(stream, overlap=overlap)
    expected = hashlib.sha256()

    try:
        for index in range(12):
            frame = _frame(seed=index)
            expected.update(bytes(memoryview(frame)))
            writer.submit(frame)
            del frame
            if index % 8 == 0:
                aggressive_cleanup()
            _churn()
        writer.finish()
    finally:
        writer.shutdown()

    assert writer.completed == 12
    assert stream.digest() == expected.hexdigest()


def test_transposed_frame_requires_an_explicit_contiguous_copy() -> None:
    """``mx.contiguous`` in the decode loop is load-bearing, and its absence is loud.

    The decode loop transposes ``(3, H, W)`` to ``(H, W, 3)``, which is a strided
    view. ``_write_all`` calls ``memoryview(...).cast("B")``, and casts are only
    defined for C-contiguous views -- so dropping the ``mx.contiguous`` call
    raises rather than writing subtly wrong bytes. This test states that, so a
    future refactor removing the call is caught here.
    """
    chw = mx.arange(3 * 4 * 5, dtype=mx.int32).astype(mx.uint8).reshape(3, 4, 5)
    strided = chw.transpose(1, 2, 0)
    mx.eval(strided)

    assert not memoryview(strided).c_contiguous
    with pytest.raises(TypeError, match="C-contiguous"):
        memoryview(strided).cast("B")

    # The contiguous copy the decode loop actually makes is castable, and holds
    # the same bytes the stride-aware path would have produced.
    contiguous = mx.contiguous(strided)
    mx.eval(contiguous)
    assert memoryview(contiguous).c_contiguous
    assert bytes(memoryview(contiguous).cast("B")) == bytes(memoryview(strided))


def test_overlapped_and_serial_writers_agree_byte_for_byte() -> None:
    """Enabling the writer thread must not change a single output byte.

    ``LTX2_MEDIA_WRITE_OVERLAP`` selects between these two paths at runtime, so
    they have to be indistinguishable in their output.
    """
    digests = []
    for overlap in (False, True):
        stream = _CollectingStream()
        writer = _OrderedFrameWriter(stream, overlap=overlap)
        try:
            for index in range(12):
                frame = _frame(seed=index)
                writer.submit(frame)
                del frame
                if index % 8 == 0:
                    aggressive_cleanup()
                _churn()
            writer.finish()
        finally:
            writer.shutdown()
        digests.append(stream.digest())

    assert digests[0] == digests[1]
