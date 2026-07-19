"""#74: the DiT must be freed before VAE decode in low-memory mode.

On 32 GB Macs, the two-stage strength-1.0 swap path enters
``_decode_and_save_video`` holding a fully-loaded (non-streamed) ~8 GB q8
transformer; the ~13 GB-peak video decode on top of that baseline pushes the
process past the jetsam limit and macOS SIGKILLs it silently (no traceback,
no crash report). The latents are already materialized at that point, so the
DiT is dead weight: low-memory mode must drop it (and reset ``_loaded`` so a
later ``load()`` reloads it on demand) before decoding starts.

These are pure unit tests — no weights, the orchestration decode is stubbed.
"""

from __future__ import annotations

import pytest

from ltx_pipelines_mlx._base import BasePipeline


class _Marker:
    """Stands in for a loaded LTXModel."""


@pytest.fixture
def pipe():
    """A BasePipeline skeleton with just the attributes decode touches."""
    p = BasePipeline.__new__(BasePipeline)
    p.low_memory = True
    p.verbose = False
    p._loaded = True
    p.dit = _Marker()
    # _decode_and_save_video forwards these to the orchestration impl.
    p.video_decoder_block = object()
    p.audio_decoder_block = object()
    return p


@pytest.fixture
def stub_decode(monkeypatch):
    """Stub the orchestration decode; records call kwargs for assertions."""
    calls = []

    def _stub(video_block, audio_block, video_latent, audio_latent, output_path, **kwargs):
        calls.append({"output_path": output_path, **kwargs})
        return output_path

    import ltx_pipelines_mlx.utils._orchestration as orch

    monkeypatch.setattr(orch, "decode_and_save_video", _stub)
    return calls


def test_low_memory_frees_dit_before_decode(pipe, stub_decode, monkeypatch):
    """In low-memory mode the DiT must already be gone when decode runs."""
    seen_dit_at_decode = []

    import ltx_pipelines_mlx.utils._orchestration as orch

    real_stub = orch.decode_and_save_video

    def _observing_stub(*args, **kwargs):
        seen_dit_at_decode.append(pipe.dit)
        return real_stub(*args, **kwargs)

    monkeypatch.setattr(orch, "decode_and_save_video", _observing_stub)

    result = pipe._decode_and_save_video(None, None, "out.mp4", frame_rate=25.0)

    assert result == "out.mp4"
    assert seen_dit_at_decode == [None], "DiT must be freed BEFORE the decode starts"
    assert pipe.dit is None
    assert pipe._loaded is False, "load() must reload the DiT on the next generation"


def test_without_low_memory_dit_is_kept(pipe, stub_decode):
    """Default-quality mode must not churn the DiT."""
    pipe.low_memory = False
    marker = pipe.dit

    pipe._decode_and_save_video(None, None, "out.mp4", frame_rate=25.0)

    assert pipe.dit is marker
    assert pipe._loaded is True


def test_low_memory_with_no_dit_is_a_noop(pipe, stub_decode):
    """Already-freed DiT (e.g. streaming path) must not flip state or crash."""
    pipe.dit = None
    pipe._loaded = True

    pipe._decode_and_save_video(None, None, "out.mp4", frame_rate=25.0)

    assert pipe.dit is None
    # Nothing was freed here; the load state must not be touched.
    assert pipe._loaded is True


def test_every_pipeline_routes_decode_through_base():
    """The fix lives in BasePipeline._decode_and_save_video; no subclass may
    override it, or that pipeline silently loses the jetsam protection."""
    # Import all pipeline modules so __subclasses__ is fully populated.
    import ltx_pipelines_mlx.distilled
    import ltx_pipelines_mlx.hdr_ic_lora
    import ltx_pipelines_mlx.ic_lora
    import ltx_pipelines_mlx.keyframe_interpolation
    import ltx_pipelines_mlx.lipdub
    import ltx_pipelines_mlx.ti2vid_one_stage
    import ltx_pipelines_mlx.ti2vid_two_stages  # noqa: F401

    def all_subclasses(cls):
        out = set(cls.__subclasses__())
        for sub in list(out):
            out |= all_subclasses(sub)
        return out

    subclasses = all_subclasses(BasePipeline)
    assert subclasses, "expected pipeline subclasses to be importable"
    overriders = [c.__name__ for c in subclasses if "_decode_and_save_video" in c.__dict__]
    assert not overriders, f"pipelines overriding _decode_and_save_video: {overriders}"
