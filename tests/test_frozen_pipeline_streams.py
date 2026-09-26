"""Pipelines mark frozen streams like upstream (a2v, lipdub, retake/extend).

Mirrors the upstream ``ModalitySpec(frozen=True)`` call sites (see
``LatentState.frozen`` docs and Task 1's ``test_frozen_modality_sigma.py``):
``a2vid_two_stage.py:264-273,302-313`` (audio frozen in both stages),
``dubit.py:319-325`` (our ``lipdub.py`` stage 2 audio), ``retake.py:267-281``
(video frozen when not regenerated -- not applicable to our port, which has
no ``regenerate_video`` flag; audio frozen when ``initial_audio_latent is
not None and not regenerate_audio``).

Each test builds the real pipeline over a synthetic (weight-less) pack dir,
stubs the heavyweight collaborators (text/audio/video encoders, DiT, LoRA
fusion, upsampler, decoders), spies on the denoising-loop call(s), and
asserts the ``frozen`` flag (and, where upstream pairs it with an explicit
zero mask, the mask) on the states actually handed to the loop.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

# ---------------------------------------------------------------------------
# Shared stub collaborators
# ---------------------------------------------------------------------------


class _LoopSpy:
    """Stand-in for a denoising loop: records kwargs, echoes the input latents."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent,
            audio_latent=kwargs["audio_state"].latent,
        )


class _FakeVaeEncoder:
    """Identity latent (de)normalization -- only shapes matter here."""

    def denormalize_latent(self, x):
        return x

    def normalize_latent(self, x):
        return x


def _fake_upsampler(x):
    """2x nearest-neighbour spatial upscale on a (B, C, F, H, W) latent."""
    return mx.repeat(mx.repeat(x, 2, axis=3), 2, axis=4)


def _write_23_pack(tmp_path):
    cfg = {"transformer": {"num_layers": 2}}
    (tmp_path / "embedded_config.json").write_text(json.dumps(cfg))
    return tmp_path


# ---------------------------------------------------------------------------
# a2v: audio frozen in both stages (a2vid_two_stage.py:264-273,302-313)
# ---------------------------------------------------------------------------


def _make_a2v(tmp_path, monkeypatch):
    import ltx_pipelines_mlx.a2vid_two_stage as a2v_mod
    from ltx_core_mlx.utils.audio import AudioData
    from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage

    _write_23_pack(tmp_path)
    pipe = A2VidPipelineTwoStage(str(tmp_path), low_memory=False)

    pipe._encode_text_with_negative = lambda prompt, negative_prompt=None: (  # type: ignore[method-assign]
        mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 4, 2048), dtype=mx.bfloat16),
        mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 4, 2048), dtype=mx.bfloat16),
    )
    pipe._load_audio_encoder = lambda: None  # type: ignore[method-assign]
    pipe.audio_encoder = object()
    pipe.audio_processor = object()
    pipe.dit = object()  # type: ignore[assignment]
    pipe._fuse_distilled_lora = lambda dit: None  # type: ignore[method-assign]
    pipe.upsampler = _fake_upsampler  # type: ignore[assignment]
    pipe.image_conditioner._encoder = _FakeVaeEncoder()
    pipe._load_decoders = lambda: None  # type: ignore[method-assign]
    pipe._save_waveform = staticmethod(lambda *a, **k: None)  # type: ignore[assignment]
    pipe.video_decoder_block.decode_and_stream = lambda *a, **k: None  # type: ignore[method-assign]

    fake_audio = AudioData(waveform=mx.zeros((1, 1, 16000)), sample_rate=16000)
    monkeypatch.setattr(a2v_mod, "load_audio", lambda *a, **k: fake_audio)
    monkeypatch.setattr(a2v_mod, "encode_audio", lambda *a, **k: mx.zeros((1, 8, 32, 16), dtype=mx.bfloat16))
    monkeypatch.setattr(a2v_mod, "X0Model", lambda dit: dit)

    guided = _LoopSpy()
    euler = _LoopSpy()
    monkeypatch.setattr(a2v_mod, "guided_denoise_loop", guided)
    monkeypatch.setattr(a2v_mod, "denoise_loop", euler)
    return pipe, guided, euler


def test_a2v_audio_is_frozen_in_both_stages(tmp_path, monkeypatch):
    pipe, guided, euler = _make_a2v(tmp_path, monkeypatch)
    pipe.generate_and_save(
        prompt="a singer",
        output_path=str(tmp_path / "out.mp4"),
        audio_path="unused.wav",
        height=64,
        width=64,
        num_frames=9,
        frame_rate=24.0,
        seed=7,
        stage1_steps=2,
    )
    assert len(guided.calls) == 1 and len(euler.calls) == 1

    stage1_audio = guided.calls[0]["audio_state"]
    assert stage1_audio.frozen is True
    assert mx.array_equal(stage1_audio.denoise_mask, mx.zeros_like(stage1_audio.denoise_mask)).item()

    stage2_audio = euler.calls[0]["audio_state"]
    assert stage2_audio.frozen is True
    assert mx.array_equal(stage2_audio.denoise_mask, mx.zeros_like(stage2_audio.denoise_mask)).item()

    # Video is generated in both stages -- never frozen.
    assert guided.calls[0]["video_state"].frozen is False
    assert euler.calls[0]["video_state"].frozen is False


# ---------------------------------------------------------------------------
# lipdub: audio frozen only in stage 2 (dubit.py:319-325)
# ---------------------------------------------------------------------------


def _make_lipdub(tmp_path, monkeypatch):
    import ltx_pipelines_mlx.lipdub as lipdub_mod
    from ltx_pipelines_mlx.lipdub import LipDubPipeline

    _write_23_pack(tmp_path)
    lora_path = tmp_path / "lora.safetensors"
    lora_path.write_bytes(b"")  # exists() is True; metadata read fails gracefully -> downscale=1

    pipe = LipDubPipeline(str(tmp_path), lora_paths=[(str(lora_path), 1.0)], low_memory=False)

    pipe._load_text_encoder = lambda: None  # type: ignore[method-assign]
    pipe._encode_text = lambda prompt: (  # type: ignore[method-assign]
        mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 4, 2048), dtype=mx.bfloat16),
    )
    pipe._encode_reference_audio_vae_latent = lambda video_path: mx.zeros(  # type: ignore[method-assign]
        (1, 8, 32, 16), dtype=mx.bfloat16
    )
    pipe.load = lambda: None  # type: ignore[method-assign]
    pipe._fuse_loras = lambda: None  # type: ignore[method-assign]
    pipe.dit = object()  # type: ignore[assignment]
    pipe.vae_encoder = _FakeVaeEncoder()  # type: ignore[assignment]
    pipe.upsampler = _fake_upsampler  # type: ignore[assignment]

    class _Meta:
        num_frames = 9
        fps = 24.0

    monkeypatch.setattr(lipdub_mod, "probe_video_info", lambda path: _Meta())
    monkeypatch.setattr(lipdub_mod, "append_ic_lora_reference_video_conditionings", lambda conds, *a, **k: None)
    monkeypatch.setattr(lipdub_mod, "X0Model", lambda dit: dit)

    loop = _LoopSpy()
    monkeypatch.setattr(lipdub_mod, "denoise_loop", loop)
    return pipe, loop


def test_lipdub_audio_frozen_only_in_stage2(tmp_path, monkeypatch):
    pipe, loop = _make_lipdub(tmp_path, monkeypatch)
    pipe.generate_lipdub(
        prompt="lip sync",
        reference_video_path="unused.mp4",
        height=64,
        width=64,
        seed=7,
        stage1_steps=1,
        stage2_steps=1,
    )
    assert len(loop.calls) == 2

    stage1_audio = loop.calls[0]["audio_state"]
    assert stage1_audio.frozen is False

    stage2_audio = loop.calls[1]["audio_state"]
    assert stage2_audio.frozen is True
    # Reference tokens are appended after freezing; the target-audio prefix
    # of the mask stays all-zero (only the appended reference block, which
    # is always clean/mask=0 too, follows it).
    assert mx.array_equal(stage2_audio.denoise_mask, mx.zeros_like(stage2_audio.denoise_mask)).item()

    # Video is generated in both stages -- never frozen.
    assert loop.calls[0]["video_state"].frozen is False
    assert loop.calls[1]["video_state"].frozen is False


# ---------------------------------------------------------------------------
# retake: audio frozen iff not regenerate_audio; video never frozen (no
# ``regenerate_video`` flag in this port -- see retake.py comment).
# ---------------------------------------------------------------------------


def _make_retake(tmp_path, monkeypatch):
    import ltx_pipelines_mlx.retake as retake_mod
    from ltx_pipelines_mlx.retake import RetakePipeline

    _write_23_pack(tmp_path)
    pipe = RetakePipeline(str(tmp_path), low_memory=False)

    pipe._encode_text_with_negative = lambda prompt, negative_prompt=None: (  # type: ignore[method-assign]
        mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 4, 2048), dtype=mx.bfloat16),
        mx.zeros((1, 4, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 4, 2048), dtype=mx.bfloat16),
    )
    pipe.dit = object()  # type: ignore[assignment]
    monkeypatch.setattr(retake_mod, "X0Model", lambda dit: dit)

    guided = _LoopSpy()
    monkeypatch.setattr(retake_mod, "guided_denoise_loop", guided)
    return pipe, guided


def _source_latents(frames=2, height=2, width=2, audio_frames=8):
    video_latent = mx.zeros((1, 128, frames, height, width), dtype=mx.bfloat16)
    audio_latent = mx.zeros((1, 8, audio_frames, 16), dtype=mx.bfloat16)
    return video_latent, audio_latent


@pytest.mark.parametrize("regenerate_audio", [True, False])
def test_retake_audio_frozen_iff_not_regenerated(tmp_path, monkeypatch, regenerate_audio):
    pipe, guided = _make_retake(tmp_path, monkeypatch)
    video_latent, audio_latent = _source_latents()
    pipe.retake(
        prompt="a different action",
        source_video_latent=video_latent,
        source_audio_latent=audio_latent,
        start_frame=0,
        end_frame=1,
        height=64,
        width=64,
        num_frames=9,
        frame_rate=24.0,
        seed=7,
        num_steps=1,
        regenerate_audio=regenerate_audio,
    )
    assert len(guided.calls) == 1
    audio_state = guided.calls[0]["audio_state"]
    assert audio_state.frozen is (not regenerate_audio)

    # Video always carries a partial (temporal-region) mask, never a fully
    # frozen whole-stream marker: our port has no ``regenerate_video`` flag.
    assert guided.calls[0]["video_state"].frozen is False


def test_extend_never_marks_a_stream_frozen(tmp_path, monkeypatch):
    """extend always grows both streams with new content -- nothing is a
    whole-stream conditioning, so neither state is ever ``frozen``."""
    pipe, guided = _make_retake(tmp_path, monkeypatch)
    video_latent, audio_latent = _source_latents()
    pipe.extend(
        prompt="continue the scene",
        source_video_latent=video_latent,
        source_audio_latent=audio_latent,
        extend_frames=1,
        height=64,
        width=64,
        num_frames=9,
        frame_rate=24.0,
        seed=7,
        num_steps=1,
    )
    assert len(guided.calls) == 1
    assert guided.calls[0]["video_state"].frozen is False
    assert guided.calls[0]["audio_state"].frozen is False
