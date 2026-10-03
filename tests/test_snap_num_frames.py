"""Off-grid frame counts are floored to the causal grid before any sizing (upstream v1.4.0).

The video latent only covers ``(F - 1) * 8 + 1`` frames, so an off-grid request such as 87
frames decodes 81. Upstream now sizes the audio from that snapped canvas ("Generated audio
could outlast the decoded video when num_frames was not on the video VAE temporal grid");
before, our audio was sized from the raw count and outlasted the video.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import pytest

from ltx_core_mlx.utils.positions import compute_audio_token_count
from ltx_pipelines_mlx.utils.blocks import snap_num_frames
from tests.test_frozen_pipeline_streams import _make_a2v
from tests.test_ltx25_distilled import _make_stubbed_pipeline, _run
from tests.test_negative_prompt import _pack


def test_snap_num_frames_floors_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        assert snap_num_frames(87) == 81
    assert "87" in caplog.text and "81" in caplog.text


def test_on_grid_counts_pass_through_silently(caplog):
    with caplog.at_level(logging.WARNING):
        assert snap_num_frames(97) == 97
    assert caplog.text == ""


@pytest.mark.parametrize("ltx25", [False, True])
def test_distilled_sizes_audio_from_the_snapped_canvas(tmp_path, monkeypatch, ltx25):
    pipe, _euler, _ancestral, noised_calls = _make_stubbed_pipeline(tmp_path, monkeypatch, ltx25=ltx25)
    video, audio = _run(pipe, num_frames=87)
    assert noised_calls[1]["base_shape"][1] == compute_audio_token_count(81, frame_rate=24.0)
    assert audio.shape[2] == compute_audio_token_count(81, frame_rate=24.0)
    assert video.shape[2] == (81 - 1) // 8 + 1


def test_a2v_sizes_audio_from_the_snapped_canvas(tmp_path, monkeypatch):
    pipe, guided, _euler = _make_a2v(tmp_path, monkeypatch)
    pipe.generate_and_save(
        prompt="a singer",
        output_path=str(tmp_path / "out.mp4"),
        audio_path="unused.wav",
        height=64,
        width=64,
        num_frames=12,
        frame_rate=24.0,
        seed=7,
        stage1_steps=2,
    )
    assert guided.calls[0]["audio_state"].latent.shape[1] == compute_audio_token_count(9, frame_rate=24.0)


class _AbortError(Exception):
    pass


def test_keyframe_sizes_audio_from_the_snapped_canvas(tmp_path, monkeypatch):
    import ltx_pipelines_mlx.keyframe_interpolation as kf_mod
    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    pipe = KeyframeInterpolationPipeline(model_dir=_pack(tmp_path), dev_transformer="transformer-dev.safetensors")
    pipe.image_conditioner = lambda fn, free_after: ([], [])
    pipe._encode_text_with_negative = lambda prompt, negative_prompt=None: (  # type: ignore[method-assign]
        *(mx.zeros((1, 4, d), dtype=mx.bfloat16) for d in (4096, 2048, 4096, 2048)),
    )
    pipe.dit = object()  # type: ignore[assignment]
    pipe.upsampler = object()  # type: ignore[assignment]
    shapes: list[tuple] = []
    real = kf_mod.create_noised_state

    def spy(**kwargs):
        shapes.append(kwargs["base_shape"])
        if len(shapes) == 2:
            raise _AbortError
        return real(**kwargs)

    monkeypatch.setattr(kf_mod, "create_noised_state", spy)
    with pytest.raises(_AbortError):
        pipe.interpolate(
            prompt="p",
            keyframe_images=[],
            keyframe_indices=[],
            height=64,
            width=64,
            num_frames=87,
            frame_rate=24.0,
            seed=7,
            cfg_scale=3.0,
        )
    assert shapes[1] == (1, compute_audio_token_count(81, frame_rate=24.0), 128)
