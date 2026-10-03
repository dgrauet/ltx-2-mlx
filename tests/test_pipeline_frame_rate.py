"""a2v and keyframe honour ``frame_rate`` (not the 24 fps default) in positions and audio length."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.utils.positions import compute_audio_token_count, compute_video_positions
from tests.test_frozen_pipeline_streams import _make_a2v
from tests.test_negative_prompt import _pack


def test_a2v_video_positions_use_the_frame_rate(tmp_path, monkeypatch):
    pipe, guided, euler = _make_a2v(tmp_path, monkeypatch)
    pipe.generate_and_save(
        prompt="a singer",
        output_path=str(tmp_path / "out.mp4"),
        audio_path="unused.wav",
        height=64,
        width=64,
        num_frames=9,
        frame_rate=30.0,
        seed=7,
        stage1_steps=2,
    )
    F, H, W = compute_video_latent_shape(9, 32, 32)
    n1 = F * H * W
    pos_1 = guided.calls[0]["video_state"].positions[:, :n1]
    pos_2 = euler.calls[0]["video_state"].positions[:, : n1 * 4]
    assert mx.allclose(pos_1, compute_video_positions(F, H, W, frame_rate=30.0)).item()
    assert mx.allclose(pos_2, compute_video_positions(F, 2 * H, 2 * W, frame_rate=30.0)).item()


class _AbortError(Exception):
    pass


def test_keyframe_audio_length_uses_the_frame_rate(tmp_path, monkeypatch):
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
        if len(shapes) == 2:  # stage-1 audio state: all we need
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
            num_frames=97,
            frame_rate=30.0,
            seed=7,
            cfg_scale=3.0,  # guided path: uses the stubbed _encode_text_with_negative
        )
    assert shapes[1] == (1, compute_audio_token_count(97, frame_rate=30.0), 128)
    assert compute_audio_token_count(97, frame_rate=30.0) != compute_audio_token_count(97)
