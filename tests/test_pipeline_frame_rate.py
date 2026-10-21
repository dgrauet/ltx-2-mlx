"""a2v honours ``frame_rate`` (not the 24 fps default) in positions and audio length."""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.utils.positions import compute_video_positions
from tests.test_frozen_pipeline_streams import _make_a2v


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
