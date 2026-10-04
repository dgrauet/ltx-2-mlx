"""LipDub (upstream Dub-It) audio reference: fitted to the clip, then rebuilt from stage 1.

Upstream ``dubit.py`` (v1.4.0) builds the stage-1 audio reference from the source clip's
audio latent sliced (or right zero-padded) to the clip window
(``audio_latent_for_layout``), and rebuilds the stage-2 reference from the stage-1
generated audio (``_audio_conditionings(latent=None)``: "Stage 2 freezes stage-1 audio and
uses it as the audio-reference tokens"). Reference positions sit in negative time, ending
0.04 s before the target (``positions - aud_dur - 0.04`` with ``aud_dur`` the end of the
last token's interval).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count
from ltx_pipelines_mlx.lipdub import patchify_lipdub_audio_reference_latent
from tests.test_frozen_pipeline_streams import _make_lipdub

AUDIO_T = compute_audio_token_count(9, frame_rate=24.0)  # _make_lipdub probes 9 frames @ 24 fps


def _run(tmp_path, monkeypatch, ref_frames: int):
    pipe, loop = _make_lipdub(tmp_path, monkeypatch)
    ref = mx.arange(ref_frames * 8 * 16, dtype=mx.float32).reshape(1, 8, ref_frames, 16) + 1.0
    pipe._encode_reference_audio_vae_latent = lambda video_path: ref  # type: ignore[method-assign]
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
    return pipe, loop, ref


def _reference_tokens(state) -> mx.array:
    return state.latent[:, AUDIO_T:, :]


@pytest.mark.parametrize("ref_frames", [AUDIO_T + 7, AUDIO_T - 3])
def test_stage1_reference_is_fitted_to_the_clip(tmp_path, monkeypatch, ref_frames):
    pipe, loop, ref = _run(tmp_path, monkeypatch, ref_frames)
    tokens = _reference_tokens(loop.calls[0]["audio_state"])
    assert tokens.shape[1] == AUDIO_T
    keep = min(ref_frames, AUDIO_T)
    expected, _ = pipe.audio_patchifier.patchify(ref[:, :, :keep, :])
    assert mx.array_equal(tokens[:, :keep, :].astype(mx.float32), expected).item()
    if keep < AUDIO_T:
        assert float(mx.abs(tokens[:, keep:, :]).max()) == 0.0, "short source must be zero-padded"


def test_stage2_reference_is_the_stage1_audio(tmp_path, monkeypatch):
    _pipe, loop, _ref = _run(tmp_path, monkeypatch, AUDIO_T)
    stage1_out = loop.calls[0]["audio_state"].latent[:, :AUDIO_T, :]  # echo loop: output = input state
    stage2_state = loop.calls[1]["audio_state"]
    assert mx.array_equal(_reference_tokens(stage2_state), stage1_out).item()
    assert mx.array_equal(stage2_state.latent[:, :AUDIO_T, :], stage1_out).item()


def test_reference_positions_end_0_04s_before_the_target():
    latent = mx.zeros((1, 8, 10, 16))
    from ltx_core_mlx.components.patchifiers import AudioPatchifier

    _tokens, positions = patchify_lipdub_audio_reference_latent(latent, AudioPatchifier())
    mids = compute_audio_positions(10)
    last_end = (10 * 4 + 1 - 4) * 160 / 16000  # end of token 9's causal interval
    assert mx.allclose(positions, mids - last_end - 0.04).item()
