"""Tests for the LTX-2.5 duration head.

Non-slow tests exercise the pure-MLX module (shape contracts, the
video-only/audio-only/both-None branches, and a numerical regression pinned
against real pack weights). The slow bidirectional load-contract test (all
15 pack tensors consumed, no model param left unfed) requires the local
``ltx-2.5-mlx-q8`` pack and is gated the same way as
``test_ltx25_load_contract.py``.
"""

from __future__ import annotations

import json
import struct

import mlx.core as mx
import pytest

from ltx_core_mlx.duration_head import AttentionPooler, DurationHead, load_duration_head
from ltx_pipelines_mlx.utils.blocks import (
    DurationPredictor,
    require_num_frames_source,
    resolve_num_frames,
    seconds_to_clamped_num_frames,
)
from ltx_pipelines_mlx.utils.types import AutoDuration
from tests.conftest import LTX25_Q8_DIR


def test_duration_head_requires_at_least_one_modality():
    model = DurationHead()
    with pytest.raises(ValueError, match="at least one of"):
        model()


def test_duration_head_video_only_shape():
    model = DurationHead()
    video_tokens = mx.random.normal((2, 8, 4096))
    out = model(video_tokens=video_tokens)
    assert out.shape == (2,)


def test_duration_head_audio_only_shape():
    model = DurationHead()
    audio_tokens = mx.random.normal((2, 8, 2048))
    out = model(audio_tokens=audio_tokens)
    assert out.shape == (2,)


def test_duration_head_both_modalities_shape():
    model = DurationHead()
    video_tokens = mx.random.normal((3, 5, 4096))
    audio_tokens = mx.random.normal((3, 7, 2048))
    out = model(video_tokens=video_tokens, audio_tokens=audio_tokens)
    assert out.shape == (3,)


def test_duration_head_output_is_positive():
    """exp() is applied internally, so output must always be > 0 regardless of sign of logits."""
    model = DurationHead()
    video_tokens = mx.random.normal((4, 3, 4096)) * 100.0  # large-magnitude logits
    out = model(video_tokens=video_tokens)
    assert bool(mx.all(out > 0))


def test_attention_pooler_pools_to_num_queries():
    pooler = AttentionPooler(hidden_dim=256, num_queries=3, num_heads=4)
    tokens = mx.random.normal((2, 10, 256))
    out = pooler(tokens)
    assert out.shape == (2, 3, 256)


def test_load_duration_head_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_duration_head(tmp_path / "does_not_exist.safetensors")


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_load_duration_head_covers_every_pack_tensor():
    """Bidirectional load contract: every pack tensor is consumed, every model param is fed.

    Mirrors ``test_ltx25_load_contract.py``'s pattern (#52 silent-no-op class):
    a weight nobody reads, or a param the pack doesn't serve, must fail loudly.
    """
    path = LTX25_Q8_DIR / "duration_head.safetensors"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    pack_keys = {k for k in header if k != "__metadata__"}
    assert len(pack_keys) == 15, f"expected 15 duration_head tensors, pack has {len(pack_keys)}"

    # load_duration_head() itself would raise via load_weights(strict=True) if any
    # model param went unfed, or silently ignore extra pack keys if not filtered --
    # so also check the raw key set explicitly maps 1:1 onto the expected schema.
    stripped = {k.removeprefix("duration_head.") for k in pack_keys}
    expected = {
        "video_input_proj.weight",
        "video_input_proj.bias",
        "video_modality_emb",
        "audio_input_proj.weight",
        "audio_input_proj.bias",
        "audio_modality_emb",
        "attention_pooler.query_tokens",
        "attention_pooler.cross_attn.in_proj_weight",
        "attention_pooler.cross_attn.in_proj_bias",
        "attention_pooler.cross_attn.out_proj.weight",
        "attention_pooler.cross_attn.out_proj.bias",
        "mlp_hidden.weight",
        "mlp_hidden.bias",
        "mlp_out.weight",
        "mlp_out.bias",
    }
    assert stripped == expected

    model = load_duration_head(path)
    mx.eval(model.parameters())

    video_tokens = mx.random.normal((1, 16, 4096))
    audio_tokens = mx.random.normal((1, 16, 2048))
    out = model(video_tokens=video_tokens, audio_tokens=audio_tokens)
    assert out.shape == (1,)
    assert bool(out.item() > 0)


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_duration_head_pinned_regression():
    """Pure-MLX regression pinned from a seeded input, derived from the torch parity harness.

    Parity numbers (see task-1-report.md): torch vs. MLX on real weights with
    identical seeded inputs (video ``(1,1024,4096)``, audio ``(1,1024,2048)``,
    ``torch.manual_seed(0)``):
        both:  torch=3.572701  mlx=3.57217   rel_err=1.5e-4
        video: torch=3.199023  mlx=3.19872   rel_err=9.4e-5
        audio: torch=3.893557  mlx=3.89286   rel_err=1.8e-4

    This test pins the MLX-only values (mx.random-seeded, independent of the
    torch harness's exact RNG stream) so future refactors can't silently drift.
    """
    path = LTX25_Q8_DIR / "duration_head.safetensors"
    model = load_duration_head(path)

    mx.random.seed(0)
    video_tokens = mx.random.normal((1, 1024, 4096))
    audio_tokens = mx.random.normal((1, 1024, 2048))

    out_both = model(video_tokens=video_tokens, audio_tokens=audio_tokens)
    out_video = model(video_tokens=video_tokens)
    out_audio = model(audio_tokens=audio_tokens)

    # Pinned against the reference pack; regenerate if the pack's weights change.
    assert out_both.item() == pytest.approx(4.09065055847168, rel=1e-5)
    assert out_video.item() == pytest.approx(3.8629565238952637, rel=1e-5)
    assert out_audio.item() == pytest.approx(4.285473346710205, rel=1e-5)


# ============================================================================
# Task 2: AutoDuration + DurationPredictor + helpers
# ============================================================================


def test_seconds_snap_to_grid():
    """Test snapping to VAE's 8k+1 causal temporal grid."""
    # 2.0s @ 24fps = 48 -> floor grille 8k+1 = 41
    assert seconds_to_clamped_num_frames(2.0, frame_rate=24.0) == 41
    # clamp haut : 100s @ 24 -> max_frames 1024 -> snap 1017
    assert seconds_to_clamped_num_frames(100.0, frame_rate=24.0) == 1017
    # remontée quand le floor passe sous min_frames
    assert seconds_to_clamped_num_frames(0.01, frame_rate=24.0, min_frames=24) == 25


def test_require_num_frames_source_raises_without_predictor():
    """AutoDuration without predictor raises; explicit int never raises."""
    with pytest.raises(ValueError, match="Pass num_frames explicitly"):
        require_num_frames_source(AutoDuration(), None)
    # Explicit int should not raise
    require_num_frames_source(97, None)


def test_resolve_passthrough_and_predict():
    """Explicit int is passed through; AutoDuration delegates to predictor."""
    # Passthrough for explicit int
    assert resolve_num_frames(97, None, video_encoding=None, audio_encoding=None, frame_rate=24.0) == 97

    # Fake predictor that records its arguments
    class FakePredictor:
        def __init__(self):
            self.call_args = None

        def __call__(self, video_encoding, audio_encoding, *, frame_rate, min_seconds=1.0, max_seconds=20.0):
            self.call_args = {
                "video_encoding": video_encoding,
                "audio_encoding": audio_encoding,
                "frame_rate": frame_rate,
                "min_seconds": min_seconds,
                "max_seconds": max_seconds,
            }
            return 41  # Dummy return value on the 8k+1 grid

    # Test AutoDuration delegation with fake predictor
    fake_predictor = FakePredictor()
    video_enc = mx.random.normal((1, 16, 4096))
    audio_enc = mx.random.normal((1, 16, 2048))
    auto_dur = AutoDuration(min_seconds=0.5, max_seconds=10.0)

    result = resolve_num_frames(
        auto_dur,
        fake_predictor,
        video_encoding=video_enc,
        audio_encoding=audio_enc,
        frame_rate=24.0,
    )

    # Verify result is passed through
    assert result == 41

    # Verify predictor was called with correct arguments
    assert fake_predictor.call_args is not None
    assert fake_predictor.call_args["video_encoding"] is video_enc
    assert fake_predictor.call_args["audio_encoding"] is audio_enc
    assert fake_predictor.call_args["frame_rate"] == 24.0
    # Verify min/max_seconds come from AutoDuration instance
    assert fake_predictor.call_args["min_seconds"] == 0.5
    assert fake_predictor.call_args["max_seconds"] == 10.0


def test_duration_predictor_from_checkpoint_missing_file(tmp_path):
    """from_checkpoint returns None when duration_head.safetensors is absent."""
    predictor = DurationPredictor.from_checkpoint(tmp_path)
    assert predictor is None


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_duration_predictor_from_checkpoint_loads(tmp_path):
    """from_checkpoint loads a real predictor from pack with duration_head.safetensors."""
    predictor = DurationPredictor.from_checkpoint(LTX25_Q8_DIR)
    assert predictor is not None

    # Test that predictor returns int on grid for random encodings
    video_encoding = mx.random.normal((1, 16, 4096))
    audio_encoding = mx.random.normal((1, 16, 2048))
    num_frames = predictor(video_encoding, audio_encoding, frame_rate=24.0)

    assert isinstance(num_frames, int)
    # Verify it's on the 8k+1 grid: (num_frames - 1) % 8 == 0
    assert (num_frames - 1) % 8 == 0
    # Verify default clamps are respected (1s @ 24fps = 24, 20s @ 24fps = 480)
    assert num_frames >= 25  # Snapped up from 24
    assert num_frames <= 1017  # Snapped down from 480


# ============================================================================
# Task 3: CLI --frames/--auto-duration collapse
# ============================================================================


def _parse_generate_args(*extra: str):
    from ltx_pipelines_mlx.cli import _build_parser

    parser = _build_parser()
    return parser.parse_args(["generate", "-p", "a fox", "-o", "out.mp4", "--frame-rate", "24", "--distilled", *extra])


def test_cli_frames_default_is_none_and_resolves_to_auto_duration():
    """No -f => parsed default is None => resolves to AutoDuration()."""
    from ltx_pipelines_mlx.cli import _resolve_num_frames_arg

    args = _parse_generate_args()
    assert args.frames is None
    assert args.auto_duration is None

    resolved = _resolve_num_frames_arg(args)
    assert resolved == AutoDuration()


def test_cli_explicit_frames_wins():
    from ltx_pipelines_mlx.cli import _resolve_num_frames_arg

    args = _parse_generate_args("-f", "25")
    assert args.frames == 25
    assert _resolve_num_frames_arg(args) == 25


def test_cli_auto_duration_flag_parses_and_resolves():
    from ltx_pipelines_mlx.cli import _resolve_num_frames_arg

    args = _parse_generate_args("--auto-duration", "2:10")
    assert args.auto_duration == AutoDuration(min_seconds=2.0, max_seconds=10.0)
    assert args.frames is None
    assert _resolve_num_frames_arg(args) == AutoDuration(min_seconds=2.0, max_seconds=10.0)


def test_cli_explicit_frames_and_auto_duration_warns_and_frames_wins(capsys):
    from ltx_pipelines_mlx.cli import _resolve_num_frames_arg

    args = _parse_generate_args("-f", "25", "--auto-duration", "2:10")
    resolved = _resolve_num_frames_arg(args)

    assert resolved == 25
    err = capsys.readouterr().err
    assert "--frames" in err and "--auto-duration" in err


def test_cli_auto_duration_rejects_malformed_value():
    from ltx_pipelines_mlx.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "-p",
                "x",
                "-o",
                "out.mp4",
                "--frame-rate",
                "24",
                "--distilled",
                "--auto-duration",
                "not-a-range",
            ]
        )


def test_cli_auto_duration_rejects_inverted_bounds(capsys):
    """Mirrors upstream AutoDurationAction: MIN must be <= MAX, checked at parse time.

    Without this guard, --auto-duration 10:2 would silently flow through
    (AutoDuration doesn't validate its own fields) and produce a
    wrong-duration render instead of a clear CLI error.
    """
    with pytest.raises(SystemExit):
        _parse_generate_args("--auto-duration", "10:2")

    err = capsys.readouterr().err
    assert "MIN_SECONDS (10.0) must be <= MAX_SECONDS (2.0)" in err


def test_cli_other_subcommands_keep_97_default():
    """keyframe/a2v/ic-lora keep their historical -f default untouched."""
    from ltx_pipelines_mlx.cli import _build_parser

    parser = _build_parser()
    kf_args = parser.parse_args(
        ["keyframe", "-p", "x", "-o", "out.mp4", "--frame-rate", "24", "--start", "a.png", "--end", "b.png"]
    )
    assert kf_args.frames == 97
    assert not hasattr(kf_args, "auto_duration")


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_duration_predictor_with_custom_bounds(tmp_path):
    """Predictor respects custom min/max_seconds bounds."""
    predictor = DurationPredictor.from_checkpoint(LTX25_Q8_DIR)
    video_encoding = mx.random.normal((1, 16, 4096))

    # Request 0.5s-1.0s duration (narrower than default 1s-20s)
    num_frames = predictor(video_encoding, None, frame_rate=24.0, min_seconds=0.5, max_seconds=1.0)

    assert isinstance(num_frames, int)
    # min_seconds=0.5 @ 24fps = 12 frames, max_frames=24
    # result should be between snapped(12) and snapped(24)
    assert (num_frames - 1) % 8 == 0
    assert num_frames >= 9  # Snapped up from 12
    assert num_frames <= 25  # Snapped up from 24
