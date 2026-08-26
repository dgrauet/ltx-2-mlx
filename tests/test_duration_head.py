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
