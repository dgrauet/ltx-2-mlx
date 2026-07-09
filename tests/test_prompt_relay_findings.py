"""Regression tests for the PR #61 review findings (temporal prompt gating).

Each test encodes the *expected* (correct) behavior after the fixes: an out-of-band
condition (zero-length beat, overflowing token range, out-of-range epsilon,
combined-prompt truncation) must be rejected loudly rather than silently mis-gated.
Pure-CPU, no model.
"""

from __future__ import annotations

import numpy as np
import pytest

from ltx_core_mlx.conditioning.prompt_relay import (
    build_relay_mask,
    distribute_segment_lengths,
    map_token_ranges,
)


class _FakeTokenizer:
    """Whitespace tokenizer: prepends a BOS id, one id per word. No EOS.

    Mirrors tests/test_prompt_relay.py's fixture (and Gemma's convention).
    """

    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        words = text.split()
        return [1] + [1000 + i for i, _ in enumerate(words)]


def _mask_np(**kw) -> np.ndarray:
    """build_relay_mask -> squeezed float32 numpy (Nv, Nk)."""
    import mlx.core as mx

    defaults = dict(
        num_video_tokens=kw.pop("nv"),
        tokens_per_frame=kw.pop("tpf"),
        latent_frames=kw.pop("frames"),
        num_text_tokens=kw.pop("nk"),
    )
    m = build_relay_mask(dtype=mx.float32, **defaults, **kw)
    return np.array(m)[0, 0]


# ---------------------------------------------------------------------------
# Finding 1 — zero-length segments must not silently attend to ALL frames
# ---------------------------------------------------------------------------


class TestZeroLengthSegmentUngated:
    def test_pinned_fills_clip_second_segment_rejected(self):
        """--segment "a" 5 --segment "b" on a 5-frame clip -> lengths [5, 0]."""
        lengths = distribute_segment_lengths(2, 5, [5, None])
        assert lengths == [5, 0]  # distribute still clamps (existing behavior)
        with pytest.raises(ValueError):
            _mask_np(
                token_ranges=[(3, 5), (5, 7)],
                segment_lengths=lengths,
                nv=5,
                tpf=1,
                frames=5,
                nk=16,
            )

    def test_auto_split_more_segments_than_it_can_fit(self):
        """3 auto segments over 4 frames -> [2, 2, 0]; the collapsed beat is rejected."""
        lengths = distribute_segment_lengths(3, 4)
        assert lengths == [2, 2, 0]
        with pytest.raises(ValueError):
            _mask_np(
                token_ranges=[(3, 5), (5, 7), (7, 9)],
                segment_lengths=lengths,
                nv=4,
                tpf=1,
                frames=4,
                nk=16,
            )


# ---------------------------------------------------------------------------
# Finding 2 — token ranges past num_text_tokens must not silently clip
# ---------------------------------------------------------------------------


class TestTokenRangeOverflow:
    def test_range_beyond_text_axis_rejected(self):
        with pytest.raises(ValueError):
            _mask_np(
                token_ranges=[(20, 30)],  # entirely past Nk=16
                segment_lengths=[4],
                nv=8,
                tpf=1,
                frames=8,
                nk=16,
            )

    def test_map_token_ranges_rejects_truncating_prompt(self):
        """A combined prompt longer than max_length is rejected (left-truncation
        would shift every surviving column)."""
        tok = _FakeTokenizer()
        global_prompt = " ".join(f"g{i}" for i in range(10))  # 10 words + BOS = 11 > 8
        with pytest.raises(ValueError):
            map_token_ranges(tok, global_prompt, ["red car"], max_length=8)

    def test_map_token_ranges_within_limit_ok(self):
        """Control: a combined prompt under the limit maps normally."""
        tok = _FakeTokenizer()
        combined, ranges = map_token_ranges(tok, "global", ["red car"], max_length=64)
        assert combined == "global red car"
        assert ranges == [(2, 4)]


# ---------------------------------------------------------------------------
# Finding 3 — odd-length segments keep their free anchor frame (floor midpoint)
# ---------------------------------------------------------------------------


class TestOddSegmentMidpointParity:
    def test_odd_length_segment_keeps_a_zero_penalty_anchor_frame(self):
        mask = _mask_np(
            token_ranges=[(3, 5)],
            segment_lengths=[3],
            nv=3,
            tpf=1,
            frames=3,
            nk=8,
        )
        seg_col = mask[:, 3]  # per-frame penalty for the segment's first token
        assert np.any(seg_col == 0.0), (
            f"length-3 segment has no zero-penalty frame (per-frame bias: "
            f"{seg_col.tolist()}); reference floor-divides the midpoint"
        )

    def test_even_length_segment_unaffected_control(self):
        mask = _mask_np(
            token_ranges=[(3, 5)],
            segment_lengths=[2],
            nv=2,
            tpf=1,
            frames=2,
            nk=8,
        )
        assert mask[1, 3] == 0.0  # frame 1 is the (floored) midpoint: free


# ---------------------------------------------------------------------------
# Finding 4 — out-of-range epsilon must be rejected, not silently defaulted
# ---------------------------------------------------------------------------


class TestEpsilonNotSilentlyDiscarded:
    def _mask_for_eps(self, eps: float) -> np.ndarray:
        return _mask_np(
            token_ranges=[(3, 5)],
            segment_lengths=[2],
            nv=6,
            tpf=1,
            frames=6,
            nk=8,
            epsilon=eps,
        )

    def test_epsilon_one_rejected(self):
        with pytest.raises(ValueError):
            self._mask_for_eps(1.0)  # sigma -> inf; must not silently sharpen

    def test_negative_epsilon_rejected(self):
        with pytest.raises(ValueError):
            self._mask_for_eps(-0.5)

    def test_in_range_epsilon_ok(self):
        """Control: the documented default stays valid."""
        m = self._mask_for_eps(1e-3)
        assert m.shape == (6, 8)
