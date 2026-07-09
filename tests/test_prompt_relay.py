"""Unit tests for Prompt Relay mask construction (ltx_core_mlx.conditioning.prompt_relay).

Pure-CPU / no model load — exercises token-range mapping, length distribution, and
the additive cross-attention penalty mask.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.conditioning.prompt_relay import (
    build_relay_mask,
    distribute_segment_lengths,
    map_token_ranges,
)
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock


class _FakeTokenizer:
    """Whitespace tokenizer: prepends a BOS id, one id per word. No EOS."""

    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        words = text.split()
        return [1] + [1000 + i for i, _ in enumerate(words)]


def test_map_token_ranges_basic():
    tok = _FakeTokenizer()
    combined, ranges = map_token_ranges(tok, "global prompt", ["red car", "blue sky"])
    # combined prompt is global + space-prefixed locals
    assert combined == "global prompt red car blue sky"
    # global = BOS + 2 words = 3 tokens; each local adds its words
    assert ranges == [(3, 5), (5, 7)]


def test_map_token_ranges_rejects_empty_local():
    tok = _FakeTokenizer()
    with pytest.raises(ValueError):
        map_token_ranges(tok, "global", ["   "])


def test_distribute_segment_lengths_auto_and_explicit():
    # auto: ceil(4/2)=2 each
    assert distribute_segment_lengths(2, 4) == [2, 2]
    # auto with clamp: ceil(5/2)=3 -> [3, 2] (second clamped to remaining)
    assert distribute_segment_lengths(2, 5) == [3, 2]
    # explicit passes through, clamped to timeline
    assert distribute_segment_lengths(3, 6, [2, 2, 10]) == [2, 2, 2]
    # mixed pinned + auto: pinned kept, leftover spread across the None beats
    assert distribute_segment_lengths(3, 12, [4, None, None]) == [4, 4, 4]
    assert distribute_segment_lengths(2, 10, [3, None]) == [3, 7]
    # pinned already fills the clip -> auto beats collapse to 0
    assert distribute_segment_lengths(2, 5, [5, None]) == [5, 0]
    with pytest.raises(ValueError):
        distribute_segment_lengths(2, 4, [1])  # count mismatch


def test_build_relay_mask_shape_and_dtype():
    mask = build_relay_mask(
        token_ranges=[(2, 3), (3, 4)],
        segment_lengths=[2, 2],
        num_video_tokens=4,
        tokens_per_frame=1,
        latent_frames=4,
        num_text_tokens=6,
    )
    assert mask.shape == (1, 1, 4, 6)
    assert mask.dtype == mx.bfloat16


def test_build_relay_mask_temporal_gating():
    # F=4 latent frames, 1 token/frame -> Nv=4; Nk=6.
    # seg0 tokens=[2,3), window centred on frame 1; seg1 tokens=[3,4), centred on frame 3.
    mask = np.array(
        build_relay_mask(
            token_ranges=[(2, 3), (3, 4)],
            segment_lengths=[2, 2],
            num_video_tokens=4,
            tokens_per_frame=1,
            latent_frames=4,
            num_text_tokens=6,
            dtype=mx.float32,
        )
    )[0, 0]

    # Global-prompt columns (0,1) and register/padding columns (4,5) are always free.
    assert np.all(mask[:, 0] == 0.0)
    assert np.all(mask[:, 1] == 0.0)
    assert np.all(mask[:, 4] == 0.0)
    assert np.all(mask[:, 5] == 0.0)

    # seg0's token (col 2): free at its midpoint frame (1), penalised far away (frame 3).
    assert mask[1, 2] == 0.0
    assert mask[3, 2] < 0.0
    # seg1's token (col 3): free at its midpoint frame (3), penalised at frame 0.
    assert mask[3, 3] == 0.0
    assert mask[0, 3] < 0.0


def test_build_relay_mask_appended_keyframe_rows_free():
    # Nv exceeds F*tokens_per_frame by 2 appended keyframe tokens -> those rows free.
    mask = np.array(
        build_relay_mask(
            token_ranges=[(2, 3)],
            segment_lengths=[2],
            num_video_tokens=6,  # 4 real + 2 appended
            tokens_per_frame=1,
            latent_frames=4,
            num_text_tokens=6,
            dtype=mx.float32,
        )
    )[0, 0]
    # Appended rows (4, 5) receive zero penalty everywhere.
    assert np.all(mask[4, :] == 0.0)
    assert np.all(mask[5, :] == 0.0)


def _make_block() -> BasicAVTransformerBlock:
    return BasicAVTransformerBlock(
        video_dim=32,
        audio_dim=16,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=4,
        av_cross_head_dim=4,
    )


def _block_kwargs(bsz: int, nv: int, nt: int):
    # Force a nonzero text-cross-attn residual gate (adaln chunk 8 = gate_ca);
    # otherwise the zero-init gate scales attn2's contribution to zero and the
    # cross-attention mask would be unobservable at the block output.
    video_adaln = mx.zeros((bsz, 9 * 32))
    video_adaln = video_adaln.at[:, 8 * 32 : 9 * 32].add(1.0)
    return dict(
        video_hidden=mx.random.normal((bsz, nv, 32)),
        audio_hidden=mx.random.normal((bsz, 4, 16)),
        video_adaln_params=video_adaln,
        audio_adaln_params=mx.zeros((bsz, 9 * 16)),
        video_prompt_adaln_params=mx.zeros((bsz, 2 * 32)),
        audio_prompt_adaln_params=mx.zeros((bsz, 2 * 16)),
        av_ca_video_params=mx.zeros((bsz, 4 * 32)),
        av_ca_audio_params=mx.zeros((bsz, 4 * 16)),
        av_ca_a2v_gate_params=mx.zeros((bsz, 32)),
        av_ca_v2a_gate_params=mx.zeros((bsz, 16)),
        video_text_embeds=mx.random.normal((bsz, nt, 32)),
        audio_text_embeds=mx.random.normal((bsz, nt, 16)),
    )


def test_cross_attention_mask_flows_through_block_and_changes_output():
    """The video_cross_attention_mask reaches attn2 and alters its contribution."""
    mx.random.seed(0)
    block = _make_block()
    B, nv, nt = 1, 8, 6
    kwargs = _block_kwargs(B, nv, nt)

    v_none, _ = block(**kwargs, video_cross_attention_mask=None)

    # Strong negative bias on the last 3 text columns for every video token.
    mask = mx.zeros((1, 1, nv, nt))
    mask = mask.at[:, :, :, 3:].add(-1e4)
    v_masked, _ = block(**kwargs, video_cross_attention_mask=mask)

    mx.synchronize()
    assert v_masked.shape == (B, nv, 32)
    # Masking a subset of text keys must change the block output.
    assert float(mx.max(mx.abs(v_masked - v_none))) > 1e-4


def test_guided_loop_masks_conditional_pass_only():
    """Under CFG, the relay mask reaches the conditional pass but never the uncond one."""
    from ltx_core_mlx.components.guiders import (
        MultiModalGuiderParams,
        create_multimodal_guider_factory,
    )
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState
    from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop

    B, nv, na, nt, c = 1, 4, 2, 6, 8
    v_state = LatentState(mx.zeros((B, nv, c)), mx.zeros((B, nv, c)), mx.ones((B, nv, 1)))
    a_state = LatentState(mx.zeros((B, na, c)), mx.zeros((B, na, c)), mx.ones((B, na, 1)))

    # cfg_scale != 1 -> unconditional pass runs; stg/modality off -> no ptb/mod passes.
    params = MultiModalGuiderParams(cfg_scale=2.0, stg_scale=0.0, modality_scale=1.0, stg_blocks=[])
    v_factory = create_multimodal_guider_factory(params, negative_context=mx.zeros((B, nt, c)))
    a_factory = create_multimodal_guider_factory(params, negative_context=mx.zeros((B, nt, c)))

    recorded: list = []

    class _Recorder:
        def __call__(self, **kw):
            recorded.append(kw.get("video_cross_attention_mask"))
            return kw["video_latent"], kw["audio_latent"]

    mask = mx.zeros((1, 1, nv, nt))
    guided_denoise_loop(
        model=_Recorder(),
        video_state=v_state,
        audio_state=a_state,
        video_text_embeds=mx.zeros((B, nt, c)),
        audio_text_embeds=mx.zeros((B, nt, c)),
        video_guider_factory=v_factory,
        audio_guider_factory=a_factory,
        sigmas=[1.0, 0.0],
        video_cross_attention_mask=mask,
        show_progress=False,
    )

    # Two passes: exactly one carries the mask (cond), one carries None (uncond).
    masked = [m for m in recorded if m is not None]
    unmasked = [m for m in recorded if m is None]
    assert len(masked) == 1, f"expected 1 masked pass, got {len(masked)} of {len(recorded)}"
    assert len(unmasked) == 1, f"expected 1 unmasked pass, got {len(unmasked)}"
