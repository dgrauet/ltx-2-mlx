"""res_2s must never re-noise the tokens its own per-token timesteps call clean.

`res2s_denoise_loop` is the stage-1 sampler `--two-stages-hq` uses. It injects SDE
noise at BOTH the substep and the step level (`_sde_step`, sigma->sub_sigma and
sigma->sigma_next) over the WHOLE token sequence, pinned conditioning block included,
and did not re-apply the conditioning mask afterwards. A preserved token
(`denoise_mask == 0`) is handed `clean_latent` by `_predict`, and
`_compute_per_token_timesteps` stamps it timestep 0 = clean — so noise landing on it
is a contradiction the model resolves by inventing content. Anchored I2V/fl2v/l2v
renders drift to the prompt's prior between the pinned ends, and the end anchor lands
as a snap rather than a landing.

`guided_denoise_loop` (`--two-stage`, a2v, retake) is deterministic Euler and
never dirties them; `euler_ancestral_denoising_loop` (distilled) re-pins after its
noise draw and says why in a comment. res_2s did neither — the fix is the same
`apply_denoise_mask` call the other two loops already make, after each `_sde_step`.

Harness note, learned the hard way: the loop calls `mx.random.seed(step_idx * 10000 +
2)` INSIDE every step, so a seed set before the call never reaches the noise it draws.
A "two different draws, pinned tokens must not move" test therefore passes on the buggy
loop and proves nothing — it was written here, and deleted. The end-state comparison
against `clean_latent` is the honest form of the check.
"""

from __future__ import annotations

import inspect

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop, res2s_denoise_loop

#: Tokens at the head of the sequence that conditioning preserves. Any count works;
#: 8 keeps the run cheap and the failure impossible to miss.
PINNED = 8
#: 4 outer steps, no terminal 0 — with a terminal 0 the loop's final `_predict`
#: re-pins everything and the defect is hidden by the last step instead of fixed.
SIGMAS = [1.0, 0.7, 0.4, 0.1, 0.05]


def _tiny_x0_model() -> X0Model:
    """The same 2-layer toy transformer the TeaCache hook tests build."""
    return X0Model(
        LTXModel(
            LTXModelConfig(
                num_layers=2,
                video_dim=32,
                audio_dim=16,
                video_num_heads=4,
                audio_num_heads=4,
                video_head_dim=8,
                audio_head_dim=4,
                av_cross_num_heads=4,
                av_cross_head_dim=4,
            )
        )
    )


def _cfg_factory(batch: int, dim: int) -> MultiModalGuiderFactory:
    """CFG-only guider factory (no STG, no modality)."""
    params = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=0.0,
        rescale_scale=0.7,
        modality_scale=1.0,
        stg_blocks=[],
    )
    neg = mx.zeros((batch, 4, dim), dtype=mx.bfloat16)
    return create_multimodal_guider_factory(params, negative_context=neg)


def _state(batch: int, n_tokens: int, dim: int, *, pinned: int) -> LatentState:
    """First ``pinned`` tokens preserved (mask 0, clean value carried), rest generated
    (mask 1, noisy start). Not seeded here — the caller seeds once per run."""
    clean = mx.random.normal((batch, n_tokens, dim)).astype(mx.bfloat16)
    generated = mx.random.normal((batch, n_tokens, dim)).astype(mx.bfloat16)
    mask = mx.ones((batch, n_tokens, 1), dtype=mx.bfloat16)
    mask[:, :pinned] = 0.0
    # Exactly what the conditioner leaves behind: preserved slots hold clean_latent.
    latent = mx.where(mask == 0, clean, generated)
    return LatentState(latent=latent, clean_latent=clean, denoise_mask=mask)


def _run(loop=res2s_denoise_loop):
    B, Nv, Na = 1, 16, 6
    mx.random.seed(7)
    video_state = _state(B, Nv, 128, pinned=PINNED)
    audio_state = _state(B, Na, 128, pinned=PINNED)  # audio latents are 128-ch too
    kwargs = dict(
        model=_tiny_x0_model(),
        video_state=video_state,
        audio_state=audio_state,
        video_text_embeds=mx.zeros((B, 4, 32), dtype=mx.bfloat16),
        audio_text_embeds=mx.zeros((B, 4, 16), dtype=mx.bfloat16),
        video_guider_factory=_cfg_factory(B, 32),
        audio_guider_factory=_cfg_factory(B, 16),
        sigmas=list(SIGMAS),
        show_progress=False,
    )
    if loop is res2s_denoise_loop:
        kwargs["bongmath"] = False  # anchor refinement is irrelevant to the mask contract
    out = loop(**kwargs)
    mx.eval(out.video_latent, out.audio_latent)
    return out, video_state, audio_state


class TestRes2sPreservesConditioning:
    def test_pinned_video_tokens_survive_the_loop(self):
        """mask == 0 in, mask == 0 out — bit-identical to ``clean_latent``.

        Fails on an unpatched loop: the step-level `_sde_step` leaves
        `alpha_ratio * clean + sigma_up * noise` at those positions, so the clip
        carries noise at tokens its own timesteps call clean.
        """
        out, video_state, _ = _run()
        got = out.video_latent[:, :PINNED].astype(mx.float32)
        want = video_state.clean_latent[:, :PINNED].astype(mx.float32)
        drift = float(mx.mean(mx.abs(got - want)).item())
        assert drift == 0.0, (
            f"res_2s re-noised {PINNED} preserved video tokens (mean |delta| {drift:.4g}); "
            "_compute_per_token_timesteps stamps them timestep 0, so the loop handed the "
            "model noise at positions it told the model were clean"
        )

    def test_pinned_audio_tokens_survive_the_loop(self):
        """The audio half has the same two `_sde_step` calls and needs the same guard."""
        out, _, audio_state = _run()
        got = out.audio_latent[:, :PINNED].astype(mx.float32)
        want = audio_state.clean_latent[:, :PINNED].astype(mx.float32)
        assert float(mx.mean(mx.abs(got - want)).item()) == 0.0

    def test_generated_tokens_actually_moved(self):
        """Anti-vacuous: if the loop were a no-op, the two tests above pass for free."""
        out, video_state, _ = _run()
        moved = out.video_latent[:, PINNED:].astype(mx.float32) - video_state.latent[:, PINNED:].astype(mx.float32)
        assert float(mx.mean(mx.abs(moved)).item()) > 0.0

    def test_guided_loop_holds_the_same_contract(self):
        """`guided_denoise_loop` — what `--two-stage`, a2v and retake run —
        already preserves pinned tokens, because its Euler step is deterministic. This
        test passes before AND after the patch on purpose: it pins the invariant
        res_2s is expected to match, so a later refactor of EITHER loop that breaks
        the contract fails here and names which loop moved.
        """
        out, video_state, _ = _run(loop=guided_denoise_loop)
        got = out.video_latent[:, :PINNED].astype(mx.float32)
        want = video_state.clean_latent[:, :PINNED].astype(mx.float32)
        assert float(mx.mean(mx.abs(got - want)).item()) == 0.0

    def test_the_loop_still_self_seeds(self):
        """Guard for the harness limitation in the module docstring: the loop seeds
        the global RNG per step (`mx.random.seed(step_idx * ...)`), so noise is fixed
        across runs. If that ever stops being true, the end-state comparisons above
        need a per-seed repeat — this test is the tripwire that says so.
        """
        assert "mx.random.seed(step_idx" in inspect.getsource(res2s_denoise_loop)
