"""Pipeline-level helpers matching upstream ltx-pipelines/utils/helpers.py.

These functions form the standard orchestration vocabulary for
building a noised LatentState from conditionings + an optional
initial latent, and for blending denoised model output with the
clean-latent reference per the denoise mask.

They mirror the upstream LTX-2 helpers 1:1, allowing pipeline code
to be read side-by-side with the upstream Python files.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import first_frame_keyframes_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, noise_latent_state

if TYPE_CHECKING:
    pass


def post_process_latent(
    denoised: mx.array,
    denoise_mask: mx.array,
    clean: mx.array,
) -> mx.array:
    """Blend denoised model output with clean state based on mask.

    Mirror of upstream ``ltx_pipelines.utils.helpers.post_process_latent``.
    For tokens with ``mask=1`` (generation), the predicted ``denoised``
    is used as-is. For tokens with ``mask=0`` (preserved keyframe /
    reference), ``clean`` is restored.

    Upstream upcasts ``clean`` to fp32 before the blend then casts back
    to ``denoised.dtype`` to reduce bf16 accumulation noise.

    Args:
        denoised: Model x0 prediction, shape ``(B, T, C)``.
        denoise_mask: Mask in ``[0, 1]``, shape ``(B, T, 1)``.
        clean: Clean latent reference, shape ``(B, T, C)``.

    Returns:
        Blended latent, same shape and dtype as ``denoised``.
    """
    out_dtype = denoised.dtype
    blended = denoised * denoise_mask + clean.astype(mx.float32) * (1.0 - denoise_mask)
    return blended.astype(out_dtype)


def state_with_conditionings(
    latent_state: LatentState,
    conditioning_items: list,
    spatial_dims: tuple[int, int, int],
) -> LatentState:
    """Apply a list of conditionings sequentially to a latent state.

    Mirror of upstream ``ltx_pipelines.utils.helpers.state_with_conditionings``.
    Iterates through the conditioning items and applies each one to the
    state. Each conditioning may append tokens (keyframe / reference)
    or modify the denoise mask (temporal region).

    Note: the upstream signature passes ``latent_tools`` as the second
    arg; in MLX our conditionings take ``spatial_dims`` directly via
    their ``apply(state, spatial_dims)`` signature, since we don't have
    the LatentTools abstraction yet.

    Args:
        latent_state: Starting state.
        conditioning_items: List with ``apply(state, spatial_dims)``
            method (e.g. ``VideoConditionByKeyframeIndex``).
        spatial_dims: ``(F, H, W)`` latent spatial dimensions.

    Returns:
        State with all conditionings applied in order.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply(latent_state, spatial_dims)
    return latent_state


def create_noised_state(
    base_shape: tuple[int, ...],
    conditionings: list,
    spatial_dims: tuple[int, int, int],
    positions: mx.array,
    seed: int,
    sigma: float = 1.0,
    initial_latent: mx.array | None = None,
    dtype: mx.Dtype = mx.bfloat16,
    legacy_scalar_blend: bool = False,
) -> LatentState:
    """Build a noised latent state from conditionings + optional initial latent.

    Mirror of upstream ``ltx_pipelines.utils.helpers.create_noised_state``.

    Order of operations (CRITICAL — matches upstream):
        1. Initial state: zero-init OR ``initial_latent`` if provided.
        2. Apply conditionings in order (may append tokens, set mask=0).
        3. Noise: ``GaussianNoiser`` semantics — keyframe tokens
           (mask=0) stay clean, generation tokens (mask=1) get
           ``noise * sigma + clean * (1 - sigma)`` blend.

    Args:
        base_shape: ``(B, N_gen, C)`` shape of the generation tokens
            BEFORE conditioning items append any reference tokens.
        conditionings: List of conditioning items (e.g.
            ``VideoConditionByKeyframeIndex``) to apply.
        spatial_dims: ``(F, H, W)`` latent spatial dims for conditioning.
        positions: ``(1, N_gen, num_axes)`` positions for the
            generation tokens. Conditioning items append their own
            positions for any tokens they introduce.
        seed: Random seed for the noise.
        sigma: Noise scale. ``1.0`` for stage 1 (pure noise on
            generated tokens); ``stage_2_sigmas[0]`` for stage 2
            (partial noise on top of ``initial_latent``).
        initial_latent: Optional starting latent for the generation
            region. ``None`` means start from zeros (typical stage 1).
            Provided as the upscaled stage-1 latent for stage 2.
        dtype: dtype of the resulting state arrays.
        legacy_scalar_blend: When True, apply scalar-sigma noise blend
            BEFORE conditionings (matches the legacy
            ``noise * sigma + clean * (1 - sigma)`` inline arithmetic
            with Python literals — bf16 cannot represent 0.05 exactly,
            so going through ``noise_latent_state``'s ``mask * sigma``
            blend introduces a ~3e-3 error per element that compounds
            across denoise steps and breaks bit-equivalence with
            pre-Phase-3 baselines). Set True at video Stage 1 + Stage 2
            callsites in ti2vid_two_stages / ti2vid_two_stages_hq /
            a2vid_two_stage / ic_lora. Default False matches the
            standard helper-style flow (init → cond → mask-aware
            noise) and preserves bit-equivalence for callsites whose
            legacy code went through ``noise_latent_state`` (audio
            Stage 2, keyframe and reference-append conditionings).

    Returns:
        Noised LatentState ready to feed into the denoising loop.
    """
    if initial_latent is None:
        latent = mx.zeros(base_shape, dtype=dtype)
    else:
        # Preserve initial_latent dtype. Stage-2 upsampler output is
        # fp32 (normalize_latent uses fp32 per-channel stats); casting
        # to bf16 here would silently lose ~16 bits of mantissa and
        # drift across denoise steps.
        latent = initial_latent

    denoise_mask = mx.ones((base_shape[0], base_shape[1], 1), dtype=dtype)
    # Video states (3 position axes) mark their first latent frame as a
    # single-pixel-frame token class, exactly like upstream
    # ``VideoLatentTools.create_initial_state``; audio states carry no marker.
    _, spatial_h, spatial_w = spatial_dims
    keyframes_mask = (
        first_frame_keyframes_mask(denoise_mask, spatial_h * spatial_w) if positions.shape[-1] == 3 else None
    )
    state = LatentState(
        latent=latent,
        clean_latent=latent,
        denoise_mask=denoise_mask,
        positions=positions,
        keyframes_mask=keyframes_mask,
    )

    if legacy_scalar_blend:
        # Apply scalar noise blend BEFORE conditionings, with Python
        # scalar sigma (no bf16 mask quantization). Then conditionings
        # may overwrite individual tokens (LatentIndex replace) or
        # append new ones (Keyframe / Reference) — they set their own
        # mask=0 at the affected tokens, exactly as the legacy
        # ``inline noise+blend`` → ``apply_conditioning`` flow did.
        mx.random.seed(seed)
        noise = mx.random.normal(state.clean_latent.shape).astype(mx.bfloat16)
        blended = noise * sigma + state.clean_latent * (1.0 - sigma)
        state = replace(state, latent=blended)
        state = state_with_conditionings(state, conditionings, spatial_dims)
        return _noise_generated_keyframe_slots(state, sigma, seed)

    state = state_with_conditionings(state, conditionings, spatial_dims)
    return noise_latent_state(state, sigma=sigma, seed=seed)


#: Decorrelates the slot noise draw from the main latent draw (which reuses ``seed``).
GENERATED_KEYFRAME_NOISE_SEED_OFFSET = 20000


def _noise_generated_keyframe_slots(state: LatentState, sigma: float, seed: int) -> LatentState:
    """Noise generated keyframe slots appended *after* the legacy scalar blend.

    Upstream noises the whole sequence after conditioning, so slots (zero latent,
    ``denoise_mask=1``) come out as ``noise * sigma + latent * (1 - sigma)``. The legacy
    path blends before conditioning for bit-equivalence with older baselines, which would
    leave slots at their seed value; this applies the same blend to the slot range only.
    """
    layout = state.generated_keyframe_layout
    if layout is None:
        return state
    mx.random.seed(seed + GENERATED_KEYFRAME_NOISE_SEED_OFFSET)
    slot = state.latent[:, layout.token_slice]
    noise = mx.random.normal(slot.shape).astype(slot.dtype)
    blended = noise * sigma + slot * (1.0 - sigma)
    latent = mx.concatenate(
        [state.latent[:, : layout.first_token], blended, state.latent[:, layout.first_token + layout.num_tokens :]],
        axis=1,
    )
    return replace(state, latent=latent)


def evenly_spaced_keyframe_positions(num_keyframes: int, num_frames: int) -> list[int]:
    """Interior pixel-frame positions for ``num_keyframes`` generated keyframes, endpoints excluded.

    Mirror of upstream: ``linspace(0, num_frames - 1, num_keyframes + 2)`` rounded
    half-to-even (torch.round semantics), computed in float32 like torch's linspace.
    """
    import numpy as np

    if num_keyframes < 0:
        raise ValueError(f"num_keyframes must be non-negative, got {num_keyframes}")
    if num_keyframes == 0:
        return []
    if num_frames < num_keyframes + 2:
        raise ValueError(
            f"Generated keyframes need at least num_keyframes + 2 target frames, got "
            f"num_keyframes={num_keyframes}, num_frames={num_frames}"
        )
    grid = np.linspace(0, num_frames - 1, num_keyframes + 2, dtype=np.float32)
    return [int(v) for v in np.rint(grid).astype(np.int64).tolist()[1:-1]]


def has_generated_keyframes(generated_keyframes: int | Sequence[int]) -> bool:
    """Whether a ``generated_keyframes`` request asks for any slots (never test truthiness of an array)."""
    if isinstance(generated_keyframes, int):
        return generated_keyframes > 0
    return len(generated_keyframes) > 0


def resolve_generated_keyframes(generated_keyframes: int | Sequence[int], num_frames: int) -> list[int]:
    """Normalize the pipeline-level ``generated_keyframes`` argument to sorted pixel-frame indices.

    An ``int`` requests that many evenly spaced interior keyframes; a sequence gives the target
    pixel-frame indices explicitly. ``0`` / empty means the feature is off.
    """
    if isinstance(generated_keyframes, int):
        return evenly_spaced_keyframe_positions(generated_keyframes, num_frames)
    positions = sorted({int(p) for p in generated_keyframes})
    if positions and (positions[0] < 0 or positions[-1] >= num_frames):
        raise ValueError(
            f"Generated keyframe positions must lie in [0, {num_frames}), got {sorted(int(p) for p in generated_keyframes)}"
        )
    return positions


def generated_keyframe_conditionings(
    generated_keyframes: int | Sequence[int],
    num_frames: int,
    *,
    frame_rate: float,
) -> list:
    """Build the generated-keyframe conditioning, or an empty list when the feature is off.

    All slots go into one :class:`VideoGeneratedKeyframeSlots` so their tokens form a single
    contiguous, exactly-locatable range.
    """
    from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots

    positions = resolve_generated_keyframes(generated_keyframes, num_frames)
    if not positions:
        return []
    return [VideoGeneratedKeyframeSlots(pixel_frame_indices=positions, frame_rate=frame_rate)]
