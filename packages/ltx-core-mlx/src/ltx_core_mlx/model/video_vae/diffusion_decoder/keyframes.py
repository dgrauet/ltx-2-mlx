"""Keyframe (dual-stream) inputs and coordinate math for the keyframe-aware diffusion decode.

A keyframe-aware decode carries two streams through the decoder: the video volume ``(B, T, H, W, C)``
and a stack of keyframe *planes* ``(B, P, H, W, C)`` whose plane axis occupies the temporal slot.
Weights are fully shared; the streams only mix inside the joint attention softmax
(:func:`~ltx_core_mlx.model.video_vae.diffusion_decoder.neighborhood_attention.joint_na3d`).
Everything here is pure geometry (upstream ``keyframes.py``); the slot tables are host-side numpy
so they never enter the lazy graph.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.video_vae.diffusion_decoder.config import DiffusionDecoderConfig
from ltx_core_mlx.model.video_vae.diffusion_decoder.layers import LinearPixelShuffleUpsample

#: Keyframe planes visible to one video query (and video frames visible to one plane query).
KEYFRAME_CONTEXT_SLOTS = 2


@dataclass(frozen=True)
class DecodeKeyframes:
    """Caller-facing keyframe input to a diffusion decode.

    Attributes:
        latents: ``(B, C, P, H, W)`` per-channel-normalised latents, one latent frame per plane,
            each encoded (or generated) as a standalone one-pixel-frame clip.
        pixel_frame_indices: Global pixel frame index of each plane; never rebased onto a tile.
        clip_start_frame: First global pixel frame of the video latent in this decode (0 for a
            whole-clip decode).
    """

    latents: mx.array
    pixel_frame_indices: tuple[int, ...]
    clip_start_frame: int = 0

    def validate(self, *, num_frames: int | None = None) -> None:
        """Raise ``ValueError`` when shapes / indices are inconsistent (optionally against a frame count)."""
        if self.latents.ndim != 5:
            raise ValueError(f"keyframe latents must be (B, C, P, H, W), got {tuple(self.latents.shape)}")
        if self.clip_start_frame < 0:
            raise ValueError(f"clip_start_frame must be non-negative, got {self.clip_start_frame}")
        planes = self.latents.shape[2]
        if planes == 0:
            raise ValueError("keyframe decode needs at least one plane; decode without keyframes instead")
        if planes != len(self.pixel_frame_indices):
            raise ValueError(
                f"keyframe plane count {planes} != len(pixel_frame_indices) {len(self.pixel_frame_indices)}"
            )
        if min(self.pixel_frame_indices) < 0:
            raise ValueError("pixel_frame_indices must be non-negative (global pixel frames)")
        if num_frames is not None and num_frames < 1:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

    def for_frame_span(self, frame_lo: int, frame_hi: int) -> DecodeKeyframes:
        """Keep the planes a decode of pixel frames ``[lo, hi]`` needs; indices stay global."""
        keep = planes_for_tile(self.pixel_frame_indices, frame_lo, frame_hi)
        idx = [i for i, k in enumerate(keep) if k]
        return DecodeKeyframes(
            latents=self.latents[:, :, idx],
            pixel_frame_indices=tuple(self.pixel_frame_indices[i] for i in idx),
            clip_start_frame=frame_lo,
        )

    def crop_spatial(self, height: slice, width: slice) -> DecodeKeyframes:
        """Crop the planes with the same latent slices the video used."""
        return dataclasses.replace(self, latents=self.latents[:, :, :, height, width])

    @property
    def num_planes(self) -> int:
        return int(self.latents.shape[2])


@dataclass(frozen=True)
class KeyframeStream:
    """The keyframe half of the dual stream at one decoder stage.

    Attributes:
        x: ``(B, P, H, W, C)`` channels-last activations; ``H``/``W`` match the video stream.
        times: ``(P,)`` float32 plane position in this stage's temporal units, tile-local origin.
        valid: ``(P,)`` bool; invalid planes are masked out of every softmax.
    """

    x: mx.array
    times: mx.array
    valid: mx.array

    def masked(self) -> KeyframeStream:
        """Re-zero invalid planes' activations."""
        return dataclasses.replace(self, x=self.x * self.valid[None, :, None, None, None].astype(self.x.dtype))

    def select_planes(self, keep: Sequence[bool]) -> KeyframeStream:
        """Subset the plane axis, keeping ``x`` / ``times`` / ``valid`` in step."""
        if len(keep) != self.num_planes:
            raise ValueError(f"keep must have {self.num_planes} entries, got {len(keep)}")
        idx = [i for i, k in enumerate(keep) if k]
        return KeyframeStream(x=self.x[:, idx], times=self.times[idx], valid=self.valid[idx])

    def crop_spatial(self, height: slice, width: slice) -> KeyframeStream:
        """Crop H/W with the same slices the video stream's tile used."""
        return dataclasses.replace(self, x=self.x[:, :, height, width, :])

    @property
    def num_planes(self) -> int:
        return int(self.x.shape[1])


def _stage_times_np(pixel_frame_indices: Sequence[int], remaining_time_stride: int) -> np.ndarray:
    if remaining_time_stride < 1:
        raise ValueError(f"remaining_time_stride must be positive, got {remaining_time_stride}")
    frames = np.asarray(pixel_frame_indices, dtype=np.float32)
    times = (frames + np.float32((remaining_time_stride - 1) / 2)) / np.float32(remaining_time_stride)
    return np.where(frames == 0, np.float32(0.0), times).astype(np.float32)


def keyframe_stage_times(pixel_frame_indices: Sequence[int], remaining_time_stride: int) -> mx.array:
    """Chunk-centre position of each plane in a stage's temporal units: ``t_s(f) = (f + (r-1)/2) / r``, ``t_s(0) = 0``."""
    return mx.array(_stage_times_np(pixel_frame_indices, remaining_time_stride))


def keyframe_clip_times(
    pixel_frame_indices: Sequence[int],
    remaining_time_stride: int,
    clip_start_frame: int,
    extra_origin: float = 0.0,
) -> mx.array:
    """Stage times relative to a decode whose first pixel frame is ``clip_start_frame``: ``t_s(idx) - t_s(start) - extra``."""
    times = _stage_times_np(pixel_frame_indices, remaining_time_stride)
    origin = _stage_times_np((clip_start_frame,), remaining_time_stride)[0]
    return mx.array((times - origin - np.float32(extra_origin)).astype(np.float32))


def planes_for_tile(
    pixel_frame_indices: Sequence[int], frame_lo: int, frame_hi: int, *, clip_start_frame: int = 0
) -> list[bool]:
    """Which planes a tile spanning pixel frames ``[lo, hi]`` (inclusive, relative to ``clip_start_frame``) carries.

    Every plane inside the span plus the nearest plane on each side outside it, so a frame at a
    tile edge ranks the same anchors as in a whole-clip decode. Selection is by value.
    """
    lo, hi = frame_lo + clip_start_frame, frame_hi + clip_start_frame
    idx = list(pixel_frame_indices)
    keep = [lo <= f <= hi for f in idx]
    before = [(f, i) for i, f in enumerate(idx) if f < lo]
    if before:
        keep[max(before)[1]] = True
    after = [(f, i) for i, f in enumerate(idx) if f > hi]
    if after:
        keep[min(after)[1]] = True
    return keep


def remaining_time_strides(config: DiffusionDecoderConfig) -> tuple[int, ...]:
    """Remaining temporal upsampling at each stage input, plus 1 for stage 5 (production ``(8, 8, 4, 2, 1)``)."""
    strides = [stride[0] for stride, _ in config.upsamples]
    out = []
    for i in range(len(strides)):
        product = 1
        for s in strides[i:]:
            product *= s
        out.append(product)
    out.append(1)
    return tuple(out)


def upsample_keyframe_planes(upsample: LinearPixelShuffleUpsample, x: mx.array) -> mx.array:
    """Spatially upsample ``(B, P, H, W, C)`` planes through the video upsample, keeping ``P`` invariant.

    Each plane is folded into the batch as a ``T = 1`` clip with ``drop_leading_frame=True``: a
    temporal stride of 2 expands ``T`` to 2 and the drop takes it back to 1, so only H/W grow.
    """
    b, p, h, w, c = x.shape
    up = upsample(x.reshape(b * p, 1, h, w, c), drop_leading_frame=True)
    if up.shape[1] != 1:
        raise RuntimeError(f"isolated keyframe upsampling must preserve one temporal plane, got T={up.shape[1]}")
    return up[:, 0].reshape(b, p, up.shape[2], up.shape[3], up.shape[4])


def _nearest_slots(
    query_times: np.ndarray, candidate_times: np.ndarray, candidate_valid: np.ndarray | None, num_slots: int
) -> np.ndarray:
    """``(Q, num_slots)`` candidate indices ranked by ``(|dt|, index)``; ``-1`` where no candidate is left."""
    d = np.abs(query_times[:, None].astype(np.float32) - candidate_times[None, :].astype(np.float32))
    if candidate_valid is not None:
        d = np.where(candidate_valid[None, :], d, np.float32(np.inf))
    order = np.argsort(d, axis=1, kind="stable")
    take = min(num_slots, candidate_times.shape[0])
    chosen = order[:, :take]
    finite = np.isfinite(np.take_along_axis(d, chosen, axis=1))
    chosen = np.where(finite, chosen, -1)
    if take < num_slots:
        chosen = np.concatenate([chosen, np.full((chosen.shape[0], num_slots - take), -1)], axis=1)
    return chosen.astype(np.int32)


def video_keyframe_slots(
    keyframe_times: np.ndarray, keyframe_valid: np.ndarray, video_length: int, num_slots: int = KEYFRAME_CONTEXT_SLOTS
) -> np.ndarray:
    """``(T, num_slots)`` plane index per video frame, ranked by ``(|t_s(plane) - t|, plane)``; independent of ``K_t``."""
    query = np.arange(video_length, dtype=np.float32)
    return _nearest_slots(
        query, np.asarray(keyframe_times, dtype=np.float32), np.asarray(keyframe_valid, dtype=bool), num_slots
    )


def keyframe_video_slots(
    keyframe_times: np.ndarray, keyframe_valid: np.ndarray, video_length: int, num_slots: int = KEYFRAME_CONTEXT_SLOTS
) -> np.ndarray:
    """``(P, num_slots)`` video frame index per plane, ranked by ``(|t' - t_s(plane)|, t')``; invalid planes -> ``-1`` rows."""
    candidates = np.arange(video_length, dtype=np.float32)
    slots = _nearest_slots(np.asarray(keyframe_times, dtype=np.float32), candidates, None, num_slots)
    valid = np.asarray(keyframe_valid, dtype=bool)
    return np.where(valid[:, None], slots, -1).astype(np.int32)
