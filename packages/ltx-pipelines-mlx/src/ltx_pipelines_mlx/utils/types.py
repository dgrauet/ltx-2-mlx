"""Pipeline-level types matching upstream ltx-pipelines/utils/types.py.

These dataclasses and protocols form the orchestration vocabulary used
by ``DiffusionStage`` and the helper functions in
``ltx_pipelines_mlx.utils.helpers``. They mirror the upstream LTX-2
abstractions 1:1 so that pipeline code can be read side-by-side with
the upstream Python files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx

if TYPE_CHECKING:
    from ltx_core_mlx.conditioning.types.latent_cond import LatentState


@dataclass(frozen=True)
class AutoDuration:
    """Request auto-predicted duration, clamped to ``[min_seconds, max_seconds]``.

    The predicted duration is converted to a frame count snapped to the VAE's causal
    temporal grid (``8k + 1``). Defaults: 1s and 20s.
    """

    min_seconds: float = 1.0
    max_seconds: float = 20.0


DEFAULT_AUTO_DURATION = AutoDuration()

#: Pixel frames per latent frame of the causal video VAE (upstream ``VIDEO_SCALE_FACTORS.time``).
_VIDEO_TEMPORAL_SCALE = 8


@dataclass(frozen=True)
class ImageReplace:
    """Pin a still by replacing the latent that starts at a pixel-frame index.

    ``frame_idx`` is zero-based on the decoded video timeline, not a latent index. Because the
    video VAE is causal, valid values are frame 0 and the first pixel frame of each later latent:
    ``1 + 8 * k`` (1, 9, 17, ...). Mirrors upstream ``ltx_pipelines.utils.types.ImageReplace``.

    Local extension: a negative ``frame_idx`` counts back from the end of the clip and is
    validated once :func:`~ltx_pipelines_mlx.utils.args.resolve_frame_indices` knows the frame
    count (upstream rejects negative indices).
    """

    frame_idx: int

    def __post_init__(self) -> None:
        if self.frame_idx >= 0:
            _assert_replacement_frame_idx(self.frame_idx)


@dataclass(frozen=True)
class ImageGuide:
    """Guide the clip with a keyframe at a zero-based decoded pixel-frame index."""

    frame_idx: int


ImageBinding = ImageReplace | ImageGuide


@dataclass(frozen=True)
class ImageConditioningInput:
    """A still to condition on, by file, with the H.264 CRF it should be re-compressed at.

    Mirrors upstream ``ltx_pipelines.utils.types.ImageConditioningInput`` (v1.5.0). All frame
    indices are zero-based pixel-frame indices. Leaving ``crf`` unset uses the model's checkpoint
    default (``ImageConditioner.resolve_crf``: 33 before LTX-2.4, 18 from 2.4 on); pass it
    explicitly to override that value, including ``0`` to skip re-compression.

    How the still binds is ``binding``: :class:`ImageReplace` (lowered to a
    ``VideoConditionByLatentIndex``) or :class:`ImageGuide` (a ``VideoConditionByKeyframeIndex``).
    Construct with :meth:`at` (frame 0 replaces, later frames guide), :meth:`replace`, or
    :meth:`guide`.
    """

    path: str
    strength: float
    binding: ImageBinding
    crf: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ImageReplace | ImageGuide):
            raise TypeError(f"binding must be ImageReplace or ImageGuide, got {type(self.binding).__name__}")

    @classmethod
    def at(cls, path: str, frame_idx: int, strength: float, crf: int | None = None) -> ImageConditioningInput:
        """Bind at a pixel frame, replacing frame 0 and guiding at every later frame."""
        if frame_idx == 0:
            return cls.replace(path, 0, strength, crf)
        return cls.guide(path, frame_idx, strength, crf)

    @classmethod
    def replace(cls, path: str, frame_idx: int, strength: float, crf: int | None = None) -> ImageConditioningInput:
        """Replace at a zero-based pixel frame that starts a causal VAE latent."""
        return cls(path=path, strength=strength, binding=ImageReplace(frame_idx), crf=crf)

    @classmethod
    def guide(cls, path: str, frame_idx: int, strength: float, crf: int | None = None) -> ImageConditioningInput:
        """Guide generation with a still at a pixel-frame index."""
        return cls(path=path, strength=strength, binding=ImageGuide(frame_idx), crf=crf)

    @property
    def frame_idx(self) -> int:
        """Pixel-frame index carried by either supported image binding."""
        return self.binding.frame_idx

    @property
    def replaces(self) -> bool:
        """Whether the still replaces a latent (``ImageReplace``) rather than guiding."""
        return isinstance(self.binding, ImageReplace)

    def map_frame_index(self, mapper: Callable[[int], int | None]) -> ImageConditioningInput | None:
        """Map the pixel-frame index through ``mapper``, preserving the replace vs guide binding.

        Upstream ``LoadedImageConditioning.map_frame_index``. ``mapper`` returning ``None``
        drops the image (``None`` is returned). A replace mapped onto a pixel frame that does not
        start a latent raises ``ValueError``, as upstream.
        """
        local = mapper(self.frame_idx)
        if local is None:
            return None
        return replace(self, binding=type(self.binding)(local))

    def with_frame_idx(self, frame_idx: int) -> ImageConditioningInput:
        """This input at another pixel frame, with the same binding."""
        return replace(self, binding=type(self.binding)(frame_idx))


def replacement_latent_index(frame_idx: int) -> int:
    """Latent index an ``ImageReplace`` at pixel frame ``frame_idx`` writes (upstream ``_replacement_frame_to_latent_index``).

    Latent 0 covers pixel frame 0 alone under the causal VAE, so pixel frame ``1 + 8 * k`` maps to
    latent ``k + 1``.
    """
    if frame_idx == 0:
        return 0
    return 1 + (frame_idx - 1) // _VIDEO_TEMPORAL_SCALE


def _assert_replacement_frame_idx(frame_idx: int) -> None:
    """Raise unless ``frame_idx`` starts a latent frame (0 or ``1 + 8k``; upstream message)."""
    if frame_idx == 0:
        return
    if frame_idx < 1 or (frame_idx - 1) % _VIDEO_TEMPORAL_SCALE != 0:
        raise ValueError(f"ImageReplace frame_idx must be 0 or 1 + {_VIDEO_TEMPORAL_SCALE} * k, got {frame_idx}")


@dataclass(frozen=True)
class ModalitySpec:
    """Specification for one modality passed to a diffusion stage.

    Carries everything needed to build the initial noised latent state
    and run the denoising loop for a single modality (video or audio).

    Attributes:
        context: Pre-projected text embeddings, shape ``(B, Nt, dim)``.
        conditionings: List of conditioning items applied to the initial
            state before noising (e.g. ``VideoConditionByKeyframeIndex``).
        noise_scale: Scalar in ``[0, 1]`` controlling how much noise is
            mixed in. ``1.0`` = pure noise (typical stage 1 with
            ``initial_latent=None``); ``stage_2_sigmas[0]`` = partial
            noise mixed with ``initial_latent``.
        frozen: When True the state is built but its ``denoise_mask`` is
            zeroed so all tokens are treated as preserved. Used when an
            audio modality is passed for cross-modal context only and
            should not be re-denoised.
        initial_latent: Optional starting latent. ``None`` means start
            from zeros (typical stage 1). Provided as the upscaled
            stage-1 output for stage 2.
    """

    context: mx.array
    conditionings: list[Any] = field(default_factory=list)
    noise_scale: float = 1.0
    frozen: bool = False
    initial_latent: mx.array | None = None


class Denoiser(Protocol):
    """Protocol for a denoiser that receives the transformer at call time.

    Mirrors upstream ``ltx_pipelines.utils.types.Denoiser``. The
    transformer is not stored — it is passed as the first argument so
    the caller (a denoising loop or a pipeline block) controls its
    lifecycle.

    Args:
        transformer: The diffusion model (typically an X0Model wrapper).
        video_state: Current video latent state, or ``None`` if absent.
        audio_state: Current audio latent state, or ``None`` if absent.
        sigmas: 1-D array of sigma values for each diffusion step.
        step_index: Index of the current denoising step.

    Returns:
        ``(denoised_video, denoised_audio)`` arrays (either may be ``None``).
    """

    def __call__(
        self,
        transformer: Any,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: mx.array,
        step_index: int,
    ) -> tuple[mx.array | None, mx.array | None]: ...
