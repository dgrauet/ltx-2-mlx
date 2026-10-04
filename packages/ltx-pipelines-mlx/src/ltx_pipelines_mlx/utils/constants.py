"""Pipeline constants and default parameters for LTX-2.3."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from safetensors import safe_open

from ltx_core_mlx.components.guiders import MultiModalGuiderParams
from ltx_core_mlx.loader.helpers import parse_model_version

# H.264 CRF an image conditioning is re-compressed at, matching the compression the model was
# trained against (upstream ``ltx_pipelines.utils.constants``). This is a property of the model
# generation, not a code-level fallback: reach it through ``PipelineParams.default_image_crf``
# (see :func:`detect_params`), which is what a pipeline's ``ImageConditioner`` resolves an unset
# ``ImageConditioningInput.crf`` against.
DEFAULT_IMAGE_CRF = 33
LTX_2_4_IMAGE_CRF = 18
VIDEO_LATENT_CHANNELS = 128

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


@dataclass
class PipelineParams:
    """Default parameters for an LTX pipeline run.

    Args:
        seed: Random seed for reproducibility. None means random.
        stage_1_height: First-stage target height in pixels.
        stage_1_width: First-stage target width in pixels.
        num_frames: Number of video frames to generate.
        frame_rate: Output video frame rate.
        num_inference_steps: Number of denoising steps.
        default_image_crf: H.264 CRF image conditionings are re-compressed at
            for this model generation (see :func:`detect_params`).
        video_guider_params: Guidance parameters for the video modality.
        audio_guider_params: Guidance parameters for the audio modality.
    """

    seed: int | None = None
    stage_1_height: int = 1088
    stage_1_width: int = 1920
    num_frames: int = 257
    frame_rate: int = 24
    num_inference_steps: int = 30
    default_image_crf: int = DEFAULT_IMAGE_CRF
    video_guider_params: MultiModalGuiderParams = field(
        default_factory=MultiModalGuiderParams,
    )
    audio_guider_params: MultiModalGuiderParams = field(
        default_factory=MultiModalGuiderParams,
    )


LTX_2_3_PARAMS = PipelineParams(
    num_inference_steps=30,
    video_guider_params=MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    ),
    audio_guider_params=MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    ),
)

LTX_2_3_HQ_PARAMS = PipelineParams(
    num_inference_steps=15,
    stage_1_height=1088 // 2,
    stage_1_width=1920 // 2,
    video_guider_params=MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=0.0,
        rescale_scale=0.45,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    ),
    audio_guider_params=MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=0.0,
        rescale_scale=1.0,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    ),
)


# 2.4 continues the 2.3 lineage, so it inherits 2.3's knobs and only moves the image CRF
# (upstream ``LTX_2_4_PARAMS``).
LTX_2_4_PARAMS = replace(LTX_2_3_PARAMS, default_image_crf=LTX_2_4_IMAGE_CRF)

# Params per model generation, newest first (upstream ``_PARAMS_SINCE_VERSION``). A checkpoint
# gets the params of the newest generation it is at or above, so an unrecognised *newer* version
# inherits the closest known one. Anything older than every row, or unversioned, falls through to
# ``_UNVERSIONED_PARAMS``.
_PARAMS_SINCE_VERSION: tuple[tuple[tuple[int, ...], PipelineParams], ...] = (
    ((2, 4), LTX_2_4_PARAMS),
    ((2, 3), LTX_2_3_PARAMS),
)

# Upstream falls back to ``LTX_2_PARAMS`` (the 2.0 defaults); this port has no 2.0 preset, and the
# only generation-dependent field read off the result is ``default_image_crf``, which is
# ``DEFAULT_IMAGE_CRF`` on both.
_UNVERSIONED_PARAMS = PipelineParams()


def detect_model_version(checkpoint_path: str | Path) -> tuple[int, ...]:
    """Read a checkpoint's ``model_version`` metadata as comparable numeric components.

    Mirrors upstream ``ltx_pipelines.utils.constants.detect_model_version``. Returns ``()``,
    which compares below every real version, when the field is unset, unparseable, or the
    file cannot be read, so callers get their oldest fallback. Pre-release tags come both
    dot- and hyphen-separated (``"2.3.rc1"``, ``"2.4-rc2"``); the hyphen is normalized to a
    dot first so a release candidate maps onto the generation it is a candidate for.

    The LTX-2.5 MLX packs carry ``model_version`` in each safetensors header; the 2.3 packs
    carry none and therefore read as unversioned.

    Args:
        checkpoint_path: Path to a ``.safetensors`` file.

    Returns:
        The parsed version tuple, or ``()``.
    """
    logger = logging.getLogger(__name__)
    try:
        with safe_open(str(checkpoint_path), framework="numpy") as f:
            metadata = f.metadata() or {}
        version = metadata.get("model_version", "")
    except Exception:
        logger.warning("Could not read checkpoint metadata from %s, treating it as unversioned", checkpoint_path)
        return ()

    parsed = parse_model_version(version.replace("-", "."))
    logger.info("Checkpoint declares model_version=%s (parsed as %s)", version or "unknown", parsed)
    return parsed


def detect_params(checkpoint_path: str | Path) -> PipelineParams:
    """Pipeline params of the newest model generation the checkpoint is at or above.

    Mirrors upstream ``ltx_pipelines.utils.constants.detect_params``: reads ``model_version``
    via :func:`detect_model_version` and walks ``_PARAMS_SINCE_VERSION``; older, unset, or
    unreadable versions fall back to ``_UNVERSIONED_PARAMS``.

    Args:
        checkpoint_path: Path to a ``.safetensors`` file.

    Returns:
        The matching :class:`PipelineParams`.
    """
    parsed = detect_model_version(checkpoint_path)
    for since, params in _PARAMS_SINCE_VERSION:
        if parsed >= since:
            return params
    return _UNVERSIONED_PARAMS
