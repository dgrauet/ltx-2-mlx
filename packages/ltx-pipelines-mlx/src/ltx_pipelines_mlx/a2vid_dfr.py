"""Audio-to-Video on the DFR base path (``a2v --dfr``, LTX-2.5 packs, experimental).

:class:`~ltx_pipelines_mlx.a2vid_distilled.A2VidDistilledPipeline`'s frozen input audio on
:class:`~ltx_pipelines_mlx.dfr.DFRPipeline`'s base path: the distilled stage 1 on the keyframe canvas, then the
full-resolution stage 2 with the official detailing IC-LoRA. The input audio is frozen in both stages, as on
``a2v --distilled``; the output carries the original waveform, trimmed to the clip.

DFR pads the clip to whole keyframe segments before stage 1, so the frozen audio has to span the padded canvas: the
encoded track covers the requested duration (it is checked to) and is right-padded with zero tokens up to the canvas.
The padding is cut with the rest of the canvas padding before decoding. Temporal rounds and the spatial epilogue are
not offered here.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline
from ltx_pipelines_mlx.dfr import DFRPipeline


def fit_audio_tokens(tokens: mx.array, num_tokens: int) -> mx.array:
    """Right-pad ``(1, T, C)`` audio tokens with zeros, or trim them, to ``num_tokens``."""
    have = tokens.shape[1]
    if have > num_tokens:
        return tokens[:, :num_tokens, :]
    if have < num_tokens:
        pad = mx.zeros((tokens.shape[0], num_tokens - have, tokens.shape[2]), dtype=tokens.dtype)
        return mx.concatenate([tokens, pad], axis=1)
    return tokens


class A2VidDFRPipeline(A2VidDistilledPipeline, DFRPipeline):
    """``a2v --dfr``: :class:`A2VidDistilledPipeline`'s frozen audio hooks on the DFR base path.

    Method resolution: the audio hooks and ``generate_and_save`` come from ``A2VidDistilledPipeline``;
    ``generate_two_stage`` (canvas, keyframe slots, detailing stage 2) and the constructor come from
    ``DFRPipeline``.
    """

    def generate_and_save(self, *args: Any, **kwargs: Any) -> str:  # type: ignore[override]
        """:meth:`A2VidDistilledPipeline.generate_and_save`, refusing a pack without keyframe slots first.

        DFR refuses such a pack (e.g. any LTX-2.3 pack) in ``generate_two_stage``; checking here
        fails before the audio encoder is loaded and the source track encoded.

        Raises:
            ValueError: on a pack without the keyframe embedding, before any model is loaded.
        """
        self._require_generated_keyframes_support([1])
        return super().generate_and_save(*args, **kwargs)

    def _frozen_audio_state(
        self,
        shape: tuple[int, ...],
        positions: mx.array,
        spatial_dims: tuple[int, int, int],
        seed: int,
    ) -> LatentState:
        """The input track frozen, padded with zero tokens to the DFR canvas."""
        source = self._source_audio_tokens
        assert source is not None
        self._source_audio_tokens = fit_audio_tokens(source, shape[1])
        try:
            return super()._frozen_audio_state(shape, positions, spatial_dims, seed)
        finally:
            self._source_audio_tokens = source


__all__ = ["A2VidDFRPipeline", "fit_audio_tokens"]
