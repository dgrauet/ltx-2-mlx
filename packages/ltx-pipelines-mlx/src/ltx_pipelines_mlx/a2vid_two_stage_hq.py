"""HQ Audio-to-Video pipeline — res_2s second-order sampler for Stage 1.

Same architecture as AudioToVideoPipeline but uses the res_2s sampler
instead of Euler for Stage 1, producing higher quality output.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_pipelines_mlx.a2vid_two_stage import AudioToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import res2s_denoise_loop


class AudioToVideoHQPipeline(AudioToVideoPipeline):
    """HQ Audio-to-Video with res_2s second-order sampler for Stage 1.

    Inherits from AudioToVideoPipeline and overrides Stage 1 denoising
    to use res_2s instead of Euler. Stage 2 is identical.
    """

    def _denoise_stage1(
        self,
        x0_model: X0Model,
        video_state: LatentState,
        audio_state: LatentState,
        video_embeds: mx.array,
        audio_embeds: mx.array,
        neg_video_embeds: mx.array,
        neg_audio_embeds: mx.array,
        sigmas: list[float],
        cfg_scale: float = 3.0,
        stg_scale: float = 0.0,
    ) -> object:
        """Run Stage 1 denoising with res_2s + CFG."""
        video_gp = MultiModalGuiderParams(cfg_scale=cfg_scale, stg_scale=stg_scale)
        audio_gp = MultiModalGuiderParams()

        video_factory = create_multimodal_guider_factory(video_gp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_gp, negative_context=neg_audio_embeds)

        return res2s_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
        )
