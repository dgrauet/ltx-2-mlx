"""Tests for two-stage pipeline components and logic.

Tests the denorm/renorm roundtrip, res2s with guidance, and pipeline
instantiation — all without requiring model weights.
"""

from pathlib import Path

import mlx.core as mx
import pytest

from ltx_core_mlx.model.video_vae.video_vae import EncoderPerChannelStatistics, VideoEncoder


# ---------------------------------------------------------------------------
# Denormalize / Normalize roundtrip
# ---------------------------------------------------------------------------
class TestDenormRenormRoundtrip:
    """Verify that denormalize → normalize recovers the original latent."""

    def _make_encoder_with_stats(self, channels: int = 4) -> VideoEncoder:
        """Create a VideoEncoder and set known per-channel stats."""
        enc = VideoEncoder.__new__(VideoEncoder)
        enc.per_channel_statistics = EncoderPerChannelStatistics.__new__(EncoderPerChannelStatistics)
        enc.per_channel_statistics.mean_of_means = mx.arange(channels, dtype=mx.float32) * 0.1
        enc.per_channel_statistics.std_of_means = mx.ones(channels, dtype=mx.float32) * 2.0
        return enc

    def test_roundtrip_identity(self):
        """normalize(denormalize(x)) should return x."""
        enc = self._make_encoder_with_stats(4)
        x = mx.random.normal((1, 2, 3, 4, 4))
        denormed = enc.denormalize_latent(x)
        renormed = enc.normalize_latent(denormed)
        assert mx.allclose(renormed, x, atol=1e-5).item()

    def test_denormalize_changes_values(self):
        """Denormalized values should differ from input (unless stats are trivial)."""
        enc = self._make_encoder_with_stats(4)
        x = mx.ones((1, 1, 1, 1, 4))
        denormed = enc.denormalize_latent(x)
        assert not mx.allclose(denormed, x, atol=1e-6).item()

    def test_shape_preserved(self):
        """Shape should be preserved through denorm/renorm."""
        enc = self._make_encoder_with_stats(8)
        shape = (2, 3, 4, 5, 8)
        x = mx.random.normal(shape)
        assert enc.denormalize_latent(x).shape == shape
        assert enc.normalize_latent(x).shape == shape

    def test_denorm_renorm_with_transpose(self):
        """Test the full transpose flow used in the pipeline: BCFHW → BFHWC → denorm → BCFHW."""
        enc = self._make_encoder_with_stats(8)
        # Simulate pipeline latent in PyTorch layout (B, C, F, H, W)
        latent_pt = mx.random.normal((1, 8, 2, 3, 4))

        # Pipeline flow
        latent_mlx = latent_pt.transpose(0, 2, 3, 4, 1)  # → (B, F, H, W, C)
        denormed = enc.denormalize_latent(latent_mlx)
        denormed_pt = denormed.transpose(0, 4, 1, 2, 3)  # → (B, C, F, H, W)

        # Reverse
        back_mlx = denormed_pt.transpose(0, 2, 3, 4, 1)
        renormed = enc.normalize_latent(back_mlx)
        renormed_pt = renormed.transpose(0, 4, 1, 2, 3)

        assert mx.allclose(renormed_pt, latent_pt, atol=1e-5).item()


# ---------------------------------------------------------------------------
# Upsampler shape with denorm/renorm
# ---------------------------------------------------------------------------
class TestUpsamplerDenormRenorm:
    """Test upsampler preserves channels through the denorm → upsample → renorm flow."""

    def test_upsampler_output_channels_match_input(self):
        """Upsampler should preserve channel count (in_channels → in_channels)."""
        from ltx_core_mlx.model.upsampler.model import LatentUpsampler

        upsampler = LatentUpsampler(in_channels=32, mid_channels=64, num_blocks_per_stage=1)
        latent = mx.zeros((1, 32, 2, 3, 4))  # BCFHW
        out = upsampler(latent)
        assert out.shape[1] == 32  # channels preserved
        assert out.shape[3] == 6  # H * 2
        assert out.shape[4] == 8  # W * 2


# ---------------------------------------------------------------------------
# res2s_denoise_loop with guidance
# ---------------------------------------------------------------------------
class TestRes2sGuidance:
    """Test that res2s_denoise_loop works with guider factories."""

    def _make_model_with_x0(self):
        """Create an X0Model with a dummy DiT that predicts velocity."""
        from ltx_core_mlx.model.transformer.model import LTXModel, X0Model

        class VelocityDiT(LTXModel):
            """DiT that predicts velocity v such that x0 = x_t - sigma * v."""

            def __init__(self):
                # Skip LTXModel.__init__ to avoid creating real layers
                pass

            def __call__(self, **kwargs):
                v = kwargs["video_latent"]
                a = kwargs["audio_latent"]
                sigma = kwargs["timestep"][0].item()
                if sigma > 0:
                    return v / sigma, a / sigma
                return mx.zeros_like(v), mx.zeros_like(a)

        return X0Model(VelocityDiT())

    def test_res2s_without_guidance_runs(self):
        """res2s without guidance should run and return correct shapes."""
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState
        from ltx_pipelines_mlx.utils.samplers import res2s_denoise_loop

        model = self._make_model_with_x0()
        N_video, N_audio, D = 8, 4, 16
        video_state = LatentState(
            latent=mx.random.normal((1, N_video, D)),
            clean_latent=mx.zeros((1, N_video, D)),
            denoise_mask=mx.ones((1, N_video, 1)),
        )
        audio_state = LatentState(
            latent=mx.random.normal((1, N_audio, D)),
            clean_latent=mx.zeros((1, N_audio, D)),
            denoise_mask=mx.ones((1, N_audio, 1)),
        )
        output = res2s_denoise_loop(
            model=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=mx.zeros((1, 4, D)),
            audio_text_embeds=mx.zeros((1, 4, D)),
            sigmas=[1.0, 0.5, 0.0],
            show_progress=False,
        )
        assert output.video_latent.shape == (1, N_video, D)
        assert output.audio_latent.shape == (1, N_audio, D)

    def test_res2s_with_guidance_runs(self):
        """res2s with CFG guidance should run and return correct shapes."""
        from ltx_core_mlx.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState
        from ltx_pipelines_mlx.utils.samplers import res2s_denoise_loop

        model = self._make_model_with_x0()
        N_video, N_audio, D = 8, 4, 16
        video_state = LatentState(
            latent=mx.random.normal((1, N_video, D)),
            clean_latent=mx.zeros((1, N_video, D)),
            denoise_mask=mx.ones((1, N_video, 1)),
        )
        audio_state = LatentState(
            latent=mx.random.normal((1, N_audio, D)),
            clean_latent=mx.zeros((1, N_audio, D)),
            denoise_mask=mx.ones((1, N_audio, 1)),
        )

        params = MultiModalGuiderParams(cfg_scale=3.0)
        neg_embeds = mx.zeros((1, 4, D))
        video_factory = create_multimodal_guider_factory(params, negative_context=neg_embeds)
        audio_factory = create_multimodal_guider_factory(params, negative_context=neg_embeds)

        output = res2s_denoise_loop(
            model=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=mx.zeros((1, 4, D)),
            audio_text_embeds=mx.zeros((1, 4, D)),
            sigmas=[1.0, 0.5, 0.0],
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            show_progress=False,
        )
        assert output.video_latent.shape == (1, N_video, D)
        assert output.audio_latent.shape == (1, N_audio, D)

    def test_res2s_with_stg_guidance_runs(self):
        """res2s with STG guidance (perturbations) should run without errors."""
        from ltx_core_mlx.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState
        from ltx_pipelines_mlx.utils.samplers import res2s_denoise_loop

        model = self._make_model_with_x0()
        N_video, N_audio, D = 8, 4, 16
        video_state = LatentState(
            latent=mx.random.normal((1, N_video, D)),
            clean_latent=mx.zeros((1, N_video, D)),
            denoise_mask=mx.ones((1, N_video, 1)),
        )
        audio_state = LatentState(
            latent=mx.random.normal((1, N_audio, D)),
            clean_latent=mx.zeros((1, N_audio, D)),
            denoise_mask=mx.ones((1, N_audio, 1)),
        )

        params = MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0, stg_blocks=[0])
        neg_embeds = mx.zeros((1, 4, D))
        video_factory = create_multimodal_guider_factory(params, negative_context=neg_embeds)
        audio_factory = create_multimodal_guider_factory(params, negative_context=neg_embeds)

        output = res2s_denoise_loop(
            model=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=mx.zeros((1, 4, D)),
            audio_text_embeds=mx.zeros((1, 4, D)),
            sigmas=[1.0, 0.5, 0.0],
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            show_progress=False,
        )
        assert output.video_latent.shape == (1, N_video, D)
        assert output.audio_latent.shape == (1, N_audio, D)


# ---------------------------------------------------------------------------
# Two-stage pipeline instantiation
# ---------------------------------------------------------------------------
class TestTwoStagePipelineInstantiation:
    """Test that pipelines can be instantiated without model weights."""

    def _make_tmpdir(self, tmp_path):
        """Create a temp dir that looks like a model dir (local path)."""
        d = tmp_path / "fake_model"
        d.mkdir()
        return str(d)

    def test_two_stage_init(self, tmp_path):
        from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline

        pipe = TI2VidTwoStagesPipeline(model_dir=self._make_tmpdir(tmp_path), low_memory=True)
        assert pipe._dev_transformer == "transformer-dev.safetensors"
        assert pipe._distilled_lora == "ltx-2.3-22b-distilled-lora-384.safetensors"
        assert pipe._distilled_lora_strength == 1.0
        assert pipe.upsampler is None
        assert pipe.vae_encoder is None

    def test_hq_inherits_from_two_stage(self):
        from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

        assert issubclass(TI2VidTwoStagesHQPipeline, TI2VidTwoStagesPipeline)

    def test_hq_init(self, tmp_path):
        from ltx_pipelines_mlx.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

        pipe = TI2VidTwoStagesHQPipeline(model_dir=self._make_tmpdir(tmp_path), low_memory=True)
        assert pipe._dev_transformer == "transformer-dev.safetensors"

    def test_custom_params(self, tmp_path):
        from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline

        pipe = TI2VidTwoStagesPipeline(
            model_dir=self._make_tmpdir(tmp_path),
            dev_transformer="custom-dev.safetensors",
            distilled_lora="custom-lora.safetensors",
            distilled_lora_strength=0.8,
        )
        assert pipe._dev_transformer == "custom-dev.safetensors"
        assert pipe._distilled_lora == "custom-lora.safetensors"
        assert pipe._distilled_lora_strength == 0.8

    def test_keyframe_inherits_from_two_stage(self):
        from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline
        from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline

        assert issubclass(KeyframeInterpolationPipeline, TI2VidTwoStagesPipeline)


# ---------------------------------------------------------------------------
# Upsampler weights must be present — silent fallback is forbidden
# ---------------------------------------------------------------------------
class TestUpsamplerMissingWeights:
    """Regression: a missing spatial upscaler must fail loud, not run untrained.

    When ``spatial_upscaler_x2_v1_1.safetensors`` was absent from the model
    dir, ``_load_upsampler`` silently kept a randomly-initialised
    ``LatentUpsampler`` (no ``load_weights`` call). Stage 2 then ran the latent
    through an untrained Conv3d + pixel-shuffle, decoding into a periodic grid
    ("mosaic") with no error or warning. The loader must raise instead.
    """

    def test_missing_weights_raises_filenotfound(self, tmp_path):
        from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline

        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()
        pipe = TI2VidTwoStagesPipeline(model_dir=str(model_dir), low_memory=True)
        assert pipe.upsampler is None

        with pytest.raises(FileNotFoundError) as exc:
            pipe._load_upsampler()

        # The message must name the missing file so the user knows what to fetch.
        assert "spatial_upscaler_x2_v1_1" in str(exc.value)
        # And no untrained upsampler may be left behind for stage 2 to use.
        assert pipe.upsampler is None


# ---------------------------------------------------------------------------
# Stage 2 dims from upscaled shape
# ---------------------------------------------------------------------------
class TestStage2Dims:
    """Test that H_full = H_half * 2 differs from compute_video_latent_shape."""

    def test_dims_diverge_at_480(self):
        """At height=480, H_half*2 != compute_video_latent_shape(480)."""
        from ltx_core_mlx.components.patchifiers import compute_video_latent_shape

        half_h = 480 // 2  # 240
        _, H_half, _ = compute_video_latent_shape(33, half_h, 352)
        _, H_target, _ = compute_video_latent_shape(33, 480, 704)

        # H_half * 2 should differ from H_target due to rounding
        H_full_correct = H_half * 2
        assert H_full_correct != H_target, (
            f"H_half*2={H_full_correct} should differ from H_target={H_target} to demonstrate why we use H_half*2"
        )

    def test_dims_match_at_512(self):
        """At height=512 (power of 2), both methods should agree."""
        from ltx_core_mlx.components.patchifiers import compute_video_latent_shape

        half_h = 512 // 2  # 256
        _, H_half, _ = compute_video_latent_shape(33, half_h, 352)
        _, H_target, _ = compute_video_latent_shape(33, 512, 704)

        H_full_correct = H_half * 2
        assert H_full_correct == H_target


# ---------------------------------------------------------------------------
# BasePipeline._resolve_safetensors
# ---------------------------------------------------------------------------
class TestResolveSafetensors:
    """Unit tests for BasePipeline._resolve_safetensors."""

    def _resolve(self, model_dir: Path, stem: str) -> Path:
        from ltx_pipelines_mlx._base import BasePipeline

        return BasePipeline._resolve_safetensors(model_dir, stem)

    def test_exact_only(self, tmp_path: Path) -> None:
        """Falls back to exact path when no versioned file exists."""
        (tmp_path / "transformer-distilled.safetensors").touch()
        result = self._resolve(tmp_path, "transformer-distilled")
        assert result.name == "transformer-distilled.safetensors"

    def test_versioned_preferred(self, tmp_path: Path) -> None:
        """Versioned file is returned when present."""
        (tmp_path / "transformer-distilled-1.1.safetensors").touch()
        result = self._resolve(tmp_path, "transformer-distilled")
        assert result.name == "transformer-distilled-1.1.safetensors"
        assert result.exists()

    def test_versioned_beats_exact(self, tmp_path: Path) -> None:
        """Versioned wins over unversioned when both exist."""
        (tmp_path / "transformer-distilled.safetensors").touch()
        (tmp_path / "transformer-distilled-1.1.safetensors").touch()
        result = self._resolve(tmp_path, "transformer-distilled")
        assert result.name == "transformer-distilled-1.1.safetensors"

    def test_latest_version_selected(self, tmp_path: Path) -> None:
        """When multiple versioned files exist, the alphabetically last is returned."""
        (tmp_path / "transformer-distilled-1.0.safetensors").touch()
        (tmp_path / "transformer-distilled-1.1.safetensors").touch()
        result = self._resolve(tmp_path, "transformer-distilled")
        assert result.name == "transformer-distilled-1.1.safetensors"

    def test_no_match_returns_canonical(self, tmp_path: Path) -> None:
        """Returns the canonical (exact) path when nothing exists, for clear error messages."""
        result = self._resolve(tmp_path, "transformer-distilled")
        assert result.name == "transformer-distilled.safetensors"
        assert not result.exists()

    def test_lora_versioned_fallback(self, tmp_path: Path) -> None:
        """Works for LoRA stems too."""
        stem = "ltx-2.3-22b-distilled-lora-384"
        (tmp_path / f"{stem}-1.1.safetensors").touch()
        result = self._resolve(tmp_path, stem)
        assert result.name == f"{stem}-1.1.safetensors"
        assert result.exists()
