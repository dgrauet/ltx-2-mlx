"""DFR spatial epilogue: building blocks (Lanczos x2, tile clamp, single-frame decode) and --spatial-upscalings 2."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

import ltx_pipelines_mlx.dfr as dfr_mod
import ltx_pipelines_mlx.utils._orchestration as orch
from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig
from ltx_core_mlx.utils.positions import compute_audio_token_count
from ltx_pipelines_mlx.dfr import (
    EPILOGUE_KEYFRAME_STRENGTH,
    EPILOGUE_NOISE_SEED_OFFSET,
    EPILOGUE_SPATIAL_OVERLAP,
    KEYFRAME_PLANE_DECODE_SEED_OFFSET,
    TEMPORAL_UPSAMPLER_STEM,
    DFRPipeline,
    clamp_tile_counts,
    floor_to_multiple,
    lanczos_x2,
)
from ltx_pipelines_mlx.scheduler import LTX_2_5_STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.args import ImageConditioningInput
from tests.test_dfr import _make, _run
from tests.test_dfr_temporal import _fake_temporal
from tests.test_ltx25_distilled import _fake_upsampler, _FakeVaeEncoder


def test_constants():
    assert EPILOGUE_SPATIAL_OVERLAP == 12
    assert EPILOGUE_KEYFRAME_STRENGTH == 1.0
    assert KEYFRAME_PLANE_DECODE_SEED_OFFSET == 4000
    assert EPILOGUE_NOISE_SEED_OFFSET == 2000


def _pil_lanczos_x2(frame: np.ndarray) -> np.ndarray:
    """Reference: uint8-round -> PIL resize x2 LANCZOS -> float32 [0, 1]."""
    height, width, channels = frame.shape
    array = np.clip(frame, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    if channels == 1:
        image = Image.fromarray(array[..., 0], mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    image = image.resize((width * 2, height * 2), resample=Image.Resampling.LANCZOS)
    resized = np.asarray(image, dtype=np.float32) / 255.0
    if resized.ndim == 2:
        resized = resized[..., None]
    return resized


class TestLanczosX2:
    def test_matches_pil_reference_rgb(self):
        rng = np.random.default_rng(0)
        frames = rng.random((3, 8, 10, 3)).astype(np.float32)
        out = lanczos_x2(frames)
        assert out.shape == (3, 16, 20, 3)
        assert out.dtype == np.float32
        expected = np.stack([_pil_lanczos_x2(f) for f in frames], axis=0)
        np.testing.assert_array_equal(out, expected)

    def test_matches_pil_reference_single_channel(self):
        rng = np.random.default_rng(1)
        frames = rng.random((2, 6, 6, 1)).astype(np.float32)
        out = lanczos_x2(frames)
        assert out.shape == (2, 12, 12, 1)
        expected = np.stack([_pil_lanczos_x2(f) for f in frames], axis=0)
        np.testing.assert_array_equal(out, expected)

    def test_output_range_clamped(self):
        frames = np.full((1, 4, 4, 3), 2.0, dtype=np.float32)  # out of [0, 1]
        out = lanczos_x2(frames)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_requires_at_least_one_frame(self):
        with pytest.raises(ValueError):
            lanczos_x2(np.zeros((0, 4, 4, 3), dtype=np.float32))

    def test_requires_fhwc(self):
        with pytest.raises(ValueError):
            lanczos_x2(np.zeros((4, 4, 3), dtype=np.float32))

    def test_rejects_bad_channel_count(self):
        with pytest.raises(ValueError):
            lanczos_x2(np.zeros((1, 4, 4, 2), dtype=np.float32))


class TestClampTileCounts:
    def test_untouched_when_it_fits(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(1, 0),
            height=DimensionTilingConfig(2, 12),
            width=DimensionTilingConfig(2, 12),
        )
        clamped = clamp_tile_counts(tiling, (9, 32, 48))
        assert clamped == tiling

    def test_dim_smaller_than_num_tiles_falls_back_to_one_tile(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(1, 0),
            height=DimensionTilingConfig(2, 12),
            width=DimensionTilingConfig(2, 12),
        )
        clamped = clamp_tile_counts(tiling, (9, 1, 48))
        assert clamped.height == DimensionTilingConfig(1, 0)
        assert clamped.width == DimensionTilingConfig(2, 12)

    def test_overlap_clamped_to_dim_minus_num_tiles(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(1, 0),
            height=DimensionTilingConfig(2, 12),
            width=DimensionTilingConfig(2, 12),
        )
        clamped = clamp_tile_counts(tiling, (9, 4, 48))
        assert clamped.height == DimensionTilingConfig(2, 2)  # dim_size - n = 4 - 2 = 2
        assert clamped.width == DimensionTilingConfig(2, 12)

    def test_single_tile_dimension_untouched(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(1, 0),
            height=DimensionTilingConfig(1, 0),
            width=DimensionTilingConfig(2, 12),
        )
        clamped = clamp_tile_counts(tiling, (9, 1, 4))
        assert clamped.height == DimensionTilingConfig(1, 0)
        assert clamped.width == DimensionTilingConfig(2, 2)


def test_floor_to_multiple():
    assert floor_to_multiple(1080, 128) == 1024
    assert floor_to_multiple(1024, 128) == 1024
    assert floor_to_multiple(127, 128) == 0


class _StubConvDecoder:
    def __init__(self):
        self.calls = []

    def decode(self, latent):
        self.calls.append(latent)
        import mlx.core as mx

        b, _c, f, h, w = latent.shape
        return mx.full((b, 3, f, h * 32, w * 32), 1.0, dtype=latent.dtype)  # mimics the real 32x spatial upsample


class _StubInnerDiffusionDecoder:
    def __init__(self):
        self.calls = []

    def decode(self, latent, *, seed=0):
        self.calls.append((latent, seed))
        b, _c, f, h, w = latent.shape
        import mlx.core as mx

        return mx.full((b, 3, f, h, w), 0.5, dtype=latent.dtype)


class _StubDiffusionBlock:
    """Mimics ``_DiffusionVideoDecoder``'s public surface for decode_single_frame."""

    def __init__(self):
        self._decoder = _StubInnerDiffusionDecoder()


class TestDecodeSingleFrame:
    def test_conv_path_calls_decoder_and_normalizes(self, monkeypatch):
        import mlx.core as mx

        from ltx_pipelines_mlx.utils.blocks import VideoDecoder

        block = VideoDecoder.__new__(VideoDecoder)
        block.video_decoder = "conv"
        block._decoder = _StubConvDecoder()
        monkeypatch.setattr(block, "load", lambda: block._decoder)

        latent = mx.zeros((1, 128, 1, 2, 3))
        out = block.decode_single_frame(latent, seed=7)

        assert len(block._decoder.calls) == 1
        assert out.shape == (1, 2 * 32, 3 * 32, 3)
        assert bool(mx.all((out >= 0) & (out <= 1)).item())

    def test_diffusion_path_passes_seed(self, monkeypatch):
        import mlx.core as mx

        from ltx_pipelines_mlx.utils.blocks import VideoDecoder

        block = VideoDecoder.__new__(VideoDecoder)
        block.video_decoder = "diffusion"
        stub = _StubDiffusionBlock()
        block._decoder = stub
        monkeypatch.setattr(block, "load", lambda: block._decoder)

        latent = mx.zeros((1, 128, 1, 2, 3))
        out = block.decode_single_frame(latent, seed=42)

        assert len(stub._decoder.calls) == 1
        _, seed = stub._decoder.calls[0]
        assert seed == 42
        assert out.shape[0] == 1
        assert out.shape[-1] == 3

    def test_rejects_multi_frame_latent(self):
        import mlx.core as mx

        from ltx_pipelines_mlx.utils.blocks import VideoDecoder

        block = VideoDecoder.__new__(VideoDecoder)
        block.video_decoder = "conv"

        latent = mx.zeros((1, 128, 2, 2, 3))
        with pytest.raises(ValueError):
            block.decode_single_frame(latent, seed=0)


# ---- --spatial-upscalings 2 end to end (test doubles of tests/test_dfr.py) ----------------------------


class _EncodingVaeEncoder(_FakeVaeEncoder):
    """Identity (de)normalisation plus an ``encode`` that records its pixel input."""

    def __init__(self):
        self.encoded: list[mx.array] = []

    def encode(self, pixels):
        self.encoded.append(pixels)
        b, _c, f, h, w = pixels.shape
        return mx.full((b, 128, f, h // 32, w // 32), float(len(self.encoded)), dtype=mx.bfloat16)


def _make_epilogue(tmp_path, monkeypatch, *, t=0, low_ram=False):
    pipe, euler, ancestral, noised, attached = _make(tmp_path, monkeypatch, low_ram=low_ram)
    (tmp_path / f"{TEMPORAL_UPSAMPLER_STEM}.safetensors").write_bytes(b"")
    pipe.spatial_upscalings = 2
    pipe.temporal_upscalings = t
    encoder = _EncodingVaeEncoder()
    pipe.vae_encoder = encoder
    monkeypatch.setattr(pipe, "_detach_detailing_lora", lambda: None)
    monkeypatch.setattr(pipe, "_load_temporal_upsampler", lambda: _fake_temporal)
    rec = {"decoded": [], "decoder_kind": [], "freed": 0, "tilers": [], "tiled": []}

    def decode_single_frame(latent, *, seed=0):
        rec["decoded"].append((tuple(latent.shape), seed))
        rec["decoder_kind"].append(pipe.video_decoder_block.video_decoder)
        _b, _c, _f, h, w = latent.shape
        return mx.full((1, h * 32, w * 32, 3), 0.5)

    def free():
        rec["freed"] += 1

    monkeypatch.setattr(pipe.video_decoder_block, "decode_single_frame", decode_single_frame)
    monkeypatch.setattr(pipe.video_decoder_block, "free", free)

    class _SpyTiler(VideoModalityTiler):
        def __init__(self, tiling, latent_shape, seams=()):
            rec["tilers"].append({"tiling": tiling, "latent_shape": latent_shape, "seams": list(seams)})
            super().__init__(tiling, latent_shape, seams=seams)

    class _SpyTiled(TiledLTXModel):
        def __init__(self, inner, tiler, normalize_positions=False):
            rec["tiled"].append({"inner": inner, "tiler": tiler, "normalize_positions": normalize_positions})
            super().__init__(inner, tiler, normalize_positions=normalize_positions)

    monkeypatch.setattr(dfr_mod, "VideoModalityTiler", _SpyTiler)
    monkeypatch.setattr(dfr_mod, "TiledLTXModel", _SpyTiled)
    return pipe, euler, noised, attached, encoder, rec


def _pack(tmp_path):
    cfg = {"transformer": {"num_layers": 48, "ff_bias": False, "use_keyframes_abs_pos_embedding": True}}
    (tmp_path / "embedded_config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_spatial_upscalings_validated_at_construction(tmp_path):
    with pytest.raises(ValueError, match="spatial_upscalings"):
        DFRPipeline(str(_pack(tmp_path)), spatial_upscalings=3)
    assert DFRPipeline(str(_pack(tmp_path))).spatial_upscalings == 1


@pytest.mark.parametrize("bad", ["relay", "tiling"])
def test_epilogue_refuses_prompt_relay_and_tiling_before_any_load(tmp_path, monkeypatch, bad):
    pipe, *_ = _make_epilogue(tmp_path, monkeypatch)
    pipe._load_text_encoder = lambda: (_ for _ in ()).throw(AssertionError("must not load"))  # type: ignore[method-assign]
    if bad == "relay":
        with pytest.raises(ValueError, match="Prompt Relay"):
            _run(pipe, height=256, width=256, prompt_relay=object())
    else:
        pipe._tile_count = object()
        with pytest.raises(ValueError, match="modality tiling"):
            _run(pipe, height=256, width=256)


def test_resolution_floors_to_128_with_a_warning(tmp_path, monkeypatch, capsys):
    pipe, _, noised, *_ = _make_epilogue(tmp_path, monkeypatch)
    video, _ = _run(pipe, height=300, width=270)
    assert "256x256" in capsys.readouterr().err
    assert video.shape[3:] == (8, 8)


def test_no_warning_on_a_multiple_of_128(tmp_path, monkeypatch, capsys):
    pipe, *_ = _make_epilogue(tmp_path, monkeypatch)
    _run(pipe, height=256, width=384)
    assert "multiples of 128" not in capsys.readouterr().err


def test_stage_resolutions_quarter_half_full(tmp_path, monkeypatch):
    pipe, euler, noised, *_ = _make_epilogue(tmp_path, monkeypatch)
    video, _ = _run(pipe, height=256, width=384, num_frames=49)
    assert noised[0]["spatial_dims"] == (7, 2, 3)  # stage 1 at H/4
    assert noised[2]["spatial_dims"] == (7, 4, 6)  # stage 2 at H/2
    assert noised[4]["spatial_dims"] == (7, 8, 12)  # epilogue at H
    assert video.shape == (1, 128, 7, 8, 12)


def test_planes_decoded_in_order_with_offset_seeds_then_lanczos_and_encoded(tmp_path, monkeypatch):
    pipe, _, _, _, encoder, rec = _make_epilogue(tmp_path, monkeypatch)
    pipe.video_decoder = "diffusion"
    _run(pipe, height=256, width=256, num_frames=49)  # stage-2 slots at [24, 48]
    assert rec["decoded"] == [((1, 128, 1, 4, 4), 7 + 4000), ((1, 128, 1, 4, 4), 7 + 4001)]
    assert rec["decoder_kind"] == ["diffusion", "diffusion"]
    assert rec["freed"] == 1  # decoder block freed after the planes
    planes = encoder.encoded[-2:]
    assert all(p.shape == (1, 3, 1, 256, 256) for p in planes)  # Lanczos x2 of the 128 px decode
    assert all(abs(float(p.min()) - 0.0) < 1e-2 and abs(float(p.max()) - 0.0) < 1e-2 for p in planes)  # 0.5 -> vae 0


def test_epilogue_conditionings(tmp_path, monkeypatch):
    pipe, _, noised, _, _, _ = _make_epilogue(tmp_path, monkeypatch, t=1)
    seen = []
    monkeypatch.setattr(orch, "combined_image_conditionings", lambda imgs, **kw: [])
    monkeypatch.setattr(dfr_mod, "combined_image_conditionings", lambda imgs, **kw: seen.append((list(imgs), kw)) or [])
    _run(pipe, height=256, width=256, num_frames=49, images=[ImageConditioningInput("a.png", 24, 1.0)])
    imgs, kw = seen[-1]
    assert [i.frame_idx for i in imgs] == [48] and kw["enc_h"] == 256 and kw["enc_w"] == 256
    assert kw["spatial_dims"] == (13, 8, 8)
    conds = noised[-2]["conditionings"]
    kfs = [c for c in conds if isinstance(c, VideoConditionByKeyframeIndex)]
    assert [k.frame_idx for k in kfs] == pipe.generated_keyframe_positions
    assert all(k.strength == 1.0 and k.keyframe_latent.shape == (1, 64, 128) for k in kfs)
    refs = [c for c in conds if isinstance(c, VideoConditionByReferenceLatent)]
    assert len(refs) == 1 and refs[0].downscale_factor == 2 and refs[0].strength == 1.0
    assert refs[0].reference_latent.shape == (1, 13 * 4 * 4, 128)  # the H/2 guide latent
    assert not [c for c in conds if isinstance(c, VideoGeneratedKeyframeSlots)]


def test_epilogue_model_tiler_sampler_and_frozen_audio_t0(tmp_path, monkeypatch):
    pipe, euler, noised, _, _, rec = _make_epilogue(tmp_path, monkeypatch)
    _run(pipe, height=512, width=512, num_frames=49)
    (tiler,) = rec["tilers"]
    assert tiler["latent_shape"] == (7, 16, 16) and tiler["seams"] == []
    t = tiler["tiling"]
    assert (t.frames.num_tiles, t.height.num_tiles, t.width.num_tiles) == (1, 2, 2)
    assert t.height.overlap == 12 and t.width.overlap == 12
    (tiled,) = rec["tiled"]
    assert tiled["inner"] is pipe.dit and tiled["normalize_positions"] is True
    call = euler.calls[-1]
    assert isinstance(call["model"], TiledLTXModel)  # X0Model is the identity in the doubles
    assert call["sigmas"] == LTX_2_5_STAGE_2_DISTILLED_SIGMAS
    assert noised[-2]["seed"] == 7 + 2000 and noised[-2]["sigma"] == LTX_2_5_STAGE_2_DISTILLED_SIGMAS[0]
    assert call["audio_state"].frozen is True and float(mx.abs(call["audio_state"].denoise_mask).max()) == 0.0
    assert call["audio_state"].latent.shape[1] == compute_audio_token_count(49, frame_rate=24.0)


def test_epilogue_tiles_on_the_last_round_seams_t1(tmp_path, monkeypatch):
    pipe, euler, noised, attached, _, rec = _make_epilogue(tmp_path, monkeypatch, t=1)
    video, _ = _run(pipe, height=256, width=256, num_frames=49)  # 97 frames @ 48 fps
    (tiler,) = rec["tilers"]
    assert tiler["latent_shape"] == (13, 8, 8)
    assert tiler["seams"] == [6, 12]  # round-1 seams [48, 96] in latent cells
    t = tiler["tiling"]
    assert (t.frames.num_tiles, t.frames.overlap) == (2, 7)
    assert (t.height.num_tiles, t.height.overlap) == (2, 6)  # clamped: 8 - 2
    assert len(attached) == 2  # re-attached after the rounds detached it
    call = euler.calls[-1]
    assert call["audio_state"].latent.shape[1] == compute_audio_token_count(97, frame_rate=60.0)
    assert video.shape == (1, 128, 13, 8, 8)


def test_detailing_lora_attached_once_without_rounds(tmp_path, monkeypatch):
    _pipe, _, _, attached, _, _ = _make_epilogue(tmp_path, monkeypatch)
    _run(_pipe, height=256, width=256)
    assert len(attached) == 1


def test_epilogue_reloads_the_freed_encoder_and_upsampler(tmp_path, monkeypatch):
    pipe, *_, encoder, _ = _make_epilogue(tmp_path, monkeypatch)
    pipe.low_memory = True
    reloads = {"enc": 0, "up": 0}

    def load_encoder():
        if pipe.image_conditioner._encoder is None:
            reloads["enc"] += 1
            pipe.image_conditioner._encoder = encoder
        return pipe.image_conditioner._encoder

    def load_upsampler():
        reloads["up"] += 1
        pipe.upsampler = _fake_upsampler

    monkeypatch.setattr(pipe.image_conditioner, "load", load_encoder)
    monkeypatch.setattr(pipe, "_load_upsampler", load_upsampler)
    video, _ = _run(pipe, height=256, width=256)
    assert reloads == {"enc": 1, "up": 1}
    assert video.shape == (1, 128, 7, 8, 8)


def test_generated_keyframes_become_the_reencoded_planes(tmp_path, monkeypatch):
    pipe, *_ = _make_epilogue(tmp_path, monkeypatch)
    _run(pipe, height=256, width=256, num_frames=49)
    assert pipe.generated_keyframe_positions == [24, 48]
    assert pipe.generated_keyframes.shape == (1, 128, 2, 8, 8)
    assert [float(pipe.generated_keyframes[0, 0, i, 0, 0]) for i in range(2)] == [1.0, 2.0]  # encode order


def test_spatial_upscalings_1_is_unchanged(tmp_path, monkeypatch):
    pipe, euler, noised, attached, _, rec = _make_epilogue(tmp_path, monkeypatch)
    pipe.spatial_upscalings = 1
    video, _ = _run(pipe, height=128, width=128, num_frames=49)
    assert len(noised) == 4 and len(euler.calls) == 1 and len(attached) == 1
    assert noised[0]["spatial_dims"] == (7, 2, 2) and noised[2]["spatial_dims"] == (7, 4, 4)
    assert rec["decoded"] == [] and rec["tilers"] == [] and video.shape == (1, 128, 7, 4, 4)


def test_single_segment_rounds_fall_back_to_a_blended_frame_split(tmp_path, monkeypatch):
    """-f 9 with T=1: the only seam is the canvas's last cell, so the frame split is a clamped count split."""
    pipe, _, _, _, _, rec = _make_epilogue(tmp_path, monkeypatch, t=1)
    video, _ = _run(pipe, height=128, width=128, num_frames=9)  # canvas 25 -> 49 frames, seam [48] -> cell 6
    (tiler,) = rec["tilers"]
    assert tiler["latent_shape"] == (7, 4, 4) and tiler["seams"] == [6]
    t = tiler["tiling"]
    assert (t.frames.num_tiles, t.frames.overlap) == (2, 5)  # overlap 7 clamped to 7 - 2
    assert (t.height.num_tiles, t.height.overlap) == (2, 2)
    assert video.shape == (1, 128, ((9 - 1) * 2) // 8 + 1, 4, 4)


def test_images_reach_every_stage_at_its_resolution_and_the_epilogue_on_the_final_grid(tmp_path, monkeypatch):
    """Review focus 5: a frame-0 and an interior image, T=1, spatial x2 — real image conditionings."""
    from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex

    pipe, _, noised, _, _, _ = _make_epilogue(tmp_path, monkeypatch, t=1)
    paths = []
    for name, value in (("a.png", 64), ("b.png", 192)):
        path = tmp_path / name
        Image.fromarray(np.full((64, 64, 3), value, dtype=np.uint8)).save(path)
        paths.append(str(path))
    enc_dims: list[tuple[str, int, int]] = []

    def spy(module, label):
        real = module.combined_image_conditionings

        def wrapped(imgs, **kw):
            enc_dims.append((label, kw["enc_h"], kw["enc_w"]))
            return real(imgs, **kw)

        monkeypatch.setattr(module, "combined_image_conditionings", wrapped)

    spy(orch, "stage")  # stages 1/2 import it from _orchestration at call time
    spy(dfr_mod, "dfr")  # the rounds and the epilogue
    images = [ImageConditioningInput(paths[0], 0, 1.0), ImageConditioningInput(paths[1], 24, 1.0)]
    _run(pipe, height=256, width=384, num_frames=49, images=images)

    assert enc_dims[0] == ("stage", 64, 96)  # stage 1 at H/4
    assert enc_dims[1] == ("stage", 128, 192)  # stage 2 at H/2
    assert all(d == ("dfr", 128, 192) for d in enc_dims[2:-1])  # temporal-round tiles at H/2
    assert enc_dims[-1] == ("dfr", 256, 384)  # epilogue at full resolution
    conds = noised[-2]["conditionings"]
    assert isinstance(conds[0], VideoConditionByLatentIndex) and conds[0].frame_indices == [0]
    assert conds[0].clean_latent.shape == (1, 8 * 12, 128)  # full-res frame-0 tokens
    assert isinstance(conds[1], VideoConditionByKeyframeIndex) and conds[1].frame_idx == 48  # 24 * 2**1
    assert conds[1].keyframe_latent.shape == (1, 8 * 12, 128)
    planes = [c for c in conds[2:] if isinstance(c, VideoConditionByKeyframeIndex)]
    assert [p.frame_idx for p in planes] == pipe.generated_keyframe_positions


def test_epilogue_refuses_a_plane_count_mismatch(tmp_path, monkeypatch):
    """Upstream ``_keyframe_conditionings_from_latents``: K latents must match the carry positions."""
    pipe, *_ = _make_epilogue(tmp_path, monkeypatch)
    real = pipe._decode_lanczos_carry_keyframes
    monkeypatch.setattr(pipe, "_decode_lanczos_carry_keyframes", lambda kfs, seed: real(kfs, seed)[:-1])
    with pytest.raises(ValueError, match="keyframe latents"):
        _run(pipe, height=256, width=256, num_frames=49)
