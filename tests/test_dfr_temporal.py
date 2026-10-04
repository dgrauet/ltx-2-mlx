"""DFR temporal-round helpers — pinned on upstream dfr_pipeline.py semantics."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import ltx_pipelines_mlx.dfr as dfr_mod
from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core_mlx.utils.positions import compute_audio_token_count
from ltx_pipelines_mlx.dfr import (
    ANCHOR_KEYFRAME_STRENGTH,
    TEMPORAL_ANCESTRAL_ETA,
    TEMPORAL_SIGMAS,
    TEMPORAL_UPSAMPLER_STEM,
    DFRPipeline,
    audio_latent_for_tile,
    conditioning_fps,
    lead_in_latent,
    rebase_image_conditionings,
    resample_audio_time,
    slot_initials_from_video,
)
from ltx_pipelines_mlx.utils.args import ImageConditioningInput
from tests.test_dfr import _make, _run


def test_constants():
    assert ANCHOR_KEYFRAME_STRENGTH == 0.95 and TEMPORAL_ANCESTRAL_ETA == 0.5
    assert TEMPORAL_SIGMAS == [0.975, 0.909375, 0.725, 0.421875, 0.0]


@pytest.mark.parametrize(("fps", "cond"), [(24.0, 24.0), (30.0, 30.0), (48.0, 60.0), (50.0, 60.0), (96.0, 60.0)])
def test_conditioning_fps(fps, cond):
    assert conditioning_fps(fps) == cond


def test_resample_audio_time_is_linear_and_clamped():
    a = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)  # values 0..3 along T
    out = resample_audio_time(a, 0.0, 4.0, 8)  # step 0.5
    assert np.allclose(np.array(out).reshape(-1), [0, 0.5, 1, 1.5, 2, 2.5, 3, 3])
    with pytest.raises(ValueError, match="empty"):
        resample_audio_time(a, 2.0, 2.0, 4)
    with pytest.raises(ValueError, match="out_frames"):
        resample_audio_time(a, 0.0, 1.0, 0)


def test_audio_latent_for_tile_window_and_token_count():
    full = mx.arange(100, dtype=mx.float32).reshape(1, 1, 100, 1)  # stage-1 audio, 4 s canvas
    out = audio_latent_for_tile(
        full, pixel_start=96, local_frames=145, playback_fps=48.0, source_duration=121 / 24.0, cond_fps=60.0
    )
    assert out.shape[2] == compute_audio_token_count(145, frame_rate=60.0)
    start = 96 / 48.0 / (121 / 24.0) * 100
    assert abs(float(out[0, 0, 0, 0]) - start) < 1e-3


def test_slot_initials_pick_the_nearest_latent_frame():
    v = mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1, 1)
    out = slot_initials_from_video(v, [0, 12, 20, 100])  # round(p/8) -> 0, 2 (1.5 rounds to 2), 2 (2.5 -> 2), clamp 4
    assert np.array(out).reshape(-1).tolist() == [0.0, 2.0, 2.0, 4.0]


def test_lead_in_latent_is_the_plane_then_the_previous_tiles_cells():
    from ltx_pipelines_mlx.dfr_layout import TilePrefix

    previous = mx.arange(19, dtype=mx.float32).reshape(1, 1, 19, 1, 1)  # tile 0 output, cell k = canvas cell k
    plane = mx.full((1, 1, 1, 1, 1), -1.0)
    prefix = TilePrefix(keyframe_position=120, video_start_cell=16, cells=4, resume_pixel=145)
    out = lead_in_latent((0, previous), prefix, plane)
    assert np.array(out).reshape(-1).tolist() == [-1.0, 16.0, 17.0, 18.0]
    # a prefixed previous tile: its cell 0 is the plane at canvas cell 15, so canvas cell 16 is its cell 1
    out = lead_in_latent((15, previous), prefix, plane)
    assert np.array(out).reshape(-1).tolist() == [-1.0, 1.0, 2.0, 3.0]
    with pytest.raises(RuntimeError, match="outside the previous tile"):
        lead_in_latent((17, previous), prefix, plane)


def test_rebase_image_conditionings_scales_filters_and_rebases():
    imgs = [ImageConditioningInput("a.png", 0, 1.0), ImageConditioningInput("b.png", 60, 0.8)]
    assert rebase_image_conditionings(imgs, pixel_scale=2, pixel_start=0, pixel_end=144) == [
        ImageConditioningInput("a.png", 0, 1.0),
        ImageConditioningInput("b.png", 120, 0.8),
    ]
    assert rebase_image_conditionings(imgs, pixel_scale=2, pixel_start=96, pixel_end=240) == [
        ImageConditioningInput("b.png", 24, 0.8)
    ]


def _pack(tmp_path: Path) -> Path:
    cfg = {"transformer": {"num_layers": 48, "ff_bias": False, "use_keyframes_abs_pos_embedding": True}}
    (tmp_path / "embedded_config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_temporal_upscalings_validated_at_construction(tmp_path):
    with pytest.raises(ValueError, match="temporal_upscalings"):
        DFRPipeline(str(_pack(tmp_path)), temporal_upscalings=3)


def test_temporal_upsampler_resolves_from_the_pack(tmp_path):
    pack = _pack(tmp_path)
    (pack / f"{TEMPORAL_UPSAMPLER_STEM}.safetensors").write_bytes(b"")
    pipe = DFRPipeline(str(pack), temporal_upscalings=1)
    assert pipe._resolve_temporal_upsampler_path() == pack / f"{TEMPORAL_UPSAMPLER_STEM}.safetensors"


def test_temporal_upsampler_override_and_missing(tmp_path):
    pack = _pack(tmp_path)
    other = tmp_path / "t.safetensors"
    other.write_bytes(b"")
    pipe = DFRPipeline(str(pack), temporal_upscalings=1, temporal_upsampler_path=str(other))
    assert pipe._resolve_temporal_upsampler_path() == other
    pipe = DFRPipeline(str(pack), temporal_upscalings=1)
    with pytest.raises(FileNotFoundError, match=TEMPORAL_UPSAMPLER_STEM):
        pipe._resolve_temporal_upsampler_path()


def test_load_temporal_upsampler_rejects_a_spatial_module(tmp_path, monkeypatch):
    """A resolved-but-spatial (or config-less) upsampler must fail loud, not silently wreck the video."""
    pack = _pack(tmp_path)
    (pack / f"{TEMPORAL_UPSAMPLER_STEM}.safetensors").write_bytes(b"")
    pipe = DFRPipeline(str(pack), temporal_upscalings=1)

    class _Stub:
        def __init__(self, temporal_upsample: bool):
            self.temporal_upsample = temporal_upsample

    monkeypatch.setattr(pipe, "_build_upsampler", lambda path: _Stub(temporal_upsample=False))
    with pytest.raises(ValueError, match="does not build a temporal upsampler"):
        pipe._load_temporal_upsampler()

    monkeypatch.setattr(pipe, "_build_upsampler", lambda path: _Stub(temporal_upsample=True))
    assert pipe._load_temporal_upsampler().temporal_upsample is True


def test_upsample_latent_takes_an_explicit_upsampler(tmp_path):
    from tests.test_ltx25_distilled import _FakeVaeEncoder

    pipe = DFRPipeline(str(_pack(tmp_path)))
    pipe.vae_encoder = _FakeVaeEncoder()
    pipe.upsampler = lambda x: x * 2
    x = mx.ones((1, 2, 3, 1, 1))
    assert float(pipe._upsample_latent(x).sum()) == 12.0  # default: self.upsampler
    assert float(pipe._upsample_latent(x, upsampler=lambda y: y * 3).sum()) == 18.0


def test_detach_removes_only_the_detailing_source_under_low_ram(tmp_path):
    pipe = DFRPipeline(str(_pack(tmp_path)), low_ram_streaming=True)

    class _Dit:
        pass

    dit = _Dit()
    user_src, detail_src = object(), object()
    object.__setattr__(dit, "_lora_sources", [user_src, detail_src])
    pipe.dit = dit
    pipe._detailing_source = detail_src
    pipe._detach_detailing_lora()
    assert pipe.dit is dit and object.__getattribute__(dit, "_lora_sources") == [user_src]
    assert pipe._detailing_source is None


def test_detach_reloads_a_clean_transformer_without_low_ram(tmp_path, monkeypatch):
    pack = _pack(tmp_path)
    (pack / "transformer-distilled.safetensors").write_bytes(b"")
    pipe = DFRPipeline(str(pack), low_ram_streaming=False)
    pipe.dit = "fused"
    loaded = []
    monkeypatch.setattr(pipe, "_load_transformer_with_optional_streaming", lambda p: loaded.append(p) or "clean")
    pipe._detach_detailing_lora()
    assert pipe.dit == "clean" and loaded[0].name.startswith("transformer")


# ---- temporal rounds end to end (test doubles of tests/test_dfr.py) -------------------------------


def _fake_temporal(x):
    rep = mx.repeat(x, 2, axis=2)
    return rep[:, :, 1:]  # 2L -> drop the first -> 2L - 1


def _make_rounds(tmp_path, monkeypatch, *, t=1, low_ram=False):
    pipe, euler, ancestral, noised, attached = _make(tmp_path, monkeypatch, low_ram=low_ram)
    # generate_two_stage resolves the temporal upsampler path up front (before any Gemma load).
    (tmp_path / f"{TEMPORAL_UPSAMPLER_STEM}.safetensors").write_bytes(b"")
    pipe.temporal_upscalings = t
    detached = []
    monkeypatch.setattr(pipe, "_detach_detailing_lora", lambda: detached.append(True))
    monkeypatch.setattr(pipe, "_load_temporal_upsampler", lambda: _fake_temporal)
    return pipe, euler, ancestral, noised, attached, detached


def test_round_1_on_121_frames(tmp_path, monkeypatch):
    pipe, _, ancestral, noised, _, detached = _make_rounds(tmp_path, monkeypatch, t=1)
    video, audio = _run(pipe, num_frames=121)
    assert detached == [True]
    assert video.shape[2] == (241 - 1) // 8 + 1  # 31 latent frames
    tiles = ancestral.calls[2:]  # calls 0/1 are stages 1/2
    assert len(tiles) == 2
    assert [c["noise_seed"] for c in tiles] == [7 + 1000, 7 + 1001]
    assert all(c["stepper"].eta == 0.5 and c["sigmas"] == [0.975, 0.909375, 0.725, 0.421875, 0.0] for c in tiles)
    # carry bag after round 1: seams (x2) + slots, every 24 frames
    assert pipe.generated_keyframe_positions == list(range(24, 241, 24))
    assert pipe.generated_keyframes.shape[2] == 10


def test_round_tiles_use_anchors_slots_and_frozen_audio(tmp_path, monkeypatch):
    from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex

    pipe, _, ancestral, noised, _, _ = _make_rounds(tmp_path, monkeypatch, t=1)
    _run(pipe, num_frames=121)
    tile_calls = [kw for kw in noised if kw["sigma"] == 0.975]
    assert len(tile_calls) == 2
    anchors = [c for c in tile_calls[0]["conditionings"] if isinstance(c, VideoConditionByKeyframeIndex)]
    assert [a.frame_idx for a in anchors] == [48, 96, 144] and all(a.strength == 0.95 for a in anchors)
    slots = [c for c in tile_calls[0]["conditionings"] if isinstance(c, VideoGeneratedKeyframeSlots)]
    assert list(slots[0].pixel_frame_indices) == [24, 72, 120]
    for call in ancestral.calls[2:]:
        assert float(mx.abs(call["audio_state"].denoise_mask).max()) == 0.0  # frozen audio
        assert call["audio_state"].frozen is True  # model conditions its audio AdaLN / A->V gate on sigma 0


def test_round_tile_1_starts_on_a_plane_with_a_pinned_prefix(tmp_path, monkeypatch):
    """Upstream v1.4.0: a non-first tile starts on the plane at the last keyframe before its seam
    (tile 0's fresh slot at 120), pins the 4 cells up to the seam (144) to tile 0's output, keeps
    only the anchors after the seam, and stitches only what it denoised."""
    from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex

    pipe, _, ancestral, noised, _, _ = _make_rounds(tmp_path, monkeypatch, t=1)
    video, _ = _run(pipe, num_frames=121)
    tile_calls = [kw for kw in noised if kw["sigma"] == 0.975]
    t1 = tile_calls[1]
    assert t1["spatial_dims"][0] == 1 + (31 - 16)  # the plane, then canvas cells 16..30
    anchors = [c for c in t1["conditionings"] if isinstance(c, VideoConditionByKeyframeIndex)]
    assert [a.frame_idx for a in anchors] == [192 - 120, 240 - 120]
    slots = [c for c in t1["conditionings"] if isinstance(c, VideoGeneratedKeyframeSlots)]
    assert list(slots[0].pixel_frame_indices) == [168 - 120, 216 - 120]
    (lead,) = [c for c in t1["conditionings"] if isinstance(c, VideoConditionByLatentIndex)]
    assert lead.frame_indices == [0, 1, 2, 3] and lead.strength == 1.0
    # tile 0 had no prefix
    assert not [c for c in tile_calls[0]["conditionings"] if isinstance(c, VideoConditionByLatentIndex)]
    # the tile's audio window starts at the plane and covers the prefix too
    assert ancestral.calls[3]["audio_state"].latent.shape[1] == compute_audio_token_count(121, frame_rate=60.0)
    assert video.shape[2] == 31  # 19 + (16 - 4): kept runs are disjoint


def test_round_2_runs_4_tiles_and_outputs_481_frames(tmp_path, monkeypatch):
    pipe, _, ancestral, _, _, _ = _make_rounds(tmp_path, monkeypatch, t=2)
    video, _ = _run(pipe, num_frames=121)
    assert len(ancestral.calls) == 2 + 2 + 4
    assert video.shape[2] == (481 - 1) // 8 + 1


def test_rounds_trim_to_the_requested_duration(tmp_path, monkeypatch):
    pipe, *_ = _make_rounds(tmp_path, monkeypatch, t=1)
    video, audio = _run(pipe, num_frames=137)  # canvas 145 -> 289 frames after round 1
    target = (137 - 1) * 2 + 1
    assert video.shape[2] == (target - 1) // 8 + 1
    assert audio.shape[2] == compute_audio_token_count(target, frame_rate=48.0)


def test_rounds_on_a_single_segment_canvas(tmp_path, monkeypatch):
    pipe, _, ancestral, _, _, _ = _make_rounds(tmp_path, monkeypatch, t=2)
    video, _ = _run(pipe, num_frames=9)  # canvas 25, one seam
    assert video.shape[2] == ((9 - 1) * 4) // 8 + 1


def test_rounds_rebase_images_into_their_tiles(tmp_path, monkeypatch):
    pipe, _, _, noised, _, _ = _make_rounds(tmp_path, monkeypatch, t=1)
    seen = []
    import ltx_pipelines_mlx.utils._orchestration as orch

    # stages 1/2 import the helper locally from _orchestration at call time; the rounds use dfr's import
    monkeypatch.setattr(orch, "combined_image_conditionings", lambda imgs, **kw: [])
    monkeypatch.setattr(dfr_mod, "combined_image_conditionings", lambda imgs, **kw: seen.append(list(imgs)) or [])
    _run(pipe, num_frames=121, images=[ImageConditioningInput("a.png", 60, 1.0)])
    # 60 * 2 = 120: inside tile 0 [0, 144] and tile 1, which starts on the plane at 120 and spans
    # [120, 240] -> local 120 and 0. Upstream rebases on the whole tile window, prefix included.
    assert [[i.frame_idx for i in s] for s in seen] == [[120], [0]]


def test_rounds_carry_the_checkpoint_image_crf(tmp_path, monkeypatch):
    """Unset CRFs are resolved once from the pack (2.5 -> 18) and reach every re-encode; explicit ones win."""
    pipe, *_ = _make_rounds(tmp_path, monkeypatch, t=1)
    mx.save_safetensors(
        str(tmp_path / "vae_encoder.safetensors"), {"w": mx.zeros((1,))}, metadata={"model_version": "2.5.0"}
    )
    seen = []
    import ltx_pipelines_mlx.utils._orchestration as orch

    monkeypatch.setattr(orch, "combined_image_conditionings", lambda imgs, **kw: seen.append(list(imgs)) or [])
    monkeypatch.setattr(dfr_mod, "combined_image_conditionings", lambda imgs, **kw: seen.append(list(imgs)) or [])
    images = [ImageConditioningInput("a.png", 0, 1.0), ImageConditioningInput("b.png", 60, 1.0, 0)]
    _run(pipe, num_frames=121, images=images)
    assert len(seen) >= 3  # stage 1, stage 2, the round tiles
    crfs = {(i.path, i.crf) for s in seen for i in s}
    assert crfs == {("a.png", 18), ("b.png", 0)}


def test_legacy_image_shorthand_gets_the_checkpoint_crf(tmp_path, monkeypatch):
    """``image=`` (frame 0, strength 1.0) is resolved like ``images=``: unversioned pack -> 33."""
    pipe, *_ = _make_rounds(tmp_path, monkeypatch, t=1)
    seen = []
    import ltx_pipelines_mlx.utils._orchestration as orch

    monkeypatch.setattr(orch, "combined_image_conditionings", lambda imgs, **kw: seen.append(list(imgs)) or [])
    monkeypatch.setattr(dfr_mod, "combined_image_conditionings", lambda imgs, **kw: seen.append(list(imgs)) or [])
    _run(pipe, num_frames=121, image="a.png")
    assert seen and {(i.path, i.frame_idx, i.crf) for s in seen for i in s} == {("a.png", 0, 33)}


def test_rounds_refuse_modality_tiling_and_prompt_relay(tmp_path, monkeypatch):
    pipe, *_ = _make_rounds(tmp_path, monkeypatch, t=1)
    with pytest.raises(ValueError, match="Prompt Relay"):
        _run(pipe, num_frames=49, prompt_relay=object())
    pipe._tile_count = object()
    with pytest.raises(ValueError, match="modality tiling"):
        _run(pipe, num_frames=49)


def test_rounds_run_with_frozen_audio_even_without_audio_output(tmp_path, monkeypatch):
    pipe, _, ancestral, _, _, _ = _make_rounds(tmp_path, monkeypatch, t=1)
    pipe.generate_audio = False
    _run(pipe, num_frames=49)
    assert all(c["audio_state"] is not None for c in ancestral.calls[2:])


def test_t0_path_is_unchanged(tmp_path, monkeypatch):
    pipe, _, ancestral, _, _, detached = _make_rounds(tmp_path, monkeypatch, t=0)
    video, _ = _run(pipe, num_frames=49)
    assert detached == [] and len(ancestral.calls) == 2 and video.shape[2] == 7


def test_decode_writes_at_the_upsampled_fps(tmp_path, monkeypatch):
    pipe, *_ = _make_rounds(tmp_path, monkeypatch, t=2)
    seen = {}
    base = next(c for c in type(pipe).__mro__[1:] if "_decode_and_save_video" in c.__dict__)
    monkeypatch.setattr(
        base,
        "_decode_and_save_video",
        lambda self, v, a, o, *, frame_rate, seed=0, keyframes=None: seen.setdefault("fps", frame_rate) and o,
    )
    pipe.generated_keyframes = None
    pipe._decode_and_save_video(mx.zeros((1, 128, 3, 2, 2)), mx.zeros((1, 8, 4, 16)), "o.mp4", frame_rate=24.0)
    assert seen["fps"] == 96.0


def test_rounds_reload_the_vae_encoder_freed_by_low_memory_stage2(tmp_path, monkeypatch):
    """low_memory (the default) frees the VAE encoder in stage 2; the rounds need it for denorm/renorm."""
    from tests.test_ltx25_distilled import _FakeVaeEncoder

    pipe, _, ancestral, _, _, _ = _make_rounds(tmp_path, monkeypatch, t=1)
    pipe.low_memory = True
    reloads = []

    def load():
        reloads.append(True)
        pipe.image_conditioner._encoder = _FakeVaeEncoder()
        return pipe.image_conditioner._encoder

    monkeypatch.setattr(pipe.image_conditioner, "load", load)
    video, _ = _run(pipe, num_frames=49)
    assert reloads == [True] and len(ancestral.calls) == 4
    assert video.shape[2] == ((49 - 1) * 2) // 8 + 1


def test_rounds_release_the_fused_stage1_dit(tmp_path, monkeypatch):
    """Stage1Result.x0_model wraps the detailing-fused DiT; it must not stay resident beside the clean reload."""
    pipe, *_ = _make_rounds(tmp_path, monkeypatch, t=1, low_ram=False)
    seen = []
    real = pipe._denoise_temporal_tile

    def spy(stage1, *args, **kwargs):
        seen.append(stage1.x0_model)
        return real(stage1, *args, **kwargs)

    monkeypatch.setattr(pipe, "_denoise_temporal_tile", spy)
    _run(pipe, num_frames=49)
    assert seen and all(x0 is None for x0 in seen)
