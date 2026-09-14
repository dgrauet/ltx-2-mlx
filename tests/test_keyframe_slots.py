"""Generated keyframe slots + keyframe absolute-position marker (#127, sub-project 2).

Layer 1: ``LatentState.keyframes_mask`` marks single-pixel-frame tokens. Upstream marks the
target's first latent frame unconditionally (the causal encoder makes it cover one pixel
frame) and every appended-token item extends the mask, marked only for generated slots.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.conditioning.mask_utils import extend_keyframes_mask
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, create_initial_state
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.utils.positions import compute_video_positions
from ltx_pipelines_mlx.utils.helpers import create_noised_state

F, H, W, C = 3, 2, 2, 16
N = F * H * W
TPF = H * W  # tokens per latent frame


def _positions() -> mx.array:
    return compute_video_positions(F, H, W, frame_rate=24.0)


def _state(num_tokens: int = N, keyframes_mask=None) -> LatentState:
    return LatentState(
        latent=mx.zeros((1, num_tokens, C)),
        clean_latent=mx.zeros((1, num_tokens, C)),
        denoise_mask=mx.ones((1, num_tokens, 1)),
        positions=_positions(),
        keyframes_mask=keyframes_mask,
    )


def test_latent_state_new_fields_default_to_none():
    s = _state()
    assert s.keyframes_mask is None
    assert s.generated_keyframe_layout is None
    assert s.generated_keyframes is None


def test_extend_keyframes_mask_stays_none_when_unmarked_and_absent():
    assert extend_keyframes_mask(_state(), 4, marked=False) is None


def test_extend_keyframes_mask_creates_mask_for_marked_tokens():
    m = extend_keyframes_mask(_state(), 4, marked=True)
    assert m.shape == (1, N + 4, 1)
    assert mx.array_equal(m[:, :N], mx.zeros((1, N, 1)))
    assert mx.array_equal(m[:, N:], mx.ones((1, 4, 1)))


def test_extend_keyframes_mask_appends_zeros_to_existing_mask_when_unmarked():
    existing = mx.concatenate([mx.ones((1, TPF, 1)), mx.zeros((1, N - TPF, 1))], axis=1)
    m = extend_keyframes_mask(_state(keyframes_mask=existing), 3, marked=False)
    assert mx.array_equal(m[:, :N], existing)
    assert mx.array_equal(m[:, N:], mx.zeros((1, 3, 1)))


@pytest.mark.parametrize("legacy", [False, True])
def test_create_noised_state_marks_first_latent_frame(legacy):
    s = create_noised_state((1, N, C), [], (F, H, W), _positions(), seed=0, legacy_scalar_blend=legacy)
    expected = mx.concatenate([mx.ones((1, TPF, 1)), mx.zeros((1, N - TPF, 1))], axis=1)
    assert mx.array_equal(s.keyframes_mask, expected)


def test_create_initial_state_marks_first_latent_frame():
    s = create_initial_state((1, N, C), seed=0, positions=_positions(), tokens_per_frame=TPF)
    assert mx.array_equal(s.keyframes_mask[:, :TPF], mx.ones((1, TPF, 1)))
    assert mx.array_equal(s.keyframes_mask[:, TPF:], mx.zeros((1, N - TPF, 1)))


def test_keyframe_index_item_extends_mask_unmarked():
    kf = VideoConditionByKeyframeIndex(
        frame_idx=8, keyframe_latent=mx.zeros((1, TPF, C)), spatial_dims=(F, H, W), frame_rate=24.0
    )
    s = create_noised_state((1, N, C), [kf], (F, H, W), _positions(), seed=0)
    assert s.latent.shape[1] == N + TPF
    assert s.keyframes_mask.shape == (1, N + TPF, 1)
    assert mx.array_equal(s.keyframes_mask[:, N:], mx.zeros((1, TPF, 1)))
    assert mx.array_equal(s.keyframes_mask[:, :TPF], mx.ones((1, TPF, 1)))


def test_reference_latent_item_extends_mask_unmarked():
    ref = VideoConditionByReferenceLatent(
        reference_latent=mx.zeros((1, N, C)), reference_positions=_positions(), strength=1.0
    )
    s = create_noised_state((1, N, C), [ref], (F, H, W), _positions(), seed=0)
    assert s.keyframes_mask.shape == (1, 2 * N, 1)
    assert mx.array_equal(s.keyframes_mask[:, N:], mx.zeros((1, N, 1)))


# ---------------------------------------------------------------------------
# Layer 2: DiT applies the learned marker; Modality / tiling / samplers carry the mask
# ---------------------------------------------------------------------------

from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler  # noqa: E402
from ltx_core_mlx.model.transformer.modality import Modality  # noqa: E402
from ltx_core_mlx.model.transformer.model import (  # noqa: E402
    LTXModel,
    LTXModelConfig,
    apply_keyframes_absolute_embedding,
)
from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig  # noqa: E402
from ltx_pipelines_mlx.utils.samplers import denoise_loop  # noqa: E402


def _tiny_config(**overrides) -> LTXModelConfig:
    kw = dict(
        num_layers=1,
        video_dim=32,
        audio_dim=16,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=4,
        av_cross_head_dim=4,
        video_patch_channels=8,
        audio_patch_channels=8,
        ff_mult=2.0,
        timestep_embedding_dim=32,
    )
    kw.update(overrides)
    return LTXModelConfig(**kw)


def _grid_positions(f: int, h: int, w: int) -> mx.array:
    f_idx = mx.arange(f)[:, None, None].astype(mx.float32) * mx.ones((f, h, w), dtype=mx.float32)
    h_idx = mx.arange(h)[None, :, None].astype(mx.float32) * mx.ones((f, h, w), dtype=mx.float32)
    w_idx = mx.arange(w)[None, None, :].astype(mx.float32) * mx.ones((f, h, w), dtype=mx.float32)
    return mx.stack([f_idx, h_idx, w_idx], axis=-1).reshape(1, f * h * w, 3)


def _forward_kwargs(cfg: LTXModelConfig, f: int = 1, h: int = 1, w: int = 16) -> dict:
    mx.random.seed(3)
    nv, na, nt = f * h * w, 4, 4
    return dict(
        video_latent=mx.random.normal((1, nv, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, na, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([0.5]),
        video_text_embeds=mx.random.normal((1, nt, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, nt, cfg.audio_dim)).astype(mx.bfloat16),
        video_positions=_grid_positions(f, h, w),
        audio_positions=mx.zeros((1, na, 1)),
    )


def test_apply_keyframes_absolute_embedding_adds_marker_only_to_marked_tokens():
    hidden = mx.zeros((1, 4, 3))
    mask = mx.array([[[1.0], [0.0], [0.0], [1.0]]])
    emb = mx.array([[1.0, 2.0, 3.0]])
    out = apply_keyframes_absolute_embedding(hidden, mask, emb)
    assert mx.array_equal(out[:, 0], emb) and mx.array_equal(out[:, 3], emb)
    assert mx.array_equal(out[:, 1:3], mx.zeros((1, 2, 3)))
    assert apply_keyframes_absolute_embedding(hidden, None, emb) is hidden
    assert apply_keyframes_absolute_embedding(hidden, mask, None) is hidden


def test_model_without_embedding_param_ignores_mask_bit_exactly():
    """2.3 checkpoints: no parameter, so a mask changes nothing."""
    cfg = _tiny_config(use_keyframes_abs_pos_embedding=False)
    mx.random.seed(7)
    model = LTXModel(cfg)
    mx.eval(model.parameters())
    kw = _forward_kwargs(cfg)
    v0, a0 = model(**kw)
    v1, a1 = model(**kw, video_keyframes_mask=mx.ones((1, 16, 1)))
    mx.eval(v0, a0, v1, a1)
    assert mx.array_equal(v0, v1) and mx.array_equal(a0, a1)


def test_model_with_learned_embedding_changes_marked_tokens_only_via_mask():
    cfg = _tiny_config(use_keyframes_abs_pos_embedding=True)
    mx.random.seed(7)
    model = LTXModel(cfg)
    model.keyframes_abs_pos_embedding = mx.random.normal((1, cfg.video_dim)) * 0.5
    mx.eval(model.parameters())
    kw = _forward_kwargs(cfg)
    v_none, _ = model(**kw)
    v_zero, _ = model(**kw, video_keyframes_mask=mx.zeros((1, 16, 1)))
    v_marked, _ = model(**kw, video_keyframes_mask=mx.ones((1, 16, 1)))
    mx.eval(v_none, v_zero, v_marked)
    assert mx.array_equal(v_none, v_zero), "an all-zero mask must equal no mask"
    assert not mx.array_equal(v_none, v_marked), "marked tokens must see the learned embedding"


def test_modality_carries_keyframes_mask_and_tiling_slices_it():
    f, h, w = 2, 4, 4
    n = f * h * w
    latent = mx.random.normal((1, n, 8))
    mask = mx.concatenate([mx.ones((1, h * w, 1)), mx.zeros((1, n - h * w, 1))], axis=1)
    modality = Modality(
        latent=latent,
        sigma=mx.zeros((1,)),
        timesteps=mx.zeros((1, n)),
        positions=_grid_positions(f, h, w),
        context=mx.zeros((1, 0, 0)),
        keyframes_mask=mask,
    )
    tiling = TileCountConfig(frames=DimensionTilingConfig(num_tiles=2, overlap=0))
    tiler = VideoModalityTiler(tiling, latent_shape=(f, h, w))
    first, _ = tiler.tile_modality(modality, tiler.tiles[0], normalize_positions=False)
    second, _ = tiler.tile_modality(modality, tiler.tiles[1], normalize_positions=False)
    assert first.keyframes_mask.shape == (1, h * w, 1) and bool(mx.all(first.keyframes_mask == 1.0).item())
    assert bool(mx.all(second.keyframes_mask == 0.0).item())


def test_tiled_wrapper_forwards_keyframes_mask_per_tile():
    cfg = _tiny_config(use_keyframes_abs_pos_embedding=True)
    mx.random.seed(7)
    model = LTXModel(cfg)
    model.keyframes_abs_pos_embedding = mx.random.normal((1, cfg.video_dim)) * 0.5
    mx.eval(model.parameters())
    kw = _forward_kwargs(cfg)
    mask = mx.concatenate([mx.ones((1, 8, 1)), mx.zeros((1, 8, 1))], axis=1)
    baseline_v, _ = model(**kw, video_keyframes_mask=mask)
    wrapped = TiledLTXModel(model, VideoModalityTiler(TileCountConfig(), latent_shape=(1, 1, 16)))
    wrapped_v, _ = wrapped(**kw, video_keyframes_mask=mask)
    mx.eval(baseline_v, wrapped_v)
    assert mx.allclose(baseline_v, wrapped_v, atol=1e-5, rtol=1e-5).item()


def test_samplers_pass_state_keyframes_mask_to_model():
    seen: list = []

    class _Recorder:
        def __call__(self, *, video_latent, audio_latent, **kwargs):
            seen.append(kwargs.get("video_keyframes_mask"))
            return mx.zeros_like(video_latent), mx.zeros_like(audio_latent)

    n = 12
    mask = mx.concatenate([mx.ones((1, 4, 1)), mx.zeros((1, n - 4, 1))], axis=1)
    video = LatentState(
        latent=mx.ones((1, n, 16)),
        clean_latent=mx.zeros((1, n, 16)),
        denoise_mask=mx.ones((1, n, 1)),
        keyframes_mask=mask,
    )
    audio = LatentState(latent=mx.ones((1, 3, 16)), clean_latent=mx.zeros((1, 3, 16)), denoise_mask=mx.ones((1, 3, 1)))
    denoise_loop(
        model=_Recorder(),
        video_state=video,
        audio_state=audio,
        sigmas=[1.0, 0.5, 0.0],
        video_text_embeds=mx.zeros((1, 4, 16)),
        audio_text_embeds=mx.zeros((1, 4, 16)),
        show_progress=False,
    )
    assert seen and all(m is not None and mx.array_equal(m, mask) for m in seen)


# ---------------------------------------------------------------------------
# Layer 3: the VideoGeneratedKeyframeSlots item, slot noising, extraction, spacing
# ---------------------------------------------------------------------------

from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier  # noqa: E402
from ltx_core_mlx.conditioning.types.keyframe_cond import _compute_keyframe_positions  # noqa: E402
from ltx_core_mlx.conditioning.types.keyframe_slots import (  # noqa: E402
    VideoGeneratedKeyframeSlots,
    extract_generated_keyframes,
)
from ltx_pipelines_mlx.utils.helpers import (  # noqa: E402
    evenly_spaced_keyframe_positions,
    generated_keyframe_conditionings,
    has_generated_keyframes,
    resolve_generated_keyframes,
)

NUM_PIXEL_FRAMES = (F - 1) * 8 + 1  # 17 for F=3


def _slots(*indices: int, initial=None) -> VideoGeneratedKeyframeSlots:
    return VideoGeneratedKeyframeSlots(pixel_frame_indices=indices, frame_rate=24.0, initial_keyframes=initial)


def test_slots_validate_indices():
    with pytest.raises(ValueError, match="non-empty"):
        _slots()
    with pytest.raises(ValueError, match="non-negative"):
        _slots(-1)
    with pytest.raises(ValueError, match="strictly increasing"):
        _slots(4, 4)
    with pytest.raises(ValueError, match="outside"):
        _slots(NUM_PIXEL_FRAMES).apply(_state(), (F, H, W))


def test_slots_append_marked_fully_denoised_tokens_with_single_frame_positions():
    s = _slots(5, 11).apply(_state(), (F, H, W))
    assert s.latent.shape == (1, N + 2 * TPF, C)
    assert mx.array_equal(s.latent[:, N:], mx.zeros((1, 2 * TPF, C)))
    assert mx.array_equal(s.clean_latent[:, N:], mx.zeros((1, 2 * TPF, C)))
    assert mx.array_equal(s.denoise_mask[:, N:], mx.ones((1, 2 * TPF, 1)))
    assert mx.array_equal(s.keyframes_mask[:, N:], mx.ones((1, 2 * TPF, 1)))
    assert mx.array_equal(s.keyframes_mask[:, :N], mx.zeros((1, N, 1)))  # bare state had no mask
    # Each slot sits at temporal span [t, t+1) -> midpoint (t + 0.5) / fps, full spatial grid.
    expected = mx.concatenate(
        [_compute_keyframe_positions(5, H, W, 24.0), _compute_keyframe_positions(11, H, W, 24.0)], axis=1
    )
    assert mx.allclose(s.positions[:, N:], expected, atol=1e-6).item()
    assert s.generated_keyframe_layout.pixel_frame_indices == (5, 11)
    assert s.generated_keyframe_layout.tokens_per_keyframe == TPF
    assert s.generated_keyframe_layout.first_token == N
    assert s.generated_keyframe_layout.token_slice == slice(N, N + 2 * TPF)


def test_slots_keep_existing_first_frame_marker():
    s = create_noised_state((1, N, C), [_slots(8)], (F, H, W), _positions(), seed=0)
    assert mx.array_equal(s.keyframes_mask[:, :TPF], mx.ones((1, TPF, 1)))
    assert mx.array_equal(s.keyframes_mask[:, TPF:N], mx.zeros((1, N - TPF, 1)))
    assert mx.array_equal(s.keyframes_mask[:, N:], mx.ones((1, TPF, 1)))


def test_slots_refuse_to_be_applied_twice():
    s = _slots(5).apply(_state(), (F, H, W))
    with pytest.raises(ValueError, match="already applied"):
        _slots(9).apply(s, (F, H, W))


def test_slots_seed_latent_from_initial_keyframes_but_not_clean():
    initial = mx.random.normal((1, C, 2, H, W))
    s = _slots(5, 11, initial=initial).apply(_state(), (F, H, W))
    tokens, _ = VideoLatentPatchifier().patchify(initial[:, :, 0:1])
    assert mx.array_equal(s.latent[:, N : N + TPF], tokens)
    assert mx.array_equal(s.clean_latent[:, N:], mx.zeros((1, 2 * TPF, C)))
    with pytest.raises(ValueError, match="K="):
        _slots(5, initial=initial)


@pytest.mark.parametrize("legacy", [False, True])
def test_slot_tokens_are_noised_on_both_noising_paths(legacy):
    """Slots are appended with zero latent and denoise_mask=1, so at sigma=1 they must be pure noise."""
    s = create_noised_state((1, N, C), [_slots(8)], (F, H, W), _positions(), seed=0, legacy_scalar_blend=legacy)
    slot = s.latent[:, N:]
    assert not mx.array_equal(slot, mx.zeros_like(slot))
    assert abs(float(mx.var(slot.astype(mx.float32)).item()) - 1.0) < 0.5


def test_extract_generated_keyframes_unpatchifies_each_slot_standalone():
    s = _slots(5, 11).apply(_state(), (F, H, W))
    filled = mx.random.normal((1, 2 * TPF, C))
    tokens = mx.concatenate([s.latent[:, :N], filled], axis=1)
    kf = extract_generated_keyframes(tokens, s.generated_keyframe_layout, VideoLatentPatchifier(), (H, W))
    assert kf.shape == (1, C, 2, H, W)
    single, _ = VideoLatentPatchifier().patchify(kf[:, :, 1:2])
    assert mx.array_equal(single, filled[:, TPF:])
    assert extract_generated_keyframes(tokens, None, VideoLatentPatchifier(), (H, W)) is None
    with pytest.raises(ValueError, match="tokens per keyframe"):
        extract_generated_keyframes(tokens, s.generated_keyframe_layout, VideoLatentPatchifier(), (H, W + 1))


def test_evenly_spaced_positions_match_upstream_linspace_round():
    assert evenly_spaced_keyframe_positions(0, 97) == []
    assert evenly_spaced_keyframe_positions(1, 97) == [48]
    assert evenly_spaced_keyframe_positions(3, 97) == [24, 48, 72]
    assert evenly_spaced_keyframe_positions(5, 121) == [20, 40, 60, 80, 100]
    assert evenly_spaced_keyframe_positions(2, 9) == [3, 5]  # linspace(0, 8, 4) = 0, 2.67, 5.33, 8
    with pytest.raises(ValueError, match="at least"):
        evenly_spaced_keyframe_positions(8, 9)
    with pytest.raises(ValueError, match="non-negative"):
        evenly_spaced_keyframe_positions(-1, 9)


def test_resolve_and_conditionings_helpers():
    assert has_generated_keyframes(0) is False and has_generated_keyframes(2) is True
    assert has_generated_keyframes([]) is False and has_generated_keyframes([4]) is True
    assert resolve_generated_keyframes([9, 3, 3], 17) == [3, 9]
    with pytest.raises(ValueError, match=r"\[0, 17\)"):
        resolve_generated_keyframes([17], 17)
    assert generated_keyframe_conditionings(0, 17, frame_rate=24.0) == []
    (item,) = generated_keyframe_conditionings(2, 17, frame_rate=24.0)
    assert isinstance(item, VideoGeneratedKeyframeSlots) and item.pixel_frame_indices == (5, 11)


# ---------------------------------------------------------------------------
# Layer 4: pipelines refuse unsupported packs up front, forward the request, CLI flag
# ---------------------------------------------------------------------------

import json  # noqa: E402

from ltx_pipelines_mlx._base import BasePipeline  # noqa: E402


def _pipe_with_config(tmp_path, transformer_cfg: dict | None) -> BasePipeline:
    if transformer_cfg is not None:
        (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer_cfg}))
    p = BasePipeline.__new__(BasePipeline)
    p.model_dir = tmp_path
    return p


def test_require_generated_keyframes_support_passes_on_keyframe_checkpoint(tmp_path):
    p = _pipe_with_config(tmp_path, {"use_keyframes_abs_pos_embedding": True})
    p._require_generated_keyframes_support(3)
    p._require_generated_keyframes_support([4, 8])


def test_require_generated_keyframes_support_fails_fast_without_embedding(tmp_path):
    p = _pipe_with_config(tmp_path, {"use_keyframes_abs_pos_embedding": False})
    with pytest.raises(ValueError, match="use_keyframes_abs_pos_embedding"):
        p._require_generated_keyframes_support(3)
    # Off is always fine, whatever the checkpoint.
    p._require_generated_keyframes_support(0)
    p._require_generated_keyframes_support([])


def test_pipeline_generated_keyframes_attribute_defaults_to_none():
    assert BasePipeline.__new__(BasePipeline).generated_keyframes is None


def _parse_generate_args(*extra: str):
    from ltx_pipelines_mlx.cli import _build_parser

    return _build_parser().parse_args(
        ["generate", "-p", "a fox", "-o", "out.mp4", "--frame-rate", "24", "-f", "17", *extra]
    )


def test_cli_num_generated_keyframes_defaults_to_zero_and_parses():
    assert _parse_generate_args("--distilled").num_generated_keyframes == 0
    assert _parse_generate_args("--distilled", "--num-generated-keyframes", "3").num_generated_keyframes == 3


@pytest.mark.parametrize("mode", ["--distilled", "--one-stage", "--two-stage", "--two-stages-hq"])
def test_cli_num_generated_keyframes_reaches_every_generate_mode(monkeypatch, mode):
    seen = {}

    class _FakePipe:
        def __init__(self, *args, **kwargs):
            pass

        def generate_and_save(self, **kwargs):
            seen.update(kwargs)

    import ltx_pipelines_mlx.distilled as distilled
    import ltx_pipelines_mlx.ti2vid_one_stage as one_stage
    import ltx_pipelines_mlx.ti2vid_two_stages as two_stages
    import ltx_pipelines_mlx.ti2vid_two_stages_hq as two_stages_hq

    monkeypatch.setattr(distilled, "DistilledPipeline", _FakePipe)
    monkeypatch.setattr(one_stage, "TI2VidOneStagePipeline", _FakePipe)
    monkeypatch.setattr(two_stages, "TI2VidTwoStagesPipeline", _FakePipe)
    monkeypatch.setattr(two_stages_hq, "TI2VidTwoStagesHQPipeline", _FakePipe)

    from ltx_pipelines_mlx.cli import _cmd_generate

    _cmd_generate(_parse_generate_args(mode, "--num-generated-keyframes", "3", "--quiet"))
    assert seen["generated_keyframes"] == 3


@pytest.mark.parametrize(
    ("module", "cls", "inner"),
    [
        ("ltx_pipelines_mlx.ti2vid_one_stage", "TI2VidOneStagePipeline", "generate_one_stage_dev"),
        ("ltx_pipelines_mlx.ti2vid_two_stages", "TI2VidTwoStagesPipeline", "generate_two_stage"),
    ],
)
def test_generate_and_save_forwards_generated_keyframes(monkeypatch, tmp_path, module, cls, inner):
    import importlib

    pipe_cls = getattr(importlib.import_module(module), cls)
    p = pipe_cls.__new__(pipe_cls)
    p.low_memory = False
    p.verbose = False
    p.dit = None
    p._loaded = False
    p.generate_audio = True
    seen = {}

    def _inner(**kwargs):
        seen.update(kwargs)
        return mx.zeros((1, 1, 1)), mx.zeros((1, 1, 1))

    monkeypatch.setattr(p, inner, _inner)
    monkeypatch.setattr(p, "_load_decoders", lambda: None)
    monkeypatch.setattr(p, "_decode_and_save_video", lambda *a, **k: str(tmp_path / "o.mp4"))
    p.generate_and_save(
        prompt="x", output_path=str(tmp_path / "o.mp4"), num_frames=17, frame_rate=24.0, generated_keyframes=2
    )
    assert seen["generated_keyframes"] == 2
