"""Unit tests for VideoModalityTiler."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler
from ltx_core_mlx.conditioning.types.keyframe_cond import _compute_keyframe_positions
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.transformer.modality import Modality
from ltx_core_mlx.model.video_vae.tiling import (
    DimensionIntervals,
    DimensionTilingConfig,
    TileCountConfig,
    identity_mapping_operation,
    split_at_seams,
    split_by_count,
)
from ltx_core_mlx.utils.positions import compute_video_positions

FPS = 24.0


def _make_modality(latent: mx.array, positions: mx.array, attention_mask: mx.array | None = None) -> Modality:
    """Build a minimal Modality for tiler tests (timesteps, sigma, context filled with zeros)."""
    B, T = latent.shape[0], latent.shape[1]
    return Modality(
        latent=latent,
        sigma=mx.zeros((B,)),
        timesteps=mx.zeros((B, T)),
        positions=positions,
        context=mx.zeros((B, 0, 0), dtype=latent.dtype),
        enabled=True,
        context_mask=None,
        attention_mask=attention_mask,
    )


def _make_positions(F: int, H: int, W: int) -> mx.array:  # noqa: N803
    """(1, F*H*W, 3) pixel-space midpoints, as the pipelines build them."""
    return compute_video_positions(F, H, W, frame_rate=FPS)


def _gen_count(tile) -> int:
    f, h, w = tile.in_coords
    return (f.stop - f.start) * (h.stop - h.start) * (w.stop - w.start)


def _cond_keep_table(tiler: VideoModalityTiler, modality: Modality) -> list[list[int]]:
    """Per tile: the kept conditioning-token indices (relative to the first cond token)."""
    table = []
    for t in tiler.tiles:
        _, ctx = tiler.tile_modality(modality, t, normalize_positions=False)
        idx = np.asarray(ctx.keep_indices)
        table.append((idx[idx >= tiler.num_generated_tokens] - tiler.num_generated_tokens).tolist())
    return table


def _tiny_model():
    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

    cfg = LTXModelConfig(
        num_layers=2,
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
    mx.random.seed(7)
    model = LTXModel(cfg)
    mx.eval(model.parameters())
    return model, cfg


def _model_kwargs(cfg, F: int, H: int, W: int) -> dict:  # noqa: N803
    Nv, Na, Nt = F * H * W, 4, 4
    return dict(
        video_latent=mx.random.normal((1, Nv, cfg.video_patch_channels)).astype(mx.bfloat16),
        audio_latent=mx.random.normal((1, Na, cfg.audio_patch_channels)).astype(mx.bfloat16),
        timestep=mx.array([0.5]),
        video_text_embeds=mx.random.normal((1, Nt, cfg.video_dim)).astype(mx.bfloat16),
        audio_text_embeds=mx.random.normal((1, Nt, cfg.audio_dim)).astype(mx.bfloat16),
        video_positions=_make_positions(F, H, W),
        audio_positions=mx.zeros((1, Na, 1)),
    )


class TestVideoModalityTiler:
    def test_tile_count_with_overlap(self):
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(num_tiles=2, overlap=1),
            height=DimensionTilingConfig(num_tiles=2, overlap=2),
            width=DimensionTilingConfig(num_tiles=1),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(4, 8, 8))
        # 2 (frames) * 2 (height) * 1 (width) = 4 tiles
        assert len(tiler.tiles) == 4

    def test_single_tile_round_trip(self):
        """1x1x1 tile: blend(tile(latent)) must equal latent itself."""
        F, H, W, D = 4, 6, 8, 16
        T = F * H * W
        tiling = TileCountConfig()  # 1 tile per dim
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        assert len(tiler.tiles) == 1

        latent = mx.random.normal((1, T, D))
        positions = _make_positions(F, H, W)
        modality = _make_modality(latent, positions)
        tile = tiler.tiles[0]

        tiled, ctx = tiler.tile_modality(modality, tile, normalize_positions=False)
        assert tiled.latent.shape == latent.shape

        out = tiler.blend(tiled.latent, tile, ctx)
        mx.eval(latent, out)
        assert mx.allclose(latent, out, atol=1e-6).item()

    def test_multi_tile_no_overlap_round_trip(self):
        """Tiling with no overlap and trapezoidal mask = 1.0 everywhere
        should reconstruct identity when blend masks are uniform."""
        F, H, W, D = 4, 8, 8, 8
        T = F * H * W
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(num_tiles=2, overlap=0),
            height=DimensionTilingConfig(num_tiles=2, overlap=0),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        assert len(tiler.tiles) == 4

        latent = mx.random.normal((1, T, D))
        positions = _make_positions(F, H, W)
        modality = _make_modality(latent, positions)

        output = mx.zeros_like(latent)
        for t in tiler.tiles:
            tiled, ctx = tiler.tile_modality(modality, t, normalize_positions=False)
            output = tiler.blend(tiled.latent, t, ctx, output=output)
        mx.eval(latent, output)
        # No overlap → blend masks are all 1.0, sum gives back identity.
        assert mx.allclose(latent, output, atol=1e-6).item()

    def test_multi_tile_with_overlap_blend_weights_sum_to_one(self):
        """With overlap, blended output must equal the original where
        per-pixel blend weights across all tiles sum to 1."""
        F, H, W, D = 4, 8, 8, 4
        T = F * H * W
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(num_tiles=1),
            height=DimensionTilingConfig(num_tiles=2, overlap=2),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))

        latent = mx.random.normal((1, T, D))
        positions = _make_positions(F, H, W)
        modality = _make_modality(latent, positions)

        output = mx.zeros_like(latent)
        for t in tiler.tiles:
            tiled, ctx = tiler.tile_modality(modality, t, normalize_positions=False)
            output = tiler.blend(tiled.latent, t, ctx, output=output)
        mx.eval(latent, output)
        # Trapezoidal masks sum to 1 across the overlap → identity.
        assert mx.allclose(latent, output, atol=1e-5).item()

    def test_position_normalization(self):
        """normalize_positions=True shifts every kept position by the tile's generated-interval
        start (upstream subtracts the interval starts' minimum), so the tile's first generated
        cell looks like the canvas origin: spatial midpoints 16 px, the first temporal midpoint
        at half a latent frame (4 px) or the causal 0.5 px when the tile starts at frame 0."""
        F, H, W = 4, 8, 8
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(num_tiles=2, overlap=0),
            height=DimensionTilingConfig(num_tiles=2, overlap=0),
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))

        latent = mx.zeros((1, F * H * W, 4))
        positions = _make_positions(F, H, W)
        modality = _make_modality(latent, positions)

        for t in tiler.tiles:
            tiled, _ = tiler.tile_modality(modality, t, normalize_positions=True)
            gen_pos = np.asarray(tiled.positions[0, : _gen_count(t), :])
            first_t = 0.5 if t.in_coords[0].start == 0 else 4.0
            np.testing.assert_allclose(gen_pos.min(axis=0), [first_t / FPS, 16.0, 16.0], rtol=1e-6)

    def test_attention_mask_subset(self):
        """When attention_mask is given, the tiled mask is the kept
        rows x kept cols submatrix."""
        F, H, W = 2, 4, 4
        T = F * H * W
        tiling = TileCountConfig(height=DimensionTilingConfig(num_tiles=2, overlap=0))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))

        latent = mx.zeros((1, T, 4))
        positions = _make_positions(F, H, W)
        full_mask = mx.random.normal((1, T, T))
        modality = _make_modality(latent, positions, attention_mask=full_mask)

        tile = tiler.tiles[0]
        tiled, _ = tiler.tile_modality(modality, tile)
        assert tiled.attention_mask is not None
        assert tiled.attention_mask.shape == (1, tiled.latent.shape[1], tiled.latent.shape[1])

    def test_cond_tokens_kept_by_overlap(self):
        """Conditioning tokens inside a tile's generated extent are kept; others are dropped."""
        F, H, W, D = 2, 4, 4, 4
        T_gen = F * H * W
        # Two frame-0 keyframe cells, one in each height half of the grid (h=0 and h=3).
        kf = _compute_keyframe_positions(0, H, W, FPS)
        cond_pos = mx.concatenate([kf[:, 0:1], kf[:, 3 * W : 3 * W + 1]], axis=1)  # (1, 2, 3)
        positions = mx.concatenate([_make_positions(F, H, W), cond_pos], axis=1)
        latent = mx.random.normal((1, T_gen + 2, D))

        tiling = TileCountConfig(height=DimensionTilingConfig(num_tiles=2, overlap=0))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        modality = _make_modality(latent, positions)

        # First tile covers h=[0,2): only the first cond (h=0); second tile h=[2,4): only the second.
        assert _cond_keep_table(tiler, modality) == [[0], [1]]

    def test_keyframe_at_seam_pixel_kept_by_both_tiles(self):
        """A keyframe at pixel 8s, where the later frame tile starts at latent s, lies inside both
        tiles' generated intervals (upstream interval overlap); the earlier one used to drop it."""
        F, H, W = 4, 1, 1
        tiling = TileCountConfig(frames=DimensionTilingConfig(num_tiles=2, overlap=1))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        s = tiler.tiles[1].in_coords[0].start
        assert (tiler.tiles[0].in_coords[0], s) == (slice(0, 3), 2)

        kf = _compute_keyframe_positions(8 * s, H, W, FPS)
        positions = mx.concatenate([_make_positions(F, H, W), kf], axis=1)
        modality = _make_modality(mx.zeros((1, F + 1, 4)), positions)
        assert _cond_keep_table(tiler, modality) == [[0], [0]]

    def test_keyframe_on_last_frame_kept_by_last_tile(self):
        """A keyframe on the canvas's last pixel frame 8(L-1) belongs to the last tile."""
        F, H, W = 4, 1, 1
        tiling = TileCountConfig(frames=DimensionTilingConfig(num_tiles=2, overlap=0))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))

        kf = _compute_keyframe_positions(8 * (F - 1), H, W, FPS)
        positions = mx.concatenate([_make_positions(F, H, W), kf], axis=1)
        modality = _make_modality(mx.zeros((1, F + 1, 4)), positions)
        assert _cond_keep_table(tiler, modality) == [[], [0]]

    def test_x2_reference_cell_on_spatial_boundary_kept_by_both(self):
        """A x2 reference cell whose midpoint is a spatial tile boundary straddles both tiles."""
        F, H, W = 1, 6, 1
        tiling = TileCountConfig(height=DimensionTilingConfig(num_tiles=2, overlap=0))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        assert tiler.tiles[1].in_coords[1] == slice(3, 6)

        ref_latent = mx.zeros((1, 3, 4))
        state = LatentState(
            latent=mx.zeros((1, F * H * W, 4)),
            clean_latent=mx.zeros((1, F * H * W, 4)),
            denoise_mask=mx.ones((1, F * H * W, 1)),
            positions=_make_positions(F, H, W),
        )
        cond = VideoConditionByReferenceLatent(
            ref_latent, reference_positions=_make_positions(1, 3, 1), downscale_factor=2
        )
        state = cond.apply(state, (F, H, W))
        # Reference cell 1: midpoint 2 * 48 = 96 px = 32 * 3, the boundary between the tiles.
        assert float(state.positions[0, F * H * W + 1, 1].item()) == 96.0

        modality = _make_modality(state.latent, state.positions)
        assert _cond_keep_table(tiler, modality) == [[0, 1], [1, 2]]

    def test_cond_blend_weights_sum_to_one(self):
        """Blending tiled conditioning tokens reproduces them exactly (weights 1/keepers)."""
        F, H, W, D = 4, 1, 1, 4
        tiling = TileCountConfig(frames=DimensionTilingConfig(num_tiles=2, overlap=1))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        cond_pos = mx.concatenate([_compute_keyframe_positions(p, H, W, FPS) for p in (0, 8, 16, 24)], axis=1)
        positions = mx.concatenate([_make_positions(F, H, W), cond_pos], axis=1)
        latent = mx.random.normal((1, F + 4, D))
        modality = _make_modality(latent, positions)

        output = None
        for t in tiler.tiles:
            tiled, ctx = tiler.tile_modality(modality, t, normalize_positions=False)
            assert ctx.cond_blend_weights is not None
            assert bool(mx.all(mx.isfinite(ctx.cond_blend_weights)).item())
            output = tiler.blend(tiled.latent, t, ctx, output=output)
        mx.eval(output)
        assert mx.allclose(latent, output, atol=1e-6).item()

    def test_tile_without_cond_tokens_keeps_none(self):
        """A tile keeping zero conditioning tokens gets an empty weight vector, not a crash."""
        F, H, W, D = 4, 1, 1, 4
        tiling = TileCountConfig(frames=DimensionTilingConfig(num_tiles=2, overlap=0))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        positions = mx.concatenate([_make_positions(F, H, W), _compute_keyframe_positions(24, H, W, FPS)], axis=1)
        modality = _make_modality(mx.random.normal((1, F + 1, D)), positions)
        tiled, ctx = tiler.tile_modality(modality, tiler.tiles[0], normalize_positions=False)
        assert tiled.latent.shape[1] == _gen_count(tiler.tiles[0])
        out = tiler.blend(tiled.latent, tiler.tiles[0], ctx)
        mx.eval(out)
        assert out.shape == (1, F + 1, D)

    def test_seams_cut_frames_with_rectangular_masks(self):
        F = 31
        tiling = TileCountConfig(frames=DimensionTilingConfig(num_tiles=2, overlap=7))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, 1, 1), seams=[6, 12, 18, 24])
        frames = [t.in_coords[0] for t in tiler.tiles]
        assert frames == [slice(0, 19), slice(12, 31)]
        masks = [np.asarray(t.masks_1d[0]) for t in tiler.tiles]
        np.testing.assert_array_equal(masks[0], np.ones(19))
        np.testing.assert_array_equal(masks[1], np.concatenate([np.zeros(7), np.ones(12)]))

        weights = np.zeros(F)
        for t, m in zip(tiler.tiles, masks, strict=True):
            weights[t.in_coords[0]] += m
        np.testing.assert_array_equal(weights, np.ones(F))
        # The seam cell 18 is owned by the earlier tile only.
        assert masks[0][18] == 1.0 and masks[1][18 - 12] == 0.0

    @pytest.mark.parametrize(
        ("num_frame_tiles", "seams"),
        [(1, [6, 12, 18, 24]), (2, []), (2, [0, 30]), (2, [31, 40])],
    )
    def test_seams_ignored_without_interior_cut(self, num_frame_tiles: int, seams: list[int]):
        tiling = TileCountConfig(frames=DimensionTilingConfig(num_tiles=num_frame_tiles, overlap=7))
        with_seams = VideoModalityTiler(tiling, latent_shape=(31, 1, 1), seams=seams)
        plain = VideoModalityTiler(tiling, latent_shape=(31, 1, 1))
        assert [t.in_coords for t in with_seams.tiles] == [t.in_coords for t in plain.tiles]
        for a, b in zip(with_seams.tiles, plain.tiles, strict=True):
            np.testing.assert_array_equal(np.asarray(a.blend_mask), np.asarray(b.blend_mask))


class TestTilingPrimitives:
    def test_split_at_seams_matches_dfr_numbers(self):
        iv = split_at_seams([0, 6, 12, 18, 24, 30], 2, 7)(31)
        assert (iv.starts, iv.ends, iv.left_ramps, iv.right_ramps) == ([0, 12], [19, 31], [0, 7], [0, 0])

    def test_split_at_seams_validates(self):
        with pytest.raises(ValueError, match="num_tiles"):
            split_at_seams([0, 6], 0)
        with pytest.raises(ValueError, match=">= 0"):
            split_at_seams([0, 6], 1, -1)
        with pytest.raises(ValueError, match="start at 0"):
            split_at_seams([1, 6], 1)
        with pytest.raises(ValueError, match="strictly increasing"):
            split_at_seams([0, 6, 6], 1)
        with pytest.raises(ValueError, match="last cell"):
            split_at_seams([0, 6], 1)(9)

    def test_identity_mapping_rectangular(self):
        iv = DimensionIntervals(starts=[0, 5], ends=[8, 12], left_ramps=[0, 3], right_ramps=[0, 0])
        slices, masks = identity_mapping_operation(iv, rectangular=True)
        assert slices == [slice(0, 8), slice(5, 12)]
        np.testing.assert_array_equal(np.asarray(masks[0]), np.ones(8))
        np.testing.assert_array_equal(np.asarray(masks[1]), np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.float32))

    def test_split_by_count_rejects_tile_not_above_overlap(self):
        with pytest.raises(ValueError, match="<= overlap"):
            split_by_count(2, 3)(4)


class TestTiledLTXModel:
    def test_single_tile_matches_baseline(self):
        """Wrapping a model in TiledLTXModel with a 1x1x1 tiler must
        produce the same output as the bare model (single tile == identity)."""
        model, cfg = _tiny_model()
        # Pixel-space: F=1, H=1, W=16 → 16 tokens (single time/height row).
        F, H, W = 1, 1, 16
        common = _model_kwargs(cfg, F, H, W)

        baseline_v, baseline_a = model(**common)

        tiler = VideoModalityTiler(TileCountConfig(), latent_shape=(F, H, W))
        wrapped = TiledLTXModel(model, tiler)
        wrapped_v, wrapped_a = wrapped(**common)

        mx.eval(baseline_v, baseline_a, wrapped_v, wrapped_a)
        assert mx.allclose(baseline_v, wrapped_v, atol=1e-5, rtol=1e-5).item()
        assert mx.allclose(baseline_a, wrapped_a, atol=1e-5, rtol=1e-5).item()

    def test_unconditioned_output_byte_identical_to_pre_iso_loop(self):
        """Without appended tokens the wrapper is the pre-change loop, bit for bit."""
        model, cfg = _tiny_model()
        F, H, W = 4, 6, 6
        common = _model_kwargs(cfg, F, H, W)
        tiling = TileCountConfig(
            frames=DimensionTilingConfig(2, 1), height=DimensionTilingConfig(2, 2), width=DimensionTilingConfig(2, 1)
        )
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        got_v, got_a = TiledLTXModel(model, tiler)(**common)

        # Reference: the pre-change keep path (generated tokens only, trapezoid blend, audio mean).
        expected_v = None
        audio_outs = []
        for t in tiler.tiles:
            f, h, w = t.in_coords
            grid = np.arange(F * H * W).reshape(F, H, W)
            idx = mx.array(grid[f, h, w].reshape(-1))
            kw = dict(common)
            kw["video_latent"] = common["video_latent"][:, idx, :]
            kw["video_positions"] = common["video_positions"][:, idx, :]
            kw["video_attention_mask"] = None
            kw["video_keyframes_mask"] = None
            v, a = model(**kw)
            if expected_v is None:
                expected_v = mx.zeros((1, F * H * W, v.shape[-1]), dtype=v.dtype)
            part = v * t.blend_mask.reshape(-1).astype(v.dtype)[None, :, None]
            expected_v[:, idx, :] = expected_v[:, idx, :] + part
            audio_outs.append(a)
        expected_a = mx.mean(mx.stack(audio_outs, axis=0), axis=0)
        mx.eval(got_v, got_a, expected_v, expected_a)
        assert mx.array_equal(got_v, expected_v).item()
        assert mx.array_equal(got_a, expected_a).item()

    def _spy_positions(self, normalize: bool | None) -> tuple[VideoModalityTiler, mx.array, list[mx.array]]:
        F, H, W = 4, 4, 4
        tiling = TileCountConfig(frames=DimensionTilingConfig(2, 0), height=DimensionTilingConfig(2, 0))
        tiler = VideoModalityTiler(tiling, latent_shape=(F, H, W))
        seen: list[mx.array] = []

        def inner(**kwargs):
            seen.append(kwargs["video_positions"])
            return kwargs["video_latent"], kwargs["audio_latent"]

        wrapped = TiledLTXModel(inner, tiler) if normalize is None else TiledLTXModel(inner, tiler, normalize)
        positions = _make_positions(F, H, W)
        wrapped(
            video_latent=mx.zeros((1, F * H * W, 4)),
            audio_latent=mx.zeros((1, 2, 4)),
            timestep=mx.array([0.5]),
            video_positions=positions,
        )
        return tiler, positions, seen

    def test_normalize_positions_forwarded(self):
        tiler, _, seen = self._spy_positions(True)
        for t, pos in zip(tiler.tiles, seen, strict=True):
            gen = np.asarray(pos[0, : _gen_count(t), :])
            first_t = 0.5 if t.in_coords[0].start == 0 else 4.0
            # Interval starts (midpoint minus half a cell) have minimum 0 on every axis.
            np.testing.assert_allclose(gen.min(axis=0) - [first_t / FPS, 16.0, 16.0], 0.0, atol=1e-6)

    def test_normalize_positions_default_off(self):
        tiler, positions, seen = self._spy_positions(None)
        for t, pos in zip(tiler.tiles, seen, strict=True):
            f, h, w = t.in_coords
            full = np.asarray(positions[0]).reshape(4, 4, 4, 3)[f, h, w].reshape(-1, 3)
            np.testing.assert_array_equal(np.asarray(pos[0]), full)
