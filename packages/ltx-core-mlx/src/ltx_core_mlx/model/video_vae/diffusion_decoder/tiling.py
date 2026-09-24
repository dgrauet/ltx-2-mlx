"""Tile schedule for the diffusion video decoder (upstream ``diffusion_tiling.py`` + ``tiling.py``).

Tiles live on the stage-4 input grid (the output of det stages 1-3). Every function here is a
verbatim transcription of the upstream formulas quoted in the design spec; keep them free of MLX
model code so they stay testable without weights.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import mlx.core as mx

from ltx_core_mlx.model.video_vae.diffusion_decoder.config import DiffusionDecoderConfig, Kernel
from ltx_core_mlx.model.video_vae.tiling import compute_trapezoidal_mask_1d


@dataclass(frozen=True)
class Interval:
    """Half-open ``[start, end)`` span on one axis with fade-in / fade-out ramp lengths.

    Attributes:
        start: First index (inclusive).
        end: Last index (exclusive).
        left_ramp: Fade-in length at the start of the span.
        right_ramp: Fade-out length at the end of the span.
    """

    start: int
    end: int
    left_ramp: int = 0
    right_ramp: int = 0

    @property
    def length(self) -> int:
        """Length of the interval: ``end - start``."""
        return self.end - self.start


def _grow_last_tile_to_min(intervals: list[Interval], min_tile_size: int) -> list[Interval]:
    """Extend a too-short last interval backwards to ``min_tile_size`` (upstream ``tl:140-154``)."""
    if len(intervals) < 2 or intervals[-1].length >= min_tile_size:
        return intervals
    prev, last = intervals[-2], intervals[-1]
    new_start = last.end - min_tile_size
    new_overlap = prev.end - new_start
    return [*intervals[:-2], replace(prev, right_ramp=new_overlap), Interval(new_start, last.end, new_overlap, 0)]


def _validate_intervals(intervals: list[Interval], length: int, min_tile_size: int | None) -> None:
    """Coverage, minimum length, ramp bounds and consistent overlaps (upstream ``tl:157-171``)."""
    if intervals[0].start != 0 or intervals[-1].end != length:
        raise ValueError(f"intervals {intervals} do not cover [0, {length})")
    for iv in intervals:
        if iv.length <= 0 or iv.left_ramp > iv.length or iv.right_ramp > iv.length:
            raise ValueError(f"invalid interval {iv}")
        if min_tile_size is not None and len(intervals) > 1 and iv.length < min_tile_size:
            raise ValueError(f"interval {iv} shorter than min tile size {min_tile_size}")
    for prev, cur in zip(intervals, intervals[1:], strict=False):
        overlap = prev.end - cur.start
        if overlap != prev.right_ramp or overlap != cur.left_ramp:
            raise ValueError(f"inconsistent overlap between {prev} and {cur}")


def split_by_size(length: int, size: int, overlap: int, min_tile_size: int | None = None) -> list[Interval]:
    """Split ``[0, length)`` into overlapping intervals of ``size`` (upstream ``tl:174-223``).

    Args:
        length: Axis length in stage-4 cells.
        size: Tile size in cells (``> 0``).
        overlap: Overlap between consecutive tiles in cells (``0 <= overlap < size``).
        min_tile_size: When set, an axis shorter than it is not split and a short last
            interval is grown backwards to this length.

    Returns:
        Intervals covering the axis; a single ``[0, length)`` interval when no split applies.

    Raises:
        ValueError: Bad arguments, or a grown last tile that breaks overlap consistency.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if min_tile_size is not None and min_tile_size < 1:
        raise ValueError("min_tile_size must be >= 1")
    if min_tile_size is not None and length < min_tile_size:
        return [Interval(0, length)]
    if length <= size:
        return [Interval(0, length)]
    n = (length + size - 2 * overlap - 1) // (size - overlap)
    intervals: list[Interval] = []
    for i in range(n):
        start = i * (size - overlap)
        if i == 0:
            intervals.append(Interval(0, size, 0, overlap))
        elif i < n - 1:
            intervals.append(Interval(start, start + size, overlap, overlap))
        else:
            intervals.append(Interval(start, length, overlap, 0))
    if min_tile_size is not None:
        intervals = _grow_last_tile_to_min(intervals, min_tile_size)
    _validate_intervals(intervals, length, min_tile_size)
    return intervals


def propagate_temporal(iv: Interval, stride: int) -> Interval:
    """Map a temporal interval through one causal pixel-shuffle hop (upstream ``dt:872-900``).

    With ``stride == 2`` the duplicated leading frame is dropped: ``end -= 1`` always, and
    ``start -= 1`` for every interval that does not start at 0, so non-origin tiles line up
    with the origin tile's frame indexing.
    """
    start, end = iv.start * stride, iv.end * stride
    if stride == 2:
        end -= 1
        if iv.start != 0:
            start -= 1
    return Interval(start, end, iv.left_ramp * stride, iv.right_ramp * stride)


def propagate_spatial(iv: Interval, stride: int) -> Interval:
    """Scale a spatial interval and its ramps by ``stride`` (non-causal hop)."""
    return Interval(iv.start * stride, iv.end * stride, iv.left_ramp * stride, iv.right_ramp * stride)


def round_up(value: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is ``>= value``."""
    return -(-value // multiple) * multiple


def padded_latent_fhw(cfg: DiffusionDecoderConfig, fhw: tuple[int, int, int]) -> tuple[int, int, int]:
    """Latent ``(F, H, W)`` after :meth:`NADiffusionDecoder.pad_to_floor` (each axis at least the floor)."""
    f_min, h_min, w_min = cfg.min_latent_shape()
    return (max(fhw[0], f_min), max(fhw[1], h_min), max(fhw[2], w_min))


@dataclass(frozen=True)
class DiffusionTileGeometry:
    """Everything the tile schedule needs from the decoder config (upstream ``dt:711-821``).

    Attributes:
        strides: Upsample strides of the four hops.
        pixel_scale: Pixels per stage-4 cell ``(t, h, w)`` = ``(st3, sh3 * patch, sw3 * patch)``.
        latent_scale: Pixels per latent cell ``(8, 32, 32)`` in production.
        patch_size: Stage-5 spatial patch.
        min_tile_s4: Minimum tile per axis in stage-4 cells ``max(k4, ceil(k5 / up3))``.
        halo4: Stage-4 receptive halo ``depth4 * (k4 // 2)``.
        halo5: Stage-5 halo in stage-4 cells ``ceil(depth5 * (k5 // 2) / up3)``.
        overlap_frames: Recommended temporal overlap in pixel frames.
        overlap_px: Recommended spatial overlap in pixels (shared by H and W).
        min_tile_frames: Smallest AUTO tile in frames.
        min_tile_px: Smallest AUTO tile in pixels.
        step_frames: AUTO candidate grid step in frames.
        step_px: AUTO candidate grid step in pixels.
        ghost_frames_s4: Ghost frames appended to the stage-4 feature by the trailing pad.
        stage4_channels: Channels of the stage-4 feature (memory model).
        stage5_channels: Channels of the diffusion stage (memory model).
    """

    strides: tuple[Kernel, Kernel, Kernel, Kernel]
    pixel_scale: Kernel
    latent_scale: Kernel
    patch_size: int
    min_tile_s4: Kernel
    halo4: Kernel
    halo5: Kernel
    overlap_frames: int
    overlap_px: int
    min_tile_frames: int
    min_tile_px: int
    step_frames: int
    step_px: int
    ghost_frames_s4: int
    stage4_channels: int
    stage5_channels: int

    @classmethod
    def from_config(cls, cfg: DiffusionDecoderConfig) -> DiffusionTileGeometry:
        """Derive the geometry from a decoder config (production values in the module docstring)."""
        strides = tuple(s for s, _ in cfg.upsamples)
        k4, k5, up3, p = cfg.stage_kernels[3], cfg.stage5_kernel, strides[3], cfg.patch_size
        pixel_scale = (up3[0], up3[1] * p, up3[2] * p)
        full = cfg.cumulative_strides()[4]
        latent_scale = (full[0], full[1] * p, full[2] * p)
        min_tile = tuple(max(k4[a], math.ceil(k5[a] / up3[a])) for a in range(3))
        halo4 = tuple(cfg.stage_depths[3] * (k4[a] // 2) for a in range(3))
        halo5 = tuple(math.ceil(cfg.diff_depth * (k5[a] // 2) / up3[a]) for a in range(3))
        dom = tuple(max(halo4[a], halo5[a]) for a in range(3))
        overlap_frames = round_up(dom[0] * pixel_scale[0], latent_scale[0])
        overlap_px = round_up(max(dom[1], dom[2]) * pixel_scale[1], latent_scale[1])
        step_frames = math.lcm(pixel_scale[0], latent_scale[0])
        step_px = math.lcm(pixel_scale[1], latent_scale[1])
        min_tile_frames = round_up(
            max(
                2 * pixel_scale[0],
                2 * overlap_frames,
                round_up(min_tile[0] * pixel_scale[0], pixel_scale[0]),
                2 * latent_scale[0],
            ),
            step_frames,
        )
        min_tile_px = round_up(
            max(
                2 * pixel_scale[1],
                2 * overlap_px,
                round_up(min_tile[1] * pixel_scale[1], pixel_scale[1]),
                2 * latent_scale[1],
            ),
            step_px,
        )
        return cls(
            strides=strides,  # type: ignore[arg-type]
            pixel_scale=pixel_scale,
            latent_scale=latent_scale,
            patch_size=p,
            min_tile_s4=min_tile,  # type: ignore[arg-type]
            halo4=halo4,  # type: ignore[arg-type]
            halo5=halo5,  # type: ignore[arg-type]
            overlap_frames=overlap_frames,
            overlap_px=overlap_px,
            min_tile_frames=min_tile_frames,
            min_tile_px=min_tile_px,
            step_frames=step_frames,
            step_px=step_px,
            ghost_frames_s4=cfg.ghost_pad_frames() * cfg.cumulative_strides()[3][0],
            stage4_channels=cfg.stage_channels[3],
            stage5_channels=cfg.stage_channels[4],
        )

    def stage4_content_thw(self, f: int, h: int, w: int) -> Kernel:
        """Stage-4 input grid of a (padded, not ghost-padded) latent (upstream ``dt:711-725``)."""
        t = f
        for st, sh, sw in self.strides[:3]:
            t, h, w = t * st, h * sh, w * sw
            if st == 2:
                t -= 1
        return (t, h, w)

    def intervals_for_axis(self, axis: int, length: int, tile_px: int, overlap_px: int) -> list[Interval]:
        """Intervals (stage-4 cells) for one axis; ``tile_px == 0`` leaves the axis untiled (``tl:860-897``)."""
        if tile_px == 0:
            return [Interval(0, length)]
        factor = self.pixel_scale[axis]
        size, overlap = tile_px // factor, overlap_px // factor
        tile = max(2, overlap + 1, size)
        return split_by_size(length, tile, overlap, self.min_tile_s4[axis])


@dataclass(frozen=True)
class DiffusionTileConfig:
    """Tile sizes and overlaps in pixel units (frames / pixels); ``0`` size = axis untiled."""

    tile_frames: int
    overlap_frames: int
    tile_px_h: int
    overlap_px_h: int
    tile_px_w: int
    overlap_px_w: int

    def axis(self, axis: int) -> tuple[int, int]:
        """``(tile, overlap)`` of axis 0 (t), 1 (h) or 2 (w)."""
        return [
            (self.tile_frames, self.overlap_frames),
            (self.tile_px_h, self.overlap_px_h),
            (self.tile_px_w, self.overlap_px_w),
        ][axis]

    def validate(self, geometry: DiffusionTileGeometry, *, allow_small_overlap: bool = False) -> DiffusionTileConfig:
        """Check grid alignment, minimum size and (unless ``allow_small_overlap``) the recommended overlap."""
        names = ("frames", "height", "width")
        recommended = (geometry.overlap_frames, geometry.overlap_px, geometry.overlap_px)
        for a in range(3):
            tile, overlap = self.axis(a)
            factor = geometry.pixel_scale[a]
            if tile == 0:
                continue
            if tile % factor or overlap % factor:
                raise ValueError(
                    f"diffusion decoder tile {names[a]}: size {tile} and overlap {overlap} must be multiples of {factor}"
                )
            if tile < 2 * factor:
                raise ValueError(f"diffusion decoder tile {names[a]}: size {tile} must be at least {2 * factor}")
            if overlap >= tile:
                raise ValueError(
                    f"diffusion decoder tile {names[a]}: overlap {overlap} must be smaller than the tile size {tile}"
                )
            if not allow_small_overlap and overlap < recommended[a]:
                raise ValueError(
                    f"diffusion decoder tile {names[a]}: overlap {overlap} is below the recommended {recommended[a]}"
                )
        return self

    @classmethod
    def from_pixels(cls, geometry: DiffusionTileGeometry, frames: int, height: int, width: int) -> DiffusionTileConfig:
        """Build a config with the recommended overlaps from explicit tile sizes (``0`` = axis untiled)."""
        ov_t = geometry.overlap_frames if frames else 0
        ov_h = geometry.overlap_px if height else 0
        ov_w = geometry.overlap_px if width else 0
        return cls(frames, ov_t, height, ov_h, width, ov_w).validate(geometry)


@dataclass(frozen=True)
class DiffusionTile:
    """One tile: input slices on the stage-4 grid, output slices + 1-D masks on the pixel grid.

    Attributes:
        index: Position in the schedule (temporal group slowest, then h, then w); seeds the tile noise.
        in_t: Temporal slice of the stage-4 **content** frames (ghost frames are appended by the decoder
            when ``pad_trailing``).
        in_h: Stage-4 H slice.
        in_w: Stage-4 W slice.
        out_t: Pixel-frame slice of the padded-latent output.
        out_h: Pixel H slice.
        out_w: Pixel W slice.
        mask_t: Float32 trapezoid mask of length ``out_t.stop - out_t.start``.
        mask_h: Float32 mask over ``out_h``.
        mask_w: Float32 mask over ``out_w``.
        is_origin: The tile starts at stage-4 frame 0 (drops the duplicated leading frame).
        pad_trailing: The tile reaches the last content frame (gets the ghost frames, then the ghost crop).
    """

    index: int
    in_t: slice
    in_h: slice
    in_w: slice
    out_t: slice
    out_h: slice
    out_w: slice
    mask_t: mx.array
    mask_h: mx.array
    mask_w: mx.array
    is_origin: bool
    pad_trailing: bool


def output_fhw(geometry: DiffusionTileGeometry, latent_fhw_padded: Kernel) -> Kernel:
    """Pixel shape ``(F_px, H_px, W_px)`` decoded from a padded latent (``(F-1)*8+1, 32H, 32W``)."""
    f, h, w = latent_fhw_padded
    ft, fh, fw = geometry.latent_scale
    return ((f - 1) * ft + 1, h * fh, w * fw)


def _axis_specs(
    geometry: DiffusionTileGeometry, axis: int, length: int, tile_px: int, overlap_px: int
) -> list[tuple[slice, slice, mx.array]]:
    """``(in_slice, out_slice, mask)`` per interval of one axis (upstream ``dt:462-491``)."""
    st3 = geometry.strides[3][axis]
    specs = []
    for iv in geometry.intervals_for_axis(axis, length, tile_px, overlap_px):
        if axis == 0:
            px = propagate_temporal(iv, st3)
        else:
            px = propagate_spatial(propagate_spatial(iv, st3), geometry.patch_size)
        mask = compute_trapezoidal_mask_1d(px.length, px.left_ramp, px.right_ramp).astype(mx.float32)
        specs.append((slice(iv.start, iv.end), slice(px.start, px.end), mask))
    return specs


def build_tile_schedule(
    geometry: DiffusionTileGeometry,
    latent_fhw_padded: Kernel,
    config: DiffusionTileConfig | None,
    *,
    allow_small_overlap: bool = False,
) -> list[DiffusionTile]:
    """Cartesian product of the per-axis intervals, temporal axis slowest (upstream ``dt:421-505``).

    Args:
        geometry: Decoder tile geometry.
        latent_fhw_padded: Latent shape after :func:`padded_latent_fhw` (content, no ghost frames).
        config: Tile sizes; ``None`` = one tile covering everything.
        allow_small_overlap: Test-only escape from the recommended-overlap check.
    """
    cfg = config or DiffusionTileConfig(0, 0, 0, 0, 0, 0)
    cfg.validate(geometry, allow_small_overlap=allow_small_overlap)
    t4, h4, w4 = geometry.stage4_content_thw(*latent_fhw_padded)
    t_specs = _axis_specs(geometry, 0, t4, *cfg.axis(0))
    h_specs = _axis_specs(geometry, 1, h4, *cfg.axis(1))
    w_specs = _axis_specs(geometry, 2, w4, *cfg.axis(2))
    tiles: list[DiffusionTile] = []
    for i, ((it, ot, mt), (ih, oh, mh), (iw, ow, mw)) in enumerate(itertools.product(t_specs, h_specs, w_specs)):
        tiles.append(
            DiffusionTile(
                index=i,
                in_t=it,
                in_h=ih,
                in_w=iw,
                out_t=ot,
                out_h=oh,
                out_w=ow,
                mask_t=mt,
                mask_h=mh,
                mask_w=mw,
                is_origin=it.start == 0,
                pad_trailing=it.stop == t4,
            )
        )
    return tiles


def masks_are_complementary(tiles: list[DiffusionTile], out_fhw: Kernel, atol: float = 1e-5) -> bool:
    """True when, per axis, the masks of the unique output slices sum to one everywhere (``tl:501-535``)."""
    for axis, (out_attr, mask_attr) in enumerate((("out_t", "mask_t"), ("out_h", "mask_h"), ("out_w", "mask_w"))):
        total = mx.zeros((out_fhw[axis],), dtype=mx.float32)
        seen: set[tuple[int, int]] = set()
        for tile in tiles:
            s: slice = getattr(tile, out_attr)
            if (s.start, s.stop) in seen:
                continue
            seen.add((s.start, s.stop))
            total[s] = total[s] + getattr(tile, mask_attr)
        if not mx.allclose(total, mx.ones_like(total), atol=atol).item():
            return False
    return True


def group_tiles_by_temporal_slice(tiles: list[DiffusionTile]) -> list[list[DiffusionTile]]:
    """Consecutive tiles sharing ``out_t`` form one temporal group (schedule order is temporal-slowest)."""
    groups: list[list[DiffusionTile]] = []
    for tile in tiles:
        if groups and groups[-1][0].out_t == tile.out_t:
            groups[-1].append(tile)
        else:
            groups.append([tile])
    return groups


def describe_tiling(geometry: DiffusionTileGeometry, latent_fhw_padded: Kernel, config: DiffusionTileConfig) -> str:
    """One-line human summary: tile counts, sizes, overlaps and the compute redundancy factor."""
    t4, h4, w4 = geometry.stage4_content_thw(*latent_fhw_padded)
    n_t = len(geometry.intervals_for_axis(0, t4, *config.axis(0)))
    n_h = len(geometry.intervals_for_axis(1, h4, *config.axis(1)))
    n_w = len(geometry.intervals_for_axis(2, w4, *config.axis(2)))
    f_px, h_px, w_px = output_fhw(geometry, latent_fhw_padded)
    tile_t = config.tile_frames or f_px
    tile_h = config.tile_px_h or h_px
    tile_w = config.tile_px_w or w_px
    redundancy = n_t * n_h * n_w * tile_t * tile_h * tile_w / (f_px * h_px * w_px)
    return (
        f"tiles={n_t}x{n_h}x{n_w} ({n_t * n_h * n_w}) frames={config.tile_frames}/{config.overlap_frames} "
        f"px={config.tile_px_h}x{config.tile_px_w}/{config.overlap_px_h} redundancy=x{redundancy:.1f}"
    )


#: Stage-5 activation coefficient: bytes per token = channels * 2 * coef. Upstream's ``chunked_eager``
#: uses 5.0; this value is calibrated separately for MLX/Metal. Calibrated 2026-09-19 against a
#: measured decode (LTX-2.5 q8, 512x768x97, M2 Pro 32 GB): the prior value of 5.0 estimated 5.90 GB
#: against a measured peak Metal memory of 20.07 GB (ratio 0.29, and the same under-estimate made the
#: automatic budget picker leave a 512x768x49 decode untiled at an 8 GB budget even though it measured
#: 10.58 GB peak — a real budget-violation bug, not just an accuracy gap). 17.5 estimates 20.11 GB for
#: that shape (ratio 1.002 against the measurement).
STAGE5_MEM_COEF = 17.5
#: Headroom kept free on top of the model and activations.
RESERVE_BYTES = 1 << 30
#: Weights are charged at least this much (upstream ``dt:77``).
MIN_MODEL_BYTES = 1 << 30
#: Budget env var, shared with the conv decoder.
BUDGET_ENV = "LTX2_VAE_DECODE_BUDGET_GB"


def _stage4_feature_bytes(geometry: DiffusionTileGeometry, latent_fhw_padded: Kernel) -> int:
    """bf16 bytes of the resident stage-4 feature including the ghost frames."""
    t4, h4, w4 = geometry.stage4_content_thw(*latent_fhw_padded)
    return (t4 + geometry.ghost_frames_s4) * h4 * w4 * geometry.stage4_channels * 2


def _stage5_bytes_per_token(geometry: DiffusionTileGeometry) -> float:
    """Activation bytes per stage-5 token: channels x 2 x STAGE5_MEM_COEF."""
    return geometry.stage5_channels * 2 * STAGE5_MEM_COEF


def _plane_tokens(geometry: DiffusionTileGeometry, h_px: int, w_px: int, keyframe_planes: int) -> int:
    """Stage-5 tokens of ``keyframe_planes`` planes at an ``h_px x w_px`` pixel extent."""
    p = geometry.patch_size
    return keyframe_planes * (h_px // p) * (w_px // p)


def estimate_untiled_bytes(
    geometry: DiffusionTileGeometry, latent_fhw_padded: Kernel, *, keyframe_planes: int = 0
) -> int:
    """Activation bytes of a one-tile decode: (video + plane) stage-5 tokens x bytes/token + the fp16
    output accumulator."""
    f_px, h_px, w_px = output_fhw(geometry, latent_fhw_padded)
    p = geometry.patch_size
    tokens = f_px * (h_px // p) * (w_px // p) + _plane_tokens(geometry, h_px, w_px, keyframe_planes)
    return int(tokens * _stage5_bytes_per_token(geometry)) + f_px * h_px * w_px * 3 * 2


def _candidates(
    geometry: DiffusionTileGeometry, axis: int, length_s4: int, length_px: int, min_px: int, step: int, overlap: int
) -> list[tuple[int, int]]:
    """``(tile_px, n_intervals)`` for every grid size from the minimum up to the full axis."""
    top = max(min_px, round_up(length_px, step))
    out = []
    for size in range(min_px, top + 1, step):
        if size <= overlap:
            continue
        out.append((size, len(geometry.intervals_for_axis(axis, length_s4, size, overlap))))
    return out


def auto_tile_config(
    geometry: DiffusionTileGeometry,
    latent_fhw_padded: Kernel,
    *,
    budget_bytes: int,
    weight_bytes: int,
    keyframe_planes: int = 0,
) -> DiffusionTileConfig | None:
    """Pick tile sizes that keep the decode under ``budget_bytes`` (upstream ``dt:261-418``, simplified).

    Returns ``None`` when the whole decode fits (no tiling). Otherwise the feasible
    ``(frames, h, w)`` on the ``(step_frames, step_px, step_px)`` grid with the least overlap
    redundancy; ties prefer the larger tile, then fewer tiles.

    ``keyframe_planes`` planes are charged at the video's per-token cost (upstream
    ``chunked_eager`` uses the same coefficient); every tile is assumed to carry all
    planes, which is an upper bound (a tile keeps only the planes inside it plus the nearest
    one on each side).

    Raises:
        ValueError: No tile fits (names ``LTX2_VAE_DECODE_BUDGET_GB``).
    """
    usable = (
        budget_bytes
        - max(weight_bytes, MIN_MODEL_BYTES)
        - RESERVE_BYTES
        - _stage4_feature_bytes(geometry, latent_fhw_padded)
    )
    if estimate_untiled_bytes(geometry, latent_fhw_padded, keyframe_planes=keyframe_planes) <= usable:
        return None
    t4, h4, w4 = geometry.stage4_content_thw(*latent_fhw_padded)
    f_px, h_px, w_px = output_fhw(geometry, latent_fhw_padded)
    ov_t, ov_hw = geometry.overlap_frames, geometry.overlap_px
    cand_t = _candidates(geometry, 0, t4, f_px, geometry.min_tile_frames, geometry.step_frames, ov_t)
    cand_h = _candidates(geometry, 1, h4, h_px, geometry.min_tile_px, geometry.step_px, ov_hw)
    cand_w = _candidates(geometry, 2, w4, w_px, geometry.min_tile_px, geometry.step_px, ov_hw)
    p = geometry.patch_size
    per_token = _stage5_bytes_per_token(geometry)
    best: tuple[float, int, int, int, int, int] | None = None
    for tile_t, n_t in cand_t:
        acc = 2 * tile_t * h_px * w_px * 6
        if acc >= usable:
            continue
        max_tokens = (usable - acc) // per_token
        for tile_h, n_h in cand_h:
            for tile_w, n_w in cand_w:
                if (
                    tile_t * (tile_h // p) * (tile_w // p) + _plane_tokens(geometry, tile_h, tile_w, keyframe_planes)
                    > max_tokens
                ):
                    continue
                redundancy = n_t * n_h * n_w * tile_t * tile_h * tile_w / (f_px * h_px * w_px)
                score = (redundancy, -(tile_t * tile_h * tile_w), n_t * n_h * n_w, tile_t, tile_h, tile_w)
                if best is None or score < best:
                    best = score
    if best is None:
        raise ValueError(
            f"diffusion decoder: no tile fits the decode budget ({budget_bytes / 2**30:.1f} GB, "
            f"{max(usable, 0) / 2**30:.1f} GB usable after weights, reserve and the stage-4 feature). "
            f"Raise {BUDGET_ENV}, lower the resolution / frame count, or use --video-decoder conv."
        )
    _, _, _, tile_t, tile_h, tile_w = best
    return DiffusionTileConfig(tile_t, ov_t, tile_h, ov_hw, tile_w, ov_hw)
