"""Video VAE Decoder and Encoder.

Ported from ltx-core/src/ltx_core/model/video_vae/video_vae.py

Weight key structure (decoder, after stripping 'vae_decoder.' prefix):
    conv_in.conv.{weight,bias}
    conv_out.conv.{weight,bias}
    per_channel_statistics.{mean,std}
    up_blocks.{0,2,4,6,8}.res_blocks.{N}.conv{1,2}.conv.{weight,bias}  (ResStages)
    up_blocks.{1,3,5,7}.conv.conv.{weight,bias}                         (DepthToSpaceUpsamples)

Weight key structure (encoder, after stripping 'vae_encoder.' prefix):
    conv_in.conv.{weight,bias}          -- (128, 3,3,3, 48)
    conv_out.conv.{weight,bias}         -- (129, 3,3,3, 1024)
    per_channel_statistics.{_mean_of_means, _std_of_means}  -- (128,)
    down_blocks.{0,2,4,6,8}.res_blocks.{N}.conv{1,2}.conv.{weight,bias}
    down_blocks.{1,3,5,7}.conv.conv.{weight,bias}

Note: The encoder weight file uses ``_mean_of_means`` / ``_std_of_means`` but MLX
nn.Module skips underscore-prefixed attributes in ``parameters()``.
We store them as ``mean_of_means`` / ``std_of_means`` and remap during loading
via :func:`~ltx_2_mlx.model.video_vae.ops.remap_encoder_weight_keys`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from ltx_core_mlx.model.video_vae.convolution import Conv3dBlock
from ltx_core_mlx.model.video_vae.normalization import pixel_norm
from ltx_core_mlx.model.video_vae.ops import EncoderPerChannelStatistics, PerChannelStatistics
from ltx_core_mlx.model.video_vae.resnet import ResBlockStage
from ltx_core_mlx.model.video_vae.sampling import (
    DepthToSpaceUpsample,
    SpaceToDepthDownsample,
    patchify_spatial,
    pixel_shuffle_3d,
    unpatchify_spatial,
)
from ltx_core_mlx.model.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    Tile,
    TilingConfig,
    prepare_tiles_for_decoding,
    prepare_tiles_for_encoding,
)
from ltx_core_mlx.utils.ffmpeg import find_ffmpeg
from ltx_core_mlx.utils.memory import aggressive_cleanup

logger: logging.Logger = logging.getLogger(__name__)


def _write_all(buffer: memoryview, stream: Any) -> None:
    """Write a contiguous buffer completely without materializing ``bytes``.

    ``BufferedWriter.write`` normally accepts the exported MLX buffer directly,
    but the explicit loop also handles short writes without a fallback copy.
    """
    view = memoryview(buffer).cast("B")
    try:
        while view:
            written = stream.write(view)
            # A stream that reports neither progress nor an error would spin
            # this loop forever. That is a broken stream implementation, not a
            # closed pipe, so it does not raise BrokenPipeError.
            if written is None or written <= 0:
                raise OSError(
                    f"stream.write() reported {written!r} bytes written; "
                    f"cannot make progress on {len(view)} remaining bytes"
                )
            view = view[written:]
    finally:
        view.release()


class _OrderedFrameWriter:
    """Serialize evaluated frames with at most one asynchronous write in flight.

    Waiting for the previous write before submitting the next keeps memory bounded
    and preserves byte order. The worker only sees buffers after ``mx.eval`` has
    completed, so it never races Metal evaluation.
    """

    def __init__(self, stream: Any, *, overlap: bool) -> None:
        self._stream = stream
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltx-media-writer") if overlap else None
        self._pending: Future[None] | None = None
        self.completed = 0

    def _complete_pending(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.result()
            self.completed += 1

    def submit(self, buffer: Any) -> None:
        if self._executor is None:
            _write_all(memoryview(buffer), self._stream)
            self.completed += 1
            return

        self._complete_pending()
        self._pending = self._executor.submit(_write_all, memoryview(buffer), self._stream)

    def finish(self) -> None:
        self._complete_pending()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)


def _media_write_overlap_enabled() -> bool:
    """Whether frame writes overlap decode on a worker thread. Off by default.

    The zero-copy write path is unconditional and free. Overlapping the write
    on a second thread measured 0.8% end to end on a 512x512x25 q8 render,
    inside run-to-run noise, so it stays opt-in rather than adding concurrency
    to the decode loop for no measured gain.
    """
    value = os.environ.get("LTX2_MEDIA_WRITE_OVERLAP", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


#: Peak-memory budget of a VAE decode in GB; shared by the conv and diffusion decoders.
VAE_DECODE_BUDGET_ENV = "LTX2_VAE_DECODE_BUDGET_GB"


#: Set to a truthy value to leave MLX's allocator cache alone during a VAE decode.
VAE_DECODE_KEEP_CACHE_ENV = "LTX2_VAE_DECODE_KEEP_CACHE"

#: Peak activation bytes of the conv decoder per output pixel-frame, measured on an M2 Pro 32 GB
#: with the LTX-2.5 q8 pack (issue #142): 740-900 B untiled, ~530-750 B inside temporal /
#: spatial tiles. The peak sits in the last up-blocks at full pixel resolution, not at block 3.
CONV_DECODE_BYTES_PER_PIXEL_FRAME = 750
#: Bytes per pixel of the fp32 accumulation state of the tiled path: ``buffer`` + ``weights``
#: for the current temporal group and the previous group's pair kept for blending.
_TILED_ACCUMULATOR_BYTES_PER_PIXEL = 4 * 3 * 4
#: Upstream ``TileSizeConfig.default()``: 80 frames / 24 overlap, 768 px / 64 overlap.
DECODE_TILE_FRAMES_MAX = 80
DECODE_TILE_FRAMES_PREFERRED_MIN = 40
DECODE_TILE_FRAMES_MIN = 16
DECODE_SPATIAL_TILE_LADDER: tuple[tuple[int, int], ...] = ((768, 64), (512, 32), (256, 32))


def decode_budget_bytes() -> int:
    """Peak-memory budget of a VAE decode: ``LTX2_VAE_DECODE_BUDGET_GB``, else half of unified memory."""
    if VAE_DECODE_BUDGET_ENV in os.environ:
        return int(float(os.environ[VAE_DECODE_BUDGET_ENV]) * 1024**3)
    return int(mx.device_info()["memory_size"]) // 2


@contextlib.contextmanager
def decode_cache_limit() -> Iterator[None]:
    """Disable MLX's allocator cache for the duration of a decode, then restore it.

    The decoder's large, short-lived activations otherwise stay parked in the free
    list between tiles and inflate the process footprint by tens of GB (issue #142
    measured 44 -> 26 GB physical peak on the same decode, pixels identical).
    A no-op when the cache is already disabled (``--low-ram``) or when
    ``LTX2_VAE_DECODE_KEEP_CACHE`` is set.
    """
    if os.environ.get(VAE_DECODE_KEEP_CACHE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        yield
        return
    previous = mx.set_cache_limit(0)
    try:
        yield
    finally:
        mx.set_cache_limit(previous)


def _spatial_tile_px(latent_axis: int, long_side: int, cfg: SpatialTilingConfig) -> int:
    """Pixel extent of one spatial tile on an axis, mirroring ``prepare_tiles_for_decoding``."""
    tile_lat = cfg.tile_size_in_pixels // 32
    overlap_lat = cfg.tile_overlap_in_pixels // 32
    adjusted = max(max(2, overlap_lat + 1), round(tile_lat * latent_axis / long_side))
    return min(latent_axis, adjusted) * 32


def estimate_decode_peak_bytes(latent_shape: tuple[int, ...], tiling: TilingConfig | None) -> int:
    """Estimated peak activation bytes of decoding ``latent_shape`` with ``tiling``.

    ``CONV_DECODE_BYTES_PER_PIXEL_FRAME`` times the pixel-frames of one tile, plus the
    fp32 accumulation buffers of the tiled path (which span the whole frame).
    """
    _, _, f_lat, h_lat, w_lat = latent_shape
    f_px, h_px, w_px = 8 * f_lat - 7, 32 * h_lat, 32 * w_lat
    if tiling is None:
        return CONV_DECODE_BYTES_PER_PIXEL_FRAME * f_px * h_px * w_px
    tile_f, tile_h, tile_w = f_px, h_px, w_px
    if tiling.temporal_config is not None:
        tile_f = min(f_px, tiling.temporal_config.tile_size_in_frames)
    if tiling.spatial_config is not None:
        long_side = max(h_lat, w_lat)
        tile_h = _spatial_tile_px(h_lat, long_side, tiling.spatial_config)
        tile_w = _spatial_tile_px(w_lat, long_side, tiling.spatial_config)
    activations = CONV_DECODE_BYTES_PER_PIXEL_FRAME * tile_f * tile_h * tile_w
    accumulators = tile_f * h_px * w_px * _TILED_ACCUMULATOR_BYTES_PER_PIXEL
    return activations + accumulators


def describe_decode_tiling(tiling: TilingConfig) -> str:
    """Short human summary of a decode tiling, e.g. ``frames=40/8 px=512/32``."""
    parts = []
    if tiling.temporal_config is not None:
        tc = tiling.temporal_config
        parts.append(f"frames={tc.tile_size_in_frames}/{tc.tile_overlap_in_frames}")
    if tiling.spatial_config is not None:
        sc = tiling.spatial_config
        parts.append(f"px={sc.tile_size_in_pixels}/{sc.tile_overlap_in_pixels}")
    return " ".join(parts) or "untiled"


def _temporal_tile(frames: int, frame_rate: float) -> TemporalTilingConfig:
    """Temporal tile of ``frames`` with a blend of ~1 s capped at 30 % of the tile (80 -> 24 like upstream)."""
    one_second = max(8, (int(frame_rate) // 8) * 8)
    overlap = min(one_second, (int(frames * 0.3) // 8) * 8)
    return TemporalTilingConfig(tile_size_in_frames=frames, tile_overlap_in_frames=overlap)


def _compute_decode_tiling(
    latent_shape: tuple[int, ...],
    frame_rate: float = 24.0,
    budget_bytes: int | None = None,
) -> TilingConfig | None:
    """Return a TilingConfig whose estimated peak fits the decode budget, or None.

    ``None`` when the whole clip fits (no tiling, no overhead). Otherwise the
    first rung of a ladder that fits: temporal tiles from the upstream default
    (80 frames) down to 40, then spatial tiles 768 -> 512 -> 256 px at 40 frames,
    then temporal tiles down to 16 frames at 256 px. Nothing fitting returns the
    smallest rung with a warning. Budget: ``budget_bytes`` or :func:`decode_budget_bytes`.
    """
    budget = decode_budget_bytes() if budget_bytes is None else budget_bytes
    if estimate_decode_peak_bytes(latent_shape, None) <= budget:
        return None
    f_px = 8 * latent_shape[2] - 7

    def fits(cfg: TilingConfig) -> bool:
        return estimate_decode_peak_bytes(latent_shape, cfg) <= budget

    temporal_sizes = [n for n in range(DECODE_TILE_FRAMES_MAX, DECODE_TILE_FRAMES_MIN - 1, -8) if n < f_px]
    preferred = [n for n in temporal_sizes if n >= DECODE_TILE_FRAMES_PREFERRED_MIN]
    smallest = [n for n in temporal_sizes if n < DECODE_TILE_FRAMES_PREFERRED_MIN]
    candidates: list[TilingConfig] = [TilingConfig(temporal_config=_temporal_tile(n, frame_rate)) for n in preferred]
    base_temporal = _temporal_tile(preferred[-1], frame_rate) if preferred else None
    for px, overlap in DECODE_SPATIAL_TILE_LADDER:
        candidates.append(TilingConfig(spatial_config=SpatialTilingConfig(px, overlap), temporal_config=base_temporal))
    last_px, last_overlap = DECODE_SPATIAL_TILE_LADDER[-1]
    for n in smallest:
        candidates.append(
            TilingConfig(
                spatial_config=SpatialTilingConfig(last_px, last_overlap),
                temporal_config=_temporal_tile(n, frame_rate),
            )
        )
    for cfg in candidates:
        if fits(cfg):
            return cfg
    logger.warning(
        "vae-decode: no tiling fits the %.1f GB budget (smallest rung estimates %.1f GB); decoding anyway",
        budget / 2**30,
        estimate_decode_peak_bytes(latent_shape, candidates[-1]) / 2**30,
    )
    return candidates[-1]


def _add_at(buffer: mx.array, coords: tuple[slice, ...], values: mx.array) -> mx.array:
    """Add values into buffer at the given slice coordinates.

    MLX arrays are immutable, so we use slice assignment via __setitem__
    on a copy. In practice MLX handles this efficiently.
    """
    # MLX supports in-place-style slice assignment that returns a new array
    buffer[coords] = buffer[coords] + values
    return buffer


def _group_tiles_by_temporal_slice(tiles: list[Tile]) -> list[list[Tile]]:
    """Group tiles by their temporal output slice."""
    if not tiles:
        return []

    groups: list[list[Tile]] = []
    current_slice = tiles[0].out_coords[2]
    current_group: list[Tile] = []

    for tile in tiles:
        tile_slice = tile.out_coords[2]
        if tile_slice == current_slice:
            current_group.append(tile)
        else:
            groups.append(current_group)
            current_slice = tile_slice
            current_group = [tile]

    if current_group:
        groups.append(current_group)

    return groups


# Bytes of ffmpeg stderr surfaced in the error when it exits non-zero.
_FFMPEG_STDERR_TAIL = 4096


@contextlib.contextmanager
def _ffmpeg_sink(cmd: list[str]) -> Iterator[subprocess.Popen[bytes]]:
    """Run an ffmpeg command that consumes raw frames on stdin.

    stderr goes to an unnamed temporary file rather than a pipe: a pipe that
    nobody reads fills at the OS buffer limit (64 KB on macOS), after which
    ffmpeg blocks on stderr, stops draining stdin, and ``wait()`` never
    returns (#92). The file is read back only when ffmpeg exits non-zero,
    so a failing encode raises with ffmpeg's own diagnostics instead of
    silently producing a truncated or missing file.

    On exit, stdin is closed and the process reaped regardless of how the
    body ended. An exception raised by the body propagates unchanged --
    ffmpeg's resulting exit status is a consequence, not the cause.

    Args:
        cmd: Full ffmpeg argv, reading video from ``-`` / ``pipe:0``.

    Yields:
        The running process; write frames to ``proc.stdin``.

    Raises:
        RuntimeError: If the body completed but ffmpeg exited non-zero.
    """
    with tempfile.TemporaryFile() as stderr_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_file)
        assert proc.stdin is not None
        try:
            yield proc
        finally:
            if not proc.stdin.closed:
                # ffmpeg may already have exited, and closing flushes
                # buffered bytes into a dead pipe.
                with contextlib.suppress(BrokenPipeError):
                    proc.stdin.close()
            proc.wait()
        if proc.returncode != 0:
            stderr_file.seek(0, os.SEEK_END)
            stderr_file.seek(max(0, stderr_file.tell() - _FFMPEG_STDERR_TAIL))
            tail = stderr_file.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg exited with status {proc.returncode}:\n{tail}")


def build_ffmpeg_command(
    ffmpeg: str,
    out_w: int,
    out_h: int,
    frame_rate: float,
    audio_path: str | None,
    output_path: str,
) -> list[str]:
    """Build the ffmpeg argv that consumes raw RGB24 frames on stdin and muxes to ``output_path``.

    Args:
        ffmpeg: Path to the ffmpeg binary.
        out_w: Output frame width in pixels.
        out_h: Output frame height in pixels.
        frame_rate: Output frames per second.
        audio_path: Optional audio file to mux alongside the video stream.
        output_path: Destination video file path.

    Returns:
        Full ffmpeg argv, reading raw video frames from ``-`` / ``pipe:0``.
    """
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{out_w}x{out_h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(frame_rate),
        "-i",
        "-",
    ]
    if audio_path:
        # Do not use -shortest: the reference muxes both streams in full
        # (ltx_pipelines.utils.media_io writes all video chunks + the entire
        # audio waveform, no truncation). Reconstructed audio can be slightly
        # shorter than the video, and -shortest would truncate tail frames on
        # extend/retake.
        cmd.extend(["-i", audio_path, "-c:a", "aac"])
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path])
    return cmd


def stream_chunks_to_ffmpeg(chunks: Iterator[mx.array], proc: subprocess.Popen[bytes]) -> None:
    """Push uint8 RGB frames from ``chunks`` into ``proc.stdin``.

    Args:
        chunks: Iterator of video chunks ``(B, 3, T, H, W)`` in ``[-1, 1]``.
        proc: Running ffmpeg process (see :func:`_ffmpeg_sink`); frames are
            written to ``proc.stdin``.
    """
    assert proc.stdin is not None
    frame_writer = _OrderedFrameWriter(proc.stdin, overlap=_media_write_overlap_enabled())
    try:
        for chunk in chunks:  # (B, 3, T, H, W)
            num_frames = chunk.shape[2]
            for i in range(num_frames):
                frame = chunk[:, :, i, :, :]
                frame = mx.clip(frame, -1.0, 1.0)
                frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
                frame_hwc = mx.contiguous(frame[0].transpose(1, 2, 0))  # (H, W, 3)
                mx.eval(frame_hwc)  # required before a worker exports the unified-memory buffer
                frame_writer.submit(frame_hwc)
                del frame, frame_hwc
                if i % 8 == 0:
                    aggressive_cleanup()
            del chunk
            aggressive_cleanup()
        frame_writer.finish()
    except BrokenPipeError:
        logger.warning(
            "ffmpeg pipe closed after %d frames; output may be truncated",
            frame_writer.completed,
        )
    finally:
        # The writer must drain before _ffmpeg_sink closes stdin and reaps
        # ffmpeg: a write error other than a closed pipe (a stalled stream
        # raises OSError) propagates from here, and the sink's own finally
        # still tears the process down.
        frame_writer.shutdown()


class VideoDecoder(nn.Module):
    """Video VAE Decoder with streaming frame output.

    Decodes latent (B, C, F', H', W') to pixels, streaming frames
    to ffmpeg for memory efficiency.

    Architecture matches the weight file exactly:
        conv_in -> up_blocks (alternating ResStage / DepthToSpaceUpsample) -> conv_out

    up_blocks layout:
        0: ResStage  1024, 2 blocks
        1: DepthToSpaceUpsample 1024 -> 4096  (pixel-shuffle 2xspatial + 2xtemporal -> 512ch)
        2: ResStage  512,  2 blocks
        3: DepthToSpaceUpsample 512 -> 4096   (pixel-shuffle 2xspatial + 2xtemporal -> 512ch)
        4: ResStage  512,  4 blocks
        5: DepthToSpaceUpsample 512 -> 512    (pixel-shuffle 2xtemporal -> 256ch)
        6: ResStage  256,  6 blocks
        7: DepthToSpaceUpsample 256 -> 512    (pixel-shuffle 2xspatial -> 128ch)
        8: ResStage  128,  4 blocks

    Args:
        causal: If True, uses causal temporal padding (replicate first frame,
            remove first frame after temporal upsample). If False (LTX-2.3
            default), uses symmetric zero-padding and no frame removal.
    """

    def __init__(self, causal: bool = False, spatial_padding_mode: str = "zeros"):
        super().__init__()
        self._causal = causal

        # LTX-2.3 model was trained with zero padding (per embedded_config.json
        # "spatial_padding_mode": "zeros"). Previously hardcoded "reflect" which
        # caused cumulative temporal divergence in decoder forward (visible as
        # the keyframe hold-cut-decay regression at the latent boundary).
        sp_mode = spatial_padding_mode

        # Input convolution: 128 latent channels -> 1024
        self.conv_in = Conv3dBlock(
            128,
            1024,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=sp_mode,
        )

        # Flat list of up_blocks -- indices must match weight keys exactly.
        self.up_blocks: list[Any] = [
            ResBlockStage(1024, num_blocks=2, causal=causal, spatial_padding_mode=sp_mode),  # 0
            DepthToSpaceUpsample(1024, 4096, causal=causal, spatial_padding_mode=sp_mode),  # 1
            ResBlockStage(512, num_blocks=2, causal=causal, spatial_padding_mode=sp_mode),  # 2
            DepthToSpaceUpsample(512, 4096, causal=causal, spatial_padding_mode=sp_mode),  # 3
            ResBlockStage(512, num_blocks=4, causal=causal, spatial_padding_mode=sp_mode),  # 4
            DepthToSpaceUpsample(512, 512, causal=causal, spatial_padding_mode=sp_mode),  # 5
            ResBlockStage(256, num_blocks=6, causal=causal, spatial_padding_mode=sp_mode),  # 6
            DepthToSpaceUpsample(256, 512, causal=causal, spatial_padding_mode=sp_mode),  # 7
            ResBlockStage(128, num_blocks=4, causal=causal, spatial_padding_mode=sp_mode),  # 8
        ]

        # Output convolution: 128 -> 48 (3 RGB x 16 for spatial pixel shuffle)
        self.conv_out = Conv3dBlock(
            128,
            48,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=sp_mode,
        )

        # Per-channel normalization statistics
        self.per_channel_statistics = PerChannelStatistics(128)

        # Upsample config: (spatial_factor, temporal_factor) per DepthToSpaceUpsample
        # up_blocks indices 1, 3, 5, 7
        self._upsample_config: list[tuple[int, int]] = [
            (2, 2),  # block 1: 4096 / (2*2*2) = 512
            (2, 2),  # block 3: 4096 / (2*2*2) = 512
            (1, 2),  # block 5: 512 / (1*1*2) = 256
            (2, 1),  # block 7: 512 / (2*2*1) = 128
        ]

    def denormalize_latent(self, latent: mx.array) -> mx.array:
        """Reverse per-channel normalization: x * std + mean.

        Args:
            latent: (B, F, H, W, C) in MLX layout.

        Returns:
            Denormalized latent.
        """
        mean = self.per_channel_statistics.mean.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std.reshape(1, 1, 1, 1, -1)
        return latent * std + mean

    def decode(self, latent: mx.array, *, _materialize_stages: bool = False) -> mx.array:
        """Decode latent to pixel frames.

        Args:
            latent: (B, C, F, H, W) latent in PyTorch layout.
            _materialize_stages: If True, force-eval after each upsample stage so
                prior activations can be freed before the next (larger) stage begins.
                Only set by :meth:`tiled_decode`; the no-tiling path omits this to
                avoid breaking kernel fusion across upsample stages.

        Returns:
            Pixels (B, 3, F, H, W) in [-1, 1], same dtype as ``latent``.
        """
        # Cast input to weights dtype and remember caller dtype to restore on
        # return. Matches Lightricks/LTX-2 PR #179 commit b604d3f — defensive
        # guard against dtype mismatch between the caller and weights.
        output_dtype = latent.dtype
        flat_params = mlx.utils.tree_flatten(self.parameters())
        weights_dtype = flat_params[0][1].dtype if flat_params else output_dtype
        if latent.dtype != weights_dtype:
            latent = latent.astype(weights_dtype)

        # Convert BCFHW -> BFHWC for MLX convolutions
        x = latent.transpose(0, 2, 3, 4, 1)
        x = self.denormalize_latent(x)

        x = self.conv_in(x)

        upsample_idx = 0
        for i, block in enumerate(self.up_blocks):
            x = block(x)

            # Apply pixel shuffle after each DepthToSpaceUpsample (odd indices)
            if i % 2 == 1:
                sf, tf = self._upsample_config[upsample_idx]
                x = pixel_shuffle_3d(x, spatial_factor=sf, temporal_factor=tf)
                # Reference: ALWAYS remove first frame after temporal upsample
                # (unconditional on causal mode, gated on stride[0]==2 only)
                if tf > 1:
                    x = x[:, 1:, :, :, :]
                upsample_idx += 1
                if _materialize_stages:
                    # Free prior-stage activations before the next, larger stage.
                    mx.eval(x)

        # Pre-activation PixelNorm + SiLU before final conv
        x = self.conv_out(nn.silu(pixel_norm(x)))

        # Final spatial unpatchify: 48 -> 3 channels, 4x spatial expansion.
        # Uses unpatchify_spatial (not pixel_shuffle_3d) because the reference
        # unpatchify has channel order (c, p, r_W, q_H) — width factor before
        # height factor — which differs from DepthToSpaceUpsample's (c, p1, p2_H, p3_W).
        x = unpatchify_spatial(x, patch_size=4)

        # BFHWC -> BCFHW, restored to caller's dtype.
        return x.transpose(0, 4, 1, 2, 3).astype(output_dtype)

    def tiled_decode(
        self,
        latent: mx.array,
        tiling_config: TilingConfig | None = None,
    ) -> Iterator[mx.array]:
        """Decode a latent tensor into video frames using tiled processing.

        Splits the latent into tiles, decodes each independently, and yields
        video chunks by temporal slice. Overlapping regions are blended using
        trapezoidal masks.

        Args:
            latent: (B, C, F', H', W') latent in PyTorch layout.
            tiling_config: Tiling configuration. If None, decodes without tiling.

        Yields:
            Video chunks (B, 3, T, H, W) in [-1, 1], by temporal slices.
        """
        if tiling_config is None:
            pixels = self.decode(latent)
            mx.eval(pixels)  # materialize now — frees block-3 intermediate activations
            aggressive_cleanup()  # release GPU cache before caller begins streaming
            yield pixels
            del pixels
            return

        tiles = prepare_tiles_for_decoding(latent.shape, tiling_config)

        # Group tiles by temporal output slice
        temporal_groups = _group_tiles_by_temporal_slice(tiles)

        # Calculate full output spatial dims from latent shape
        _, _, F_lat, H_lat, W_lat = latent.shape
        out_H = H_lat * 32
        out_W = W_lat * 32

        # State for temporal overlap blending
        previous_chunk: mx.array | None = None
        previous_weights: mx.array | None = None
        previous_temporal_slice: slice | None = None

        f_px = 8 * F_lat - 7
        for temporal_group_tiles in temporal_groups:
            # Spatial-only tiling leaves the temporal slice as slice(None): normalise it to the
            # clip's frame range so the offsets below work for every tiling shape.
            curr_temporal_slice = slice(*temporal_group_tiles[0].out_coords[2].indices(f_px)[:2])
            temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start

            # Initialize accumulation buffers for this temporal group.
            # TODO: switch to bfloat16 (matching decoder output dtype) to halve
            # buffer memory — deferred to isolate decode RAM auditing to its own PR.
            buffer = mx.zeros((latent.shape[0], 3, temporal_len, out_H, out_W))
            weights = mx.zeros_like(buffer)

            for tile in temporal_group_tiles:
                # Decode tile and immediately materialize to free intermediate activations
                decoded_tile = self.decode(latent[tile.in_coords], _materialize_stages=True)
                mx.eval(decoded_tile)
                aggressive_cleanup()

                mask = tile.blend_mask

                tile_t_start, tile_t_stop = tile.out_coords[2].indices(f_px)[:2]
                temporal_offset = tile_t_start - curr_temporal_slice.start
                expected_temporal_len = tile_t_stop - tile_t_start
                decoded_temporal_len = decoded_tile.shape[2]
                actual_temporal_len = min(
                    expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset
                )

                chunk_coords = (
                    slice(None),  # batch
                    slice(None),  # channels
                    slice(temporal_offset, temporal_offset + actual_temporal_len),
                    tile.out_coords[3],  # height
                    tile.out_coords[4],  # width
                )

                decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
                mask_slice = mask[:, :, :actual_temporal_len, :, :] if mask.shape[2] > 1 else mask

                buffer = _add_at(buffer, chunk_coords, decoded_slice * mask_slice)
                weights = _add_at(weights, chunk_coords, mask_slice)
                # Force accumulation buffers to materialize so decoded_tile's graph can be freed
                mx.eval(buffer, weights)

                del decoded_tile, mask, decoded_slice, mask_slice
                aggressive_cleanup()

            # Blend with previous temporal chunk if overlap exists
            if previous_chunk is not None and previous_temporal_slice is not None:
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    prev_overlap_start = curr_temporal_slice.start - previous_temporal_slice.start

                    # Add current overlap into previous buffers
                    prev_overlap = previous_chunk[:, :, prev_overlap_start:, :, :]
                    prev_w_overlap = previous_weights[:, :, prev_overlap_start:, :, :]
                    curr_overlap = buffer[:, :, :overlap_len, :, :]
                    curr_w_overlap = weights[:, :, :overlap_len, :, :]

                    merged = prev_overlap + curr_overlap
                    merged_w = prev_w_overlap + curr_w_overlap

                    # Write merged back into both buffers
                    previous_chunk = mx.concatenate([previous_chunk[:, :, :prev_overlap_start, :, :], merged], axis=2)
                    previous_weights = mx.concatenate(
                        [previous_weights[:, :, :prev_overlap_start, :, :], merged_w], axis=2
                    )
                    buffer = mx.concatenate([merged, buffer[:, :, overlap_len:, :, :]], axis=2)
                    weights = mx.concatenate([merged_w, weights[:, :, overlap_len:, :, :]], axis=2)

                # Yield the non-overlapping part of the previous chunk
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                if yield_len > 0:
                    safe_weights = mx.maximum(previous_weights, 1e-8)
                    chunk = (previous_chunk / safe_weights)[:, :, :yield_len, :, :]
                    mx.eval(chunk)
                    yield chunk

            previous_chunk = buffer
            previous_weights = weights
            previous_temporal_slice = curr_temporal_slice

        # Yield remaining chunk
        if previous_chunk is not None and previous_weights is not None:
            safe_weights = mx.maximum(previous_weights, 1e-8)
            chunk = previous_chunk / safe_weights
            mx.eval(chunk)
            yield chunk

    def decode_and_stream(
        self,
        latent: mx.array,
        output_path: str,
        *,
        frame_rate: float,
        audio_path: str | None = None,
        seed: int = 0,
    ) -> None:
        """Decode latent and stream frames to ffmpeg.

        Automatically applies temporal, then spatial, tiling when the estimated
        peak of a full-volume decode exceeds the memory budget
        (``LTX2_VAE_DECODE_BUDGET_GB``, default half of unified memory); see
        :func:`estimate_decode_peak_bytes` and :func:`_compute_decode_tiling`.
        The peak is ~750 bytes per output pixel-frame (issue #142), e.g. ~14 GB
        for 512x768x49 and ~150 GB for 1080p x 97 frames untiled. MLX's
        allocator cache is disabled for the duration of the decode
        (:func:`decode_cache_limit`). Falls through to a single-pass decode with
        no overhead for clips that fit.

        Args:
            latent: (B, C, F, H, W) latent.
            output_path: Path to output video file.
            frame_rate: Output frames per second.
            audio_path: Optional audio file to mux.
            seed: Unused by the deterministic conv decoder; accepted so callers
                can pass it uniformly across video-decoder backends (the
                diffusion decoder uses it to seed its noise draw).
        """
        del seed
        ffmpeg = find_ffmpeg()
        tiling = _compute_decode_tiling(latent.shape, frame_rate=frame_rate)
        if tiling is not None:
            logger.info("vae-decode tiled: %s", describe_decode_tiling(tiling))

        # Estimate output dimensions from latent
        _, _, _F_lat, H_lat, W_lat = latent.shape
        out_H = H_lat * 32
        out_W = W_lat * 32

        cmd = build_ffmpeg_command(ffmpeg, out_W, out_H, frame_rate, audio_path, output_path)

        try:
            with decode_cache_limit(), _ffmpeg_sink(cmd) as proc:
                self._stream_frames(latent, tiling, proc)
        finally:
            aggressive_cleanup()

    def _stream_frames(self, latent: mx.array, tiling: TilingConfig | None, proc: subprocess.Popen[bytes]) -> None:
        """Decode ``latent`` tile by tile and push uint8 RGB frames into ``proc.stdin``."""
        stream_chunks_to_ffmpeg(self.tiled_decode(latent, tiling), proc)


class VideoEncoder(nn.Module):
    """Video VAE Encoder.

    Encodes pixel frames (B, 3, F, H, W) to latent (B, C, F', H', W').
    Temporal 8x, spatial 32x compression with 128 latent channels.

    Reference architecture:
        patchify(4x4 spatial) -> conv_in -> down_blocks -> norm+silu -> conv_out

    down_blocks layout (from config encoder_blocks):
        0: ResStage  128,  4 blocks
        1: SpaceToDepthDownsample 128->256, stride=(1,2,2) -- spatial 2x
        2: ResStage  256,  6 blocks
        3: SpaceToDepthDownsample 256->512, stride=(2,1,1) -- temporal 2x
        4: ResStage  512,  4 blocks
        5: SpaceToDepthDownsample 512->1024, stride=(2,2,2) -- all 2x
        6: ResStage  1024, 2 blocks
        7: SpaceToDepthDownsample 1024->1024, stride=(2,2,2) -- all 2x (mult=1)
        8: ResStage  1024, 2 blocks

    Weight loading: use :func:`~ltx_2_mlx.model.video_vae.ops.remap_encoder_weight_keys`
    before calling ``load_weights`` to handle the underscore-prefixed per-channel stats keys.
    """

    def __init__(self):
        super().__init__()

        # Input convolution: 48 channels (3 RGB x 4x4 spatial patchify) -> 128
        self.conv_in = Conv3dBlock(48, 128, kernel_size=3, padding=1, causal=True)

        # Flat list of down_blocks -- indices must match weight keys exactly.
        self.down_blocks: list = [
            ResBlockStage(128, num_blocks=4, causal=True),  # 0
            SpaceToDepthDownsample(128, 256, stride=(1, 2, 2)),  # 1
            ResBlockStage(256, num_blocks=6, causal=True),  # 2
            SpaceToDepthDownsample(256, 512, stride=(2, 1, 1)),  # 3
            ResBlockStage(512, num_blocks=4, causal=True),  # 4
            SpaceToDepthDownsample(512, 1024, stride=(2, 2, 2)),  # 5
            ResBlockStage(1024, num_blocks=2, causal=True),  # 6
            SpaceToDepthDownsample(1024, 1024, stride=(2, 2, 2)),  # 7
            ResBlockStage(1024, num_blocks=2, causal=True),  # 8
        ]

        # Output convolution: 1024 -> 129 channels
        self.conv_out = Conv3dBlock(1024, 129, kernel_size=3, padding=1, causal=True)

        # Per-channel normalization statistics
        self.per_channel_statistics = EncoderPerChannelStatistics(128)

    def normalize_latent(self, latent: mx.array) -> mx.array:
        """Apply per-channel normalization: (x - mean) / std.

        Args:
            latent: (B, F, H, W, C) in MLX layout.

        Returns:
            Normalized latent.
        """
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, 1, 1, -1)
        return (latent - mean) / std

    def denormalize_latent(self, latent: mx.array) -> mx.array:
        """Reverse per-channel normalization: x * std + mean.

        Used to unwrap encoder normalization before the upsampler (which
        operates in un-normalized space) and re-normalize after.

        Args:
            latent: (B, F, H, W, C) in MLX layout.

        Returns:
            Denormalized latent.
        """
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, 1, 1, -1)
        return latent * std + mean

    def encode(self, pixels: mx.array) -> mx.array:
        """Encode pixel frames to latent.

        Args:
            pixels: (B, 3, F, H, W) in [-1, 1], PyTorch layout.

        Returns:
            Latent (B, C, F', H', W') in PyTorch layout.
        """
        # BCFHW -> BFHWC for MLX convolutions
        x = pixels.transpose(0, 2, 3, 4, 1)

        # Spatial patchification: (B, F, H, W, 3) -> (B, F, H/4, W/4, 48)
        # Reference: patchify(sample, patch_size_hw=4, patch_size_t=1)
        x = patchify_spatial(x, patch_size=4)

        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)

        # PixelNorm + SiLU before conv_out (reference: conv_norm_out + conv_act)
        x = self.conv_out(nn.silu(pixel_norm(x)))

        # Take first 128 channels (mean), discard the rest (log_var or dummy)
        x = x[:, :, :, :, :128]

        x = self.normalize_latent(x)

        # BFHWC -> BCFHW
        return x.transpose(0, 4, 1, 2, 3)

    def tiled_encode(
        self,
        video: mx.array,
        tiling_config: TilingConfig | None = None,
    ) -> mx.array:
        """Encode video to latent using tiled processing.

        Splits the video into overlapping tiles, encodes each independently,
        and blends overlapping regions using rectangular masks.

        Args:
            video: (B, 3, F, H, W) in [-1, 1], PyTorch layout.
            tiling_config: Tiling configuration. If None, encodes without tiling.

        Returns:
            Latent (B, 128, F', H', W') in PyTorch layout.
        """
        if tiling_config is None:
            return self.encode(video)

        batch, _, frames, height, width = video.shape

        # Crop frames to valid count (1 + 8*k)
        if (frames - 1) % 8 != 0:
            frames_to_crop = (frames - 1) % 8
            logger.warning(
                "Invalid frame count %d for encode; cropping last %d frames.",
                frames,
                frames_to_crop,
            )
            video = video[:, :, :-frames_to_crop, :, :]
            frames = video.shape[2]

        # Calculate output latent shape
        latent_F = (frames - 1) // 8 + 1
        latent_H = height // 32
        latent_W = width // 32

        tiles = prepare_tiles_for_encoding(video.shape, tiling_config)

        # Accumulation buffers
        latent_buffer = mx.zeros((batch, 128, latent_F, latent_H, latent_W))
        weights_buffer = mx.zeros_like(latent_buffer)

        for tile in tiles:
            video_tile = video[tile.in_coords]
            latent_tile = self.encode(video_tile)
            mask = tile.blend_mask

            latent_buffer = _add_at(latent_buffer, tile.out_coords, latent_tile * mask)
            weights_buffer = _add_at(weights_buffer, tile.out_coords, mask)

            del latent_tile, mask, video_tile
            aggressive_cleanup()

        safe_weights = mx.maximum(weights_buffer, 1e-8)
        return latent_buffer / safe_weights
