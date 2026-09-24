"""``NADiffusionDecoder``: the LTX-2.5 diffusion video decoder, plain path, tiled, keyframe-aware.

Flow (spec §3): size floor + ghost pad -> de-normalise -> conv_in -> det stages 1-3 with
upsamples -> stage 4 blocks (upsample 3 deferred to the diffusion blocks) -> ghost crop ->
pure-noise ``x_t`` -> one diffusion step at ``t = 1`` (``x0`` output is the pixels) -> crop.
:meth:`NADiffusionDecoder.tiled_decode` streams this per stage-4/stage-5 tile, blending
overlaps with trapezoid masks and yielding each temporal group's exclusive frames as soon
as its tiles are decoded, so peak activation memory stays bounded regardless of clip length.

The keyframe (dual-stream) path (``decode``/``tiled_decode`` with ``keyframes=``) carries a
second stream of keyframe planes alongside the video volume through the same weights, at every
stage; the two streams only mix inside the joint attention softmax
(:func:`~ltx_core_mlx.model.video_vae.diffusion_decoder.neighborhood_attention.joint_na3d`).
The plain path above is unaffected and stays byte-identical.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.diffusion_decoder.blocks import NABlock
from ltx_core_mlx.model.video_vae.diffusion_decoder.chunked import ChunkedDiffusionNABlock
from ltx_core_mlx.model.video_vae.diffusion_decoder.config import LTX_2_5_DIFFUSION_DECODER, DiffusionDecoderConfig
from ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes import (
    DecodeKeyframes,
    KeyframeStream,
    keyframe_clip_times,
    planes_for_tile,
    remaining_time_strides,
    upsample_keyframe_planes,
)
from ltx_core_mlx.model.video_vae.diffusion_decoder.layers import (
    LinearPixelShuffleUpsample,
    RMSNorm,
    SharedAdaLN,
    TimestepEmbedder,
)
from ltx_core_mlx.model.video_vae.diffusion_decoder.patching import patchify_pixels, unpatchify_pixels
from ltx_core_mlx.model.video_vae.diffusion_decoder.tiling import (
    DiffusionTile,
    DiffusionTileConfig,
    DiffusionTileGeometry,
    build_tile_schedule,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
    output_fhw,
)
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.weights import load_split_safetensors

#: Decorrelates the decoder's noise draw from the sampler's (which reuses ``seed``).
DIFFVAE_NOISE_SEED_OFFSET = 30000


class PerChannelStats(nn.Module):
    """Per-channel de-normalisation statistics (``mean``, ``std``) for the input latent."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mean = mx.zeros((channels,))
        self.std = mx.ones((channels,))


class NADiffusionDecoder(nn.Module):
    """See module docstring. Parameter names mirror the pack keys (prefix ``vae_decoder_av.``)."""

    def __init__(self, config: DiffusionDecoderConfig = LTX_2_5_DIFFUSION_DECODER) -> None:
        super().__init__()
        self.config = config
        c = config
        self.per_channel_statistics = PerChannelStats(c.in_channels)
        self.conv_in = nn.Linear(c.in_channels, c.stage_channels[0])
        # Added to keyframe latents in the keyframe-aware decode (DecodeKeyframes) right
        # before conv_in; the plain decode never reads it.
        self.type_emb = mx.zeros((c.in_channels,))
        self.det_stages = [
            [NABlock(c.stage_channels[s], c.head_dim, c.stage_kernels[s]) for _ in range(c.stage_depths[s])]
            for s in range(c.num_det_stages)
        ]
        self.upsamples = [
            LinearPixelShuffleUpsample(c.stage_channels[s], stride, reduction)
            for s, (stride, reduction) in enumerate(c.upsamples)
        ]
        pp = c.patch_size * c.patch_size
        self.conv_in_x_t = nn.Linear(c.out_channels * pp, c.diff_channels)
        self.t_embedder = TimestepEmbedder(c.t_freq_dim, c.t_embed_hidden)
        self.shared_adaln = SharedAdaLN(c.t_embed_hidden, c.diff_channels)
        self.diff_blocks = [
            ChunkedDiffusionNABlock(c.diff_channels, c.head_dim, c.stage5_kernel, context_channels=c.diff_channels)
            for _ in range(c.diff_depth)
        ]
        self.norm_out = RMSNorm(c.diff_channels)
        self.conv_out = nn.Linear(c.diff_channels, c.out_channels * pp)

        st, sh, sw = c.cumulative_strides()[4]
        self.temporal_scale = st
        self.spatial_scale = (sh * c.patch_size, sw * c.patch_size)

    def weights_dtype(self) -> mx.Dtype:
        """dtype the decoder runs in (its parameters'); inputs are cast to it on entry."""
        return self.conv_in.weight.dtype

    # ---- geometry -----------------------------------------------------------------
    def denormalize_latent(self, z: mx.array) -> mx.array:
        s, m = self.per_channel_statistics.std, self.per_channel_statistics.mean
        return z * s.reshape(1, -1, 1, 1, 1).astype(z.dtype) + m.reshape(1, -1, 1, 1, 1).astype(z.dtype)

    def pad_to_floor(
        self, latent: mx.array, *, temporal: bool = True
    ) -> tuple[mx.array, tuple[int, int, int, int, int]]:
        """Pad ``(B, C, F, H, W)`` up to the config's minimum latent shape (T: repeat last; H/W: symmetric edge).

        ``temporal=False`` skips the T pad (used to pad keyframe planes, which carry their own
        pixel-frame indices and must never gain a repeated-frame temporal pad).
        """
        f_min, h_min, w_min = self.config.min_latent_shape()
        _, _, f, h, w = latent.shape
        t_pad = max(f_min - f, 0) if temporal else 0
        h_need, w_need = max(h_min - h, 0), max(w_min - w, 0)
        h_b, h_a = h_need // 2, h_need - h_need // 2
        w_b, w_a = w_need // 2, w_need - w_need // 2
        if t_pad:
            latent = mx.concatenate([latent, mx.repeat(latent[:, :, -1:], t_pad, axis=2)], axis=2)
        if h_need:
            latent = mx.concatenate(
                [mx.repeat(latent[:, :, :, :1], h_b, axis=3), latent, mx.repeat(latent[:, :, :, -1:], h_a, axis=3)],
                axis=3,
            )
        if w_need:
            latent = mx.concatenate(
                [
                    mx.repeat(latent[:, :, :, :, :1], w_b, axis=4),
                    latent,
                    mx.repeat(latent[:, :, :, :, -1:], w_a, axis=4),
                ],
                axis=4,
            )
        return latent, (t_pad, h_b, h_a, w_b, w_a)

    def _ghost_crop_keep(self, t4: int) -> int:
        strides = self.config.cumulative_strides()[3]  # stride at stage-4 input relative to the latent
        ghost_at_s4 = self.config.ghost_pad_frames() * strides[0]
        return min(t4, max(t4 - ghost_at_s4, math.ceil(self.config.stage5_kernel[0] / 2)))

    def stage5_canvas(self, t4_kept: int, h4: int, w4: int, *, is_origin: bool = True) -> tuple[int, int, int]:
        """Noise canvas ``(F5, H5, W5)`` of a stage-4 feature; the origin tile drops the duplicated frame."""
        (st, sh, sw), _ = self.config.upsamples[3]
        p = self.config.patch_size
        frames = t4_kept * st - 1 if (is_origin and st == 2) else t4_kept * st
        return frames, h4 * sh * p, w4 * sw * p

    # ---- forward pieces -----------------------------------------------------------
    def forward_stages_1_to_3(
        self, latent_padded: mx.array, *, tap: Callable[[str, mx.array], None] | None = None
    ) -> mx.array:
        """De-normalise, ghost-pad, run det stages 1-3 with their upsamples; keep the ghost frames.

        Returns the stage-4 input feature ``(B, T4 + ghost, H4, W4, C4)``; it stays resident for the
        whole (tiled) decode. ``tap`` receives ``("s{s+1}.out", x)`` after each stage's last block.
        """
        z = self.denormalize_latent(latent_padded)
        ghost = self.config.ghost_pad_frames()
        z = mx.concatenate([z, mx.repeat(z[:, :, -1:], ghost, axis=2)], axis=2)
        x = self.conv_in(z.transpose(0, 2, 3, 4, 1))  # (B, T, H, W, C0)
        for s in range(3):
            for block in self.det_stages[s]:
                x = block(x)
            if tap is not None:
                tap(f"s{s + 1}.out", x)
            x = self.upsamples[s](x, drop_leading_frame=True)
        return x

    def forward_stage_4(
        self, feat: mx.array, *, pad_trailing: bool, tap: Callable[[str, mx.array], None] | None = None
    ) -> mx.array:
        """Stage-4 blocks on a (tile of the) stage-4 input feature; ghost-crop when ``pad_trailing``.

        ``upsamples[3]`` is deferred to the diffusion blocks (chunked_eager). ``tap`` receives
        ``("s4.out", x)``.
        """
        x = feat
        for block in self.det_stages[3]:
            x = block(x)
        if tap is not None:
            tap("s4.out", x)
        if pad_trailing:
            x = x[:, : self._ghost_crop_keep(x.shape[1])]
        return x

    def forward_stages_1_to_4(
        self, latent_padded: mx.array, *, tap: Callable[[str, mx.array], None] | None = None
    ) -> mx.array:
        """Whole-volume stages 1-4 + ghost crop (the one-tile path; parity goldens tap here)."""
        return self.forward_stage_4(self.forward_stages_1_to_3(latent_padded, tap=tap), pad_trailing=True, tap=tap)

    def forward_stage_5(
        self,
        x_t: mx.array,
        stage4_feat: mx.array,
        t: mx.array,
        *,
        drop_leading_frame: bool = True,
        tap: Callable[[str, mx.array], None] | None = None,
    ) -> mx.array:
        """One diffusion evaluation at timestep ``t`` (``(B,)``); returns pixels ``(B, 3, F5, H_px, W_px)``.

        ``drop_leading_frame`` is False for non-origin tiles (they keep all ``2 * T4`` shuffled
        frames). The shared context upsample runs once here and is injected by every block.
        ``tap``, when given, receives ``("s5.b{i}.out", x)`` after each diffusion block.
        """
        context = self.upsamples[3](stage4_feat, drop_leading_frame=drop_leading_frame)
        x = self.conv_in_x_t(patchify_pixels(x_t, self.config.patch_size))
        t_emb = self.t_embedder(t * self.config.timestep_scale_multiplier)
        modulation = self.shared_adaln(t_emb)
        for i, block in enumerate(self.diff_blocks):
            x = block(x, context, modulation)
            if tap is not None:
                tap(f"s5.b{i}.out", x)
        x = self.conv_out(self.norm_out(x))
        return unpatchify_pixels(x, self.config.patch_size, self.config.out_channels)

    # ---- keyframe (dual-stream) path ----------------------------------------------
    def keyframe_stream_from_latents(
        self, keyframes: DecodeKeyframes, *, valid: mx.array | None = None
    ) -> KeyframeStream:
        """Un-normalise the planes, add ``type_emb``, project with the shared ``conv_in``; times at the stage-1 stride."""
        z = self.denormalize_latent(keyframes.latents)
        x = z.transpose(0, 2, 3, 4, 1) + self.type_emb.astype(z.dtype)
        x = self.conv_in(x)
        if valid is None:
            valid = mx.ones((keyframes.num_planes,), dtype=mx.bool_)
        strides = remaining_time_strides(self.config)
        times = keyframe_clip_times(keyframes.pixel_frame_indices, strides[0], keyframes.clip_start_frame)
        return KeyframeStream(x, times, valid).masked()

    def forward_stages_1_to_3_with_keyframes(
        self,
        latent_padded: mx.array,
        keyframes_padded: DecodeKeyframes,
        *,
        tap: Callable[[str, mx.array], None] | None = None,
    ) -> tuple[mx.array, KeyframeStream]:
        """Dual-stream stages 1-3: video as :meth:`forward_stages_1_to_3`, planes alongside (times rebuilt per stage)."""
        z = self.denormalize_latent(latent_padded)
        ghost = self.config.ghost_pad_frames()
        z = mx.concatenate([z, mx.repeat(z[:, :, -1:], ghost, axis=2)], axis=2)
        x = self.conv_in(z.transpose(0, 2, 3, 4, 1))
        stream = self.keyframe_stream_from_latents(keyframes_padded)
        strides = remaining_time_strides(self.config)
        idx, cs = keyframes_padded.pixel_frame_indices, keyframes_padded.clip_start_frame
        for s in range(3):
            for block in self.det_stages[s]:
                x, stream = block.forward_with_keyframes(x, stream)
            if tap is not None:
                tap(f"s{s + 1}.out", x)
                tap(f"s{s + 1}.kf", stream.x)
            x = self.upsamples[s](x, drop_leading_frame=True)
            kx = upsample_keyframe_planes(self.upsamples[s], stream.x)
            stream = KeyframeStream(kx, keyframe_clip_times(idx, strides[s + 1], cs), stream.valid).masked()
        return x, stream

    def forward_stage_4_with_keyframes(
        self,
        feat: mx.array,
        stream: KeyframeStream,
        pixel_frame_indices: tuple[int, ...],
        *,
        clip_start_frame: int = 0,
        pad_trailing: bool,
        stage4_time_origin: float = 0.0,
        pixel_time_origin: float = 0.0,
        tap: Callable[[str, mx.array], None] | None = None,
    ) -> tuple[mx.array, KeyframeStream]:
        """Dual-stream stage 4 on a (tile of the) stage-4 feature; ghost-crop video only; returns stage-5 plane times.

        Two origins at two scales: ``stage4_time_origin`` (stage-4 cells, the tile's ``in_t.start``) for the
        stage-4 blocks, ``pixel_time_origin`` (pixel frames, the tile's ``out_t.start``) for stage 5.
        """
        strides = remaining_time_strides(self.config)
        times4 = keyframe_clip_times(pixel_frame_indices, strides[3], clip_start_frame, extra_origin=stage4_time_origin)
        stream = dataclasses.replace(stream, times=times4)
        x = feat
        for block in self.det_stages[3]:
            x, stream = block.forward_with_keyframes(x, stream)
        if tap is not None:
            tap("s4.out", x)
            tap("s4.kf", stream.x)
        if pad_trailing:
            x = x[:, : self._ghost_crop_keep(x.shape[1])]
        times5 = keyframe_clip_times(pixel_frame_indices, strides[4], clip_start_frame, extra_origin=pixel_time_origin)
        return x, dataclasses.replace(stream, times=times5)

    def forward_stage_5_with_keyframes(
        self,
        x_t: mx.array,
        kf_x_t: mx.array,
        stage4_feat: mx.array,
        kf_feat: mx.array,
        keyframe_times: mx.array,
        keyframe_valid: mx.array,
        t: mx.array,
        *,
        drop_leading_frame: bool = True,
        tap: Callable[[str, mx.array], None] | None = None,
    ) -> mx.array:
        """One dual-stream diffusion evaluation at ``t``; returns the video pixels only (the plane prediction is discarded)."""
        context = self.upsamples[3](stage4_feat, drop_leading_frame=drop_leading_frame)
        kf_context = upsample_keyframe_planes(self.upsamples[3], kf_feat)
        p = self.config.patch_size
        x = self.conv_in_x_t(patchify_pixels(x_t, p))
        kx = self.conv_in_x_t(patchify_pixels(kf_x_t, p)) * keyframe_valid[None, :, None, None, None].astype(x.dtype)
        t_emb = self.t_embedder(t * self.config.timestep_scale_multiplier)
        modulation = self.shared_adaln(t_emb)
        for i, block in enumerate(self.diff_blocks):
            x, kx = block.forward_with_keyframes(x, context, kx, kf_context, modulation, keyframe_times, keyframe_valid)
            if tap is not None:
                tap(f"s5.b{i}.out", x)
                tap(f"s5.b{i}.kf", kx)
        x = self.conv_out(self.norm_out(x))
        return unpatchify_pixels(x, p, self.config.out_channels)

    def _keyframe_noise_key(self, seed: int, tile_index: int = 0) -> mx.array:
        """Plane-noise key: the second half of the tile's split key (the video draw keeps the unsplit key)."""
        return mx.random.split(self.noise_key(seed, tile_index))[1]

    def decode_tile_with_keyframes(
        self,
        feat_s4: mx.array,
        stream: KeyframeStream,
        keyframes: DecodeKeyframes,
        tile: DiffusionTile,
        *,
        seed: int = 0,
        noise: mx.array | None = None,
        keyframe_noise: mx.array | None = None,
    ) -> mx.array:
        """Stages 4 + 5 on one tile with the planes it needs (upstream ``_decode_one_tile_with_keyframes``)."""
        cs = keyframes.clip_start_frame
        keep = planes_for_tile(
            keyframes.pixel_frame_indices, tile.out_t.start, tile.out_t.stop - 1, clip_start_frame=cs
        )
        if not any(keep):
            raise RuntimeError(f"diffusion decoder tile {tile.index} selected no keyframe plane")
        tile_idx = tuple(i for i, k in zip(keyframes.pixel_frame_indices, keep, strict=True) if k)
        tile_stream = stream.select_planes(keep).crop_spatial(tile.in_h, tile.in_w)
        t1 = feat_s4.shape[1] if tile.pad_trailing else tile.in_t.stop
        feat, tile_stream = self.forward_stage_4_with_keyframes(
            feat_s4[:, tile.in_t.start : t1, tile.in_h, tile.in_w],
            tile_stream,
            tile_idx,
            clip_start_frame=cs,
            pad_trailing=tile.pad_trailing,
            stage4_time_origin=float(tile.in_t.start),
            pixel_time_origin=float(tile.out_t.start),
        )
        canvas = self.stage5_canvas(feat.shape[1], feat.shape[2], feat.shape[3], is_origin=tile.is_origin)
        shape = (feat.shape[0], self.config.out_channels, *canvas)
        kshape = (feat.shape[0], self.config.out_channels, tile_stream.num_planes, canvas[1], canvas[2])
        if noise is None:
            noise = mx.random.normal(shape, key=self.noise_key(seed, tile.index))
        elif tuple(noise.shape) != shape:
            raise ValueError(f"noise must have shape {shape}, got {noise.shape}")
        if keyframe_noise is None:
            keyframe_noise = mx.random.normal(kshape, key=self._keyframe_noise_key(seed, tile.index))
        elif tuple(keyframe_noise.shape) != kshape:
            raise ValueError(f"keyframe_noise must have shape {kshape}, got {keyframe_noise.shape}")
        pixels = self.forward_stage_5_with_keyframes(
            noise.astype(feat.dtype),
            keyframe_noise.astype(feat.dtype),
            feat,
            tile_stream.x,
            tile_stream.times,
            tile_stream.valid,
            mx.array([1.0]),
            drop_leading_frame=tile.is_origin,
        )
        return pixels[:, :, : tile.out_t.stop - tile.out_t.start]

    def _prepare_keyframes(
        self, latent: mx.array, keyframes: DecodeKeyframes, h_pads: tuple[int, int, int, int]
    ) -> DecodeKeyframes:
        """Validate the planes against ``latent`` and pad them to the same spatial floor (no temporal pad)."""
        _, _, f, h, w = latent.shape
        keyframes.validate(num_frames=(f - 1) * self.temporal_scale + 1)
        if tuple(keyframes.latents.shape[3:]) != (h, w):
            raise ValueError(
                f"keyframe planes must share the video's spatial latent grid ({h}, {w}), got {keyframes.latents.shape[3:]}"
            )
        padded, (_, *kpads) = self.pad_to_floor(keyframes.latents.astype(latent.dtype), temporal=False)
        if tuple(kpads) != tuple(h_pads):
            raise RuntimeError(f"keyframe spatial pad {tuple(kpads)} != video pad {tuple(h_pads)}")
        return dataclasses.replace(keyframes, latents=padded)

    def noise_key(self, seed: int, tile_index: int = 0) -> mx.array:
        """PRNG key of a tile's noise draw: ``seed + DIFFVAE_NOISE_SEED_OFFSET + tile_index``."""
        return mx.random.key(seed + DIFFVAE_NOISE_SEED_OFFSET + tile_index)

    def decode_tile(
        self, feat_s4: mx.array, tile: DiffusionTile, *, noise: mx.array | None = None, seed: int = 0
    ) -> mx.array:
        """Stage 4 + 5 on one tile of the stage-4 input feature (upstream ``_decode_one_tile``).

        Returns un-masked pixels ``(B, 3, len(out_t), 8*Lh, 8*Lw)`` in ``feat_s4``'s dtype. A
        trailing tile takes the ghost frames from ``feat_s4`` and is cropped back to its output
        extent.
        """
        t1 = feat_s4.shape[1] if tile.pad_trailing else tile.in_t.stop
        feat = self.forward_stage_4(
            feat_s4[:, tile.in_t.start : t1, tile.in_h, tile.in_w], pad_trailing=tile.pad_trailing
        )
        canvas = self.stage5_canvas(feat.shape[1], feat.shape[2], feat.shape[3], is_origin=tile.is_origin)
        shape = (feat.shape[0], self.config.out_channels, *canvas)
        if noise is None:
            noise = mx.random.normal(shape, key=self.noise_key(seed, tile.index))
        elif tuple(noise.shape) != shape:
            raise ValueError(f"noise must have shape {shape}, got {noise.shape}")
        pixels = self.forward_stage_5(
            noise.astype(feat.dtype), feat, mx.array([1.0]), drop_leading_frame=tile.is_origin
        )
        return pixels[:, :, : tile.out_t.stop - tile.out_t.start]

    # ---- public API ---------------------------------------------------------------
    def decode(
        self,
        latent: mx.array,
        *,
        noise: mx.array | None = None,
        seed: int = 0,
        tap: Callable[[str, mx.array], None] | None = None,
        keyframes: DecodeKeyframes | None = None,
        keyframe_noise: mx.array | None = None,
    ) -> mx.array:
        """Decode ``(B, 128, F, H, W)`` normalised latent to ``(B, 3, 8F-7, 32H, 32W)`` pixels in ``[-1, 1]``.

        ``tap`` is the parity instrumentation hook; see :meth:`forward_stages_1_to_4` and
        :meth:`forward_stage_5` for the names it is called with. When ``keyframes`` is given, the
        dual-stream keyframe-aware path runs instead (see :meth:`forward_stages_1_to_3_with_keyframes`
        onwards); ``keyframes=None`` is the plain path above and stays byte-identical.
        """
        if latent.shape[0] != 1:
            raise ValueError("NADiffusionDecoder decodes one video at a time (batch size 1)")
        # Run in the weights' dtype whatever the caller hands over (an fp32 latent would
        # otherwise promote every activation and double the peak), restore it on the output.
        output_dtype = latent.dtype
        latent = latent.astype(self.weights_dtype())
        _, _, f, h, w = latent.shape
        # T pads live at the end (repeat-last-frame) and are removed by the [:f_px] crop below.
        padded, (_t_pad, h_b, _h_a, w_b, _w_a) = self.pad_to_floor(latent)
        if keyframes is not None:
            kf = self._prepare_keyframes(latent, keyframes, (h_b, _h_a, w_b, _w_a))
            feat, stream = self.forward_stages_1_to_3_with_keyframes(padded, kf, tap=tap)
            feat, stream = self.forward_stage_4_with_keyframes(
                feat, stream, kf.pixel_frame_indices, clip_start_frame=kf.clip_start_frame, pad_trailing=True, tap=tap
            )
            t4, h4, w4 = feat.shape[1], feat.shape[2], feat.shape[3]
            canvas = self.stage5_canvas(t4, h4, w4)
            kshape = (1, self.config.out_channels, kf.num_planes, canvas[1], canvas[2])
            if noise is None:
                noise = mx.random.normal((1, self.config.out_channels, *canvas), key=self.noise_key(seed))
            elif tuple(noise.shape) != (1, self.config.out_channels, *canvas):
                raise ValueError(f"noise must have shape {(1, self.config.out_channels, *canvas)}, got {noise.shape}")
            if keyframe_noise is None:
                keyframe_noise = mx.random.normal(kshape, key=self._keyframe_noise_key(seed))
            elif tuple(keyframe_noise.shape) != kshape:
                raise ValueError(f"keyframe_noise must have shape {kshape}, got {keyframe_noise.shape}")
            pixels = self.forward_stage_5_with_keyframes(
                noise.astype(latent.dtype),
                keyframe_noise.astype(latent.dtype),
                feat,
                stream.x,
                stream.times,
                stream.valid,
                mx.array([1.0]),
                tap=tap,
            )
        else:
            feat = self.forward_stages_1_to_4(padded, tap=tap)
            t4, h4, w4 = feat.shape[1], feat.shape[2], feat.shape[3]
            canvas = self.stage5_canvas(t4, h4, w4)
            if noise is None:
                noise = mx.random.normal((1, self.config.out_channels, *canvas), key=self.noise_key(seed))
            elif tuple(noise.shape) != (1, self.config.out_channels, *canvas):
                raise ValueError(f"noise must have shape {(1, self.config.out_channels, *canvas)}, got {noise.shape}")
            pixels = self.forward_stage_5(noise.astype(latent.dtype), feat, mx.array([1.0]), tap=tap)
        sh, sw = self.spatial_scale
        f_px, h_px, w_px = (f - 1) * self.temporal_scale + 1, h * sh, w * sw
        hb, wb = h_b * sh, w_b * sw
        return pixels[:, :, :f_px, hb : hb + h_px, wb : wb + w_px].astype(output_dtype)

    def tiled_decode(
        self,
        latent: mx.array,
        tiling: DiffusionTileConfig | None = None,
        *,
        seed: int = 0,
        allow_small_overlap: bool = False,
        keyframes: DecodeKeyframes | None = None,
    ) -> Iterator[mx.array]:
        """Stream a (tiled) decode as ``(1, 3, T_chunk, H, W)`` chunks in ``[-1, 1]``, content-cropped.

        Upstream ``_decode_pixels``: stages 1-3 once; per temporal group an fp16 accumulator over the
        padded spatial extent receives every tile ``x`` its separable trapezoid masks; the previous
        group's overlap stub is added, the group's exclusive frames are yielded and the tail is
        carried. A one-tile schedule bypasses the accumulator (byte-identical to :meth:`decode`).
        With ``keyframes``, stages 1-3 run the dual-stream path once and each tile selects (via
        :func:`~ltx_core_mlx.model.video_vae.diffusion_decoder.keyframes.planes_for_tile`) only the
        planes its span needs before running stage 4 + 5 (:meth:`decode_tile_with_keyframes`).
        """
        if latent.shape[0] != 1:
            raise ValueError("NADiffusionDecoder decodes one video at a time (batch size 1)")
        output_dtype = latent.dtype
        latent = latent.astype(self.weights_dtype())  # see decode()
        _, _, f, h, w = latent.shape
        padded, (_t_pad, h_b, _h_a, w_b, _w_a) = self.pad_to_floor(latent)
        geometry = DiffusionTileGeometry.from_config(self.config)
        fhw = (padded.shape[2], padded.shape[3], padded.shape[4])
        tiles = build_tile_schedule(geometry, fhw, tiling, allow_small_overlap=allow_small_overlap)
        f_full, h_full, w_full = output_fhw(geometry, fhw)
        sh, sw = self.spatial_scale
        f_px, h_px, w_px = (f - 1) * self.temporal_scale + 1, h * sh, w * sw
        hb, wb = h_b * sh, w_b * sw

        def crop(chunk: mx.array, start: int) -> mx.array | None:
            keep = min(chunk.shape[2], f_px - start)
            if keep <= 0:
                return None
            return chunk[:, :, :keep, hb : hb + h_px, wb : wb + w_px]

        if len(tiles) > 1 and not masks_are_complementary(tiles, (f_full, h_full, w_full)):
            raise ValueError("diffusion decoder tile masks are not complementary; refusing to blend")
        if keyframes is not None:
            kf = self._prepare_keyframes(latent, keyframes, (h_b, _h_a, w_b, _w_a))
            feat_s4, stream = self.forward_stages_1_to_3_with_keyframes(padded, kf)
            mx.eval(feat_s4, stream.x)

            def decode_one(tile: DiffusionTile) -> mx.array:
                # Pre-select this tile's planes so a tile never carries planes it doesn't need
                # (whole-stream residency would defeat the point of tiling): `keep` is the same
                # rule `decode_tile_with_keyframes` applies internally, so its own re-derivation
                # of `keep` from the already-narrowed `tile_kf` is an identity no-op.
                keep = planes_for_tile(
                    kf.pixel_frame_indices, tile.out_t.start, tile.out_t.stop - 1, clip_start_frame=kf.clip_start_frame
                )
                idx = [i for i, k in enumerate(keep) if k]
                tile_kf = dataclasses.replace(
                    kf,
                    latents=kf.latents[:, :, idx],
                    pixel_frame_indices=tuple(kf.pixel_frame_indices[i] for i in idx),
                )
                return self.decode_tile_with_keyframes(feat_s4, stream.select_planes(keep), tile_kf, tile, seed=seed)
        else:
            feat_s4 = self.forward_stages_1_to_3(padded)
            mx.eval(feat_s4)

            def decode_one(tile: DiffusionTile) -> mx.array:
                return self.decode_tile(feat_s4, tile, seed=seed)

        if len(tiles) == 1:
            chunk = crop(decode_one(tiles[0]).astype(output_dtype), 0)
            if chunk is None:
                raise RuntimeError("diffusion decoder: one-tile decode produced no content frames")
            yield chunk
            return
        acc_dtype = mx.float16 if feat_s4.dtype == mx.bfloat16 else feat_s4.dtype
        groups = group_tiles_by_temporal_slice(tiles)
        starts = [g[0].out_t.start for g in groups]
        stub: mx.array | None = None
        for gi, group in enumerate(groups):
            g_start, g_stop = group[0].out_t.start, group[0].out_t.stop
            buffer = mx.zeros((1, self.config.out_channels, g_stop - g_start, h_full, w_full), dtype=acc_dtype)
            for tile in group:
                px = decode_one(tile)
                w = tile.mask_t[:, None, None] * tile.mask_h[None, :, None] * tile.mask_w[None, None, :]
                coords = (slice(None), slice(None), slice(None), tile.out_h, tile.out_w)
                buffer[coords] = (buffer[coords] + px * w[None, None]).astype(acc_dtype)
                mx.eval(buffer)
                del px, w
                aggressive_cleanup()
            if stub is not None:
                n = stub.shape[2]
                if n > buffer.shape[2]:
                    raise ValueError(
                        f"diffusion decoder tiling: overlap stub of {n} frames exceeds the next "
                        f"temporal group ({buffer.shape[2]} frames)"
                    )
                buffer[:, :, :n] = (buffer[:, :, :n] + stub).astype(acc_dtype)
            if gi < len(groups) - 1:
                exclusive = min(max(0, starts[gi + 1] - g_start), g_stop - g_start)
                chunk = crop(buffer[:, :, :exclusive].astype(output_dtype), g_start)
                stub = buffer[:, :, exclusive:]
            else:
                chunk = crop(buffer.astype(output_dtype), g_start)
            mx.eval(chunk if chunk is not None else buffer, *([stub] if stub is not None else []))
            del buffer
            if chunk is not None:
                yield chunk


def load_diffusion_decoder(path: str | Path, config: DiffusionDecoderConfig | None = None) -> NADiffusionDecoder:
    """Build and load an :class:`NADiffusionDecoder` from a pack's ``vae_decoder_av.safetensors`` (strict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"diffusion video decoder weights not found at {path}")
    cfg = config or DiffusionDecoderConfig.from_safetensors_metadata(path)
    model = NADiffusionDecoder(cfg)
    weights = load_split_safetensors(path, prefix="vae_decoder_av.")
    model.load_weights(list(weights.items()))  # strict: every param fed, no unknown key
    return model
