"""``NADiffusionDecoder``: the LTX-2.5 diffusion video decoder, plain path, tiled.

Flow (spec §3): size floor + ghost pad -> de-normalise -> conv_in -> det stages 1-3 with
upsamples -> stage 4 blocks (upsample 3 deferred to the diffusion blocks) -> ghost crop ->
pure-noise ``x_t`` -> one diffusion step at ``t = 1`` (``x0`` output is the pixels) -> crop.
:meth:`NADiffusionDecoder.tiled_decode` streams this per stage-4/stage-5 tile, blending
overlaps with trapezoid masks and yielding each temporal group's exclusive frames as soon
as its tiles are decoded, so peak activation memory stays bounded regardless of clip length.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.diffusion_decoder.blocks import NABlock
from ltx_core_mlx.model.video_vae.diffusion_decoder.chunked import ChunkedDiffusionNABlock
from ltx_core_mlx.model.video_vae.diffusion_decoder.config import LTX_2_5_DIFFUSION_DECODER, DiffusionDecoderConfig
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
        # Added by upstream only to keyframe latents in the keyframe-aware decode
        # (DecodeKeyframes); the plain decode never reads it. Loaded for the load
        # contract; inert in v1.
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

    # ---- geometry -----------------------------------------------------------------
    def denormalize_latent(self, z: mx.array) -> mx.array:
        s, m = self.per_channel_statistics.std, self.per_channel_statistics.mean
        return z * s.reshape(1, -1, 1, 1, 1).astype(z.dtype) + m.reshape(1, -1, 1, 1, 1).astype(z.dtype)

    def pad_to_floor(self, latent: mx.array) -> tuple[mx.array, tuple[int, int, int, int, int]]:
        """Pad ``(B, C, F, H, W)`` up to the config's minimum latent shape (T: repeat last; H/W: symmetric edge)."""
        f_min, h_min, w_min = self.config.min_latent_shape()
        _, _, f, h, w = latent.shape
        t_pad = max(f_min - f, 0)
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
    ) -> mx.array:
        """Decode ``(B, 128, F, H, W)`` normalised latent to ``(B, 3, 8F-7, 32H, 32W)`` pixels in ``[-1, 1]``.

        ``tap`` is the parity instrumentation hook; see :meth:`forward_stages_1_to_4` and
        :meth:`forward_stage_5` for the names it is called with.
        """
        if latent.shape[0] != 1:
            raise ValueError("NADiffusionDecoder decodes one video at a time (batch size 1)")
        _, _, f, h, w = latent.shape
        # T pads live at the end (repeat-last-frame) and are removed by the [:f_px] crop below.
        padded, (_t_pad, h_b, _h_a, w_b, _w_a) = self.pad_to_floor(latent)
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
        return pixels[:, :, :f_px, hb : hb + h_px, wb : wb + w_px]

    def tiled_decode(
        self,
        latent: mx.array,
        tiling: DiffusionTileConfig | None = None,
        *,
        seed: int = 0,
        allow_small_overlap: bool = False,
    ) -> Iterator[mx.array]:
        """Stream a (tiled) decode as ``(1, 3, T_chunk, H, W)`` chunks in ``[-1, 1]``, content-cropped.

        Upstream ``_decode_pixels``: stages 1-3 once; per temporal group an fp16 accumulator over the
        padded spatial extent receives every tile ``x`` its separable trapezoid masks; the previous
        group's overlap stub is added, the group's exclusive frames are yielded and the tail is
        carried. A one-tile schedule bypasses the accumulator (byte-identical to :meth:`decode`).
        """
        if latent.shape[0] != 1:
            raise ValueError("NADiffusionDecoder decodes one video at a time (batch size 1)")
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
        feat_s4 = self.forward_stages_1_to_3(padded)
        mx.eval(feat_s4)
        if len(tiles) == 1:
            chunk = crop(self.decode_tile(feat_s4, tiles[0], seed=seed).astype(latent.dtype), 0)
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
                px = self.decode_tile(feat_s4, tile, seed=seed)
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
                chunk = crop(buffer[:, :, :exclusive].astype(latent.dtype), g_start)
                stub = buffer[:, :, exclusive:]
            else:
                chunk = crop(buffer.astype(latent.dtype), g_start)
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
