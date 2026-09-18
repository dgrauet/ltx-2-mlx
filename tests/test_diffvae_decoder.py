"""NADiffusionDecoder assembly: geometry, parameter contract, end-to-end shapes on the tiny config, pack load contract."""

from __future__ import annotations

import json
import struct

import mlx.core as mx
import mlx.utils
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder import (
    DIFFVAE_NOISE_SEED_OFFSET,
    NADiffusionDecoder,
    load_diffusion_decoder,
)
from ltx_core_mlx.model.video_vae.diffusion_decoder.config import LTX_2_5_DIFFUSION_DECODER
from ltx_core_mlx.model.video_vae.diffusion_decoder.tiling import (
    DiffusionTileConfig,
    DiffusionTileGeometry,
    build_tile_schedule,
)
from tests.conftest import LTX25_Q8_DIR
from tests.diffvae_tiny import TINY

TINY_CFG = DiffusionTileConfig(16, 8, 64, 32, 64, 32)


def test_pad_to_floor_replicates_last_frame_and_symmetric_edges():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 2, 2, 5))  # floor is (3, 3, 3)
    zp, pads = dec.pad_to_floor(z)
    assert zp.shape == (1, 8, 3, 3, 5) and pads == (1, 0, 1, 0, 0)
    assert mx.array_equal(zp[:, :, 2], zp[:, :, 1])  # T: repeat last frame
    assert mx.array_equal(zp[:, :, :, 2], zp[:, :, :, 1])  # H after-pad: edge replicate
    z2 = mx.random.normal((1, 8, 3, 1, 3))
    _, pads2 = dec.pad_to_floor(z2)
    assert pads2 == (0, 1, 1, 0, 0)  # need 2 -> before 1, after 1


def test_stage_shapes_on_tiny_config():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 3, 3, 3))
    feat = dec.forward_stages_1_to_4(z)  # ghost pad 2 -> T=5 latent frames
    # stage chain on T: 5 -> up0 (1,2,2): 5 -> up1 (2,1,1) drop: 9 -> up2 (2,2,2) drop: 17 = T4 ; ghost crop keep = min(17, max(17-8, 2)) = 9
    assert feat.shape == (1, 9, 12, 12, 8)
    assert dec.stage5_canvas(9, 12, 12) == (17, 96, 96)
    x_t = mx.random.normal((1, 3, 17, 96, 96))
    px = dec.forward_stage_5(x_t, feat, mx.array([1.0]))
    assert px.shape == (1, 3, 17, 96, 96)


def test_decode_crops_to_content_and_is_deterministic_for_a_seed():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 2, 2, 3))  # content 9 frames, 64 x 96 px
    a = dec.decode(z, seed=3)
    b = dec.decode(z, seed=3)
    assert a.shape == (1, 3, 9, 64, 96) and mx.array_equal(a, b)
    assert not mx.array_equal(a, dec.decode(z, seed=4))
    chunks = list(dec.tiled_decode(z))
    assert len(chunks) == 1 and chunks[0].shape == a.shape


def test_injected_noise_bypasses_the_rng():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 3, 3, 3))
    noise = mx.random.normal((1, 3, 17, 96, 96))
    assert mx.array_equal(dec.decode(z, noise=noise), dec.decode(z, noise=noise))
    # Explicit noise makes `seed` irrelevant.
    assert mx.array_equal(dec.decode(z, noise=noise, seed=1), dec.decode(z, noise=noise, seed=2))


def test_decode_rejects_batch_and_bad_noise_shape():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((2, 8, 3, 3, 3))
    with pytest.raises(ValueError):
        dec.decode(z)
    z1 = mx.random.normal((1, 8, 3, 3, 3))
    bad_noise = mx.random.normal((1, 3, 16, 96, 96))  # wrong T
    with pytest.raises(ValueError):
        dec.decode(z1, noise=bad_noise)


def test_pixel_scales_derived_from_config():
    dec = NADiffusionDecoder(LTX_2_5_DIFFUSION_DECODER)
    assert dec.temporal_scale == 8
    assert dec.spatial_scale == (32, 32)


def test_parameter_names_match_the_pack_schema():
    dec = NADiffusionDecoder(TINY)
    keys = set(dict(mlx.utils.tree_flatten(dec.parameters())))
    for k in [
        "conv_in.weight",
        "conv_in_x_t.bias",
        "conv_out.weight",
        "type_emb",
        "det_stages.0.0.attn.qkv.weight",
        "det_stages.3.0.mlp.w_down.weight",
        "upsamples.3.proj.bias",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.bias",
        "shared_adaln.proj.weight",
        "diff_blocks.1.scale_shift_table",
        "diff_blocks.0.context_proj.weight",
        "norm_out.weight",
        "per_channel_statistics.mean",
        "per_channel_statistics.std",
    ]:
        assert k in keys, k
    assert dec.diff_blocks[0].scale_shift_table.shape == (7, 4)
    assert dec.shared_adaln.proj.weight.shape == (28, 8)


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_pack_load_contract_bidirectional():
    path = LTX25_Q8_DIR / "vae_decoder_av.safetensors"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    pack_keys = {k.removeprefix("vae_decoder_av.") for k in header if k != "__metadata__"}
    dec = load_diffusion_decoder(path)  # strict: raises if a param is unfed
    model_keys = set(dict(mlx.utils.tree_flatten(dec.parameters())))
    assert pack_keys == model_keys, pack_keys ^ model_keys
    assert dec.per_channel_statistics.mean.shape == (128,)


def test_split_forward_composes_to_the_old_forward():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 3, 3, 3))
    feat_s4 = dec.forward_stages_1_to_3(z)
    assert feat_s4.shape == (1, 17, 12, 12, 8)  # 9 content + 8 ghost frames
    assert mx.array_equal(dec.forward_stage_4(feat_s4, pad_trailing=True), dec.forward_stages_1_to_4(z))
    assert dec.forward_stage_4(feat_s4[:, 2:10], pad_trailing=False).shape == (1, 8, 12, 12, 8)
    assert dec.stage5_canvas(8, 12, 12, is_origin=False) == (16, 96, 96)
    assert dec.stage5_canvas(8, 12, 12) == (15, 96, 96)


def test_noise_key_offsets_by_tile_index():
    dec = NADiffusionDecoder(TINY)
    assert mx.array_equal(dec.noise_key(3), mx.random.key(3 + DIFFVAE_NOISE_SEED_OFFSET))
    assert not mx.array_equal(dec.noise_key(3, 1), dec.noise_key(3, 0))


def test_one_tile_tiled_decode_equals_decode():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 2, 2, 3))
    chunks = list(dec.tiled_decode(z, None, seed=3))
    assert len(chunks) == 1 and mx.array_equal(chunks[0], dec.decode(z, seed=3))


def _tiles_and_feat(dec, z):
    geometry = DiffusionTileGeometry.from_config(dec.config)
    tiles = build_tile_schedule(geometry, tuple(z.shape[2:]), TINY_CFG)
    return tiles, dec.forward_stages_1_to_3(z)


def test_origin_tile_matches_untiled_on_its_exclusive_region():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 5, 3, 3))  # at the floor: no padding; output 33 x 96 x 96
    tiles, feat_s4 = _tiles_and_feat(dec, z)
    full_noise = mx.random.normal((1, 3, 33, 96, 96))
    untiled = dec.decode(z, noise=full_noise)
    origin = tiles[0]
    assert origin.is_origin and not origin.pad_trailing and origin.out_t == slice(0, 15)
    px = dec.decode_tile(feat_s4, origin, noise=full_noise[:, :, :15, :64, :64])
    assert px.shape == (1, 3, 15, 64, 64)
    # exclusive region = up to the next tile's start on each axis (frames 7, px 32); receptive field
    # of TINY is 2 stage-4 cells (4 frames / 16 px) so those pixels see identical inputs.
    assert mx.allclose(px[:, :, :7, :32, :32], untiled[:, :, :7, :32, :32], atol=1e-5, rtol=1e-5).item()


def test_non_origin_tile_is_frame_aligned_with_untiled():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 5, 3, 3))
    tiles, feat_s4 = _tiles_and_feat(dec, z)
    full_noise = mx.random.normal((1, 3, 33, 96, 96))
    untiled = dec.decode(z, noise=full_noise)
    tile = tiles[4]  # in_t [4,12) -> out_t [7,23), h/w [0,64)
    assert not tile.is_origin and tile.out_t == slice(7, 23)
    px = dec.decode_tile(feat_s4, tile, noise=full_noise[:, :, 7:23, :64, :64])
    assert px.shape == (1, 3, 16, 64, 64)
    # interior frames (4 frames in from both tile ends) and the interior spatial block
    assert mx.allclose(px[:, :, 4:12, :32, :32], untiled[:, :, 11:19, :32, :32], atol=1e-5, rtol=1e-5).item()


def test_trailing_tile_gets_ghost_frames_and_is_cropped_to_its_output():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 5, 3, 3))
    tiles, feat_s4 = _tiles_and_feat(dec, z)
    tile = tiles[12]  # in_t [12,17) -> out_t [23,33); pad_trailing
    assert tile.pad_trailing and not tile.is_origin
    px = dec.decode_tile(feat_s4, tile, seed=0)
    assert px.shape == (1, 3, 10, 64, 64)


def test_tiled_decode_streams_exclusive_frames_and_covers_the_content():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 5, 3, 3))
    chunks = list(dec.tiled_decode(z, TINY_CFG, seed=1))
    assert [c.shape for c in chunks] == [(1, 3, 7, 96, 96), (1, 3, 8, 96, 96), (1, 3, 8, 96, 96), (1, 3, 10, 96, 96)]
    video = mx.concatenate(chunks, axis=2)
    assert video.dtype == z.dtype and mx.all(mx.isfinite(video)).item()
    assert mx.abs(video).max().item() < 50  # blended, not summed twice
    # determinism per seed
    again = mx.concatenate(list(dec.tiled_decode(z, TINY_CFG, seed=1)), axis=2)
    assert mx.array_equal(video, again)


def test_tiled_decode_crops_padding_and_content():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 4, 2, 3))  # H below floor -> padded to 3 then cropped back; 25 frames
    video = mx.concatenate(list(dec.tiled_decode(z, TINY_CFG, seed=1)), axis=2)
    assert video.shape == (1, 3, 25, 64, 96)
