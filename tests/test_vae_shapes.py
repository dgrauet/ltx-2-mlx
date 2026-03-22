"""Shape tests for the Video VAE."""

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, compute_video_latent_shape
from ltx_core_mlx.model.video_vae.convolution import Conv3dBlock
from ltx_core_mlx.model.video_vae.ops import PerChannelStatistics
from ltx_core_mlx.model.video_vae.resnet import ResBlock3d, ResBlockStage
from ltx_core_mlx.model.video_vae.sampling import DepthToSpaceUpsample as UpsampleConv
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder, VideoEncoder

# ---------------------------------------------------------------------------
# Patchifier tests
# ---------------------------------------------------------------------------


class TestVideoLatentPatchifier:
    def test_patchify_unpatchify_roundtrip(self):
        patchifier = VideoLatentPatchifier()
        latent = mx.zeros((1, 128, 4, 2, 3))  # BCFHW
        tokens, dims = patchifier.patchify(latent)
        assert tokens.shape == (1, 4 * 2 * 3, 128)
        assert dims == (4, 2, 3)

        recovered = patchifier.unpatchify(tokens, dims)
        assert recovered.shape == latent.shape

    def test_values_preserved(self):
        patchifier = VideoLatentPatchifier()
        latent = mx.random.normal((1, 8, 2, 3, 4))
        tokens, dims = patchifier.patchify(latent)
        recovered = patchifier.unpatchify(tokens, dims)
        assert mx.allclose(latent, recovered).item()


class TestAudioPatchifier:
    def test_patchify_unpatchify_roundtrip(self):
        patchifier = AudioPatchifier()
        latent = mx.zeros((1, 8, 10, 16))  # B, 8, T, 16
        tokens, T = patchifier.patchify(latent)
        assert tokens.shape == (1, 10, 128)
        assert T == 10

        recovered = patchifier.unpatchify(tokens)
        assert recovered.shape == latent.shape


class TestComputeVideoLatentShape:
    def test_standard_dims(self):
        F, H, W = compute_video_latent_shape(97, 480, 704)
        assert H == 480 // 32
        assert W == 704 // 32

    def test_temporal_compression(self):
        F, H, W = compute_video_latent_shape(97, 480, 704, temporal_compression=8)
        assert F == 13  # ceil(97/8)


# ---------------------------------------------------------------------------
# Building block tests
# ---------------------------------------------------------------------------


class TestConv3dBlock:
    def test_output_shape_causal(self):
        conv = Conv3dBlock(16, 32, kernel_size=3, padding=1, causal=True)
        x = mx.zeros((1, 4, 8, 8, 16))
        y = conv(x)
        assert y.shape == (1, 4, 8, 8, 32)

    def test_output_shape_non_causal(self):
        conv = Conv3dBlock(16, 32, kernel_size=3, padding=1, causal=False)
        x = mx.zeros((1, 4, 8, 8, 16))
        y = conv(x)
        assert y.shape == (1, 4, 8, 8, 32)

    def test_key_structure(self):
        conv = Conv3dBlock(16, 32)
        params = dict(conv.parameters())
        flat = {k: v for k, v in nn.utils.tree_flatten(params)}
        assert "conv.weight" in flat
        assert "conv.bias" in flat


class TestResBlock3d:
    def test_output_shape(self):
        block = ResBlock3d(64)
        x = mx.zeros((1, 2, 4, 4, 64))
        y = block(x)
        assert y.shape == x.shape

    def test_key_structure(self):
        block = ResBlock3d(64)
        flat = {k: v for k, v in nn.utils.tree_flatten(block.parameters())}
        assert "conv1.conv.weight" in flat
        assert "conv1.conv.bias" in flat
        assert "conv2.conv.weight" in flat
        assert "conv2.conv.bias" in flat
        # No norm keys
        norm_keys = [k for k in flat if "norm" in k]
        assert len(norm_keys) == 0


class TestResBlockStage:
    def test_key_structure(self):
        stage = ResBlockStage(64, num_blocks=3)
        flat = {k: v for k, v in nn.utils.tree_flatten(stage.parameters())}
        for i in range(3):
            assert f"res_blocks.{i}.conv1.conv.weight" in flat
            assert f"res_blocks.{i}.conv2.conv.weight" in flat


class TestUpsampleConv:
    def test_key_structure(self):
        up = UpsampleConv(64, 128)
        flat = {k: v for k, v in nn.utils.tree_flatten(up.parameters())}
        assert "conv.conv.weight" in flat
        assert "conv.conv.bias" in flat


class TestPerChannelStatistics:
    def test_key_structure(self):
        stats = PerChannelStatistics(128)
        flat = {k: v for k, v in nn.utils.tree_flatten(stats.parameters())}
        assert "mean" in flat
        assert "std" in flat


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------


class TestVideoDecoder:
    def test_key_structure(self):
        """Verify the decoder produces weight keys matching the safetensors file."""
        decoder = VideoDecoder()
        flat = {k: v for k, v in nn.utils.tree_flatten(decoder.parameters())}

        # Top-level keys
        assert "conv_in.conv.weight" in flat
        assert "conv_in.conv.bias" in flat
        assert "conv_out.conv.weight" in flat
        assert "conv_out.conv.bias" in flat
        assert "per_channel_statistics.mean" in flat
        assert "per_channel_statistics.std" in flat

        # ResStage blocks (even indices)
        assert "up_blocks.0.res_blocks.0.conv1.conv.weight" in flat
        assert "up_blocks.0.res_blocks.1.conv2.conv.bias" in flat
        assert "up_blocks.4.res_blocks.3.conv1.conv.weight" in flat
        assert "up_blocks.6.res_blocks.5.conv2.conv.weight" in flat
        assert "up_blocks.8.res_blocks.3.conv1.conv.weight" in flat

        # UpsampleConv blocks (odd indices)
        assert "up_blocks.1.conv.conv.weight" in flat
        assert "up_blocks.1.conv.conv.bias" in flat
        assert "up_blocks.3.conv.conv.weight" in flat
        assert "up_blocks.5.conv.conv.weight" in flat
        assert "up_blocks.7.conv.conv.weight" in flat

        # Should NOT have norm or mid_block keys
        norm_keys = [k for k in flat if "norm" in k]
        assert len(norm_keys) == 0, f"Unexpected norm keys: {norm_keys}"
        mid_keys = [k for k in flat if "mid_block" in k]
        assert len(mid_keys) == 0, f"Unexpected mid_block keys: {mid_keys}"

    def test_no_extra_top_level_keys(self):
        """Ensure only expected top-level parameter groups exist."""
        decoder = VideoDecoder()
        flat = {k: v for k, v in nn.utils.tree_flatten(decoder.parameters())}
        prefixes = {k.split(".")[0] for k in flat}
        expected = {"conv_in", "conv_out", "up_blocks", "per_channel_statistics"}
        assert prefixes == expected, f"Unexpected prefixes: {prefixes - expected}"


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------


class TestVideoEncoder:
    def test_key_structure(self):
        """Verify the encoder produces weight keys matching the safetensors file."""
        encoder = VideoEncoder()
        flat = {k: v for k, v in nn.utils.tree_flatten(encoder.parameters())}

        # Top-level keys
        assert "conv_in.conv.weight" in flat
        assert "conv_out.conv.weight" in flat
        assert "per_channel_statistics.mean_of_means" in flat
        assert "per_channel_statistics.std_of_means" in flat

        # ResStage blocks (even indices)
        assert "down_blocks.0.res_blocks.0.conv1.conv.weight" in flat
        assert "down_blocks.0.res_blocks.3.conv2.conv.weight" in flat
        assert "down_blocks.2.res_blocks.5.conv1.conv.weight" in flat
        assert "down_blocks.8.res_blocks.1.conv2.conv.weight" in flat

        # DownsampleConv blocks (odd indices)
        assert "down_blocks.1.conv.conv.weight" in flat
        assert "down_blocks.3.conv.conv.weight" in flat
        assert "down_blocks.5.conv.conv.weight" in flat
        assert "down_blocks.7.conv.conv.weight" in flat

        # Should NOT have norm or mid_block keys
        norm_keys = [k for k in flat if "norm" in k]
        assert len(norm_keys) == 0
        mid_keys = [k for k in flat if "mid_block" in k]
        assert len(mid_keys) == 0

    def test_no_extra_top_level_keys(self):
        encoder = VideoEncoder()
        flat = {k: v for k, v in nn.utils.tree_flatten(encoder.parameters())}
        prefixes = {k.split(".")[0] for k in flat}
        expected = {"conv_in", "conv_out", "down_blocks", "per_channel_statistics"}
        assert prefixes == expected, f"Unexpected prefixes: {prefixes - expected}"
