"""Tests for video and audio patchifiers."""

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    compute_video_latent_shape,
)


# ---------------------------------------------------------------------------
# VideoLatentPatchifier
# ---------------------------------------------------------------------------
class TestVideoLatentPatchifier:
    def test_patchify_shape(self):
        patchifier = VideoLatentPatchifier()
        latent = mx.zeros((1, 128, 4, 2, 3))  # (B, C, F, H, W)
        tokens, spatial_dims = patchifier.patchify(latent)
        assert tokens.shape == (1, 4 * 2 * 3, 128)  # (B, F*H*W, C)
        assert spatial_dims == (4, 2, 3)

    def test_unpatchify_shape(self):
        patchifier = VideoLatentPatchifier()
        tokens = mx.zeros((1, 24, 128))  # (B, N=4*2*3, C)
        latent = patchifier.unpatchify(tokens, spatial_dims=(4, 2, 3))
        assert latent.shape == (1, 128, 4, 2, 3)

    def test_roundtrip_identity(self):
        """patchify -> unpatchify should be an identity operation."""
        patchifier = VideoLatentPatchifier()
        original = mx.random.normal((1, 128, 3, 2, 4))
        tokens, dims = patchifier.patchify(original)
        reconstructed = patchifier.unpatchify(tokens, dims)
        assert mx.allclose(reconstructed, original, atol=1e-6).item()

    def test_roundtrip_unpatchify_patchify(self):
        """unpatchify -> patchify should also be identity."""
        patchifier = VideoLatentPatchifier()
        tokens_orig = mx.random.normal((1, 12, 64))
        latent = patchifier.unpatchify(tokens_orig, spatial_dims=(3, 2, 2))
        tokens_back, dims = patchifier.patchify(latent)
        assert dims == (3, 2, 2)
        assert mx.allclose(tokens_back, tokens_orig, atol=1e-6).item()

    def test_single_frame(self):
        patchifier = VideoLatentPatchifier()
        latent = mx.zeros((1, 128, 1, 4, 4))
        tokens, dims = patchifier.patchify(latent)
        assert tokens.shape == (1, 16, 128)
        assert dims == (1, 4, 4)

    def test_batch_size_two(self):
        patchifier = VideoLatentPatchifier()
        latent = mx.zeros((2, 64, 2, 3, 3))
        tokens, dims = patchifier.patchify(latent)
        assert tokens.shape == (2, 18, 64)
        assert dims == (2, 3, 3)

    def test_values_preserved(self):
        """Check that values are not corrupted during patchify."""
        patchifier = VideoLatentPatchifier()
        # Create known pattern: channel c, frame f, h, w -> value = c + f*100 + h*10 + w
        B, C, F, H, W = 1, 4, 2, 2, 2
        latent = mx.zeros((B, C, F, H, W))
        for c in range(C):
            for f in range(F):
                for h in range(H):
                    for w in range(W):
                        val = c + f * 100 + h * 10 + w
                        latent = latent.at[0, c, f, h, w].add(mx.array(float(val)))
        tokens, dims = patchifier.patchify(latent)
        reconstructed = patchifier.unpatchify(tokens, dims)
        assert mx.allclose(reconstructed, latent, atol=1e-5).item()

    def test_token_ordering(self):
        """Token 0 should correspond to (f=0, h=0, w=0), token 1 to (f=0, h=0, w=1), etc."""
        patchifier = VideoLatentPatchifier()
        # Make a latent where each spatial position has a unique value in channel 0
        F, H, W = 2, 2, 3
        latent = mx.zeros((1, 1, F, H, W))
        for f in range(F):
            for h in range(H):
                for w in range(W):
                    val = f * 100 + h * 10 + w
                    latent = latent.at[0, 0, f, h, w].add(mx.array(float(val)))
        tokens, _ = patchifier.patchify(latent)
        # Token ordering: (f, h, w) iterating w fastest, then h, then f
        expected_order = [f * 100 + h * 10 + w for f in range(F) for h in range(H) for w in range(W)]
        for i, expected_val in enumerate(expected_order):
            assert abs(float(tokens[0, i, 0]) - expected_val) < 1e-5

    def test_1x1x1_spatial(self):
        """Minimal spatial dimensions."""
        patchifier = VideoLatentPatchifier()
        latent = mx.ones((1, 128, 1, 1, 1))
        tokens, dims = patchifier.patchify(latent)
        assert tokens.shape == (1, 1, 128)
        assert dims == (1, 1, 1)
        reconstructed = patchifier.unpatchify(tokens, dims)
        assert mx.allclose(reconstructed, latent, atol=1e-6).item()


# ---------------------------------------------------------------------------
# AudioPatchifier
# ---------------------------------------------------------------------------
class TestAudioPatchifier:
    def test_patchify_shape(self):
        patchifier = AudioPatchifier()
        latent = mx.zeros((1, 8, 10, 16))  # (B, 8, T, 16)
        tokens, T = patchifier.patchify(latent)
        assert tokens.shape == (1, 10, 128)  # (B, T, 8*16)
        assert T == 10

    def test_unpatchify_shape(self):
        patchifier = AudioPatchifier()
        tokens = mx.zeros((1, 10, 128))
        latent = patchifier.unpatchify(tokens)
        assert latent.shape == (1, 8, 10, 16)

    def test_roundtrip_identity(self):
        patchifier = AudioPatchifier()
        original = mx.random.normal((1, 8, 7, 16))
        tokens, T = patchifier.patchify(original)
        reconstructed = patchifier.unpatchify(tokens)
        assert mx.allclose(reconstructed, original, atol=1e-6).item()

    def test_roundtrip_unpatchify_patchify(self):
        patchifier = AudioPatchifier()
        tokens_orig = mx.random.normal((1, 5, 128))
        latent = patchifier.unpatchify(tokens_orig)
        tokens_back, T = patchifier.patchify(latent)
        assert T == 5
        assert mx.allclose(tokens_back, tokens_orig, atol=1e-6).item()

    def test_single_token(self):
        patchifier = AudioPatchifier()
        latent = mx.ones((1, 8, 1, 16))
        tokens, T = patchifier.patchify(latent)
        assert tokens.shape == (1, 1, 128)
        assert T == 1

    def test_batch_size_two(self):
        patchifier = AudioPatchifier()
        latent = mx.zeros((2, 8, 5, 16))
        tokens, T = patchifier.patchify(latent)
        assert tokens.shape == (2, 5, 128)
        assert T == 5

    def test_unpatchify_ignores_time_dim_arg(self):
        """_time_dim parameter is ignored (API symmetry)."""
        patchifier = AudioPatchifier()
        tokens = mx.zeros((1, 10, 128))
        latent1 = patchifier.unpatchify(tokens)
        latent2 = patchifier.unpatchify(tokens, _time_dim=999)
        assert mx.allclose(latent1, latent2).item()


# ---------------------------------------------------------------------------
# compute_video_latent_shape
# ---------------------------------------------------------------------------
class TestComputeVideoLatentShape:
    def test_basic(self):
        F, H, W = compute_video_latent_shape(num_frames=24, height=512, width=768)
        assert F == 3  # ceil(24/8)
        assert H == 16  # 512/32
        assert W == 24  # 768/32

    def test_exact_divisible(self):
        F, H, W = compute_video_latent_shape(num_frames=8, height=32, width=64)
        assert F == 1
        assert H == 1
        assert W == 2

    def test_temporal_rounding_up(self):
        """Temporal dimension should round up (ceiling division)."""
        F1, _, _ = compute_video_latent_shape(num_frames=1, height=32, width=32)
        assert F1 == 1  # ceil(1/8) = 1
        F9, _, _ = compute_video_latent_shape(num_frames=9, height=32, width=32)
        assert F9 == 2  # ceil(9/8) = 2
        F16, _, _ = compute_video_latent_shape(num_frames=16, height=32, width=32)
        assert F16 == 2  # ceil(16/8) = 2
        F17, _, _ = compute_video_latent_shape(num_frames=17, height=32, width=32)
        assert F17 == 3  # ceil(17/8) = 3

    def test_spatial_floor_division(self):
        """Spatial dimensions use floor division."""
        _, H, W = compute_video_latent_shape(num_frames=8, height=100, width=100)
        assert H == 3  # 100 // 32
        assert W == 3

    def test_custom_compression(self):
        F, H, W = compute_video_latent_shape(
            num_frames=16, height=256, width=256, temporal_compression=4, spatial_compression=16
        )
        assert F == 4  # ceil(16/4)
        assert H == 16  # 256/16
        assert W == 16

    def test_standard_resolutions(self):
        """Test common generation resolutions."""
        # 512x512, 49 frames (common LTX config)
        F, H, W = compute_video_latent_shape(num_frames=49, height=512, width=512)
        assert F == 7  # ceil(49/8)
        assert H == 16
        assert W == 16

        # 768x512, 97 frames
        F, H, W = compute_video_latent_shape(num_frames=97, height=512, width=768)
        assert F == 13  # ceil(97/8)
        assert H == 16
        assert W == 24

    def test_single_frame(self):
        F, H, W = compute_video_latent_shape(num_frames=1, height=256, width=256)
        assert F == 1
        assert H == 8
        assert W == 8
