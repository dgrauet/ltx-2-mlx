"""Tests for pixel_shuffle_3d and unpatchify_spatial correctness.

Verifies that:
1. pixel_shuffle_3d correctly implements the DepthToSpaceUpsample pattern
   ``"b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"``
2. unpatchify_spatial correctly implements the VAE output unpatchify pattern
   ``"b (c p r q) f h w -> b c (f p) (h q) (w r)"``
3. patchify_spatial and unpatchify_spatial are exact inverses (roundtrip)
4. Adjacent output pixels are smooth (no checkerboard) for gradient inputs
"""

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.video_vae.sampling import (
    patchify_spatial,
    pixel_shuffle_3d,
    space_to_depth,
    unpatchify_spatial,
)


class TestPixelShuffle3d:
    """Tests for pixel_shuffle_3d (DepthToSpaceUpsample pattern)."""

    def test_shape(self):
        x = mx.zeros((1, 2, 3, 4, 128))  # BDHWC, C=128=16*2*2*2
        out = pixel_shuffle_3d(x, spatial_factor=2, temporal_factor=2)
        assert out.shape == (1, 4, 6, 8, 16)

    def test_inverse_of_space_to_depth(self):
        """pixel_shuffle_3d should be the exact inverse of space_to_depth."""
        original = mx.random.normal((1, 4, 6, 8, 16))
        packed = space_to_depth(original, stride=(2, 2, 2))
        recovered = pixel_shuffle_3d(packed, spatial_factor=2, temporal_factor=2)
        np.testing.assert_allclose(
            np.array(recovered),
            np.array(original),
            atol=1e-6,
            err_msg="pixel_shuffle_3d is not the inverse of space_to_depth",
        )

    def test_channel_placement_matches_reference(self):
        """Verify specific channel->position mapping matches reference.

        Reference: "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"
        with p1=temporal, p2=height, p3=width.

        For sf=2, tf=1: channel layout is (c, p2_H, p3_W).
        Channel 0 (c=0, p2=0, p3=0) -> (h=0, w=0)
        Channel 1 (c=0, p2=0, p3=1) -> (h=0, w=1)  -- p3 innermost -> W
        Channel 2 (c=0, p2=1, p3=0) -> (h=1, w=0)  -- p2 -> H
        Channel 3 (c=0, p2=1, p3=1) -> (h=1, w=1)
        """
        x = mx.zeros((1, 1, 1, 1, 4))
        x = x.at[0, 0, 0, 0, 0].add(10.0)  # c=0, p2=0, p3=0 -> h=0, w=0
        x = x.at[0, 0, 0, 0, 1].add(20.0)  # c=0, p2=0, p3=1 -> h=0, w=1
        x = x.at[0, 0, 0, 0, 2].add(30.0)  # c=0, p2=1, p3=0 -> h=1, w=0
        x = x.at[0, 0, 0, 0, 3].add(40.0)  # c=0, p2=1, p3=1 -> h=1, w=1

        out = pixel_shuffle_3d(x, spatial_factor=2, temporal_factor=1)
        # Output: (1, 1, 2, 2, 1)
        assert out.shape == (1, 1, 2, 2, 1)

        out_np = np.array(out[0, 0, :, :, 0])
        expected = np.array([[10.0, 20.0], [30.0, 40.0]])
        np.testing.assert_allclose(out_np, expected, err_msg="pixel_shuffle_3d channel->position mapping is wrong")


class TestUnpatchifySpatial:
    """Tests for unpatchify_spatial (VAE output unpatchify pattern)."""

    def test_shape(self):
        x = mx.zeros((1, 2, 3, 4, 48))  # BFHWC, 48 = 3 * 4 * 4
        out = unpatchify_spatial(x, patch_size=4)
        assert out.shape == (1, 2, 12, 16, 3)

    def test_channel_placement_matches_reference(self):
        """Verify channel->position mapping matches reference unpatchify.

        Reference: "b (c p r q) f h w -> b c (f p) (h q) (w r)"
        with p=1, r=patch_size, q=patch_size.

        Channel ordering: (c, r_W, q_H) -- r (width) before q (height).
        For ps=2:
        Channel 0 (c=0, r=0, q=0) -> (h=0, w=0)
        Channel 1 (c=0, r=0, q=1) -> (h=1, w=0)  -- q innermost -> H
        Channel 2 (c=0, r=1, q=0) -> (h=0, w=1)  -- r -> W
        Channel 3 (c=0, r=1, q=1) -> (h=1, w=1)
        """
        x = mx.zeros((1, 1, 1, 1, 4))
        x = x.at[0, 0, 0, 0, 0].add(10.0)  # c=0, r=0, q=0 -> h=0, w=0
        x = x.at[0, 0, 0, 0, 1].add(20.0)  # c=0, r=0, q=1 -> h=1, w=0
        x = x.at[0, 0, 0, 0, 2].add(30.0)  # c=0, r=1, q=0 -> h=0, w=1
        x = x.at[0, 0, 0, 0, 3].add(40.0)  # c=0, r=1, q=1 -> h=1, w=1

        out = unpatchify_spatial(x, patch_size=2)
        # Output: (1, 1, 2, 2, 1)
        assert out.shape == (1, 1, 2, 2, 1)

        out_np = np.array(out[0, 0, :, :, 0])
        expected = np.array([[10.0, 30.0], [20.0, 40.0]])
        np.testing.assert_allclose(out_np, expected, err_msg="unpatchify_spatial channel->position mapping is wrong")

    def test_differs_from_pixel_shuffle(self):
        """unpatchify_spatial should differ from pixel_shuffle_3d for non-symmetric data.

        They use different channel orderings:
        - pixel_shuffle_3d: (c, tf, sf_H, sf_W) -- H before W
        - unpatchify_spatial: (c, r_W, q_H) -- W before H

        For asymmetric channel values, the outputs must differ.
        """
        x = mx.zeros((1, 1, 1, 1, 4))
        x = x.at[0, 0, 0, 0, 1].add(1.0)  # This channel goes to different places

        out_shuffle = pixel_shuffle_3d(x, spatial_factor=2, temporal_factor=1)
        out_unpatch = unpatchify_spatial(x, patch_size=2)

        # They should produce different results because H/W mapping differs
        shuffle_np = np.array(out_shuffle[0, 0, :, :, 0])
        unpatch_np = np.array(out_unpatch[0, 0, :, :, 0])

        assert not np.allclose(shuffle_np, unpatch_np), (
            "unpatchify_spatial should differ from pixel_shuffle_3d "
            f"but got identical results: {shuffle_np} vs {unpatch_np}"
        )


class TestPatchifyRoundtrip:
    """Test that patchify_spatial and unpatchify_spatial are exact inverses."""

    def test_roundtrip(self):
        """Patchify -> unpatchify should recover the original."""
        original = mx.random.normal((1, 2, 16, 16, 3))  # BFHWC, RGB
        packed = patchify_spatial(original, patch_size=4)
        assert packed.shape == (1, 2, 4, 4, 48)

        recovered = unpatchify_spatial(packed, patch_size=4)
        assert recovered.shape == original.shape
        np.testing.assert_allclose(
            np.array(recovered),
            np.array(original),
            atol=1e-6,
            err_msg="patchify_spatial -> unpatchify_spatial roundtrip failed",
        )

    def test_roundtrip_patch2(self):
        """Roundtrip with patch_size=2."""
        original = mx.random.normal((2, 4, 8, 8, 3))
        packed = patchify_spatial(original, patch_size=2)
        recovered = unpatchify_spatial(packed, patch_size=2)
        np.testing.assert_allclose(
            np.array(recovered),
            np.array(original),
            atol=1e-6,
            err_msg="patchify_spatial -> unpatchify_spatial roundtrip failed for ps=2",
        )


class TestNoCheckerboard:
    """Verify that a smooth gradient input produces smooth output (no checkerboard)."""

    def test_gradient_smoothness(self):
        """A horizontal gradient should remain smooth after unpatchify_spatial.

        If H/W sub-pixels are swapped, a horizontal gradient will show
        alternating bright/dark columns (checkerboard).
        """
        ps = 4
        H_out, W_out = 8, 8
        _, _ = H_out // ps, W_out // ps  # 2, 2

        # Create a smooth horizontal gradient in pixel space
        gradient = mx.broadcast_to(
            mx.arange(W_out).reshape(1, 1, 1, W_out, 1).astype(mx.float32),
            (1, 1, H_out, W_out, 1),
        )

        # Pack it with patchify_spatial, then unpack with unpatchify_spatial
        packed = patchify_spatial(gradient, patch_size=ps)
        recovered = unpatchify_spatial(packed, patch_size=ps)

        # Check smoothness: adjacent pixels in W should differ by exactly 1
        recovered_np = np.array(recovered[0, 0, 0, :, 0])  # first row
        diffs = np.diff(recovered_np)
        np.testing.assert_allclose(
            diffs, 1.0, atol=1e-6, err_msg="Horizontal gradient shows non-smooth output (checkerboard artifact)"
        )

    def test_vertical_gradient_smoothness(self):
        """A vertical gradient should remain smooth after unpatchify_spatial."""
        ps = 4
        H_out, W_out = 8, 8

        gradient = mx.broadcast_to(
            mx.arange(H_out).reshape(1, 1, H_out, 1, 1).astype(mx.float32),
            (1, 1, H_out, W_out, 1),
        )

        packed = patchify_spatial(gradient, patch_size=ps)
        recovered = unpatchify_spatial(packed, patch_size=ps)

        # Check smoothness: adjacent pixels in H should differ by exactly 1
        recovered_np = np.array(recovered[0, 0, :, 0, 0])  # first column
        diffs = np.diff(recovered_np)
        np.testing.assert_allclose(
            diffs, 1.0, atol=1e-6, err_msg="Vertical gradient shows non-smooth output (checkerboard artifact)"
        )
