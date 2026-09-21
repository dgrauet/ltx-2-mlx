import mlx.core as mx

from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.utils.positions import compute_video_positions
from ltx_pipelines_mlx.iclora_utils import reference_conditioning_from_latent


def test_reference_conditioning_from_latent_tokens_positions_and_scale():
    latent = mx.random.normal((1, 128, 3, 2, 4))
    cond = reference_conditioning_from_latent(latent, frame_rate=24.0, downscale_factor=2)
    assert isinstance(cond, VideoConditionByReferenceLatent)
    assert cond.reference_latent.shape == (1, 3 * 2 * 4, 128)
    assert mx.array_equal(cond.reference_latent[0, 5], latent[0, :, 0, 1, 1])  # token 5 = (f0, h1, w1)
    assert mx.array_equal(cond.reference_positions, compute_video_positions(3, 2, 4, frame_rate=24.0))
    assert cond.downscale_factor == 2 and cond.strength == 1.0
