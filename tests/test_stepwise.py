"""Tests for stepwise previews (utils/stepwise.py + the sampler on_step hook).

None of these need real weights: the sampler tests drive the loops with a stub
X0Model, and the preview tests drive StepwisePreview with a stub decoder.
"""

import mlx.core as mx
import pytest
from PIL import Image

from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop
from ltx_pipelines_mlx.utils.stepwise import (
    MAX_ANIMATION_FRAMES,
    StepwiseConfig,
    StepwisePreview,
    resolve_frame_index,
    to_pil_image,
)

F, H, W = 4, 2, 3
CHANNELS = 128
NUM_TOKENS = F * H * W


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class StubX0Model:
    """Returns a constant x0 prediction, ignoring all inputs."""

    def __call__(self, *, video_latent, audio_latent, **kwargs):
        return mx.zeros_like(video_latent), mx.zeros_like(audio_latent)


class StubDecoder:
    """Stands in for the VAE decoder: (B,C,1,h,w) -> (B,3,1,h*32,w*32).

    Each call returns a different fill value. libwebp collapses identical
    consecutive frames into one, so a constant stub would make every animation
    look like a single frame.
    """

    def __init__(self, fill: float = -1.0):
        self.fill = fill
        self.calls: list[tuple[int, ...]] = []

    def decode(self, latent):
        self.calls.append(latent.shape)
        b, _c, f, h, w = latent.shape
        # Cycle through distinct values so consecutive frames never match.
        value = self.fill + 1.8 * (len(self.calls) % 16) / 16
        return mx.full((b, 3, f, h * 32, w * 32), value)


class StubDecoderBlock:
    def __init__(self, decoder=None):
        self.decoder = decoder if decoder is not None else StubDecoder()
        self.load_count = 0

    def load(self):
        self.load_count += 1
        return self.decoder


class ExplodingDecoderBlock:
    def load(self):
        raise RuntimeError("decoder unavailable")


def make_state(num_tokens=NUM_TOKENS, channels=CHANNELS):
    return LatentState(
        latent=mx.ones((1, num_tokens, channels), dtype=mx.bfloat16),
        clean_latent=mx.zeros((1, num_tokens, channels), dtype=mx.bfloat16),
        denoise_mask=mx.ones((1, num_tokens, 1), dtype=mx.bfloat16),
    )


def make_preview(tmp_path, **overrides):
    config = StepwiseConfig(output_dir=tmp_path, seed=42, **overrides)
    return StepwisePreview(config, verbose=False)


def run_denoise_loop(on_step, num_steps=4):
    sigmas = [1.0 - i / num_steps for i in range(num_steps + 1)]
    return denoise_loop(
        model=StubX0Model(),
        video_state=make_state(),
        audio_state=make_state(num_tokens=8),
        video_text_embeds=mx.zeros((1, 4, 16)),
        audio_text_embeds=mx.zeros((1, 4, 16)),
        sigmas=sigmas,
        show_progress=False,
        on_step=on_step,
    )


# ---------------------------------------------------------------------------
# frame index resolution
# ---------------------------------------------------------------------------
class TestResolveFrameIndex:
    def test_defaults_to_middle(self):
        assert resolve_frame_index(13, None) == 6
        assert resolve_frame_index(4, None) == 2

    def test_negative_counts_from_end(self):
        assert resolve_frame_index(13, -1) == 12
        assert resolve_frame_index(13, -3) == 10

    def test_explicit_index_passes_through(self):
        assert resolve_frame_index(13, 0) == 0
        assert resolve_frame_index(13, 7) == 7

    def test_clamps_out_of_range(self):
        assert resolve_frame_index(13, 99) == 12
        assert resolve_frame_index(13, -99) == 0

    def test_single_frame(self):
        assert resolve_frame_index(1, None) == 0
        assert resolve_frame_index(1, -1) == 0


# ---------------------------------------------------------------------------
# pixel conversion
# ---------------------------------------------------------------------------
class TestToPilImage:
    def test_shape_and_mode(self):
        pixels = mx.zeros((1, 3, 1, 8, 5))
        image = to_pil_image(pixels)
        assert image.mode == "RGB"
        assert image.size == (5, 8)  # PIL is (width, height)

    def test_value_mapping(self):
        # -1 -> 0, 0 -> 127, +1 -> 255
        assert to_pil_image(mx.full((1, 3, 1, 2, 2), -1.0)).getpixel((0, 0)) == (0, 0, 0)
        assert to_pil_image(mx.full((1, 3, 1, 2, 2), 1.0)).getpixel((0, 0)) == (255, 255, 255)
        assert to_pil_image(mx.zeros((1, 3, 1, 2, 2))).getpixel((0, 0)) == (127, 127, 127)

    def test_clips_out_of_range(self):
        assert to_pil_image(mx.full((1, 3, 1, 2, 2), 5.0)).getpixel((0, 0)) == (255, 255, 255)
        assert to_pil_image(mx.full((1, 3, 1, 2, 2), -5.0)).getpixel((0, 0)) == (0, 0, 0)


# ---------------------------------------------------------------------------
# bind()
# ---------------------------------------------------------------------------
class TestBind:
    def test_returns_callable(self, tmp_path):
        hook = make_preview(tmp_path).bind(
            latent_frames=F,
            latent_height=H,
            latent_width=W,
            decoder_block=StubDecoderBlock(),
            patchifier=VideoLatentPatchifier(),
        )
        assert callable(hook)

    def test_returns_none_once_disabled(self, tmp_path):
        preview = make_preview(tmp_path)
        preview._disable(RuntimeError("boom"))
        hook = preview.bind(
            latent_frames=F,
            latent_height=H,
            latent_width=W,
            decoder_block=StubDecoderBlock(),
            patchifier=VideoLatentPatchifier(),
        )
        assert hook is None


# ---------------------------------------------------------------------------
# writing previews
# ---------------------------------------------------------------------------
class TestPreviewOutput:
    def _hook(self, tmp_path, decoder_block, *, stage=None, **overrides):
        preview = make_preview(tmp_path, **overrides)
        return preview, preview.bind(
            latent_frames=F,
            latent_height=H,
            latent_width=W,
            decoder_block=decoder_block,
            patchifier=VideoLatentPatchifier(),
            stage=stage,
        )

    def test_writes_png_and_animation(self, tmp_path):
        block = StubDecoderBlock()
        _, hook = self._hook(tmp_path, block)
        hook(0, 4, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)

        png = tmp_path / "seed_42_step001of004.png"
        assert png.exists()
        with Image.open(png) as image:
            assert image.size == (W * 32, H * 32)
        assert (tmp_path / "seed_42_progress.webp").exists()

    def test_decodes_exactly_one_latent_frame(self, tmp_path):
        decoder = StubDecoder()
        _, hook = self._hook(tmp_path, StubDecoderBlock(decoder))
        hook(0, 4, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)
        # (B, C, F=1, H, W) -- the whole point is not decoding all F frames.
        assert decoder.calls == [(1, CHANNELS, 1, H, W)]

    def test_strips_appended_keyframe_tokens(self, tmp_path):
        decoder = StubDecoder()
        _, hook = self._hook(tmp_path, StubDecoderBlock(decoder))
        # Multi-anchor conditioning appends tokens past the F*H*W grid.
        hook(0, 4, mx.zeros((1, NUM_TOKENS + 7, CHANNELS)), 1.0)
        assert decoder.calls == [(1, CHANNELS, 1, H, W)]

    def test_stage_tag_separates_files(self, tmp_path):
        _, hook1 = self._hook(tmp_path, StubDecoderBlock(), stage=1)
        _, hook2 = self._hook(tmp_path, StubDecoderBlock(), stage=2)
        hook1(0, 4, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)
        hook2(0, 4, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)

        assert (tmp_path / "seed_42_s1_step001of004.png").exists()
        assert (tmp_path / "seed_42_s2_step001of004.png").exists()
        assert (tmp_path / "seed_42_s1_progress.webp").exists()
        assert (tmp_path / "seed_42_s2_progress.webp").exists()

    def test_animation_gains_a_frame_per_preview(self, tmp_path):
        _, hook = self._hook(tmp_path, StubDecoderBlock())
        animation = tmp_path / "seed_42_progress.webp"

        for step in range(3):
            hook(step, 3, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)
            with Image.open(animation) as image:
                assert image.n_frames == step + 1

    def test_no_temp_files_left_behind(self, tmp_path):
        _, hook = self._hook(tmp_path, StubDecoderBlock())
        for step in range(3):
            hook(step, 3, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_decoder_loaded_once_across_previews(self, tmp_path):
        block = StubDecoderBlock()
        _, hook = self._hook(tmp_path, block)
        for step in range(3):
            hook(step, 3, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)
        # load() is cached in the real block; we must not free/reload per preview.
        assert block.load_count == 3

    def test_frame_list_is_capped(self, tmp_path):
        _, hook = self._hook(tmp_path, StubDecoderBlock())
        total = MAX_ANIMATION_FRAMES + 5
        for step in range(total):
            hook(step, total, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)

        with Image.open(tmp_path / "seed_42_progress.webp") as image:
            # Halved once the cap is passed, so bounded but still accumulating.
            assert 1 < image.n_frames <= MAX_ANIMATION_FRAMES
        assert len(list(tmp_path.glob("*.png"))) == total  # every step still written


# ---------------------------------------------------------------------------
# interval
# ---------------------------------------------------------------------------
class TestInterval:
    def _count_pngs(self, tmp_path, interval, num_steps):
        preview = make_preview(tmp_path, interval=interval)
        hook = preview.bind(
            latent_frames=F,
            latent_height=H,
            latent_width=W,
            decoder_block=StubDecoderBlock(),
            patchifier=VideoLatentPatchifier(),
        )
        for step in range(num_steps):
            hook(step, num_steps, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)
        return len(list(tmp_path.glob("*.png")))

    def test_every_step(self, tmp_path):
        assert self._count_pngs(tmp_path, interval=1, num_steps=9) == 9

    def test_every_third_step(self, tmp_path):
        # steps 0, 3, 6 by interval, plus step 8 because it is last.
        assert self._count_pngs(tmp_path, interval=3, num_steps=9) == 4

    def test_final_step_always_previewed(self, tmp_path):
        preview = make_preview(tmp_path, interval=100)
        hook = preview.bind(
            latent_frames=F,
            latent_height=H,
            latent_width=W,
            decoder_block=StubDecoderBlock(),
            patchifier=VideoLatentPatchifier(),
        )
        for step in range(5):
            hook(step, 5, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)
        # step 0 (interval) and step 4 (last)
        assert sorted(p.name for p in tmp_path.glob("*.png")) == [
            "seed_42_step001of005.png",
            "seed_42_step005of005.png",
        ]


# ---------------------------------------------------------------------------
# failure isolation
# ---------------------------------------------------------------------------
class TestFailureIsolation:
    def test_decode_failure_does_not_raise(self, tmp_path):
        preview = make_preview(tmp_path)
        hook = preview.bind(
            latent_frames=F,
            latent_height=H,
            latent_width=W,
            decoder_block=ExplodingDecoderBlock(),
            patchifier=VideoLatentPatchifier(),
        )
        hook(0, 4, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)  # must not raise
        assert preview._disabled

    def test_disables_after_first_failure(self, tmp_path):
        preview = make_preview(tmp_path)
        block = ExplodingDecoderBlock()
        calls = []
        original_load = block.load

        def counting_load():
            calls.append(1)
            return original_load()

        block.load = counting_load
        hook = preview.bind(
            latent_frames=F, latent_height=H, latent_width=W, decoder_block=block, patchifier=VideoLatentPatchifier()
        )
        for step in range(5):
            hook(step, 5, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)
        assert len(calls) == 1  # not retried on every subsequent step

    def test_generation_completes_despite_broken_preview(self, tmp_path):
        preview = make_preview(tmp_path)
        hook = preview.bind(
            latent_frames=F,
            latent_height=H,
            latent_width=W,
            decoder_block=ExplodingDecoderBlock(),
            patchifier=VideoLatentPatchifier(),
        )
        output = run_denoise_loop(hook)
        assert output.video_latent.shape == (1, NUM_TOKENS, CHANNELS)

    def test_keyboard_interrupt_still_propagates(self, tmp_path):
        class Interrupting:
            def load(self):
                raise KeyboardInterrupt

        preview = make_preview(tmp_path)
        hook = preview.bind(
            latent_frames=F,
            latent_height=H,
            latent_width=W,
            decoder_block=Interrupting(),
            patchifier=VideoLatentPatchifier(),
        )
        with pytest.raises(KeyboardInterrupt):
            hook(0, 4, mx.zeros((1, NUM_TOKENS, CHANNELS)), 1.0)


# ---------------------------------------------------------------------------
# sampler hook integration
# ---------------------------------------------------------------------------
class TestSamplerHook:
    def test_denoise_loop_fires_once_per_step(self):
        seen = []
        run_denoise_loop(lambda idx, total, x0, sigma: seen.append((idx, total)), num_steps=4)
        assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]

    def test_denoise_loop_passes_x0_not_noisy_latent(self):
        # StubX0Model predicts zeros; the running latent starts at ones.
        seen = []
        run_denoise_loop(lambda idx, total, x0, sigma: seen.append(x0))
        assert all(mx.all(x0 == 0).item() for x0 in seen)

    def test_denoise_loop_passes_descending_sigma(self):
        seen = []
        run_denoise_loop(lambda idx, total, x0, sigma: seen.append(sigma), num_steps=4)
        assert seen == sorted(seen, reverse=True)

    def test_denoise_loop_without_hook_is_unchanged(self):
        without = run_denoise_loop(None)
        with_hook = run_denoise_loop(lambda *a: None)
        assert mx.allclose(without.video_latent, with_hook.video_latent).item()

    def test_guided_loop_fires_once_per_step(self):
        from ltx_core_mlx.components.guiders import (
            MultiModalGuiderParams,
            create_multimodal_guider_factory,
        )

        seen = []
        factory = create_multimodal_guider_factory(MultiModalGuiderParams(), negative_context=None)
        guided_denoise_loop(
            model=StubX0Model(),
            video_state=make_state(),
            audio_state=make_state(num_tokens=8),
            video_text_embeds=mx.zeros((1, 4, 16)),
            audio_text_embeds=mx.zeros((1, 4, 16)),
            video_guider_factory=factory,
            sigmas=[1.0, 0.75, 0.5, 0.25, 0.0],
            show_progress=False,
            on_step=lambda idx, total, x0, sigma: seen.append((idx, total)),
        )
        assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]
