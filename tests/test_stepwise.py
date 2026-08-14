"""Tests for stepwise previews (utils/stepwise.py + the sampler on_step hook).

None of these need real weights: the sampler tests drive the loops with a stub
X0Model, and the preview tests drive StepwisePreview with a stub decoder that
reproduces the VAE's 8x temporal upsampling (N latent frames -> 8N-7 pixels).
"""

from pathlib import Path

import mlx.core as mx
import pytest
from PIL import Image

from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop
from ltx_pipelines_mlx.utils.stepwise import (
    DEFAULT_PREVIEW_FRAMES,
    StepwiseConfig,
    StepwisePreview,
    resolve_window,
    to_pil_images,
)

F, H, W = 4, 2, 3
CHANNELS = 128
NUM_TOKENS = F * H * W

# Two latent frames -> 9 pixel frames: enough to prove the window without slow tests.
WINDOW = 2
WINDOW_PIXEL_FRAMES = 8 * WINDOW - 7


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class StubX0Model:
    """Returns a constant x0 prediction, ignoring all inputs."""

    def __call__(self, *, video_latent, audio_latent, **kwargs):
        return mx.zeros_like(video_latent), mx.zeros_like(audio_latent)


class StubDecoder:
    """Stands in for the VAE decoder, including its 8x temporal upsampling.

    Frames within a chunk get distinct values: libwebp collapses identical
    consecutive frames into one, so a constant stub would make every chunk look
    like a single frame.
    """

    def __init__(self):
        self.calls: list[tuple[int, ...]] = []

    def decode(self, latent):
        self.calls.append(latent.shape)
        b, _c, f, h, w = latent.shape
        out_frames = max(1, 8 * f - 7)
        frames = [mx.full((b, 3, 1, h * 32, w * 32), -1.0 + 1.8 * i / max(1, out_frames)) for i in range(out_frames)]
        return mx.concatenate(frames, axis=2)


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
    overrides.setdefault("frames", WINDOW)
    return StepwisePreview(StepwiseConfig(output_dir=tmp_path, seed=42, **overrides), verbose=False)


def bind(preview, decoder_block, *, stage=None):
    return preview.bind(
        latent_frames=F,
        latent_height=H,
        latent_width=W,
        decoder_block=decoder_block,
        patchifier=VideoLatentPatchifier(),
        stage=stage,
    )


def x0(extra_tokens: int = 0):
    return mx.zeros((1, NUM_TOKENS + extra_tokens, CHANNELS))


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
# window resolution
# ---------------------------------------------------------------------------
class TestResolveWindow:
    def test_centres_on_the_middle_by_default(self):
        # 16 latent frames, 8 wide -> starts at 4, spanning the middle.
        assert resolve_window(16, None, 8) == (4, 8)

    def test_explicit_centre(self):
        assert resolve_window(16, 10, 4) == (8, 4)

    def test_negative_centre_counts_from_the_end(self):
        # Centred on the last frame, then clamped to keep the window inside the clip.
        assert resolve_window(16, -1, 4) == (12, 4)

    def test_clamps_to_the_start(self):
        assert resolve_window(16, 0, 6) == (0, 6)

    def test_clamps_to_the_end(self):
        assert resolve_window(16, 99, 6) == (10, 6)

    def test_window_wider_than_clip_is_truncated(self):
        assert resolve_window(4, None, 8) == (0, 4)

    def test_single_frame_clip(self):
        assert resolve_window(1, None, 8) == (0, 1)

    def test_count_is_at_least_one(self):
        assert resolve_window(16, None, 0) == (8, 1)

    def test_default_window_is_eight_latent_frames(self):
        assert DEFAULT_PREVIEW_FRAMES == 8
        assert StepwiseConfig(output_dir=Path(".")).frames == 8


# ---------------------------------------------------------------------------
# pixel conversion
# ---------------------------------------------------------------------------
class TestToPilImages:
    def test_one_image_per_temporal_index(self):
        images = to_pil_images(mx.zeros((1, 3, 5, 8, 5)))
        assert len(images) == 5
        assert images[0].mode == "RGB"
        assert images[0].size == (5, 8)  # PIL is (width, height)

    def test_value_mapping(self):
        # -1 -> 0, 0 -> 127, +1 -> 255
        assert to_pil_images(mx.full((1, 3, 1, 2, 2), -1.0))[0].getpixel((0, 0)) == (0, 0, 0)
        assert to_pil_images(mx.full((1, 3, 1, 2, 2), 1.0))[0].getpixel((0, 0)) == (255, 255, 255)
        assert to_pil_images(mx.zeros((1, 3, 1, 2, 2)))[0].getpixel((0, 0)) == (127, 127, 127)

    def test_clips_out_of_range(self):
        assert to_pil_images(mx.full((1, 3, 1, 2, 2), 5.0))[0].getpixel((0, 0)) == (255, 255, 255)
        assert to_pil_images(mx.full((1, 3, 1, 2, 2), -5.0))[0].getpixel((0, 0)) == (0, 0, 0)


# ---------------------------------------------------------------------------
# bind()
# ---------------------------------------------------------------------------
class TestBind:
    def test_returns_callable(self, tmp_path):
        assert callable(bind(make_preview(tmp_path), StubDecoderBlock()))

    def test_returns_none_once_disabled(self, tmp_path):
        preview = make_preview(tmp_path)
        preview._disable(RuntimeError("boom"))
        assert bind(preview, StubDecoderBlock()) is None


# ---------------------------------------------------------------------------
# writing chunks
# ---------------------------------------------------------------------------
class TestChunkOutput:
    def test_writes_one_animated_chunk_per_step(self, tmp_path):
        hook = bind(make_preview(tmp_path), StubDecoderBlock())
        hook(0, 4, x0(), 1.0)

        chunk = tmp_path / "seed_42_step001of004.webp"
        assert chunk.exists()
        with Image.open(chunk) as image:
            assert image.n_frames == WINDOW_PIXEL_FRAMES
            assert image.size == (W * 32, H * 32)

    def test_decodes_the_whole_window_in_one_call(self, tmp_path):
        decoder = StubDecoder()
        hook = bind(make_preview(tmp_path), StubDecoderBlock(decoder))
        hook(0, 4, x0(), 1.0)
        # One decode of (B, C, WINDOW, H, W) — not WINDOW separate decodes.
        assert decoder.calls == [(1, CHANNELS, WINDOW, H, W)]

    def test_window_is_truncated_to_the_clip(self, tmp_path):
        decoder = StubDecoder()
        hook = bind(make_preview(tmp_path, frames=64), StubDecoderBlock(decoder))
        assert decoder.calls == []
        hook(0, 4, x0(), 1.0)
        assert decoder.calls == [(1, CHANNELS, F, H, W)]

    def test_strips_appended_keyframe_tokens(self, tmp_path):
        decoder = StubDecoder()
        hook = bind(make_preview(tmp_path), StubDecoderBlock(decoder))
        # Multi-anchor conditioning appends tokens past the F*H*W grid.
        hook(0, 4, x0(extra_tokens=7), 1.0)
        assert decoder.calls == [(1, CHANNELS, WINDOW, H, W)]

    def test_chunks_are_never_rewritten(self, tmp_path):
        """The point of per-step chunks: write cost stays linear in the step count."""
        hook = bind(make_preview(tmp_path), StubDecoderBlock())
        hook(0, 3, x0(), 1.0)
        first = tmp_path / "seed_42_step001of003.webp"
        stamp = first.stat().st_mtime_ns

        hook(1, 3, x0(), 0.5)
        hook(2, 3, x0(), 0.0)

        assert first.stat().st_mtime_ns == stamp
        assert sorted(p.name for p in tmp_path.glob("*.webp")) == [
            "seed_42_step001of003.webp",
            "seed_42_step002of003.webp",
            "seed_42_step003of003.webp",
        ]

    def test_chunks_sort_into_playback_order(self, tmp_path):
        """Stage 1 before stage 2, and step 10 after step 9 — under a plain name sort."""
        hook1 = bind(make_preview(tmp_path), StubDecoderBlock(), stage=1)
        hook2 = bind(make_preview(tmp_path), StubDecoderBlock(), stage=2)
        for step in range(11):
            hook1(step, 11, x0(), 1.0)
        hook2(0, 2, x0(), 1.0)

        names = sorted(p.name for p in tmp_path.glob("*.webp"))
        assert names[0] == "seed_42_s1_step001of011.webp"
        assert names[9] == "seed_42_s1_step010of011.webp"
        assert names[-1] == "seed_42_s2_step001of002.webp"

    def test_stage_tag_separates_files(self, tmp_path):
        bind(make_preview(tmp_path), StubDecoderBlock(), stage=1)(0, 4, x0(), 1.0)
        bind(make_preview(tmp_path), StubDecoderBlock(), stage=2)(0, 4, x0(), 1.0)
        assert (tmp_path / "seed_42_s1_step001of004.webp").exists()
        assert (tmp_path / "seed_42_s2_step001of004.webp").exists()

    def test_no_temp_files_left_behind(self, tmp_path):
        hook = bind(make_preview(tmp_path), StubDecoderBlock())
        for step in range(3):
            hook(step, 3, x0(), 1.0)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_decoder_comes_from_the_block_cache_each_time(self, tmp_path):
        block = StubDecoderBlock()
        hook = bind(make_preview(tmp_path), block)
        for step in range(3):
            hook(step, 3, x0(), 1.0)
        assert block.load_count == 3

    def test_single_frame_window_still_works(self, tmp_path):
        hook = bind(make_preview(tmp_path, frames=1), StubDecoderBlock())
        hook(0, 2, x0(), 1.0)
        with Image.open(tmp_path / "seed_42_step001of002.webp") as image:
            assert image.n_frames == 1


# ---------------------------------------------------------------------------
# interval
# ---------------------------------------------------------------------------
class TestInterval:
    def _count_chunks(self, tmp_path, interval, num_steps):
        hook = bind(make_preview(tmp_path, interval=interval), StubDecoderBlock())
        for step in range(num_steps):
            hook(step, num_steps, x0(), 1.0)
        return len(list(tmp_path.glob("*.webp")))

    def test_every_step(self, tmp_path):
        assert self._count_chunks(tmp_path, interval=1, num_steps=9) == 9

    def test_every_third_step(self, tmp_path):
        # steps 0, 3, 6 by interval, plus step 8 because it is last.
        assert self._count_chunks(tmp_path, interval=3, num_steps=9) == 4

    def test_final_step_always_previewed(self, tmp_path):
        hook = bind(make_preview(tmp_path, interval=100), StubDecoderBlock())
        for step in range(5):
            hook(step, 5, x0(), 1.0)
        assert sorted(p.name for p in tmp_path.glob("*.webp")) == [
            "seed_42_step001of005.webp",
            "seed_42_step005of005.webp",
        ]


# ---------------------------------------------------------------------------
# failure isolation
# ---------------------------------------------------------------------------
class TestFailureIsolation:
    def test_decode_failure_does_not_raise(self, tmp_path):
        preview = make_preview(tmp_path)
        bind(preview, ExplodingDecoderBlock())(0, 4, x0(), 1.0)  # must not raise
        assert preview._disabled

    def test_disables_after_first_failure(self, tmp_path):
        block = ExplodingDecoderBlock()
        calls = []
        original_load = block.load

        def counting_load():
            calls.append(1)
            return original_load()

        block.load = counting_load
        hook = bind(make_preview(tmp_path), block)
        for step in range(5):
            hook(step, 5, x0(), 1.0)
        assert len(calls) == 1  # not retried on every subsequent step

    def test_generation_completes_despite_broken_preview(self, tmp_path):
        hook = bind(make_preview(tmp_path), ExplodingDecoderBlock())
        output = run_denoise_loop(hook)
        assert output.video_latent.shape == (1, NUM_TOKENS, CHANNELS)

    def test_keyboard_interrupt_still_propagates(self, tmp_path):
        class Interrupting:
            def load(self):
                raise KeyboardInterrupt

        hook = bind(make_preview(tmp_path), Interrupting())
        with pytest.raises(KeyboardInterrupt):
            hook(0, 4, x0(), 1.0)


# ---------------------------------------------------------------------------
# sampler hook integration
# ---------------------------------------------------------------------------
class TestSamplerHook:
    def test_denoise_loop_fires_once_per_step(self):
        seen = []
        run_denoise_loop(lambda idx, total, pred, sigma: seen.append((idx, total)), num_steps=4)
        assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]

    def test_denoise_loop_passes_x0_not_noisy_latent(self):
        # StubX0Model predicts zeros; the running latent starts at ones.
        seen = []
        run_denoise_loop(lambda idx, total, pred, sigma: seen.append(pred))
        assert all(mx.all(pred == 0).item() for pred in seen)

    def test_denoise_loop_passes_descending_sigma(self):
        seen = []
        run_denoise_loop(lambda idx, total, pred, sigma: seen.append(sigma), num_steps=4)
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
            on_step=lambda idx, total, pred, sigma: seen.append((idx, total)),
        )
        assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]


# ---------------------------------------------------------------------------
# CLI construction
# ---------------------------------------------------------------------------
def stepwise_args(tmp_path, **overrides):
    import argparse

    values = dict(
        stepwise_image_output_dir=str(tmp_path / "previews"),
        stepwise_interval=1,
        stepwise_frame=None,
        stepwise_frames=WINDOW,
        frame_rate=24.0,
        seed=42,
        quiet=False,
        low_ram=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class TestCLIMemoryWarning:
    """The memory warning must not be conditional on --low-ram.

    Previews hold the decoder and the transformer in memory at once, and the
    resulting jetsam kill is not an exception — the handler's failure
    containment cannot catch it, so this line is the only signal the user gets.
    """

    def test_warns_without_low_ram(self, tmp_path, capsys):
        from ltx_pipelines_mlx.cli import _build_stepwise

        assert _build_stepwise(stepwise_args(tmp_path)) is not None
        err = capsys.readouterr().err
        assert "VAE decoder resident" in err
        assert "--low-ram" not in err

    def test_low_ram_adds_a_second_line(self, tmp_path, capsys):
        from ltx_pipelines_mlx.cli import _build_stepwise

        _build_stepwise(stepwise_args(tmp_path, low_ram=True))
        err = capsys.readouterr().err
        assert "VAE decoder resident" in err
        assert "--low-ram" in err

    def test_silent_when_previews_are_off(self, tmp_path, capsys):
        from ltx_pipelines_mlx.cli import _build_stepwise

        assert _build_stepwise(stepwise_args(tmp_path, stepwise_image_output_dir=None)) is None
        assert capsys.readouterr().err == ""
