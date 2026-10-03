"""HDR IC-LoRA (upstream v1.4 single-stage ACEScct) — pipeline contracts without weights."""

from __future__ import annotations

import json
import subprocess

import mlx.core as mx
import numpy as np
import pytest

import ltx_pipelines_mlx.hdr_ic_lora as hdr_mod
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.keyframe_slots import VideoGeneratedKeyframeSlots
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.utils.ffmpeg import find_ffmpeg
from ltx_core_mlx.utils.positions import compute_video_positions
from ltx_pipelines_mlx.hdr_ic_lora import (
    DEFAULT_DENOISE_SIGMAS,
    DEFAULT_KEYFRAME_STRENGTH,
    HDRICLoraPipeline,
    conditioning_fps,
    dfr_seam_roles,
    load_video_context,
)
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.blocks import ImageConditioner, VideoDecoder
from ltx_pipelines_mlx.utils.hdr_media import VideoInput
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput


def _pack(tmp_path):
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {"num_layers": 48, "ff_bias": False}}))
    return tmp_path


def test_vae_blocks_default_to_bf16_and_accept_fp32(tmp_path):
    _pack(tmp_path)
    assert VideoDecoder(tmp_path, video_decoder="diffusion").dtype == mx.bfloat16
    assert VideoDecoder(tmp_path, video_decoder="diffusion", dtype=mx.float32).dtype == mx.float32
    assert ImageConditioner(tmp_path).dtype is None
    assert ImageConditioner(tmp_path, dtype=mx.float32).dtype == mx.float32


class _StubInner:
    def tiled_decode(self, video_latent, tiling, *, seed, keyframes):
        assert seed == 7 and keyframes is None
        yield mx.full((1, 3, 2, 4, 6), -1.0)
        yield mx.full((1, 3, 3, 4, 6), 1.0)


class _StubDiffusion:
    _decoder = _StubInner()

    def resolve_tiling(self, latent_shape, *, keyframe_planes=0):
        return None


def test_iter_frames_yields_float32_unit_range(tmp_path, monkeypatch):
    _pack(tmp_path)
    dec = VideoDecoder(tmp_path, video_decoder="diffusion")
    monkeypatch.setattr(VideoDecoder, "load", lambda self: _StubDiffusion())
    chunks = list(dec.iter_frames(mx.zeros((1, 128, 1, 1, 1)), seed=7, keyframes=None))
    assert [c.shape for c in chunks] == [(2, 4, 6, 3), (3, 4, 6, 3)]
    assert all(c.dtype == np.float32 for c in chunks)
    assert chunks[0].max() == 0.0 and chunks[1].min() == 1.0


def test_iter_frames_rejects_conv_decoder(tmp_path):
    _pack(tmp_path)
    dec = VideoDecoder(tmp_path)
    with pytest.raises(ValueError):
        next(dec.iter_frames(mx.zeros((1, 128, 1, 1, 1)), seed=0, keyframes=None))


# --- HDRICLoraPipeline ----------------------------------------------------------------------


def _pack25(tmp_path, *, ltx25=True):
    transformer = {"num_layers": 48}
    if ltx25:
        transformer |= {"ff_bias": False, "use_keyframes_abs_pos_embedding": True}
    (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": transformer}))
    return tmp_path


def _embeddings(tmp_path):
    emb = tmp_path / "e.safetensors"
    mx.save_safetensors(str(emb), {"video_context": mx.zeros((1, 4, 4096))})
    return emb


def _source(tmp_path, *, frames, size="64x64", rate=24):
    src = tmp_path / f"s_{frames}_{size}_{rate}.mp4"
    subprocess.run(
        [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=gray:s={size}:r={rate}",
         "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )  # fmt: skip
    return src


def test_conditioning_fps_snaps_to_30():
    assert conditioning_fps(24.0) == 24.0 and conditioning_fps(30.0) == 30.0 and conditioning_fps(60.0) == 30.0


def test_seam_roles_match_upstream():
    # resolve_canvas(97) ties 24/32 on zero padding and keeps the larger segment (upstream
    # ``choose_segment_length``), so seams sit every 32 frames; 73 frames pick 24.
    assert dfr_seam_roles(97, high_quality_hdr=False) == ([32, 64, 96], [32, 64, 96])
    assert dfr_seam_roles(97, high_quality_hdr=True) == ([64, 128, 192], [64, 128, 192])
    assert dfr_seam_roles(73, high_quality_hdr=False) == ([24, 48, 72], [24, 48, 72])
    assert dfr_seam_roles(1, high_quality_hdr=False) == ([], [])
    assert DEFAULT_KEYFRAME_STRENGTH == 0.95


def test_default_sigmas_are_the_distilled_table():
    assert list(DISTILLED_SIGMAS) == DEFAULT_DENOISE_SIGMAS


def test_missing_scene_embeddings_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Text embeddings"):
        load_video_context(tmp_path / "nope.safetensors")
    p = tmp_path / "e.safetensors"
    mx.save_safetensors(str(p), {"other": mx.zeros((1, 2))})
    with pytest.raises(KeyError, match="video_context"):
        load_video_context(p)


def test_video_prompt_embeds_key_is_accepted(tmp_path):
    p = tmp_path / "e.safetensors"
    mx.save_safetensors(str(p), {"video_prompt_embeds": mx.ones((1, 3, 4096))})
    assert load_video_context(p).shape == (1, 3, 4096)


def test_rejects_23_pack_before_any_load(tmp_path):
    emb = _embeddings(tmp_path)
    with pytest.raises(ValueError, match=r"LTX-2\.5"):
        HDRICLoraPipeline(str(_pack25(tmp_path, ltx25=False)), hdr_lora=str(emb), text_embeddings=str(emb))


def test_off_grid_source_raises_before_loading(tmp_path, monkeypatch):
    src = _source(tmp_path, frames=100)
    emb = _embeddings(tmp_path)
    pipe = HDRICLoraPipeline(str(_pack25(tmp_path)), hdr_lora=str(emb), text_embeddings=str(emb))
    monkeypatch.setattr(pipe, "load", lambda: (_ for _ in ()).throw(AssertionError("must not load")))
    with pytest.raises(ValueError, match=r"8k\+1.*97"):
        pipe.generate(VideoInput(src, gamma_encoded=True), seed=1)


class _FakeEncoder:
    def __init__(self):
        self.calls: list[tuple[int, ...]] = []

    def encode(self, pixels):
        self.calls.append(tuple(pixels.shape))
        assert pixels.dtype == mx.float32
        _, _, f, h, w = pixels.shape
        return mx.ones((1, 128, (f - 1) // 8 + 1, h // 32, w // 32), dtype=mx.float32)


class _FakeConditioner:
    def __init__(self):
        self.encoder = _FakeEncoder()

    def __call__(self, fn, *, free_after=True):
        return fn(self.encoder)

    def free(self):
        pass


class _FakeDecoderBlock:
    """Stands in for ``VideoDecoder``: frame ``i`` is filled with ``i / 1000`` so decimation is checkable."""

    def __init__(self, chunk_sizes, height, width, *, type_emb=1.0):
        self.chunk_sizes = chunk_sizes
        self.hw = (height, width)
        self.calls: list[dict] = []
        self._decoder = type("D", (), {"type_emb": mx.full((128,), type_emb)})()

    def load(self):
        return self

    def free(self):
        pass

    def iter_frames(self, video_latent, *, seed, keyframes=None):
        self.calls.append({"shape": tuple(video_latent.shape), "seed": seed, "keyframes": keyframes})
        start = 0
        for n in self.chunk_sizes:
            idx = np.arange(start, start + n, dtype=np.float32) / 1000.0
            yield np.broadcast_to(idx[:, None, None, None], (n, *self.hw, 3)).copy()
            start += n


def _wire(tmp_path, monkeypatch, *, frames, size, rate, chunk_sizes, type_emb=1.0):
    src = _source(tmp_path, frames=frames, size=size, rate=rate)
    emb = _embeddings(tmp_path)
    pipe = HDRICLoraPipeline(str(_pack25(tmp_path)), hdr_lora=str(emb), text_embeddings=str(emb))
    pipe.load = lambda: None  # type: ignore[method-assign]
    pipe.dit = object()  # type: ignore[assignment]
    pipe.image_conditioner = _FakeConditioner()  # type: ignore[assignment]
    w, h = (int(v) for v in size.split("x"))
    gen_h, gen_w = -(-h // 32) * 32, -(-w // 32) * 32
    pipe.video_decoder_block = _FakeDecoderBlock(chunk_sizes, gen_h, gen_w, type_emb=type_emb)  # type: ignore[assignment]
    monkeypatch.setattr(hdr_mod, "X0Model", lambda dit: dit)
    loops: list[dict] = []

    def _loop(model, video_state, audio_state=None, video_text_embeds=None, audio_text_embeds=None, sigmas=None, **kw):
        loops.append(
            dict(model=model, video_state=video_state, audio_state=audio_state, video_text_embeds=video_text_embeds,
                 audio_text_embeds=audio_text_embeds, sigmas=sigmas)
        )  # fmt: skip
        return DenoiseOutput(video_latent=video_state.latent, audio_latent=None)

    monkeypatch.setattr(hdr_mod, "denoise_loop", _loop)
    noised: list[dict] = []
    real = hdr_mod.create_noised_state

    def _spy(**kwargs):
        noised.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(hdr_mod, "create_noised_state", _spy)
    return pipe, src, loops, noised


def test_generate_wiring_matches_upstream(tmp_path, monkeypatch):
    pipe, src, loops, noised = _wire(tmp_path, monkeypatch, frames=73, size="48x40", rate=60, chunk_sizes=[40, 33])
    chunks, fps = pipe.generate(VideoInput(src, gamma_encoded=True), seed=3)
    assert fps == 60.0
    assert len(loops) == 1
    assert pipe.video_decoder_block.calls == [], "the decode must stay lazy until the chunks are consumed"
    out = list(chunks)

    assert len(loops) == 1
    loop = loops[0]
    assert loop["audio_state"] is None and loop["audio_text_embeds"] is None
    assert loop["sigmas"] == DEFAULT_DENOISE_SIGMAS
    assert loop["video_text_embeds"].dtype == mx.bfloat16 and loop["video_text_embeds"].shape == (1, 4, 4096)

    cfps = conditioning_fps(60.0)
    (call,) = noised
    conds = call["conditionings"]
    assert isinstance(conds[0], VideoConditionByReferenceLatent)
    assert conds[0].downscale_factor == 1 and conds[0].strength == 1.0
    assert conds[0].reference_latent.dtype == mx.bfloat16
    guides = conds[1:4]
    assert all(isinstance(g, VideoConditionByKeyframeIndex) for g in guides)
    assert [g.frame_idx for g in guides] == [24, 48, 72]
    assert all(g.strength == 0.95 for g in guides)
    assert all(g.keyframe_latent.shape == (1, 4, 128) and g.keyframe_latent.dtype == mx.bfloat16 for g in guides)
    assert isinstance(conds[4], VideoGeneratedKeyframeSlots) and len(conds) == 5
    assert list(conds[4].pixel_frame_indices) == [24, 48, 72]
    assert call["spatial_dims"] == (10, 2, 2)
    assert call["sigma"] == DEFAULT_DENOISE_SIGMAS[0]
    assert mx.array_equal(call["positions"], compute_video_positions(10, 2, 2, frame_rate=cfps))
    assert mx.array_equal(conds[0].reference_positions, compute_video_positions(10, 2, 2, frame_rate=cfps))

    # Encoder: the whole reference once, then one 1-frame encode per guide.
    assert pipe.image_conditioner.encoder.calls == [(1, 3, 73, 64, 64)] + [(1, 3, 1, 64, 64)] * 3

    (dec,) = pipe.video_decoder_block.calls
    assert dec["shape"] == (1, 128, 10, 2, 2) and dec["seed"] == 3
    assert dec["keyframes"] is not None and dec["keyframes"].pixel_frame_indices == (24, 48, 72)
    assert [c.shape for c in out] == [(40, 40, 48, 3), (33, 40, 48, 3)]


def test_high_quality_keeps_every_other_frame_across_chunks(tmp_path, monkeypatch):
    # 2 * 73 - 1 = 145 internal frames, split in odd-sized chunks so the parity flips between them.
    pipe, src, loops, noised = _wire(tmp_path, monkeypatch, frames=73, size="64x64", rate=24, chunk_sizes=[3, 5, 137])
    chunks, _ = pipe.generate(VideoInput(src, gamma_encoded=False), seed=0, high_quality_hdr=True)
    frames = np.concatenate(list(chunks), axis=0)
    assert frames.shape[0] == 73
    np.testing.assert_allclose(frames[:, 0, 0, 0], np.arange(0, 145, 2, dtype=np.float32) / 1000.0)
    conds = noised[0]["conditionings"]
    assert [c.frame_idx for c in conds[1:4]] == [48, 96, 144]
    assert pipe.image_conditioner.encoder.calls[0] == (1, 3, 145, 64, 64)
    assert noised[0]["spatial_dims"] == (19, 2, 2)


def test_no_keyframes_is_plain_ic_lora(tmp_path, monkeypatch):
    pipe, src, loops, noised = _wire(tmp_path, monkeypatch, frames=9, size="64x64", rate=24, chunk_sizes=[9])
    list(pipe.generate(VideoInput(src, gamma_encoded=True), seed=0, keyframe_strength=None)[0])
    conds = noised[0]["conditionings"]
    assert len(conds) == 1 and isinstance(conds[0], VideoConditionByReferenceLatent)
    assert pipe.video_decoder_block.calls[0]["keyframes"] is None


def test_zero_type_emb_refuses_keyframe_decode(tmp_path, monkeypatch):
    pipe, src, _, _ = _wire(tmp_path, monkeypatch, frames=25, size="64x64", rate=24, chunk_sizes=[25], type_emb=0.0)
    chunks, _ = pipe.generate(VideoInput(src, gamma_encoded=True), seed=0)
    with pytest.raises(RuntimeError, match="type_emb"):
        list(chunks)


def test_generate_and_save_feeds_the_hdr_encoder(tmp_path, monkeypatch):
    pipe, src, _, _ = _wire(tmp_path, monkeypatch, frames=9, size="64x64", rate=24, chunk_sizes=[9])
    seen: dict = {}

    def _encode(chunks, output_path, fps, color_space):
        seen.update(frames=sum(c.shape[0] for c in chunks), out=output_path, fps=fps, cs=color_space)
        return tmp_path / "x_exr"

    monkeypatch.setattr(hdr_mod, "encode_hdr_outputs", _encode)
    from ltx_pipelines_mlx.utils.hdr_media import EXRColorSpace

    result = pipe.generate_and_save(VideoInput(src, gamma_encoded=True), str(tmp_path / "o.mp4"), seed=0)
    assert result == tmp_path / "x_exr"
    assert seen == {"frames": 9, "out": str(tmp_path / "o.mp4"), "fps": 24, "cs": EXRColorSpace.ACESCG}
