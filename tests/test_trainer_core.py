"""Unit tests for trainer strategy factory, LoRA checkpoint format,
model loader auto-detect, preprocessing helpers, LoRA target discovery,
and config validation.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors

# ---------------------------------------------------------------------------
# 1. TestStrategyFactory
# ---------------------------------------------------------------------------


class TestStrategyFactory:
    """Tests for get_training_strategy factory function."""

    def test_pydantic_to_t2v(self) -> None:
        """TrainingStrategyConfig(name='text_to_video') creates TextToVideoStrategy."""
        from ltx_trainer_mlx.config import TrainingStrategyConfig
        from ltx_trainer_mlx.training_strategies import TextToVideoStrategy, get_training_strategy

        config = TrainingStrategyConfig(name="text_to_video")
        strategy = get_training_strategy(config)
        assert isinstance(strategy, TextToVideoStrategy)

    def test_pydantic_to_v2v(self) -> None:
        """TrainingStrategyConfig(name='video_to_video') creates VideoToVideoStrategy."""
        from ltx_trainer_mlx.config import TrainingStrategyConfig
        from ltx_trainer_mlx.training_strategies import VideoToVideoStrategy, get_training_strategy

        config = TrainingStrategyConfig(name="video_to_video")
        strategy = get_training_strategy(config)
        assert isinstance(strategy, VideoToVideoStrategy)

    def test_audio_passthrough(self) -> None:
        """generate_audio=True propagates to strategy.requires_audio=True."""
        from ltx_trainer_mlx.config import TrainingStrategyConfig
        from ltx_trainer_mlx.training_strategies import get_training_strategy

        config = TrainingStrategyConfig(name="text_to_video", generate_audio=True)
        strategy = get_training_strategy(config)
        assert strategy.requires_audio is True

    def test_audio_disabled_by_default(self) -> None:
        """generate_audio=False disables audio in strategy."""
        from ltx_trainer_mlx.config import TrainingStrategyConfig
        from ltx_trainer_mlx.training_strategies import get_training_strategy

        config = TrainingStrategyConfig(name="text_to_video", generate_audio=False)
        strategy = get_training_strategy(config)
        assert strategy.requires_audio is False

    def test_unknown_name_raises(self) -> None:
        """ValueError for invalid strategy name."""
        from ltx_trainer_mlx.training_strategies import get_training_strategy

        class _Fake:
            name = "banana"

        with pytest.raises(ValueError, match="Unknown training strategy"):
            get_training_strategy(_Fake())

    def test_native_config_passthrough(self) -> None:
        """TextToVideoConfig instance passes through without conversion."""
        from ltx_trainer_mlx.training_strategies import (
            TextToVideoConfig,
            TextToVideoStrategy,
            get_training_strategy,
        )

        native_config = TextToVideoConfig(with_audio=True)
        strategy = get_training_strategy(native_config)
        assert isinstance(strategy, TextToVideoStrategy)
        assert strategy.requires_audio is True


# ---------------------------------------------------------------------------
# 2. TestLoraCheckpointFormat
# ---------------------------------------------------------------------------


class TestLoraCheckpointFormat:
    """Tests that the LoRA checkpoint save logic produces the correct format.

    The fixture-based tests simulate the save logic from trainer.py
    _save_checkpoint by creating a small nn.Module with LoRALinear layers and
    running the same conversion. test_save_checkpoint_round_trip goes through
    the real LtxvTrainer._setup_lora + _save_checkpoint path.
    """

    @pytest.fixture()
    def lora_checkpoint(self, tmp_path: Path) -> tuple[Path, int, int, int]:
        """Create a small model with LoRA layers and save a checkpoint."""
        from mlx_lm.tuner.lora import LoRALinear

        in_dim, out_dim, rank = 64, 32, 8

        # Build a minimal model with LoRA layers
        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = [type("Block", (), {"attn": type("Attn", (), {})()})()]
                block = self.transformer_blocks[0]
                block.attn.to_q = LoRALinear(in_dim, out_dim, r=rank)
                block.attn.to_k = LoRALinear(in_dim, out_dim, r=rank)

        model = TinyModel()
        mx.metal.clear_cache()
        mx.synchronize()

        # Reproduce _save_checkpoint LoRA logic
        state_dict: dict[str, np.ndarray] = {}
        for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
            key = f"diffusion_model.{name}"
            if key.endswith(".lora_a"):
                key = key[: -len(".lora_a")] + ".lora_A.weight"
                param = mx.transpose(param)
            elif key.endswith(".lora_b"):
                key = key[: -len(".lora_b")] + ".lora_B.weight"
                param = mx.transpose(param)
            state_dict[key] = np.array(param.astype(mx.float32))

        out_path = tmp_path / "lora_weights.safetensors"
        save_safetensors(state_dict, str(out_path))
        return out_path, in_dim, out_dim, rank

    def test_key_naming(self, lora_checkpoint: tuple[Path, int, int, int]) -> None:
        """Keys use .lora_A.weight / .lora_B.weight format."""
        path, *_ = lora_checkpoint
        tensors = load_safetensors(str(path))
        for key in tensors:
            assert ".lora_A.weight" in key or ".lora_B.weight" in key, f"Unexpected key format: {key}"
            assert ".lora_a" not in key and ".lora_b" not in key

    def test_key_prefix(self, lora_checkpoint: tuple[Path, int, int, int]) -> None:
        """All keys start with 'diffusion_model.'."""
        path, *_ = lora_checkpoint
        tensors = load_safetensors(str(path))
        for key in tensors:
            assert key.startswith("diffusion_model."), f"Key missing prefix: {key}"

    def test_transpose_shapes(self, lora_checkpoint: tuple[Path, int, int, int]) -> None:
        """lora_A.weight is (rank, in_dim), lora_B.weight is (out_dim, rank)."""
        path, in_dim, out_dim, rank = lora_checkpoint
        tensors = load_safetensors(str(path))
        for key, tensor in tensors.items():
            if ".lora_A.weight" in key:
                assert tensor.shape == (rank, in_dim), f"lora_A shape {tensor.shape} != ({rank}, {in_dim})"
            elif ".lora_B.weight" in key:
                assert tensor.shape == (out_dim, rank), f"lora_B shape {tensor.shape} != ({out_dim}, {rank})"

    def test_dtype(self, lora_checkpoint: tuple[Path, int, int, int]) -> None:
        """All saved tensors are float32."""
        path, *_ = lora_checkpoint
        tensors = load_safetensors(str(path))
        for key, tensor in tensors.items():
            assert tensor.dtype == np.float32, f"{key} has dtype {tensor.dtype}"

    def test_save_checkpoint_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real _save_checkpoint writes C-contiguous factors that reload exactly.

        safetensors >= 0.8.0 serializes a numpy array's raw buffer and ignores
        its strides, so a transposed (non-C-contiguous) factor is saved
        scrambled. The lock pins 0.7.0, which still round-trips such arrays
        correctly, so the value check alone passes without the fix: the
        contiguity check on what reaches save_file is what catches it.
        """
        from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
        from ltx_trainer_mlx import trainer as trainer_module
        from ltx_trainer_mlx.config import LtxTrainerConfig
        from ltx_trainer_mlx.trainer import LtxvTrainer

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "transformer.safetensors").touch()
        config = LtxTrainerConfig(
            model={"model_path": str(model_dir)},
            lora={"rank": 4, "alpha": 4},
            data={"preprocessed_data_root": str(tmp_path / "data")},
            output_dir=str(tmp_path / "out"),
        )

        # Tiny DiT, no weights: its to_q/to_k/to_v layers are 16x16, 32x32 and 16x32.
        mx.random.seed(0)
        model_config = LTXModelConfig(
            num_layers=1,
            video_dim=32,
            audio_dim=16,
            video_num_heads=4,
            audio_num_heads=4,
            video_head_dim=8,
            audio_head_dim=4,
            av_cross_num_heads=4,
            av_cross_head_dim=4,
            video_patch_channels=8,
            audio_patch_channels=8,
            ff_mult=2.0,
            timestep_embedding_dim=32,
        )

        # Bypass __init__, which loads the text encoder and model weights.
        trainer = LtxvTrainer.__new__(LtxvTrainer)
        trainer._config = config
        trainer._transformer = LTXModel(model_config)
        trainer._global_step = 1
        trainer._checkpoint_paths = []
        trainer._setup_lora()

        # LoRALinear zero-initializes lora_b; randomize both factors so a
        # scrambled save cannot compare equal by accident.
        trained = [
            (k, mx.random.normal(v.shape))
            for k, v in nn.utils.tree_flatten(trainer._transformer.trainable_parameters())
        ]
        trainer._transformer.load_weights(trained, strict=False)
        in_memory = {k: np.array(v) for k, v in nn.utils.tree_flatten(trainer._transformer.trainable_parameters())}
        assert in_memory, "_setup_lora attached no LoRA layers"

        contiguous: dict[str, bool] = {}
        real_save = trainer_module.save_safetensors

        def recording_save(tensors: dict[str, np.ndarray], filename: str) -> None:
            contiguous.update({k: bool(v.flags["C_CONTIGUOUS"]) for k, v in tensors.items()})
            real_save(tensors, filename)

        monkeypatch.setattr(trainer_module, "save_safetensors", recording_save)

        saved = load_safetensors(str(trainer._save_checkpoint()))

        assert set(contiguous) == set(saved), "_save_checkpoint did not save through save_safetensors"
        not_contiguous = sorted(k for k, ok in contiguous.items() if not ok)
        assert not not_contiguous, f"non-C-contiguous arrays passed to save_file: {not_contiguous}"

        expected: dict[str, np.ndarray] = {}
        for name, value in in_memory.items():
            if name.endswith(".lora_a"):
                expected[f"diffusion_model.{name[: -len('.lora_a')]}.lora_A.weight"] = value.T
            else:
                assert name.endswith(".lora_b"), f"unexpected trainable parameter: {name}"
                expected[f"diffusion_model.{name[: -len('.lora_b')]}.lora_B.weight"] = value.T
        assert set(saved) == set(expected)
        for key, value in expected.items():
            np.testing.assert_array_equal(saved[key], value, err_msg=key)


# ---------------------------------------------------------------------------
# 3. TestModelLoaderAutoDetect
# ---------------------------------------------------------------------------


class TestModelLoaderAutoDetect:
    """Tests for the transformer file auto-detection logic in load_transformer."""

    def _auto_detect(self, model_dir: Path) -> Path:
        """Run the same auto-detect logic as load_transformer without loading weights."""
        for stem in ["transformer", "transformer-distilled", "transformer-dev"]:
            exact = model_dir / f"{stem}.safetensors"
            if exact.exists():
                return exact
            if stem != "transformer":
                versioned = sorted(model_dir.glob(f"{stem}*.safetensors"))
                if versioned:
                    return versioned[-1]
        raise FileNotFoundError(f"No transformer safetensors found in {model_dir}")

    def test_finds_transformer(self, tmp_path: Path) -> None:
        """Directory with transformer.safetensors finds it."""
        (tmp_path / "transformer.safetensors").touch()
        result = self._auto_detect(tmp_path)
        assert result.name == "transformer.safetensors"

    def test_finds_distilled(self, tmp_path: Path) -> None:
        """Directory with only transformer-distilled.safetensors finds it."""
        (tmp_path / "transformer-distilled.safetensors").touch()
        result = self._auto_detect(tmp_path)
        assert result.name == "transformer-distilled.safetensors"

    def test_finds_dev(self, tmp_path: Path) -> None:
        """Directory with only transformer-dev.safetensors finds it."""
        (tmp_path / "transformer-dev.safetensors").touch()
        result = self._auto_detect(tmp_path)
        assert result.name == "transformer-dev.safetensors"

    def test_prefers_transformer(self, tmp_path: Path) -> None:
        """Prefers transformer.safetensors over transformer-distilled.safetensors."""
        (tmp_path / "transformer.safetensors").touch()
        (tmp_path / "transformer-distilled.safetensors").touch()
        result = self._auto_detect(tmp_path)
        assert result.name == "transformer.safetensors"

    def test_no_file_raises(self, tmp_path: Path) -> None:
        """Empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No transformer safetensors"):
            self._auto_detect(tmp_path)


# ---------------------------------------------------------------------------
# 4. TestPreprocessHelpers
# ---------------------------------------------------------------------------


class TestPreprocessHelpers:
    """Tests for preprocessing utility functions."""

    def test_resolve_captions_from_dir(self, tmp_path: Path) -> None:
        """Matching .txt caption files are read correctly."""
        from ltx_trainer_mlx.preprocess import _resolve_captions

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        captions_dir = tmp_path / "captions"
        captions_dir.mkdir()

        video_files = []
        for name, caption in [("clip_one", "A cat running"), ("clip_two", "A dog sleeping")]:
            vf = videos_dir / f"{name}.mp4"
            vf.touch()
            video_files.append(vf)
            (captions_dir / f"{name}.txt").write_text(caption)

        result = _resolve_captions(video_files, str(captions_dir), ".txt")
        assert result == ["A cat running", "A dog sleeping"]

    def test_resolve_captions_no_dir(self, tmp_path: Path) -> None:
        """No captions_dir uses video stem with underscore/hyphen replacement."""
        from ltx_trainer_mlx.preprocess import _resolve_captions

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        video_files = [videos_dir / "my_cool-video.mp4"]
        video_files[0].touch()

        result = _resolve_captions(video_files, None, ".txt")
        assert result == ["my cool video"]

    def test_resolve_captions_missing_file(self, tmp_path: Path) -> None:
        """Missing .txt files fall back to filename for those entries."""
        from ltx_trainer_mlx.preprocess import _resolve_captions

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        captions_dir = tmp_path / "captions"
        captions_dir.mkdir()

        v1 = videos_dir / "has_caption.mp4"
        v1.touch()
        (captions_dir / "has_caption.txt").write_text("Real caption")

        v2 = videos_dir / "no-caption.mp4"
        v2.touch()
        # Intentionally no .txt for v2

        result = _resolve_captions([v1, v2], str(captions_dir), ".txt")
        assert result[0] == "Real caption"
        assert result[1] == "no caption"  # hyphen -> space

    def test_resize_video(self) -> None:
        """Resize produces correct output shape."""
        from ltx_trainer_mlx.preprocess import _resize_video

        # (F, C, H, W) in [0, 1]
        video = mx.random.uniform(shape=(4, 3, 100, 200))
        resized = _resize_video(video, target_h=50, target_w=100)
        assert resized.shape == (4, 3, 50, 100)

    def test_resize_video_preserves_range(self) -> None:
        """Resized values stay in [0, 1]."""
        from ltx_trainer_mlx.preprocess import _resize_video

        video = mx.random.uniform(shape=(2, 3, 64, 64))
        resized = _resize_video(video, target_h=32, target_w=32)
        resized_np = np.array(resized)
        assert resized_np.min() >= 0.0
        assert resized_np.max() <= 1.0


# ---------------------------------------------------------------------------
# 5. TestFindLoraTargets
# ---------------------------------------------------------------------------


class TestFindLoraTargets:
    """Tests for _find_lora_targets helper."""

    @pytest.fixture()
    def model_with_layers(self) -> nn.Module:
        """Create a small nn.Module hierarchy with Linear and QuantizedLinear."""

        class SubBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.to_q = nn.Linear(32, 32)
                self.to_k = nn.Linear(32, 32)
                self.to_v = nn.Linear(32, 32)
                self.ff = nn.Linear(32, 32)

        class TinyTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = SubBlock()

        model = TinyTransformer()
        mx.metal.clear_cache()
        mx.synchronize()
        return model

    def test_finds_linear(self, model_with_layers: nn.Module) -> None:
        """Matches nn.Linear by target name."""
        from ltx_trainer_mlx.trainer import _find_lora_targets

        results = _find_lora_targets(model_with_layers, ["to_q", "to_k"])
        paths = [path for path, _ in results]
        assert "attn.to_q" in paths
        assert "attn.to_k" in paths
        assert "attn.ff" not in paths

    def test_finds_quantized(self) -> None:
        """Matches nn.QuantizedLinear by target name."""
        from ltx_trainer_mlx.trainer import _find_lora_targets

        class QModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.to_q = nn.QuantizedLinear(32, 32, group_size=32)
                self.proj = nn.Linear(32, 32)

        model = QModel()
        mx.metal.clear_cache()
        mx.synchronize()

        results = _find_lora_targets(model, ["to_q"])
        assert len(results) == 1
        path, module = results[0]
        assert path == "to_q"
        assert isinstance(module, nn.QuantizedLinear)

    def test_no_match(self, model_with_layers: nn.Module) -> None:
        """Non-matching names return empty list."""
        from ltx_trainer_mlx.trainer import _find_lora_targets

        results = _find_lora_targets(model_with_layers, ["nonexistent_layer"])
        assert results == []


# ---------------------------------------------------------------------------
# 6. TestConfig
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for LtxTrainerConfig validation."""

    @pytest.fixture()
    def model_dir(self, tmp_path: Path) -> Path:
        """Create a fake model directory that passes validation."""
        d = tmp_path / "model"
        d.mkdir()
        (d / "transformer.safetensors").touch()
        return d

    def test_valid_minimal_config(self, model_dir: Path) -> None:
        """LtxTrainerConfig with required fields parses OK."""
        from ltx_trainer_mlx.config import LtxTrainerConfig

        config = LtxTrainerConfig(
            model={"model_path": str(model_dir)},
            lora={"rank": 16, "alpha": 16},
            data={"preprocessed_data_root": "/tmp/data"},
        )
        assert config.model.training_mode == "lora"
        assert config.lora is not None
        assert config.lora.rank == 16

    def test_lora_required_when_lora_mode(self, model_dir: Path) -> None:
        """training_mode='lora' without lora section raises ValidationError."""
        from pydantic import ValidationError

        from ltx_trainer_mlx.config import LtxTrainerConfig

        with pytest.raises(ValidationError, match="LoRA configuration must be provided"):
            LtxTrainerConfig(
                model={"model_path": str(model_dir), "training_mode": "lora"},
                lora=None,
                data={"preprocessed_data_root": "/tmp/data"},
            )

    def test_v2v_requires_lora_mode(self, model_dir: Path) -> None:
        """video_to_video strategy with training_mode='full' raises ValidationError."""
        from pydantic import ValidationError

        from ltx_trainer_mlx.config import LtxTrainerConfig

        with pytest.raises(ValidationError, match="Training mode must be 'lora'"):
            LtxTrainerConfig(
                model={"model_path": str(model_dir), "training_mode": "full"},
                training_strategy={"name": "video_to_video"},
                validation={"interval": None},
                data={"preprocessed_data_root": "/tmp/data"},
            )


# ---------------------------------------------------------------------------
# 7. TestGradientCheckpointing
# ---------------------------------------------------------------------------


class TestGradientCheckpointing:
    """Tests for mx.checkpoint-based gradient checkpointing in LTXModel block loop.

    The critical invariant: block.trainable_parameters() must be passed as an
    explicit input to the mx.checkpoint closure so that mx.checkpoint can
    differentiate through them.  If params are captured from the closure instead,
    mx.checkpoint gives zero gradients for the LoRA params (silent no-op adapter).
    """

    def test_grads_match_on_off(self) -> None:
        """Checkpointing gives identical LoRA-param grads to the non-checkpoint path."""
        mx.random.seed(42)
        dim, rank = 16, 4

        # Frozen base + trainable LoRA adapter (both non-zero to ensure non-zero grads).
        # mlx_lm's LoRALinear initializes lora_b=0, which zeros out lora_a's gradient;
        # we use a custom block so both adapter matrices are randomly initialized.
        class LoRAStyleBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = nn.Linear(dim, dim, bias=False)  # frozen below
                self.lora_a = mx.random.normal((dim, rank)) * 0.1
                self.lora_b = mx.random.normal((rank, dim)) * 0.1

            def __call__(self, x: mx.array) -> mx.array:
                return self.base(x) + (x @ self.lora_a) @ self.lora_b

        block = LoRAStyleBlock()
        block.base.freeze()  # only the base Linear is frozen; lora_a/lora_b stay trainable

        x = mx.random.normal((2, dim))
        target = mx.random.normal((2, dim))

        def loss_no_ckpt(params: dict, x: mx.array, target: mx.array) -> mx.array:
            block.update(params)
            return mx.mean((block(x) - target) ** 2)

        # Mirrors the pattern in model.py:
        #   def _run_block(params, vh, ah, _block=block, _bidx=block_idx):
        #       _block.update(params); return _block(...)
        #   video_hidden, audio_hidden = mx.checkpoint(_run_block)(
        #       block.trainable_parameters(), video_hidden, audio_hidden
        #   )
        def loss_ckpt(params: dict, x: mx.array, target: mx.array) -> mx.array:
            def _run(params: dict, x: mx.array, _b: nn.Module = block) -> mx.array:
                _b.update(params)
                return _b(x)

            return mx.mean((mx.checkpoint(_run)(params, x) - target) ** 2)

        params = block.trainable_parameters()
        grads_normal = mx.grad(loss_no_ckpt)(params, x, target)
        grads_ckpt = mx.grad(loss_ckpt)(params, x, target)
        mx.eval(grads_normal, grads_ckpt)

        flat_normal = dict(nn.utils.tree_flatten(grads_normal))
        flat_ckpt = dict(nn.utils.tree_flatten(grads_ckpt))

        assert flat_ckpt, "trainable_parameters() returned nothing — freeze setup broken"
        for name, g_c in flat_ckpt.items():
            assert mx.any(g_c != 0).item(), f"Zero grad for '{name}' with checkpointing on"
            g_n = flat_normal[name]
            assert mx.allclose(g_n, g_c, atol=1e-5).item(), (
                f"Grad mismatch for '{name}': max_diff={mx.max(mx.abs(g_n - g_c)).item():.2e}"
            )


# ---------------------------------------------------------------------------
# 8. TestFitAudioTokens
# ---------------------------------------------------------------------------


class TestFitAudioTokens:
    """Tests for _fit_audio_tokens alignment with compute_audio_token_count.

    _fit_audio_tokens must produce a token count that exactly matches
    compute_audio_token_count(num_frames, fps) regardless of whether the raw
    encoded latent comes in slightly over or under the canonical length.
    """

    def test_trim(self) -> None:
        """Oversized latent is trimmed to target_tokens on the time axis."""
        from ltx_trainer_mlx.preprocess import _fit_audio_tokens

        latent = mx.zeros((1, 8, 20, 16))
        result = _fit_audio_tokens(latent, 15)
        assert result.shape == (1, 8, 15, 16)

    def test_pad(self) -> None:
        """Undersized latent is edge-padded to target_tokens on the time axis."""
        from ltx_trainer_mlx.preprocess import _fit_audio_tokens

        latent = mx.zeros((1, 8, 10, 16))
        result = _fit_audio_tokens(latent, 15)
        assert result.shape == (1, 8, 15, 16)

    def test_exact_is_noop(self) -> None:
        """Exactly-sized latent is returned with shape unchanged."""
        from ltx_trainer_mlx.preprocess import _fit_audio_tokens

        latent = mx.zeros((1, 8, 12, 16))
        result = _fit_audio_tokens(latent, 12)
        assert result.shape == (1, 8, 12, 16)

    def test_pad_replicates_last_token(self) -> None:
        """Padding repeats the last time step rather than injecting zeros."""
        from ltx_trainer_mlx.preprocess import _fit_audio_tokens

        sentinel = 99.0
        latent = mx.ones((1, 8, 5, 16)) * sentinel
        result = _fit_audio_tokens(latent, 8)
        assert result.shape == (1, 8, 8, 16)
        padded = result[:, :, 5:, :]
        assert mx.allclose(padded, mx.ones_like(padded) * sentinel).item()

    def test_matches_compute_audio_token_count(self) -> None:
        """Output token count equals compute_audio_token_count for canonical params.

        Verifies that trim and pad paths both converge to the exact canonical
        count that the audio preprocessing pipeline uses.
        """
        from ltx_core_mlx.utils.positions import compute_audio_token_count
        from ltx_trainer_mlx.preprocess import _fit_audio_tokens

        cases = [(33, 24.0), (97, 24.0), (65, 25.0), (9, 24.0)]
        for num_frames, fps in cases:
            target = compute_audio_token_count(num_frames, fps)
            for raw_t in (target - 3, target, target + 3):
                latent = mx.zeros((1, 8, raw_t, 16))
                result = _fit_audio_tokens(latent, target)
                assert result.shape[2] == target, (
                    f"frames={num_frames} fps={fps}: expected {target}, got {result.shape[2]}"
                )
