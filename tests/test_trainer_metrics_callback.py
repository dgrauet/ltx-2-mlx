"""Tests for the trainer's per-optimizer-step metrics callback.

Drives the real ``LtxvTrainer.train`` loop on a tiny linear model: model
loading, dataset loading and checkpointing are stubbed, so no weights are
needed.
"""

from __future__ import annotations

import inspect
import logging
import math
from pathlib import Path
from typing import Any, get_args

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_trainer_mlx.trainer import (
    LtxvTrainer,
    MetricsCallback,
    StepCallback,
    StepMetrics,
    _emit_step_metrics,
)


def _make_trainer(tmp_path: Path, *, steps: int, accum: int, scheduler: str = "linear") -> LtxvTrainer:
    """Build an ``LtxvTrainer`` around a tiny linear model, bypassing model loading."""
    from ltx_trainer_mlx.config import LtxTrainerConfig

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "transformer.safetensors").touch()

    cfg = LtxTrainerConfig(
        model={"model_path": str(model_dir)},
        lora={"rank": 4, "alpha": 4},
        data={"preprocessed_data_root": str(tmp_path / "data")},
        optimization={
            "steps": steps,
            "gradient_accumulation_steps": accum,
            "learning_rate": 1e-2,
            "scheduler_type": scheduler,
            "batch_size": 1,
        },
        validation={"interval": None},
        checkpoints={"interval": None},
        output_dir=str(tmp_path / "out"),
    )

    trainer = object.__new__(LtxvTrainer)
    trainer._config = cfg
    trainer._global_step = -1
    trainer._checkpoint_paths = []
    trainer._wandb_run = None

    mx.random.seed(0)
    model = nn.Linear(4, 1)
    trainer._transformer = model

    xs = [mx.random.normal((2, 4)) for _ in range(3)]
    ys = [mx.random.normal((2, 1)) for _ in range(3)]
    batches = [{"x": x, "y": y} for x, y in zip(xs, ys, strict=True)]

    def _init_dataloader() -> None:
        trainer._dataloader = batches

    def _build_loss_fn() -> Any:
        def loss_fn(batch: dict[str, mx.array]) -> mx.array:
            return mx.mean((model(batch["x"]) - batch["y"]) ** 2)

        return loss_fn

    trainer._init_dataloader = _init_dataloader  # type: ignore[method-assign]
    trainer._build_loss_fn = _build_loss_fn  # type: ignore[method-assign]
    trainer._save_checkpoint = lambda: tmp_path / "out" / "final.safetensors"  # type: ignore[method-assign]
    return trainer


class TestMetricsCallback:
    """End-to-end tests on a tiny training loop."""

    @pytest.mark.parametrize("accum", [1, 2])
    def test_one_metrics_per_optimizer_step(self, tmp_path: Path, accum: int) -> None:
        """One StepMetrics per optimizer step, monotonically increasing, finite loss."""
        steps = 5
        trainer = _make_trainer(tmp_path, steps=steps, accum=accum)
        received: list[StepMetrics] = []

        trainer.train(disable_progress_bars=True, metrics_callback=received.append)

        assert [m.step for m in received] == list(range(1, steps + 1))
        for m in received:
            assert m.total_steps == steps
            assert math.isfinite(m.loss) and m.loss >= 0.0
            assert m.step_time_s >= 0.0
            assert m.peak_memory_gb is not None and m.peak_memory_gb >= 0.0

    def test_lr_matches_scheduler(self, tmp_path: Path) -> None:
        """lr is the rate applied by the update: base lr, then schedule(step - 1)."""
        steps = 5
        trainer = _make_trainer(tmp_path, steps=steps, accum=1, scheduler="linear")
        received: list[StepMetrics] = []

        trainer.train(disable_progress_bars=True, metrics_callback=received.append)

        schedule = trainer._lr_schedule
        assert schedule is not None
        base_lr = trainer._config.optimization.learning_rate
        expected = [base_lr] + [schedule(k - 1) for k in range(2, steps + 1)]
        assert [m.lr for m in received] == pytest.approx(expected, rel=1e-5)
        # Linear decay: strictly decreasing after the first step.
        assert received[-1].lr < received[1].lr

    def test_loss_is_mean_over_micro_batches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With gradient accumulation, loss is the mean of the micro-batch losses."""
        trainer = _make_trainer(tmp_path, steps=2, accum=2)
        micro_losses: list[float] = []

        # Record the scalar loss of each micro-batch through value_and_grad.
        orig_vag = nn.value_and_grad

        def _vag(model: nn.Module, fn: Any) -> Any:
            wrapped = orig_vag(model, fn)

            def call(batch: Any) -> Any:
                loss, grads = wrapped(batch)
                micro_losses.append(float(loss.item()))
                return loss, grads

            return call

        monkeypatch.setattr("ltx_trainer_mlx.trainer.nn.value_and_grad", _vag)
        received: list[StepMetrics] = []

        trainer.train(disable_progress_bars=True, metrics_callback=received.append)

        assert len(micro_losses) == 4
        assert received[0].loss == pytest.approx((micro_losses[0] + micro_losses[1]) / 2)
        assert received[1].loss == pytest.approx((micro_losses[2] + micro_losses[3]) / 2)

    def test_step_callback_unchanged(self, tmp_path: Path) -> None:
        """Legacy step_callback still gets (step, total_steps, paths) once per optimizer step."""
        steps = 3
        trainer = _make_trainer(tmp_path, steps=steps, accum=2)
        calls: list[tuple[int, int, list[Path]]] = []

        trainer.train(
            disable_progress_bars=True,
            step_callback=lambda s, t, p: calls.append((s, t, p)),
        )

        assert calls == [(1, steps, []), (2, steps, []), (3, steps, [])]

    def test_raising_metrics_callback_does_not_abort(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A raising metrics_callback is disabled with a warning; training completes."""
        steps = 4
        trainer = _make_trainer(tmp_path, steps=steps, accum=1)
        calls: list[int] = []
        legacy: list[int] = []

        def bad(m: StepMetrics) -> None:
            calls.append(m.step)
            raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger="ltx_trainer_mlx.trainer"):
            saved, _stats = trainer.train(
                disable_progress_bars=True,
                step_callback=lambda s, t, p: legacy.append(s),
                metrics_callback=bad,
            )

        assert calls == [1]  # disabled after the first failure
        assert legacy == list(range(1, steps + 1))
        assert trainer._global_step == steps
        assert saved.name == "final.safetensors"
        records = [r for r in caplog.records if "metrics_callback raised" in r.getMessage()]
        assert len(records) == 1 and records[0].exc_info is not None


class TestApi:
    """Signature / helper tests."""

    def test_step_callback_arity_is_three(self) -> None:
        params, _ret = get_args(StepCallback)
        assert len(params) == 3

    def test_metrics_callback_takes_step_metrics(self) -> None:
        params, _ret = get_args(MetricsCallback)
        assert params == [StepMetrics]

    def test_train_signature(self) -> None:
        params = inspect.signature(LtxvTrainer.train).parameters
        assert list(params) == ["self", "disable_progress_bars", "step_callback", "metrics_callback"]
        assert params["metrics_callback"].default is None

    def test_step_metrics_is_frozen(self) -> None:
        m = StepMetrics(step=1, total_steps=2, loss=0.5, lr=1e-4, step_time_s=0.1)
        assert m.peak_memory_gb is None
        with pytest.raises(AttributeError):
            m.loss = 1.0  # type: ignore[misc]

    def test_emit_helper(self) -> None:
        m = StepMetrics(step=1, total_steps=1, loss=0.0, lr=0.0, step_time_s=0.0)
        assert _emit_step_metrics(None, m) is None
        seen: list[StepMetrics] = []
        assert _emit_step_metrics(seen.append, m) is not None
        assert seen == [m]

        def bad(_: StepMetrics) -> None:
            raise ValueError("x")

        assert _emit_step_metrics(bad, m) is None
