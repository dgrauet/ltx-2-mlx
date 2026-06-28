"""Regression tests for the ``_pending_loras`` dispatch contract.

Every pipeline ``load()`` override must route DiT construction through
:meth:`BasePipeline._load_transformer_with_optional_streaming` so that
LoRAs set via ``pipe._pending_loras = [...]`` are fused at load time.
Forgetting the dispatch produces a silent failure: T2V generation runs
with no LoRA applied (no error, output identical to a base run).

These tests mock the heavy weight-loading deps so they run in <1s and
exercise the dispatching logic in isolation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ltx_core_mlx.loader.sd_ops import LTXV_LORA_BLOCK_PREFIX, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines_mlx._base import BasePipeline


@pytest.fixture
def pipeline_stub():
    """A bare-minimum stub that exposes the dispatch attributes only."""

    class _Stub:
        verbose = False
        low_ram_streaming = False

        def _fuse_pending_loras(self, weights, pending):
            # Spy: record the call, return a tagged dict so we can assert
            # downstream consumed it.
            self._fuse_spy = (weights, pending)
            return {"__fused__": True}

    return _Stub()


# Bound via ``__get__`` so the stub instance receives ``self`` correctly.
_load_under_test = BasePipeline._load_transformer_with_optional_streaming


def test_no_pending_loras_delegates_to_orchestration(pipeline_stub):
    """Without pending LoRAs, falls through to ``utils._orchestration.load_transformer``."""
    sentinel = MagicMock(name="dit_from_orchestration")
    with patch(
        "ltx_pipelines_mlx.utils._orchestration.load_transformer",
        return_value=sentinel,
    ) as orch_load:
        result = _load_under_test(pipeline_stub, Path("/fake/transformer.safetensors"))

    assert result is sentinel
    orch_load.assert_called_once()
    assert not hasattr(pipeline_stub, "_fuse_spy")


def test_pending_loras_takes_fusion_path(pipeline_stub):
    """With pending LoRAs, the orchestration helper is bypassed for the fusion path."""
    pipeline_stub._pending_loras = [("/fake/lora.safetensors", 0.75)]

    with (
        patch("ltx_pipelines_mlx._base.load_split_safetensors", return_value={"raw": "weights"}) as load_sft,
        patch("ltx_pipelines_mlx._base.apply_quantization") as apply_q,
        patch("ltx_pipelines_mlx._base.LTXModel") as LTXModel_cls,
        patch("ltx_pipelines_mlx._base.aggressive_cleanup"),
        patch("ltx_pipelines_mlx.utils._orchestration.load_transformer") as orch_load,
    ):
        dit_instance = MagicMock(name="dit")
        LTXModel_cls.return_value = dit_instance

        result = _load_under_test(pipeline_stub, Path("/fake/transformer.safetensors"))

    orch_load.assert_not_called()  # critical: the no-LoRA path must NOT run
    load_sft.assert_called_once()
    apply_q.assert_called_once_with(dit_instance, {"__fused__": True})
    dit_instance.load_weights.assert_called_once()
    assert result is dit_instance
    # The spy verifies the LoRA list reached _fuse_pending_loras unchanged.
    assert pipeline_stub._fuse_spy[1] == [("/fake/lora.safetensors", 0.75)]


def test_pending_loras_with_streaming_attaches_lora_sources(pipeline_stub):
    """``low_ram_streaming`` + LoRA attaches ``BlockLoraSource`` via bind-time fusion."""
    pipeline_stub._pending_loras = [("/fake/lora.safetensors", 1.0)]
    pipeline_stub.low_ram_streaming = True

    mock_model = MagicMock(name="streaming_dit")
    object.__setattr__(mock_model, "_lora_sources", [])
    mock_source = MagicMock(name="block_lora_source")

    with (
        patch(
            "ltx_pipelines_mlx.utils._orchestration.load_transformer",
            return_value=mock_model,
        ) as orch_load,
        patch(
            "ltx_core_mlx.loader.block_streaming.BlockLoraSource",
            return_value=mock_source,
        ) as BlockLoraSource_cls,
        patch(
            "ltx_pipelines_mlx.utils._orchestration.resolve_lora_path",
            return_value="/fake/lora.safetensors",
        ),
    ):
        result = _load_under_test(pipeline_stub, Path("/fake/transformer.safetensors"))

    orch_load.assert_called_once_with(Path("/fake/transformer.safetensors"), low_ram_streaming=True)
    BlockLoraSource_cls.assert_called_once_with(
        "/fake/lora.safetensors",
        block_prefix=LTXV_LORA_BLOCK_PREFIX,
        strength=1.0,
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
    )
    assert result is mock_model
    attached = object.__getattribute__(mock_model, "_lora_sources")
    assert mock_source in attached


@pytest.mark.parametrize(
    "loras",
    [
        [("/fake/lora_a.safetensors", 0.8), ("/fake/lora_b.safetensors", 1.2)],
    ],
)
def test_pending_loras_with_streaming_multi_lora(pipeline_stub, loras):
    """Multiple pending LoRAs each produce one ``BlockLoraSource`` attachment."""
    pipeline_stub._pending_loras = loras
    pipeline_stub.low_ram_streaming = True

    mock_model = MagicMock(name="streaming_dit")
    object.__setattr__(mock_model, "_lora_sources", [])

    mock_sources = [MagicMock(name=f"source_{i}") for i in range(len(loras))]

    with (
        patch(
            "ltx_pipelines_mlx.utils._orchestration.load_transformer",
            return_value=mock_model,
        ),
        patch(
            "ltx_core_mlx.loader.block_streaming.BlockLoraSource",
            side_effect=mock_sources,
        ) as BlockLoraSource_cls,
        patch(
            "ltx_pipelines_mlx.utils._orchestration.resolve_lora_path",
            side_effect=[path for path, _ in loras],
        ),
    ):
        result = _load_under_test(pipeline_stub, Path("/fake/transformer.safetensors"))

    assert BlockLoraSource_cls.call_count == len(loras)
    for (path, strength), _mock_source in zip(loras, mock_sources):
        BlockLoraSource_cls.assert_any_call(
            path,
            block_prefix=LTXV_LORA_BLOCK_PREFIX,
            strength=strength,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
    attached = object.__getattribute__(mock_model, "_lora_sources")
    assert set(mock_sources).issubset(set(attached))
    assert result is mock_model


@pytest.mark.parametrize(
    "pipeline_module,pipeline_class",
    [
        ("ltx_pipelines_mlx.distilled", "DistilledPipeline"),
        ("ltx_pipelines_mlx.ti2vid_one_stage", "TI2VidOneStagePipeline"),
        ("ltx_pipelines_mlx.ti2vid_two_stages", "TI2VidTwoStagesPipeline"),
        ("ltx_pipelines_mlx.ic_lora", "ICLoraPipeline"),
    ],
)
def test_pipeline_load_routes_through_wrapper(pipeline_module, pipeline_class):
    """Every pipeline that ships a custom ``load()`` must call the wrapper.

    Asserts that the source of each pipeline's ``load()`` references
    ``_load_transformer_with_optional_streaming``. A future override that
    sidesteps the wrapper (e.g. calling ``_orchestration.load_transformer``
    directly) would silently break LoRA T2V — this test catches that.
    """
    import importlib
    import inspect

    mod = importlib.import_module(pipeline_module)
    cls = getattr(mod, pipeline_class)
    source = inspect.getsource(cls.load)
    # Either the direct wrapper, or ``_load_dev_transformer`` (which itself
    # routes through the wrapper for the dev-transformer variant). Both
    # honor ``_pending_loras``; only sidestepping ``_orchestration.load_transformer``
    # directly would silently drop LoRAs.
    routes_via_wrapper = "_load_transformer_with_optional_streaming" in source or "_load_dev_transformer" in source
    assert routes_via_wrapper, (
        f"{pipeline_class}.load() must route DiT construction through one of "
        "self._load_transformer_with_optional_streaming(...) or "
        "self._load_dev_transformer() so that _pending_loras is honored. "
        "See _base.py for the contract."
    )
