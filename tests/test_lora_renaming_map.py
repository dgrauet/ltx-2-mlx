"""Contract tests for LTXV_LORA_COMFY_RENAMING_MAP.

The four audio/joint-block replacements (.linear_1., .linear_2.,
audio_ff.net.0.proj., audio_ff.net.2.) are load-bearing for the
BlockLoraSource streaming path, which uses this map directly.  A
regression that drops any of these silently produces a no-op LoRA
(keys never match model weights → no contribution).
"""

from __future__ import annotations

import pytest

from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP


@pytest.mark.parametrize(
    "input_key,expected",
    [
        # Original video feed-forward replacements
        (
            "transformer_blocks.0.ff.net.0.proj.lora_A.weight",
            "transformer_blocks.0.ff.proj_in.lora_A.weight",
        ),
        (
            "transformer_blocks.0.ff.net.2.lora_B.weight",
            "transformer_blocks.0.ff.proj_out.lora_B.weight",
        ),
        # New: audio/joint-block linear replacements
        (
            "transformer_blocks.0.linear_1.lora_A.weight",
            "transformer_blocks.0.linear1.lora_A.weight",
        ),
        (
            "transformer_blocks.0.linear_2.lora_B.weight",
            "transformer_blocks.0.linear2.lora_B.weight",
        ),
        # New: audio feed-forward replacements
        (
            "transformer_blocks.0.audio_ff.net.0.proj.lora_A.weight",
            "transformer_blocks.0.audio_ff.proj_in.lora_A.weight",
        ),
        (
            "transformer_blocks.0.audio_ff.net.2.lora_B.weight",
            "transformer_blocks.0.audio_ff.proj_out.lora_B.weight",
        ),
    ],
)
def test_renaming_map_apply_to_key(input_key: str, expected: str) -> None:
    assert LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(input_key) == expected
