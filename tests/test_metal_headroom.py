"""#79: the Gemma encoder must not widen a stricter configured cache limit.

`--low-ram` sets `mx.set_cache_limit(0)` at pipeline init so freed buffers
return to the OS. `_ensure_metal_headroom()` (called at every prompt
encoding) used to unconditionally set the limit to 0.9x device memory,
silently clobbering the low-ram setting for the whole rest of the run —
observed as 8-21 GB of retained cache during video decode on 32 GB Macs.

The headroom call was meant to CAP the cache below the default, never to
raise a stricter limit: min semantics.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

HEADROOM = int(mx.device_info()["memory_size"] * 0.9)


def _current_cache_limit() -> int:
    """mx has no getter; set_cache_limit returns the previous value."""
    cur = mx.set_cache_limit(0)
    mx.set_cache_limit(cur)
    return cur


@pytest.fixture
def restore_cache_limit():
    prev = _current_cache_limit()
    yield
    mx.set_cache_limit(prev)


def test_stricter_limit_is_preserved(restore_cache_limit):
    """The low-ram case: a configured 0 must survive prompt encoding."""
    mx.set_cache_limit(0)
    GemmaLanguageModel._ensure_metal_headroom()
    assert _current_cache_limit() == 0


def test_intermediate_stricter_limit_is_preserved(restore_cache_limit):
    """Any limit below the cap is a deliberate user setting — keep it."""
    mx.set_cache_limit(HEADROOM // 4)
    GemmaLanguageModel._ensure_metal_headroom()
    assert _current_cache_limit() == HEADROOM // 4


def test_looser_limit_is_capped(restore_cache_limit):
    """The original purpose: cap a looser/default limit to 0.9x memory."""
    mx.set_cache_limit(HEADROOM * 2)
    GemmaLanguageModel._ensure_metal_headroom()
    assert _current_cache_limit() == HEADROOM
