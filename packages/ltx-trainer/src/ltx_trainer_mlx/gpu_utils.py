"""Metal memory management utilities for training and inference.

Ported from ltx-trainer (Lightricks). Replaces CUDA/nvidia-smi with MLX
Metal memory introspection via ``ltx_core_mlx.utils.memory``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TypeVar

from ltx_core_mlx.utils.memory import aggressive_cleanup, get_memory_stats

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def free_gpu_memory(log: bool = False) -> None:
    """Free Metal memory by running garbage collection and clearing the MLX cache.

    Args:
        log: If True, log memory stats after clearing.
    """
    aggressive_cleanup()
    if log:
        stats = get_memory_stats()
        logger.debug(
            "Metal memory freed. Active: %.2fGB, Peak: %.2fGB, Cache: %.2fGB",
            stats["active_gb"],
            stats["peak_gb"],
            stats["cache_gb"],
        )


class free_gpu_memory_context:  # noqa: N801
    """Context manager and decorator to free Metal memory before and/or after execution.

    Can be used as a decorator::

        @free_gpu_memory_context(after=True)
        def my_function():
            ...

    Or as a context manager::

        with free_gpu_memory_context():
            heavy_operation()

    Args:
        before: Free memory before execution (default: False).
        after: Free memory after execution (default: True).
        log: Log memory stats when freeing (default: False).
    """

    def __init__(self, *, before: bool = False, after: bool = True, log: bool = False) -> None:
        self.before = before
        self.after = after
        self.log = log

    def __enter__(self) -> free_gpu_memory_context:
        if self.before:
            free_gpu_memory(log=self.log)
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        if self.after:
            free_gpu_memory(log=self.log)

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            with self:
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


def get_gpu_memory_gb() -> float:
    """Get current Metal active memory usage in GB.

    Returns:
        Current active Metal memory usage in GB.
    """
    stats = get_memory_stats()
    return stats["active_gb"]
