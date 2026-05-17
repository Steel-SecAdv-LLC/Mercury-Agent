"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Performance profiling utilities.

Decorators and context managers for measuring CPU time, memory, and wall clock
of Mercury detector code paths. Per-module loggers are obtained via
``logging.getLogger(__name__)`` so library output composes with the host
application's logging configuration.

Profiling can be globally disabled via :func:`set_profiling_enabled` to remove
overhead in production.
"""

import cProfile
import functools
import io
import logging
import pstats
import time
import tracemalloc
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_logger = logging.getLogger(__name__)
_profiling_enabled = True


def set_profiling_enabled(enabled: bool) -> None:
    """
    Enable or disable profiling globally.

    Args:
        enabled: Whether to enable profiling.
    """
    global _profiling_enabled
    _profiling_enabled = enabled


def is_profiling_enabled() -> bool:
    """Return whether profiling is globally enabled."""
    return _profiling_enabled


def profile_func(
    top_n: int = 10,
    sort_by: str = "cumtime",
    enabled: bool | None = None,
) -> Callable[..., Any]:
    """
    Decorator to profile a function's CPU time with :mod:`cProfile`.

    Args:
        top_n: Number of top functions to display in the profile output.
        sort_by: Sort key passed to :meth:`pstats.Stats.sort_stats`
            (``"cumtime"``, ``"tottime"``, ``"ncalls"``).
        enabled: Override the global enable flag for this decorator instance.

    Returns:
        The wrapped function. When profiling is disabled the wrapper is a
        thin pass-through with no measurable overhead.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            should_profile = enabled if enabled is not None else _profiling_enabled
            if not should_profile:
                return func(*args, **kwargs)

            profiler = cProfile.Profile()
            profiler.enable()
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
                ps.print_stats(top_n)
                _logger.info("CPU profile for %s:\n%s", func.__name__, s.getvalue())
            return result

        return wrapper

    return decorator


def profile_memory(
    enabled: bool | None = None,
) -> Callable[..., Any]:
    """
    Decorator to profile a function's peak memory usage with :mod:`tracemalloc`.

    Args:
        enabled: Override the global enable flag.

    Returns:
        The wrapped function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            should_profile = enabled if enabled is not None else _profiling_enabled
            if not should_profile:
                return func(*args, **kwargs)

            tracemalloc.start()
            try:
                result = func(*args, **kwargs)
                current, peak = tracemalloc.get_traced_memory()
                _logger.info(
                    "Memory profile for %s: current=%.2f MB peak=%.2f MB",
                    func.__name__,
                    current / 1024 / 1024,
                    peak / 1024 / 1024,
                )
            finally:
                tracemalloc.stop()
            return result

        return wrapper

    return decorator


def profile_time(
    enabled: bool | None = None,
    log_args: bool = False,
) -> Callable[..., Any]:
    """
    Decorator to profile a function's wall-clock execution time.

    Args:
        enabled: Override the global enable flag.
        log_args: When True, include a truncated view of ``args``/``kwargs``
            in the log line. Disabled by default to avoid logging PII or
            large payloads.

    Returns:
        The wrapped function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            should_profile = enabled if enabled is not None else _profiling_enabled
            if not should_profile:
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                arg_info = ""
                if log_args:
                    arg_info = f" args={args[:3]!r} kwargs={list(kwargs.keys())[:3]}"
                _logger.info(
                    "Execution time for %s%s: %.2f ms",
                    func.__name__,
                    arg_info,
                    elapsed_ms,
                )
            return result

        return wrapper

    return decorator


def profile_complete(
    top_n: int = 10,
    sort_by: str = "cumtime",
    enabled: bool | None = None,
) -> Callable[..., Any]:
    """
    Decorator combining CPU, memory, and wall-clock profiling in one pass.

    Args:
        top_n: Number of top functions to display in the CPU profile.
        sort_by: Sort key for the CPU profile.
        enabled: Override the global enable flag.

    Returns:
        The wrapped function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            should_profile = enabled if enabled is not None else _profiling_enabled
            if not should_profile:
                return func(*args, **kwargs)

            profiler = cProfile.Profile()
            profiler.enable()
            tracemalloc.start()
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                profiler.disable()

                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
                ps.print_stats(top_n)

                current, peak = tracemalloc.get_traced_memory()
                _logger.info(
                    "Complete profile for %s: time=%.2f ms current_mem=%.2f MB peak_mem=%.2f MB\n%s",
                    func.__name__,
                    elapsed_ms,
                    current / 1024 / 1024,
                    peak / 1024 / 1024,
                    s.getvalue(),
                )
            except Exception:
                profiler.disable()
                raise
            finally:
                tracemalloc.stop()
            return result

        return wrapper

    return decorator


class PerformanceBenchmark:
    """
    Context manager for benchmarking arbitrary code blocks.

    Captures wall-clock elapsed time and peak memory between ``__enter__`` and
    ``__exit__``. Emits a single log line on exit; the elapsed time and peak
    memory remain accessible on the instance for programmatic use.

    Example::

        with PerformanceBenchmark("detector.forward") as bench:
            detector.run(payload)
        latency_ms = bench.elapsed_ms
    """

    def __init__(
        self,
        name: str,
        enabled: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Args:
            name: Human-readable benchmark name (appears in log output).
            enabled: Override the global enable flag.
            logger: Logger to emit the benchmark line to. Defaults to this
                module's logger.
        """
        self.name = name
        self.enabled = enabled if enabled is not None else _profiling_enabled
        self.logger = logger or _logger
        self.start_time = 0.0
        self.elapsed_ms = 0.0
        self.peak_mem_bytes = 0
        self.current_mem_bytes = 0

    def __enter__(self) -> PerformanceBenchmark:
        if self.enabled:
            self.start_time = time.perf_counter()
            tracemalloc.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if not self.enabled:
            return
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.current_mem_bytes, self.peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.logger.info(
            "Benchmark '%s': %.2f ms, current=%.2f MB peak=%.2f MB",
            self.name,
            self.elapsed_ms,
            self.current_mem_bytes / 1024 / 1024,
            self.peak_mem_bytes / 1024 / 1024,
        )


def benchmark_function(
    func: Callable[..., Any],
    *args: Any,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Benchmark a function over multiple iterations and return summary statistics.

    Args:
        func: Function to benchmark.
        *args: Positional arguments to pass to ``func`` each call.
        iterations: Number of timed iterations.
        warmup: Number of un-timed warmup iterations (e.g. for JIT or cache
            warming).
        **kwargs: Keyword arguments to pass to ``func`` each call.

    Returns:
        Dict with keys ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``,
        ``median_ms``, ``iterations``, ``warmup``.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    for _ in range(warmup):
        func(*args, **kwargs)

    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "iterations": iterations,
        "warmup": warmup,
    }
