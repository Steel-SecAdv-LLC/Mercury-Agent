"""Performance profiling utilities for Mercury Agent.

Provides four decorators, one context manager, and one benchmarking helper
for measuring CPU time, memory consumption, and wall-clock execution of
synchronous functions. All entry points are no-ops when profiling is
globally disabled via :func:`set_profiling_enabled`.

Source provenance:
    Ported from Omni-AXA-Engine ``omni_anomaly_engine.utils.profiling``
    (commit history preserved in CHANGELOG.md). Mercury-specific changes:

    * Logger namespace switched to ``omni_mercury_engine.profiling`` via
      :func:`omni_mercury_engine.utils.logging.get_logger`.
    * ``tracemalloc`` is treated as always-available (mandatory stdlib in
      Python 3.11+, Mercury's minimum supported runtime); the original's
      ``TRACEMALLOC_AVAILABLE`` import guard was dead code on supported
      interpreters and has been removed.
    * Public surface fully typed for ``mypy --strict``; ``Any``-typed
      decorator returns replaced with explicit ``Callable[P, R]`` /
      ``Callable[P, Awaitable[R]]`` signatures using :pep:`612` paramspecs.
    * Nested ``tracemalloc.start()`` / ``tracemalloc.stop()`` calls are
      reference-counted by the stdlib, but a single Mercury invocation
      that wraps itself in two profiling decorators would previously
      attribute the inner allocation totals to the outer call. The port
      records the tracemalloc state on entry and only stops tracing if
      it was started by the same decorator instance.
    * Exception-path cleanup hardened in :func:`profile_func` so
      ``profiler.disable()`` runs even when stats emission raises.

Example:
    Basic decorator usage::

        from omni_mercury_engine.utils.profiling import profile_time

        @profile_time()
        def expensive_op(x: int) -> int:
            return sum(range(x))

    Benchmark a function over many iterations::

        from omni_mercury_engine.utils.profiling import benchmark_function

        stats = benchmark_function(expensive_op, 10_000, iterations=200)
        print(stats["mean_ms"], stats["std_ms"])
"""

# Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see https://www.gnu.org/licenses/.

from __future__ import annotations

import asyncio
import cProfile
import functools
import io
import pstats
import time
import tracemalloc
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import numpy as np

from omni_mercury_engine.utils.logging import get_logger

if TYPE_CHECKING:
    import logging
    from collections.abc import Awaitable, Callable
    from types import TracebackType

P = ParamSpec("P")
R = TypeVar("R")

# Module-level state. Mutated through the public accessors only.
_profiling_enabled: bool = True
_logger: logging.Logger | None = None

_VALID_SORT_KEYS: frozenset[str] = frozenset(
    {
        "calls",
        "cumtime",
        "cumulative",
        "file",
        "filename",
        "line",
        "module",
        "name",
        "ncalls",
        "nfl",
        "pcalls",
        "stdname",
        "time",
        "tottime",
    }
)


def set_profiling_enabled(enabled: bool) -> None:
    """Enable or disable profiling for every decorator and context manager.

    Args:
        enabled: ``True`` to enable profiling, ``False`` to make every
            decorator a transparent no-op.
    """
    global _profiling_enabled
    _profiling_enabled = bool(enabled)


def is_profiling_enabled() -> bool:
    """Return whether profiling is currently enabled globally.

    Returns:
        Current value of the global profiling switch.
    """
    return _profiling_enabled


def set_profiling_logger(logger: logging.Logger) -> None:
    """Set the logger used for profiling output.

    Args:
        logger: Logger instance to receive profiling output. Replaces any
            previously configured logger.
    """
    global _logger
    _logger = logger


def get_profiling_logger() -> logging.Logger:
    """Return the logger used for profiling output.

    The logger is lazily created on first access and namespaced under
    ``omni_mercury_engine.profiling`` so it inherits Mercury Agent's
    handler configuration.

    Returns:
        Logger used for profiling output.
    """
    global _logger
    if _logger is None:
        _logger = get_logger("omni_mercury_engine.profiling")
    return _logger


def _resolve_enabled(override: bool | None) -> bool:
    """Resolve the effective enable flag for a single decorator invocation."""
    return _profiling_enabled if override is None else bool(override)


def _validate_sort_key(sort_by: str) -> str:
    """Validate ``sort_by`` against :mod:`pstats` accepted keys.

    Args:
        sort_by: Candidate sort key.

    Returns:
        Validated ``sort_by`` string.

    Raises:
        ValueError: If ``sort_by`` is not a key accepted by
            :meth:`pstats.Stats.sort_stats`.
    """
    if sort_by not in _VALID_SORT_KEYS:
        raise ValueError(
            f"Invalid pstats sort key: {sort_by!r}. Expected one of " f"{sorted(_VALID_SORT_KEYS)}."
        )
    return sort_by


def _format_cpu_stats(profiler: cProfile.Profile, top_n: int, sort_by: str) -> str:
    """Format a pstats summary as a string.

    Args:
        profiler: A disabled :class:`cProfile.Profile` instance.
        top_n: Number of top entries to include.
        sort_by: Sort key for :meth:`pstats.Stats.sort_stats`.

    Returns:
        Multi-line formatted pstats output.
    """
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats(sort_by).print_stats(top_n)
    return stream.getvalue()


def profile_func(
    top_n: int = 10,
    sort_by: str = "cumtime",
    enabled: bool | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to profile its CPU usage with :mod:`cProfile`.

    Args:
        top_n: Number of top functions to display in the pstats summary.
            Must be a positive integer.
        sort_by: Sort key passed to :meth:`pstats.Stats.sort_stats`. One of
            ``"cumtime"``, ``"tottime"``, ``"ncalls"``, etc.
        enabled: Per-decorator override of the global enable flag. When
            ``None`` (the default), the global flag is used.

    Returns:
        Decorator that wraps a callable with a CPU profiler.

    Raises:
        ValueError: If ``top_n`` is non-positive or ``sort_by`` is invalid.

    Example:
        Profile a single function call::

            @profile_func(top_n=20)
            def detect_anomalies(stream):
                ...
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    validated_sort = _validate_sort_key(sort_by)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _resolve_enabled(enabled):
                return func(*args, **kwargs)

            logger = get_profiling_logger()
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                return func(*args, **kwargs)
            finally:
                profiler.disable()
                try:
                    output = _format_cpu_stats(profiler, top_n, validated_sort)
                except Exception:
                    logger.exception("Failed to format CPU profile for %s", func.__qualname__)
                else:
                    logger.info("CPU Profile for %s:\n%s", func.__qualname__, output)

        return wrapper

    return decorator


def profile_memory(
    enabled: bool | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to profile its memory usage with :mod:`tracemalloc`.

    Reports the current and peak resident memory in megabytes after the
    wrapped function returns. If ``tracemalloc`` was already tracing on
    entry (e.g. an outer decorator started it), the inner decorator does
    not stop tracing — it simply records the delta.

    Args:
        enabled: Per-decorator override of the global enable flag.

    Returns:
        Decorator that wraps a callable with a memory profiler.

    Example:
        >>> @profile_memory()
        ... def allocate_tensor(n: int) -> bytes:
        ...     return bytes(n)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _resolve_enabled(enabled):
                return func(*args, **kwargs)

            logger = get_profiling_logger()
            we_started_tracing = not tracemalloc.is_tracing()
            if we_started_tracing:
                tracemalloc.start()
            try:
                result = func(*args, **kwargs)
            finally:
                current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                if we_started_tracing:
                    tracemalloc.stop()
                logger.info(
                    "Memory Profile for %s: Current=%.2f MB, Peak=%.2f MB",
                    func.__qualname__,
                    current_bytes / (1024.0 * 1024.0),
                    peak_bytes / (1024.0 * 1024.0),
                )
            return result

        return wrapper

    return decorator


def profile_time(
    enabled: bool | None = None,
    log_args: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to log its execution time in milliseconds.

    Args:
        enabled: Per-decorator override of the global enable flag.
        log_args: When ``True``, include a truncated argument summary in
            the emitted log line (first three positional args and first
            three keyword arg names).

    Returns:
        Decorator that wraps a callable with a wall-clock timer.

    Example:
        >>> @profile_time(log_args=True)
        ... def detect(data: list[int]) -> int:
        ...     return sum(data)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _resolve_enabled(enabled):
                return func(*args, **kwargs)

            logger = get_profiling_logger()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                arg_info = ""
                if log_args:
                    arg_info = (
                        f" with args={list(args)[:3]!r}, " f"kwargs={list(kwargs.keys())[:3]!r}"
                    )
                logger.info(
                    "Execution time for %s%s: %.2f ms",
                    func.__qualname__,
                    arg_info,
                    elapsed_ms,
                )

        return wrapper

    return decorator


def profile_time_async(
    enabled: bool | None = None,
    log_args: bool = False,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Async-aware counterpart of :func:`profile_time` for coroutines.

    Mercury Agent uses :mod:`asyncio` extensively in ``utils/comm.py`` and
    the FastAPI server; this decorator is provided so async hot paths can
    be timed without manually wrapping ``await`` sites.

    Args:
        enabled: Per-decorator override of the global enable flag.
        log_args: Include a truncated argument summary.

    Returns:
        Decorator that wraps a coroutine function with a wall-clock timer.

    Raises:
        TypeError: If the wrapped callable is not a coroutine function.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f"profile_time_async expects a coroutine function, got {func!r}")

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _resolve_enabled(enabled):
                return await func(*args, **kwargs)

            logger = get_profiling_logger()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                arg_info = ""
                if log_args:
                    arg_info = (
                        f" with args={list(args)[:3]!r}, " f"kwargs={list(kwargs.keys())[:3]!r}"
                    )
                logger.info(
                    "Execution time for %s%s: %.2f ms",
                    func.__qualname__,
                    arg_info,
                    elapsed_ms,
                )

        return wrapper

    return decorator


def profile_complete(
    top_n: int = 10,
    sort_by: str = "cumtime",
    enabled: bool | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function with combined CPU, memory, and wall-clock profiling.

    Emits a single multi-line log record summarising all three dimensions.
    On exception the underlying ``cProfile`` and ``tracemalloc`` resources
    are cleaned up before the exception propagates.

    Args:
        top_n: Number of top functions to display in the pstats summary.
        sort_by: Sort key for :meth:`pstats.Stats.sort_stats`.
        enabled: Per-decorator override of the global enable flag.

    Returns:
        Decorator producing a complete profile log on every call.

    Raises:
        ValueError: If ``top_n`` is non-positive or ``sort_by`` is invalid.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    validated_sort = _validate_sort_key(sort_by)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _resolve_enabled(enabled):
                return func(*args, **kwargs)

            logger = get_profiling_logger()
            profiler = cProfile.Profile()
            profiler.enable()
            we_started_tracing = not tracemalloc.is_tracing()
            if we_started_tracing:
                tracemalloc.start()
            start = time.perf_counter()

            try:
                result = func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                profiler.disable()
                current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                if we_started_tracing:
                    tracemalloc.stop()

                try:
                    cpu_summary = _format_cpu_stats(profiler, top_n, validated_sort)
                except Exception:
                    logger.exception("Failed to format CPU profile for %s", func.__qualname__)
                    cpu_summary = "<unavailable: pstats formatting failed>"

                divider = "=" * 80
                logger.info(
                    "\n%s\nComplete Profile for %s\n%s\n\n"
                    "Execution Time: %.2f ms\n"
                    "Memory: Current=%.2f MB, Peak=%.2f MB\n\n"
                    "CPU Profile (top %d):\n%s%s\n",
                    divider,
                    func.__qualname__,
                    divider,
                    elapsed_ms,
                    current_bytes / (1024.0 * 1024.0),
                    peak_bytes / (1024.0 * 1024.0),
                    top_n,
                    cpu_summary,
                    divider,
                )
            return result

        return wrapper

    return decorator


class PerformanceBenchmark(AbstractContextManager["PerformanceBenchmark"]):
    """Context manager for benchmarking an arbitrary code block.

    Records wall-clock time and ``tracemalloc`` current/peak allocations,
    then emits a single log line on exit. Cooperates with nested
    benchmarks: an inner instance will not stop tracemalloc if it was
    already running when the inner context was entered.

    Attributes:
        name: Human-readable benchmark label included in the log line.
        enabled: Whether this instance will record any measurements.
        logger: Logger instance receiving the emitted log line.
        elapsed_ms: Wall-clock time of the most recent ``__exit__``.
            Zero if the benchmark was disabled.
        current_mb: Tracemalloc current allocation (MB) at exit. Zero if
            disabled.
        peak_mb: Tracemalloc peak allocation (MB) at exit. Zero if
            disabled.

    Example:
        >>> with PerformanceBenchmark("feature_extraction") as bench:
        ...     features = extract(data)
        >>> print(bench.elapsed_ms)
    """

    def __init__(
        self,
        name: str,
        enabled: bool | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the benchmark.

        Args:
            name: Label for the benchmark in log output.
            enabled: Per-instance override of the global enable flag.
            logger: Logger to emit results to. Defaults to the module
                profiling logger.
        """
        self.name: str = name
        self.enabled: bool = _resolve_enabled(enabled)
        self.logger: logging.Logger = logger or get_profiling_logger()
        self._start_time: float = 0.0
        self._we_started_tracing: bool = False
        self.elapsed_ms: float = 0.0
        self.current_mb: float = 0.0
        self.peak_mb: float = 0.0

    def __enter__(self) -> PerformanceBenchmark:
        """Start the benchmark and return ``self``."""
        if not self.enabled:
            return self
        self._start_time = time.perf_counter()
        self._we_started_tracing = not tracemalloc.is_tracing()
        if self._we_started_tracing:
            tracemalloc.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the benchmark and emit a log line.

        Does not suppress exceptions raised within the ``with`` block.
        """
        if not self.enabled:
            return
        self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        if self._we_started_tracing:
            tracemalloc.stop()
        self.current_mb = current_bytes / (1024.0 * 1024.0)
        self.peak_mb = peak_bytes / (1024.0 * 1024.0)
        self.logger.info(
            "Benchmark '%s': %.2f ms, Memory: %.2f MB (Peak: %.2f MB)",
            self.name,
            self.elapsed_ms,
            self.current_mb,
            self.peak_mb,
        )


def benchmark_function[**P, R](
    func: Callable[P, R],
    *args: Any,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs: Any,
) -> dict[str, float | int]:
    """Measure the execution time of ``func`` over many iterations.

    Runs a configurable number of warmup iterations whose results are
    discarded, then collects ``iterations`` timings and returns summary
    statistics in milliseconds.

    Args:
        func: Callable to benchmark.
        *args: Positional arguments forwarded to ``func``.
        iterations: Number of measured iterations. Must be positive.
        warmup: Number of warmup iterations whose timings are discarded.
            Must be non-negative.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        Mapping with the keys ``mean_ms``, ``std_ms``, ``min_ms``,
        ``max_ms``, ``median_ms``, ``iterations``, and ``warmup``.

    Raises:
        ValueError: If ``iterations`` is non-positive or ``warmup`` is
            negative.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    untyped_func = cast("Callable[..., R]", func)
    for _ in range(warmup):
        untyped_func(*args, **kwargs)

    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        untyped_func(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000.0)

    times_array = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(times_array)),
        "std_ms": float(np.std(times_array)),
        "min_ms": float(np.min(times_array)),
        "max_ms": float(np.max(times_array)),
        "median_ms": float(np.median(times_array)),
        "iterations": int(iterations),
        "warmup": int(warmup),
    }


__all__ = [
    "PerformanceBenchmark",
    "benchmark_function",
    "get_profiling_logger",
    "is_profiling_enabled",
    "profile_complete",
    "profile_func",
    "profile_memory",
    "profile_time",
    "profile_time_async",
    "set_profiling_enabled",
    "set_profiling_logger",
]
