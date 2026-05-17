"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from omni_mercury_engine.utils import profiling
from omni_mercury_engine.utils.profiling import (
    PerformanceBenchmark,
    benchmark_function,
    is_profiling_enabled,
    profile_complete,
    profile_func,
    profile_memory,
    profile_time,
    set_profiling_enabled,
)

# ---------------------------------------------------------------------------
# Deleted-API contract
# ---------------------------------------------------------------------------


def test_deleted_global_setter_apis() -> None:
    """``set_profiling_logger`` / ``get_profiling_logger`` were removed by spec."""
    assert not hasattr(profiling, "set_profiling_logger")
    assert not hasattr(profiling, "get_profiling_logger")


def test_module_uses_dunder_name_logger() -> None:
    """The module-level ``_logger`` must be bound to ``logging.getLogger(__name__)``."""
    expected = logging.getLogger("omni_mercury_engine.utils.profiling")
    assert profiling._logger is expected


# ---------------------------------------------------------------------------
# Enable / disable toggle
# ---------------------------------------------------------------------------


def test_enable_disable_toggle() -> None:
    """``set_profiling_enabled`` flips the global flag in both directions."""
    set_profiling_enabled(False)
    try:
        assert is_profiling_enabled() is False
        set_profiling_enabled(True)
        assert is_profiling_enabled() is True
    finally:
        set_profiling_enabled(True)


def test_decorators_pass_through_when_disabled(caplog: pytest.LogCaptureFixture) -> None:
    """When profiling is globally off, every decorator must be a no-op pass-through."""
    set_profiling_enabled(False)
    try:
        caplog.set_level(logging.INFO, logger="omni_mercury_engine.utils.profiling")

        @profile_func()
        def f1(x: int) -> int:
            return x + 1

        @profile_memory()
        def f2(x: int) -> int:
            return x * 2

        @profile_time()
        def f3(x: int) -> int:
            return x - 1

        @profile_complete()
        def f4(x: int) -> int:
            return x // 2

        assert f1(10) == 11
        assert f2(10) == 20
        assert f3(10) == 9
        assert f4(10) == 5
        assert caplog.records == []
    finally:
        set_profiling_enabled(True)


# ---------------------------------------------------------------------------
# CPU profile decorator
# ---------------------------------------------------------------------------


def test_profile_func_returns_value_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    """``profile_func`` must return the wrapped value and emit a CPU profile log line."""
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.utils.profiling")

    @profile_func(top_n=3)
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert any("CPU profile for add" in r.getMessage() for r in caplog.records)


def test_profile_func_local_enable_override(caplog: pytest.LogCaptureFixture) -> None:
    """Per-decorator ``enabled=False`` must override the global flag."""
    set_profiling_enabled(True)
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.utils.profiling")

    @profile_func(enabled=False)
    def f(x: int) -> int:
        return x

    f(42)
    assert caplog.records == []


# ---------------------------------------------------------------------------
# Memory decorator
# ---------------------------------------------------------------------------


def test_profile_memory_logs_peak(caplog: pytest.LogCaptureFixture) -> None:
    """``profile_memory`` must emit current / peak memory in the log line."""
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.utils.profiling")

    @profile_memory()
    def allocate() -> list[int]:
        return list(range(100_000))

    result = allocate()
    assert len(result) == 100_000
    messages = [r.getMessage() for r in caplog.records]
    assert any("Memory profile for allocate" in m and "peak=" in m for m in messages)


def test_profile_memory_stops_tracemalloc_on_exception() -> None:
    """``tracemalloc`` must be stopped even when the wrapped function raises."""
    import tracemalloc

    @profile_memory()
    def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        boom()
    assert not tracemalloc.is_tracing()


# ---------------------------------------------------------------------------
# Time decorator
# ---------------------------------------------------------------------------


def test_profile_time_logs_elapsed(caplog: pytest.LogCaptureFixture) -> None:
    """``profile_time`` must log an ``Execution time`` line in milliseconds."""
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.utils.profiling")

    @profile_time()
    def sleeper() -> None:
        time.sleep(0.005)

    sleeper()
    messages = [r.getMessage() for r in caplog.records]
    assert any("Execution time for sleeper" in m and "ms" in m for m in messages)


def test_profile_time_log_args_includes_argv(caplog: pytest.LogCaptureFixture) -> None:
    """When ``log_args=True`` the log line includes a truncated args snippet."""
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.utils.profiling")

    @profile_time(log_args=True)
    def f(*args: Any, **kwargs: Any) -> int:
        return len(args)

    f(1, 2, "secret-token", extra=99)
    messages = [r.getMessage() for r in caplog.records]
    joined = "\n".join(messages)
    assert "args=" in joined
    assert "kwargs=" in joined


# ---------------------------------------------------------------------------
# Complete (combined) decorator
# ---------------------------------------------------------------------------


def test_profile_complete_logs_all_three_dimensions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``profile_complete`` must report time, current memory, and peak memory."""
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.utils.profiling")

    @profile_complete(top_n=2)
    def worker() -> int:
        time.sleep(0.001)
        return sum(range(1000))

    assert worker() == sum(range(1000))
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "Complete profile for worker" in joined
    assert "time=" in joined
    assert "peak_mem=" in joined


def test_profile_complete_stops_tracemalloc_on_exception() -> None:
    """``profile_complete`` must clean up tracemalloc and re-raise on error."""
    import tracemalloc

    @profile_complete()
    def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        boom()
    assert not tracemalloc.is_tracing()


# ---------------------------------------------------------------------------
# Benchmark context manager
# ---------------------------------------------------------------------------


def test_performance_benchmark_captures_elapsed_and_peak() -> None:
    """The context manager exposes elapsed_ms and peak_mem_bytes after exit."""
    with PerformanceBenchmark("ctx-test") as bench:
        # Allocate something measurable so tracemalloc captures it.
        _data = bytearray(50_000)
        time.sleep(0.001)

    assert bench.elapsed_ms > 0
    assert bench.peak_mem_bytes >= 0


def test_performance_benchmark_noop_when_disabled() -> None:
    """When disabled, the context manager must not start tracemalloc."""
    import tracemalloc

    with PerformanceBenchmark("disabled", enabled=False) as bench:
        pass
    assert bench.elapsed_ms == 0
    assert not tracemalloc.is_tracing()


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_returns_expected_keys() -> None:
    """``benchmark_function`` returns a stats dict with the documented keys."""

    def cheap() -> int:
        return 1 + 1

    stats = benchmark_function(cheap, iterations=20, warmup=2)
    assert set(stats.keys()) == {
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "median_ms",
        "iterations",
        "warmup",
    }
    assert stats["iterations"] == 20
    assert stats["warmup"] == 2
    assert stats["mean_ms"] >= 0
    assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]


def test_benchmark_function_passes_args_through() -> None:
    """The benchmark loop forwards positional and keyword arguments to the callable."""
    seen: list[tuple[Any, ...]] = []

    def record(*args: Any, **kwargs: Any) -> None:
        seen.append((args, tuple(sorted(kwargs.items()))))

    benchmark_function(record, 1, 2, iterations=3, warmup=1, extra="z")
    # 1 warmup + 3 iterations = 4 invocations
    assert len(seen) == 4
    for args, kwargs in seen:
        assert args == (1, 2)
        assert kwargs == (("extra", "z"),)


def test_benchmark_function_rejects_nonpositive_iterations() -> None:
    """Zero or negative iterations is a programmer error and must raise."""
    with pytest.raises(ValueError):
        benchmark_function(lambda: None, iterations=0)
    with pytest.raises(ValueError):
        benchmark_function(lambda: None, iterations=-1)


def test_benchmark_function_rejects_negative_warmup() -> None:
    """Negative warmup is a programmer error and must raise."""
    with pytest.raises(ValueError):
        benchmark_function(lambda: None, iterations=1, warmup=-1)
