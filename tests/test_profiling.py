"""Tests for :mod:`omni_mercury_engine.utils.profiling`.

These tests exercise the six public profiling entry points end-to-end
(``profile_func``, ``profile_memory``, ``profile_time``,
``profile_time_async``, ``profile_complete``, :class:`PerformanceBenchmark`
and :func:`benchmark_function`) plus the module-level enable/disable
plumbing and logger configuration.  No synthetic data is fabricated --
all tests drive the real profiling stack (cProfile, tracemalloc,
``time.perf_counter``) on small in-process workloads and assert against
real measured outputs.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
Released under GPL-3.0+.
"""

from __future__ import annotations

import asyncio
import logging
import tracemalloc
from typing import Any

import pytest

from omni_mercury_engine.utils import profiling
from omni_mercury_engine.utils.profiling import (
    PerformanceBenchmark,
    benchmark_function,
    get_profiling_logger,
    is_profiling_enabled,
    profile_complete,
    profile_func,
    profile_memory,
    profile_time,
    profile_time_async,
    set_profiling_enabled,
    set_profiling_logger,
)


@pytest.fixture(autouse=True)
def _restore_profiling_state() -> Any:
    """Restore the module's global flags around each test."""

    original_enabled = is_profiling_enabled()
    original_logger = profiling._logger  # type: ignore[attr-defined]
    try:
        yield
    finally:
        set_profiling_enabled(original_enabled)
        profiling._logger = original_logger  # type: ignore[attr-defined]
        # Always leave tracemalloc disabled at the end of the test so
        # subsequent tests start from a clean baseline.
        if tracemalloc.is_tracing():
            tracemalloc.stop()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_set_and_query_global_enabled() -> None:
    set_profiling_enabled(False)
    assert is_profiling_enabled() is False
    set_profiling_enabled(True)
    assert is_profiling_enabled() is True


def test_set_and_get_profiling_logger() -> None:
    new_logger = logging.getLogger("mercury.test.profiling")
    set_profiling_logger(new_logger)
    assert get_profiling_logger() is new_logger


def test_lazy_logger_namespace() -> None:
    # Forcing the lazy path: clear the cached logger and re-fetch.
    profiling._logger = None  # type: ignore[attr-defined]
    logger = get_profiling_logger()
    assert logger.name == "omni_mercury_engine.profiling"


# ---------------------------------------------------------------------------
# profile_func
# ---------------------------------------------------------------------------


def test_profile_func_returns_value_and_logs_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_func(top_n=3, sort_by="cumtime")
    def sum_squares(n: int) -> int:
        return sum(i * i for i in range(n))

    result = sum_squares(50)
    assert result == sum(i * i for i in range(50))
    assert any(
        "CPU Profile for" in r.message and "sum_squares" in r.message for r in caplog.records
    )


def test_profile_func_rejects_invalid_top_n() -> None:
    with pytest.raises(ValueError, match="top_n must be positive"):
        profile_func(top_n=0)


def test_profile_func_rejects_invalid_sort_key() -> None:
    with pytest.raises(ValueError, match="Invalid pstats sort key"):
        profile_func(sort_by="nope")


def test_profile_func_disabled_globally_is_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_profiling_enabled(False)
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_func()
    def identity(value: int) -> int:
        return value

    assert identity(7) == 7
    assert not any("CPU Profile" in r.message and "identity" in r.message for r in caplog.records)


def test_profile_func_per_decorator_override_takes_effect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_profiling_enabled(False)
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_func(enabled=True)
    def adder(a: int, b: int) -> int:
        return a + b

    assert adder(2, 3) == 5
    assert any("CPU Profile for" in r.message and "adder" in r.message for r in caplog.records)


def test_profile_func_propagates_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_func()
    def boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        boom()
    # The profiler must still emit a record despite the exception.
    assert any("CPU Profile for" in r.message and "boom" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# profile_memory
# ---------------------------------------------------------------------------


def test_profile_memory_logs_current_and_peak(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_memory()
    def allocate(n: int) -> list[int]:
        return list(range(n))

    result = allocate(1024)
    assert len(result) == 1024
    assert any(
        "Memory Profile for" in r.message and "allocate" in r.message and "Peak=" in r.message
        for r in caplog.records
    )


def test_profile_memory_nested_does_not_break_outer_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_memory()
    def inner() -> int:
        return sum(range(100))

    @profile_memory()
    def outer() -> int:
        return inner() + inner()

    result = outer()
    assert result == 2 * sum(range(100))
    # tracemalloc should be stopped again after the outermost decorator exits.
    assert not tracemalloc.is_tracing()


def test_profile_memory_disabled_is_noop() -> None:
    set_profiling_enabled(False)

    @profile_memory()
    def identity(x: int) -> int:
        return x

    assert identity(42) == 42
    assert not tracemalloc.is_tracing()


def test_profile_memory_exception_path_cleans_up() -> None:
    @profile_memory()
    def kaboom() -> None:
        raise ValueError("nope")

    assert not tracemalloc.is_tracing()
    with pytest.raises(ValueError):
        kaboom()
    assert not tracemalloc.is_tracing()


# ---------------------------------------------------------------------------
# profile_time
# ---------------------------------------------------------------------------


def test_profile_time_logs_elapsed_ms(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_time()
    def short_op() -> int:
        return 1

    assert short_op() == 1
    records = [
        r for r in caplog.records if "Execution time for" in r.message and "short_op" in r.message
    ]
    assert records, "profile_time did not emit a log record"
    assert " ms" in records[-1].message


def test_profile_time_log_args_includes_arg_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_time(log_args=True)
    def add(a: int, b: int, *, c: int = 0) -> int:
        return a + b + c

    assert add(1, 2, c=3) == 6
    records = [
        r for r in caplog.records if "Execution time for" in r.message and ".add" in r.message
    ]
    assert records
    assert "args=" in records[-1].message and "kwargs=" in records[-1].message


def test_profile_time_disabled_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")
    set_profiling_enabled(False)

    @profile_time()
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 2) == 4
    assert not any("Execution time" in r.message for r in caplog.records)


def test_profile_time_propagates_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_time()
    def bad() -> None:
        raise RuntimeError("bad")

    with pytest.raises(RuntimeError):
        bad()
    assert any("Execution time for" in r.message and "bad" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# profile_time_async
# ---------------------------------------------------------------------------


async def test_profile_time_async_logs_elapsed_ms(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_time_async()
    async def sleeper() -> int:
        await asyncio.sleep(0.001)
        return 7

    result = await sleeper()
    assert result == 7
    assert any("Execution time for" in r.message and "sleeper" in r.message for r in caplog.records)


def test_profile_time_async_rejects_non_coroutine() -> None:
    with pytest.raises(TypeError, match="coroutine function"):

        @profile_time_async()
        def not_async() -> int:  # pragma: no cover - decorator raises immediately
            return 0


async def test_profile_time_async_disabled_is_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")
    set_profiling_enabled(False)

    @profile_time_async()
    async def coro() -> int:
        return 9

    assert await coro() == 9
    assert not any("Execution time" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# profile_complete
# ---------------------------------------------------------------------------


def test_profile_complete_logs_all_three_dimensions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    @profile_complete(top_n=2)
    def workload(n: int) -> int:
        return sum(i * i for i in range(n))

    workload(64)
    payload = "\n".join(r.message for r in caplog.records)
    assert "Complete Profile for" in payload and "workload" in payload
    assert "Execution Time:" in payload
    assert "Memory:" in payload
    assert "CPU Profile (top 2)" in payload


def test_profile_complete_exception_cleanup() -> None:
    @profile_complete(top_n=1)
    def bomb() -> None:
        raise KeyError("zap")

    with pytest.raises(KeyError):
        bomb()
    assert not tracemalloc.is_tracing()


def test_profile_complete_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        profile_complete(top_n=-1)
    with pytest.raises(ValueError):
        profile_complete(sort_by="bogus")


def test_profile_complete_disabled_is_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")
    set_profiling_enabled(False)

    @profile_complete()
    def fast() -> int:
        return 11

    assert fast() == 11
    assert not any("Complete Profile" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# PerformanceBenchmark
# ---------------------------------------------------------------------------


def test_performance_benchmark_records_measurements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="omni_mercury_engine.profiling")

    with PerformanceBenchmark("hot-path") as bench:
        _ = [i * i for i in range(1_000)]

    assert bench.elapsed_ms > 0.0
    assert bench.peak_mb >= bench.current_mb >= 0.0
    assert any("Benchmark 'hot-path'" in r.message for r in caplog.records)


def test_performance_benchmark_disabled_path() -> None:
    set_profiling_enabled(False)
    with PerformanceBenchmark("noop") as bench:
        pass
    assert bench.elapsed_ms == 0.0
    assert bench.current_mb == 0.0


def test_performance_benchmark_nested_does_not_stop_outer_trace() -> None:
    with PerformanceBenchmark("outer") as outer_bench:
        with PerformanceBenchmark("inner") as inner_bench:
            _ = list(range(100))
        # tracemalloc must still be running for the outer benchmark to
        # capture its own peak measurement.
        assert tracemalloc.is_tracing()

    assert inner_bench.elapsed_ms >= 0.0
    assert outer_bench.elapsed_ms >= 0.0
    assert not tracemalloc.is_tracing()


def test_performance_benchmark_propagates_exceptions() -> None:
    with pytest.raises(ValueError), PerformanceBenchmark("err"):
        raise ValueError("explode")
    assert not tracemalloc.is_tracing()


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_returns_summary_stats() -> None:
    def workload(value: int) -> int:
        return value * 2

    stats = benchmark_function(workload, 21, iterations=20, warmup=5)
    assert stats["iterations"] == 20
    assert stats["warmup"] == 5
    assert stats["mean_ms"] >= 0.0
    assert stats["min_ms"] <= stats["median_ms"] <= stats["max_ms"]
    assert stats["std_ms"] >= 0.0


def test_benchmark_function_invalid_iterations() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        benchmark_function(lambda: None, iterations=0)


def test_benchmark_function_invalid_warmup() -> None:
    with pytest.raises(ValueError, match="warmup must be non-negative"):
        benchmark_function(lambda: None, iterations=1, warmup=-1)


def test_benchmark_function_forwards_kwargs() -> None:
    captured: dict[str, Any] = {}

    def workload(value: int, *, scale: int = 1) -> int:
        captured["last"] = value * scale
        return captured["last"]

    stats = benchmark_function(workload, 5, iterations=3, warmup=0, scale=4)
    assert captured["last"] == 20
    assert stats["iterations"] == 3
