# Performance Profiling

Applies to Mercury Agent **v1.7.x**. Last updated: 2026-05-20.

`omni_mercury_engine.utils.profiling` is the first-party profiling
toolkit ported from Omni-AXA-Engine and hardened for Mercury Agent's
asyncio paths and `mypy --strict` configuration. It provides six
public decorators, one context manager, and one benchmarking helper
for measuring CPU time, memory consumption, and wall-clock execution.

**Locking tests:** `tests/test_profiling.py` (32 unit tests).

> **All entry points are no-ops when profiling is globally disabled.**
> The default is **disabled**; enable explicitly via
> `set_profiling_enabled(True)`. This keeps Mercury's production hot
> paths instrumentation-free unless an operator opts in.

---

## Public surface

| Symbol | Kind | Measures |
|--------|------|----------|
| `set_profiling_enabled(enabled: bool)` | toggle | Global on/off (default off) |
| `is_profiling_enabled()` | query | Current state |
| `set_profiling_logger(logger)` | injection | Override the default logger |
| `get_profiling_logger()` | query | Current logger |
| `@profile_func` | decorator | Full `cProfile` stats (top N callers) |
| `@profile_memory` | decorator | `tracemalloc` peak / current bytes |
| `@profile_time` | decorator | Wall-clock seconds (sync functions) |
| `@profile_time_async` | decorator | Wall-clock seconds (`async def` functions) — **Mercury addition** |
| `@profile_complete` | decorator | All three of the above stacked |
| `PerformanceBenchmark` | context manager | Scoped timing block |
| `benchmark_function(fn, *args, iterations=...)` | function | Repeat-call timing statistics |

### Mercury-specific changes from upstream

- Logger namespace switched to `omni_mercury_engine.profiling` via
  `omni_mercury_engine.utils.logging.get_logger`.
- `tracemalloc` is treated as always-available (mandatory stdlib in
  Python 3.11+, Mercury's minimum supported runtime); the upstream's
  `TRACEMALLOC_AVAILABLE` import guard was dead code on supported
  interpreters and has been removed.
- Public surface fully typed for `mypy --strict`; `Any`-typed
  decorator returns replaced with explicit
  `Callable[P, R]` / `Callable[P, Awaitable[R]]` using PEP 612
  ParamSpecs.
- Nested `tracemalloc.start()` / `tracemalloc.stop()` calls are
  reference-counted by the stdlib, but a single Mercury invocation
  that wraps itself in two profiling decorators would previously
  attribute the inner allocation totals to the outer call. The port
  records the tracemalloc state on entry and only stops tracing if
  it was started by the same decorator instance.
- Exception-path cleanup in `profile_func` was hardened so
  `profiler.disable()` runs even when stats emission raises.
- Added `@profile_time_async` for Mercury's asyncio paths.
- Added the opt-in global enable flag exposed via
  `set_profiling_enabled(...)` / `is_profiling_enabled()`.

---

## Usage

### Decorators

```python
from omni_mercury_engine.utils.profiling import (
    set_profiling_enabled,
    profile_time, profile_time_async,
    profile_memory, profile_func, profile_complete,
)

set_profiling_enabled(True)

@profile_time()
def expensive_op(x: int) -> int:
    return sum(range(x))

@profile_time_async()
async def expensive_async_op(x: int) -> int:
    await asyncio.sleep(0)
    return sum(range(x))

@profile_memory()
def allocates_lots(n: int) -> list[int]:
    return list(range(n))

@profile_func(top_n=20, sort_by="cumulative")
def cpu_bound(x: int) -> int:
    ...

@profile_complete()  # time + memory + cProfile in one shot
def fully_instrumented() -> None:
    ...
```

Each decorator emits a single structured log line per call when
profiling is enabled. Override the logger with
`set_profiling_logger(...)` to route the output (e.g. to a Prometheus
push-gateway adapter, a JSONL sink, or a structured-logging pipeline).

### `PerformanceBenchmark` context manager

```python
from omni_mercury_engine.utils.profiling import PerformanceBenchmark

with PerformanceBenchmark("oracle_pipeline_warmup") as bench:
    detector.fit(training_data)
    detector.detect(validation_data)

print(bench.elapsed_seconds, bench.peak_memory_bytes)
```

The context manager records start/stop wall-clock and peak
`tracemalloc` allocation for the block. Useful for scoped timing of
multi-step pipelines without decorating every helper function.

### `benchmark_function`

```python
from omni_mercury_engine.utils.profiling import benchmark_function

stats = benchmark_function(
    expensive_op,
    10_000,                  # positional args to expensive_op
    iterations=200,          # how many times to call it
    warmup_iterations=10,    # discarded JIT-warmup calls
)

print(stats["mean_ms"], stats["std_ms"], stats["p99_ms"])
```

Returns a dict with `mean_ms`, `median_ms`, `std_ms`, `min_ms`,
`max_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `total_seconds`,
`iterations`, and `warmup_iterations`.

---

## Integration with Mercury's benchmark pipeline

The profiling toolkit is **not** wired into the benchmark gates by
default. `benchmarks/mercury_benchmark.py` records its own per-dataset
timing into `mercury_benchmark_results.json`. The profiling toolkit is
intended for ad-hoc investigations (e.g. tracking down a hotspot in
the Oracle pipeline) where `cProfile`-grade detail or per-call memory
attribution is required.

To enable profiling for a one-off benchmark run, prepend
`MERCURY_PROFILING=1` and call `set_profiling_enabled(True)` in the
relevant entry point — there is no environment-variable side-effect
inside the module, by design.

---

## See also

- [`docs/API_REFERENCE.md`](API_REFERENCE.md) — quick-import index.
- `tests/test_profiling.py` — the 32-test regression suite that pins
  the public surface.
- `benchmarks/mercury_benchmark.py` — the benchmark harness (separate
  timing pipeline, not built on this toolkit).
