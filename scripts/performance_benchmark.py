#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Performance gate for the cache and gradient-predictor hot paths.

Measures the TTL cache and the synthetic-gradient predictor, writes the
measurements to a JSON file, then asserts each against its threshold. The
file is written first and the assertions run after, so the CI upload step
(``if: always()``) always has an artifact to attach, even when a threshold
trips.

Usage::

    python scripts/performance_benchmark.py --output benchmark-results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

#: Per-operation time budgets in milliseconds. The single source for both the
#: recorded JSON and the assertions, so the two cannot drift apart.
THRESHOLDS_MS: dict[str, float] = {
    "ttl_write": 500.0,
    "ttl_read": 100.0,
    "gradient": 1000.0,
}

TTL_OPERATIONS = 1000
GRADIENT_OPERATIONS = 100


class BenchmarkFailure(AssertionError):
    """A measured operation exceeded its time budget."""


def measure_ttl_cache() -> dict[str, float]:
    """Time writes and reads on a TTL cache over a repeating key set."""
    from omni_mercury_engine.core.gosnn_integration import TTLCache

    cache = TTLCache(max_size=1000)
    data = np.random.randn(100, 64)

    start = time.perf_counter()
    for i in range(TTL_OPERATIONS):
        cache.set(data[i % 100], {"result": i})
    write_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    hits = 0
    for i in range(TTL_OPERATIONS):
        if cache.get(data[i % 100]) is not None:
            hits += 1
    read_ms = (time.perf_counter() - start) * 1000

    return {
        "ttl_cache_write_ms": write_ms,
        "ttl_cache_read_ms": read_ms,
        "ttl_cache_hit_rate": hits / TTL_OPERATIONS,
    }


def measure_gradient_predictor() -> dict[str, float]:
    """Time repeated cached forward passes of the synthetic-gradient predictor."""
    from omni_mercury_engine.ml.optimizers import SyntheticGradientPredictor, get_gradient_cache

    predictor = SyntheticGradientPredictor(input_dim=128, hidden_dim=64)
    get_gradient_cache().clear()
    activations = np.random.randn(32, 128).astype(np.float32)

    start = time.perf_counter()
    for _ in range(GRADIENT_OPERATIONS):
        predictor.forward(activations, use_cache=True)
    gradient_ms = (time.perf_counter() - start) * 1000

    return {
        "synthetic_gradient_ms": gradient_ms,
        "gradient_cache_hit_rate": float(get_gradient_cache().stats["hit_rate"]),
    }


def run_benchmarks() -> dict[str, Any]:
    """Collect every measurement plus the thresholds they were judged against."""
    results: dict[str, Any] = {}
    results.update(measure_ttl_cache())
    results.update(measure_gradient_predictor())
    results["thresholds_ms"] = dict(THRESHOLDS_MS)
    return results


def check_thresholds(results: dict[str, Any]) -> None:
    """Raise :class:`BenchmarkFailure` for the first measurement over budget."""
    checks = (
        ("ttl_cache_write_ms", "ttl_write", "TTLCache write too slow"),
        ("ttl_cache_read_ms", "ttl_read", "TTLCache read too slow"),
        ("synthetic_gradient_ms", "gradient", "Gradient prediction too slow"),
    )
    for key, threshold_key, message in checks:
        measured = float(results[key])
        budget = float(results["thresholds_ms"][threshold_key])
        if measured >= budget:
            raise BenchmarkFailure(f"{message}: {measured:.2f}ms (budget {budget:.0f}ms)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="JSON results path")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("MERCURY AGENT PERFORMANCE BENCHMARK")
    print("=" * 60)
    results = run_benchmarks()
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for key in sorted(k for k in results if k != "thresholds_ms"):
        print(f"{key}: {results[key]}")

    try:
        check_thresholds(results)
    except BenchmarkFailure as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print("=" * 60)
    print("ALL PERFORMANCE BENCHMARKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
