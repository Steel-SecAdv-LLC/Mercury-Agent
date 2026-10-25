# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CI performance gate: thresholds, the JSON record, and the failure path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "performance_benchmark.py"


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("performance_benchmark", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing(bench: ModuleType) -> dict[str, object]:
    return {
        "ttl_cache_write_ms": 1.0,
        "ttl_cache_read_ms": 1.0,
        "synthetic_gradient_ms": 1.0,
        "thresholds_ms": dict(bench.THRESHOLDS_MS),
    }


def test_thresholds_are_the_documented_budgets(bench: ModuleType) -> None:
    assert bench.THRESHOLDS_MS == {"ttl_write": 500.0, "ttl_read": 100.0, "gradient": 1000.0}


def test_passing_measurements_are_accepted(bench: ModuleType) -> None:
    bench.check_thresholds(_passing(bench))


@pytest.mark.parametrize(
    ("key", "threshold", "message"),
    [
        ("ttl_cache_write_ms", "ttl_write", "TTLCache write too slow"),
        ("ttl_cache_read_ms", "ttl_read", "TTLCache read too slow"),
        ("synthetic_gradient_ms", "gradient", "Gradient prediction too slow"),
    ],
)
def test_each_budget_is_enforced_at_its_boundary(
    bench: ModuleType, key: str, threshold: str, message: str
) -> None:
    results = _passing(bench)
    results[key] = bench.THRESHOLDS_MS[threshold]  # equal to the budget is over it
    with pytest.raises(bench.BenchmarkFailure, match=message):
        bench.check_thresholds(results)


def test_main_writes_the_record_and_exits_zero(bench: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "benchmark-results.json"
    assert bench.main(["--output", str(out)]) == 0
    record = json.loads(out.read_text())
    assert set(record) >= {
        "ttl_cache_write_ms",
        "ttl_cache_read_ms",
        "ttl_cache_hit_rate",
        "synthetic_gradient_ms",
        "gradient_cache_hit_rate",
        "thresholds_ms",
    }
    assert record["thresholds_ms"] == bench.THRESHOLDS_MS


def test_failing_run_still_leaves_the_record(
    bench: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upload step needs the file even when a budget trips."""
    slow = _passing(bench) | {"synthetic_gradient_ms": 1e9}
    monkeypatch.setattr(bench, "run_benchmarks", lambda: slow)
    out = tmp_path / "benchmark-results.json"
    assert bench.main(["--output", str(out)]) == 1
    assert json.loads(out.read_text())["synthetic_gradient_ms"] == 1e9
