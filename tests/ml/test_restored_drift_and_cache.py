# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restored ``ml.drift.OnlineDriftDetector`` and ``ml.optimization`` helpers.

Removed in 7ab59916 as uncalled. The released drift detector ran an
uncorrected KS test per feature on every update (12-31 false drifts per 2000
stationary samples), and the released cache keyed calls by ``str(args)``,
which numpy abbreviates past 1000 elements, so different arrays shared a key.
The tests marked as such fail on the released code.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.ml.drift import OnlineDriftDetector
from omni_mercury_engine.ml.optimization import (
    get_optimal_batch_size,
    memory_efficient_lru_cache,
    parallel_map,
)


def _stream(seed: int, n: int = 2000, shift_at: int | None = None) -> np.ndarray:
    data = np.random.default_rng(seed).normal(0.0, 1.0, (n, 3))
    if shift_at is not None:
        data[shift_at:, 1] += 1.0  # a 1-sigma mean shift in one feature
    return data


@pytest.mark.parametrize("seed", range(5))
def test_stationary_stream_raises_no_drift(seed: int) -> None:
    """Fails on the released KS detector (12-31 false drifts per stream)."""
    detector = OnlineDriftDetector()
    drifts = sum(detector.update(row) is not None for row in _stream(seed))
    assert drifts == 0


@pytest.mark.parametrize("seed", range(5))
def test_one_sigma_shift_is_detected_promptly(seed: int) -> None:
    detector = OnlineDriftDetector()
    data = _stream(seed, n=1400, shift_at=1000)
    first = next((i, r) for i, row in enumerate(data) if (r := detector.update(row)) is not None)
    index, result = first
    assert 1000 <= index < 1100  # 20-72 samples measured over 20 seeds
    assert result.details["feature"] == 1
    assert result.details["method"] == "adwin0"
    assert result.test_statistic > result.threshold
    assert len(detector.window) < 1000  # the pre-change part was dropped


def test_reset_clears_state() -> None:
    detector = OnlineDriftDetector()
    for row in _stream(0, n=1200, shift_at=600):
        detector.update(row)
    assert detector.drift_count >= 1
    detector.reset()
    assert (detector.window, detector.drift_detected, detector.drift_count) == ([], False, 0)


def test_cache_distinguishes_large_arrays() -> None:
    """Fails on the released ``str(args)`` key, which served the first sum twice."""

    @memory_efficient_lru_cache()
    def total(x: np.ndarray) -> float:
        return float(np.sum(x))

    a = np.arange(5000.0)
    b = a.copy()
    b[2500] += 1000.0
    assert total(a) == float(a.sum())
    assert total(b) == float(b.sum())
    assert total(a) == float(a.sum())
    assert total.cache_stats()["hits"] == 1


def test_cache_keeps_a_none_result_and_types_scalars() -> None:
    calls: list[object] = []

    @memory_efficient_lru_cache()
    def record(x: object) -> None:
        calls.append(x)

    record(1)
    record(1)
    record(1.0)
    record(True)
    assert calls == [1, 1.0, True]


def test_cache_bypasses_arguments_without_an_exact_key() -> None:
    calls: list[int] = []

    class Model:
        pass

    @memory_efficient_lru_cache()
    def score(model: Model, k: int = 1) -> int:
        calls.append(k)
        return k

    score(Model())
    score(Model())
    assert calls == [1, 1]
    assert score.cache_stats()["entries"] == 0
    score.cache_clear()


def test_parallel_map_keeps_input_order() -> None:
    assert parallel_map(abs, [-3, 1, -2], n_jobs=2) == [3, 1, 2]


def test_optimal_batch_size_defaults_without_cuda() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU default only")
    assert get_optimal_batch_size(torch.nn.Linear(4, 2), (4,)) == 32
