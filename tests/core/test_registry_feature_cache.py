# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in registry feature cache: correctness guards and metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.detector_registry import DetectorRegistry
from omni_mercury_engine.security.realtime_threat_detection import RealTimeThreatDetector
from omni_mercury_engine.utils.feature_cache import (
    CacheConfig,
    MemoryEfficientFeatureCache,
    QuantizationType,
)


class _Counting:
    """Pure detector with a cache token that counts real computations."""

    def __init__(self) -> None:
        self.calls = 0
        self.version = 1

    def feature_cache_token(self) -> tuple[str, int]:
        return ("counting", self.version)

    def extract_features(self, data: Any) -> np.ndarray:
        self.calls += 1
        # np.asarray: ``.sum()`` types as Any under the Python-3.11 numpy
        # stubs, failing the tests mypy lane with [no-any-return].
        return np.asarray(np.asarray(data, dtype=float).sum(axis=1, keepdims=True) * self.version)

    def predict(self, data: Any) -> dict[str, Any]:
        return {"scores": np.asarray(data, dtype=float).mean(axis=1)}


class _Stateful(_Counting):
    """No token: must never be cached."""

    feature_cache_token = None  # type: ignore[assignment]


def _metric(name: str, detector: str) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(name, {"detector_name": detector})
    return float(value or 0.0)


def test_cache_off_by_default() -> None:
    reg = DetectorRegistry()
    det = _Counting()
    reg.register("c", det)
    x = np.ones((4, 3))
    reg.extract_features("c", x)
    reg.extract_features("c", x)
    assert det.calls == 2
    assert reg.feature_cache_stats() is None


def test_repeated_payload_hits_and_records_metrics() -> None:
    reg = DetectorRegistry()
    reg.enable_feature_cache()
    det = _Counting()
    reg.register("cache_probe", det)
    x = np.arange(12.0).reshape(4, 3)
    hits0 = _metric("omni_feature_cache_hits_total", "cache_probe")
    misses0 = _metric("omni_feature_cache_misses_total", "cache_probe")
    first = reg.extract_features("cache_probe", x)
    second = reg.extract_features("cache_probe", x)
    assert det.calls == 1
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.scores, second.scores)
    assert _metric("omni_feature_cache_hits_total", "cache_probe") == hits0 + 1
    assert _metric("omni_feature_cache_misses_total", "cache_probe") == misses0 + 1


def test_token_change_and_different_data_miss() -> None:
    reg = DetectorRegistry()
    reg.enable_feature_cache()
    det = _Counting()
    reg.register("c", det)
    x = np.ones((4, 3))
    reg.extract_features("c", x)
    det.version = 2  # e.g. a refit
    refreshed = reg.extract_features("c", x)
    assert det.calls == 2
    assert float(np.asarray(refreshed.features)[0, 0]) == 6.0
    reg.extract_features("c", x + 1.0)
    reg.extract_features("c", x.astype(np.float32))  # same values, other dtype
    assert det.calls == 4


def test_stateful_detectors_and_dict_payloads_are_never_cached() -> None:
    reg = DetectorRegistry()
    reg.enable_feature_cache()
    stateful = _Stateful()
    reg.register("s", stateful)
    x = np.ones((2, 2))
    reg.extract_features("s", x)
    reg.extract_features("s", x)
    assert stateful.calls == 2
    assert reg._feature_cache_key("c", _Counting(), {"a": 1}) is None


def test_refit_of_real_detector_invalidates() -> None:
    rng = np.random.default_rng(0)
    reg = DetectorRegistry()
    reg.enable_feature_cache()
    det = RealTimeThreatDetector().fit(rng.normal(size=(300, 3)))
    reg.register("rt", det)
    batch = rng.normal(size=(20, 3))
    before = reg.extract_features("rt", batch)
    det.fit(rng.normal(2.0, 1.0, (300, 3)))
    after = reg.extract_features("rt", batch)
    assert before.scores is not None and after.scores is not None
    assert not np.array_equal(before.scores, after.scores)


def test_tensor_features_round_trip_as_tensors() -> None:
    torch = pytest.importorskip("torch")
    from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector

    rng = np.random.default_rng(1)
    reg = DetectorRegistry()
    reg.enable_feature_cache()
    reg.register("statistical", StatisticalAnomalyDetector().fit(rng.normal(size=(300, 4))))
    batch = rng.normal(size=(30, 4))
    first = reg.extract_features("statistical", batch)
    second = reg.extract_features("statistical", batch)
    assert isinstance(second.features, torch.Tensor)
    assert torch.equal(first.features, second.features)


def test_entry_larger_than_budget_is_refused_without_evicting() -> None:
    cache = MemoryEfficientFeatureCache(CacheConfig(max_memory_mb=0.001))
    small = np.zeros(10)
    assert cache.put("small", small)
    assert not cache.put("huge", np.zeros(100_000))
    assert cache.contains("small")


def test_int8_reconstruction_is_within_one_quantum() -> None:
    cache = MemoryEfficientFeatureCache(CacheConfig(quantization=QuantizationType.INT8))
    data = np.linspace(-1.0, 1.0, 50)
    cache.put("k", data)
    out = cache.get("k")
    assert out is not None
    np.testing.assert_allclose(out, data, atol=2.0 / 255)


class TestCacheCoherency:
    """A hit must carry both features and scores, or be a miss.

    Regression: the |f and |s entries evict independently, so a hit on
    features after the scores entry was evicted silently returned
    scores=None for a detector that does produce scores; and the
    tensor-provenance side dict grew one key per distinct payload forever.
    """

    def test_evicted_scores_entry_degrades_to_full_miss(self) -> None:
        reg = DetectorRegistry()
        reg.enable_feature_cache()
        det = _Counting()
        reg.register("coherent", det)
        x = np.ones((4, 3))

        first = reg.extract_features("coherent", x)
        assert first.scores is not None
        key = reg._feature_cache_key("coherent", det, x)
        assert key is not None
        assert reg._cached_result(key) is not None

        # Simulate the scores companion being evicted independently.
        reg._feature_cache._remove_entry(f"{key}|s")
        assert (
            reg._cached_result(key) is None
        ), "features-only hit must degrade to a miss, not drop the scores"
        refreshed = reg.extract_features("coherent", x)
        assert det.calls == 2  # honest recompute
        assert refreshed.scores is not None

    def test_no_unbounded_side_state(self) -> None:
        reg = DetectorRegistry()
        reg.enable_feature_cache(max_entries=4)
        det = _Counting()
        reg.register("bounded", det)
        for i in range(32):
            reg.extract_features("bounded", np.full((2, 3), float(i)))
        assert not hasattr(
            reg, "_feature_cache_tensor"
        ), "tensor provenance must live in the cache keys, not side state"
        assert reg.feature_cache_stats() is not None
