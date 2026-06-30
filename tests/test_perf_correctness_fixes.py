# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the perf + correctness hot-spot fixes.

Covers: blake3 import hoist, FeatureCache key sampling, kinematic sliding-max
vectorization parity, residual-FFT memoization, distributed partition merge
ordering, and the fail-loud-on-unfit engine guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine import crypto
from omni_mercury_engine.detectors.statistical import _sliding_backmax

if TYPE_CHECKING:
    from typing import Any


class TestBlake3Hoist:
    def test_blake3_detected_at_import_time(self) -> None:
        # Module-level detection booleans exist (no per-call import).
        assert hasattr(crypto, "_BLAKE3_AVAILABLE")

    def test_blake3_hash_is_32_bytes_and_no_per_call_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        digest = crypto.hash_data(b"mercury", "blake3")
        assert isinstance(digest, bytes) and len(digest) == 32

        # The hot path must not run the import machinery for blake3.
        real_import = builtins.__import__

        def _no_blake3(name, *a, **k):
            assert name != "blake3", "blake3 imported on the hash hot path"
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_blake3)
        again = crypto.hash_data(b"mercury", "blake3")
        assert again == digest


class TestSlidingBackmaxParity:
    @staticmethod
    def _old_spread(score: np.ndarray, n_samples: int, window: int, weight: float) -> np.ndarray:
        padded = np.zeros(n_samples)
        n = len(score)
        padded[:n] = weight * score
        spread = padded.copy()
        for shift in range(1, window):
            shifted = np.zeros(n_samples)
            shifted[shift : shift + n] = weight * score[: n_samples - shift]
            np.maximum(spread, shifted, out=spread)
        return spread

    @pytest.mark.parametrize("n_samples", [4, 5, 10, 23, 50])
    @pytest.mark.parametrize("window,weight", [(3, 0.4), (4, 0.6)])
    def test_matches_old_loop(self, n_samples: int, window: int, weight: float) -> None:
        rng = np.random.default_rng(n_samples + window)
        # derivative score is shorter than n_samples (n-2 / n-3 style)
        score = rng.random(max(0, n_samples - (window - 1)))
        old = self._old_spread(score, n_samples, window, weight)
        new = _sliding_backmax(weight * score, n_samples, window)
        assert np.allclose(old, new)


class TestFeatureCacheKey:
    def test_identical_arrays_same_key_distinct_arrays_differ(self) -> None:
        from omni_mercury_engine.engine import FeatureCache

        cache = FeatureCache()
        a = np.arange(10_000, dtype=float).reshape(100, 100)
        b = a.copy()
        assert cache._make_key(a) == cache._make_key(b)
        c = a.copy()
        c[0, 0] += 7.0  # differs within the sampled region (index 0)
        assert cache._make_key(a) != cache._make_key(c)

    def test_shape_and_dtype_separate_keys(self) -> None:
        from omni_mercury_engine.engine import FeatureCache

        cache = FeatureCache()
        a = np.zeros((10, 10))
        b = np.zeros((100,))
        assert cache._make_key(a) != cache._make_key(b)


class TestResidualFilterCache:
    def test_cache_hit_matches_uncached(self) -> None:
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.default_rng(0)
        scores = np.clip(rng.random(128), 0, 1)
        uncached = MercuryAnomalyDetector._residual_frequency_filter(scores)
        cache: dict[Any, Any] = {}
        first = MercuryAnomalyDetector._residual_frequency_filter(scores, cache=cache)
        second = MercuryAnomalyDetector._residual_frequency_filter(scores, cache=cache)
        assert np.array_equal(first, second)
        assert np.allclose(uncached, first)
        assert len(cache) == 1  # second call was a hit, not a new entry

    def test_cache_avoids_rfft_on_hit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.detectors import statistical

        rng = np.random.default_rng(1)
        scores = np.clip(rng.random(128), 0, 1)
        cache: dict[Any, Any] = {}
        statistical.MercuryAnomalyDetector._residual_frequency_filter(scores, cache=cache)

        calls = {"n": 0}
        real_rfft = np.fft.rfft

        def _counting_rfft(*a, **k):
            calls["n"] += 1
            return real_rfft(*a, **k)

        monkeypatch.setattr(np.fft, "rfft", _counting_rfft)
        statistical.MercuryAnomalyDetector._residual_frequency_filter(scores, cache=cache)
        assert calls["n"] == 0  # served from cache, no FFT


class TestPartitionMergeOrdering:
    def test_merge_orders_by_partition_start(self) -> None:
        from omni_mercury_engine.distributed.cluster import (
            ResultAggregator,
            TaskResult,
            TaskStatus,
        )

        agg = ResultAggregator()
        # Add out of order; expect reassembly in input index order.
        agg.add_result(
            TaskResult(
                task_id="b",
                status=TaskStatus.COMPLETED,
                result={"anomaly_scores": np.array([3.0, 4.0]), "predictions": np.array([1, 1])},
                data_indices=(2, 4),
            )
        )
        agg.add_result(
            TaskResult(
                task_id="a",
                status=TaskStatus.COMPLETED,
                result={"anomaly_scores": np.array([1.0, 2.0]), "predictions": np.array([0, 0])},
                data_indices=(0, 2),
            )
        )
        out = agg.aggregate()
        assert np.array_equal(out["anomaly_scores"], np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.array_equal(out["predictions"], np.array([0, 0, 1, 1]))


class TestFailLoudOnUnfit:
    def test_detect_with_fusion_raises_when_unfit_by_default(self) -> None:
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion")  # require_explicit_fit defaults True
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (8, 6))
        with pytest.raises(RuntimeError, match="not fitted"):
            engine.detect_with_fusion(X)

    def test_legacy_autofit_opt_in_does_not_raise(self) -> None:
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (8, 6))
        result = engine.detect_with_fusion(X)
        assert isinstance(result, dict)
        # The auto-fit leak is audited.
        assert engine._inference_auto_fit_detectors
