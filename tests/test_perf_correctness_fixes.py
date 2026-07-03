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

    def test_off_sample_point_mutation_changes_key(self) -> None:
        """A change confined to an index the 256-point stride never samples
        must still change the key -- otherwise get_or_compute() would return
        stale cached features for genuinely different data (a silent wrong-
        cache-hit, not just a missed optimization)."""
        from omni_mercury_engine.engine import FeatureCache

        cache = FeatureCache()
        n = 10_000
        a = np.arange(n, dtype=float)
        sampled = set(np.linspace(0, n - 1, 256).astype(int).tolist())
        off_sample = next(i for i in range(n) if i not in sampled)
        b = a.copy()
        b[off_sample] += 1.0
        assert cache._make_key(a) != cache._make_key(b)

    def test_torch_cpu_noncontiguous_distinct_tensors_do_not_collide(self) -> None:
        """Distinct non-contiguous CPU tensors of the same shape/dtype must key
        differently (regression guard against the ptr=0 collapse Copilot flagged,
        and against the address-reuse alias identity keying leaves open)."""
        torch = pytest.importorskip("torch")
        from omni_mercury_engine.engine import FeatureCache

        cache = FeatureCache()
        a = torch.arange(64, dtype=torch.float32).reshape(8, 8)[:, ::2]
        b = (torch.arange(64, dtype=torch.float32).reshape(8, 8) + 100.0)[:, ::2]
        assert not a.is_contiguous()
        assert cache._make_key(a) != cache._make_key(b)
        # A contiguous tensor and its (non-contiguous) transpose differ too.
        m = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        assert cache._make_key(m) != cache._make_key(m.t())

    def test_torch_cpu_inplace_mutation_changes_key(self) -> None:
        """In-place mutation of a CPU tensor (same storage) must change the key.

        Pure identity keying (data_ptr + view metadata) would return the SAME
        key for the mutated tensor -> a stale cache hit for different contents.
        CPU tensors are content-keyed to close this; see ``FeatureCache._make_key``.
        """
        torch = pytest.importorskip("torch")
        from omni_mercury_engine.engine import FeatureCache

        cache = FeatureCache()
        x = torch.arange(10_000, dtype=torch.float32)
        k1 = cache._make_key(x)
        x[500] = 9999.0  # in-place, same storage
        assert cache._make_key(x) != k1
        # Off-sample mutation is caught by the folded finite-aware checksum too.
        y = torch.arange(10_000, dtype=torch.float32)
        k_y = cache._make_key(y)
        sampled = set(np.linspace(0, y.numel() - 1, 256).astype(int).tolist())
        off = next(i for i in range(y.numel()) if i not in sampled)
        y[off] += 1.0
        assert cache._make_key(y) != k_y

    def test_torch_cpu_equal_content_same_key(self) -> None:
        """Two distinct CPU tensors with equal content key identically (content
        keying) -- correct: equal data yields equal cached features."""
        torch = pytest.importorskip("torch")
        from omni_mercury_engine.engine import FeatureCache

        cache = FeatureCache()
        a = torch.arange(1000, dtype=torch.float32)
        b = torch.arange(1000, dtype=torch.float32)
        assert cache._make_key(a) == cache._make_key(b)


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

    def test_n_results_counts_only_merged_partitions(self) -> None:
        # A failed and an empty partition are recorded but not merged; n_results
        # must report the count actually concatenated, not len(self._results).
        from omni_mercury_engine.distributed.cluster import (
            ResultAggregator,
            TaskResult,
            TaskStatus,
        )

        agg = ResultAggregator()
        agg.add_result(
            TaskResult(
                task_id="ok",
                status=TaskStatus.COMPLETED,
                result={"anomaly_scores": np.array([1.0, 2.0]), "predictions": np.array([0, 0])},
                data_indices=(0, 2),
            )
        )
        agg.add_result(
            TaskResult(task_id="failed", status=TaskStatus.FAILED, result=None, data_indices=(2, 4))
        )
        agg.add_result(
            TaskResult(
                task_id="empty",
                status=TaskStatus.COMPLETED,
                result={"anomaly_scores": np.array([]), "predictions": np.array([])},
                data_indices=(4, 4),
            )
        )
        out = agg.aggregate()
        assert out["n_results"] == 1  # only "ok" was merged, not the 3 recorded
        assert np.array_equal(out["anomaly_scores"], np.array([1.0, 2.0]))

    def test_empty_result_shape_matches_merged(self) -> None:
        # Both empty paths -- no results recorded, and all recorded results
        # filtered out (failed/empty) -- must return the same key shape as a
        # successful merge so callers can read ``n_results`` /
        # ``aggregation_method`` unconditionally (no KeyError on the empty path).
        from omni_mercury_engine.distributed.cluster import (
            ResultAggregator,
            TaskResult,
            TaskStatus,
        )

        expected_keys = {"anomaly_scores", "predictions", "n_results", "aggregation_method"}

        # (1) No results at all -> ResultAggregator.aggregate() empty path.
        agg = ResultAggregator(aggregation_method="weighted_fusion")
        out = agg.aggregate()
        assert set(out) == expected_keys
        assert out["n_results"] == 0
        assert out["aggregation_method"] == "weighted_fusion"
        assert len(out["anomaly_scores"]) == 0

        # (2) Results recorded but all filtered out (failed + empty) ->
        #     _merge_partitions() empty path.
        agg.add_result(
            TaskResult(task_id="failed", status=TaskStatus.FAILED, result=None, data_indices=(0, 2))
        )
        agg.add_result(
            TaskResult(
                task_id="empty",
                status=TaskStatus.COMPLETED,
                result={"anomaly_scores": np.array([]), "predictions": np.array([])},
                data_indices=(2, 2),
            )
        )
        out2 = agg.aggregate()
        assert set(out2) == expected_keys
        assert out2["n_results"] == 0
        assert out2["aggregation_method"] == "weighted_fusion"
        assert len(out2["anomaly_scores"]) == 0


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
