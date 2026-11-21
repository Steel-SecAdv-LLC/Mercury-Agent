# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``OnlineDriftDetector`` keeps its decisions and stops rebuilding its window.

Every update used to turn the window, a list of up to ``max_window_size``
arrays, into a new 2-D array before testing a single split. The window now
lives in one preallocated array. The test below holds the rewritten detector
to the earlier one, kept here verbatim as the reference: on every tick of
every stream the same drift, at the same split, with the same statistic and
bound to the last bit.
"""

from __future__ import annotations

import pickle
import time
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors.drift import OnlineDriftDetector


class _ReferenceOnlineDriftDetector:
    """The detector as it was: ADWIN0 over a list window rebuilt every update."""

    def __init__(
        self,
        max_window_size: int = 1000,
        min_window_size: int = 50,
        significance_level: float = 0.05,
    ) -> None:
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.significance_level = significance_level
        self.window: list[np.ndarray] = []
        self.drift_count = 0
        self._n_features = 1

    def _first_cut(self, values: np.ndarray) -> tuple[int, float, float] | None:
        n = values.size
        min_part = max(5, self.min_window_size // 5)
        if n < 2 * min_part:
            return None
        variance = float(values.var())
        if variance == 0.0:
            return None
        value_range = float(values.max() - values.min())
        cumulative = np.cumsum(values)
        total = cumulative[-1]
        cuts = np.arange(min_part, n - min_part + 1)
        n0 = cuts.astype(float)
        n1 = n - n0
        mean0 = cumulative[cuts - 1] / n0
        mean1 = (total - cumulative[cuts - 1]) / n1
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        log_term = np.log(2.0 / (self.significance_level / (n * self._n_features)))
        eps = np.sqrt(2.0 / m * variance * log_term) + 2.0 * value_range / (3.0 * m) * log_term
        exceeded = np.flatnonzero(np.abs(mean0 - mean1) > eps)
        if exceeded.size == 0:
            return None
        k = int(exceeded[0])
        return int(cuts[k]), float(abs(mean0[k] - mean1[k])), float(eps[k])

    def update(self, sample: np.ndarray) -> dict[str, Any] | None:
        sample = np.asarray(sample, dtype=float).flatten()
        self.window.append(sample)
        self._n_features = sample.shape[0]
        if len(self.window) > self.max_window_size:
            self.window = self.window[-self.max_window_size :]
        if len(self.window) < self.min_window_size:
            return None
        data = np.array(self.window)
        hits = [
            (feature, cut)
            for feature in range(data.shape[1])
            if (cut := self._first_cut(data[:, feature])) is not None
        ]
        if not hits:
            return None
        feature, (index, difference, bound) = min(hits, key=lambda h: h[1][0])
        self.drift_count += 1
        self.window = self.window[index:]
        return {
            "test_statistic": difference,
            "threshold": bound,
            "window_size": len(self.window),
            "drift_count": self.drift_count,
            "feature": feature,
        }


def _stream(kind: str, seed: int, n: int, features: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n, features))
    if kind == "stationary":
        return data
    if kind == "one_shift":
        data[n // 2 :, features - 1] += 4.0
        return data
    if kind == "small_shift":
        data[n // 2 :, features - 1] += 1.0  # found by a long window only
        return data
    if kind == "many_shifts":
        for start in range(n // 6, n, n // 6):
            data[start:, rng.integers(features)] += rng.choice([-4.0, 4.0])
        return data
    if kind == "variance_change":
        data[n // 2 :] *= 4.0
        return data
    if kind == "heavy_tails":
        data = rng.standard_t(5.0, size=(n, features))
        data[n // 2 :, 0] += 8.0
        return data
    if kind == "constant_column":
        data[:, 0] = 0.5  # zero variance, exactly: the column is never testable
        if features > 1:
            data[n // 2 :, features - 1] += 4.0
        return data
    if kind == "ramp":
        data[:, 0] += np.linspace(0.0, 60.0, n)
        return data
    if kind == "integers":
        return rng.integers(0, 3, size=(n, features)).astype(float)
    raise AssertionError(kind)


KINDS = (
    "stationary",
    "one_shift",
    "small_shift",
    "many_shifts",
    "variance_change",
    "heavy_tails",
    "constant_column",
    "ramp",
    "integers",
)

#: (max_window_size, min_window_size, features, samples)
SHAPES = ((1024, 50, 1, 3000), (200, 50, 3, 1500), (160, 30, 8, 1000), (1000, 50, 3, 2500))


def _decisions(detector: Any, data: np.ndarray) -> list[tuple[Any, ...]]:
    """One entry per drift: (tick, feature, statistic, bound, window size, count)."""
    out: list[tuple[Any, ...]] = []
    for tick, row in enumerate(data):
        result = detector.update(row)
        if result is None:
            continue
        if isinstance(result, dict):
            details, statistic, bound = result, result["test_statistic"], result["threshold"]
        else:
            details, statistic, bound = result.details, result.test_statistic, result.threshold
        out.append(
            (
                tick,
                details["feature"],
                statistic,
                bound,
                details["window_size"],
                details["drift_count"],
            )
        )
    return out


class TestDecisionsAreTheReferenceDecisions:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_every_drift_matches_bit_for_bit(self, kind: str, shape: tuple[int, ...]) -> None:
        max_window, min_window, features, samples = shape
        drifts_seen = 0
        for seed in range(3):
            data = _stream(kind, seed, samples, features)
            reference = _ReferenceOnlineDriftDetector(max_window, min_window)
            detector = OnlineDriftDetector(max_window, min_window)

            expected = _decisions(reference, data)
            actual = _decisions(detector, data)

            # Exact equality: tick, feature, statistic and bound, not approximately.
            assert actual == expected
            np.testing.assert_array_equal(
                np.array(detector.window), np.array(reference.window), strict=True
            )
            drifts_seen += len(expected)
        if kind in {"one_shift", "many_shifts", "heavy_tails", "ramp"}:
            assert drifts_seen > 0, "the stream was meant to exercise the drift path"

    def test_a_stricter_delta_matches_too(self) -> None:
        data = _stream("many_shifts", 11, 2000, 3)
        reference = _ReferenceOnlineDriftDetector(500, 50, significance_level=0.001)
        detector = OnlineDriftDetector(500, 50, significance_level=0.001)

        assert _decisions(detector, data) == _decisions(reference, data)


class TestTheWindowIsNotRebuilt:
    def test_one_buffer_serves_a_full_window_of_updates(self) -> None:
        """No allocation per update: the same array, appended to in place."""
        detector = OnlineDriftDetector(max_window_size=256, min_window_size=50)
        data = _stream("stationary", 0, 2000, 3)
        for row in data[:300]:
            detector.update(row)
        buffer = detector._buffer
        assert buffer is not None

        for row in data[300:]:
            detector.update(row)

        assert detector._buffer is buffer
        assert buffer.shape == (3, 512)
        assert len(detector.window) == 256

    def test_the_window_is_the_most_recent_samples(self) -> None:
        detector = OnlineDriftDetector(max_window_size=100, min_window_size=50)
        data = _stream("stationary", 1, 1234, 2)

        for row in data:
            detector.update(row)

        np.testing.assert_array_equal(np.array(detector.window), data[-100:])

    def test_reset_empties_the_window(self) -> None:
        detector = OnlineDriftDetector()
        for row in _stream("one_shift", 0, 1200, 3):
            detector.update(row)

        detector.reset()

        assert (detector.window, detector.drift_detected, detector.drift_count) == ([], False, 0)
        assert detector.update(np.zeros(5)) is None  # a new width is fine after a reset

    def test_a_sample_of_another_width_is_refused(self) -> None:
        detector = OnlineDriftDetector()
        detector.update(np.zeros(3))

        with pytest.raises(ValueError, match="3"):
            detector.update(np.zeros(4))

    def test_the_window_can_be_assigned_and_survives_pickling(self) -> None:
        data = _stream("stationary", 2, 400, 2)
        detector = OnlineDriftDetector(max_window_size=300)
        detector.window = list(data[:120])
        assert len(detector.window) == 120

        restored = pickle.loads(pickle.dumps(detector))
        fresh = OnlineDriftDetector(max_window_size=300)
        fresh.window = list(data[:120])

        assert _decisions(restored, data[120:]) == _decisions(fresh, data[120:])
        np.testing.assert_array_equal(np.array(restored.window), np.array(fresh.window))

    def test_a_detector_pickled_with_a_list_window_still_loads(self) -> None:
        """State written before the buffer existed carried ``window`` as a list."""
        data = _stream("stationary", 3, 200, 2)
        legacy_state = {
            "max_window_size": 1000,
            "min_window_size": 50,
            "significance_level": 0.05,
            "window": [row.copy() for row in data[:80]],
            "drift_detected": False,
            "drift_count": 0,
            "_n_features": 2,
        }
        restored = OnlineDriftDetector.__new__(OnlineDriftDetector)
        restored.__setstate__(legacy_state)

        np.testing.assert_array_equal(np.array(restored.window), data[:80])
        assert restored.update(data[80]) is None


class TestAnUpdateIsCheaper:
    def test_a_full_window_update_costs_less_than_the_reference(self) -> None:
        """At 1,024 samples the rebuild alone was most of a single-feature update."""
        window = 1024
        data = _stream("stationary", 5, window + 1500, 1)

        def median_tick(detector: Any) -> float:
            for row in data[:window]:
                detector.update(row)
            ticks = np.empty(len(data) - window)
            for i, row in enumerate(data[window:]):
                started = time.perf_counter()
                detector.update(row)
                ticks[i] = time.perf_counter() - started
            return float(np.median(ticks))

        # Interleaved, best of several: a ratio of medians taken in one process.
        reference = min(median_tick(_ReferenceOnlineDriftDetector(window)) for _ in range(3))
        current = min(median_tick(OnlineDriftDetector(window)) for _ in range(3))

        assert current < 0.75 * reference, (current, reference)


class TestConstantFeatureIsNotDrift:
    """A feature that never changes cannot drift.

    Its variance is rounding dust rather than 0 (``var`` of 2,000 copies of 0.1
    is ~1e-34), so the ``variance == 0`` guard let it through and the dust in
    its two half-means beat a bound built from the same dust: ~195 false
    drifts per 2,000 stationary samples.
    """

    @pytest.mark.parametrize("constant", [0.1, 0.3, 7.7])
    def test_a_stationary_stream_with_a_constant_feature_raises_nothing(
        self, constant: float
    ) -> None:
        rng = np.random.default_rng(0)
        detector = OnlineDriftDetector()
        drifts = 0
        for _ in range(2000):
            sample = np.array([rng.normal(), constant, rng.normal()])
            drifts += detector.update(sample) is not None
        assert drifts == 0

    def test_a_real_shift_beside_a_constant_feature_is_still_detected(self) -> None:
        rng = np.random.default_rng(1)
        detector = OnlineDriftDetector()
        detected: list[int] = []
        for tick in range(1500):
            level = 3.0 if tick >= 800 else 0.0
            if detector.update(np.array([rng.normal(level), 0.1])) is not None:
                detected.append(tick)
        assert detected, "the mean shift at tick 800 was not detected"
        assert 800 <= detected[0] <= 900
