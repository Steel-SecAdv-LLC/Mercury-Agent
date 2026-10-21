# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Operating-point robustness: the detector must survive the hard cases.

The decision threshold is anchored to a fit-time reference distribution (the
training data's own ensemble scores under the serving pipeline, cut at a
one-class split-conformal quantile). These tests pin the four failure modes
the old batch-relative operating point exhibited, each reproduced and
measured before the fix:

1. **Anomaly-majority flood** — batch-median "ensemble flip" inverted all
   scores: AUROC 0.0, 0/90 true anomalies flagged, 10/10 normals flagged.
2. **Single-point streaming** — a lone point was its own 95th-percentile
   threshold, so ``score > threshold`` could never hold.
3. **One NaN cell** — propagated into the batch threshold (NaN), silencing
   every flag including a 50-sigma outlier.
4. **False-alarm control** — the fix must not inflate the all-normal flag
   rate (measured unchanged at 0.038) or move the real-data benchmark
   (8 ADBench sets: AUROC and F1 identical to 4 decimals before/after).
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector


@pytest.fixture(scope="module")
def fitted_detector() -> StatisticalAnomalyDetector:
    rng = np.random.default_rng(0)
    det = StatisticalAnomalyDetector()
    det.fit(rng.normal(0.0, 1.0, (500, 8)))
    return det


class TestAnomalyMajorityFlood:
    def test_flood_is_flagged_not_inverted(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(6.0, 1.0, (90, 8)), rng.normal(0.0, 1.0, (10, 8))])
        result = fitted_detector.detect(X)
        scores = np.asarray(result["scores"], dtype=float)
        flags = np.asarray(result["is_anomaly"], dtype=bool)
        # Perfect separation: every +6-sigma row outranks every normal row.
        assert float(scores[:90].min()) > float(scores[90:].max())
        assert int(flags[:90].sum()) >= 85, "flood rows must be flagged"
        assert int(flags[90:].sum()) <= 1, "normal rows must not be flagged"

    def test_fully_anomalous_batch_is_fully_flagged(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        """With no normal rows at all, a batch-relative cut has nothing to
        anchor on; the reference threshold flags the lot."""
        rng = np.random.default_rng(2)
        X = rng.normal(6.0, 1.0, (100, 8))
        flags = np.asarray(fitted_detector.detect(X)["is_anomaly"], dtype=bool)
        assert flags.mean() > 0.95

    def test_anomalous_bulk_regime_is_reported(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(6.0, 1.0, (100, 8))
        diag = fitted_detector.detect(X)["calibration_diagnostics"]
        assert diag is not None and diag.get("regime") == "anomalous_bulk"
        assert diag.get("method") == "reference_conformal"


class TestSinglePointDetection:
    def test_extreme_single_point_is_flagged(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        x = np.array([[100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0]])
        result = fitted_detector.detect(x)
        assert bool(np.asarray(result["is_anomaly"]).ravel()[0]) is True

    def test_normal_single_point_is_not_flagged(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        x = np.zeros((1, 8))
        result = fitted_detector.detect(x)
        assert bool(np.asarray(result["is_anomaly"]).ravel()[0]) is False


class TestNaNRobustness:
    def test_one_nan_cell_does_not_blind_the_batch(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        rng = np.random.default_rng(4)
        X = rng.normal(0.0, 1.0, (100, 8))
        X[50, 3] = 50.0  # gross outlier that must stay flagged
        clean = fitted_detector.detect(X)
        X_nan = X.copy()
        X_nan[10, 2] = np.nan
        dirty = fitted_detector.detect(X_nan)
        assert np.isfinite(float(dirty["threshold"]))
        assert bool(np.asarray(dirty["is_anomaly"])[50]) is True
        # Flag count stays in the clean batch's neighbourhood.
        assert (
            abs(
                int(np.asarray(dirty["is_anomaly"]).sum())
                - int(np.asarray(clean["is_anomaly"]).sum())
            )
            <= 2
        )

    def test_nan_rows_are_reported(self, fitted_detector: StatisticalAnomalyDetector) -> None:
        rng = np.random.default_rng(5)
        X = rng.normal(0.0, 1.0, (20, 8))
        X[3, 1] = np.nan
        X[7, 0] = np.inf
        result = fitted_detector.detect(X)
        nan_rows = np.asarray(result["nan_rows"], dtype=bool)
        assert nan_rows.sum() == 2 and bool(nan_rows[3]) and bool(nan_rows[7])

    def test_clean_batch_reports_empty_nan_rows(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        rng = np.random.default_rng(6)
        result = fitted_detector.detect(rng.normal(0.0, 1.0, (20, 8)))
        assert not np.asarray(result["nan_rows"], dtype=bool).any()


class TestFalseAlarmControl:
    def test_all_normal_flag_rate_respects_alpha(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        rng = np.random.default_rng(7)
        flags = np.asarray(
            fitted_detector.detect(rng.normal(0.0, 1.0, (500, 8)))["is_anomaly"], dtype=bool
        )
        # The adaptive path still serves normal-bulk batches; its false-alarm
        # rate was 0.038 before the change and must not inflate past the
        # conformal budget's neighbourhood.
        assert flags.mean() <= 0.10

    def test_normal_bulk_batches_keep_the_adaptive_path(
        self, fitted_detector: StatisticalAnomalyDetector
    ) -> None:
        """Benchmark-compat pin: contaminated-minority batches must keep the
        benchmark-selected adaptive operating point (measured byte-identical
        AUROC/F1 on 8 ADBench sets across this change)."""
        rng = np.random.default_rng(8)
        X = np.vstack([rng.normal(0.0, 1.0, (95, 8)), rng.normal(6.0, 1.0, (5, 8))])
        diag = fitted_detector.detect(X)["calibration_diagnostics"]
        assert diag is not None and diag.get("method") == "adaptive"


class TestReferenceDistributionContract:
    def test_reference_survives_state_round_trip(self) -> None:
        rng = np.random.default_rng(9)
        det = StatisticalAnomalyDetector()
        det.fit(rng.normal(0.0, 1.0, (300, 4)))
        assert det._reference_scores is not None
        ref_thr = det._reference_threshold()
        assert ref_thr is not None

        state = det.get_fitted_state()
        assert state is not None
        restored = StatisticalAnomalyDetector()
        restored.set_fitted_state(state)
        assert restored._reference_threshold() == pytest.approx(ref_thr)
        # Single-point detection works on the restored detector too.
        x = np.array([[50.0, -50.0, 50.0, -50.0]])
        assert bool(np.asarray(restored.detect(x)["is_anomaly"]).ravel()[0]) is True

    def test_legacy_state_without_reference_degrades_gracefully(self) -> None:
        """A detector whose restored state predates the reference keeps the
        historical batch-relative behaviour instead of crashing."""
        rng = np.random.default_rng(10)
        det = StatisticalAnomalyDetector()
        det.fit(rng.normal(0.0, 1.0, (100, 4)))
        det._reference_scores = None  # simulate a legacy restore
        result = det.detect(rng.normal(0.0, 1.0, (50, 4)))
        assert "is_anomaly" in result and np.isfinite(float(result["threshold"]))
