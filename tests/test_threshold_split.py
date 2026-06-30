# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for leakage-free 3-way threshold tuning (val) / reporting (test).

Covers the fix for the evaluation defect where the operating threshold was
selected on the same data it was reported on (optimistic, leaky). The honest
path tunes on a validation split and reports on a disjoint test split.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.evaluation.metrics import (
    compute_best_f1,
    evaluate_anomaly_detection,
    evaluate_anomaly_detection_split,
    fit_threshold,
    split_three_way,
)
from omni_mercury_engine.metrics.anomaly_metrics import AnomalyMetrics


class TestSplitThreeWay:
    """The seeded, stratified / temporal splitter."""

    def test_partitions_are_disjoint_and_cover_everything(self) -> None:
        train, val, test = split_three_way(100, np.array([0, 1] * 50), random_state=7)
        union = np.concatenate([train, val, test])
        assert len(union) == 100
        assert len(np.unique(union)) == 100  # disjoint
        assert set(union.tolist()) == set(range(100))

    def test_stratification_keeps_both_classes_in_each_split(self) -> None:
        # 80 normal, 20 anomaly.
        y = np.array([0] * 80 + [1] * 20)
        train, val, test = split_three_way(
            100, y, val_frac=0.2, test_frac=0.4, random_state=0, stratify=True
        )
        for split in (train, val, test):
            assert y[split].sum() >= 1  # at least one anomaly
            assert (y[split] == 0).sum() >= 1  # at least one normal

    def test_determinism(self) -> None:
        a = split_three_way(50, np.array([0, 1] * 25), random_state=3)
        b = split_three_way(50, np.array([0, 1] * 25), random_state=3)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)

    def test_timeseries_split_is_contiguous(self) -> None:
        train, val, test = split_three_way(
            100, val_frac=0.2, test_frac=0.4, is_timeseries=True
        )
        # No shuffling: each split is a contiguous, ordered block.
        assert np.array_equal(train, np.sort(train))
        assert train[-1] < val[0] < test[0]
        assert np.array_equal(val, np.arange(val[0], val[-1] + 1))


class TestThresholdLeakage:
    """The core regression: in-sample F1 is optimistic vs the honest test F1."""

    def _leaky_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        # Scores carry real but imperfect signal; a threshold perfectly tuned on
        # the whole set overfits the boundary point.
        rng = np.random.default_rng(0)
        n = 400
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        score = np.where(
            y == 1, rng.normal(0.6, 0.25, n), rng.normal(0.4, 0.25, n)
        )
        return y, np.clip(score, 0.0, 1.0)

    def test_split_reports_honest_not_optimistic_f1(self) -> None:
        y, score = self._leaky_dataset()
        # Reconstruct the same test split the honest evaluator uses.
        _, _, test_idx = split_three_way(
            len(y), y, val_frac=0.2, test_frac=0.4, random_state=0
        )
        # Oracle: the best achievable F1 on the *test* set if its threshold were
        # tuned in-sample (this is exactly what the leaky path would report).
        oracle_test_f1, _ = compute_best_f1(y[test_idx], score[test_idx])

        split = evaluate_anomaly_detection_split(y, score, random_state=0)
        # The honest, val-tuned threshold can never beat the test-set oracle...
        assert split.f1 <= oracle_test_f1 + 1e-9
        # ...and on noisy-but-separable data it is strictly worse: that gap is
        # precisely the optimism the in-sample path hides.
        assert split.f1 < oracle_test_f1
        # The reported threshold is the val-tuned one, applied to test.
        assert 0.0 <= split.best_threshold <= 1.0

    def test_tune_on_val_matches_split_entrypoint(self) -> None:
        y, score = self._leaky_dataset()
        via_flag = evaluate_anomaly_detection(y, score, tune_on="val", random_state=0)
        via_fn = evaluate_anomaly_detection_split(y, score, random_state=0)
        assert via_flag.f1 == via_fn.f1
        assert via_flag.best_threshold == via_fn.best_threshold

    def test_fit_threshold_returns_only_threshold(self) -> None:
        y, score = self._leaky_dataset()
        thr = fit_threshold(y, score)
        assert isinstance(thr, float)
        assert score.min() <= thr <= score.max()


class TestBackwardCompatAndFallback:
    """Defaults are preserved; tiny inputs degrade gracefully."""

    def test_default_is_in_sample_unchanged(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        legacy = evaluate_anomaly_detection(y, score)
        explicit = evaluate_anomaly_detection(y, score, tune_on="in_sample")
        assert legacy.best_f1 == explicit.best_f1
        assert legacy.f1 == explicit.f1

    def test_tiny_n_falls_back_to_in_sample(self) -> None:
        y = np.array([0, 1, 0, 1])
        score = np.array([0.2, 0.8, 0.3, 0.7])
        # Should not raise; falls back to the in-sample result.
        split = evaluate_anomaly_detection_split(y, score)
        legacy = evaluate_anomaly_detection(y, score)
        assert split.f1 == legacy.f1
        assert split.best_threshold == legacy.best_threshold

    def test_compute_all_default_matches_legacy(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1])
        score = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.7])
        default = AnomalyMetrics.compute_all(y, score)
        in_sample = AnomalyMetrics.compute_all(y, score, tune_on="in_sample")
        assert default["f1_max"] == in_sample["f1_max"]
        assert default["auroc"] == in_sample["auroc"]

    def test_compute_all_val_is_honest_on_large_data(self) -> None:
        rng = np.random.default_rng(1)
        n = 300
        y = np.array([0] * 150 + [1] * 150)
        score = np.clip(
            np.where(y == 1, rng.normal(0.6, 0.2, n), rng.normal(0.4, 0.2, n)), 0, 1
        )
        _, _, test_idx = split_three_way(n, y, val_frac=0.2, test_frac=0.4, random_state=0)
        oracle_test = AnomalyMetrics.compute_all(y[test_idx], score[test_idx])
        honest = AnomalyMetrics.compute_all(y, score, tune_on="val", random_state=0)
        # Honest f1 on the held-out test split cannot beat the test-set oracle.
        assert honest["f1_max"] <= oracle_test["f1_max"] + 1e-9
