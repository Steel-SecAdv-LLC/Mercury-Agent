"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for evaluation metrics module.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.evaluation.metrics import (
    AnomalyMetrics,
    compute_auc_pr,
    compute_auc_roc,
    compute_best_f1,
    compute_f1,
    compute_point_adjusted_f1,
    compute_precision_at_k,
    compute_range_based_f1,
    evaluate_anomaly_detection,
    print_metrics_report,
)


class TestAUCROC:
    """Tests for AUC-ROC computation."""

    def test_perfect_classification(self) -> None:
        """Perfect separation should give AUC-ROC = 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        auc = compute_auc_roc(y_true, y_score)
        assert auc == 1.0

    def test_random_classification(self) -> None:
        """Random guessing should give AUC-ROC ~ 0.5."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_score = np.random.rand(1000)
        auc = compute_auc_roc(y_true, y_score)
        assert 0.4 <= auc <= 0.6

    def test_inverted_classification(self) -> None:
        """Inverted predictions should give AUC-ROC = 0.0."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        auc = compute_auc_roc(y_true, y_score)
        assert auc == 0.0

    def test_all_same_class(self) -> None:
        """All same class should give AUC-ROC = 0.5."""
        y_true = np.ones(10)
        y_score = np.random.rand(10)
        auc = compute_auc_roc(y_true, y_score)
        assert auc == 0.5


class TestAUCPR:
    """Tests for AUC-PR computation."""

    def test_perfect_classification(self) -> None:
        """Perfect separation should give AUC-PR = 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        auc = compute_auc_pr(y_true, y_score)
        assert auc == 1.0

    def test_no_positives(self) -> None:
        """No positives should give AUC-PR = 0.0."""
        y_true = np.zeros(10)
        y_score = np.random.rand(10)
        auc = compute_auc_pr(y_true, y_score)
        assert auc == 0.0

    def test_imbalanced_data(self) -> None:
        """AUC-PR should handle imbalanced data."""
        y_true = np.array([0] * 90 + [1] * 10)
        y_score = np.concatenate([np.random.rand(90) * 0.5, np.random.rand(10) * 0.5 + 0.5])
        auc = compute_auc_pr(y_true, y_score)
        assert 0.0 <= auc <= 1.0


class TestF1Score:
    """Tests for F1-score computation."""

    def test_perfect_f1(self) -> None:
        """Perfect predictions should give F1 = 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        f1 = compute_f1(y_true, y_pred)
        assert f1 == 1.0

    def test_no_predictions(self) -> None:
        """No positive predictions should give F1 = 0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        f1 = compute_f1(y_true, y_pred)
        assert f1 == 0.0

    def test_all_false_positives(self) -> None:
        """All false positives should give F1 = 0.0."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        f1 = compute_f1(y_true, y_pred)
        assert f1 == 0.0

    def test_partial_match(self) -> None:
        """Partial matches should give F1 between 0 and 1."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        f1 = compute_f1(y_true, y_pred)
        # Precision = 1/2, Recall = 1/2, F1 = 0.5
        assert f1 == 0.5


class TestBestF1:
    """Tests for optimal threshold F1 computation."""

    def test_finds_optimal_threshold(self) -> None:
        """Should find threshold that maximizes F1."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 0.8, 0.9, 1.0])
        best_f1, threshold = compute_best_f1(y_true, y_score)
        assert best_f1 == 1.0
        assert 0.45 <= threshold <= 0.55


class TestPrecisionAtK:
    """Tests for Precision@K computation."""

    def test_precision_at_k_perfect(self) -> None:
        """Top-K all anomalies should give P@K = 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        p_at_k = compute_precision_at_k(y_true, y_score, k=5)
        assert p_at_k == 1.0

    def test_precision_at_k_partial(self) -> None:
        """Partial top-K should give P@K < 1.0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.95, 0.6, 0.7, 0.8, 0.9, 1.0])
        p_at_k = compute_precision_at_k(y_true, y_score, k=5)
        # Top 5: [0.95, 1.0, 0.9, 0.8, 0.7] -> [0, 1, 1, 1, 1] = 4/5
        assert p_at_k == 0.8


class TestPointAdjustedF1:
    """Tests for point-adjusted F1 (time-series)."""

    def test_segment_adjustment(self) -> None:
        """Detecting one point in segment should mark whole segment."""
        # GT: segment from 5-10
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Only detect point 7
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

        # Without adjustment: only 1 TP
        regular_f1 = compute_f1(y_true, y_pred)

        # With adjustment: whole segment detected
        adjusted_f1 = compute_point_adjusted_f1(y_true, y_pred, adjust_predicts=True)

        assert adjusted_f1 > regular_f1
        assert adjusted_f1 == 1.0

    def test_no_detection_in_segment(self) -> None:
        """No detection in segment should still give F1 = 0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        adjusted_f1 = compute_point_adjusted_f1(y_true, y_pred)
        assert adjusted_f1 == 0.0

    def test_multiple_segments(self) -> None:
        """Should handle multiple anomaly segments."""
        # Two segments: 2-4 and 7-9
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
        # Detect one point in each segment
        y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        adjusted_f1 = compute_point_adjusted_f1(y_true, y_pred)
        assert adjusted_f1 == 1.0


class TestRangeBasedF1:
    """Tests for range-based F1 (time-series)."""

    def test_perfect_overlap(self) -> None:
        """Perfect segment overlap should give F1 = 1.0."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0])
        f1 = compute_range_based_f1(y_true, y_pred)
        assert f1 == 1.0

    def test_partial_overlap(self) -> None:
        """Partial overlap should give 0 < F1 < 1."""
        y_true = np.array([0, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 0])
        f1 = compute_range_based_f1(y_true, y_pred)
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self) -> None:
        """No overlap should give F1 = 0.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1])
        f1 = compute_range_based_f1(y_true, y_pred)
        assert f1 == 0.0

    def test_empty_segments(self) -> None:
        """No anomalies in both should give F1 = 1.0."""
        y_true = np.zeros(10)
        y_pred = np.zeros(10)
        f1 = compute_range_based_f1(y_true, y_pred)
        assert f1 == 1.0


class TestEvaluateAnomalyDetection:
    """Tests for comprehensive evaluation function."""

    def test_returns_anomaly_metrics(self) -> None:
        """Should return AnomalyMetrics dataclass."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = evaluate_anomaly_detection(y_true, y_score)
        assert isinstance(metrics, AnomalyMetrics)

    def test_all_metrics_present(self) -> None:
        """Should compute all standard metrics."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = evaluate_anomaly_detection(y_true, y_score)

        assert metrics.auc_roc is not None
        assert metrics.auc_pr is not None
        assert metrics.best_f1 is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1 is not None
        assert metrics.accuracy is not None

    def test_timeseries_metrics(self) -> None:
        """Should compute time-series metrics when flag set."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.2, 0.1])
        metrics = evaluate_anomaly_detection(y_true, y_score, is_timeseries=True)

        assert metrics.point_adjusted_f1 is not None
        assert metrics.range_based_f1 is not None

    def test_custom_threshold(self) -> None:
        """Should use custom threshold when provided."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = evaluate_anomaly_detection(y_true, y_score, threshold=0.5)

        # With threshold 0.5, all predictions should be correct
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = evaluate_anomaly_detection(y_true, y_score)

        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "auc_roc" in d
        assert "precision" in d


class TestPrintMetricsReport:
    """Tests for metrics report formatting."""

    def test_generates_report(self) -> None:
        """Should generate formatted report string."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = evaluate_anomaly_detection(y_true, y_score)

        report = print_metrics_report(metrics, "TestDataset")

        assert "TestDataset" in report
        assert "AUC-ROC" in report
        assert "Precision" in report
        assert "Recall" in report

    def test_timeseries_report(self) -> None:
        """Should include time-series metrics in report."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.2, 0.1])
        metrics = evaluate_anomaly_detection(y_true, y_score, is_timeseries=True)

        report = print_metrics_report(metrics, "TimeSeriesTest")

        assert "Point-Adjusted F1" in report
        assert "Range-Based F1" in report
