"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Tests for Anomaly Detection Metrics module.

Tests AUROC, AUPRC, F1-max, pixel-level metrics, and PRO score.
"""

import numpy as np
import pytest

pytest.importorskip("torch")  # TODO: install torch in CI — see issue "Install torch in CI for full test coverage"


class TestAUROC:
    """Tests for AUROC computation."""

    def test_auroc_perfect_classifier(self):
        """Test AUROC with perfect predictions."""
        from omni_mercury_engine.metrics import compute_auroc

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        auroc = compute_auroc(y_true, y_score)
        assert auroc == 1.0

    def test_auroc_random_classifier(self):
        """Test AUROC with random predictions."""
        from omni_mercury_engine.metrics import compute_auroc

        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.random.rand(10)

        auroc = compute_auroc(y_true, y_score)
        assert 0.0 <= auroc <= 1.0

    def test_auroc_all_same_label(self):
        """Test AUROC with all same labels (edge case)."""
        from omni_mercury_engine.metrics import compute_auroc

        y_true = np.array([0, 0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        auroc = compute_auroc(y_true, y_score)
        assert auroc == 0.5  # Should return 0.5 for undefined case

    def test_auroc_with_fixture(self, binary_labels, anomaly_scores):
        """Test AUROC with test fixtures."""
        from omni_mercury_engine.metrics import compute_auroc

        auroc = compute_auroc(binary_labels, anomaly_scores)
        assert 0.0 <= auroc <= 1.0


class TestAUPRC:
    """Tests for AUPRC computation."""

    def test_auprc_perfect_classifier(self):
        """Test AUPRC with perfect predictions."""
        from omni_mercury_engine.metrics import compute_auprc

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        auprc = compute_auprc(y_true, y_score)
        assert auprc == 1.0

    def test_auprc_range(self, binary_labels, anomaly_scores):
        """Test AUPRC is in valid range."""
        from omni_mercury_engine.metrics import compute_auprc

        auprc = compute_auprc(binary_labels, anomaly_scores)
        assert 0.0 <= auprc <= 1.0


class TestF1Max:
    """Tests for F1-max score computation."""

    def test_f1_max_perfect_classifier(self):
        """Test F1-max with perfect predictions."""
        from omni_mercury_engine.metrics import compute_f1_max

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 0.8, 0.9, 1.0])

        f1_max, threshold = compute_f1_max(y_true, y_score)
        assert f1_max == 1.0

    def test_f1_max_returns_threshold(self, binary_labels, anomaly_scores):
        """Test F1-max returns optimal threshold."""
        from omni_mercury_engine.metrics import compute_f1_max

        f1_max, threshold = compute_f1_max(binary_labels, anomaly_scores)

        assert 0.0 <= f1_max <= 1.0
        assert threshold is not None


class TestOptimalThreshold:
    """Tests for optimal threshold computation."""

    def test_optimal_threshold_f1(self, binary_labels, anomaly_scores):
        """Test optimal threshold for F1 metric."""
        from omni_mercury_engine.metrics import compute_optimal_threshold

        threshold = compute_optimal_threshold(binary_labels, anomaly_scores, metric="f1")
        assert threshold is not None

    def test_optimal_threshold_accuracy(self, binary_labels, anomaly_scores):
        """Test optimal threshold for accuracy metric."""
        from omni_mercury_engine.metrics import compute_optimal_threshold

        threshold = compute_optimal_threshold(binary_labels, anomaly_scores, metric="accuracy")
        assert threshold is not None

    def test_optimal_threshold_youden(self, binary_labels, anomaly_scores):
        """Test optimal threshold using Youden's J."""
        from omni_mercury_engine.metrics import compute_optimal_threshold

        threshold = compute_optimal_threshold(binary_labels, anomaly_scores, metric="youden")
        assert threshold is not None


class TestPixelLevelMetrics:
    """Tests for pixel-level anomaly localization metrics."""

    def test_pixel_auroc(self, pixel_masks, pixel_scores):
        """Test pixel-level AUROC computation."""
        from omni_mercury_engine.metrics import compute_pixel_auroc

        pixel_auroc = compute_pixel_auroc(pixel_masks, pixel_scores)
        assert 0.0 <= pixel_auroc <= 1.0

    def test_pixel_auroc_perfect_localization(self):
        """Test pixel AUROC with perfect localization."""
        from omni_mercury_engine.metrics import compute_pixel_auroc

        masks = np.zeros((5, 64, 64))
        masks[:, 20:40, 20:40] = 1

        scores = np.zeros((5, 64, 64))
        scores[:, 20:40, 20:40] = 1.0

        pixel_auroc = compute_pixel_auroc(masks, scores)
        assert pixel_auroc == 1.0


class TestPRO:
    """Tests for Per-Region Overlap (PRO) metric."""

    def test_pro_basic(self, pixel_masks, pixel_scores):
        """Test PRO score computation."""
        from omni_mercury_engine.metrics import compute_pro

        pro = compute_pro(pixel_masks, pixel_scores)
        assert 0.0 <= pro <= 1.0

    def test_pro_perfect_localization(self):
        """Test PRO with perfect region overlap."""
        from omni_mercury_engine.metrics import compute_pro

        masks = np.zeros((5, 64, 64))
        masks[:, 20:40, 20:40] = 1

        scores = np.zeros((5, 64, 64))
        scores[:, 20:40, 20:40] = 1.0

        pro = compute_pro(masks, scores)
        assert pro > 0.8  # Should be high for perfect localization


class TestAnomalyMetrics:
    """Tests for unified AnomalyMetrics class."""

    def test_compute_all_metrics(self, binary_labels, anomaly_scores):
        """Test computing all metrics at once."""
        from omni_mercury_engine.metrics import AnomalyMetrics

        results = AnomalyMetrics.compute_all(binary_labels, anomaly_scores)

        assert "auroc" in results
        assert "auprc" in results
        assert "f1_max" in results
        assert "optimal_threshold" in results

    def test_compute_all_with_predictions(self, binary_labels, anomaly_scores):
        """Test computing metrics with binary predictions."""
        from omni_mercury_engine.metrics import AnomalyMetrics

        y_pred = (anomaly_scores > 0.5).astype(int)

        results = AnomalyMetrics.compute_all(binary_labels, anomaly_scores, y_pred=y_pred)

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results

    def test_compute_all_with_masks(self, binary_labels, anomaly_scores, pixel_masks, pixel_scores):
        """Test computing metrics with pixel masks."""
        from omni_mercury_engine.metrics import AnomalyMetrics

        # Use subset matching length
        results = AnomalyMetrics.compute_all(
            binary_labels[:10],
            anomaly_scores[:10],
            masks_true=pixel_masks,
            masks_score=pixel_scores,
        )

        assert "pixel_auroc" in results
        assert "pro" in results

    def test_compute_per_category(self, binary_labels, anomaly_scores):
        """Test per-category metric computation."""
        from omni_mercury_engine.metrics import AnomalyMetrics

        categories = ["cat_a"] * 50 + ["cat_b"] * 50

        results = AnomalyMetrics.compute_per_category(binary_labels, anomaly_scores, categories)

        assert "cat_a" in results
        assert "cat_b" in results


class TestBenchmarkEvaluator:
    """Tests for BenchmarkEvaluator class."""

    def test_evaluator_initialization(self, tmp_path):
        """Test BenchmarkEvaluator initialization."""
        from omni_mercury_engine.metrics import BenchmarkEvaluator

        evaluator = BenchmarkEvaluator(output_dir=tmp_path)
        assert evaluator.output_dir.exists()

    def test_evaluation_result_to_dict(self):
        """Test EvaluationResult serialization."""
        from omni_mercury_engine.metrics import EvaluationResult

        result = EvaluationResult(
            detector_name="test_detector",
            dataset_name="test_dataset",
            metrics={"auroc": 0.95, "f1_max": 0.9},
        )

        d = result.to_dict()
        assert d["detector_name"] == "test_detector"
        assert d["metrics"]["auroc"] == 0.95

    def test_evaluation_result_save_load(self, tmp_path):
        """Test saving and loading evaluation results."""
        from omni_mercury_engine.metrics import EvaluationResult

        result = EvaluationResult(
            detector_name="test_detector",
            dataset_name="test_dataset",
            metrics={"auroc": 0.95},
        )

        path = tmp_path / "result.json"
        result.save(path)

        loaded = EvaluationResult.load(path)
        assert loaded.detector_name == "test_detector"
        assert loaded.metrics["auroc"] == 0.95

    def test_compare_results(self, tmp_path):
        """Test comparing multiple evaluation results."""
        from omni_mercury_engine.metrics import BenchmarkEvaluator, EvaluationResult

        evaluator = BenchmarkEvaluator(output_dir=tmp_path)

        results = [
            EvaluationResult(
                detector_name="detector_a",
                dataset_name="dataset_1",
                metrics={"auroc": 0.95},
            ),
            EvaluationResult(
                detector_name="detector_b",
                dataset_name="dataset_1",
                metrics={"auroc": 0.90},
            ),
        ]

        comparison = evaluator.compare(results, metric="auroc")
        assert "Comparison" in comparison
        assert "detector_a" in comparison

    def test_generate_report(self, tmp_path):
        """Test generating evaluation report."""
        from omni_mercury_engine.metrics import BenchmarkEvaluator, EvaluationResult

        evaluator = BenchmarkEvaluator(output_dir=tmp_path)

        results = [
            EvaluationResult(
                detector_name="detector_a",
                dataset_name="dataset_1",
                metrics={"auroc": 0.95, "f1_max": 0.9},
            ),
        ]

        report_path = tmp_path / "report.md"
        evaluator.generate_report(results, report_path)
        assert report_path.exists()
