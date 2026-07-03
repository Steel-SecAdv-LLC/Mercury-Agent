# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Anomaly Detection Metrics module.

Tests AUROC, AUPRC, F1-max, pixel-level metrics, and PRO score.
"""

from __future__ import annotations

from typing import Any

import pytest

# ``omni_mercury_engine.metrics`` re-exports both NumPy- and torch-
# backed metrics; importing the package transitively pulls in the
# pixel-level metric helpers that depend on torchvision.  Skip the
# whole module cleanly when the optional ``ml`` extra is absent.
pytest.importorskip("torch")

import numpy as np


class TestAUROC:
    """Tests for AUROC computation."""

    def test_auroc_perfect_classifier(self) -> None:
        """Test AUROC with perfect predictions."""
        from omni_mercury_engine.metrics import compute_auroc

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        auroc = compute_auroc(y_true, y_score)
        assert auroc == 1.0

    def test_auroc_random_classifier(self) -> None:
        """Test AUROC with random predictions."""
        from omni_mercury_engine.metrics import compute_auroc

        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.random.rand(10)

        auroc = compute_auroc(y_true, y_score)
        assert 0.0 <= auroc <= 1.0

    def test_auroc_all_same_label(self) -> None:
        """Test AUROC with all same labels (edge case)."""
        from omni_mercury_engine.metrics import compute_auroc

        y_true = np.array([0, 0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        auroc = compute_auroc(y_true, y_score)
        assert auroc == 0.5  # Should return 0.5 for undefined case

    def test_auroc_with_fixture(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test AUROC with test fixtures."""
        from omni_mercury_engine.metrics import compute_auroc

        auroc = compute_auroc(binary_labels, anomaly_scores)
        assert 0.0 <= auroc <= 1.0


class TestAUPRC:
    """Tests for AUPRC computation."""

    def test_auprc_perfect_classifier(self) -> None:
        """Test AUPRC with perfect predictions."""
        from omni_mercury_engine.metrics import compute_auprc

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        auprc = compute_auprc(y_true, y_score)
        assert auprc == 1.0

    def test_auprc_range(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test AUPRC is in valid range."""
        from omni_mercury_engine.metrics import compute_auprc

        auprc = compute_auprc(binary_labels, anomaly_scores)
        assert 0.0 <= auprc <= 1.0


class TestF1Max:
    """Tests for F1-max score computation."""

    def test_f1_max_perfect_classifier(self) -> None:
        """Test F1-max with perfect predictions."""
        from omni_mercury_engine.metrics import compute_f1_max

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 0.8, 0.9, 1.0])

        f1_max, threshold = compute_f1_max(y_true, y_score)
        assert f1_max == 1.0

    def test_f1_max_returns_threshold(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test F1-max returns optimal threshold."""
        from omni_mercury_engine.metrics import compute_f1_max

        f1_max, threshold = compute_f1_max(binary_labels, anomaly_scores)

        assert 0.0 <= f1_max <= 1.0
        assert threshold is not None


class TestOptimalThreshold:
    """Tests for optimal threshold computation."""

    def test_optimal_threshold_f1(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test optimal threshold for F1 metric."""
        from omni_mercury_engine.metrics import compute_optimal_threshold

        threshold = compute_optimal_threshold(binary_labels, anomaly_scores, metric="f1")
        assert threshold is not None

    def test_optimal_threshold_accuracy(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test optimal threshold for accuracy metric."""
        from omni_mercury_engine.metrics import compute_optimal_threshold

        threshold = compute_optimal_threshold(binary_labels, anomaly_scores, metric="accuracy")
        assert threshold is not None

    def test_optimal_threshold_youden(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test optimal threshold using Youden's J."""
        from omni_mercury_engine.metrics import compute_optimal_threshold

        threshold = compute_optimal_threshold(binary_labels, anomaly_scores, metric="youden")
        assert threshold is not None


class TestPixelLevelMetrics:
    """Tests for pixel-level anomaly localization metrics."""

    def test_pixel_auroc(self, pixel_masks: Any, pixel_scores: Any) -> None:
        """Test pixel-level AUROC computation."""
        from omni_mercury_engine.metrics import compute_pixel_auroc

        pixel_auroc = compute_pixel_auroc(pixel_masks, pixel_scores)
        assert 0.0 <= pixel_auroc <= 1.0

    def test_pixel_auroc_perfect_localization(self) -> None:
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

    def test_pro_basic(self, pixel_masks: Any, pixel_scores: Any) -> None:
        """Test PRO score computation."""
        from omni_mercury_engine.metrics import compute_pro

        pro = compute_pro(pixel_masks, pixel_scores)
        assert 0.0 <= pro <= 1.0

    def test_pro_perfect_localization(self) -> None:
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

    def test_compute_all_metrics(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test computing all metrics at once."""
        from omni_mercury_engine.metrics import AnomalyMetrics

        results = AnomalyMetrics.compute_all(binary_labels, anomaly_scores)

        assert "auroc" in results
        assert "auprc" in results
        assert "f1_max" in results
        assert "optimal_threshold" in results

    def test_compute_all_with_predictions(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test computing metrics with binary predictions."""
        from omni_mercury_engine.metrics import AnomalyMetrics

        y_pred = (anomaly_scores > 0.5).astype(int)

        results = AnomalyMetrics.compute_all(binary_labels, anomaly_scores, y_pred=y_pred)

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results

    def test_compute_all_with_masks(
        self, binary_labels: Any, anomaly_scores: Any, pixel_masks: Any, pixel_scores: Any
    ) -> None:
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

    def test_compute_per_category(self, binary_labels: Any, anomaly_scores: Any) -> None:
        """Test per-category metric computation."""
        from omni_mercury_engine.metrics import AnomalyMetrics

        categories = ["cat_a"] * 50 + ["cat_b"] * 50

        results = AnomalyMetrics.compute_per_category(binary_labels, anomaly_scores, categories)

        assert "cat_a" in results
        assert "cat_b" in results

    def test_compute_all_val_split_masks_torch_and_list(self) -> None:
        """``tune_on='val'`` must index masks that are not NumPy arrays.

        Regression: the val-split pixel path indexed ``masks_true[test_idx]``
        with a NumPy int array *before* converting to NumPy. torch tensors and
        Python lists do not support NumPy advanced indexing with an int array
        (a list raises ``TypeError``), so the path raised at runtime whenever a
        caller passed per-sample masks as anything other than a NumPy array.
        The conversion is now hoisted ahead of the split index.
        """
        import torch

        from omni_mercury_engine.metrics import AnomalyMetrics

        rng = np.random.default_rng(0)
        n = 60
        y_true = np.array([0, 1] * (n // 2))
        y_score = rng.random(n)

        # Annotated ``Any`` on purpose: the point of this test is to feed the
        # ``compute_all`` mask params (typed ``ndarray | None``) the non-ndarray
        # arraylikes a real caller might pass -- a torch tensor and a Python
        # list -- and prove the val-split path converts before indexing.
        masks_true_t: Any = torch.from_numpy((rng.random((n, 4, 4)) > 0.7).astype(np.float32))
        masks_score_t: Any = torch.from_numpy(rng.random((n, 4, 4)).astype(np.float32))
        r_torch = AnomalyMetrics.compute_all(
            y_true, y_score, tune_on="val", masks_true=masks_true_t, masks_score=masks_score_t
        )
        assert "pixel_auroc" in r_torch and "pro" in r_torch

        masks_true_l: Any = [(rng.random((4, 4)) > 0.7).astype(np.float32) for _ in range(n)]
        masks_score_l: Any = [rng.random((4, 4)).astype(np.float32) for _ in range(n)]
        r_list = AnomalyMetrics.compute_all(
            y_true, y_score, tune_on="val", masks_true=masks_true_l, masks_score=masks_score_l
        )
        assert "pixel_auroc" in r_list and "pro" in r_list


class TestBenchmarkEvaluator:
    """Tests for BenchmarkEvaluator class."""

    def test_evaluator_initialization(self, tmp_path: Any) -> None:
        """Test BenchmarkEvaluator initialization."""
        from omni_mercury_engine.metrics import BenchmarkEvaluator

        evaluator = BenchmarkEvaluator(output_dir=tmp_path)
        assert evaluator.output_dir.exists()

    def test_evaluation_result_to_dict(self) -> None:
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

    def test_evaluation_result_save_load(self, tmp_path: Any) -> None:
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

    def test_compare_results(self, tmp_path: Any) -> None:
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

    def test_generate_report(self, tmp_path: Any) -> None:
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
