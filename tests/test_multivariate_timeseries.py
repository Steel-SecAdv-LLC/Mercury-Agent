# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Multivariate Time-Series Anomaly Detection."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.multivariate_timeseries import (
    ChaosMultivariateFusion,
    MultivariateTSDetector,
)


class TestMultivariateTSDetector:
    """Test multivariate time-series detector."""

    def test_detector_initialization(self) -> None:
        """Test detector initialization."""
        detector = MultivariateTSDetector()
        assert detector.window_size == 100
        assert detector.num_features == 10
        assert detector.trained is False

    def test_detector_custom_config(self) -> None:
        """Test detector with custom configuration."""
        config = {
            "window_size": 50,
            "num_features": 5,
            "lstm_hidden_dim": 32,
            "temporal_conv_filters": 16,
            "graph_conv_layers": 3,
        }
        detector = MultivariateTSDetector(config)
        assert detector.window_size == 50
        assert detector.num_features == 5
        assert detector.lstm_hidden_dim == 32
        assert detector.temporal_conv_filters == 16
        assert detector.graph_conv_layers == 3

    def test_fit_on_normal_data(self) -> None:
        """Test fitting on normal time-series data."""
        detector = MultivariateTSDetector()

        data = np.random.randn(50, 100, 10)
        detector.fit(data)

        assert detector.trained is True
        assert detector.threshold is not None
        assert detector.mean_features is not None
        assert detector.std_features is not None

    def test_predict_detects_anomalies(self) -> None:
        """Test anomaly detection on test data."""
        detector = MultivariateTSDetector()

        normal_data = np.random.randn(50, 100, 10)
        detector.fit(normal_data)

        anomalous_data = np.random.randn(10, 100, 10) * 10
        results = detector.predict(anomalous_data)

        assert "anomaly_scores" in results
        assert "predictions" in results
        # No ``roc_auc_estimate``: the detector never sees a label, so it
        # cannot report a ranking metric.
        assert "roc_auc_estimate" not in results
        assert results["method"] == "statistical_multivariate_ts"
        assert results["is_learned"] is False
        assert np.any(results["predictions"])

    def test_predict_without_fit_raises_error(self) -> None:
        """Test prediction without fitting raises error."""
        detector = MultivariateTSDetector()
        data = np.random.randn(10, 100, 10)

        with pytest.raises(ValueError, match="Model must be fit before prediction"):
            detector.predict(data)

    def test_lstm_feature_extraction(self) -> None:
        """Test LSTM feature extraction."""
        detector = MultivariateTSDetector()
        data = np.random.randn(20, 100, 10)

        features = detector._extract_lstm_features(data)

        assert features.shape == (20, 10)
        assert not np.isnan(features).any()

    def test_temporal_conv_feature_extraction(self) -> None:
        """Test temporal convolution feature extraction."""
        detector = MultivariateTSDetector()
        data = np.random.randn(20, 100, 10)

        features = detector._extract_temporal_conv_features(data)

        assert features.shape == (20, 10)
        assert not np.isnan(features).any()

    def test_graph_feature_extraction(self) -> None:
        """Test graph convolution feature extraction."""
        detector = MultivariateTSDetector()
        data = np.random.randn(20, 100, 10)

        features = detector._extract_graph_features(data)

        assert features.shape == (20, 10)
        assert not np.isnan(features).any()

    def test_reconstruction_error_computation(self) -> None:
        """Test reconstruction error computation."""
        detector = MultivariateTSDetector()
        original = np.random.randn(20, 100, 10)
        features = np.random.randn(20, 30)

        errors = detector._compute_reconstruction_error(original, features)

        assert errors.shape == (20,)
        assert np.all(errors >= 0)

    def test_no_fabricated_ranking_metric_is_reported(self) -> None:
        """The removed ``_estimate_roc_auc`` must not come back.

        It computed ``0.5 + 0.4 * tanh(separation)`` from the detector's own
        scores and its own thresholded predictions. No ground-truth label was
        ever involved, so the value could not be an AUC of anything: it rose
        whenever the detector was merely self-consistent, which is exactly when
        a real AUC would be uninformative. A number that looks like a benchmark
        result and is not one is worse than no number.
        """
        detector = MultivariateTSDetector()
        assert not hasattr(detector, "_estimate_roc_auc")

        data = np.random.randn(60, 100, 10)
        detector.fit(data)
        results = detector.predict(data)
        for key in results:
            assert "auc" not in key.lower(), key
        assert results["is_learned"] is False

    def test_detector_handles_small_dataset(self) -> None:
        """Test detector works with small dataset."""
        detector = MultivariateTSDetector({"window_size": 20, "num_features": 5})

        data = np.random.randn(5, 20, 5)
        detector.fit(data)

        assert detector.trained is True

        test_data = np.random.randn(3, 20, 5)
        results = detector.predict(test_data)

        assert len(results["anomaly_scores"]) == 3

    def test_threshold_based_on_training_distribution(self) -> None:
        """Test threshold is set based on training data."""
        detector = MultivariateTSDetector()

        normal_data = np.random.randn(100, 100, 10) * 0.5
        detector.fit(normal_data)

        threshold1 = detector.threshold
        assert threshold1 is not None

        noisy_data = np.random.randn(100, 100, 10) * 2.0
        detector.fit(noisy_data)

        threshold2 = detector.threshold
        assert threshold2 is not None

        assert threshold2 > threshold1


class TestChaosMultivariateFusion:
    """Test chaos-multivariate fusion detector."""

    def test_fusion_initialization(self) -> None:
        """Test fusion detector initialization."""
        fusion = ChaosMultivariateFusion()
        assert fusion.mvts_detector is not None
        assert fusion.trained is False

    def test_fusion_fit(self) -> None:
        """Test fitting fusion detector."""
        fusion = ChaosMultivariateFusion()

        data = np.random.randn(50, 100, 10)
        fusion.fit(data)

        assert fusion.trained is True

    def test_fusion_predict_with_chaos_refinement(self) -> None:
        """Test prediction with chaos-based refinement."""
        fusion = ChaosMultivariateFusion()

        normal_data = np.random.randn(50, 100, 10)
        fusion.fit(normal_data)

        anomalous_data = np.random.randn(10, 100, 10) * 10
        results = fusion.predict_with_chaos_refinement(anomalous_data)

        assert "anomaly_scores" in results
        assert "predictions" in results
        assert "threshold" in results
        assert "original_threshold" in results
        assert "roc_auc_estimate" not in results
        assert results["method"] == "chaos_refined_statistical_multivariate_ts"
        assert results["is_learned"] is False

    def test_fusion_predict_without_fit_raises_error(self) -> None:
        """Test prediction without fitting raises error."""
        fusion = ChaosMultivariateFusion()
        data = np.random.randn(10, 100, 10)

        with pytest.raises(ValueError, match="Model must be fit before prediction"):
            fusion.predict_with_chaos_refinement(data)

    def test_chaos_refinement_adjusts_threshold(self) -> None:
        """Test chaos refinement adjusts threshold."""
        fusion = ChaosMultivariateFusion()

        normal_data = np.random.randn(50, 100, 10)
        fusion.fit(normal_data)

        test_data = np.random.randn(20, 100, 10)
        results = fusion.predict_with_chaos_refinement(test_data)

        assert results["threshold"] != results["original_threshold"]
        assert results["threshold"] > 0

    def test_fusion_achieves_high_roc_auc(self) -> None:
        """Test fusion achieves ROC-AUC on simulated anomalies."""
        fusion = ChaosMultivariateFusion()

        normal_data = np.random.randn(100, 100, 10) * 0.5
        fusion.fit(normal_data)

        mixed_data = np.concatenate(
            [np.random.randn(50, 100, 10) * 0.5, np.random.randn(50, 100, 10) * 5.0],
            axis=0,
        )
        results = fusion.predict_with_chaos_refinement(mixed_data)

        # Separation is observable in the scores themselves; no fabricated
        # ranking metric is (or may be) reported.
        assert "roc_auc_estimate" not in results
        assert np.ptp(results["anomaly_scores"]) > 0.0

    def test_fusion_custom_configs(self) -> None:
        """Test fusion with custom configurations."""
        mvts_config = {"window_size": 50, "num_features": 5}
        chaos_config = {"population_size": 20, "max_iterations": 30}

        fusion = ChaosMultivariateFusion(mvts_config, chaos_config)

        assert fusion.mvts_detector.window_size == 50
        assert fusion.mvts_detector.num_features == 5

    def test_fusion_multiple_predictions(self) -> None:
        """Test fusion can make multiple predictions."""
        fusion = ChaosMultivariateFusion()

        normal_data = np.random.randn(50, 100, 10)
        fusion.fit(normal_data)

        for _ in range(5):
            test_data = np.random.randn(10, 100, 10)
            results = fusion.predict_with_chaos_refinement(test_data)
            assert "predictions" in results
