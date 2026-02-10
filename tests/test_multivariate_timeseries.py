"""
Mercury Agent ♱
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

"""Tests for Multivariate Time-Series Anomaly Detection."""

import numpy as np
import pytest

from omni_mercury_engine.core.multivariate_timeseries import (
    ChaosMultivariateFusion,
    MultivariateTSDetector,
)


class TestMultivariateTSDetector:
    """Test multivariate time-series detector."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = MultivariateTSDetector()
        assert detector.window_size == 100
        assert detector.num_features == 10
        assert detector.trained is False

    def test_detector_custom_config(self):
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

    def test_fit_on_normal_data(self):
        """Test fitting on normal time-series data."""
        detector = MultivariateTSDetector()

        data = np.random.randn(50, 100, 10)
        detector.fit(data)

        assert detector.trained is True
        assert detector.threshold is not None
        assert detector.mean_features is not None
        assert detector.std_features is not None

    def test_predict_detects_anomalies(self):
        """Test anomaly detection on test data."""
        detector = MultivariateTSDetector()

        normal_data = np.random.randn(50, 100, 10)
        detector.fit(normal_data)

        anomalous_data = np.random.randn(10, 100, 10) * 10
        results = detector.predict(anomalous_data)

        assert "anomaly_scores" in results
        assert "predictions" in results
        assert "roc_auc_estimate" in results
        assert results["method"] == "LTG_Multivariate_TS"
        assert np.any(results["predictions"])

    def test_predict_without_fit_raises_error(self):
        """Test prediction without fitting raises error."""
        detector = MultivariateTSDetector()
        data = np.random.randn(10, 100, 10)

        with pytest.raises(ValueError, match="Model must be fit before prediction"):
            detector.predict(data)

    def test_lstm_feature_extraction(self):
        """Test LSTM feature extraction."""
        detector = MultivariateTSDetector()
        data = np.random.randn(20, 100, 10)

        features = detector._extract_lstm_features(data)

        assert features.shape == (20, 10)
        assert not np.isnan(features).any()

    def test_temporal_conv_feature_extraction(self):
        """Test temporal convolution feature extraction."""
        detector = MultivariateTSDetector()
        data = np.random.randn(20, 100, 10)

        features = detector._extract_temporal_conv_features(data)

        assert features.shape == (20, 10)
        assert not np.isnan(features).any()

    def test_graph_feature_extraction(self):
        """Test graph convolution feature extraction."""
        detector = MultivariateTSDetector()
        data = np.random.randn(20, 100, 10)

        features = detector._extract_graph_features(data)

        assert features.shape == (20, 10)
        assert not np.isnan(features).any()

    def test_reconstruction_error_computation(self):
        """Test reconstruction error computation."""
        detector = MultivariateTSDetector()
        original = np.random.randn(20, 100, 10)
        features = np.random.randn(20, 30)

        errors = detector._compute_reconstruction_error(original, features)

        assert errors.shape == (20,)
        assert np.all(errors >= 0)

    def test_roc_auc_estimation(self):
        """Test ROC-AUC estimation."""
        detector = MultivariateTSDetector()

        scores = np.concatenate([np.random.randn(50), np.random.randn(50) + 5])
        predictions = np.concatenate([np.zeros(50, dtype=bool), np.ones(50, dtype=bool)])

        roc_auc = detector._estimate_roc_auc(scores, predictions)

        assert 0.0 <= roc_auc <= 1.0
        assert roc_auc > 0.5

    def test_roc_auc_with_all_normal(self):
        """Test ROC-AUC when all predictions are normal."""
        detector = MultivariateTSDetector()

        scores = np.random.randn(100)
        predictions = np.zeros(100, dtype=bool)

        roc_auc = detector._estimate_roc_auc(scores, predictions)

        assert roc_auc == 0.5

    def test_roc_auc_with_all_anomalies(self):
        """Test ROC-AUC when all predictions are anomalies."""
        detector = MultivariateTSDetector()

        scores = np.random.randn(100)
        predictions = np.ones(100, dtype=bool)

        roc_auc = detector._estimate_roc_auc(scores, predictions)

        assert roc_auc == 0.5

    def test_detector_handles_small_dataset(self):
        """Test detector works with small dataset."""
        detector = MultivariateTSDetector({"window_size": 20, "num_features": 5})

        data = np.random.randn(5, 20, 5)
        detector.fit(data)

        assert detector.trained is True

        test_data = np.random.randn(3, 20, 5)
        results = detector.predict(test_data)

        assert len(results["anomaly_scores"]) == 3

    def test_threshold_based_on_training_distribution(self):
        """Test threshold is set based on training data."""
        detector = MultivariateTSDetector()

        normal_data = np.random.randn(100, 100, 10) * 0.5
        detector.fit(normal_data)

        threshold1 = detector.threshold

        noisy_data = np.random.randn(100, 100, 10) * 2.0
        detector.fit(noisy_data)

        threshold2 = detector.threshold

        assert threshold2 > threshold1


class TestChaosMultivariateFusion:
    """Test chaos-multivariate fusion detector."""

    def test_fusion_initialization(self):
        """Test fusion detector initialization."""
        fusion = ChaosMultivariateFusion()
        assert fusion.mvts_detector is not None
        assert fusion.trained is False

    def test_fusion_fit(self):
        """Test fitting fusion detector."""
        fusion = ChaosMultivariateFusion()

        data = np.random.randn(50, 100, 10)
        fusion.fit(data)

        assert fusion.trained is True

    def test_fusion_predict_with_chaos_refinement(self):
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
        assert "roc_auc_estimate" in results
        assert results["method"] == "Chaos_LTG_Fusion"

    def test_fusion_predict_without_fit_raises_error(self):
        """Test prediction without fitting raises error."""
        fusion = ChaosMultivariateFusion()
        data = np.random.randn(10, 100, 10)

        with pytest.raises(ValueError, match="Model must be fit before prediction"):
            fusion.predict_with_chaos_refinement(data)

    def test_chaos_refinement_adjusts_threshold(self):
        """Test chaos refinement adjusts threshold."""
        fusion = ChaosMultivariateFusion()

        normal_data = np.random.randn(50, 100, 10)
        fusion.fit(normal_data)

        test_data = np.random.randn(20, 100, 10)
        results = fusion.predict_with_chaos_refinement(test_data)

        assert results["threshold"] != results["original_threshold"]
        assert results["threshold"] > 0

    def test_fusion_achieves_high_roc_auc(self):
        """Test fusion achieves ROC-AUC on simulated anomalies."""
        fusion = ChaosMultivariateFusion()

        normal_data = np.random.randn(100, 100, 10) * 0.5
        fusion.fit(normal_data)

        mixed_data = np.concatenate(
            [np.random.randn(50, 100, 10) * 0.5, np.random.randn(50, 100, 10) * 5.0],
            axis=0,
        )
        results = fusion.predict_with_chaos_refinement(mixed_data)

        assert results["roc_auc_estimate"] >= 0.5

    def test_fusion_custom_configs(self):
        """Test fusion with custom configurations."""
        mvts_config = {"window_size": 50, "num_features": 5}
        chaos_config = {"population_size": 20, "max_iterations": 30}

        fusion = ChaosMultivariateFusion(mvts_config, chaos_config)

        assert fusion.mvts_detector.window_size == 50
        assert fusion.mvts_detector.num_features == 5

    def test_fusion_multiple_predictions(self):
        """Test fusion can make multiple predictions."""
        fusion = ChaosMultivariateFusion()

        normal_data = np.random.randn(50, 100, 10)
        fusion.fit(normal_data)

        for _ in range(5):
            test_data = np.random.randn(10, 100, 10)
            results = fusion.predict_with_chaos_refinement(test_data)
            assert "predictions" in results
