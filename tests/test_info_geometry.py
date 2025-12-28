"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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

"""Tests for Information Geometry integration."""

import numpy as np
import pytest

from omni_anomaly_engine.core.info_geometry import InformationGeometryDetector


class TestInformationGeometryDetector:
    """Test InformationGeometryDetector class."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = InformationGeometryDetector()
        assert detector.distance_metric == "fisher_rao"
        assert detector.manifold_dim == 10
        assert detector.approximation_method == "closed_form"
        assert detector.reference_distribution is None
        assert detector.fisher_matrix is None

    def test_detector_custom_config(self):
        """Test detector with custom configuration."""
        config = {
            "distance_metric": "kl_divergence",
            "manifold_dim": 20,
            "approximation_method": "sampling",
        }
        detector = InformationGeometryDetector(config)
        assert detector.distance_metric == "kl_divergence"
        assert detector.manifold_dim == 20
        assert detector.approximation_method == "sampling"

    def test_fit_reference_distribution(self):
        """Test fitting reference distribution."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)

        detector.fit_reference_distribution(in_dist_data)

        assert detector.reference_distribution is not None
        assert "mean" in detector.reference_distribution
        assert "cov" in detector.reference_distribution
        assert detector.fisher_matrix is not None

    def test_reference_mean_shape(self):
        """Test reference distribution mean shape."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)

        detector.fit_reference_distribution(in_dist_data)

        assert detector.reference_distribution["mean"].shape == (10,)

    def test_reference_cov_shape(self):
        """Test reference distribution covariance shape."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)

        detector.fit_reference_distribution(in_dist_data)

        assert detector.reference_distribution["cov"].shape == (10, 10)

    def test_fisher_matrix_shape(self):
        """Test Fisher matrix shape."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)

        detector.fit_reference_distribution(in_dist_data)

        assert detector.fisher_matrix.shape == (10, 10)

    def test_compute_fisher_matrix(self):
        """Test Fisher matrix computation."""
        detector = InformationGeometryDetector()
        mean = np.zeros(5)
        cov = np.eye(5)

        fisher = detector._compute_fisher_matrix(mean, cov)

        assert fisher.shape == (5, 5)
        assert np.all(np.isfinite(fisher))

    def test_fisher_matrix_invertible(self):
        """Test that Fisher matrix is invertible."""
        detector = InformationGeometryDetector()
        mean = np.zeros(5)
        cov = np.eye(5)

        fisher = detector._compute_fisher_matrix(mean, cov)

        det = np.linalg.det(fisher)
        assert det > 0

    def test_fisher_rao_distance_same_distribution(self):
        """Test Fisher-Rao distance is zero for same distribution."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        dist1 = {"mean": np.zeros(10), "cov": np.eye(10)}
        dist2 = {"mean": np.zeros(10), "cov": np.eye(10)}

        distance = detector.fisher_rao_distance(dist1, dist2)
        assert distance == 0.0

    def test_fisher_rao_distance_different_distributions(self):
        """Test Fisher-Rao distance for different distributions."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        dist1 = {"mean": np.zeros(10), "cov": np.eye(10)}
        dist2 = {"mean": np.ones(10), "cov": np.eye(10)}

        distance = detector.fisher_rao_distance(dist1, dist2)
        assert distance > 0.0

    def test_fisher_rao_distance_positive(self):
        """Test Fisher-Rao distance is always non-negative."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        dist1 = {"mean": np.random.randn(10), "cov": np.eye(10)}
        dist2 = {"mean": np.random.randn(10), "cov": np.eye(10)}

        distance = detector.fisher_rao_distance(dist1, dist2)
        assert distance >= 0.0

    def test_detect_ood_without_fit_raises_error(self):
        """Test that detect_ood raises error if not fitted."""
        detector = InformationGeometryDetector()
        test_data = np.random.randn(50, 10)

        with pytest.raises(ValueError, match="Must fit reference distribution first"):
            detector.detect_ood(test_data)

    def test_detect_ood_basic(self):
        """Test basic OOD detection."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert "ood_score" in results
        assert "is_ood" in results
        assert "threshold" in results
        assert "method" in results

    def test_detect_ood_method_label(self):
        """Test that method label is correct."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert results["method"] == "fisher_rao_geometry"

    def test_detect_ood_custom_threshold(self):
        """Test OOD detection with custom threshold."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data, threshold=5.0)

        assert results["threshold"] == 5.0

    def test_detect_ood_default_threshold(self):
        """Test OOD detection with default threshold."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert results["threshold"] == 3.0

    def test_detect_ood_score_type(self):
        """Test OOD score is float."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert isinstance(results["ood_score"], float)

    def test_detect_ood_is_ood_type(self):
        """Test is_ood is boolean."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(50, 10)
        results = detector.detect_ood(test_data)

        assert isinstance(results["is_ood"], (bool, np.bool_))

    def test_detect_ood_single_sample(self):
        """Test OOD detection with single sample."""
        detector = InformationGeometryDetector()
        in_dist_data = np.random.randn(100, 10)
        detector.fit_reference_distribution(in_dist_data)

        test_data = np.random.randn(1, 10)
        results = detector.detect_ood(test_data)

        assert "ood_score" in results

    def test_distance_metric_config(self):
        """Test distance metric configuration."""
        config = {"distance_metric": "kl_divergence"}
        detector = InformationGeometryDetector(config)
        assert detector.distance_metric == "kl_divergence"

    def test_manifold_dim_config(self):
        """Test manifold dimension configuration."""
        config = {"manifold_dim": 15}
        detector = InformationGeometryDetector(config)
        assert detector.manifold_dim == 15

    def test_approximation_method_config(self):
        """Test approximation method configuration."""
        config = {"approximation_method": "sampling"}
        detector = InformationGeometryDetector(config)
        assert detector.approximation_method == "sampling"
