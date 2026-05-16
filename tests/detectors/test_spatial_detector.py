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

import pytest

pytest.importorskip("torch")

"""Tests for SpatialAnomalyDetector."""

import numpy as np
import pytest
import torch

from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector


class TestSpatialAnomalyDetector:
    """Tests for SpatialAnomalyDetector."""

    @pytest.fixture
    def spatial_data(self, deterministic_rng):
        """Generate sample spatial data (2D coordinates)."""
        return deterministic_rng.randn(100, 2) * 10

    @pytest.fixture
    def spatial_data_3d(self, deterministic_rng):
        """Generate sample 3D spatial data."""
        return deterministic_rng.randn(100, 3) * 10

    def test_initialization_default(self):
        """Test initialization with default config."""
        detector = SpatialAnomalyDetector()
        assert detector.n_neighbors == 20
        assert detector.contamination == 0.1
        assert not detector._is_fitted

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = {"n_neighbors": 10, "contamination": 0.05}
        detector = SpatialAnomalyDetector(config=config)
        assert detector.n_neighbors == 10
        assert detector.contamination == 0.05

    def test_fit_numpy_array(self, spatial_data):
        """Test fitting with numpy array."""
        detector = SpatialAnomalyDetector()
        result = detector.fit(spatial_data)

        assert result is detector
        assert detector._is_fitted
        assert detector.center is not None
        assert detector.radius_threshold is not None

    def test_fit_torch_tensor(self, spatial_data):
        """Test fitting with torch tensor."""
        tensor_data = torch.tensor(spatial_data, dtype=torch.float32)
        detector = SpatialAnomalyDetector()
        result = detector.fit(tensor_data)

        assert result is detector
        assert detector._is_fitted

    def test_fit_computes_center(self, spatial_data):
        """Test that fitting computes correct center."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)

        expected_center = np.mean(spatial_data, axis=0)
        assert detector.center is not None
        np.testing.assert_array_almost_equal(detector.center, expected_center)

    def test_fit_computes_radius_threshold(self, spatial_data):
        """Test that fitting computes 95th percentile radius."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)

        center = np.mean(spatial_data, axis=0)
        distances = np.linalg.norm(spatial_data - center, axis=1)
        expected_threshold = np.percentile(distances, 95)

        assert abs(detector.radius_threshold - expected_threshold) < 1e-6

    def test_fit_insufficient_dimensions_raises(self):
        """Test that 1D data raises exception."""
        data = np.random.randn(100, 1)
        detector = SpatialAnomalyDetector()

        with pytest.raises(DetectorException, match="at least 2 dimensions"):
            detector.fit(data)

    def test_detect_unfitted_raises(self, spatial_data):
        """Test detection on unfitted detector raises exception."""
        detector = SpatialAnomalyDetector()

        with pytest.raises(DetectorException, match="must be fitted"):
            detector.detect(spatial_data)

    def test_detect_numpy_array(self, spatial_data):
        """Test detection with numpy input."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)
        result = detector.detect(spatial_data)

        assert "is_anomaly" in result
        assert "scores" in result
        assert "distance_scores" in result
        assert "lof_scores" in result
        assert "detector_type" in result
        assert result["detector_type"] == "spatial"
        assert len(result["scores"]) == len(spatial_data)

    def test_detect_torch_tensor(self, spatial_data):
        """Test detection with torch tensor input."""
        tensor_data = torch.tensor(spatial_data, dtype=torch.float32)
        detector = SpatialAnomalyDetector()
        detector.fit(tensor_data)
        result = detector.detect(tensor_data)

        assert "is_anomaly" in result
        assert len(result["scores"]) == len(spatial_data)

    def test_detect_identifies_far_outliers(self, spatial_data):
        """Test that far outliers are detected."""
        detector = SpatialAnomalyDetector(config={"threshold": 0.3})
        detector.fit(spatial_data)

        # Create data with far outliers
        outliers = np.array([[100, 100], [-100, -100]])
        test_data = np.vstack([spatial_data[:5], outliers])

        result = detector.detect(test_data)

        # The outliers should have higher scores
        outlier_scores = result["scores"][-2:]
        normal_scores = result["scores"][:-2]
        assert np.mean(outlier_scores) > np.mean(normal_scores)

    def test_detect_3d_data(self, spatial_data_3d):
        """Test detection with 3D spatial data."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data_3d)
        result = detector.detect(spatial_data_3d)

        assert len(result["scores"]) == len(spatial_data_3d)

    def test_extract_features_unfitted(self, spatial_data):
        """Test feature extraction auto-fits if not fitted."""
        detector = SpatialAnomalyDetector()
        features = detector.extract_features(spatial_data)

        assert detector._is_fitted
        assert isinstance(features, torch.Tensor)

    def test_extract_features_numpy(self, spatial_data):
        """Test feature extraction with numpy input."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)
        features = detector.extract_features(spatial_data)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == len(spatial_data)
        # Should be padded to at least 32 features
        assert features.shape[1] >= 32

    def test_extract_features_torch(self, spatial_data):
        """Test feature extraction with torch tensor input."""
        tensor_data = torch.tensor(spatial_data, dtype=torch.float32)
        detector = SpatialAnomalyDetector()
        detector.fit(tensor_data)
        features = detector.extract_features(tensor_data)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == len(spatial_data)

    def test_extract_features_content(self, spatial_data):
        """Test that extracted features contain expected components."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)
        features = detector.extract_features(spatial_data)

        # Features should include: coordinates (2), distance (1), angle (1)
        # Total = 4, padded to 32
        assert features.shape[1] == 32

    def test_compute_distance_scores(self, spatial_data):
        """Test distance score computation."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)

        scores = detector._compute_distance_scores(spatial_data)

        assert len(scores) == len(spatial_data)
        assert np.all(scores >= 0)

    def test_scores_normalized(self, spatial_data):
        """Test that combined scores are properly normalized."""
        detector = SpatialAnomalyDetector()
        detector.fit(spatial_data)
        result = detector.detect(spatial_data)

        assert np.min(result["scores"]) >= 0
        assert np.max(result["scores"]) <= 1
