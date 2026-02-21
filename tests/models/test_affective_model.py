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

"""Tests for Affective Anomaly Model."""

import numpy as np
import pytest

from omni_mercury_engine.models.affective import AffectiveAnomalyModel
from omni_mercury_engine.utils.rng import DeterministicRNG


class TestAffectiveAnomalyModelInitialization:
    """Tests for AffectiveAnomalyModel initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = AffectiveAnomalyModel()

        assert model.config == {}
        assert model._rng is not None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {"threshold": 0.5, "mode": "aggressive"}
        model = AffectiveAnomalyModel(config=config)

        assert model.config == config
        assert model.config["threshold"] == 0.5

    def test_initialization_with_rng(self):
        """Test initialization with custom RNG."""
        rng = DeterministicRNG(seed=123)
        model = AffectiveAnomalyModel(rng=rng)

        assert model._rng is rng


class TestFeatureExtraction:
    """Tests for feature extraction functionality."""

    @pytest.fixture
    def model(self):
        """Create model with deterministic RNG."""
        return AffectiveAnomalyModel(rng=DeterministicRNG(seed=42))

    def test_extract_features_numpy_2d(self, model):
        """Test feature extraction with 2D numpy array."""
        data = np.random.randn(10, 20)
        features = model.extract_features(data)

        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.shape == (10, 64)

    def test_extract_features_numpy_1d(self, model):
        """Test feature extraction with 1D numpy array (reshaped)."""
        data = np.random.randn(100)
        features = model.extract_features(data)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 64)

    def test_extract_features_dict(self, model):
        """Test feature extraction with dict input."""
        data = {"signal": np.random.randn(50)}
        features = model.extract_features(data)

        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 64

    def test_extract_features_list(self, model):
        """Test feature extraction with list input."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        features = model.extract_features(data)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 64)

    def test_extract_features_deterministic(self):
        """Test that feature extraction is deterministic with same seed."""
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=42)
        model1 = AffectiveAnomalyModel(rng=rng1)
        model2 = AffectiveAnomalyModel(rng=rng2)

        data = np.random.randn(5, 10)
        features1 = model1.extract_features(data)
        features2 = model2.extract_features(data)

        np.testing.assert_array_almost_equal(features1, features2)


class TestPrediction:
    """Tests for prediction functionality."""

    @pytest.fixture
    def model(self):
        """Create model with deterministic RNG."""
        return AffectiveAnomalyModel(rng=DeterministicRNG(seed=42))

    def test_predict_numpy_2d(self, model):
        """Test prediction with 2D numpy array."""
        data = np.random.randn(10, 20)
        result = model.predict(data)

        assert isinstance(result, dict)
        assert "anomaly_scores" in result
        assert "emotion_scores" in result
        assert "distress_levels" in result

        assert result["anomaly_scores"].shape == (10,)
        assert result["emotion_scores"].shape == (10, 6)
        assert result["distress_levels"].shape == (10,)

    def test_predict_numpy_1d(self, model):
        """Test prediction with 1D numpy array."""
        data = np.random.randn(100)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (1,)
        assert result["emotion_scores"].shape == (1, 6)
        assert result["distress_levels"].shape == (1,)

    def test_predict_dict(self, model):
        """Test prediction with dict input."""
        data = {"signal": np.random.randn(50)}
        result = model.predict(data)

        assert "anomaly_scores" in result
        assert "emotion_scores" in result
        assert "distress_levels" in result

    def test_predict_scores_in_valid_range(self, model):
        """Test that scores are in valid range [0, 1]."""
        data = np.random.randn(20, 10)
        result = model.predict(data)

        # Anomaly scores and distress levels should be in [0, 1]
        assert np.all(result["anomaly_scores"] >= 0)
        assert np.all(result["anomaly_scores"] <= 1)
        assert np.all(result["distress_levels"] >= 0)
        assert np.all(result["distress_levels"] <= 1)

    def test_predict_float32_output(self, model):
        """Test that outputs are float32."""
        data = np.random.randn(5, 10)
        result = model.predict(data)

        assert result["anomaly_scores"].dtype == np.float32
        assert result["emotion_scores"].dtype == np.float32
        assert result["distress_levels"].dtype == np.float32

    def test_predict_deterministic(self):
        """Test that prediction is deterministic with same seed."""
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=42)
        model1 = AffectiveAnomalyModel(rng=rng1)
        model2 = AffectiveAnomalyModel(rng=rng2)

        data = np.random.randn(5, 10)
        result1 = model1.predict(data)
        result2 = model2.predict(data)

        np.testing.assert_array_almost_equal(result1["anomaly_scores"], result2["anomaly_scores"])
        np.testing.assert_array_almost_equal(result1["emotion_scores"], result2["emotion_scores"])
        np.testing.assert_array_almost_equal(result1["distress_levels"], result2["distress_levels"])


class TestEmotionScores:
    """Tests for emotion score functionality."""

    @pytest.fixture
    def model(self):
        """Create model with deterministic RNG."""
        return AffectiveAnomalyModel(rng=DeterministicRNG(seed=42))

    def test_emotion_scores_shape(self, model):
        """Test that emotion scores have correct shape (6 emotions)."""
        data = np.random.randn(10, 20)
        result = model.predict(data)

        # Should have 6 emotion dimensions
        assert result["emotion_scores"].shape[1] == 6

    def test_emotion_scores_batch_dimension(self, model):
        """Test that emotion scores match batch dimension."""
        for batch_size in [1, 5, 20, 100]:
            data = np.random.randn(batch_size, 10)
            result = model.predict(data)

            assert result["emotion_scores"].shape[0] == batch_size


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def model(self):
        """Create model with deterministic RNG."""
        return AffectiveAnomalyModel(rng=DeterministicRNG(seed=42))

    def test_single_sample(self, model):
        """Test with single sample."""
        data = np.random.randn(1, 10)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (1,)

    def test_large_batch(self, model):
        """Test with large batch."""
        data = np.random.randn(1000, 50)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (1000,)

    def test_small_feature_dim(self, model):
        """Test with small feature dimension."""
        data = np.random.randn(10, 2)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (10,)

    def test_large_feature_dim(self, model):
        """Test with large feature dimension."""
        data = np.random.randn(10, 500)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (10,)

    def test_dict_with_multiple_keys(self, model):
        """Test with dict containing multiple keys."""
        data = {
            "signal": np.random.randn(50),
            "metadata": np.array([1, 2, 3]),
        }
        # Should use first value
        result = model.predict(data)

        assert "anomaly_scores" in result
