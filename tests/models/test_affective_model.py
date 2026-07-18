# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Affective Anomaly Model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.models.affective import AffectiveAnomalyModel
from omni_mercury_engine.utils.rng import DeterministicRNG


class TestAffectiveAnomalyModelInitialization:
    """Tests for AffectiveAnomalyModel initialization."""

    def test_default_initialization(self) -> None:
        """Test initialization with default parameters."""
        model = AffectiveAnomalyModel()

        assert model.config == {}
        assert model._rng is not None

    def test_initialization_with_config(self) -> None:
        """Test initialization with custom config."""
        config = {"threshold": 0.5, "mode": "aggressive"}
        model = AffectiveAnomalyModel(config=config)

        assert model.config == config
        assert model.config["threshold"] == 0.5

    def test_initialization_with_rng(self) -> None:
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

    def test_extract_features_numpy_2d(self, model: Any) -> None:
        """Test feature extraction with 2D numpy array."""
        data = np.random.randn(10, 20)
        features = model.extract_features(data)

        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.shape == (10, 64)

    def test_extract_features_numpy_1d(self, model: Any) -> None:
        """Test feature extraction with 1D numpy array (reshaped)."""
        data = np.random.randn(100)
        features = model.extract_features(data)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 64)

    def test_extract_features_dict(self, model: Any) -> None:
        """Test feature extraction with dict input."""
        data = {"signal": np.random.randn(50)}
        features = model.extract_features(data)

        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 64

    def test_extract_features_list(self, model: Any) -> None:
        """Test feature extraction with list input."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        features = model.extract_features(data)

        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 64)

    def test_extract_features_deterministic(self) -> None:
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

    def test_predict_numpy_2d(self, model: Any) -> None:
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

    def test_predict_numpy_1d(self, model: Any) -> None:
        """Test prediction with 1D numpy array."""
        data = np.random.randn(100)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (1,)
        assert result["emotion_scores"].shape == (1, 6)
        assert result["distress_levels"].shape == (1,)

    def test_predict_dict(self, model: Any) -> None:
        """Test prediction with dict input."""
        data = {"signal": np.random.randn(50)}
        result = model.predict(data)

        assert "anomaly_scores" in result
        assert "emotion_scores" in result
        assert "distress_levels" in result

    def test_predict_scores_in_valid_range(self, model: Any) -> None:
        """Test that scores are in valid range [0, 1]."""
        data = np.random.randn(20, 10)
        result = model.predict(data)

        # Anomaly scores and distress levels should be in [0, 1]
        assert np.all(result["anomaly_scores"] >= 0)
        assert np.all(result["anomaly_scores"] <= 1)
        assert np.all(result["distress_levels"] >= 0)
        assert np.all(result["distress_levels"] <= 1)

    def test_predict_float32_output(self, model: Any) -> None:
        """Test that outputs are float32."""
        data = np.random.randn(5, 10)
        result = model.predict(data)

        assert result["anomaly_scores"].dtype == np.float32
        assert result["emotion_scores"].dtype == np.float32
        assert result["distress_levels"].dtype == np.float32

    def test_predict_deterministic(self) -> None:
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

    def test_emotion_scores_shape(self, model: Any) -> None:
        """Test that emotion scores have correct shape (6 emotions)."""
        data = np.random.randn(10, 20)
        result = model.predict(data)

        # Should have 6 emotion dimensions
        assert result["emotion_scores"].shape[1] == 6

    def test_emotion_scores_batch_dimension(self, model: Any) -> None:
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

    def test_single_sample(self, model: Any) -> None:
        """Test with single sample."""
        data = np.random.randn(1, 10)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (1,)

    def test_large_batch(self, model: Any) -> None:
        """Test with large batch."""
        data = np.random.randn(1000, 50)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (1000,)

    def test_small_feature_dim(self, model: Any) -> None:
        """Test with small feature dimension."""
        data = np.random.randn(10, 2)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (10,)

    def test_large_feature_dim(self, model: Any) -> None:
        """Test with large feature dimension."""
        data = np.random.randn(10, 500)
        result = model.predict(data)

        assert result["anomaly_scores"].shape == (10,)

    def test_dict_with_multiple_keys(self, model: Any) -> None:
        """Test with dict containing multiple keys."""
        data = {
            "signal": np.random.randn(50),
            "metadata": np.array([1, 2, 3]),
        }
        # Should use first value
        result = model.predict(data)

        assert "anomaly_scores" in result


class TestDeclaredAffectiveModality:
    """The declared-emotions path: real deterministic analysis, no fabrication."""

    def _model(self) -> AffectiveAnomalyModel:
        return AffectiveAnomalyModel()

    def test_calm_series_scores_low_distress(self) -> None:
        model = self._model()
        # Sustained happy/neutral affect: rows are (neutral, happy, sad, angry, fearful, surprised)
        calm = np.tile(np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]), (20, 1))
        result = model.predict({"emotions": calm})
        assert result["anomaly_scores"].shape == (1,)
        assert result["distress_levels"][0] < 0.3
        # Aggregated emotions mirror the declared distribution
        np.testing.assert_allclose(result["emotion_scores"][0][:2], [0.5, 0.5], atol=1e-6)

    def test_sustained_negative_series_scores_high_distress(self) -> None:
        model = self._model()
        distressed = np.tile(np.array([0.0, 0.0, 0.4, 0.4, 0.2, 0.0]), (20, 1))
        calm = np.tile(np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]), (20, 1))
        r_distress = model.predict({"emotions": distressed})
        r_calm = model.predict({"emotions": calm})
        assert r_distress["distress_levels"][0] > r_calm["distress_levels"][0]
        assert r_distress["anomaly_scores"][0] > 0.5

    def test_batched_series(self) -> None:
        model = self._model()
        batch = np.stack(
            [
                np.tile(np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]), (10, 1)),
                np.tile(np.array([0.0, 0.0, 0.5, 0.3, 0.2, 0.0]), (10, 1)),
            ]
        )
        result = model.predict({"emotions": batch})
        assert result["emotion_scores"].shape == (2, 6)
        assert result["distress_levels"][1] > result["distress_levels"][0]

    def test_declared_path_is_deterministic(self) -> None:
        series = np.abs(np.random.default_rng(3).normal(size=(15, 6)))
        r1 = self._model().predict({"emotions": series})
        r2 = self._model().predict({"emotions": series})
        np.testing.assert_array_equal(r1["anomaly_scores"], r2["anomaly_scores"])
        np.testing.assert_array_equal(r1["emotion_scores"], r2["emotion_scores"])

    def test_malformed_declared_input_fails_loud(self) -> None:
        model = self._model()
        with pytest.raises(ValueError, match="shape"):
            model.predict({"emotions": np.zeros((10, 4))})  # wrong emotion count
        with pytest.raises(ValueError, match="non-finite"):
            bad = np.zeros((5, 6))
            bad[2, 3] = np.nan
            model.predict({"emotions": bad})
        with pytest.raises(ValueError, match="non-negative"):
            model.predict({"emotions": np.full((5, 6), -1.0)})

    def test_empty_time_series_fails_loud_instead_of_nan(self) -> None:
        """A declared series with zero timesteps must raise, not emit NaNs.

        Regression: shape (batch, 0, 6) passed the shape check, then the
        temporal mean over the empty axis produced NaN anomaly/emotion scores.
        """
        model = self._model()
        with pytest.raises(ValueError, match="empty emotion time series"):
            model.predict({"emotions": np.zeros((0, 6))})
        with pytest.raises(ValueError, match="empty emotion time series"):
            model.predict({"emotions": np.zeros((2, 0, 6))})

    def test_generic_dict_still_neutral(self) -> None:
        """Dicts without the declared key keep the neutral quarantine prior."""
        model = self._model()
        result = model.predict({"signal": np.random.default_rng(0).normal(size=50)})
        assert float(result["anomaly_scores"][0]) == 0.5
        assert not result["emotion_scores"].any()

    def test_fusion_features_stay_neutral_for_declared_input(self) -> None:
        """The fusion feature contract (zeros) is independent of modality."""
        model = self._model()
        feats = model.extract_features(np.random.default_rng(1).normal(size=(4, 8)))
        assert not feats.any()
