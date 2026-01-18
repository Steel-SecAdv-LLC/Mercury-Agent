"""
Mercury Agent ♱
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


"""Tests for DimensionalAnalyzer detector."""

import numpy as np
import pytest
import torch

from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer, NeuralProjection


class TestNeuralProjection:
    """Tests for NeuralProjection module."""

    def test_initialization(self):
        """Test NeuralProjection initialization with various dimensions."""
        model = NeuralProjection(input_dim=20, latent_dim=5)
        assert model.encoder is not None
        assert model.decoder is not None

    def test_forward_pass(self):
        """Test forward pass through NeuralProjection."""
        model = NeuralProjection(input_dim=20, latent_dim=5)
        x = torch.randn(10, 20)
        latent, reconstructed = model(x)

        assert latent.shape == (10, 5)
        assert reconstructed.shape == (10, 20)

    def test_encoder_decoder_consistency(self):
        """Test that encoder reduces dims and decoder restores them."""
        model = NeuralProjection(input_dim=50, latent_dim=10)
        x = torch.randn(5, 50)
        latent, reconstructed = model(x)

        assert latent.shape[1] < x.shape[1]
        assert reconstructed.shape == x.shape


class TestDimensionalAnalyzer:
    """Tests for DimensionalAnalyzer detector."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        analyzer = DimensionalAnalyzer()
        assert analyzer.n_components == 10
        assert analyzer.reconstruction_threshold == 2.0
        assert analyzer.use_db_term is True
        assert not analyzer._is_fitted

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "n_components": 5,
            "reconstruction_threshold": 1.5,
            "use_db_term": False,
        }
        analyzer = DimensionalAnalyzer(config=config)
        assert analyzer.n_components == 5
        assert analyzer.reconstruction_threshold == 1.5
        assert analyzer.use_db_term is False

    def test_fit_numpy_array(self, sample_data):
        """Test fitting with numpy array."""
        analyzer = DimensionalAnalyzer()
        result = analyzer.fit(sample_data)

        assert result is analyzer
        assert analyzer._is_fitted
        assert analyzer.pca is not None
        assert analyzer.autoencoder is not None
        assert analyzer.input_dim == sample_data.shape[1]

    def test_fit_torch_tensor(self):
        """Test fitting with torch tensor."""
        data = torch.randn(100, 15)
        analyzer = DimensionalAnalyzer()
        result = analyzer.fit(data)

        assert result is analyzer
        assert analyzer._is_fitted
        assert analyzer.input_dim == 15

    def test_fit_with_db_term(self, sample_data):
        """Test fitting computes baseline spectral signature when DB term enabled."""
        analyzer = DimensionalAnalyzer(config={"use_db_term": True})
        analyzer.fit(sample_data)

        assert analyzer.baseline_spectral_signature is not None
        assert len(analyzer.baseline_spectral_signature) > 0

    def test_fit_without_db_term(self, sample_data):
        """Test fitting skips spectral signature when DB term disabled."""
        analyzer = DimensionalAnalyzer(config={"use_db_term": False})
        analyzer.fit(sample_data)

        assert analyzer.baseline_spectral_signature is None

    def test_detect_unfitted_raises(self, sample_data):
        """Test detection on unfitted detector raises exception."""
        analyzer = DimensionalAnalyzer()
        with pytest.raises(DetectorException, match="must be fitted"):
            analyzer.detect(sample_data)

    def test_detect_numpy_array(self, sample_data):
        """Test detection with numpy array input."""
        analyzer = DimensionalAnalyzer()
        analyzer.fit(sample_data)
        result = analyzer.detect(sample_data)

        assert "is_anomaly" in result
        assert "scores" in result
        assert "pca_errors" in result
        assert "autoencoder_errors" in result
        assert "detector_type" in result
        assert result["detector_type"] == "dimensional"
        assert len(result["scores"]) == len(sample_data)

    def test_detect_torch_tensor(self):
        """Test detection with torch tensor input."""
        data = torch.randn(50, 20)
        analyzer = DimensionalAnalyzer()
        analyzer.fit(data)
        result = analyzer.detect(data)

        assert "is_anomaly" in result
        assert "scores" in result
        assert len(result["scores"]) == 50

    def test_detect_with_db_term(self, sample_data):
        """Test detection includes DB scores when enabled."""
        analyzer = DimensionalAnalyzer(config={"use_db_term": True})
        analyzer.fit(sample_data)
        result = analyzer.detect(sample_data)

        assert result["db_scores"] is not None
        assert len(result["db_scores"]) == len(sample_data)

    def test_detect_without_db_term(self, sample_data):
        """Test detection returns None for DB scores when disabled."""
        analyzer = DimensionalAnalyzer(config={"use_db_term": False})
        analyzer.fit(sample_data)
        result = analyzer.detect(sample_data)

        assert result["db_scores"] is None

    def test_detect_anomalies_in_outliers(self, sample_data):
        """Test that clear outliers are detected as anomalies."""
        normal_data = sample_data
        # Create outliers with much larger values
        outlier_data = np.vstack([sample_data[:5], sample_data[5:10] * 10])

        analyzer = DimensionalAnalyzer(config={"threshold": 0.3})
        analyzer.fit(normal_data)
        result = analyzer.detect(outlier_data)

        # At least some outliers should be detected
        assert np.sum(result["is_anomaly"]) > 0

    def test_extract_features_unfitted(self, sample_data):
        """Test feature extraction auto-fits if not fitted."""
        analyzer = DimensionalAnalyzer()
        features = analyzer.extract_features(sample_data)

        assert analyzer._is_fitted
        assert isinstance(features, torch.Tensor)

    def test_extract_features_numpy(self, sample_data):
        """Test feature extraction with numpy input."""
        analyzer = DimensionalAnalyzer()
        analyzer.fit(sample_data)
        features = analyzer.extract_features(sample_data)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == sample_data.shape[0]
        # Should have at least 50 features (or padded to 50)
        assert features.shape[1] >= 50

    def test_extract_features_torch(self):
        """Test feature extraction with torch tensor input."""
        data = torch.randn(30, 20)
        analyzer = DimensionalAnalyzer()
        analyzer.fit(data)
        features = analyzer.extract_features(data)

        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == 30
        assert features.shape[1] >= 50

    def test_extract_features_padding(self):
        """Test feature extraction pads to minimum dimension."""
        data = np.random.randn(20, 5)  # Small features
        analyzer = DimensionalAnalyzer(config={"n_components": 3})
        analyzer.fit(data)
        features = analyzer.extract_features(data)

        # Should be padded to at least 50 features
        assert features.shape[1] >= 50


class TestDBTerm:
    """Tests for Dimensional Code-Breaking (DB) term functions."""

    def test_compute_spectral_signature_1d(self):
        """Test spectral signature computation with 1D input."""
        analyzer = DimensionalAnalyzer()
        signal = np.sin(np.linspace(0, 10, 100))
        signature = analyzer._compute_spectral_signature(signal)

        assert signature is not None
        assert len(signature) > 0

    def test_compute_spectral_signature_2d(self):
        """Test spectral signature computation with 2D input."""
        analyzer = DimensionalAnalyzer()
        data = np.random.randn(100, 5)
        signature = analyzer._compute_spectral_signature(data)

        assert signature is not None
        assert len(signature) > 0

    def test_dimensional_code_breaking(self, sample_data):
        """Test DB score computation."""
        analyzer = DimensionalAnalyzer()
        analyzer.fit(sample_data)

        scores = analyzer._dimensional_code_breaking(sample_data)

        assert len(scores) == len(sample_data)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_phase_coherence(self):
        """Test phase coherence computation."""
        analyzer = DimensionalAnalyzer()

        # Coherent signal (pure sine)
        coherent = np.sin(np.linspace(0, 4 * np.pi, 100))
        coherence = analyzer._compute_phase_coherence(coherent)
        assert 0.0 <= coherence <= 1.0

        # Random signal (less coherent)
        random_signal = np.random.randn(100)
        random_coherence = analyzer._compute_phase_coherence(random_signal)
        assert 0.0 <= random_coherence <= 1.0

    def test_phase_coherence_short_signal(self):
        """Test phase coherence with very short signal."""
        analyzer = DimensionalAnalyzer()
        short_signal = np.array([1.0, 2.0])
        coherence = analyzer._compute_phase_coherence(short_signal)
        assert coherence == 1.0  # Should return 1.0 for signals < 4 samples

    def test_harmonic_distortion(self):
        """Test harmonic distortion computation."""
        analyzer = DimensionalAnalyzer()

        # Pure sine wave (low distortion)
        pure_sine = np.sin(np.linspace(0, 4 * np.pi, 100))
        thd = analyzer._compute_harmonic_distortion(pure_sine)
        assert 0.0 <= thd <= 1.0

        # Random signal
        random_signal = np.random.randn(100)
        random_thd = analyzer._compute_harmonic_distortion(random_signal)
        assert 0.0 <= random_thd <= 1.0

    def test_harmonic_distortion_short_signal(self):
        """Test harmonic distortion with very short signal."""
        analyzer = DimensionalAnalyzer()
        short_signal = np.array([1.0, 2.0, 3.0])
        thd = analyzer._compute_harmonic_distortion(short_signal)
        assert thd == 0.0  # Should return 0.0 for signals < 8 samples
