# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Spectral Vibration Analysis Module."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np
import pytest
import torch

from omni_mercury_engine.detectors.spectral_vibration import (
    MLIPVibrationEncoder,
    PhononInteractionNetwork,
    SpectralCNN,
    SpectralFeatures,
    SpectralGNN,
    SpectralGraphLayer,
    SpectralVibrationDetector,
    VibrationDiagnostic,
    VibrationSignatureType,
    compute_short_time_fourier_transform,
    compute_wavelet_decomposition,
    detect_peaks_with_harmonics,
)


class TestSpectralVibrationDetector:
    """Tests for SpectralVibrationDetector."""

    def test_init_default_config(self) -> None:
        """Test initialization with default configuration."""
        detector = SpectralVibrationDetector()
        assert detector is not None
        assert detector.threshold == 0.5
        assert not detector.is_fitted()

    def test_init_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = {
            "analysis_mode": "hybrid_fusion",
            "sample_rate": 10000.0,
            "fft_size": 2048,
            "threshold": 0.7,
        }
        detector = SpectralVibrationDetector(config)
        assert detector.threshold == 0.7
        assert detector._spectral_config.sample_rate == 10000.0
        assert detector._spectral_config.fft_size == 2048

    def test_fit_simple_signal(self) -> None:
        """Test fitting on a simple sinusoidal signal."""
        detector = SpectralVibrationDetector()

        # Generate test signal: 100 Hz sine wave
        sample_rate = 1000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 100 * t)

        detector.fit(signal)
        assert detector.is_fitted()
        assert detector._reference_spectrum is not None

    def test_fit_batched_signals(self) -> None:
        """Test fitting on batched signals."""
        detector = SpectralVibrationDetector()

        # Generate multiple test signals
        sample_rate = 1000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        signals = np.array(
            [
                np.sin(2 * np.pi * 100 * t),
                np.sin(2 * np.pi * 150 * t),
                np.sin(2 * np.pi * 200 * t),
            ]
        )

        detector.fit(signals)
        assert detector.is_fitted()

    def test_fit_empty_data_raises(self) -> None:
        """Test that fitting with empty data raises exception."""
        from omni_mercury_engine.core.exceptions import DetectorException

        detector = SpectralVibrationDetector()
        with pytest.raises((ValueError, RuntimeError, DetectorException)):
            detector.fit(np.array([]))

    def test_detect_normal_signal(self) -> None:
        """Test detection on a normal signal."""
        detector = SpectralVibrationDetector({"threshold": 0.6})

        # Generate training signal
        sample_rate = 1000
        t = np.linspace(0, 1, sample_rate)
        train_signal = np.sin(2 * np.pi * 100 * t)

        detector.fit(train_signal)

        # Test on similar signal
        test_signal = np.sin(2 * np.pi * 100 * t) * 0.95
        result = detector.detect(test_signal)

        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "spectral_features" in result
        assert "diagnostic" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_detect_anomalous_signal(self) -> None:
        """Test detection on an anomalous signal."""
        detector = SpectralVibrationDetector({"threshold": 0.3})

        # Train on single frequency
        sample_rate = 1000
        t = np.linspace(0, 1, sample_rate)
        train_signal = np.sin(2 * np.pi * 100 * t)
        detector.fit(train_signal)

        # Test with very different signal (impulse + noise)
        test_signal = np.random.randn(sample_rate) * 5
        test_signal[500] = 50  # Add impulse

        result = detector.detect(test_signal)
        # The anomalous signal should have higher score
        assert result["anomaly_score"] > 0.0

    def test_detect_returns_signature_type(self) -> None:
        """Test that detection returns signature type."""
        detector = SpectralVibrationDetector()

        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 100 * t)
        detector.fit(signal)

        result = detector.detect(signal)
        assert "signature_type" in result
        assert result["signature_type"] in [e.value for e in VibrationSignatureType]

    def test_extract_features(self) -> None:
        """Test feature extraction for ML fusion."""
        detector = SpectralVibrationDetector()

        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 100 * t)
        detector.fit(signal)

        features = detector.extract_features(signal)
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2  # [batch, features]
        assert features.shape[0] == 1  # Single sample

    def test_detect_not_fitted_raises(self) -> None:
        """Test that detection before fitting raises exception."""
        from omni_mercury_engine.core.exceptions import DetectorException

        detector = SpectralVibrationDetector()
        with pytest.raises((ValueError, RuntimeError, DetectorException)):
            detector.detect(np.random.randn(100))


class TestSpectralGraphLayer:
    """Tests for SpectralGraphLayer."""

    def test_forward_pass(self) -> None:
        """Test forward pass through graph layer."""
        layer = SpectralGraphLayer(in_features=8, out_features=16)

        num_nodes = 32
        node_features = torch.randn(num_nodes, 8)

        # Create simple edge connectivity
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
            ]
        )
        edge_type = torch.zeros(5, dtype=torch.long)

        output = layer(node_features, edge_index, edge_type)

        assert output.shape == (num_nodes, 16)


class TestSpectralGNN:
    """Tests for SpectralGNN."""

    def test_forward_pass(self) -> None:
        """Test forward pass through GNN."""
        gnn = SpectralGNN(
            input_dim=2,
            hidden_dim=32,
            output_dim=16,
            num_layers=2,
        )

        num_nodes = 64
        node_features = torch.randn(num_nodes, 2)

        # Create edge connectivity
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.tensor(edges).T
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)

        output = gnn(node_features, edge_index, edge_type)

        assert output.shape == (16,)  # Graph-level output


class TestSpectralCNN:
    """Tests for SpectralCNN."""

    def test_forward_pass_unbatched(self) -> None:
        """Test forward pass with unbatched input."""
        cnn = SpectralCNN(
            input_channels=1,
            num_filters=16,
            output_dim=8,
        )

        spectrum = torch.randn(1, 256)  # [channels, length]
        output = cnn(spectrum)

        assert output.shape == (8,)

    def test_forward_pass_batched(self) -> None:
        """Test forward pass with batched input."""
        cnn = SpectralCNN(
            input_channels=1,
            num_filters=16,
            output_dim=8,
        )

        spectra = torch.randn(4, 1, 256)  # [batch, channels, length]
        output = cnn(spectra)

        assert output.shape == (4, 8)


class TestPhononInteractionNetwork:
    """Tests for PhononInteractionNetwork."""

    def test_forward_pass(self) -> None:
        """Test forward pass through phonon network."""
        network = PhononInteractionNetwork(
            num_modes=32,
            hidden_dim=16,
            interaction_order=3,
        )

        amplitudes = torch.randn(32).abs()
        result = network(amplitudes)

        assert "total_energy" in result
        assert "coupling_matrix" in result
        assert "scattering_rates" in result
        assert "anharmonic_score" in result

        assert result["coupling_matrix"].shape == (32, 32)


class TestMLIPVibrationEncoder:
    """Tests for MLIPVibrationEncoder."""

    def test_forward_pass(self) -> None:
        """Test forward pass through MLIP encoder."""
        encoder = MLIPVibrationEncoder(
            num_freq_bins=128,
            descriptor_dim=32,
            output_dim=16,
        )

        spectrum = torch.randn(128).abs()
        output = encoder(spectrum)

        assert output.shape == (16,)

    def test_forward_pass_batched(self) -> None:
        """Test forward pass with batched input."""
        encoder = MLIPVibrationEncoder(
            num_freq_bins=128,
            descriptor_dim=32,
            output_dim=16,
        )

        spectra = torch.randn(4, 128).abs()
        output = encoder(spectra)

        assert output.shape == (4, 16)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_stft(self) -> None:
        """Test Short-Time Fourier Transform computation."""
        signal = np.random.randn(1000)
        freqs, times, spectrogram = compute_short_time_fourier_transform(
            signal, window_size=128, hop_size=32
        )

        assert len(freqs) == 64  # Half of window_size
        assert len(times) > 0
        assert spectrogram.shape[0] == 64

    def test_compute_wavelet_decomposition(self) -> None:
        """Test wavelet decomposition."""
        signal = np.random.randn(256)
        coeffs = compute_wavelet_decomposition(signal, levels=3)

        assert len(coeffs) == 4  # approx + 3 detail levels
        assert len(coeffs[0]) < len(signal)

    def test_detect_peaks_with_harmonics(self) -> None:
        """Test peak detection with harmonics."""
        # Create spectrum with clear peaks
        freqs = np.linspace(0, 500, 512)
        spectrum = np.zeros(512)

        # Add peaks at 100 Hz and harmonics
        fundamental_idx = 102
        spectrum[fundamental_idx] = 1.0
        spectrum[fundamental_idx * 2] = 0.5  # 2nd harmonic
        spectrum[fundamental_idx * 3] = 0.25  # 3rd harmonic

        # Add some noise
        spectrum += np.random.randn(512) * 0.05

        peaks = detect_peaks_with_harmonics(spectrum, freqs, num_harmonics=3, min_prominence=0.1)

        assert len(peaks) > 0
        # First peak should be the fundamental
        assert peaks[0]["amplitude"] >= peaks[1]["amplitude"] if len(peaks) > 1 else True


class TestVibrationDiagnostic:
    """Tests for VibrationDiagnostic dataclass."""

    def test_diagnostic_creation(self) -> None:
        """Test creating a diagnostic result."""
        diagnostic = VibrationDiagnostic(
            signature_type=VibrationSignatureType.BEARING_FAULT,
            confidence=0.85,
            fault_frequency=123.5,
            severity_score=0.7,
            recommended_action="Inspect bearing immediately",
            time_to_failure=168.0,
            supporting_features={"crest_factor": 4.5},
        )

        assert diagnostic.signature_type == VibrationSignatureType.BEARING_FAULT
        assert diagnostic.confidence == 0.85
        assert diagnostic.time_to_failure == 168.0


class TestSpectralFeatures:
    """Tests for SpectralFeatures dataclass."""

    def test_features_creation(self) -> None:
        """Test creating spectral features."""
        features = SpectralFeatures(
            power_spectrum=np.zeros(512),
            dominant_frequencies=[(100.0, 1.0), (200.0, 0.5)],
            harmonic_ratios=np.array([2.0, 3.0]),
            spectral_centroid=150.0,
            spectral_bandwidth=50.0,
            spectral_rolloff=400.0,
            spectral_flatness=0.3,
            crest_factor=3.5,
            kurtosis=2.0,
            phonon_coupling=0.4,
            graph_laplacian_spectrum=np.zeros(10),
            cnn_features=np.zeros(32),
            schumann_alignment=0.1,
        )

        assert len(features.dominant_frequencies) == 2
        assert features.spectral_centroid == 150.0
