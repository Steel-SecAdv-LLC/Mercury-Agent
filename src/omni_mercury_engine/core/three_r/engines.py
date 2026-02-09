"""
Mercury Agent - 3R Mechanism Engines
Copyright (C) 2025 Steel Security Advisory LLC

Core engines for the 3R (Recursion-Resonance-Refactoring) Mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import fft, signal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class RecursionEngine:
    """
    Implements recursive self-referential processing for hierarchical
    feature extraction and multi-level optimization.

    The recursion component R(x) of the AAFE provides hierarchical feature
    extraction through self-referential processing patterns.
    """

    def __init__(self, max_depth: int = 5) -> None:
        """
        Initialize RecursionEngine.

        Args:
            max_depth: Maximum recursion depth to prevent infinite loops
        """
        self.max_depth = max_depth
        self.recursion_cache: dict[str, Any] = {}

    def recursive_transform(
        self,
        data: NDArray[Any],
        transform_fn: Callable[..., Any],
        depth: int = 0,
        threshold: float = 0.01,
    ) -> NDArray[Any]:
        """
        Apply recursive transformation until convergence.

        Args:
            data: Input data array
            transform_fn: Transformation function to apply
            depth: Current recursion depth
            threshold: Convergence threshold

        Returns:
            Transformed data array
        """
        if depth >= self.max_depth:
            return data

        transformed = transform_fn(data)

        diff = np.linalg.norm(transformed - data)
        if diff < threshold:
            return transformed

        return self.recursive_transform(transformed, transform_fn, depth + 1, threshold)

    def hierarchical_feature_extraction(
        self, data: NDArray[Any], num_levels: int = 3
    ) -> list[NDArray[Any]]:
        """
        Extract features at multiple hierarchical levels.

        Args:
            data: Input data array
            num_levels: Number of hierarchy levels

        Returns:
            List of feature arrays at each level
        """
        features = []
        current_data = data

        for level in range(num_levels):
            level_features = self._extract_level_features(current_data, level)
            features.append(level_features)

            if level < num_levels - 1:
                current_data = self._downsample(level_features)

        return features

    def compute_recursion_score(self, data: NDArray[Any]) -> float:
        """
        Compute the R(x) recursion component score.

        Args:
            data: Input data for recursion analysis

        Returns:
            Recursion score in [0, 1] range
        """
        features = self.hierarchical_feature_extraction(data)

        # Compute feature consistency across levels
        if len(features) < 2:
            return 0.5

        consistency_scores = []
        for i in range(len(features) - 1):
            f1 = features[i].flatten()
            f2 = features[i + 1].flatten()
            min_len = min(len(f1), len(f2))
            if min_len > 0:
                corr = np.corrcoef(f1[:min_len], f2[:min_len])[0, 1]
                if np.isfinite(corr):
                    consistency_scores.append(abs(corr))

        if not consistency_scores:
            return 0.5

        return float(np.clip(np.mean(consistency_scores), 0.0, 1.0))

    def _extract_level_features(self, data: NDArray[Any], level: int) -> NDArray[Any]:
        """Extract features at a specific hierarchy level."""
        if data.ndim == 1:
            window_size = max(3, len(data) // (2**level))
            return self._sliding_window_stats(data, window_size)
        else:
            return np.mean(data, axis=1, keepdims=True)

    def _sliding_window_stats(self, data: NDArray[Any], window_size: int) -> NDArray[Any]:
        """Compute sliding window statistics."""
        if len(data) < window_size:
            return np.array([np.mean(data), np.std(data), np.max(data)])

        features = []
        for i in range(0, len(data) - window_size + 1, window_size // 2):
            window = data[i : i + window_size]
            features.extend([np.mean(window), np.std(window), np.max(window) - np.min(window)])

        return np.array(features)

    def _downsample(self, data: NDArray[Any]) -> NDArray[Any]:
        """Downsample data by factor of 2."""
        if len(data) <= 2:
            return data
        return data[::2]


class ResonanceEngine:
    """
    Implements frequency-domain signal amplification using Fourier analysis
    for pattern enhancement and anomaly detection.

    The resonance component H(omega) of the AAFE provides frequency-domain
    analysis for detecting harmonic patterns in data.
    """

    def __init__(self, sampling_rate: float = 1.0) -> None:
        """
        Initialize ResonanceEngine.

        Args:
            sampling_rate: Sampling rate for frequency analysis
        """
        self.sampling_rate = sampling_rate

    def compute_resonance_spectrum(
        self, signal_data: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Compute frequency spectrum using FFT.

        Args:
            signal_data: Input signal data

        Returns:
            Tuple of (frequencies, magnitudes)
        """
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()

        fft_result = np.array(fft.fft(signal_data))
        frequencies = np.array(fft.fftfreq(len(signal_data), 1.0 / self.sampling_rate))
        magnitudes = np.abs(fft_result)

        positive_freq_idx = frequencies >= 0
        return frequencies[positive_freq_idx], magnitudes[positive_freq_idx]

    def amplify_resonant_frequencies(
        self,
        signal_data: NDArray[Any],
        target_frequencies: list[float] | None = None,
        amplification_factor: float = 2.0,
    ) -> NDArray[Any]:
        """
        Amplify specific resonant frequencies in the signal.

        Args:
            signal_data: Input signal
            target_frequencies: Frequencies to amplify (auto-detected if None)
            amplification_factor: Amplification multiplier

        Returns:
            Signal with amplified frequencies
        """
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()

        fft_result = np.array(fft.fft(signal_data))
        frequencies = np.array(fft.fftfreq(len(signal_data), 1.0 / self.sampling_rate))

        if target_frequencies is None:
            target_frequencies = self._detect_dominant_frequencies(frequencies, np.abs(fft_result))

        for target_freq in target_frequencies:
            freq_idx = np.argmin(np.abs(frequencies - target_freq))
            fft_result[freq_idx] *= amplification_factor

            mirror_idx = len(fft_result) - freq_idx
            if mirror_idx < len(fft_result):
                fft_result[mirror_idx] *= amplification_factor

        return np.real(np.array(fft.ifft(fft_result)))

    def compute_resonance_score(self, signal_data: NDArray[Any]) -> float:
        """
        Compute the H(omega) resonance component score.

        Args:
            signal_data: Input signal for resonance analysis

        Returns:
            Resonance score in [0, 1] range
        """
        frequencies, magnitudes = self.compute_resonance_spectrum(signal_data)

        if len(magnitudes) == 0:
            return 0.5

        # Compute harmonic content ratio
        total_energy = np.sum(magnitudes**2)
        if total_energy == 0:
            return 0.5

        # Find dominant frequencies
        dominant_idx = magnitudes > np.mean(magnitudes)
        harmonic_energy = np.sum(magnitudes[dominant_idx] ** 2)

        harmonic_ratio = harmonic_energy / total_energy

        return float(np.clip(harmonic_ratio, 0.0, 1.0))

    def _detect_dominant_frequencies(
        self, frequencies: NDArray[Any], magnitudes: NDArray[Any], num_peaks: int = 5
    ) -> list[float]:
        """Detect dominant frequency peaks in the spectrum."""
        peaks, _ = signal.find_peaks(magnitudes, height=np.max(magnitudes) * 0.1)

        if len(peaks) == 0:
            return []

        peak_magnitudes = magnitudes[peaks]
        top_peak_idx = np.argsort(peak_magnitudes)[-num_peaks:]

        return [frequencies[peaks[i]] for i in top_peak_idx]

    def detect_resonance_anomalies(
        self, signal_data: NDArray[Any], threshold_std: float = 3.0
    ) -> dict[str, Any]:
        """
        Detect anomalous frequency components.

        Args:
            signal_data: Input signal
            threshold_std: Standard deviation threshold for anomaly detection

        Returns:
            Dictionary with anomaly detection results
        """
        frequencies, magnitudes = self.compute_resonance_spectrum(signal_data)

        mean_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)
        threshold = mean_magnitude + threshold_std * std_magnitude

        anomalous_freq_idx = magnitudes > threshold
        anomalous_frequencies = frequencies[anomalous_freq_idx]
        anomalous_magnitudes = magnitudes[anomalous_freq_idx]

        return {
            "is_anomalous": len(anomalous_frequencies) > 0,
            "num_anomalies": len(anomalous_frequencies),
            "anomalous_frequencies": anomalous_frequencies.tolist(),
            "anomalous_magnitudes": anomalous_magnitudes.tolist(),
            "threshold": threshold,
            "max_magnitude": np.max(magnitudes),
            "mean_magnitude": mean_magnitude,
        }
