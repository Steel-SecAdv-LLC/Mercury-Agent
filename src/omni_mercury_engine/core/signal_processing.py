# Copyright (C) 2025 Steel Security Advisors LLC
"""Advanced Signal Processing for Enhanced Noise Reduction.

This module provides adaptive filtering techniques for improved anomaly detection:
- Wavelet Denoising: Better for non-stationary signals (seismic, medical vitals)
- Kalman Filtering: Optimal for temporal noise in time-series data
- Savitzky-Golay Filters: Preserve signal features while removing noise
- Adaptive Bandpass Filtering: Dynamically adjust filter parameters

These methods complement the basic FFT-based filtering in the fusion module,
providing domain-specific noise reduction for humanitarian applications like
crisis detection, pandemic monitoring, and medical diagnostics.

Research sources:
- Donoho & Johnstone (1994): Wavelet shrinkage denoising
- Kalman (1960): Optimal linear filtering
- Savitzky & Golay (1964): Smoothing and differentiation of data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


class FilterType(Enum):
    """Available filter types for noise reduction."""

    FFT_LOWPASS = "fft_lowpass"
    WAVELET = "wavelet"
    KALMAN = "kalman"
    SAVITZKY_GOLAY = "savitzky_golay"
    ADAPTIVE_BANDPASS = "adaptive_bandpass"
    MEDIAN = "median"
    EXPONENTIAL_MOVING_AVERAGE = "ema"
    EMA = "ema"


@dataclass
class FilterConfig:
    """Configuration for adaptive filtering.

    Attributes:
        filter_type: Type of filter to apply
        window_size: Window size for windowed filters (Savitzky-Golay, median)
        poly_order: Polynomial order for Savitzky-Golay filter
        cutoff_freq: Cutoff frequency for bandpass/lowpass filters (0-1 normalized)
        kalman_process_noise: Process noise covariance for Kalman filter
        kalman_measurement_noise: Measurement noise covariance for Kalman filter
        wavelet_level: Decomposition level for wavelet denoising
        wavelet_threshold: Threshold multiplier for wavelet coefficient shrinkage
        ema_alpha: Smoothing factor for exponential moving average (0-1)
    """

    filter_type: FilterType = FilterType.FFT_LOWPASS
    window_size: int = 5
    poly_order: int = 2
    cutoff_freq: float = 0.5
    kalman_process_noise: float = 1e-5
    kalman_measurement_noise: float = 1e-2
    wavelet_level: int = 3
    wavelet_threshold: float = 1.0
    ema_alpha: float = 0.3
    extra_params: dict[str, Any] = field(default_factory=dict)


class AdaptiveNoiseFilter:
    """Adaptive noise filtering for enhanced anomaly detection.

    Provides multiple filtering strategies that can be selected based on
    signal characteristics and domain requirements.

    Example:
        >>> filter = AdaptiveNoiseFilter(FilterConfig(filter_type=FilterType.KALMAN))
        >>> clean_signal = filter.apply(noisy_signal)
    """

    def __init__(self, config: FilterConfig | None = None) -> None:
        """Initialize adaptive noise filter.

        Args:
            config: Filter configuration. Uses defaults if None.
        """
        self.config = config or FilterConfig()
        self._kalman_state: np.ndarray[Any, Any] | None = None
        self._kalman_covariance: float = 1.0

    def apply(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply configured filter to input data.

        Args:
            data: Input signal array

        Returns:
            Filtered signal array
        """
        if len(data) < 3:
            return data.copy()

        filter_methods = {
            FilterType.FFT_LOWPASS: self._fft_lowpass,
            FilterType.WAVELET: self._wavelet_denoise,
            FilterType.KALMAN: self._kalman_filter,
            FilterType.SAVITZKY_GOLAY: self._savitzky_golay,
            FilterType.ADAPTIVE_BANDPASS: self._adaptive_bandpass,
            FilterType.MEDIAN: self._median_filter,
            FilterType.EXPONENTIAL_MOVING_AVERAGE: self._ema_filter,
        }

        method = filter_methods.get(self.config.filter_type, self._fft_lowpass)
        return method(data)

    def _fft_lowpass(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """FFT-based lowpass filter (original method from fusion.py)."""
        fft_vals = np.fft.fft(data)
        cutoff_idx = int(len(fft_vals) * self.config.cutoff_freq)
        fft_vals[cutoff_idx:] = 0
        filtered = np.fft.ifft(fft_vals)
        return np.real(filtered)

    def _wavelet_denoise(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Wavelet denoising using soft thresholding.

        Implements Donoho-Johnstone wavelet shrinkage for non-stationary signals. Particularly
        effective for seismic data, medical vitals, and crisis signals.
        """
        level = min(self.config.wavelet_level, int(np.log2(len(data))) - 1)
        if level < 1:
            return data.copy()

        coeffs = self._haar_wavelet_decompose(data, level)

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = self.config.wavelet_threshold * sigma * np.sqrt(2 * np.log(len(data)))

        denoised_coeffs = []
        denoised_coeffs.append(coeffs[0])
        for i in range(1, len(coeffs)):
            denoised = self._soft_threshold(coeffs[i], threshold)
            denoised_coeffs.append(denoised)

        return self._haar_wavelet_reconstruct(denoised_coeffs, len(data))

    def _haar_wavelet_decompose(
        self, data: np.ndarray[Any, Any], level: int
    ) -> list[np.ndarray[Any, Any]]:
        """Simple Haar wavelet decomposition."""
        coeffs = []
        current = data.copy()

        for _ in range(level):
            n = len(current)
            if n < 2:
                break
            n_half = n // 2
            approx = np.zeros(n_half)
            detail = np.zeros(n_half)

            for i in range(n_half):
                approx[i] = (current[2 * i] + current[2 * i + 1]) / np.sqrt(2)
                detail[i] = (current[2 * i] - current[2 * i + 1]) / np.sqrt(2)

            coeffs.append(detail)
            current = approx

        coeffs.insert(0, current)
        return coeffs

    def _haar_wavelet_reconstruct(
        self, coeffs: list[np.ndarray[Any, Any]], original_length: int
    ) -> np.ndarray[Any, Any]:
        """Simple Haar wavelet reconstruction."""
        if len(coeffs) == 0:
            return np.zeros(original_length)

        current = coeffs[0]

        for i in range(len(coeffs) - 1, 0, -1):
            detail = coeffs[i]
            n = min(len(current), len(detail))
            reconstructed = np.zeros(2 * n)

            for j in range(n):
                reconstructed[2 * j] = (current[j] + detail[j]) / np.sqrt(2)
                reconstructed[2 * j + 1] = (current[j] - detail[j]) / np.sqrt(2)

            current = reconstructed

        if len(current) > original_length:
            current = current[:original_length]
        elif len(current) < original_length:
            current = np.pad(current, (0, original_length - len(current)))

        return current

    def _soft_threshold(self, data: np.ndarray[Any, Any], threshold: float) -> np.ndarray[Any, Any]:
        """Apply soft thresholding to wavelet coefficients."""
        return np.asarray(np.sign(data) * np.maximum(np.abs(data) - threshold, 0))  # type: ignore[no-any-return, unused-ignore]

    def _kalman_filter(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Kalman filter for optimal temporal noise reduction.

        Implements a simple 1D Kalman filter optimal for time-series data with Gaussian noise.
        Particularly effective for sensor data and continuous monitoring applications.
        """
        n = len(data)
        filtered = np.zeros(n)

        x_est = data[0]
        p_est = 1.0

        q = self.config.kalman_process_noise
        r = self.config.kalman_measurement_noise

        for i in range(n):
            x_pred = x_est
            p_pred = p_est + q

            k = p_pred / (p_pred + r)

            x_est = x_pred + k * (data[i] - x_pred)
            p_est = (1 - k) * p_pred

            filtered[i] = x_est

        self._kalman_state = np.array([x_est])
        self._kalman_covariance = p_est

        return filtered

    def _savitzky_golay(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Savitzky-Golay filter for smoothing while preserving features.

        Preserves higher moments of the signal (peaks, valleys) better than simple moving average.
        Ideal for spectroscopic data and signals where feature preservation is critical.
        """
        window = self.config.window_size
        if window % 2 == 0:
            window += 1
        window = min(window, len(data))
        if window < 3:
            return data.copy()

        poly_order = min(self.config.poly_order, window - 1)

        try:
            return np.asarray(signal.savgol_filter(data, window, poly_order))  # type: ignore[no-any-return, unused-ignore]
        except ValueError:
            return data.copy()

    def _adaptive_bandpass(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Adaptive bandpass filter with automatic frequency selection.

        Analyzes signal spectrum to determine optimal passband, then applies Butterworth bandpass
        filter. Useful when signal characteristics vary.
        """
        fft_vals = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        magnitude = np.abs(fft_vals)

        positive_mask = freqs > 0
        if not np.any(positive_mask):
            return data.copy()

        pos_freqs = freqs[positive_mask]
        pos_magnitude = magnitude[positive_mask]

        if len(pos_magnitude) == 0:
            return data.copy()

        peak_idx = np.argmax(pos_magnitude)
        center_freq = pos_freqs[peak_idx]

        bandwidth = self.config.cutoff_freq * 0.5
        low_freq = max(0.01, center_freq - bandwidth)
        high_freq = min(0.49, center_freq + bandwidth)

        if low_freq >= high_freq:
            return data.copy()

        try:
            b, a = signal.butter(2, [low_freq * 2, high_freq * 2], btype="band")
            return np.asarray(signal.filtfilt(b, a, data))  # type: ignore[no-any-return, unused-ignore]
        except ValueError:
            return data.copy()

    def _median_filter(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Median filter for impulse noise removal.

        Effective for removing salt-and-pepper noise and outliers while preserving edges. Useful for
        sensor data with occasional spikes.
        """
        return np.asarray(signal.medfilt(data, kernel_size=min(self.config.window_size, len(data))))  # type: ignore[no-any-return, unused-ignore]

    def _ema_filter(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Exponential moving average filter.

        Simple but effective for real-time applications where computational efficiency is important.
        """
        alpha = self.config.ema_alpha
        filtered = np.zeros_like(data)
        filtered[0] = data[0]

        for i in range(1, len(data)):
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1]

        return filtered

    def reset_state(self) -> None:
        """Reset filter state (for Kalman filter)."""
        self._kalman_state = None
        self._kalman_covariance = 1.0


class MultiStageFilter:
    """Multi-stage filtering pipeline for comprehensive noise reduction.

    Combines multiple filtering techniques in sequence for optimal results.
    Useful for complex signals requiring different noise reduction strategies.

    Example:
        >>> pipeline = MultiStageFilter([
        ...     FilterConfig(filter_type=FilterType.MEDIAN),
        ...     FilterConfig(filter_type=FilterType.SAVITZKY_GOLAY),
        ... ])
        >>> clean_signal = pipeline.apply(noisy_signal)
    """

    def __init__(self, stages: list[FilterConfig] | None = None) -> None:
        """Initialize multi-stage filter.

        Args:
            stages: List of filter configurations to apply in sequence.
        """
        self.stages = stages or []
        self.filters = [AdaptiveNoiseFilter(config) for config in self.stages]

    def apply(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply all filter stages in sequence.

        Args:
            data: Input signal array

        Returns:
            Filtered signal array
        """
        result = data.copy()
        for filter_instance in self.filters:
            result = filter_instance.apply(result)
        return result

    def add_stage(self, config: FilterConfig) -> None:
        """Add a filter stage to the pipeline.

        Args:
            config: Filter configuration for new stage
        """
        self.stages.append(config)
        self.filters.append(AdaptiveNoiseFilter(config))


def compute_rolling_statistics(
    data: np.ndarray[Any, Any],
    window_size: int = 10,
) -> dict[str, np.ndarray[Any, Any]]:
    """Compute rolling window statistics for feature engineering.

    Args:
        data: Input signal array
        window_size: Size of rolling window

    Returns:
        Dictionary containing rolling mean, std, min, max
    """
    if len(data) < window_size:
        window_size = max(1, len(data))

    rolling_mean = uniform_filter1d(data, size=window_size, mode="nearest")

    rolling_std = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        rolling_std[i] = np.std(data[start:end])

    rolling_min = np.zeros_like(data)
    rolling_max = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        rolling_min[i] = np.min(data[start:end])
        rolling_max[i] = np.max(data[start:end])

    return {
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_min": rolling_min,
        "rolling_max": rolling_max,
    }


def compute_temporal_lag_features(
    data: np.ndarray[Any, Any],
    lags: list[int] | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """Compute temporal lag features for multi-scale dependency analysis.

    Args:
        data: Input signal array
        lags: List of lag values (default: [1, 2, 5, 10])

    Returns:
        Dictionary containing lagged features and differences
    """
    if lags is None:
        lags = [1, 2, 5, 10]

    features: dict[str, np.ndarray[Any, Any]] = {}

    for lag in lags:
        if lag >= len(data):
            continue

        lagged = np.zeros_like(data)
        lagged[lag:] = data[:-lag]
        lagged[:lag] = data[0]
        features[f"lag_{lag}"] = lagged

        diff = data - lagged
        features[f"diff_{lag}"] = diff

    return features


def compute_interaction_features(
    features_dict: dict[str, np.ndarray[Any, Any]],
) -> dict[str, np.ndarray[Any, Any]]:
    """Compute interaction features between detector outputs.

    Creates cross-correlation and product features for enhanced detection.

    Args:
        features_dict: Dictionary mapping feature names to arrays

    Returns:
        Dictionary containing interaction features
    """
    interactions: dict[str, np.ndarray[Any, Any]] = {}
    feature_names = list(features_dict.keys())

    for i, name1 in enumerate(feature_names):
        for name2 in feature_names[i + 1 :]:
            feat1 = features_dict[name1]
            feat2 = features_dict[name2]

            if feat1.shape != feat2.shape:
                continue

            interactions[f"{name1}_x_{name2}"] = feat1 * feat2

            if len(feat1) > 1:
                corr = np.corrcoef(feat1.flatten(), feat2.flatten())[0, 1]
                if not np.isnan(corr):
                    interactions[f"{name1}_corr_{name2}"] = np.full_like(feat1, corr)

    return interactions
