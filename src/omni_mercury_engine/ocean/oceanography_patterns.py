# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Oceanography-Inspired Pattern Recognition Module.

Inspired by oceanographic principles: acoustic sensing (sonar/echo sounders),
wave pattern analysis (currents, tides, geophysical fluid dynamics), depth-based
stratification (deep sea soundings), multi-sensor fusion (interdisciplinary approach),
and HMS Challenger expedition's systematic exploration (70,000 nautical miles).

Key influences:
- Acoustic sensing and sonar for noisy environment signal processing
- Wave pattern analysis for cyclical/periodic anomaly detection
- Depth-based hierarchical analysis (surface, mid-level, deep features)
- Multi-sensor fusion from interdisciplinary oceanography
- Remote sensing for distributed detection
- HMS Challenger systematic sampling (492 soundings, 133 dredges, 151 trawls)

Research source: Wikipedia - Oceanography
(https://en.wikipedia.org/wiki/Oceanography)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class DepthLevel(Enum):
    """Data depth levels inspired by ocean stratification."""

    SURFACE = 1
    MID_LEVEL = 2
    DEEP = 3


@dataclass
class WavePattern:
    """Represents a wave pattern in data."""

    frequency: float
    amplitude: float
    phase: float
    period: float


class OceanographyPatterns:
    """Pattern recognition system inspired by oceanographic methods.

    Implements acoustic sensing, wave analysis, depth-based stratification, and multi-sensor fusion
    for anomaly detection.
    """

    def __init__(
        self,
        depth_levels: int = 3,
        acoustic_sensitivity: float = 0.7,
        wave_detection_threshold: float = 0.5,
    ):
        """Initialize oceanography-inspired pattern recognition.

        Args:
            depth_levels: Number of depth levels for hierarchical analysis
            acoustic_sensitivity: Sensitivity for acoustic sensing (0-1)
            wave_detection_threshold: Threshold for wave pattern detection
        """
        self.depth_levels = depth_levels
        self.acoustic_sensitivity = acoustic_sensitivity
        self.wave_detection_threshold = wave_detection_threshold
        self.pattern_history: list[WavePattern] = []

    def acoustic_sensing(
        self, data: np.ndarray[Any, Any], pulse_frequency: float = 1.0
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Acoustic sensing analogous to sonar/echo sounder.

        Inspired by 1914 first acoustic sea depth measurement and
        echo sounders used in oceanography for depth profiling.

        Args:
            data: Input data to "scan" with acoustic pulse
            pulse_frequency: Frequency of acoustic pulse

        Returns:
            Dictionary with reflection patterns and depths
        """
        pulse = np.sin(2 * np.pi * pulse_frequency * np.arange(len(data)))

        reflection = np.correlate(data.flatten(), pulse, mode="same")

        depths = np.abs(reflection)
        depths = depths / (np.max(depths) + 1e-8)

        time_of_flight = np.argmax(np.abs(reflection))

        return {
            "reflection_pattern": reflection.astype(np.float32),
            "depth_profile": depths.astype(np.float32),
            "time_of_flight": float(time_of_flight),
            "signal_strength": float(np.max(np.abs(reflection))),
        }

    def wave_pattern_analysis(self, time_series: np.ndarray[Any, Any]) -> list[WavePattern]:
        """Analyze wave patterns in time-series data.

        Inspired by oceanographic analysis of ocean currents, waves,
        tides, and geophysical fluid dynamics.

        Args:
            time_series: Time-series data to analyze

        Returns:
            List of detected wave patterns
        """
        fft = np.fft.fft(time_series)
        frequencies = np.fft.fftfreq(len(time_series))

        amplitudes = np.abs(fft)
        phases = np.angle(fft)

        significant_indices = np.where(
            amplitudes > self.wave_detection_threshold * np.max(amplitudes)
        )[0]

        patterns = []
        for idx in significant_indices[:10]:
            if frequencies[idx] > 0:
                pattern = WavePattern(
                    frequency=float(np.abs(frequencies[idx])),
                    amplitude=float(amplitudes[idx]),
                    phase=float(phases[idx]),
                    period=1.0 / (np.abs(frequencies[idx]) + 1e-8),
                )
                patterns.append(pattern)

        self.pattern_history.extend(patterns)

        return patterns

    def depth_based_stratification(
        self, data: np.ndarray[Any, Any]
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Hierarchical analysis inspired by ocean depth stratification.

        Inspired by deep sea soundings, pressure/depth relationships,
        and vertical profiling in oceanography.

        Args:
            data: Input data to stratify

        Returns:
            Dictionary with features at different depth levels
        """
        flat_data = data.flatten()
        segment_size = len(flat_data) // self.depth_levels

        stratified = {}

        for level in range(self.depth_levels):
            start_idx = level * segment_size
            end_idx = start_idx + segment_size if level < self.depth_levels - 1 else len(flat_data)

            segment = flat_data[start_idx:end_idx]

            level_name = ["surface", "mid_level", "deep"][level] if level < 3 else f"level_{level}"

            stratified[level_name] = np.array(
                [np.mean(segment), np.std(segment), np.min(segment), np.max(segment)]
            ).astype(np.float32)

        return stratified

    def multi_sensor_fusion(
        self, sensor_data: dict[str, np.ndarray[Any, Any]], weights: dict[str, float] | None = None
    ) -> np.ndarray[Any, Any]:
        """Multi-sensor fusion inspired by interdisciplinary oceanography.

        Oceanographers combine astronomy, biology, chemistry, geography,
        geology, hydrology, meteorology, and physics for comprehensive
        understanding - analogous to multi-modal data fusion.

        Args:
            sensor_data: Dictionary of sensor readings (different modalities)
            weights: Optional weights for each sensor

        Returns:
            Fused feature representation
        """
        if weights is None:
            weights = {key: 1.0 / len(sensor_data) for key in sensor_data}

        fused_features: list[Any] = []  # type: ignore[var-annotated, unused-ignore]

        for sensor_name, data in sensor_data.items():
            weight = weights.get(sensor_name, 1.0)

            features = (
                np.array(
                    [np.mean(data), np.std(data), np.percentile(data, 25), np.percentile(data, 75)]
                )
                * weight
            )

            fused_features.extend(features)

        return np.array(fused_features).astype(np.float32)

    def systematic_sampling(
        self, data: np.ndarray[Any, Any], num_samples: int = 492
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Systematic sampling inspired by HMS Challenger expedition.

        HMS Challenger (1872-1876): 492 deep sea soundings, 133 bottom dredges,
        151 open water trawls across 70,000 nautical miles for comprehensive
        ocean exploration.

        Args:
            data: Data to sample from
            num_samples: Number of systematic samples (default 492 like Challenger)

        Returns:
            Tuple of (sample_indices, sample_values)
        """
        flat_data = data.flatten()
        total_points = len(flat_data)

        step = max(1, total_points // num_samples)

        sample_indices = np.arange(0, total_points, step)[:num_samples]
        sample_values = flat_data[sample_indices]

        return sample_indices, sample_values

    def tidal_pattern_detection(
        self, time_series: np.ndarray[Any, Any], expected_period: float | None = None
    ) -> dict[str, float]:
        """Detect periodic patterns analogous to tides.

        Inspired by tidal observations recorded by Aristotle and Strabo
        (384-322 BC) and modern tidal prediction systems. The unsupervised
        path finds the dominant autocorrelation peak; when a constituent
        period is known a priori (the standard situation in tidal work,
        e.g. the M2 semidiurnal constituent), a classical harmonic
        least-squares fit at that exact frequency quantifies it:
        ``y ~ a cos(2 pi t / P) + b sin(2 pi t / P) + c`` solved in closed
        form, as in tidal harmonic analysis.

        Args:
            time_series: Time-series data (uniform sampling).
            expected_period: Known constituent period in samples. Must
                satisfy ``2 < expected_period <= n / 2`` so the fit sees at
                least two full cycles above the Nyquist limit.

        Returns:
            Dictionary with tidal pattern characteristics. With
            ``expected_period`` it additionally carries
            ``harmonic_amplitude`` / ``harmonic_phase_rad`` (of the fitted
            cosine), ``harmonic_r2`` (variance fraction the constituent
            explains) and ``period_deviation`` (relative gap between the
            autocorrelation-detected and the expected period).

        Raises:
            ValueError: ``expected_period`` outside ``(2, n / 2]``.
        """
        # Linear autocorrelation via zero-padded Wiener-Khinchin: O(n log n)
        # where np.correlate's direct form is O(n^2) — tide-gauge records are
        # long time series. Equal to the direct form to floating point.
        n_acf = len(time_series)
        padded = np.zeros(2 * n_acf)
        padded[:n_acf] = time_series
        acf = np.fft.irfft(np.abs(np.fft.rfft(padded)) ** 2)[:n_acf]
        acf = acf / acf[0]

        peaks = []
        for i in range(1, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.3:
                peaks.append(i)

        if peaks:
            dominant_period = peaks[0]
            strength = acf[dominant_period]
        else:
            dominant_period = 0
            strength = 0.0

        result = {
            "detected_period": float(dominant_period),
            "pattern_strength": float(strength),
            "is_periodic": bool(strength > 0.5),
            "autocorrelation_max": float(np.max(acf[1:])),
        }

        if expected_period is not None:
            n = len(time_series)
            if not 2.0 < expected_period <= n / 2.0:
                raise ValueError(
                    f"expected_period {expected_period} outside (2, {n / 2}]: the "
                    "harmonic fit needs two full cycles above the Nyquist limit"
                )
            t = np.arange(n, dtype=float)
            omega = 2.0 * np.pi / float(expected_period)
            design = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones(n)])
            coeffs, _, _, _ = np.linalg.lstsq(design, time_series, rcond=None)
            a, b, _c = (float(v) for v in coeffs)
            fitted = design @ coeffs
            residual = time_series - fitted
            total_var = float(np.var(time_series))
            r2 = 1.0 - float(np.var(residual)) / total_var if total_var > 1e-12 else 0.0
            result["harmonic_amplitude"] = float(np.hypot(a, b))
            # y = A cos(omega t - phase) with phase = atan2(b, a).
            result["harmonic_phase_rad"] = float(np.arctan2(b, a))
            result["harmonic_r2"] = max(0.0, r2)
            result["period_deviation"] = abs(dominant_period - expected_period) / float(
                expected_period
            )

        return result

    def climate_drift_detection(
        self, historical_data: np.ndarray[Any, Any], recent_data: np.ndarray[Any, Any]
    ) -> dict[str, float]:
        """Detect gradual drift analogous to climate change monitoring.

        Inspired by modern oceanographic research: ocean acidification,
        ocean heat content, sea level rise, coral bleaching monitoring.

        Args:
            historical_data: Historical baseline data
            recent_data: Recent data to compare

        Returns:
            Dictionary with drift metrics
        """
        hist_mean = np.mean(historical_data)
        hist_std = np.std(historical_data)

        recent_mean = np.mean(recent_data)
        recent_std = np.std(recent_data)

        mean_drift = (recent_mean - hist_mean) / (hist_std + 1e-8)

        std_ratio = recent_std / (hist_std + 1e-8)

        trend = np.polyfit(np.arange(len(recent_data)), recent_data.flatten(), 1)[0]

        return {
            "mean_drift": float(mean_drift),
            "std_ratio": float(std_ratio),
            "trend_slope": float(trend),
            "drift_detected": bool(abs(mean_drift) > 2.0),
            "distribution_shift": bool(abs(std_ratio - 1.0) > 0.3),
        }
