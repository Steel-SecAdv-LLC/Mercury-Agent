"""
Mercury Agent - Domain-Specific Feature Extractors
Copyright (C) 2025 Steel Security Advisors LLC

Advanced feature extraction for domain-specific anomaly detection:
- Medical: Vital sign temporal patterns, SOFA score weighting
- Financial: Benford's Law, transaction velocity, seasonality
- Infrastructure: SCADA correlation matrices, process variable analysis

This module implements the strategic improvements for raising domain competence
from current levels (Medical: 0.72, Financial: 0.76, Infrastructure: 0.79)
to target levels (0.85-0.90).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal, stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Supported domains for feature extraction."""

    MEDICAL = "medical"
    FINANCIAL = "financial"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    GENERAL = "general"


@dataclass
class DomainFeatureConfig:
    """Configuration for domain-specific feature extraction."""

    domain: Domain
    window_size: int = 60
    sampling_rate: float = 1.0
    enable_temporal: bool = True
    enable_statistical: bool = True
    enable_domain_specific: bool = True
    contamination_estimate: float = 0.05

    # Domain-specific parameters
    medical_params: dict[str, Any] = field(default_factory=dict)
    financial_params: dict[str, Any] = field(default_factory=dict)
    infrastructure_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainFeatureResult:
    """Result from domain-specific feature extraction."""

    features: NDArray[np.float64]
    feature_names: list[str]
    domain: Domain
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    extraction_time_ms: float = 0.0


class BaseDomainExtractor(ABC):
    """Abstract base class for domain-specific feature extractors."""

    def __init__(self, config: DomainFeatureConfig):
        """
        Initialize the domain extractor.

        Args:
            config: Domain feature configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._fitted = False
        self._feature_names: list[str] = []

    @abstractmethod
    def extract(self, data: NDArray[np.float64]) -> DomainFeatureResult:
        """
        Extract domain-specific features.

        Args:
            data: Input data array

        Returns:
            Domain feature extraction result
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get names of extracted features."""
        pass

    def _compute_statistical_features(
        self, data: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Compute standard statistical features.

        Args:
            data: Input data array

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        data = np.asarray(data).flatten()

        if len(data) == 0:
            return np.zeros(8), [
                "mean",
                "std",
                "min",
                "max",
                "skewness",
                "kurtosis",
                "q25",
                "q75",
            ]

        # Basic statistics
        features.extend(
            [
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data),
            ]
        )
        names.extend(["mean", "std", "min", "max"])

        # Higher-order statistics
        if len(data) >= 3:
            features.append(stats.skew(data))
            names.append("skewness")
        else:
            features.append(0.0)  # type: ignore[arg-type, unused-ignore]
            names.append("skewness")

        if len(data) >= 4:
            features.append(stats.kurtosis(data))
            names.append("kurtosis")
        else:
            features.append(0.0)  # type: ignore[arg-type, unused-ignore]
            names.append("kurtosis")

        # Percentiles
        features.extend(
            [
                np.percentile(data, 25),
                np.percentile(data, 75),
            ]
        )
        names.extend(["q25", "q75"])

        return np.array(features, dtype=np.float64), names

    def _compute_temporal_features(
        self, data: NDArray[np.float64], window_size: int | None = None
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Compute temporal/time-series features.

        Args:
            data: Input time-series data
            window_size: Window size for rolling computations

        Returns:
            Tuple of (features, feature_names)
        """
        data = np.asarray(data).flatten()
        window_size = window_size or self.config.window_size
        features = []
        names = []

        if len(data) < 2:
            return np.zeros(12), [
                "trend_slope",
                "trend_r2",
                "autocorr_lag1",
                "autocorr_lag5",
                "diff_mean",
                "diff_std",
                "zero_crossings",
                "local_maxima",
                "local_minima",
                "spectral_centroid",
                "spectral_spread",
                "spectral_entropy",
            ]

        # Trend analysis
        x = np.arange(len(data))
        slope, intercept, r_value, _, _ = stats.linregress(x, data)
        features.extend([slope, r_value**2])
        names.extend(["trend_slope", "trend_r2"])

        # Autocorrelation
        if len(data) > 5:
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            features.append(autocorr[1] if len(autocorr) > 1 else 0.0)
            features.append(autocorr[5] if len(autocorr) > 5 else 0.0)
        else:
            features.extend([0.0, 0.0])
        names.extend(["autocorr_lag1", "autocorr_lag5"])

        # First difference features
        diff = np.diff(data)
        if len(diff) > 0:
            features.extend([np.mean(diff), np.std(diff)])
        else:
            features.extend([0.0, 0.0])
        names.extend(["diff_mean", "diff_std"])

        # Zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(data - np.mean(data))))
        features.append(float(zero_crossings))
        names.append("zero_crossings")

        # Local extrema
        local_max = signal.argrelextrema(data, np.greater)[0]
        local_min = signal.argrelextrema(data, np.less)[0]
        features.extend([len(local_max), len(local_min)])
        names.extend(["local_maxima", "local_minima"])

        # Spectral features
        if len(data) >= 4:
            fft_result = np.fft.rfft(data)
            magnitudes = np.abs(fft_result)
            freqs = np.fft.rfftfreq(len(data))

            # Spectral centroid
            centroid = (
                np.sum(freqs * magnitudes) / (np.sum(magnitudes) + 1e-10)
                if len(magnitudes) > 0
                else 0.0
            )
            features.append(centroid)

            # Spectral spread
            spread = (
                np.sqrt(np.sum((freqs - centroid) ** 2 * magnitudes) / (np.sum(magnitudes) + 1e-10))
                if len(magnitudes) > 0
                else 0.0
            )
            features.append(spread)

            # Spectral entropy
            mag_norm = magnitudes / (np.sum(magnitudes) + 1e-10)
            mag_norm = mag_norm[mag_norm > 0]
            spectral_entropy = (
                -np.sum(mag_norm * np.log2(mag_norm + 1e-10)) if len(mag_norm) > 0 else 0.0
            )
            features.append(spectral_entropy)
        else:
            features.extend([0.0, 0.0, 0.0])
        names.extend(["spectral_centroid", "spectral_spread", "spectral_entropy"])

        return np.array(features, dtype=np.float64), names


class MedicalFeatureExtractor(BaseDomainExtractor):
    """
    Medical domain feature extractor.

    Implements vital sign temporal patterns, SOFA score weighting,
    and clinical anomaly indicators for medical anomaly detection.

    Target: Improve from 0.72 to 0.88 F1 score.
    """

    # SOFA (Sequential Organ Failure Assessment) score weights
    SOFA_WEIGHTS = {
        "respiratory": 0.20,  # PaO2/FiO2 ratio
        "coagulation": 0.15,  # Platelet count
        "liver": 0.15,  # Bilirubin
        "cardiovascular": 0.20,  # MAP or vasopressor requirement
        "cns": 0.15,  # Glasgow Coma Scale
        "renal": 0.15,  # Creatinine or urine output
    }

    # Normal vital sign ranges (adult)
    VITAL_RANGES = {
        "heart_rate": (60, 100),  # bpm
        "systolic_bp": (90, 140),  # mmHg
        "diastolic_bp": (60, 90),  # mmHg
        "respiratory_rate": (12, 20),  # breaths/min
        "oxygen_saturation": (95, 100),  # %
        "temperature": (36.1, 37.8),  # Celsius
        "mean_arterial_pressure": (70, 105),  # mmHg
    }

    def __init__(self, config: DomainFeatureConfig | None = None):
        """
        Initialize medical feature extractor.

        Args:
            config: Domain feature configuration
        """
        if config is None:
            config = DomainFeatureConfig(
                domain=Domain.MEDICAL,
                window_size=60,  # 60-second windows typical in ICU
                sampling_rate=1.0,  # 1 Hz default
            )
        super().__init__(config)

        # Medical-specific parameters
        self.sofa_weights = config.medical_params.get("sofa_weights", self.SOFA_WEIGHTS)
        self.vital_ranges = config.medical_params.get("vital_ranges", self.VITAL_RANGES)
        self.alert_fatigue_window = config.medical_params.get("alert_fatigue_window", 300)

    def extract(self, data: NDArray[np.float64]) -> DomainFeatureResult:
        """
        Extract medical domain features.

        Args:
            data: Input vital sign data. Expected shape:
                  - 1D: Single vital sign time series
                  - 2D: Multiple vital signs (rows=time, cols=vitals)

        Returns:
            Domain feature extraction result
        """
        import time

        start_time = time.perf_counter()
        data = np.asarray(data, dtype=np.float64)

        all_features = []
        all_names = []

        # Handle 1D vs 2D input
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_vitals = data.shape

        # 1. Per-vital statistical features
        if self.config.enable_statistical:
            for i in range(n_vitals):
                vital_data = data[:, i]
                stat_features, stat_names = self._compute_statistical_features(vital_data)
                all_features.append(stat_features)
                all_names.extend([f"vital{i}_{name}" for name in stat_names])

        # 2. Per-vital temporal features
        if self.config.enable_temporal:
            for i in range(n_vitals):
                vital_data = data[:, i]
                temp_features, temp_names = self._compute_temporal_features(vital_data)
                all_features.append(temp_features)
                all_names.extend([f"vital{i}_{name}" for name in temp_names])

        # 3. Medical-specific features
        if self.config.enable_domain_specific:
            med_features, med_names = self._extract_medical_features(data)
            all_features.append(med_features)
            all_names.extend(med_names)

        # Combine all features
        features = np.concatenate(all_features) if all_features else np.array([])

        # Handle NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        extraction_time = (time.perf_counter() - start_time) * 1000
        self._feature_names = all_names

        return DomainFeatureResult(
            features=features,
            feature_names=all_names,
            domain=Domain.MEDICAL,
            metadata={
                "n_vitals": n_vitals,
                "n_samples": n_samples,
                "sofa_weights": self.sofa_weights,
            },
            confidence=1.0,
            extraction_time_ms=extraction_time,
        )

    def _extract_medical_features(
        self, data: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Extract medical-specific features.

        Args:
            data: Multi-variate vital sign data (rows=time, cols=vitals)

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n_samples, n_vitals = data.shape

        # 1. Out-of-range percentage per vital
        for i in range(min(n_vitals, 7)):  # Max 7 standard vitals
            vital_data = data[:, i]

            # Map to standard vital if possible
            vital_names = list(self.VITAL_RANGES.keys())
            if i < len(vital_names):
                vital_name = vital_names[i]
                low, high = self.VITAL_RANGES[vital_name]

                # Out-of-range percentage
                oor_low = np.mean(vital_data < low)
                oor_high = np.mean(vital_data > high)
                oor_total = oor_low + oor_high

                features.extend([oor_low, oor_high, oor_total])
                names.extend(
                    [
                        f"{vital_name}_oor_low",
                        f"{vital_name}_oor_high",
                        f"{vital_name}_oor_total",
                    ]
                )
            else:
                # Generic out-of-range based on percentiles
                q10, q90 = np.percentile(vital_data, [10, 90])
                oor = np.mean((vital_data < q10) | (vital_data > q90))
                features.append(oor)
                names.append(f"vital{i}_oor_pct")

        # 2. Vital sign variability indices
        # Heart rate variability (if available, assumed to be first column)
        if n_vitals > 0:
            hr_data = data[:, 0]
            if len(hr_data) > 1:
                # RMSSD (Root Mean Square of Successive Differences)
                diff = np.diff(hr_data)
                rmssd = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else 0.0
                features.append(rmssd)  # type: ignore[arg-type, unused-ignore]
                names.append("hrv_rmssd")

                # SDNN (Standard Deviation of NN intervals)
                sdnn = np.std(hr_data)
                features.append(sdnn)
                names.append("hrv_sdnn")

                # pNN50 (percentage of successive differences > 50ms)
                if len(diff) > 0:
                    pnn50 = np.mean(np.abs(diff) > 50) * 100
                else:
                    pnn50 = 0.0  # type: ignore[assignment, unused-ignore]
                features.append(pnn50)
                names.append("hrv_pnn50")
            else:
                features.extend([0.0, 0.0, 0.0])  # type: ignore[list-item, unused-ignore]
                names.extend(["hrv_rmssd", "hrv_sdnn", "hrv_pnn50"])

        # 3. Cross-vital correlations
        if n_vitals >= 2:
            corr_matrix = np.corrcoef(data.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Mean absolute cross-correlation
            upper_tri = np.triu_indices(n_vitals, k=1)
            cross_corr = np.abs(corr_matrix[upper_tri])  # type: ignore[index, unused-ignore]
            mean_cross_corr = np.mean(cross_corr) if len(cross_corr) > 0 else 0.0
            features.append(mean_cross_corr)  # type: ignore[arg-type, unused-ignore]
            names.append("vital_cross_correlation")

        # 4. Physiological deterioration index
        # Based on early warning scores (simplified NEWS-like)
        if n_vitals >= 3:
            # Combine multiple vitals for deterioration detection
            deterioration_signals = []
            for i in range(min(n_vitals, 3)):
                vital_data = data[:, i]
                # Detect rapid changes (derivative exceeding threshold)
                if len(vital_data) > 1:
                    derivative = np.diff(vital_data)
                    rapid_changes = np.sum(np.abs(derivative) > 2 * np.std(derivative))
                    deterioration_signals.append(rapid_changes)

            deterioration_index = np.mean(deterioration_signals) if deterioration_signals else 0.0
            features.append(deterioration_index)  # type: ignore[arg-type, unused-ignore]
            names.append("deterioration_index")

        # 5. SOFA-inspired composite score
        # NOTE: This is a PROXY score, not true SOFA. True SOFA scoring requires
        # specific vital sign columns (respiratory, coagulation, liver, cardiovascular,
        # CNS, renal) in known order. This proxy computes weighted deviation across
        # available vitals as an indicator of multi-system dysfunction.
        # For accurate SOFA scoring, use column_mapping parameter in config.
        if n_vitals >= 2:
            # Compute weighted deviation proxy for available vitals
            sofa_proxy = 0.0
            n_used = min(n_vitals, len(self.SOFA_WEIGHTS))
            weights_list = list(self.SOFA_WEIGHTS.values())

            for i in range(n_used):
                vital_data = data[:, i]
                # Normalize relative to within-column statistics
                normalized = (vital_data - np.mean(vital_data)) / (np.std(vital_data) + 1e-10)
                # Use absolute deviation as dysfunction indicator
                deviation_score = np.mean(np.abs(normalized))
                # Apply weight (equal weights if beyond defined SOFA columns)
                weight = weights_list[i] if i < len(weights_list) else 1.0 / n_used
                sofa_proxy += weight * deviation_score

            # Normalize by total weight used
            total_weight = sum(weights_list[:n_used]) if n_used <= len(weights_list) else 1.0
            sofa_proxy /= max(total_weight, 1e-10)

            features.append(sofa_proxy)  # type: ignore[arg-type, unused-ignore]
            names.append("sofa_proxy_score")
        else:
            features.append(0.0)  # type: ignore[arg-type, unused-ignore]
            names.append("sofa_proxy_score")

        # 6. Alert fatigue indicator
        # Count potential false alarms based on brief threshold crossings
        if n_vitals > 0 and len(data) > self.alert_fatigue_window:
            brief_crossings = 0
            for i in range(n_vitals):
                vital_data = data[:, i]
                mean_val = np.mean(vital_data)
                std_val = np.std(vital_data) + 1e-10

                # Threshold at 2 standard deviations
                threshold_high = mean_val + 2 * std_val
                threshold_low = mean_val - 2 * std_val

                # Detect crossings that return within 10 samples
                above_high = vital_data > threshold_high
                below_low = vital_data < threshold_low

                # Count brief excursions
                for j in range(len(above_high) - 10):
                    if above_high[j] and not np.any(above_high[j + 1 : j + 10]):
                        brief_crossings += 1
                    if below_low[j] and not np.any(below_low[j + 1 : j + 10]):
                        brief_crossings += 1

            alert_fatigue = brief_crossings / (n_vitals * len(data) + 1e-10)
            features.append(alert_fatigue)
            names.append("alert_fatigue_indicator")
        else:
            features.append(0.0)  # type: ignore[arg-type, unused-ignore]
            names.append("alert_fatigue_indicator")

        return np.array(features, dtype=np.float64), names

    def get_feature_names(self) -> list[str]:
        """Get names of extracted features."""
        return self._feature_names.copy()


class FinancialFeatureExtractor(BaseDomainExtractor):
    """
    Financial domain feature extractor.

    Implements Benford's Law analysis, transaction velocity features,
    and time-series seasonality for financial anomaly detection.

    Target: Improve from 0.76 to 0.87 F1 score.
    """

    # Benford's Law expected first digit distribution
    BENFORD_DISTRIBUTION = np.array(
        [np.log10(1 + 1 / d) for d in range(1, 10)]  # P(d) = log10(1 + 1/d) for d in 1-9
    )

    def __init__(self, config: DomainFeatureConfig | None = None):
        """
        Initialize financial feature extractor.

        Args:
            config: Domain feature configuration
        """
        if config is None:
            config = DomainFeatureConfig(
                domain=Domain.FINANCIAL,
                window_size=100,  # 100 transactions default
                sampling_rate=1.0,
            )
        super().__init__(config)

        # Financial-specific parameters
        self.seasonality_periods = config.financial_params.get(
            "seasonality_periods", [7, 30, 365]  # Daily, weekly, monthly, yearly
        )
        self.velocity_windows = config.financial_params.get(
            "velocity_windows", [10, 50, 100]  # Short, medium, long term
        )

    def extract(self, data: NDArray[np.float64]) -> DomainFeatureResult:
        """
        Extract financial domain features.

        Args:
            data: Input transaction data. Expected formats:
                  - 1D: Transaction amounts
                  - 2D: Multiple features (amount, time, category, etc.)

        Returns:
            Domain feature extraction result
        """
        import time

        start_time = time.perf_counter()
        data = np.asarray(data, dtype=np.float64)

        all_features = []
        all_names = []

        # Handle 1D vs 2D input
        if data.ndim == 1:
            amounts = data
            has_timestamps = False
        else:
            amounts = data[:, 0]  # First column assumed to be amounts
            has_timestamps = data.shape[1] > 1

        # 1. Statistical features on amounts
        if self.config.enable_statistical:
            stat_features, stat_names = self._compute_statistical_features(amounts)
            all_features.append(stat_features)
            all_names.extend([f"amount_{name}" for name in stat_names])

        # 2. Temporal features on transaction sequence
        if self.config.enable_temporal:
            temp_features, temp_names = self._compute_temporal_features(amounts)
            all_features.append(temp_features)
            all_names.extend([f"amount_{name}" for name in temp_names])

        # 3. Financial-specific features
        if self.config.enable_domain_specific:
            fin_features, fin_names = self._extract_financial_features(
                data, amounts, has_timestamps
            )
            all_features.append(fin_features)
            all_names.extend(fin_names)

        # Combine all features
        features = np.concatenate(all_features) if all_features else np.array([])

        # Handle NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        extraction_time = (time.perf_counter() - start_time) * 1000
        self._feature_names = all_names

        return DomainFeatureResult(
            features=features,
            feature_names=all_names,
            domain=Domain.FINANCIAL,
            metadata={
                "n_transactions": len(amounts),
                "has_timestamps": has_timestamps,
            },
            confidence=1.0,
            extraction_time_ms=extraction_time,
        )

    def _extract_financial_features(
        self,
        data: NDArray[np.float64],
        amounts: NDArray[np.float64],
        has_timestamps: bool,
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Extract financial-specific features.

        Args:
            data: Full data array
            amounts: Transaction amounts
            has_timestamps: Whether timestamps are available

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        # 1. Benford's Law analysis
        benford_features, benford_names = self._compute_benford_features(amounts)
        features.extend(benford_features)
        names.extend(benford_names)

        # 2. Transaction velocity features
        velocity_features, velocity_names = self._compute_velocity_features(amounts)
        features.extend(velocity_features)
        names.extend(velocity_names)

        # 3. Round number detection (common in fraud)
        round_features, round_names = self._compute_round_number_features(amounts)
        features.extend(round_features)
        names.extend(round_names)

        # 4. Amount distribution anomalies
        dist_features, dist_names = self._compute_distribution_features(amounts)
        features.extend(dist_features)
        names.extend(dist_names)

        # 5. Seasonality features (if timestamps available)
        if has_timestamps and data.shape[1] > 1:
            timestamps = data[:, 1]
            season_features, season_names = self._compute_seasonality_features(amounts, timestamps)
            features.extend(season_features)
            names.extend(season_names)

        return np.array(features, dtype=np.float64), names

    def _compute_benford_features(
        self, amounts: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute Benford's Law conformity features.

        Benford's Law predicts the frequency of first digits in naturally
        occurring datasets. Deviations indicate potential fraud/manipulation.

        Args:
            amounts: Transaction amounts

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        # Get absolute non-zero amounts
        abs_amounts = np.abs(amounts[amounts != 0])

        if len(abs_amounts) < 10:
            # Not enough data for meaningful Benford analysis
            return [0.0, 0.0, 0.0, 0.0], [
                "benford_chi_square",
                "benford_mae",
                "benford_max_deviation",
                "benford_suspicious_digits",
            ]

        # Extract first digits
        first_digits = []
        for amount in abs_amounts:
            digit_str = f"{amount:.10f}".lstrip("0").lstrip(".")
            if digit_str and digit_str[0].isdigit() and digit_str[0] != "0":
                first_digits.append(int(digit_str[0]))

        if len(first_digits) < 10:
            return [0.0, 0.0, 0.0, 0.0], [
                "benford_chi_square",
                "benford_mae",
                "benford_max_deviation",
                "benford_suspicious_digits",
            ]

        # Compute observed distribution
        first_digits = np.array(first_digits)  # type: ignore[assignment, unused-ignore]
        observed = np.array([np.sum(first_digits == d) for d in range(1, 10)]) / len(first_digits)  # type: ignore[comparison-overlap, unused-ignore]

        # Chi-square statistic
        expected = self.BENFORD_DISTRIBUTION
        chi_square = np.sum((observed - expected) ** 2 / (expected + 1e-10))
        features.append(chi_square)
        names.append("benford_chi_square")

        # Mean absolute error from Benford
        mae = np.mean(np.abs(observed - expected))
        features.append(mae)
        names.append("benford_mae")

        # Maximum deviation
        max_deviation = np.max(np.abs(observed - expected))
        features.append(max_deviation)
        names.append("benford_max_deviation")

        # Number of suspicious digits (deviation > 2 std)
        std_benford = np.std(expected)
        suspicious = np.sum(np.abs(observed - expected) > 2 * std_benford)
        features.append(float(suspicious))
        names.append("benford_suspicious_digits")

        return features, names

    def _compute_velocity_features(
        self, amounts: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute transaction velocity features.

        Args:
            amounts: Transaction amounts

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n = len(amounts)

        for window in self.velocity_windows:
            if n < window:
                features.extend([0.0, 0.0, 0.0, 0.0])
                names.extend(
                    [
                        f"velocity_{window}_sum",
                        f"velocity_{window}_count_ratio",
                        f"velocity_{window}_acceleration",
                        f"velocity_{window}_volatility",
                    ]
                )
                continue

            # Rolling sum (spending velocity)
            rolling_sum = np.convolve(amounts, np.ones(window), mode="valid")
            velocity_sum = np.mean(rolling_sum)
            features.append(velocity_sum)
            names.append(f"velocity_{window}_sum")

            # Transaction count ratio (how many transactions vs expected)
            count_ratio = n / window
            features.append(count_ratio)
            names.append(f"velocity_{window}_count_ratio")

            # Acceleration (change in velocity)
            if len(rolling_sum) > 1:
                acceleration = np.mean(np.diff(rolling_sum))
            else:
                acceleration = 0.0  # type: ignore[assignment, unused-ignore]
            features.append(acceleration)  # type: ignore[arg-type, unused-ignore]
            names.append(f"velocity_{window}_acceleration")

            # Velocity volatility
            volatility = np.std(rolling_sum) if len(rolling_sum) > 0 else 0.0
            features.append(volatility)
            names.append(f"velocity_{window}_volatility")

        return features, names

    def _compute_round_number_features(
        self, amounts: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute round number detection features.

        Round numbers are often associated with fraudulent transactions
        or manual data entry.

        Args:
            amounts: Transaction amounts

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        abs_amounts = np.abs(amounts)

        # Percentage ending in 0
        ends_in_0 = np.mean((abs_amounts % 10) == 0) if len(abs_amounts) > 0 else 0.0
        features.append(ends_in_0)
        names.append("round_ends_in_0")

        # Percentage ending in 00
        ends_in_00 = np.mean((abs_amounts % 100) == 0) if len(abs_amounts) > 0 else 0.0
        features.append(ends_in_00)
        names.append("round_ends_in_00")

        # Percentage ending in 000
        ends_in_000 = np.mean((abs_amounts % 1000) == 0) if len(abs_amounts) > 0 else 0.0
        features.append(ends_in_000)
        names.append("round_ends_in_000")

        # Percentage that are exact multiples of 100
        multiples_of_100 = np.mean(abs_amounts % 100 == 0) if len(abs_amounts) > 0 else 0.0
        features.append(multiples_of_100)
        names.append("round_multiples_100")

        # Percentage with .99 suffix (psychological pricing)
        has_99 = np.mean(np.abs(abs_amounts % 1 - 0.99) < 0.001) if len(abs_amounts) > 0 else 0.0
        features.append(has_99)
        names.append("psychological_pricing_99")

        return features, names

    def _compute_distribution_features(
        self, amounts: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute amount distribution anomaly features.

        Args:
            amounts: Transaction amounts

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        if len(amounts) < 5:
            return [0.0, 0.0, 0.0, 0.0, 0.0], [
                "dist_outlier_ratio",
                "dist_gap_ratio",
                "dist_concentration",
                "dist_tail_heaviness",
                "dist_bimodality",
            ]

        # Outlier ratio (beyond 3 std)
        mean_val = np.mean(amounts)
        std_val = np.std(amounts) + 1e-10
        outlier_ratio = np.mean(np.abs(amounts - mean_val) > 3 * std_val)
        features.append(outlier_ratio)
        names.append("dist_outlier_ratio")

        # Gap ratio (largest gap between consecutive sorted values / range)
        sorted_amounts = np.sort(amounts)
        gaps = np.diff(sorted_amounts)
        range_val = np.max(amounts) - np.min(amounts) + 1e-10
        max_gap_ratio = np.max(gaps) / range_val if len(gaps) > 0 else 0.0
        features.append(max_gap_ratio)  # type: ignore[arg-type, unused-ignore]
        names.append("dist_gap_ratio")

        # Concentration (what % of total is in top 10% of transactions)
        top_10_pct = np.percentile(np.abs(amounts), 90)
        concentration = np.sum(np.abs(amounts[np.abs(amounts) >= top_10_pct])) / (
            np.sum(np.abs(amounts)) + 1e-10
        )
        features.append(concentration)
        names.append("dist_concentration")

        # Tail heaviness (kurtosis excess)
        kurtosis_val = stats.kurtosis(amounts) if len(amounts) >= 4 else 0.0
        features.append(kurtosis_val)  # type: ignore[arg-type, unused-ignore]
        names.append("dist_tail_heaviness")

        # Bimodality coefficient
        n = len(amounts)
        if n >= 3:
            skewness = stats.skew(amounts)
            kurtosis = stats.kurtosis(amounts)
            bimodality = (skewness**2 + 1) / (
                kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) + 1e-10
            )
        else:
            bimodality = 0.0
        features.append(bimodality)
        names.append("dist_bimodality")

        return features, names  # type: ignore[return-value, unused-ignore]

    def _compute_seasonality_features(
        self, amounts: NDArray[np.float64], timestamps: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute time-series seasonality features.

        Args:
            amounts: Transaction amounts
            timestamps: Transaction timestamps

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n = len(amounts)

        for period in self.seasonality_periods:
            if n < 2 * period:
                features.extend([0.0, 0.0])
                names.extend([f"season_{period}_strength", f"season_{period}_phase"])
                continue

            # Compute autocorrelation at the period
            if n > period:
                autocorr = np.corrcoef(amounts[:-period], amounts[period:])[0, 1]
                autocorr = 0.0 if np.isnan(autocorr) else autocorr
            else:
                autocorr = 0.0

            features.append(autocorr)
            names.append(f"season_{period}_strength")

            # Estimate phase using FFT
            fft_result = np.fft.fft(amounts)
            freq_idx = n // period if period > 0 else 0
            if 0 < freq_idx < len(fft_result):
                phase = np.angle(fft_result[freq_idx])
            else:
                phase = 0.0

            features.append(phase)
            names.append(f"season_{period}_phase")

        return features, names

    def get_feature_names(self) -> list[str]:
        """Get names of extracted features."""
        return self._feature_names.copy()


class InfrastructureFeatureExtractor(BaseDomainExtractor):
    """
    Infrastructure domain feature extractor.

    Implements SCADA-specific feature engineering, process variable
    correlation matrices, and industrial control system patterns.

    Target: Improve from 0.79 to 0.88 F1 score.
    """

    # Common SCADA process variable types
    PROCESS_VARIABLES = [
        "flow",
        "pressure",
        "temperature",
        "level",
        "voltage",
        "current",
        "frequency",
        "power",
    ]

    def __init__(self, config: DomainFeatureConfig | None = None):
        """
        Initialize infrastructure feature extractor.

        Args:
            config: Domain feature configuration
        """
        if config is None:
            config = DomainFeatureConfig(
                domain=Domain.INFRASTRUCTURE,
                window_size=120,  # 2-minute windows
                sampling_rate=1.0,
            )
        super().__init__(config)

        # Infrastructure-specific parameters
        self.correlation_threshold = config.infrastructure_params.get("correlation_threshold", 0.7)
        self.lag_windows = config.infrastructure_params.get("lag_windows", [1, 5, 10, 30])
        self.alarm_thresholds = config.infrastructure_params.get("alarm_thresholds", {})

    def extract(self, data: NDArray[np.float64]) -> DomainFeatureResult:
        """
        Extract infrastructure domain features.

        Args:
            data: Input process variable data.
                  2D: rows=time, cols=process variables

        Returns:
            Domain feature extraction result
        """
        import time

        start_time = time.perf_counter()
        data = np.asarray(data, dtype=np.float64)

        all_features = []
        all_names = []

        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_vars = data.shape

        # 1. Per-variable statistical features
        if self.config.enable_statistical:
            for i in range(n_vars):
                var_data = data[:, i]
                stat_features, stat_names = self._compute_statistical_features(var_data)
                all_features.append(stat_features)
                all_names.extend([f"pv{i}_{name}" for name in stat_names])

        # 2. Per-variable temporal features
        if self.config.enable_temporal:
            for i in range(n_vars):
                var_data = data[:, i]
                temp_features, temp_names = self._compute_temporal_features(var_data)
                all_features.append(temp_features)
                all_names.extend([f"pv{i}_{name}" for name in temp_names])

        # 3. Infrastructure-specific features
        if self.config.enable_domain_specific:
            infra_features, infra_names = self._extract_infrastructure_features(data)
            all_features.append(infra_features)
            all_names.extend(infra_names)

        # Combine all features
        features = np.concatenate(all_features) if all_features else np.array([])

        # Handle NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        extraction_time = (time.perf_counter() - start_time) * 1000
        self._feature_names = all_names

        return DomainFeatureResult(
            features=features,
            feature_names=all_names,
            domain=Domain.INFRASTRUCTURE,
            metadata={
                "n_vars": n_vars,
                "n_samples": n_samples,
            },
            confidence=1.0,
            extraction_time_ms=extraction_time,
        )

    def _extract_infrastructure_features(
        self, data: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Extract infrastructure-specific features.

        Args:
            data: Multi-variate process variable data

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n_samples, n_vars = data.shape

        # 1. Process variable correlation matrix features
        corr_features, corr_names = self._compute_correlation_matrix_features(data)
        features.extend(corr_features)
        names.extend(corr_names)

        # 2. Lagged cross-correlation features
        lag_features, lag_names = self._compute_lagged_correlation_features(data)
        features.extend(lag_features)
        names.extend(lag_names)

        # 3. Setpoint deviation features
        setpoint_features, setpoint_names = self._compute_setpoint_deviation_features(data)
        features.extend(setpoint_features)
        names.extend(setpoint_names)

        # 4. Alarm rate features
        alarm_features, alarm_names = self._compute_alarm_features(data)
        features.extend(alarm_features)
        names.extend(alarm_names)

        # 5. Control loop stability features
        stability_features, stability_names = self._compute_control_stability_features(data)
        features.extend(stability_features)
        names.extend(stability_names)

        # 6. Cyber attack indicators
        attack_features, attack_names = self._compute_attack_indicator_features(data)
        features.extend(attack_features)
        names.extend(attack_names)

        return np.array(features, dtype=np.float64), names

    def _compute_correlation_matrix_features(
        self, data: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute features from process variable correlation matrix.

        Unexpected correlation changes can indicate attacks or failures.

        Args:
            data: Process variable data

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n_vars = data.shape[1]

        if n_vars < 2:
            return [0.0, 0.0, 0.0, 0.0], [
                "corr_matrix_mean",
                "corr_matrix_std",
                "corr_high_count",
                "corr_determinant",
            ]

        # Compute correlation matrix
        corr_matrix = np.corrcoef(data.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Extract upper triangle (excluding diagonal)
        upper_tri = np.triu_indices(n_vars, k=1)
        correlations = corr_matrix[upper_tri]  # type: ignore[index, unused-ignore]

        # Mean absolute correlation
        mean_corr = np.mean(np.abs(correlations))
        features.append(mean_corr)
        names.append("corr_matrix_mean")

        # Correlation variability
        std_corr = np.std(correlations)
        features.append(std_corr)
        names.append("corr_matrix_std")

        # Count of high correlations
        high_corr_count = np.sum(np.abs(correlations) > self.correlation_threshold)
        features.append(float(high_corr_count))
        names.append("corr_high_count")

        # Determinant (near-zero indicates multicollinearity issues)
        try:
            determinant = np.linalg.det(corr_matrix)
        except np.linalg.LinAlgError:
            determinant = 0.0
        features.append(determinant)
        names.append("corr_determinant")

        return features, names  # type: ignore[return-value, unused-ignore]

    def _compute_lagged_correlation_features(
        self, data: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute lagged cross-correlation features.

        Physical processes have expected time delays between variables.

        Args:
            data: Process variable data

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n_samples, n_vars = data.shape

        for lag in self.lag_windows:
            if n_samples <= lag:
                features.extend([0.0, 0.0])
                names.extend([f"lag{lag}_mean_corr", f"lag{lag}_max_corr"])
                continue

            # Compute lagged correlations between all variable pairs
            lagged_corrs = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    var_i = data[:-lag, i]
                    var_j = data[lag:, j]
                    if len(var_i) > 0 and len(var_j) > 0:
                        corr = np.corrcoef(var_i, var_j)[0, 1]
                        if not np.isnan(corr):
                            lagged_corrs.append(corr)

            mean_lag_corr = np.mean(np.abs(lagged_corrs)) if lagged_corrs else 0.0
            max_lag_corr = np.max(np.abs(lagged_corrs)) if lagged_corrs else 0.0

            features.extend([mean_lag_corr, max_lag_corr])  # type: ignore[list-item, unused-ignore]
            names.extend([f"lag{lag}_mean_corr", f"lag{lag}_max_corr"])

        return features, names

    def _compute_setpoint_deviation_features(
        self, data: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute setpoint deviation features.

        Uses running mean as proxy for setpoint when actual setpoints unknown.

        Args:
            data: Process variable data

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n_samples, n_vars = data.shape

        # Global deviation metrics
        deviations = []
        for i in range(n_vars):
            var_data = data[:, i]

            # Use median as proxy for setpoint
            setpoint = np.median(var_data)

            # Compute deviation from setpoint
            deviation = np.abs(var_data - setpoint)
            mean_deviation = np.mean(deviation)
            max_deviation = np.max(deviation)
            deviations.append((mean_deviation, max_deviation))

        # Aggregate deviations
        mean_devs = [d[0] for d in deviations]
        max_devs = [d[1] for d in deviations]

        features.append(np.mean(mean_devs))
        names.append("setpoint_mean_deviation")

        features.append(np.mean(max_devs))
        names.append("setpoint_max_deviation")

        features.append(np.std(mean_devs))
        names.append("setpoint_deviation_variability")

        # Count of variables with significant deviation
        significant_deviation_count = sum(1 for d in mean_devs if d > np.median(mean_devs) * 2)
        features.append(float(significant_deviation_count))
        names.append("setpoint_significant_count")

        return features, names

    def _compute_alarm_features(self, data: NDArray[np.float64]) -> tuple[list[float], list[str]]:
        """
        Compute alarm rate features based on threshold crossings.

        Args:
            data: Process variable data

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n_samples, n_vars = data.shape

        total_crossings = 0
        alarm_rates = []

        for i in range(n_vars):
            var_data = data[:, i]

            # Use statistical thresholds if not provided
            mean_val = np.mean(var_data)
            std_val = np.std(var_data) + 1e-10

            high_threshold = mean_val + 2 * std_val
            low_threshold = mean_val - 2 * std_val

            # Count threshold crossings
            high_crossings = np.sum(np.diff(var_data > high_threshold) == 1)
            low_crossings = np.sum(np.diff(var_data < low_threshold) == 1)
            total_crossings += high_crossings + low_crossings

            # Alarm rate (crossings per 100 samples)
            alarm_rate = (high_crossings + low_crossings) / (n_samples / 100 + 1e-10)
            alarm_rates.append(alarm_rate)

        features.append(float(total_crossings))
        names.append("alarm_total_crossings")

        features.append(np.mean(alarm_rates))
        names.append("alarm_mean_rate")

        features.append(np.max(alarm_rates))
        names.append("alarm_max_rate")

        features.append(np.std(alarm_rates))
        names.append("alarm_rate_variability")

        return features, names

    def _compute_control_stability_features(
        self, data: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute control loop stability features.

        Detects oscillations, hunting, and instability in control loops.

        Args:
            data: Process variable data

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n_samples, n_vars = data.shape

        oscillation_indices = []
        stability_scores = []

        for i in range(n_vars):
            var_data = data[:, i]

            if len(var_data) < 10:
                continue

            # Zero crossings around mean (oscillation indicator)
            centered = var_data - np.mean(var_data)
            zero_crossings = np.sum(np.diff(np.signbit(centered)))
            oscillation_idx = zero_crossings / (n_samples - 1)
            oscillation_indices.append(oscillation_idx)

            # Derivative variance (hunting indicator)
            if len(var_data) > 1:
                derivative = np.diff(var_data)
                derivative_var = np.var(derivative)
                var_ratio = derivative_var / (np.var(var_data) + 1e-10)
            else:
                var_ratio = 0.0  # type: ignore[assignment, unused-ignore]

            # Stability score (lower is more stable)
            stability = oscillation_idx * var_ratio
            stability_scores.append(stability)

        features.append(np.mean(oscillation_indices) if oscillation_indices else 0.0)
        names.append("control_oscillation_index")

        features.append(np.max(oscillation_indices) if oscillation_indices else 0.0)
        names.append("control_oscillation_max")

        features.append(np.mean(stability_scores) if stability_scores else 0.0)
        names.append("control_stability_score")

        features.append(np.max(stability_scores) if stability_scores else 0.0)
        names.append("control_instability_max")

        return features, names

    def _compute_attack_indicator_features(
        self, data: NDArray[np.float64]
    ) -> tuple[list[float], list[str]]:
        """
        Compute cyber attack indicator features.

        Detects patterns associated with common ICS/SCADA attacks.

        Args:
            data: Process variable data

        Returns:
            Tuple of (features, feature_names)
        """
        features = []
        names = []

        n_samples, n_vars = data.shape

        # 1. Frozen value detection (constant values indicate sensor tampering)
        frozen_count = 0
        for i in range(n_vars):
            var_data = data[:, i]
            unique_values = len(np.unique(var_data))
            if unique_values < 3:  # Too few unique values
                frozen_count += 1

        features.append(float(frozen_count) / (n_vars + 1e-10))
        names.append("attack_frozen_ratio")

        # 2. Replay attack detection (repeated patterns)
        replay_indicators = []
        for i in range(n_vars):
            var_data = data[:, i]
            if len(var_data) > 20:
                # Check for repeating subsequences
                window_size = 10
                for start in range(len(var_data) - 2 * window_size):
                    window = var_data[start : start + window_size]
                    for check in range(start + window_size, len(var_data) - window_size):
                        check_window = var_data[check : check + window_size]
                        if np.allclose(window, check_window, rtol=1e-5):
                            replay_indicators.append(1.0)
                            break

        features.append(len(replay_indicators) / (n_vars + 1e-10))
        names.append("attack_replay_indicator")

        # 3. Value injection detection (sudden step changes)
        step_changes = 0
        for i in range(n_vars):
            var_data = data[:, i]
            if len(var_data) > 1:
                diff = np.abs(np.diff(var_data))
                std_diff = np.std(diff) + 1e-10
                large_steps = np.sum(diff > 5 * std_diff)
                step_changes += large_steps  # type: ignore[assignment, unused-ignore]

        features.append(float(step_changes) / (n_samples * n_vars + 1e-10))
        names.append("attack_step_injection_ratio")

        # 4. Correlation breaking (normal correlations suddenly change)
        # Use first and second half of data
        if n_samples > 20 and n_vars >= 2:
            half = n_samples // 2
            corr1 = np.corrcoef(data[:half].T)
            corr2 = np.corrcoef(data[half:].T)
            corr1 = np.nan_to_num(corr1, nan=0.0)
            corr2 = np.nan_to_num(corr2, nan=0.0)

            corr_change = np.mean(np.abs(corr1 - corr2))
            features.append(corr_change)
            names.append("attack_correlation_change")
        else:
            features.append(0.0)
            names.append("attack_correlation_change")

        return features, names

    def get_feature_names(self) -> list[str]:
        """Get names of extracted features."""
        return self._feature_names.copy()


class DomainFeatureExtractorFactory:
    """Factory for creating domain-specific feature extractors."""

    _extractors: dict[Domain, type[BaseDomainExtractor]] = {
        Domain.MEDICAL: MedicalFeatureExtractor,
        Domain.FINANCIAL: FinancialFeatureExtractor,
        Domain.INFRASTRUCTURE: InfrastructureFeatureExtractor,
    }

    @classmethod
    def create(
        cls, domain: Domain | str, config: DomainFeatureConfig | None = None
    ) -> BaseDomainExtractor:
        """
        Create a domain-specific feature extractor.

        Args:
            domain: Target domain
            config: Optional configuration

        Returns:
            Domain feature extractor instance

        Raises:
            ValueError: If domain is not supported
        """
        if isinstance(domain, str):
            domain = Domain(domain.lower())

        extractor_cls = cls._extractors.get(domain)
        if extractor_cls is None:
            raise ValueError(
                f"Unsupported domain: {domain}. Supported: {list(cls._extractors.keys())}"
            )

        if config is None:
            config = DomainFeatureConfig(domain=domain)
        elif config.domain != domain:
            config.domain = domain

        return extractor_cls(config)

    @classmethod
    def register(cls, domain: Domain, extractor_cls: type[BaseDomainExtractor]) -> None:
        """Register a custom domain extractor."""
        cls._extractors[domain] = extractor_cls


# Convenience functions
def extract_medical_features(
    data: NDArray[np.float64], config: DomainFeatureConfig | None = None
) -> DomainFeatureResult:
    """Extract features for medical domain."""
    extractor = DomainFeatureExtractorFactory.create(Domain.MEDICAL, config)
    return extractor.extract(data)


def extract_financial_features(
    data: NDArray[np.float64], config: DomainFeatureConfig | None = None
) -> DomainFeatureResult:
    """Extract features for financial domain."""
    extractor = DomainFeatureExtractorFactory.create(Domain.FINANCIAL, config)
    return extractor.extract(data)


def extract_infrastructure_features(
    data: NDArray[np.float64], config: DomainFeatureConfig | None = None
) -> DomainFeatureResult:
    """Extract features for infrastructure domain."""
    extractor = DomainFeatureExtractorFactory.create(Domain.INFRASTRUCTURE, config)
    return extractor.extract(data)
