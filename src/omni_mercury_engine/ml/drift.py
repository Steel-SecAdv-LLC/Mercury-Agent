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

"""
Data and Model Drift Detection Module

Provides comprehensive drift detection capabilities:
- Statistical drift detection (Kolmogorov-Smirnov, Chi-squared, PSI)
- Feature drift monitoring for anomaly detection models
- Concept drift detection for classification models
- Online drift detection with adaptive windowing

Inspired by Alibi-Detect and Evidently frameworks.
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftType(StrEnum):
    """Types of drift that can be detected."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    LABEL_DRIFT = "label_drift"


class DriftSeverity(StrEnum):
    """Severity levels for detected drift."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a drift detection test."""

    is_drift: bool
    drift_type: DriftType
    severity: DriftSeverity
    p_value: float
    test_statistic: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    feature_drifts: dict[str, float] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_drift": self.is_drift,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "threshold": self.threshold,
            "details": self.details,
            "feature_drifts": self.feature_drifts,
            "message": self.message,
        }


class KolmogorovSmirnovDriftDetector:
    """
    Kolmogorov-Smirnov test based drift detector.

    The KS test compares the cumulative distribution functions of
    reference and current data distributions to detect drift.

    Null hypothesis: Both samples come from the same distribution.
    If p-value < threshold, we reject H0 and conclude drift has occurred.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        correction: str = "bonferroni",
    ):
        """
        Initialize KS drift detector.

        Args:
            p_value_threshold: Significance level for drift detection
            correction: Multiple testing correction method
        """
        self.p_value_threshold = p_value_threshold
        self.correction = correction
        self.reference_data: np.ndarray | None = None

    def fit(self, reference_data: np.ndarray) -> KolmogorovSmirnovDriftDetector:
        """
        Fit the detector with reference (baseline) data.

        Args:
            reference_data: Reference data array [n_samples, n_features]

        Returns:
            Self for method chaining
        """
        self.reference_data = np.asarray(reference_data)
        if self.reference_data.ndim == 1:
            self.reference_data = self.reference_data.reshape(-1, 1)
        logger.info(f"KS drift detector fitted with {len(self.reference_data)} reference samples")
        return self

    def detect(
        self,
        current_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current data to compare [n_samples, n_features]
            feature_names: Optional feature names for reporting

        Returns:
            DriftResult with detection outcome
        """
        if self.reference_data is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        current_data = np.asarray(current_data)
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        n_features = current_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Run KS test for each feature
        p_values = []
        statistics = []
        feature_drifts = {}

        for i in range(n_features):
            ref_col = (
                self.reference_data[:, i]
                if self.reference_data.shape[1] > i
                else self.reference_data[:, 0]
            )
            cur_col = current_data[:, i]

            statistic, p_value = stats.ks_2samp(ref_col, cur_col)
            p_values.append(p_value)
            statistics.append(statistic)
            feature_drifts[feature_names[i]] = float(statistic)

        # Apply multiple testing correction
        if self.correction == "bonferroni" and n_features > 1:
            adjusted_threshold = self.p_value_threshold / n_features
        else:
            adjusted_threshold = self.p_value_threshold

        # Determine if drift occurred
        min_p_value = min(p_values)
        max_statistic = max(statistics)
        is_drift = min_p_value < adjusted_threshold

        # Determine severity
        severity = self._compute_severity(min_p_value, adjusted_threshold)

        # Count drifting features
        n_drifting = sum(1 for p in p_values if p < adjusted_threshold)

        return DriftResult(
            is_drift=is_drift,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            p_value=float(min_p_value),
            test_statistic=float(max_statistic),
            threshold=adjusted_threshold,
            feature_drifts=feature_drifts,
            details={
                "all_p_values": [float(p) for p in p_values],
                "all_statistics": [float(s) for s in statistics],
                "n_drifting_features": n_drifting,
                "total_features": n_features,
                "correction_method": self.correction,
            },
            message=(
                f"Drift detected in {n_drifting}/{n_features} features"
                if is_drift
                else f"No significant drift detected (p={min_p_value:.4f})"
            ),
        )

    def _compute_severity(self, p_value: float, threshold: float) -> DriftSeverity:
        """Compute drift severity based on p-value."""
        if p_value >= threshold:
            return DriftSeverity.NONE
        elif p_value >= threshold * 0.1:
            return DriftSeverity.LOW
        elif p_value >= threshold * 0.01:
            return DriftSeverity.MEDIUM
        elif p_value >= threshold * 0.001:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


class PopulationStabilityIndexDetector:
    """
    Population Stability Index (PSI) based drift detector.

    PSI is commonly used in credit scoring to measure how much
    a variable has shifted over time. It compares the distribution
    of values between two datasets.

    PSI interpretation:
    - PSI < 0.10: No significant change
    - 0.10 <= PSI < 0.25: Some change, investigate
    - PSI >= 0.25: Major shift, action required
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold_low: float = 0.10,
        psi_threshold_high: float = 0.25,
    ):
        """
        Initialize PSI detector.

        Args:
            n_bins: Number of bins for discretization
            psi_threshold_low: Threshold for low drift
            psi_threshold_high: Threshold for high drift
        """
        self.n_bins = n_bins
        self.psi_threshold_low = psi_threshold_low
        self.psi_threshold_high = psi_threshold_high
        self.reference_data: np.ndarray | None = None
        self.bin_edges: list[np.ndarray] = []

    def fit(self, reference_data: np.ndarray) -> PopulationStabilityIndexDetector:
        """
        Fit the detector with reference data.

        Args:
            reference_data: Reference data array [n_samples, n_features]

        Returns:
            Self for method chaining
        """
        self.reference_data = np.asarray(reference_data)
        if self.reference_data.ndim == 1:
            self.reference_data = self.reference_data.reshape(-1, 1)

        # Compute bin edges for each feature
        self.bin_edges = []
        for i in range(self.reference_data.shape[1]):
            _, edges = np.histogram(self.reference_data[:, i], bins=self.n_bins)
            self.bin_edges.append(edges)

        logger.info(f"PSI drift detector fitted with {len(self.reference_data)} reference samples")
        return self

    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bin_edges: np.ndarray,
    ) -> float:
        """Calculate PSI between two distributions."""
        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions with small epsilon to avoid division by zero
        eps = 1e-8
        ref_pct = (ref_hist + eps) / (len(reference) + eps * len(ref_hist))
        cur_pct = (cur_hist + eps) / (len(current) + eps * len(cur_hist))

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def detect(
        self,
        current_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """
        Detect drift using PSI.

        Args:
            current_data: Current data to compare [n_samples, n_features]
            feature_names: Optional feature names for reporting

        Returns:
            DriftResult with detection outcome
        """
        if self.reference_data is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        current_data = np.asarray(current_data)
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        n_features = current_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Calculate PSI for each feature
        psi_values = []
        feature_drifts = {}

        for i in range(n_features):
            ref_col = (
                self.reference_data[:, i]
                if self.reference_data.shape[1] > i
                else self.reference_data[:, 0]
            )
            cur_col = current_data[:, i]
            edges = self.bin_edges[i] if i < len(self.bin_edges) else self.bin_edges[0]

            psi = self._calculate_psi(ref_col, cur_col, edges)
            psi_values.append(psi)
            feature_drifts[feature_names[i]] = psi

        # Determine if drift occurred
        max_psi = max(psi_values)
        mean_psi = np.mean(psi_values)

        is_drift = max_psi >= self.psi_threshold_low
        severity = self._compute_severity(max_psi)

        # Count drifting features
        n_high_drift = sum(1 for p in psi_values if p >= self.psi_threshold_high)
        n_low_drift = sum(
            1 for p in psi_values if self.psi_threshold_low <= p < self.psi_threshold_high
        )

        return DriftResult(
            is_drift=is_drift,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            p_value=1.0 - min(max_psi, 1.0),  # Invert PSI as pseudo p-value
            test_statistic=float(max_psi),
            threshold=self.psi_threshold_low,
            feature_drifts=feature_drifts,
            details={
                "all_psi_values": [float(p) for p in psi_values],
                "mean_psi": float(mean_psi),
                "n_high_drift_features": n_high_drift,
                "n_low_drift_features": n_low_drift,
                "total_features": n_features,
            },
            message=(
                f"PSI drift detected: {n_high_drift} high, {n_low_drift} moderate"
                if is_drift
                else f"No significant PSI drift detected (max_psi={max_psi:.4f})"
            ),
        )

    def _compute_severity(self, psi: float) -> DriftSeverity:
        """Compute drift severity based on PSI value."""
        if psi < self.psi_threshold_low:
            return DriftSeverity.NONE
        elif psi < self.psi_threshold_high:
            return DriftSeverity.LOW
        elif psi < 0.50:
            return DriftSeverity.MEDIUM
        elif psi < 1.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


class ChiSquaredDriftDetector:
    """
    Chi-squared test based drift detector for categorical features.

    Uses the chi-squared test of independence to detect changes
    in the distribution of categorical variables.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
    ):
        """
        Initialize Chi-squared drift detector.

        Args:
            p_value_threshold: Significance level for drift detection
        """
        self.p_value_threshold = p_value_threshold
        self.reference_data: np.ndarray | None = None
        self.reference_counts: list[dict[Any, int]] = []

    def fit(self, reference_data: np.ndarray) -> ChiSquaredDriftDetector:
        """
        Fit the detector with reference data.

        Args:
            reference_data: Reference data array [n_samples, n_features]

        Returns:
            Self for method chaining
        """
        self.reference_data = np.asarray(reference_data)
        if self.reference_data.ndim == 1:
            self.reference_data = self.reference_data.reshape(-1, 1)

        # Compute value counts for each feature
        self.reference_counts = []
        for i in range(self.reference_data.shape[1]):
            unique, counts = np.unique(self.reference_data[:, i], return_counts=True)
            self.reference_counts.append(dict(zip(unique, counts)))

        logger.info(
            f"Chi-squared drift detector fitted with {len(self.reference_data)} reference samples"
        )
        return self

    def detect(
        self,
        current_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """
        Detect drift using Chi-squared test.

        Args:
            current_data: Current data to compare [n_samples, n_features]
            feature_names: Optional feature names for reporting

        Returns:
            DriftResult with detection outcome
        """
        if self.reference_data is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        current_data = np.asarray(current_data)
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        n_features = current_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Run chi-squared test for each feature
        p_values = []
        statistics = []
        feature_drifts = {}

        for i in range(n_features):
            ref_counts = (
                self.reference_counts[i]
                if i < len(self.reference_counts)
                else self.reference_counts[0]
            )

            # Get current counts
            cur_unique, cur_counts = np.unique(current_data[:, i], return_counts=True)
            cur_counts_dict = dict(zip(cur_unique, cur_counts))

            # Align categories
            all_categories = set(ref_counts.keys()) | set(cur_counts_dict.keys())
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            cur_aligned = [cur_counts_dict.get(cat, 0) for cat in all_categories]

            # Chi-squared test
            if len(all_categories) > 1:
                statistic, p_value = stats.chisquare(cur_aligned, f_exp=ref_aligned)
            else:
                statistic, p_value = 0.0, 1.0

            p_values.append(p_value)
            statistics.append(statistic)
            feature_drifts[feature_names[i]] = float(statistic)

        # Determine if drift occurred
        min_p_value = min(p_values)
        max_statistic = max(statistics)
        is_drift = min_p_value < self.p_value_threshold

        severity = self._compute_severity(min_p_value, self.p_value_threshold)
        n_drifting = sum(1 for p in p_values if p < self.p_value_threshold)

        return DriftResult(
            is_drift=is_drift,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            p_value=float(min_p_value),
            test_statistic=float(max_statistic),
            threshold=self.p_value_threshold,
            feature_drifts=feature_drifts,
            details={
                "all_p_values": [float(p) for p in p_values],
                "all_statistics": [float(s) for s in statistics],
                "n_drifting_features": n_drifting,
                "total_features": n_features,
            },
            message=(
                f"Categorical drift detected in {n_drifting}/{n_features} features"
                if is_drift
                else f"No significant categorical drift detected (p={min_p_value:.4f})"
            ),
        )

    def _compute_severity(self, p_value: float, threshold: float) -> DriftSeverity:
        """Compute drift severity based on p-value."""
        if p_value >= threshold:
            return DriftSeverity.NONE
        elif p_value >= threshold * 0.1:
            return DriftSeverity.LOW
        elif p_value >= threshold * 0.01:
            return DriftSeverity.MEDIUM
        elif p_value >= threshold * 0.001:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


class OnlineDriftDetector:
    """
    Online drift detection with adaptive windowing.

    Implements ADWIN (Adaptive Windowing) inspired online drift detection
    that maintains a sliding window and detects when the distribution
    changes significantly.
    """

    def __init__(
        self,
        max_window_size: int = 1000,
        min_window_size: int = 50,
        significance_level: float = 0.05,
    ):
        """
        Initialize online drift detector.

        Args:
            max_window_size: Maximum samples to keep in window
            min_window_size: Minimum samples before detection starts
            significance_level: Significance level for drift detection
        """
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.significance_level = significance_level
        self.window: list[np.ndarray] = []
        self.drift_detected = False
        self.drift_count = 0

    def update(self, sample: np.ndarray) -> DriftResult | None:
        """
        Update detector with new sample and check for drift.

        Args:
            sample: New sample to add to window

        Returns:
            DriftResult if drift detected, None otherwise
        """
        sample = np.asarray(sample).flatten()
        self.window.append(sample)

        # Trim window if too large
        if len(self.window) > self.max_window_size:
            self.window = self.window[-self.max_window_size :]

        # Check for drift if we have enough samples
        if len(self.window) < self.min_window_size:
            return None

        # Split window in half and compare distributions
        mid = len(self.window) // 2
        old_window = np.array(self.window[:mid])
        new_window = np.array(self.window[mid:])

        # Use KS test to compare distributions
        p_values = []
        for i in range(sample.shape[0]):
            _, p = stats.ks_2samp(old_window[:, i], new_window[:, i])
            p_values.append(p)

        min_p = min(p_values)

        if min_p < self.significance_level:
            self.drift_detected = True
            self.drift_count += 1

            # Reset window after drift
            self.window = self.window[mid:]

            return DriftResult(
                is_drift=True,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=DriftSeverity.MEDIUM,
                p_value=float(min_p),
                test_statistic=float(max(p_values) - min_p),
                threshold=self.significance_level,
                details={
                    "window_size": len(self.window),
                    "drift_count": self.drift_count,
                },
                message=f"Online drift detected (total drifts: {self.drift_count})",
            )

        return None

    def reset(self) -> None:
        """Reset the detector state."""
        self.window = []
        self.drift_detected = False
        self.drift_count = 0


class EnsembleDriftDetector:
    """
    Ensemble drift detector combining multiple methods.

    Combines KS, PSI, and Chi-squared tests for robust drift detection.
    Uses majority voting to determine final drift decision.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.10,
        voting: str = "majority",
    ):
        """
        Initialize ensemble drift detector.

        Args:
            p_value_threshold: Significance level for statistical tests
            psi_threshold: Threshold for PSI test
            voting: Voting strategy ('majority', 'any', 'all')
        """
        self.voting = voting
        self.ks_detector = KolmogorovSmirnovDriftDetector(p_value_threshold)
        self.psi_detector = PopulationStabilityIndexDetector(psi_threshold_low=psi_threshold)
        self.is_fitted = False

    def fit(self, reference_data: np.ndarray) -> EnsembleDriftDetector:
        """
        Fit all detectors with reference data.

        Args:
            reference_data: Reference data array [n_samples, n_features]

        Returns:
            Self for method chaining
        """
        self.ks_detector.fit(reference_data)
        self.psi_detector.fit(reference_data)
        self.is_fitted = True
        logger.info("Ensemble drift detector fitted")
        return self

    def detect(
        self,
        current_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """
        Detect drift using ensemble of methods.

        Args:
            current_data: Current data to compare
            feature_names: Optional feature names

        Returns:
            Combined DriftResult
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        # Run all detectors
        ks_result = self.ks_detector.detect(current_data, feature_names)
        psi_result = self.psi_detector.detect(current_data, feature_names)

        results = [ks_result, psi_result]
        drift_votes = [r.is_drift for r in results]

        # Determine final decision based on voting strategy
        if self.voting == "majority":
            is_drift = sum(drift_votes) > len(drift_votes) / 2
        elif self.voting == "any":
            is_drift = any(drift_votes)
        elif self.voting == "all":
            is_drift = all(drift_votes)
        else:
            is_drift = sum(drift_votes) > len(drift_votes) / 2

        # Combine severities (take max)
        severities = [r.severity for r in results]
        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]
        max_severity = max(severities, key=severity_order.index)

        # Combine feature drifts
        combined_feature_drifts: dict[str, list[float]] = {}
        for result in results:
            for feature, value in result.feature_drifts.items():
                if feature not in combined_feature_drifts:
                    combined_feature_drifts[feature] = []
                combined_feature_drifts[feature].append(value)

        avg_feature_drifts = {k: float(np.mean(v)) for k, v in combined_feature_drifts.items()}

        return DriftResult(
            is_drift=is_drift,
            drift_type=DriftType.DATA_DRIFT,
            severity=max_severity,
            p_value=float(np.mean([r.p_value for r in results])),
            test_statistic=float(np.mean([r.test_statistic for r in results])),
            threshold=0.0,  # Ensemble doesn't have single threshold
            feature_drifts=avg_feature_drifts,
            details={
                "ks_result": ks_result.to_dict(),
                "psi_result": psi_result.to_dict(),
                "voting_strategy": self.voting,
                "votes": drift_votes,
            },
            message=(
                f"Ensemble drift: {sum(drift_votes)}/{len(drift_votes)} detectors triggered"
                if is_drift
                else "No ensemble drift detected"
            ),
        )


DriftDetectorType = (
    KolmogorovSmirnovDriftDetector
    | PopulationStabilityIndexDetector
    | ChiSquaredDriftDetector
    | EnsembleDriftDetector
)


def create_drift_detector(
    detector_type: str = "ks",
    **kwargs: Any,
) -> DriftDetectorType:
    """
    Factory function to create drift detectors.

    Args:
        detector_type: Type of detector ('ks', 'psi', 'chi2', 'ensemble')
        **kwargs: Detector-specific parameters

    Returns:
        Configured drift detector
    """
    detectors: dict[str, type[DriftDetectorType]] = {
        "ks": KolmogorovSmirnovDriftDetector,
        "psi": PopulationStabilityIndexDetector,
        "chi2": ChiSquaredDriftDetector,
        "ensemble": EnsembleDriftDetector,
    }

    if detector_type not in detectors:
        raise ValueError(
            f"Unknown detector type: {detector_type}. Choose from {list(detectors.keys())}"
        )

    detector_class = detectors[detector_type]
    return detector_class(**kwargs)
