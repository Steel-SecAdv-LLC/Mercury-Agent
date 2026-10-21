# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Data and Model Drift Detection Module.

Provides comprehensive drift detection capabilities:
- Statistical drift detection (Kolmogorov-Smirnov, Chi-squared, PSI)
- Covariance-structure drift (affine-invariant SPD distance), which catches
  changes that leave every marginal intact (decoupled sensors, a broken
  physical relationship) and so are invisible to per-feature tests
- Principal-axis mean shift (a per-axis KS test in the reference's PCA basis),
  which catches shifts along low-variance directions of correlated data that
  every per-feature test dilutes
- Feature drift monitoring for anomaly detection models
- Concept drift detection for classification models
- Online drift detection with adaptive windowing

Inspired by Alibi-Detect and Evidently frameworks.
"""

from __future__ import annotations

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
    COVARIANCE_DRIFT = "covariance_drift"
    PRINCIPAL_AXIS_DRIFT = "principal_axis_drift"


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


class DriftIncomparableError(ValueError):
    """The current data has a different number of features than the reference.

    Earlier versions compared each extra feature against reference column 0
    and reported the result as drift evidence, a comparison with no meaning.
    """


def _require_n_features(expected: int, current: np.ndarray[Any, Any]) -> None:
    """Raise unless ``current`` has ``expected`` feature columns.

    Args:
        expected: Number of raw features the reference was fitted on.
        current: Current batch ``[n_samples, n_features]``.

    Raises:
        DriftIncomparableError: When the widths differ.
    """
    if current.shape[1] != expected:
        raise DriftIncomparableError(
            f"current data has {current.shape[1]} features; the reference has "
            f"{expected}, so the distributions cannot be compared"
        )


def _require_width(reference: np.ndarray[Any, Any], current: np.ndarray[Any, Any]) -> None:
    _require_n_features(reference.shape[1], current)


def _severity_for_p_value(p_value: float, threshold: float) -> DriftSeverity:
    """Severity ladder shared by the p-value tests: a decade of p per step."""
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


class KolmogorovSmirnovDriftDetector:
    """Kolmogorov-Smirnov test based drift detector.

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
        """Initialize KS drift detector.

        Args:
            p_value_threshold: Significance level for drift detection
            correction: Multiple testing correction method
        """
        self.p_value_threshold = p_value_threshold
        self.correction = correction
        self.reference_data: np.ndarray[Any, Any] | None = None

    def fit(self, reference_data: np.ndarray[Any, Any]) -> KolmogorovSmirnovDriftDetector:
        """Fit the detector with reference (baseline) data.

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
        current_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """Detect drift between reference and current data.

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

        _require_width(self.reference_data, current_data)
        for i in range(n_features):
            ref_col = self.reference_data[:, i]
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
        return _severity_for_p_value(p_value, threshold)


class PopulationStabilityIndexDetector:
    """Population Stability Index (PSI) based drift detector.

    PSI is commonly used in credit scoring to measure how much
    a variable has shifted over time. It compares the distribution
    of values between two datasets.

    PSI interpretation:
    - PSI < 0.10: No significant change
    - 0.10 <= PSI < 0.25: Some change, investigate
    - PSI >= 0.25: Major shift, action required

    Bins are reference quantiles with open outer edges, so every bin holds
    about ``n_ref / n_bins`` reference rows and values outside the reference
    range still land in a bin. Counts carry a Jeffreys (+0.5) prior instead
    of a vanishing epsilon, so an empty bin cannot blow PSI up.

    PSI is an effect size, and on small windows its sampling noise alone
    exceeds the 0.10 cut: under no drift, ``PSI * n_eff`` is approximately
    chi-squared with ``n_bins - 1`` degrees of freedom, where
    ``1 / n_eff = 1 / n_ref + 1 / n_cur``. A feature therefore drifts only
    when its PSI reaches the threshold AND that chi-squared test rejects at
    ``p_value_threshold`` (Bonferroni-corrected across features).
    Measured on drift-free N(0, 1) windows: PSI >= 0.10 alone fired on 92%
    of 30-row and 78% of 50-row windows; the chi-squared test held 1.7-4.9%.
    On 8 correlated features (2000 reference rows, 100 drift-free windows)
    the previous equal-width/epsilon detector raised drift on 100% / 100% /
    52% of 50- / 200- / 500-row windows; this one on 2% / 1% / 0%.
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold_low: float = 0.10,
        psi_threshold_high: float = 0.25,
        p_value_threshold: float = 0.05,
    ):
        """Initialize PSI detector.

        Args:
            n_bins: Number of bins for discretization
            psi_threshold_low: Threshold for low drift
            psi_threshold_high: Threshold for high drift
            p_value_threshold: Significance level of the chi-squared null
                test a feature must also pass (Bonferroni across features)
        """
        self.n_bins = n_bins
        self.psi_threshold_low = psi_threshold_low
        self.psi_threshold_high = psi_threshold_high
        self.p_value_threshold = p_value_threshold
        self.reference_data: np.ndarray[Any, Any] | None = None
        self.bin_edges: list[np.ndarray[Any, Any]] = []

    def fit(self, reference_data: np.ndarray[Any, Any]) -> PopulationStabilityIndexDetector:
        """Fit the detector with reference data.

        Args:
            reference_data: Reference data array [n_samples, n_features]

        Returns:
            Self for method chaining
        """
        self.reference_data = np.asarray(reference_data)
        if self.reference_data.ndim == 1:
            self.reference_data = self.reference_data.reshape(-1, 1)

        # Quantile bin edges with open outer edges for each feature
        self.bin_edges = [
            self._quantile_edges(self.reference_data[:, i])
            for i in range(self.reference_data.shape[1])
        ]

        logger.info(f"PSI drift detector fitted with {len(self.reference_data)} reference samples")
        return self

    def _quantile_edges(self, column: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Reference-quantile bin edges, outermost edges at -inf / +inf.

        Repeated quantiles (discrete or constant features) are merged, so a
        feature always gets at least two bins.
        """
        column = np.asarray(column, dtype=np.float64)
        column = column[np.isfinite(column)]
        if column.size == 0:
            return np.array([-np.inf, 0.0, np.inf])
        inner = np.unique(np.quantile(column, np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]))
        if inner.size == 0:
            inner = np.array([float(np.median(column))])
        return np.concatenate([[-np.inf], inner, [np.inf]])

    def _calculate_psi(
        self,
        reference: np.ndarray[Any, Any],
        current: np.ndarray[Any, Any],
        bin_edges: np.ndarray[Any, Any],
    ) -> float:
        """Calculate PSI between two distributions."""
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        # Jeffreys (+0.5) prior: bounded contribution from empty bins.
        prior = 0.5
        ref_pct = (ref_hist + prior) / (ref_hist.sum() + prior * len(ref_hist))
        cur_pct = (cur_hist + prior) / (cur_hist.sum() + prior * len(cur_hist))

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    @staticmethod
    def _psi_p_value(psi: float, n_ref: int, n_cur: int, n_bins: int) -> float:
        """Chi-squared null p-value of a PSI value (no-drift hypothesis)."""
        if n_ref <= 0 or n_cur <= 0 or n_bins < 2:
            return 1.0
        n_eff = 1.0 / (1.0 / n_ref + 1.0 / n_cur)
        return float(stats.chi2.sf(psi * n_eff, n_bins - 1))

    def detect(
        self,
        current_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """Detect drift using PSI.

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

        # Calculate PSI (and its chi-squared null p-value) for each feature
        psi_values = []
        p_values = []
        feature_drifts = {}
        n_ref = len(self.reference_data)

        _require_width(self.reference_data, current_data)
        for i in range(n_features):
            ref_col = self.reference_data[:, i]
            cur_col = current_data[:, i]
            edges = self.bin_edges[i]

            psi = self._calculate_psi(ref_col, cur_col, edges)
            psi_values.append(psi)
            p_values.append(self._psi_p_value(psi, n_ref, len(cur_col), len(edges) - 1))
            feature_drifts[feature_names[i]] = psi

        # A feature drifts when its PSI reaches the threshold and the shift
        # is statistically significant (Bonferroni across features).
        alpha = self.p_value_threshold / max(n_features, 1)
        significant = [
            psi >= self.psi_threshold_low and p < alpha
            for psi, p in zip(psi_values, p_values, strict=True)
        ]
        max_psi = max(psi_values)
        mean_psi = np.mean(psi_values)
        is_drift = any(significant)
        drifting_psi = [psi for psi, sig in zip(psi_values, significant, strict=True) if sig]
        severity = self._compute_severity(max(drifting_psi)) if is_drift else DriftSeverity.NONE

        # Count drifting features
        n_high_drift = sum(1 for p in drifting_psi if p >= self.psi_threshold_high)
        n_low_drift = sum(1 for p in drifting_psi if p < self.psi_threshold_high)
        n_eff = 1.0 / (1.0 / max(n_ref, 1) + 1.0 / max(len(current_data), 1))

        return DriftResult(
            is_drift=is_drift,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            p_value=float(min(p_values)),
            test_statistic=float(max_psi),
            threshold=self.psi_threshold_low,
            feature_drifts=feature_drifts,
            details={
                "all_psi_values": [float(p) for p in psi_values],
                "all_p_values": [float(p) for p in p_values],
                "mean_psi": float(mean_psi),
                "expected_null_psi": float((self.n_bins - 1) / n_eff),
                "adjusted_p_value_threshold": float(alpha),
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
    """Chi-squared test based drift detector for categorical features.

    Uses the chi-squared test of independence to detect changes in the distribution of categorical
    variables.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
    ):
        """Initialize Chi-squared drift detector.

        Args:
            p_value_threshold: Significance level for drift detection
        """
        self.p_value_threshold = p_value_threshold
        self.reference_data: np.ndarray[Any, Any] | None = None
        self.reference_counts: list[dict[Any, int]] = []

    def fit(self, reference_data: np.ndarray[Any, Any]) -> ChiSquaredDriftDetector:
        """Fit the detector with reference data.

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
        current_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """Detect drift using Chi-squared test.

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

        _require_width(self.reference_data, current_data)
        for i in range(n_features):
            ref_counts = self.reference_counts[i]

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
        return _severity_for_p_value(p_value, threshold)


class CovarianceDriftDetector:
    """Covariance-structure drift via the affine-invariant SPD distance.

    Per-feature tests (KS, PSI, chi-squared) compare marginals, so a change
    that keeps every marginal but alters the joint structure -- decoupled
    sensors, a spoofed channel that keeps its own statistics, a broken
    physical relationship -- is invisible to them. This test compares the
    window's covariance with the reference covariance under the
    affine-invariant geodesic distance of the SPD manifold
    (:meth:`SPDManifold.geodesic_distance`).

    The null is calibrated at fit time by leave-window-out resampling: a
    reference window is drawn and compared with the covariance of the
    *remaining* reference rows, so the null includes the reference's own
    estimation noise (an in-sample bootstrap omits it and was measured at
    35% false alarms). Nulls are computed lazily per window-size bucket;
    the bucket is the largest grid size not above the window, i.e. a
    noisier (conservative) null. The p-value is empirical,
    ``(1 + #{null >= observed}) / (B + 1)``.

    Measured (8 features, 2000 reference rows, 100 trials per scenario,
    windows of 50 / 200 / 500 rows): false alarms 0.00-0.02; all
    correlations removed, and one correlation flipped (+0.6 -> -0.6) with
    every marginal still N(0, 1), detected in 100% of windows at every size,
    where the marginal KS/PSI ensemble detected 0-2%.
    """

    #: Features beyond this are projected onto the reference's leading
    #: principal axes so each distance stays an O(d^3) eigenproblem on a
    #: bounded d.
    MAX_DIMENSION: int = 64

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        n_null: int = 200,
        random_state: int = 0,
    ):
        """Initialize the covariance drift detector.

        Args:
            p_value_threshold: Significance level for drift
            n_null: Leave-window-out resamples per window-size bucket
            random_state: Seed of the null resampling (deterministic)
        """
        self.p_value_threshold = p_value_threshold
        self.n_null = int(n_null)
        self.random_state = int(random_state)
        self.reference_data: np.ndarray[Any, Any] | None = None
        self._center: np.ndarray[Any, Any] | None = None
        self._projection: np.ndarray[Any, Any] | None = None
        self._reference_cov: np.ndarray[Any, Any] | None = None
        self._null_cache: dict[int, np.ndarray[Any, Any]] = {}
        # Raw width seen at fit. ``reference_data`` holds the *projected*
        # reference (constant columns dropped, PCA-truncated to
        # MAX_DIMENSION), so its width is not the width a batch must have.
        self._n_features: int | None = None
        self.unavailable_reason: str | None = "not fitted"

    @property
    def min_window(self) -> int:
        """Smallest window (rows) whose covariance is estimated."""
        d = 0 if self._projection is None else self._projection.shape[1]
        return max(2 * d, 20)

    def fit(self, reference_data: np.ndarray[Any, Any]) -> CovarianceDriftDetector:
        """Fit the reference covariance (the null is built lazily per window size).

        Args:
            reference_data: Reference data array [n_samples, n_features]

        Returns:
            Self for method chaining
        """
        ref = np.asarray(reference_data, dtype=np.float64)
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        ref = ref[np.isfinite(ref).all(axis=1)]
        self._null_cache = {}
        self._n_features = int(ref.shape[1])
        self.reference_data = None
        self._projection = None
        self._reference_cov = None
        self.unavailable_reason = None

        if ref.shape[1] < 2:
            self.unavailable_reason = "covariance structure needs at least two features"
            return self
        self._center = ref.mean(axis=0)
        scale = ref.std(axis=0)
        keep = scale > 1e-12
        if int(keep.sum()) < 2:
            self.unavailable_reason = "fewer than two non-constant features"
            return self
        # Standardise (the distance is affine-invariant; this only helps the
        # conditioning) and, for wide data, keep the leading principal axes.
        basis = np.eye(ref.shape[1])[:, keep] / scale[keep]
        z = (ref - self._center) @ basis
        if z.shape[1] > self.MAX_DIMENSION:
            _, _, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
            basis = basis @ vt[: self.MAX_DIMENSION].T
            z = (ref - self._center) @ basis
        self._projection = basis
        if len(z) < 2 * self.min_window:
            self.unavailable_reason = (
                f"reference has {len(z)} rows; needs at least {2 * self.min_window} "
                "to calibrate a leave-window-out null"
            )
            self._projection = None
            return self
        self.reference_data = z
        self._reference_cov = self._covariance(z)
        logger.info(
            "Covariance drift detector fitted with %d reference samples (d=%d)",
            len(z),
            z.shape[1],
        )
        return self

    @staticmethod
    def _covariance(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Sample covariance with a tiny ridge so it is numerically SPD."""
        cov = np.atleast_2d(np.cov(x, rowvar=False))
        ridge = 1e-9 * max(float(np.trace(cov)) / cov.shape[0], 1e-12)
        return cov + ridge * np.eye(cov.shape[0])

    def _distance(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        from omni_mercury_engine.core.riemannian_optimization import SPDManifold

        return SPDManifold(a.shape[0]).geodesic_distance(a, b)

    def _bucket(self, n_cur: int) -> int:
        """Largest sqrt(2)-grid window size <= ``n_cur`` that the reference allows."""
        assert self.reference_data is not None
        lo = self.min_window
        hi = len(self.reference_data) - lo
        size = lo
        while int(size * np.sqrt(2.0)) <= min(n_cur, hi):
            size = int(size * np.sqrt(2.0))
        return min(size, hi)

    def _null(self, window: int) -> np.ndarray[Any, Any]:
        cached = self._null_cache.get(window)
        if cached is not None:
            return cached
        assert self.reference_data is not None
        ref = self.reference_data
        rng = np.random.default_rng(self.random_state + window)
        null = np.empty(self.n_null, dtype=np.float64)
        for b in range(self.n_null):
            perm = rng.permutation(len(ref))
            win, rest = ref[perm[:window]], ref[perm[window:]]
            null[b] = self._distance(self._covariance(rest), self._covariance(win))
        null.sort()
        self._null_cache[window] = null
        return null

    def detect(
        self,
        current_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """Test the current window's covariance against the reference.

        Args:
            current_data: Current data to compare [n_samples, n_features]
            feature_names: Accepted for interface parity (the test is joint)

        Returns:
            DriftResult; ``details["skipped"]`` explains a window or reference
            too small to test (``is_drift`` is then False).
        """
        cur = np.asarray(current_data, dtype=np.float64)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)
        cur = cur[np.isfinite(cur).all(axis=1)]
        skipped: str | None = self.unavailable_reason
        if skipped is None and len(cur) < self.min_window:
            skipped = f"window has {len(cur)} rows; needs at least {self.min_window}"
        if skipped is not None:
            return DriftResult(
                is_drift=False,
                drift_type=DriftType.COVARIANCE_DRIFT,
                severity=DriftSeverity.NONE,
                p_value=1.0,
                test_statistic=0.0,
                threshold=self.p_value_threshold,
                details={"skipped": skipped},
                message=f"Covariance drift test skipped: {skipped}",
            )
        assert self._center is not None and self._projection is not None
        assert self._reference_cov is not None and self.reference_data is not None
        assert self._n_features is not None
        _require_n_features(self._n_features, cur)
        z = (cur - self._center) @ self._projection
        observed = self._distance(self._reference_cov, self._covariance(z))
        window = self._bucket(len(z))
        null = self._null(window)
        exceed = len(null) - int(np.searchsorted(null, observed, side="left"))
        p_value = (1.0 + exceed) / (len(null) + 1.0)
        is_drift = p_value < self.p_value_threshold
        null_q95 = float(np.quantile(null, 0.95))
        if not is_drift:
            severity = DriftSeverity.NONE
        elif observed < 1.5 * null_q95:
            severity = DriftSeverity.LOW
        elif observed < 2.0 * null_q95:
            severity = DriftSeverity.MEDIUM
        elif observed < 3.0 * null_q95:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL
        return DriftResult(
            is_drift=is_drift,
            drift_type=DriftType.COVARIANCE_DRIFT,
            severity=severity,
            p_value=float(p_value),
            test_statistic=float(observed),
            threshold=self.p_value_threshold,
            details={
                "null_window_size": window,
                "null_q95": null_q95,
                "n_null": len(null),
                "dimension": int(z.shape[1]),
                "window_size": len(z),
            },
            message=(
                f"Covariance structure drift (SPD distance {observed:.3f}, p={p_value:.4f})"
                if is_drift
                else f"No covariance drift (SPD distance {observed:.3f}, p={p_value:.4f})"
            ),
        )


class PrincipalAxisDriftDetector:
    """Mean-shift drift along the reference's principal axes.

    Per-feature tests compare one column at a time, so in correlated data a
    shift along a direction of low variance moves every column by only a
    fraction of its spread and goes unseen; the covariance test cannot see it
    either, because it centres each window on its own mean. This test rotates
    the window onto the reference's principal axes and runs a
    Kolmogorov-Smirnov test on every axis, Bonferroni-corrected: the PCA
    variant of the aggregated univariate tests that Rabanser, Guennemann and
    Lipton (2019, arXiv:1810.11953, section 5) found comparable to
    multivariate two-sample tests.

    The axes are learned from the even reference rows and the compared
    distribution comes from the odd rows. Fitting both on the same rows
    measured 6% (64 features, 500 rows), 25% (128 features, 1000 rows) and
    100% (128 features, 400 rows) false alarms, because the axes follow the
    reference's own noise; split, they measured 0%, 2% and 0%.

    Measured by ``benchmarks/drift_power.py`` (100 windows per cell over 5
    independent references; 8 features correlated at 0.9, 2000 reference rows):
    added to the KS, PSI and covariance ensemble, false alarms went from 0-2% to
    1-5%. A shift of one standard deviation along the weakest axis (about 0.1 SD
    in any single feature) was flagged in 100% of 64, 200 and 1000-row windows,
    where the ensemble without it flagged 1%, 8% and 0%. With 128 features
    correlated at 0.9 and 1000 reference rows, one SD went from 0% to 57% and two
    SD from 2% to 100%. Limit: the axes need rows to be learned from; with 128
    features correlated at 0.95 and only 400 reference rows it flags nothing
    beyond the other tests (and adds no false alarms).
    """

    #: Reference rows below which the axes are not fitted.
    MIN_REFERENCE: int = 40
    #: Window rows below which the test is skipped.
    MIN_WINDOW: int = 8

    def __init__(self, p_value_threshold: float = 0.01):
        """Initialize the principal-axis drift detector.

        Args:
            p_value_threshold: Family-wise significance level, split over the
                axes by Bonferroni. Lower than the other tests' 0.05 because
                this test is added to their vote, not substituted for it.
        """
        self.p_value_threshold = p_value_threshold
        self._center: np.ndarray[Any, Any] | None = None
        self._axes: np.ndarray[Any, Any] | None = None
        self._reference_scores: np.ndarray[Any, Any] | None = None
        self._n_features: int | None = None
        self.unavailable_reason: str | None = "not fitted"

    def fit(self, reference_data: np.ndarray[Any, Any]) -> PrincipalAxisDriftDetector:
        """Learn the principal axes and the reference distribution along them.

        Args:
            reference_data: Reference data array [n_samples, n_features]

        Returns:
            Self for method chaining
        """
        ref = np.asarray(reference_data, dtype=np.float64)
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        ref = ref[np.isfinite(ref).all(axis=1)]
        self._n_features = int(ref.shape[1])
        self._center = self._axes = self._reference_scores = None
        self.unavailable_reason = None

        if ref.shape[1] < 2:
            self.unavailable_reason = "principal axes need at least two features"
        elif len(ref) < max(self.MIN_REFERENCE, 4 * ref.shape[1]):
            self.unavailable_reason = (
                f"reference has {len(ref)} rows; needs at least "
                f"{max(self.MIN_REFERENCE, 4 * ref.shape[1])} to split into axes and distribution"
            )
        if self.unavailable_reason is not None:
            return self

        axis_rows, distribution_rows = ref[0::2], ref[1::2]
        center = axis_rows.mean(axis=0)
        _, spread, axes = np.linalg.svd(axis_rows - center, full_matrices=False)
        keep = spread > 1e-9 * spread[0] if spread[0] > 0 else np.zeros(len(spread), dtype=bool)
        if not keep.any():
            self.unavailable_reason = "reference has no variance"
            return self
        self._center = center
        self._axes = axes[keep].T
        self._reference_scores = np.sort((distribution_rows - center) @ self._axes, axis=0)
        logger.info(
            "Principal-axis drift detector fitted on %d reference rows (%d axes)",
            len(ref),
            self._axes.shape[1],
        )
        return self

    def _ks_statistics(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Two-sample KS statistic of every axis, against the pre-sorted reference.

        Equal to ``scipy.stats.ks_2samp(...).statistic`` per column (both
        evaluate the ECDF difference at every pooled point, right-continuous),
        without sorting the reference again for every request.
        """
        assert self._reference_scores is not None
        reference = self._reference_scores
        n_reference, n_window = len(reference), len(scores)
        statistics = np.empty(scores.shape[1])
        for axis in range(scores.shape[1]):
            ref, window = reference[:, axis], np.sort(scores[:, axis])
            at_window = np.searchsorted(ref, window, side="right") / n_reference - (
                np.searchsorted(window, window, side="right") / n_window
            )
            at_reference = np.searchsorted(ref, ref, side="right") / n_reference - (
                np.searchsorted(window, ref, side="right") / n_window
            )
            statistics[axis] = max(np.abs(at_window).max(), np.abs(at_reference).max())
        return statistics

    def detect(
        self,
        current_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """Test the window's distribution along every principal axis.

        Args:
            current_data: Current data to compare [n_samples, n_features]
            feature_names: Accepted for interface parity (the test is joint)

        Returns:
            DriftResult; ``details["skipped"]`` explains a window or reference
            too small to test (``is_drift`` is then False).
        """
        cur = np.asarray(current_data, dtype=np.float64)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)
        cur = cur[np.isfinite(cur).all(axis=1)]
        skipped = self.unavailable_reason
        if skipped is None and len(cur) < self.MIN_WINDOW:
            skipped = f"window has {len(cur)} rows; needs at least {self.MIN_WINDOW}"
        if skipped is not None:
            return DriftResult(
                is_drift=False,
                drift_type=DriftType.PRINCIPAL_AXIS_DRIFT,
                severity=DriftSeverity.NONE,
                p_value=1.0,
                test_statistic=0.0,
                threshold=self.p_value_threshold,
                details={"skipped": skipped},
                message=f"Principal-axis drift test skipped: {skipped}",
            )
        assert self._center is not None and self._axes is not None
        assert self._reference_scores is not None and self._n_features is not None
        _require_n_features(self._n_features, cur)

        scores = (cur - self._center) @ self._axes
        statistics = self._ks_statistics(scores)
        threshold = self.p_value_threshold / len(statistics)
        # Every axis compares the same two sample sizes, and the KS p-value
        # falls as the statistic grows, so the smallest p-value belongs to the
        # axis with the largest statistic: only that one needs its exact p.
        worst = int(np.argmax(statistics))
        p_value = float(stats.ks_2samp(self._reference_scores[:, worst], scores[:, worst]).pvalue)
        is_drift = p_value < threshold
        return DriftResult(
            is_drift=is_drift,
            drift_type=DriftType.PRINCIPAL_AXIS_DRIFT,
            severity=_severity_for_p_value(p_value, threshold),
            p_value=p_value,
            test_statistic=float(statistics.max()),
            threshold=threshold,
            details={
                "n_axes": len(statistics),
                "worst_axis": worst,
                "all_statistics": [float(s) for s in statistics],
                "window_size": len(cur),
            },
            message=(
                f"Mean shifted along principal axis {worst} (p={p_value:.2e})"
                if is_drift
                else f"No principal-axis drift (p={p_value:.4f})"
            ),
        )


class EnsembleDriftDetector:
    """Ensemble drift detector combining multiple methods.

    The marginal tests (KS, PSI) vote according to ``voting``. The
    covariance-structure and principal-axis tests answer a different
    question -- whether the joint distribution changed while the marginals
    barely did -- so they are not voters: drift is reported when the marginal
    vote passes OR either joint test rejects. ``covariance=False`` and
    ``principal_axes=False`` drop them.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.10,
        voting: str = "majority",
        covariance: bool = True,
        principal_axes: bool = True,
    ):
        """Initialize ensemble drift detector.

        Args:
            p_value_threshold: Significance level for statistical tests
            psi_threshold: Threshold for PSI test
            voting: Voting strategy ('majority', 'any', 'all')
            covariance: Also run the covariance-structure drift test
            principal_axes: Also run the principal-axis mean-shift test
        """
        self.voting = voting
        self.ks_detector = KolmogorovSmirnovDriftDetector(p_value_threshold)
        self.psi_detector = PopulationStabilityIndexDetector(
            psi_threshold_low=psi_threshold, p_value_threshold=p_value_threshold
        )
        self.cov_detector: CovarianceDriftDetector | None = (
            CovarianceDriftDetector(p_value_threshold) if covariance else None
        )
        self.axis_detector: PrincipalAxisDriftDetector | None = (
            PrincipalAxisDriftDetector() if principal_axes else None
        )
        self.is_fitted = False

    def fit(self, reference_data: np.ndarray[Any, Any]) -> EnsembleDriftDetector:
        """Fit all detectors with reference data.

        Args:
            reference_data: Reference data array [n_samples, n_features]

        Returns:
            Self for method chaining
        """
        self.ks_detector.fit(reference_data)
        self.psi_detector.fit(reference_data)
        if self.cov_detector is not None:
            self.cov_detector.fit(reference_data)
        if self.axis_detector is not None:
            self.axis_detector.fit(reference_data)
        self.is_fitted = True
        logger.info("Ensemble drift detector fitted")
        return self

    def detect(
        self,
        current_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
    ) -> DriftResult:
        """Detect drift using ensemble of methods.

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

        cov_result = (
            self.cov_detector.detect(current_data, feature_names)
            if self.cov_detector is not None
            else None
        )
        axis_result = (
            self.axis_detector.detect(current_data, feature_names)
            if self.axis_detector is not None
            else None
        )
        marginal_drift = bool(is_drift)
        covariance_drift = bool(cov_result is not None and cov_result.is_drift)
        axis_drift = bool(axis_result is not None and axis_result.is_drift)
        is_drift = marginal_drift or covariance_drift or axis_drift

        # Severity comes only from the tests that carried the decision. A
        # voter that rejected alone, outvoted by the others, used to set the
        # ensemble severity anyway, so "no drift" could arrive with severity
        # HIGH and the decision layer deferred on it.
        severities = [DriftSeverity.NONE]
        if marginal_drift:
            severities += [r.severity for r in results if r.is_drift]
        if covariance_drift and cov_result is not None:
            severities.append(cov_result.severity)
        if axis_drift and axis_result is not None:
            severities.append(axis_result.severity)
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

        joint_findings = [
            note
            for found, note in (
                (covariance_drift, "covariance structure changed"),
                (axis_drift, "mean shifted along a principal axis"),
            )
            if found
        ]
        if joint_findings and not marginal_drift:
            message = f"Ensemble drift: {' and '.join(joint_findings)} (marginals unchanged)"
        elif is_drift:
            message = f"Ensemble drift: {sum(drift_votes)}/{len(drift_votes)} detectors triggered"
            if joint_findings:
                message += f" and {' and '.join(joint_findings)}"
        else:
            message = "No ensemble drift detected"

        return DriftResult(
            is_drift=is_drift,
            drift_type=(
                DriftType.DATA_DRIFT
                if marginal_drift
                else (
                    DriftType.COVARIANCE_DRIFT
                    if covariance_drift
                    else DriftType.PRINCIPAL_AXIS_DRIFT if axis_drift else DriftType.DATA_DRIFT
                )
            ),
            severity=max_severity,
            p_value=float(np.mean([r.p_value for r in results])),
            test_statistic=float(np.mean([r.test_statistic for r in results])),
            threshold=0.0,  # Ensemble doesn't have single threshold
            feature_drifts=avg_feature_drifts,
            details={
                "ks_result": ks_result.to_dict(),
                "psi_result": psi_result.to_dict(),
                "covariance_result": cov_result.to_dict() if cov_result is not None else None,
                "principal_axis_result": (
                    axis_result.to_dict() if axis_result is not None else None
                ),
                "voting_strategy": self.voting,
                "votes": drift_votes,
                "marginal_drift": marginal_drift,
                "covariance_drift": covariance_drift,
                "principal_axis_drift": axis_drift,
            },
            message=message,
        )


DriftDetectorType = (
    KolmogorovSmirnovDriftDetector
    | PopulationStabilityIndexDetector
    | ChiSquaredDriftDetector
    | CovarianceDriftDetector
    | PrincipalAxisDriftDetector
    | EnsembleDriftDetector
)


def create_drift_detector(
    detector_type: str = "ks",
    **kwargs: Any,
) -> DriftDetectorType:
    """Factory function to create drift detectors.

    Args:
        detector_type: Type of detector ('ks', 'psi', 'chi2', 'covariance',
            'principal_axes', 'ensemble')
        **kwargs: Detector-specific parameters

    Returns:
        Configured drift detector
    """
    detectors: dict[str, type[DriftDetectorType]] = {
        "ks": KolmogorovSmirnovDriftDetector,
        "psi": PopulationStabilityIndexDetector,
        "chi2": ChiSquaredDriftDetector,
        "covariance": CovarianceDriftDetector,
        "principal_axes": PrincipalAxisDriftDetector,
        "ensemble": EnsembleDriftDetector,
    }

    if detector_type not in detectors:
        raise ValueError(
            f"Unknown detector type: {detector_type}. Choose from {list(detectors.keys())}"
        )

    detector_class = detectors[detector_type]
    return detector_class(**kwargs)


class OnlineDriftDetector:
    """Online drift detection with adaptive windowing (ADWIN).

    ADWIN0 of Bifet & Gavalda (2007, "Learning from Time-Changing Data with
    Adaptive Windowing", SDM): after each sample, every split of the window
    into an older part ``W0`` and a newer part ``W1`` is tested, and drift is
    declared when the means differ by more than

        eps_cut = sqrt(2 / m * var * ln(2 / d')) + 2 R / (3 m) * ln(2 / d'),

    with ``m`` the harmonic mean of the part sizes, ``var`` the window
    variance, ``R`` the window's range and ``d' = delta / (n * features)``
    (the paper's bound, its ``delta / n`` correction for testing every
    split, and a Bonferroni share per feature). The paper takes values in
    [0, 1]; ``R`` carries the bound to any units, so the same stream raises
    the same drifts in metres or millimetres. The older part is then dropped. ``significance_level`` is
    ``delta``; ADWIN computes no p-value, so a result's ``p_value`` carries
    ``delta``. The former check ran a KS test per feature at ``delta`` on
    every update with no correction, so a stationary 3-feature stream raised
    12-31 false drifts per 2000 samples.
    """

    def __init__(
        self,
        max_window_size: int = 1000,
        min_window_size: int = 50,
        significance_level: float = 0.05,
    ):
        """Initialize online drift detector.

        Args:
            max_window_size: Maximum samples to keep in window
            min_window_size: Minimum samples before detection starts (and the
                smallest part a split may leave on either side, a fifth of it)
            significance_level: ADWIN ``delta``
        """
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.significance_level = significance_level
        self.window: list[np.ndarray[Any, Any]] = []
        self.drift_detected = False
        self.drift_count = 0
        self._n_features = 1

    def _first_cut(self, values: np.ndarray[Any, Any]) -> tuple[int, float, float] | None:
        """Earliest split ``(index, |mean0 - mean1|, eps_cut)`` that exceeds its bound."""
        n = values.size
        min_part = max(5, self.min_window_size // 5)
        if n < 2 * min_part:
            return None
        variance = float(values.var())
        if variance == 0.0:
            return None
        value_range = float(values.max() - values.min())
        cumulative = np.cumsum(values)
        total = cumulative[-1]
        cuts = np.arange(min_part, n - min_part + 1)
        n0 = cuts.astype(float)
        n1 = n - n0
        mean0 = cumulative[cuts - 1] / n0
        mean1 = (total - cumulative[cuts - 1]) / n1
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        log_term = np.log(2.0 / (self.significance_level / (n * self._n_features)))
        eps = np.sqrt(2.0 / m * variance * log_term) + 2.0 * value_range / (3.0 * m) * log_term
        exceeded = np.flatnonzero(np.abs(mean0 - mean1) > eps)
        if exceeded.size == 0:
            return None
        k = int(exceeded[0])
        return int(cuts[k]), float(abs(mean0[k] - mean1[k])), float(eps[k])

    def update(self, sample: np.ndarray[Any, Any]) -> DriftResult | None:
        """Update detector with new sample and check for drift.

        Args:
            sample: New sample to add to window

        Returns:
            DriftResult if drift detected, None otherwise
        """
        sample = np.asarray(sample, dtype=float).flatten()
        self.window.append(sample)
        self._n_features = sample.shape[0]

        # Trim window if too large
        if len(self.window) > self.max_window_size:
            self.window = self.window[-self.max_window_size :]

        # Check for drift if we have enough samples
        if len(self.window) < self.min_window_size:
            return None

        data = np.array(self.window)
        hits = [
            (feature, cut)
            for feature in range(data.shape[1])
            if (cut := self._first_cut(data[:, feature])) is not None
        ]
        if not hits:
            return None

        feature, (index, difference, bound) = min(hits, key=lambda h: h[1][0])
        self.drift_detected = True
        self.drift_count += 1
        # Drop the part of the window that predates the change.
        self.window = self.window[index:]
        return DriftResult(
            is_drift=True,
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=DriftSeverity.MEDIUM,
            p_value=self.significance_level,
            test_statistic=difference,
            threshold=bound,
            details={
                "window_size": len(self.window),
                "drift_count": self.drift_count,
                "feature": feature,
                "method": "adwin0",
            },
            message=f"Online drift detected (total drifts: {self.drift_count})",
        )

    def reset(self) -> None:
        """Reset the detector state."""
        self.window = []
        self.drift_detected = False
        self.drift_count = 0
