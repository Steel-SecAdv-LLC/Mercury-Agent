"""
Mercury Agent - Score Calibration System
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Comprehensive score calibration system to solve the F1=0 problem:
- Root cause: Good ROC-AUC (0.88) but zero F1 because scores < threshold
- Solution: Auto-calibrate thresholds based on score distribution

This module provides:
1. ScoreCalibrationManager - Unified calibration orchestrator
2. AutoThresholdOptimizer - Multiple threshold selection strategies
3. ScoreDiagnostics - Debug tools for score distribution analysis
4. CalibrationPipeline - End-to-end calibration workflow
5. ThresholdConfidenceInterval - Bootstrap-based confidence intervals for thresholds
6. LabelSmoothingCalibrator - Label smoothing for improved calibration
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

# P2: Import from centralized constants


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Available threshold calibration methods."""

    FIXED = "fixed"  # Use fixed threshold (default 0.5)
    PERCENTILE = "percentile"  # Use percentile based on contamination
    OTSU = "otsu"  # Otsu's bimodal threshold selection
    MAD = "mad"  # Median Absolute Deviation based
    KNEE = "knee"  # Knee/elbow detection in sorted scores
    OPTIMAL_F1 = "optimal_f1"  # Find threshold maximizing F1 (requires labels)
    YOUDEN_J = "youden_j"  # Find threshold maximizing Youden's J (requires labels)
    ADAPTIVE_IQR = "adaptive_iqr"  # IQR-based adaptive threshold
    GAUSSIAN_MIXTURE = "gmm"  # Gaussian Mixture Model separation
    AUTO = "auto"  # Automatically select best method


@dataclass
class CalibrationDiagnostics:
    """Diagnostic information about score distribution and calibration.

    This class provides the diagnostic output requested by the user:
    - Score range, mean, std
    - Threshold used
    - Predictions above threshold
    - Distribution characteristics
    """

    # Basic statistics
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    score_median: float

    # Threshold information
    threshold: float
    calibration_method: str

    # Prediction statistics
    n_samples: int
    n_above_threshold: int
    predicted_anomaly_ratio: float

    # Distribution characteristics
    is_bimodal: bool
    skewness: float
    kurtosis: float

    # Percentiles for understanding distribution
    percentiles: dict[int, float] = field(default_factory=dict)

    # Calibration quality metrics (if labels available)
    estimated_contamination: float | None = None
    actual_contamination: float | None = None

    def __str__(self) -> str:
        """Format diagnostics for display."""
        lines = [
            "=" * 60,
            "SCORE CALIBRATION DIAGNOSTICS",
            "=" * 60,
            "",
            "Score Distribution:",
            f"  Range: [{self.score_min:.4f}, {self.score_max:.4f}]",
            f"  Mean:  {self.score_mean:.4f}",
            f"  Std:   {self.score_std:.4f}",
            f"  Median: {self.score_median:.4f}",
            "",
            f"Threshold: {self.threshold:.4f} (method: {self.calibration_method})",
            f"Predictions above threshold: {self.n_above_threshold}/{self.n_samples}",
            f"Predicted anomaly ratio: {self.predicted_anomaly_ratio:.4f}",
            "",
            "Distribution Characteristics:",
            f"  Bimodal: {self.is_bimodal}",
            f"  Skewness: {self.skewness:.4f}",
            f"  Kurtosis: {self.kurtosis:.4f}",
            "",
            "Percentiles:",
        ]

        for p, v in sorted(self.percentiles.items()):
            lines.append(f"  P{p}: {v:.4f}")

        if self.actual_contamination is not None:
            lines.extend(
                [
                    "",
                    "Contamination:",
                    (
                        f"  Estimated: {self.estimated_contamination:.4f}"
                        if self.estimated_contamination
                        else "  Estimated: N/A"
                    ),
                    f"  Actual: {self.actual_contamination:.4f}",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_median": self.score_median,
            "threshold": self.threshold,
            "calibration_method": self.calibration_method,
            "n_samples": self.n_samples,
            "n_above_threshold": self.n_above_threshold,
            "predicted_anomaly_ratio": self.predicted_anomaly_ratio,
            "is_bimodal": self.is_bimodal,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "percentiles": self.percentiles,
            "estimated_contamination": self.estimated_contamination,
            "actual_contamination": self.actual_contamination,
        }


@dataclass
class ThresholdConfidenceInterval:
    """Confidence interval for a calibrated threshold.

    Provides statistical uncertainty quantification for threshold estimates
    using bootstrap resampling.

    Example:
        >>> ci = ThresholdConfidenceInterval(
        ...     threshold=0.42, lower=0.38, upper=0.46,
        ...     confidence_level=0.95, method="bootstrap"
        ... )
        >>> print(f"Threshold: {ci.threshold:.3f} (95% CI: [{ci.lower:.3f}, {ci.upper:.3f}])")
    """

    threshold: float  # Point estimate
    lower: float  # Lower bound of CI
    upper: float  # Upper bound of CI
    confidence_level: float  # e.g., 0.95 for 95% CI
    method: str  # How CI was computed (bootstrap, analytical, etc.)
    n_bootstrap: int = 0  # Number of bootstrap samples used
    std_error: float = 0.0  # Standard error of threshold estimate

    def __str__(self) -> str:
        """Format confidence interval for display."""
        pct = int(self.confidence_level * 100)
        return (
            f"Threshold: {self.threshold:.4f} " f"({pct}% CI: [{self.lower:.4f}, {self.upper:.4f}])"
        )

    def contains(self, value: float) -> bool:
        """Check if a value falls within the confidence interval."""
        return self.lower <= value <= self.upper

    def width(self) -> float:
        """Return the width of the confidence interval."""
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "threshold": self.threshold,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "n_bootstrap": self.n_bootstrap,
            "std_error": self.std_error,
        }


@dataclass
class CalibrationResult:
    """Result of threshold calibration."""

    threshold: float
    method: CalibrationMethod
    predictions: NDArray[np.bool_]
    diagnostics: CalibrationDiagnostics
    confidence: float = 1.0  # Confidence in the calibration (0-1)

    # Confidence interval for the threshold (optional)
    confidence_interval: ThresholdConfidenceInterval | None = None

    # Metadata
    method_specific_info: dict[str, Any] = field(default_factory=dict)


class ScoreDiagnostics:
    """Tools for analyzing and diagnosing score distributions.

    Use this class to understand why F1=0 when ROC-AUC is good:
    - Are scores clustered below the threshold?
    - Is the score distribution bimodal?
    - What percentile does the current threshold correspond to?
    """

    @staticmethod
    def _validate_scores(scores: NDArray[np.float64], min_samples: int = 1) -> NDArray[np.float64]:
        """Validate and clean score array.

        Args:
            scores: Input scores array
            min_samples: Minimum required samples

        Returns:
            Validated and cleaned score array

        Raises:
            ValueError: If scores are invalid
        """
        scores = np.asarray(scores).flatten().astype(np.float64)

        if len(scores) == 0:
            raise ValueError("Cannot process empty score array")

        if len(scores) < min_samples:
            raise ValueError(f"Need at least {min_samples} samples, got {len(scores)}")

        # Check for NaN/Inf
        n_invalid = np.sum(~np.isfinite(scores))
        if n_invalid > 0:
            logger.warning(
                f"Found {n_invalid}/{len(scores)} NaN/Inf values in scores. "
                "Replacing with median of valid values."
            )
            valid_scores = scores[np.isfinite(scores)]
            if len(valid_scores) == 0:
                raise ValueError("All scores are NaN or Inf - cannot calibrate")
            median_val = np.median(valid_scores)
            scores = np.where(np.isfinite(scores), scores, median_val)

        return scores

    @staticmethod
    def _validate_labels(labels: NDArray[np.int32] | None) -> NDArray[np.int32] | None:
        """Validate binary labels array.

        Args:
            labels: Input labels array (optional)

        Returns:
            Validated labels array or None

        Raises:
            ValueError: If labels are invalid
        """
        if labels is None:
            return None

        labels = np.asarray(labels).flatten()

        # Convert to int if needed
        if labels.dtype not in (np.int32, np.int64, np.uint8, np.bool_):
            labels = labels.astype(np.int32)

        # Validate binary
        unique_vals = np.unique(labels)
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(f"Labels must be binary (0 or 1), got unique values: {unique_vals}")

        return labels.astype(np.int32)

    @staticmethod
    def analyze(
        scores: NDArray[np.float64],
        threshold: float = 0.5,
        labels: NDArray[np.int32] | None = None,
        method: str = "unknown",
    ) -> CalibrationDiagnostics:
        """
        Comprehensive analysis of score distribution.

        Args:
            scores: Anomaly scores array (must be non-empty, finite values)
            threshold: Current threshold value
            labels: Optional ground truth labels for contamination calculation
            method: Calibration method name

        Returns:
            CalibrationDiagnostics with full analysis

        Raises:
            ValueError: If scores array is empty or all NaN/Inf
        """
        # Validate inputs
        try:
            scores = ScoreDiagnostics._validate_scores(scores, min_samples=1)
        except ValueError:
            # Return empty diagnostics for edge cases
            return CalibrationDiagnostics(
                score_min=0.0,
                score_max=0.0,
                score_mean=0.0,
                score_std=0.0,
                score_median=0.0,
                threshold=threshold,
                calibration_method=method,
                n_samples=0,
                n_above_threshold=0,
                predicted_anomaly_ratio=0.0,
                is_bimodal=False,
                skewness=0.0,
                kurtosis=0.0,
                percentiles={},
            )

        labels = ScoreDiagnostics._validate_labels(labels)
        n = len(scores)

        if n == 0:
            return CalibrationDiagnostics(
                score_min=0.0,
                score_max=0.0,
                score_mean=0.0,
                score_std=0.0,
                score_median=0.0,
                threshold=threshold,
                calibration_method=method,
                n_samples=0,
                n_above_threshold=0,
                predicted_anomaly_ratio=0.0,
                is_bimodal=False,
                skewness=0.0,
                kurtosis=0.0,
                percentiles={},
            )

        # Basic statistics
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores))
        score_median = float(np.median(scores))

        # Prediction statistics
        n_above = int(np.sum(scores > threshold))
        pred_ratio = n_above / n if n > 0 else 0.0

        # Distribution characteristics
        is_bimodal = ScoreDiagnostics._detect_bimodality(scores)
        skewness = ScoreDiagnostics._compute_skewness(scores)
        kurtosis = ScoreDiagnostics._compute_kurtosis(scores)

        # Key percentiles
        percentiles = {
            1: float(np.percentile(scores, 1)),
            5: float(np.percentile(scores, 5)),
            10: float(np.percentile(scores, 10)),
            25: float(np.percentile(scores, 25)),
            50: float(np.percentile(scores, 50)),
            75: float(np.percentile(scores, 75)),
            90: float(np.percentile(scores, 90)),
            95: float(np.percentile(scores, 95)),
            99: float(np.percentile(scores, 99)),
        }

        # Contamination estimates
        estimated_contamination = ScoreDiagnostics._estimate_contamination(scores)
        actual_contamination = None
        if labels is not None:
            labels = np.asarray(labels).flatten()
            actual_contamination = float(np.mean(labels))

        return CalibrationDiagnostics(
            score_min=score_min,
            score_max=score_max,
            score_mean=score_mean,
            score_std=score_std,
            score_median=score_median,
            threshold=threshold,
            calibration_method=method,
            n_samples=n,
            n_above_threshold=n_above,
            predicted_anomaly_ratio=pred_ratio,
            is_bimodal=is_bimodal,
            skewness=skewness,
            kurtosis=kurtosis,
            percentiles=percentiles,
            estimated_contamination=estimated_contamination,
            actual_contamination=actual_contamination,
        )

    @staticmethod
    def _detect_bimodality(scores: NDArray[np.float64]) -> bool:
        """Detect if score distribution is bimodal using dip test heuristic."""
        if len(scores) < 20:
            return False

        # Use histogram-based heuristic
        hist, bin_edges = np.histogram(scores, bins=50)

        # Smooth histogram
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        smoothed = np.convolve(hist, kernel, mode="same")

        # Count local maxima
        local_maxima = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                # Only count significant peaks (> 5% of max)
                if smoothed[i] > 0.05 * np.max(smoothed):
                    local_maxima += 1

        return local_maxima >= 2

    @staticmethod
    def _compute_skewness(scores: NDArray[np.float64]) -> float:
        """Compute skewness of distribution."""
        n = len(scores)
        if n < 3:
            return 0.0

        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-10:
            return 0.0

        return float(np.mean(((scores - mean) / std) ** 3))

    @staticmethod
    def _compute_kurtosis(scores: NDArray[np.float64]) -> float:
        """Compute excess kurtosis of distribution."""
        n = len(scores)
        if n < 4:
            return 0.0

        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-10:
            return 0.0

        return float(np.mean(((scores - mean) / std) ** 4) - 3.0)

    @staticmethod
    def _estimate_contamination(scores: NDArray[np.float64]) -> float:
        """Estimate contamination ratio from score distribution."""
        if len(scores) < 10:
            return 0.05  # Default

        # Use IQR-based estimation
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1

        if iqr < 1e-10:
            # Scores too uniform, use tail-based estimate
            p95 = np.percentile(scores, 95)
            return float(np.mean(scores > p95))

        upper_fence = q3 + 1.5 * iqr
        return float(np.clip(np.mean(scores > upper_fence), 0.001, 0.5))

    @staticmethod
    def print_quick_diagnostic(
        scores: NDArray[np.float64],
        threshold: float = 0.5,
        detector_name: str = "Unknown",
    ) -> None:
        """
        Print quick diagnostic matching user's requested format.

        This implements the exact diagnostic the user requested:
        ```
        Score range: [min, max]
        Score mean: mean
        Threshold: threshold
        Predictions above threshold: count/total
        ```
        """
        scores = np.asarray(scores).flatten()

        print(f"\n--- {detector_name} Score Diagnostics ---")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"Score mean: {scores.mean():.4f}")
        print(f"Threshold: {threshold}")
        print(f"Predictions above threshold: {(scores > threshold).sum()}/{len(scores)}")
        print("-" * 40)


class AutoThresholdOptimizer:
    """
    Automatic threshold optimization using multiple strategies.

    This is the core solution to the F1=0 problem: instead of using
    a fixed 0.5 threshold, we compute an optimal threshold based on
    the actual score distribution.

    Attributes:
        default_contamination: Expected anomaly rate (0.0-1.0)
        min_contamination: Minimum contamination to enforce
        max_contamination: Maximum contamination to allow

    Example:
        >>> optimizer = AutoThresholdOptimizer(contamination=0.05)
        >>> result = optimizer.optimize(scores, method=CalibrationMethod.AUTO)
        >>> print(f"Threshold: {result.threshold}")
    """

    def __init__(
        self,
        default_contamination: float = 0.05,
        min_contamination: float = 0.001,
        max_contamination: float = 0.5,
    ):
        """
        Initialize threshold optimizer.

        Args:
            default_contamination: Expected anomaly rate (default 5%)
            min_contamination: Minimum contamination to enforce
            max_contamination: Maximum contamination to allow

        Raises:
            ValueError: If contamination parameters are invalid
        """
        # Validate contamination parameters
        if not (0.0 < default_contamination < 1.0):
            raise ValueError(
                f"default_contamination must be in (0, 1), got {default_contamination}"
            )
        if not (0.0 < min_contamination < max_contamination < 1.0):
            raise ValueError(
                f"Require 0 < min_contamination < max_contamination < 1, "
                f"got min={min_contamination}, max={max_contamination}"
            )
        if not (min_contamination <= default_contamination <= max_contamination):
            raise ValueError(
                f"default_contamination must be in [min, max], "
                f"got {default_contamination} not in [{min_contamination}, {max_contamination}]"
            )

        self.default_contamination = default_contamination
        self.min_contamination = min_contamination
        self.max_contamination = max_contamination

        # Method registry
        self._methods: dict[CalibrationMethod, Callable[..., tuple[float, dict[str, Any]]]] = {
            CalibrationMethod.FIXED: self._fixed_threshold,
            CalibrationMethod.PERCENTILE: self._percentile_threshold,
            CalibrationMethod.OTSU: self._otsu_threshold,
            CalibrationMethod.MAD: self._mad_threshold,
            CalibrationMethod.KNEE: self._knee_threshold,
            CalibrationMethod.ADAPTIVE_IQR: self._adaptive_iqr_threshold,
            CalibrationMethod.GAUSSIAN_MIXTURE: self._gmm_threshold,
        }

    def optimize(
        self,
        scores: NDArray[np.float64],
        method: CalibrationMethod = CalibrationMethod.AUTO,
        contamination: float | None = None,
        labels: NDArray[np.int32] | None = None,
        fixed_threshold: float = 0.5,
    ) -> CalibrationResult:
        """
        Find optimal threshold for the given scores.

        Args:
            scores: Anomaly scores (higher = more anomalous)
            method: Calibration method to use
            contamination: Expected anomaly ratio (overrides default)
            labels: Optional ground truth for optimal F1 calculation
            fixed_threshold: Threshold for FIXED method

        Returns:
            CalibrationResult with threshold, predictions, and diagnostics
        """
        scores = np.asarray(scores).flatten().astype(np.float64)
        contamination = contamination or self.default_contamination

        if len(scores) == 0:
            return CalibrationResult(
                threshold=fixed_threshold,
                method=CalibrationMethod.FIXED,
                predictions=np.array([], dtype=bool),
                diagnostics=ScoreDiagnostics.analyze(scores, fixed_threshold),
                confidence=0.0,
            )

        # Handle AUTO method
        if method == CalibrationMethod.AUTO:
            method, confidence = self._select_best_method(scores, labels)
        else:
            confidence = 1.0

        # Handle OPTIMAL_F1 separately (requires labels)
        if method == CalibrationMethod.OPTIMAL_F1:
            if labels is None:
                logger.warning("OPTIMAL_F1 requires labels, falling back to PERCENTILE")
                method = CalibrationMethod.PERCENTILE
            else:
                return self._optimal_f1_threshold(scores, labels)

        # Handle YOUDEN_J separately (requires labels)
        if method == CalibrationMethod.YOUDEN_J:
            if labels is None:
                logger.warning("YOUDEN_J requires labels, falling back to PERCENTILE")
                method = CalibrationMethod.PERCENTILE
            else:
                return self._youden_j_threshold(scores, labels)

        # Get threshold from selected method
        method_func: Callable[..., tuple[float, dict[str, Any]]] = self._methods.get(
            method, self._percentile_threshold
        )
        threshold, method_info = method_func(scores, contamination, fixed_threshold)

        # Ensure threshold is within score range
        threshold = float(np.clip(threshold, scores.min() - 0.001, scores.max() + 0.001))

        # Generate predictions
        predictions = scores > threshold

        # Generate diagnostics
        diagnostics = ScoreDiagnostics.analyze(scores, threshold, labels, method.value)

        return CalibrationResult(
            threshold=threshold,
            method=method,
            predictions=predictions,
            diagnostics=diagnostics,
            confidence=confidence,
            method_specific_info=method_info,
        )

    def _select_best_method(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32] | None = None,
    ) -> tuple[CalibrationMethod, float]:
        """
        Automatically select the best calibration method.

        Decision logic:
        1. If labels available -> OPTIMAL_F1
        2. If bimodal distribution -> OTSU
        3. If heavy-tailed -> MAD
        4. Otherwise -> PERCENTILE (most robust)
        """
        if labels is not None:
            return CalibrationMethod.OPTIMAL_F1, 1.0

        # Check for bimodality
        is_bimodal = ScoreDiagnostics._detect_bimodality(scores)
        if is_bimodal:
            return CalibrationMethod.OTSU, 0.9

        # Check for heavy tails (high kurtosis)
        kurtosis = ScoreDiagnostics._compute_kurtosis(scores)
        if kurtosis > 3.0:  # Heavy-tailed
            return CalibrationMethod.MAD, 0.85

        # Check score spread
        score_range = scores.max() - scores.min()
        if score_range < 0.1:
            # Very narrow range, use IQR-based
            return CalibrationMethod.ADAPTIVE_IQR, 0.8

        # Default to percentile
        return CalibrationMethod.PERCENTILE, 0.9

    def _fixed_threshold(
        self,
        scores: NDArray[np.float64],
        contamination: float,
        fixed_threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        """Use fixed threshold."""
        return fixed_threshold, {"method": "fixed"}

    def _percentile_threshold(
        self,
        scores: NDArray[np.float64],
        contamination: float,
        fixed_threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        """
        Percentile-based threshold.

        Key insight: If contamination=0.05, we want the top 5% to be anomalies,
        so threshold = 95th percentile.
        """
        effective_contamination = np.clip(
            contamination,
            self.min_contamination,
            self.max_contamination,
        )

        percentile = 100 * (1 - effective_contamination)
        threshold = float(np.percentile(scores, percentile))

        return threshold, {
            "method": "percentile",
            "percentile": percentile,
            "contamination": effective_contamination,
        }

    def _otsu_threshold(
        self,
        scores: NDArray[np.float64],
        contamination: float,
        fixed_threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        """
        Otsu's method for bimodal threshold selection.

        Finds the threshold that maximizes between-class variance,
        assuming scores come from two distributions (normal vs anomaly).
        """
        score_min = scores.min()
        score_max = scores.max()

        if score_max - score_min < 1e-10:
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        # Normalize to [0, 255] for histogram
        normalized = ((scores - score_min) / (score_max - score_min) * 255).astype(np.int32)

        # Compute histogram
        hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()

        if total == 0:
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        # Otsu's algorithm
        sum_total = np.dot(np.arange(256), hist)
        sum_b = 0.0
        w_b = 0.0
        max_variance = 0.0
        best_threshold_bin = 0

        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue

            w_f = total - w_b
            if w_f == 0:
                break

            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f

            variance = w_b * w_f * (m_b - m_f) ** 2

            if variance > max_variance:
                max_variance = variance
                best_threshold_bin = t

        # Convert back to original scale
        threshold = score_min + (best_threshold_bin / 255) * (score_max - score_min)

        return threshold, {
            "method": "otsu",
            "between_class_variance": max_variance,
            "threshold_bin": best_threshold_bin,
        }

    def _mad_threshold(
        self,
        scores: NDArray[np.float64],
        contamination: float,
        fixed_threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        """
        Median Absolute Deviation based threshold.

        Robust to outliers, good for heavy-tailed distributions.
        Threshold = median + k * MAD where k controls sensitivity.
        """
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        if mad < 1e-10:
            # MAD is zero, fall back to percentile
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        # Consistency factor for normal distribution
        consistency_factor = 1.4826

        # k = 3 corresponds to ~99.7% for normal distribution
        # Adjust k based on contamination
        k = 3.0 - 2.0 * contamination  # Higher contamination -> lower k
        k = max(k, 1.5)  # Don't go below 1.5

        threshold = median + k * consistency_factor * mad

        # Validate threshold produces reasonable predictions
        pred_ratio = np.mean(scores > threshold)
        if pred_ratio < self.min_contamination:
            # Threshold too high, fall back to percentile
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        return threshold, {
            "method": "mad",
            "median": float(median),
            "mad": float(mad),
            "k": k,
        }

    def _knee_threshold(
        self,
        scores: NDArray[np.float64],
        contamination: float,
        fixed_threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        """
        Knee/elbow detection in sorted scores.

        Finds the point where scores transition from "normal" to "anomalous"
        by detecting the elbow in the sorted score curve.
        """
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)

        if n < 10:
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        # Create normalized curve
        np.arange(n) / (n - 1)
        y = (sorted_scores - sorted_scores.min()) / (
            sorted_scores.max() - sorted_scores.min() + 1e-10
        )

        # Find maximum curvature point
        # Use second derivative of smoothed curve
        window = max(n // 20, 5)
        if window % 2 == 0:
            window += 1

        # Simple moving average smoothing
        kernel = np.ones(window) / window
        y_smooth = np.convolve(y, kernel, mode="valid")

        if len(y_smooth) < 3:
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        # Second derivative
        second_deriv = np.diff(np.diff(y_smooth))

        if len(second_deriv) == 0:
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        # Find knee point (maximum curvature)
        knee_idx = np.argmax(second_deriv)

        # Map back to original indices
        knee_idx_original = knee_idx + window // 2
        knee_idx_original = min(knee_idx_original, n - 1)

        threshold = sorted_scores[knee_idx_original]

        return threshold, {
            "method": "knee",
            "knee_index": knee_idx_original,
            "knee_percentile": 100 * knee_idx_original / n,
        }

    def _adaptive_iqr_threshold(
        self,
        scores: NDArray[np.float64],
        contamination: float,
        fixed_threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        """
        Adaptive IQR-based threshold.

        Uses the score distribution's IQR to estimate contamination
        and set threshold accordingly.
        """
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1

        if iqr < 1e-10:
            # Scores too uniform, use mean + 2*std
            mean = np.mean(scores)
            std = np.std(scores)
            if std < 1e-10:
                return self._percentile_threshold(scores, contamination, fixed_threshold)
            threshold = mean + 2 * std
        else:
            # Standard IQR upper fence
            upper_fence = q3 + 1.5 * iqr

            # Estimate contamination from upper fence
            estimated_contamination = np.mean(scores > upper_fence)

            # Use max of estimated and default contamination
            effective_contamination = max(
                estimated_contamination,
                contamination,
            )
            effective_contamination = np.clip(
                effective_contamination,
                self.min_contamination,
                self.max_contamination,
            )

            # Set threshold at corresponding percentile
            percentile = 100 * (1 - effective_contamination)
            threshold = float(np.percentile(scores, percentile))

        return threshold, {
            "method": "adaptive_iqr",
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
        }

    def _gmm_threshold(
        self,
        scores: NDArray[np.float64],
        contamination: float,
        fixed_threshold: float,
    ) -> tuple[float, dict[str, Any]]:
        """
        Gaussian Mixture Model based threshold.

        Fits a 2-component GMM and uses the intersection point
        as the threshold. Fallback to percentile if GMM fails.
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            logger.warning("sklearn not available for GMM, falling back to percentile")
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        if len(scores) < 20:
            return self._percentile_threshold(scores, contamination, fixed_threshold)

        try:
            # Fit 2-component GMM
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm = GaussianMixture(
                    n_components=2,
                    random_state=42,
                    max_iter=100,
                )
                gmm.fit(scores.reshape(-1, 1))

            # Get means and identify normal vs anomaly component
            means = gmm.means_.flatten()
            if means[0] < means[1]:
                normal_idx, anomaly_idx = 0, 1
            else:
                normal_idx, anomaly_idx = 1, 0

            # Find intersection point between the two Gaussians
            # Use midpoint weighted by component sizes as approximation
            weights = gmm.weights_
            stds = np.sqrt(gmm.covariances_.flatten())

            # Weighted midpoint
            w_normal = weights[normal_idx]
            w_anomaly = weights[anomaly_idx]

            threshold = (w_anomaly * means[normal_idx] + w_normal * means[anomaly_idx]) / (
                w_normal + w_anomaly
            )

            # Validate threshold is between means
            threshold = float(np.clip(threshold, means.min(), means.max()))

            return threshold, {
                "method": "gmm",
                "means": means.tolist(),
                "weights": weights.tolist(),
                "stds": stds.tolist(),
            }

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.warning(
                f"GMM fitting failed ({type(e).__name__}): {e}. "
                "Falling back to percentile method."
            )
            return self._percentile_threshold(scores, contamination, fixed_threshold)
        except Exception as e:
            logger.error(
                f"Unexpected error in GMM fitting: {e}. "
                "This may indicate a bug - please report.",
                exc_info=True,
            )
            return self._percentile_threshold(scores, contamination, fixed_threshold)

    def _optimal_f1_threshold(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32],
    ) -> CalibrationResult:
        """
        Find threshold that maximizes F1 score.

        This is the gold standard when labels are available.
        """
        labels = np.asarray(labels).flatten().astype(np.int32)

        # Try many thresholds
        thresholds = np.percentile(scores, np.linspace(0, 100, 200))

        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds:
            predictions = scores > threshold

            tp = np.sum((labels == 1) & predictions)
            fp = np.sum((labels == 0) & predictions)
            fn = np.sum((labels == 1) & ~predictions)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        predictions = scores > best_threshold
        diagnostics = ScoreDiagnostics.analyze(scores, best_threshold, labels, "optimal_f1")

        return CalibrationResult(
            threshold=best_threshold,
            method=CalibrationMethod.OPTIMAL_F1,
            predictions=predictions,
            diagnostics=diagnostics,
            confidence=1.0,
            method_specific_info={
                "method": "optimal_f1",
                "best_f1": best_f1,
            },
        )

    def _youden_j_threshold(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32],
    ) -> CalibrationResult:
        """
        Find threshold that maximizes Youden's J statistic.

        Youden's J = Sensitivity + Specificity - 1 = TPR - FPR

        This optimizes for the best trade-off between true positive rate
        and false positive rate, independent of class imbalance.

        Args:
            scores: Anomaly scores array
            labels: Binary ground truth labels

        Returns:
            CalibrationResult with optimal threshold
        """
        sorted_scores = np.sort(np.unique(scores))

        # Try threshold candidates at midpoints between unique scores
        thresholds = (sorted_scores[:-1] + sorted_scores[1:]) / 2
        thresholds = np.concatenate(
            [[sorted_scores[0] - 0.01], thresholds, [sorted_scores[-1] + 0.01]]
        )

        best_j = -1.0
        best_threshold = float(np.median(scores))
        best_tpr = 0.0
        best_fpr = 0.0

        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)

        for threshold in thresholds:
            predictions = scores > threshold

            tp = np.sum((labels == 1) & predictions)
            fp = np.sum((labels == 0) & predictions)

            tpr = tp / n_pos if n_pos > 0 else 0.0  # Sensitivity
            fpr = fp / n_neg if n_neg > 0 else 0.0  # 1 - Specificity

            # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
            j = tpr - fpr

            if j > best_j:
                best_j = j
                best_threshold = threshold
                best_tpr = tpr
                best_fpr = fpr

        predictions = scores > best_threshold
        diagnostics = ScoreDiagnostics.analyze(scores, best_threshold, labels, "youden_j")

        return CalibrationResult(
            threshold=best_threshold,
            method=CalibrationMethod.YOUDEN_J,
            predictions=predictions,
            diagnostics=diagnostics,
            confidence=1.0,
            method_specific_info={
                "method": "youden_j",
                "best_youden_j": best_j,
                "tpr_at_threshold": best_tpr,
                "fpr_at_threshold": best_fpr,
            },
        )


class ScoreCalibrationManager:
    """
    Unified calibration manager for anomaly detection scores.

    This is the main interface for solving the F1=0 problem.
    It combines:
    - Threshold optimization (finding the right decision boundary)
    - Probability calibration (making scores well-calibrated)
    - Score diagnostics (understanding score distributions)

    Usage:
        manager = ScoreCalibrationManager(contamination=0.05)

        # Calibrate scores and get predictions
        result = manager.calibrate(scores)
        predictions = result.predictions
        print(result.diagnostics)

        # Or get calibrated threshold for a detector
        threshold = manager.get_calibrated_threshold(scores)
        detector.threshold = threshold
    """

    def __init__(
        self,
        contamination: float = 0.05,
        method: CalibrationMethod = CalibrationMethod.AUTO,
        min_contamination: float = 0.001,
        max_contamination: float = 0.5,
        enable_probability_calibration: bool = False,
    ):
        """
        Initialize calibration manager.

        Args:
            contamination: Expected fraction of anomalies (0.0-1.0)
            method: Default calibration method
            min_contamination: Minimum contamination to enforce
            max_contamination: Maximum contamination to allow
            enable_probability_calibration: Whether to apply probability
                calibration (Platt/Isotonic) after threshold calibration
        """
        self.contamination = contamination
        self.method = method
        self.enable_probability_calibration = enable_probability_calibration

        self._threshold_optimizer = AutoThresholdOptimizer(
            default_contamination=contamination,
            min_contamination=min_contamination,
            max_contamination=max_contamination,
        )

        self._probability_calibrator = None
        self._last_result: CalibrationResult | None = None

    def calibrate(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32] | None = None,
        method: CalibrationMethod | None = None,
        contamination: float | None = None,
    ) -> CalibrationResult:
        """
        Calibrate scores and compute optimal threshold.

        Args:
            scores: Raw anomaly scores
            labels: Optional ground truth labels
            method: Override default calibration method
            contamination: Override default contamination

        Returns:
            CalibrationResult with threshold, predictions, and diagnostics
        """
        method = method or self.method
        contamination = contamination or self.contamination

        result = self._threshold_optimizer.optimize(
            scores=scores,
            method=method,
            contamination=contamination,
            labels=labels,
        )

        # Apply probability calibration if enabled and labels available
        if self.enable_probability_calibration and labels is not None:
            result = self._apply_probability_calibration(result, scores, labels)

        self._last_result = result
        return result

    def get_calibrated_threshold(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32] | None = None,
        method: CalibrationMethod | None = None,
    ) -> float:
        """
        Get calibrated threshold value.

        Convenience method for detectors that just need the threshold.

        Args:
            scores: Raw anomaly scores
            labels: Optional ground truth labels
            method: Override default calibration method

        Returns:
            Calibrated threshold value
        """
        result = self.calibrate(scores, labels, method)
        return result.threshold

    def get_diagnostics(
        self,
        scores: NDArray[np.float64],
        threshold: float | None = None,
        labels: NDArray[np.int32] | None = None,
    ) -> CalibrationDiagnostics:
        """
        Get detailed diagnostics for score distribution.

        Args:
            scores: Anomaly scores
            threshold: Threshold to analyze (if None, uses calibrated)
            labels: Optional ground truth for contamination calculation

        Returns:
            CalibrationDiagnostics with full analysis
        """
        if threshold is None:
            if self._last_result is not None:
                threshold = self._last_result.threshold
            else:
                result = self.calibrate(scores, labels)
                threshold = result.threshold

        return ScoreDiagnostics.analyze(scores, threshold, labels, self.method.value)

    def print_diagnostics(
        self,
        scores: NDArray[np.float64],
        threshold: float | None = None,
        labels: NDArray[np.int32] | None = None,
        detector_name: str = "Unknown",
    ) -> None:
        """
        Print formatted diagnostics to console.

        Args:
            scores: Anomaly scores
            threshold: Threshold to analyze
            labels: Optional ground truth labels
            detector_name: Name for display
        """
        diagnostics = self.get_diagnostics(scores, threshold, labels)
        print(f"\n--- {detector_name} ---")
        print(diagnostics)

    def _apply_probability_calibration(
        self,
        result: CalibrationResult,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32],
    ) -> CalibrationResult:
        """Apply probability calibration (Platt/Isotonic) to improve calibration."""
        try:
            from omni_mercury_engine.core.calibration import CalibrationEnsemble

            if self._probability_calibrator is None:
                self._probability_calibrator = CalibrationEnsemble()  # type: ignore[assignment]

            # Fit and transform - calibrator is guaranteed non-None after above check
            calibrator = self._probability_calibrator
            assert calibrator is not None  # Type guard after assignment
            calibrator.fit(scores, labels)
            calibrated_scores = calibrator.calibrate(scores)

            # Re-calibrate threshold on calibrated scores
            # After Platt/Isotonic, 0.5 threshold is usually appropriate
            predictions = calibrated_scores > 0.5

            # Update diagnostics
            diagnostics = ScoreDiagnostics.analyze(
                calibrated_scores, 0.5, labels, f"{result.method.value}+probability"
            )

            return CalibrationResult(
                threshold=0.5,
                method=result.method,
                predictions=predictions,
                diagnostics=diagnostics,
                confidence=result.confidence,
                method_specific_info={
                    **result.method_specific_info,
                    "probability_calibration": True,
                    "calibrator": calibrator.best_method,
                },
            )

        except ImportError:
            logger.warning("Probability calibration module not available")
            return result


# ============================================================================
# Confidence Interval Calculator
# ============================================================================


class ThresholdConfidenceIntervalCalculator:
    """Bootstrap-based confidence interval calculator for thresholds.

    Provides statistical uncertainty quantification for threshold estimates
    using bootstrap resampling. This helps understand the reliability of
    the calibrated threshold.

    When to use confidence intervals:
    - Production systems where threshold stability matters
    - Comparing thresholds across different datasets
    - Understanding sensitivity to sampling variation
    - Regulatory/compliance requirements for uncertainty quantification

    Example:
        >>> calculator = ThresholdConfidenceIntervalCalculator(n_bootstrap=1000)
        >>> ci = calculator.compute(scores, method=CalibrationMethod.PERCENTILE)
        >>> print(ci)  # "Threshold: 0.42 (95% CI: [0.38, 0.46])"
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = 42,
    ):
        """Initialize confidence interval calculator.

        Args:
            n_bootstrap: Number of bootstrap samples (more = more precise, slower)
            confidence_level: Confidence level for interval (e.g., 0.95 for 95% CI)
            random_state: Random seed for reproducibility
        """
        if n_bootstrap < 100:
            logger.warning(
                f"n_bootstrap={n_bootstrap} is low. Consider using >= 1000 for stable CIs."
            )
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state

    def compute(
        self,
        scores: NDArray[np.float64],
        method: CalibrationMethod = CalibrationMethod.PERCENTILE,
        contamination: float = 0.05,
        labels: NDArray[np.int32] | None = None,
    ) -> ThresholdConfidenceInterval:
        """Compute bootstrap confidence interval for threshold.

        Args:
            scores: Anomaly scores array
            method: Calibration method to use for each bootstrap sample
            contamination: Expected contamination rate
            labels: Optional labels (used for optimal_f1 method)

        Returns:
            ThresholdConfidenceInterval with point estimate and CI bounds
        """
        scores = np.asarray(scores).flatten().astype(np.float64)
        n = len(scores)

        if n < 20:
            logger.warning(f"Only {n} samples - CI may be unreliable. Consider >= 100 samples.")

        # Initialize RNG
        rng = np.random.default_rng(self.random_state)

        # Compute point estimate
        optimizer = AutoThresholdOptimizer(default_contamination=contamination)
        point_result = optimizer.optimize(scores, method=method, labels=labels)
        point_threshold = point_result.threshold

        # Bootstrap resampling
        bootstrap_thresholds = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            # Resample with replacement
            indices = rng.choice(n, size=n, replace=True)
            boot_scores = scores[indices]
            boot_labels = labels[indices] if labels is not None else None

            # Compute threshold for bootstrap sample
            boot_result = optimizer.optimize(boot_scores, method=method, labels=boot_labels)
            bootstrap_thresholds[i] = boot_result.threshold

        # Compute percentile-based CI
        alpha = 1 - self.confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        ci_lower = float(np.percentile(bootstrap_thresholds, lower_pct))
        ci_upper = float(np.percentile(bootstrap_thresholds, upper_pct))
        std_error = float(np.std(bootstrap_thresholds))

        return ThresholdConfidenceInterval(
            threshold=point_threshold,
            lower=ci_lower,
            upper=ci_upper,
            confidence_level=self.confidence_level,
            method="bootstrap_percentile",
            n_bootstrap=self.n_bootstrap,
            std_error=std_error,
        )

    def compute_bca(
        self,
        scores: NDArray[np.float64],
        method: CalibrationMethod = CalibrationMethod.PERCENTILE,
        contamination: float = 0.05,
        labels: NDArray[np.int32] | None = None,
    ) -> ThresholdConfidenceInterval:
        """Compute bias-corrected and accelerated (BCa) bootstrap CI.

        BCa intervals are more accurate than percentile intervals,
        especially for skewed distributions or small samples.

        Args:
            scores: Anomaly scores array
            method: Calibration method to use
            contamination: Expected contamination rate
            labels: Optional labels

        Returns:
            ThresholdConfidenceInterval with BCa bounds
        """
        scores = np.asarray(scores).flatten().astype(np.float64)
        n = len(scores)

        rng = np.random.default_rng(self.random_state)

        # Compute point estimate
        optimizer = AutoThresholdOptimizer(default_contamination=contamination)
        point_result = optimizer.optimize(scores, method=method, labels=labels)
        point_threshold = point_result.threshold

        # Bootstrap resampling
        bootstrap_thresholds = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            boot_scores = scores[indices]
            boot_labels = labels[indices] if labels is not None else None
            boot_result = optimizer.optimize(boot_scores, method=method, labels=boot_labels)
            bootstrap_thresholds[i] = boot_result.threshold

        # Bias correction factor (z0)
        prop_less = np.mean(bootstrap_thresholds < point_threshold)
        # Handle edge cases
        prop_less = np.clip(prop_less, 0.001, 0.999)
        z0 = float(
            np.sqrt(2) * np.erfinv(2 * prop_less - 1)
        )  # Inverse normal CDF  # type: ignore[attr-defined]

        # Acceleration factor (a) using jackknife
        jackknife_thresholds = np.zeros(n)
        for i in range(n):
            jack_indices = np.concatenate([np.arange(i), np.arange(i + 1, n)])
            jack_scores = scores[jack_indices]
            jack_labels = labels[jack_indices] if labels is not None else None
            jack_result = optimizer.optimize(jack_scores, method=method, labels=jack_labels)
            jackknife_thresholds[i] = jack_result.threshold

        theta_bar = np.mean(jackknife_thresholds)
        numerator = np.sum((theta_bar - jackknife_thresholds) ** 3)
        denominator = 6 * (np.sum((theta_bar - jackknife_thresholds) ** 2) ** 1.5)
        a = numerator / (denominator + 1e-10) if denominator != 0 else 0.0

        # Compute BCa percentiles
        alpha = 1 - self.confidence_level
        z_alpha_low = float(np.sqrt(2) * np.erfinv(2 * (alpha / 2) - 1))
        z_alpha_high = float(np.sqrt(2) * np.erfinv(2 * (1 - alpha / 2) - 1))

        # BCa adjusted percentiles
        def bca_percentile(z_alpha: float) -> float:
            numerator = z0 + z_alpha
            adjusted = z0 + numerator / (1 - a * numerator)
            # Convert back to percentile using normal CDF
            return float((1 + np.tanh(adjusted / np.sqrt(2))) / 2 * 100)

        pct_low = bca_percentile(z_alpha_low)
        pct_high = bca_percentile(z_alpha_high)

        # Clip to valid range
        pct_low = np.clip(pct_low, 0.5, 99.5)
        pct_high = np.clip(pct_high, 0.5, 99.5)

        ci_lower = float(np.percentile(bootstrap_thresholds, pct_low))
        ci_upper = float(np.percentile(bootstrap_thresholds, pct_high))
        std_error = float(np.std(bootstrap_thresholds))

        return ThresholdConfidenceInterval(
            threshold=point_threshold,
            lower=ci_lower,
            upper=ci_upper,
            confidence_level=self.confidence_level,
            method="bootstrap_bca",
            n_bootstrap=self.n_bootstrap,
            std_error=std_error,
        )


# ============================================================================
# Label Smoothing Calibrator
# ============================================================================


class LabelSmoothingCalibrator:
    """Label smoothing for improved calibration in anomaly detection.

    Label smoothing softens the target distribution, preventing the model
    from becoming overconfident. Instead of hard targets [0, 1], it uses
    soft targets like [epsilon/2, 1 - epsilon/2].

    Benefits:
    - Reduces overconfidence in predictions
    - Improves expected calibration error (ECE)
    - More robust to label noise
    - Better generalization

    Example:
        >>> calibrator = LabelSmoothingCalibrator(smoothing=0.1)
        >>> smoothed_labels = calibrator.smooth(labels)
        >>> # Use smoothed_labels for training
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        adaptive: bool = False,
        min_smoothing: float = 0.01,
        max_smoothing: float = 0.3,
    ):
        """Initialize label smoothing calibrator.

        Args:
            smoothing: Base smoothing factor (0-1). 0.1 recommended.
            adaptive: Whether to adapt smoothing based on class imbalance
            min_smoothing: Minimum smoothing when adaptive=True
            max_smoothing: Maximum smoothing when adaptive=True
        """
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

        self.smoothing = smoothing
        self.adaptive = adaptive
        self.min_smoothing = min_smoothing
        self.max_smoothing = max_smoothing

    def smooth(
        self,
        labels: NDArray[np.int32],
        class_weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Apply label smoothing to binary labels.

        Args:
            labels: Binary labels array (0 or 1)
            class_weights: Optional per-class smoothing weights

        Returns:
            Smoothed labels as float array
        """
        labels = np.asarray(labels).flatten().astype(np.float64)

        # Determine effective smoothing
        if self.adaptive:
            # Adapt smoothing based on class imbalance
            pos_ratio = np.mean(labels)
            # More smoothing for minority class
            imbalance = abs(0.5 - pos_ratio) * 2  # 0 to 1
            effective_smoothing = self.min_smoothing + imbalance * (
                self.max_smoothing - self.min_smoothing
            )
        else:
            effective_smoothing = self.smoothing

        # Apply smoothing: y_smooth = y * (1 - eps) + eps/2
        smoothed = labels * (1 - effective_smoothing) + effective_smoothing / 2

        return smoothed

    def smooth_with_confidence(
        self,
        labels: NDArray[np.int32],
        confidences: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply confidence-weighted label smoothing.

        Higher confidence samples get less smoothing, lower confidence
        samples get more smoothing. This is useful when label quality varies.

        Args:
            labels: Binary labels array
            confidences: Per-sample confidence scores (0-1)

        Returns:
            Smoothed labels with confidence-adjusted smoothing
        """
        labels = np.asarray(labels).flatten().astype(np.float64)
        confidences = np.asarray(confidences).flatten()
        confidences = np.clip(confidences, 0.0, 1.0)

        # Less confident samples get more smoothing
        sample_smoothing = self.smoothing * (1 - confidences)

        # Apply per-sample smoothing
        smoothed = labels * (1 - sample_smoothing) + sample_smoothing / 2

        return smoothed

    def get_calibration_targets(
        self,
        labels: NDArray[np.int32],
        predictions: NDArray[np.float64],
        temperature: float = 1.0,
    ) -> NDArray[np.float64]:
        """Get calibration targets using temperature scaling and smoothing.

        Combines label smoothing with temperature scaling for optimal calibration.

        Args:
            labels: Ground truth binary labels
            predictions: Model predictions (probabilities)
            temperature: Temperature for scaling (>1 softens, <1 sharpens)

        Returns:
            Calibration targets combining smoothing and temperature scaling
        """
        labels = np.asarray(labels).flatten().astype(np.float64)
        predictions = np.asarray(predictions).flatten()

        # Apply label smoothing
        smoothed = self.smooth(labels)

        # Temperature-scaled predictions
        # For binary case: p_scaled = 1 / (1 + exp(-logit/T))
        # where logit = log(p / (1-p))
        eps = 1e-7
        predictions = np.clip(predictions, eps, 1 - eps)
        logits = np.log(predictions / (1 - predictions))
        scaled_preds = 1 / (1 + np.exp(-logits / temperature))

        # Blend smoothed labels with scaled predictions
        # Use a soft target that's between hard label and prediction
        blend_weight = 0.8  # Mostly trust the smoothed label
        targets = blend_weight * smoothed + (1 - blend_weight) * scaled_preds

        return targets


# ============================================================================
# Convenience Functions
# ============================================================================


def compute_threshold_confidence_interval(
    scores: NDArray[np.float64],
    method: CalibrationMethod | str = "percentile",
    contamination: float = 0.05,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    labels: NDArray[np.int32] | None = None,
) -> ThresholdConfidenceInterval:
    """Compute confidence interval for a calibrated threshold.

    Args:
        scores: Anomaly scores array
        method: Calibration method
        contamination: Expected contamination rate
        confidence_level: Confidence level (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples
        labels: Optional ground truth labels

    Returns:
        ThresholdConfidenceInterval with bounds

    Example:
        >>> ci = compute_threshold_confidence_interval(
        ...     scores, method="percentile", confidence_level=0.95
        ... )
        >>> print(f"Threshold: {ci.threshold:.3f} ({ci.confidence_level*100:.0f}% CI: "
        ...       f"[{ci.lower:.3f}, {ci.upper:.3f}])")
    """
    if isinstance(method, str):
        method = CalibrationMethod(method.lower())

    calculator = ThresholdConfidenceIntervalCalculator(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )

    return calculator.compute(scores, method=method, contamination=contamination, labels=labels)


def calibrate_scores(
    scores: NDArray[np.float64],
    contamination: float = 0.05,
    method: str | CalibrationMethod = "auto",
    labels: NDArray[np.int32] | None = None,
) -> tuple[float, NDArray[np.bool_], CalibrationDiagnostics]:
    """
    Convenience function to calibrate scores and get threshold.

    Args:
        scores: Anomaly scores array
        contamination: Expected anomaly ratio
        method: Calibration method ("auto", "percentile", "otsu", etc.)
        labels: Optional ground truth labels

    Returns:
        Tuple of (threshold, predictions, diagnostics)

    Example:
        threshold, predictions, diagnostics = calibrate_scores(
            scores, contamination=0.05
        )
        print(diagnostics)
    """
    if isinstance(method, str):
        method = CalibrationMethod(method.lower())

    manager = ScoreCalibrationManager(
        contamination=contamination,
        method=method,
    )

    result = manager.calibrate(scores, labels)
    return result.threshold, result.predictions, result.diagnostics


def diagnose_scores(
    scores: NDArray[np.float64],
    threshold: float = 0.5,
    labels: NDArray[np.int32] | None = None,
    print_output: bool = True,
) -> CalibrationDiagnostics:
    """
    Diagnose score distribution and threshold issues.

    This implements the exact diagnostic the user requested:

    Args:
        scores: Anomaly scores array
        threshold: Current threshold value
        labels: Optional ground truth labels
        print_output: Whether to print diagnostics

    Returns:
        CalibrationDiagnostics object

    Example:
        # In the benchmark, add this after detection:
        diagnostics = diagnose_scores(
            result["scores"],
            detector.threshold,
            y_true,
            print_output=True
        )
    """
    diagnostics = ScoreDiagnostics.analyze(scores, threshold, labels, "diagnosis")

    if print_output:
        print(diagnostics)

        # Print actionable recommendations
        if diagnostics.predicted_anomaly_ratio == 0:
            print("\nDIAGNOSIS: All predictions are NEGATIVE (no anomalies detected)")
            print(f"CAUSE: Threshold ({threshold:.4f}) is higher than all scores")
            print(f"       Score max: {diagnostics.score_max:.4f}")
            print("\nRECOMMENDED ACTIONS:")
            print("1. Use percentile-based threshold: threshold = percentile(scores, 95)")
            print("2. Use adaptive calibration: calibrate_scores(scores, method='auto')")
            print("3. Check if scores need normalization to [0, 1]")

    return diagnostics


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "AutoThresholdOptimizer",
    "CalibrationDiagnostics",
    "CalibrationMethod",
    "CalibrationResult",
    "LabelSmoothingCalibrator",
    "ScoreCalibrationManager",
    "ScoreDiagnostics",
    "ThresholdConfidenceInterval",
    "ThresholdConfidenceIntervalCalculator",
    "calibrate_scores",
    "compute_threshold_confidence_interval",
    "diagnose_scores",
]
