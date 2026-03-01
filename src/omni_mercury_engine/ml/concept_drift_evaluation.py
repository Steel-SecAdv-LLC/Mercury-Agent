"""
Mercury Agent - Concept Drift Evaluation Framework
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Comprehensive concept drift evaluation framework providing:
- Temporal split strategies (expanding window, sliding window, fixed)
- Performance degradation measurement over time
- Statistical drift detection with multiple methods
- Automatic retraining trigger mechanisms
- Cross-validation with temporal ordering preservation
- Degradation curve analysis and forecasting

This addresses the critical gap: Mercury can demonstrate architectural
advantages over pure supervised methods through drift-aware evaluation.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from scipy import stats

from omni_mercury_engine.ml.drift import (
    DriftResult,
    DriftSeverity,
    DriftType,
    EnsembleDriftDetector,
    KolmogorovSmirnovDriftDetector,
    PopulationStabilityIndexDetector,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _warn_deprecated_module() -> None:
    import warnings

    warnings.warn(
        f"{__name__} previously required sklearn. "
        "Now uses Mercury-native implementations. "
        "Module will be removed in a future release — see issue #NNN.",
        DeprecationWarning,
        stacklevel=3,
    )


class TemporalSplitStrategy(StrEnum):
    """Strategies for temporal train/test splitting."""

    EXPANDING_WINDOW = "expanding_window"  # Train on all past data
    SLIDING_WINDOW = "sliding_window"  # Fixed-size training window
    FIXED_ORIGIN = "fixed_origin"  # Train once, test on future periods
    WALK_FORWARD = "walk_forward"  # Retrain at each step
    BLOCKED = "blocked"  # Non-overlapping blocks


class DegradationTrend(StrEnum):
    """Types of performance degradation trends."""

    STABLE = "stable"  # No significant degradation
    LINEAR_DECLINE = "linear_decline"  # Steady performance decrease
    EXPONENTIAL_DECAY = "exponential_decay"  # Accelerating decline
    SUDDEN_SHIFT = "sudden_shift"  # Abrupt performance drop
    OSCILLATING = "oscillating"  # Performance varies cyclically
    RECOVERING = "recovering"  # Performance improving after decline


@dataclass
class TemporalSplit:
    """A single temporal train/test split."""

    split_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    timestamp: float | None = None

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass
class SplitPerformance:
    """Performance metrics for a single temporal split."""

    split: TemporalSplit
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None
    train_time: float
    inference_time: float

    # Drift indicators
    data_drift_detected: bool = False
    drift_severity: DriftSeverity = DriftSeverity.NONE
    drift_score: float = 0.0

    # Additional metrics
    predictions: np.ndarray | None = None
    true_labels: np.ndarray | None = None
    feature_importance: dict[str, float] | None = None


@dataclass
class DegradationAnalysis:
    """Analysis of performance degradation over time."""

    trend: DegradationTrend
    degradation_rate: float  # Performance loss per time unit
    half_life: float | None  # Time for performance to halve (for exponential)
    inflection_points: list[int]  # Indices where trend changes
    stability_score: float  # 0-1, higher = more stable
    forecast: list[float]  # Predicted future performance
    retraining_recommended: bool
    retraining_urgency: str  # 'none', 'low', 'medium', 'high', 'critical'

    # Statistical measures
    trend_slope: float
    trend_p_value: float
    variance: float
    coefficient_of_variation: float


@dataclass
class ConceptDriftEvaluationResult:
    """Complete result from concept drift evaluation."""

    # Overall metrics
    mean_accuracy: float
    mean_f1: float
    mean_auc: float | None

    # Per-split results
    split_performances: list[SplitPerformance]

    # Degradation analysis
    degradation_analysis: DegradationAnalysis

    # Drift detection
    total_drifts_detected: int
    drift_timeline: list[tuple[int, DriftSeverity]]

    # Timing
    total_evaluation_time: float

    # Configuration
    strategy: TemporalSplitStrategy
    n_splits: int

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean_accuracy": self.mean_accuracy,
            "mean_f1": self.mean_f1,
            "mean_auc": self.mean_auc,
            "n_splits": self.n_splits,
            "strategy": self.strategy.value,
            "degradation": {
                "trend": self.degradation_analysis.trend.value,
                "rate": self.degradation_analysis.degradation_rate,
                "stability_score": self.degradation_analysis.stability_score,
                "retraining_recommended": self.degradation_analysis.retraining_recommended,
                "retraining_urgency": self.degradation_analysis.retraining_urgency,
            },
            "drift": {
                "total_detected": self.total_drifts_detected,
                "timeline": [{"split": s, "severity": sev.value} for s, sev in self.drift_timeline],
            },
            "total_time_seconds": self.total_evaluation_time,
            "metadata": self.metadata,
        }


class ModelProtocol(Protocol):
    """Protocol for models compatible with concept drift evaluation."""

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> Any: ...
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]: ...
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


class TemporalSplitter:
    """
    Generates temporal train/test splits preserving time ordering.

    Unlike random cross-validation, temporal splitting ensures:
    1. Training data always precedes test data
    2. No future information leakage
    3. Realistic evaluation of time-series models
    """

    def __init__(
        self,
        n_splits: int = 5,
        strategy: TemporalSplitStrategy = TemporalSplitStrategy.EXPANDING_WINDOW,
        train_ratio: float = 0.7,
        gap: int = 0,
        min_train_size: int = 100,
        window_size: int | None = None,
    ):
        """
        Initialize temporal splitter.

        Args:
            n_splits: Number of train/test splits to generate
            strategy: Splitting strategy
            train_ratio: Fraction of data for training (for fixed strategies)
            gap: Number of samples to skip between train and test (prevent leakage)
            min_train_size: Minimum training set size
            window_size: Size of sliding window (for SLIDING_WINDOW strategy)
        """
        self.n_splits = n_splits
        self.strategy = strategy
        self.train_ratio = train_ratio
        self.gap = gap
        self.min_train_size = min_train_size
        self.window_size = window_size

    def split(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64] | None = None,
        timestamps: NDArray[np.float64] | None = None,
    ) -> list[TemporalSplit]:
        """
        Generate temporal splits.

        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels (optional, for stratification)
            timestamps: Timestamps for samples (optional)

        Returns:
            List of TemporalSplit objects
        """
        n_samples = len(X)

        if self.strategy == TemporalSplitStrategy.EXPANDING_WINDOW:
            return self._expanding_window_split(n_samples, timestamps)
        elif self.strategy == TemporalSplitStrategy.SLIDING_WINDOW:
            return self._sliding_window_split(n_samples, timestamps)
        elif self.strategy == TemporalSplitStrategy.FIXED_ORIGIN:
            return self._fixed_origin_split(n_samples, timestamps)
        elif self.strategy == TemporalSplitStrategy.WALK_FORWARD:
            return self._walk_forward_split(n_samples, timestamps)
        elif self.strategy == TemporalSplitStrategy.BLOCKED:
            return self._blocked_split(n_samples, timestamps)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _expanding_window_split(
        self,
        n_samples: int,
        timestamps: NDArray[np.float64] | None,
    ) -> list[TemporalSplit]:
        """Expanding window: train on all past data."""
        splits = []

        # Reserve initial training period
        initial_train_end = max(self.min_train_size, int(n_samples * 0.3))

        # Calculate test block size
        remaining = n_samples - initial_train_end - self.gap
        test_block_size = max(1, remaining // self.n_splits)

        for i in range(self.n_splits):
            train_start = 0
            train_end = initial_train_end + i * test_block_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_block_size, n_samples)

            if test_start >= n_samples:
                break

            splits.append(
                TemporalSplit(
                    split_index=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    timestamp=timestamps[test_start] if timestamps is not None else None,
                )
            )

        return splits

    def _sliding_window_split(
        self,
        n_samples: int,
        timestamps: NDArray[np.float64] | None,
    ) -> list[TemporalSplit]:
        """Sliding window: fixed-size training window."""
        splits = []
        window_size = self.window_size or max(self.min_train_size, int(n_samples * 0.3))

        # Calculate stride
        available = n_samples - window_size - self.gap
        stride = max(1, available // self.n_splits)

        for i in range(self.n_splits):
            train_start = i * stride
            train_end = train_start + window_size
            test_start = train_end + self.gap
            test_end = min(test_start + stride, n_samples)

            if test_start >= n_samples:
                break

            splits.append(
                TemporalSplit(
                    split_index=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    timestamp=timestamps[test_start] if timestamps is not None else None,
                )
            )

        return splits

    def _fixed_origin_split(
        self,
        n_samples: int,
        timestamps: NDArray[np.float64] | None,
    ) -> list[TemporalSplit]:
        """Fixed origin: train once, test on multiple future periods."""
        splits = []

        train_end = max(self.min_train_size, int(n_samples * self.train_ratio))
        test_available = n_samples - train_end - self.gap
        test_block_size = max(1, test_available // self.n_splits)

        for i in range(self.n_splits):
            test_start = train_end + self.gap + i * test_block_size
            test_end = min(test_start + test_block_size, n_samples)

            if test_start >= n_samples:
                break

            splits.append(
                TemporalSplit(
                    split_index=i,
                    train_start=0,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    timestamp=timestamps[test_start] if timestamps is not None else None,
                )
            )

        return splits

    def _walk_forward_split(
        self,
        n_samples: int,
        timestamps: NDArray[np.float64] | None,
    ) -> list[TemporalSplit]:
        """Walk-forward: retrain at each step with new data."""
        splits = []

        initial_train = max(self.min_train_size, int(n_samples * 0.3))
        step_size = max(1, (n_samples - initial_train - self.gap) // self.n_splits)

        for i in range(self.n_splits):
            train_start = 0
            train_end = initial_train + i * step_size
            test_start = train_end + self.gap
            test_end = min(test_start + step_size, n_samples)

            if test_start >= n_samples:
                break

            splits.append(
                TemporalSplit(
                    split_index=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    timestamp=timestamps[test_start] if timestamps is not None else None,
                )
            )

        return splits

    def _blocked_split(
        self,
        n_samples: int,
        timestamps: NDArray[np.float64] | None,
    ) -> list[TemporalSplit]:
        """Blocked: non-overlapping train/test blocks."""
        splits = []

        # Divide data into (n_splits + 1) blocks
        total_blocks = self.n_splits + 1
        block_size = n_samples // total_blocks

        for i in range(self.n_splits):
            # Each split uses blocks 0..i for training, block i+1 for testing
            train_start = 0
            train_end = (i + 1) * block_size
            test_start = train_end + self.gap
            test_end = min(test_start + block_size, n_samples)

            if test_start >= n_samples:
                break

            splits.append(
                TemporalSplit(
                    split_index=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    timestamp=timestamps[test_start] if timestamps is not None else None,
                )
            )

        return splits


class DegradationAnalyzer:
    """
    Analyzes performance degradation patterns over time.

    Uses statistical methods to identify:
    - Trend type (linear, exponential, sudden shift, etc.)
    - Degradation rate
    - Inflection points
    - Retraining triggers
    """

    def __init__(
        self,
        min_samples: int = 3,
        significance_level: float = 0.05,
        degradation_threshold: float = 0.05,
        forecast_horizon: int = 3,
    ):
        """
        Initialize degradation analyzer.

        Args:
            min_samples: Minimum samples for analysis
            significance_level: P-value threshold for trend significance
            degradation_threshold: Minimum decline to trigger retraining
            forecast_horizon: Number of future periods to forecast
        """
        self.min_samples = min_samples
        self.significance_level = significance_level
        self.degradation_threshold = degradation_threshold
        self.forecast_horizon = forecast_horizon

    def analyze(self, performances: list[float]) -> DegradationAnalysis:
        """
        Analyze performance degradation over time.

        Args:
            performances: List of performance metrics over time (e.g., F1 scores)

        Returns:
            DegradationAnalysis with trend, rate, and recommendations
        """
        n = len(performances)

        if n < self.min_samples:
            return self._insufficient_data_result(performances)

        perf_array = np.array(performances)
        time_index = np.arange(n)

        # Compute basic statistics
        variance = float(np.var(perf_array))
        mean = float(np.mean(perf_array))
        cv = float(np.std(perf_array) / mean) if mean > 0 else 0.0

        # Linear trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, perf_array)

        # Detect trend type
        trend = self._detect_trend_type(perf_array, slope, p_value, variance)

        # Find inflection points
        inflection_points = self._find_inflection_points(perf_array)

        # Calculate degradation rate and half-life
        degradation_rate = -slope if slope < 0 else 0.0
        half_life = self._calculate_half_life(perf_array, trend)

        # Calculate stability score
        stability_score = self._calculate_stability_score(perf_array, variance, cv)

        # Forecast future performance
        forecast = self._forecast_performance(perf_array, trend, slope)

        # Determine retraining recommendation
        retraining_recommended, urgency = self._recommend_retraining(
            trend, degradation_rate, stability_score, perf_array
        )

        return DegradationAnalysis(
            trend=trend,
            degradation_rate=degradation_rate,
            half_life=half_life,
            inflection_points=inflection_points,
            stability_score=stability_score,
            forecast=forecast,
            retraining_recommended=retraining_recommended,
            retraining_urgency=urgency,
            trend_slope=float(slope),
            trend_p_value=float(p_value),
            variance=variance,
            coefficient_of_variation=cv,
        )

    def _detect_trend_type(
        self,
        performances: NDArray[np.float64],
        slope: float,
        p_value: float,
        variance: float,
    ) -> DegradationTrend:
        """Detect the type of performance trend."""
        _ = len(performances)  # Keep for potential future use

        # Check for significant trend
        if p_value > self.significance_level:
            # No significant linear trend
            if variance < 0.01:
                return DegradationTrend.STABLE
            else:
                # Check for oscillation
                if self._is_oscillating(performances):
                    return DegradationTrend.OSCILLATING
                return DegradationTrend.STABLE

        # Significant trend detected
        if slope < 0:
            # Declining performance
            # Check if exponential decay fits better
            if self._is_exponential_decay(performances):
                return DegradationTrend.EXPONENTIAL_DECAY

            # Check for sudden shift
            if self._has_sudden_shift(performances):
                return DegradationTrend.SUDDEN_SHIFT

            return DegradationTrend.LINEAR_DECLINE

        else:
            # Improving performance
            return DegradationTrend.RECOVERING

    def _is_exponential_decay(self, performances: NDArray[np.float64]) -> bool:
        """Check if performance follows exponential decay."""
        n = len(performances)
        if n < 4:
            return False

        # Log-transform and fit linear
        # Avoid log(0) by adding small epsilon
        log_perf = np.log(performances + 1e-10)
        time_index = np.arange(n)

        # Linear fit to log-transformed data
        log_slope, _, log_r, _, _ = stats.linregress(time_index, log_perf)

        # Linear fit to original data
        _, _, lin_r, _, _ = stats.linregress(time_index, performances)

        # Exponential fits better if log-linear has higher R^2
        return bool(abs(log_r) > abs(lin_r) + 0.1 and log_slope < 0)

    def _has_sudden_shift(self, performances: NDArray[np.float64]) -> bool:
        """Check for sudden performance shift."""
        n = len(performances)
        if n < 4:
            return False

        # Calculate differences
        diffs = np.diff(performances)
        std_diff = np.std(diffs)
        mean_diff = np.mean(diffs)

        # Check for outlier differences (sudden shifts)
        threshold = abs(mean_diff) + 3 * std_diff
        return bool(np.any(np.abs(diffs) > threshold))  # type: ignore[return-value, unused-ignore]

    def _is_oscillating(self, performances: NDArray[np.float64]) -> bool:
        """Check for oscillating pattern."""
        n = len(performances)
        if n < 5:
            return False

        # Count sign changes in differences
        diffs = np.diff(performances)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) == 2)

        # Oscillating if many sign changes relative to length
        return sign_changes > n * 0.4

    def _find_inflection_points(self, performances: NDArray[np.float64]) -> list[int]:
        """Find points where trend changes direction."""
        n = len(performances)
        if n < 4:
            return []

        inflection_points = []

        # Calculate second derivative (discrete)
        first_diff = np.diff(performances)
        second_diff = np.diff(first_diff)

        # Find sign changes in second derivative
        for i in range(len(second_diff) - 1):
            if second_diff[i] * second_diff[i + 1] < 0:
                inflection_points.append(i + 1)

        return inflection_points

    def _calculate_half_life(
        self,
        performances: NDArray[np.float64],
        trend: DegradationTrend,
    ) -> float | None:
        """Calculate time for performance to halve (exponential decay)."""
        if trend != DegradationTrend.EXPONENTIAL_DECAY:
            return None

        n = len(performances)
        if n < 3:
            return None

        # Fit exponential: y = a * exp(-b * t)
        log_perf = np.log(performances + 1e-10)
        time_index = np.arange(n)

        slope, _, _, _, _ = stats.linregress(time_index, log_perf)

        if slope >= 0:
            return None

        # Half-life = ln(2) / |decay_rate|
        half_life = np.log(2) / abs(slope)
        return float(half_life)

    def _calculate_stability_score(
        self,
        performances: NDArray[np.float64],
        variance: float,
        cv: float,
    ) -> float:
        """Calculate stability score (0-1, higher = more stable)."""
        n = len(performances)

        # Components of stability:
        # 1. Low variance
        variance_score = np.exp(-10 * variance)  # Decays with variance

        # 2. Low coefficient of variation
        cv_score = np.exp(-5 * cv)

        # 3. Maintained minimum performance
        min_perf = np.min(performances)
        mean_perf = np.mean(performances)
        min_ratio = min_perf / (mean_perf + 1e-10)
        min_score = max(0, min_ratio)

        # 4. Trend stability (small absolute slope)
        time_index = np.arange(n)
        slope, _, _, _, _ = stats.linregress(time_index, performances)
        trend_score = np.exp(-10 * abs(slope))

        # Weighted combination
        stability = 0.25 * variance_score + 0.25 * cv_score + 0.25 * min_score + 0.25 * trend_score

        return float(np.clip(stability, 0, 1))

    def _forecast_performance(
        self,
        performances: NDArray[np.float64],
        trend: DegradationTrend,
        slope: float,
    ) -> list[float]:
        """Forecast future performance."""
        n = len(performances)
        forecast = []

        for i in range(1, self.forecast_horizon + 1):
            future_time = n + i

            if trend == DegradationTrend.STABLE:
                # Predict mean
                pred = float(np.mean(performances))

            elif trend == DegradationTrend.LINEAR_DECLINE:
                # Linear extrapolation
                intercept = np.mean(performances) - slope * (n - 1) / 2
                pred = intercept + slope * future_time  # type: ignore[assignment, unused-ignore]
                pred = max(0, pred)  # Clip to non-negative

            elif trend == DegradationTrend.EXPONENTIAL_DECAY:
                # Exponential extrapolation
                log_perf = np.log(performances + 1e-10)
                log_slope, log_intercept, _, _, _ = stats.linregress(np.arange(n), log_perf)
                pred = np.exp(log_intercept + log_slope * future_time)
                pred = max(0, min(1, pred))

            elif trend == DegradationTrend.RECOVERING:
                # Optimistic linear extrapolation with ceiling
                intercept = np.mean(performances) - slope * (n - 1) / 2
                pred = intercept + slope * future_time  # type: ignore[assignment, unused-ignore]
                pred = min(1, pred)  # Clip to maximum 1

            else:
                # Default to last value
                pred = float(performances[-1])

            forecast.append(float(pred))

        return forecast

    def _recommend_retraining(
        self,
        trend: DegradationTrend,
        degradation_rate: float,
        stability_score: float,
        performances: NDArray[np.float64],
    ) -> tuple[bool, str]:
        """Determine if retraining is recommended and urgency level."""
        latest_perf = performances[-1]
        initial_perf = performances[0]
        perf_drop = initial_perf - latest_perf

        # Immediate retraining triggers
        if trend == DegradationTrend.SUDDEN_SHIFT:
            return True, "critical"

        if trend == DegradationTrend.EXPONENTIAL_DECAY:
            return True, "high"

        if perf_drop > self.degradation_threshold * 2:
            return True, "high"

        if trend == DegradationTrend.LINEAR_DECLINE:
            if perf_drop > self.degradation_threshold:
                return True, "medium"
            else:
                return True, "low"

        if stability_score < 0.3:
            return True, "medium"

        if trend == DegradationTrend.STABLE:
            return False, "none"

        if trend == DegradationTrend.RECOVERING:
            return False, "none"

        # Oscillating - may benefit from retraining
        if trend == DegradationTrend.OSCILLATING:
            return True, "low"

        return False, "none"

    def _insufficient_data_result(self, performances: list[float]) -> DegradationAnalysis:
        """Return result when insufficient data for analysis."""
        mean_perf = np.mean(performances) if performances else 0.0

        return DegradationAnalysis(
            trend=DegradationTrend.STABLE,
            degradation_rate=0.0,
            half_life=None,
            inflection_points=[],
            stability_score=1.0,
            forecast=[float(mean_perf)] * self.forecast_horizon,
            retraining_recommended=False,
            retraining_urgency="none",
            trend_slope=0.0,
            trend_p_value=1.0,
            variance=0.0,
            coefficient_of_variation=0.0,
        )


class ConceptDriftEvaluator:
    """
    Comprehensive concept drift evaluation framework.

    Integrates:
    - Temporal splitting strategies
    - Multi-metric performance tracking
    - Statistical drift detection
    - Degradation analysis
    - Automatic retraining triggers

    Usage:
        evaluator = ConceptDriftEvaluator(
            n_splits=10,
            strategy=TemporalSplitStrategy.EXPANDING_WINDOW,
            detect_drift=True,
        )
        result = evaluator.evaluate(model, X, y, timestamps)
        print(result.degradation_analysis.retraining_recommended)
    """

    def __init__(
        self,
        n_splits: int = 5,
        strategy: TemporalSplitStrategy = TemporalSplitStrategy.EXPANDING_WINDOW,
        detect_drift: bool = True,
        drift_detector: str = "ensemble",
        metric: str = "f1",
        gap: int = 0,
        min_train_size: int = 100,
        window_size: int | None = None,
        retrain_on_drift: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize concept drift evaluator.

        Args:
            n_splits: Number of temporal splits
            strategy: Temporal splitting strategy
            detect_drift: Whether to detect data/concept drift
            drift_detector: Drift detector type ('ks', 'psi', 'ensemble')
            metric: Primary metric for degradation analysis ('f1', 'accuracy', 'auc')
            gap: Gap between train and test to prevent leakage
            min_train_size: Minimum training set size
            window_size: Window size for sliding window strategy
            retrain_on_drift: Whether to retrain when drift detected
            verbose: Print progress information
        """
        _warn_deprecated_module()
        self.n_splits = n_splits
        self.strategy = strategy
        self.detect_drift = detect_drift
        self.drift_detector_type = drift_detector
        self.metric = metric
        self.gap = gap
        self.min_train_size = min_train_size
        self.window_size = window_size
        self.retrain_on_drift = retrain_on_drift
        self.verbose = verbose

        # Initialize components
        self.splitter = TemporalSplitter(
            n_splits=n_splits,
            strategy=strategy,
            gap=gap,
            min_train_size=min_train_size,
            window_size=window_size,
        )
        self.degradation_analyzer = DegradationAnalyzer()

        # Drift detectors
        self._drift_detectors: dict[str, Any] = {}

    def evaluate(
        self,
        model: Any,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        timestamps: NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        clone_model: bool = True,
    ) -> ConceptDriftEvaluationResult:
        """
        Evaluate model under concept drift conditions.

        Args:
            model: Model implementing fit/predict/predict_proba
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
            timestamps: Sample timestamps (optional)
            feature_names: Feature names for drift reporting
            clone_model: Whether to clone model for each split

        Returns:
            ConceptDriftEvaluationResult with comprehensive metrics
        """
        start_time = time.time()

        # Generate temporal splits
        splits = self.splitter.split(X, y, timestamps)

        if self.verbose:
            logger.info(f"Evaluating {len(splits)} temporal splits using {self.strategy.value}")

        # Initialize drift detector
        if self.detect_drift:
            self._initialize_drift_detector(X[: self.min_train_size])

        # Evaluate each split
        split_performances = []
        drift_timeline = []
        current_model = model

        for split in splits:
            if self.verbose:
                logger.info(
                    f"Split {split.split_index + 1}/{len(splits)}: "
                    f"train[{split.train_start}:{split.train_end}] -> "
                    f"test[{split.test_start}:{split.test_end}]"
                )

            # Get data for this split
            X_train = X[split.train_start : split.train_end]
            y_train = y[split.train_start : split.train_end]
            X_test = X[split.test_start : split.test_end]
            y_test = y[split.test_start : split.test_end]

            # Check for drift
            drift_result = None
            if self.detect_drift:
                drift_result = self._detect_drift(X_train, X_test, feature_names)
                if drift_result.is_drift:
                    drift_timeline.append((split.split_index, drift_result.severity))

            # Clone or reuse model
            if clone_model:
                try:
                    from omni_mercury_engine.ml._native_utils import native_clone

                    current_model = native_clone(model)
                except (ImportError, TypeError):
                    current_model = model

            # Train
            train_start = time.time()
            try:
                current_model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Training failed for split {split.split_index}: {e}")
                continue
            train_time = time.time() - train_start

            # Predict
            inference_start = time.time()
            try:
                y_pred = current_model.predict(X_test)

                # Get probabilities if available
                y_proba = None
                if hasattr(current_model, "predict_proba"):
                    try:
                        proba = current_model.predict_proba(X_test)
                        y_proba = proba[:, 1] if proba.ndim > 1 else proba
                    except Exception as e:
                        logger.debug(
                            f"Failed to get prediction probabilities for split {split.split_index}: {e}"
                        )
            except Exception as e:
                logger.warning(f"Prediction failed for split {split.split_index}: {e}")
                continue
            inference_time = time.time() - inference_start

            # Calculate metrics
            from omni_mercury_engine.ml._native_utils import (
                native_accuracy_score as accuracy_score,
                native_f1_score as f1_score,
                native_precision_score as precision_score,
                native_recall_score as recall_score,
                native_roc_auc_score as roc_auc_score,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                auc = None
                if y_proba is not None and len(np.unique(y_test)) > 1:
                    try:
                        auc = roc_auc_score(y_test, y_proba)
                    except ValueError as e:
                        logger.debug(f"ROC-AUC computation failed: {e}")

            # Create performance record
            perf = SplitPerformance(
                split=split,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                auc_roc=auc,
                train_time=train_time,
                inference_time=inference_time,
                data_drift_detected=drift_result.is_drift if drift_result else False,
                drift_severity=drift_result.severity if drift_result else DriftSeverity.NONE,
                drift_score=drift_result.test_statistic if drift_result else 0.0,
                predictions=y_pred,
                true_labels=y_test,
            )

            split_performances.append(perf)

        # Analyze degradation
        metric_values = self._extract_metric_values(split_performances)
        degradation_analysis = self.degradation_analyzer.analyze(metric_values)

        # Calculate summary statistics
        mean_accuracy = np.mean([p.accuracy for p in split_performances])
        mean_f1 = np.mean([p.f1 for p in split_performances])
        aucs = [p.auc_roc for p in split_performances if p.auc_roc is not None]
        mean_auc = np.mean(aucs) if aucs else None

        total_time = time.time() - start_time

        return ConceptDriftEvaluationResult(
            mean_accuracy=float(mean_accuracy),
            mean_f1=float(mean_f1),
            mean_auc=float(mean_auc) if mean_auc is not None else None,
            split_performances=split_performances,
            degradation_analysis=degradation_analysis,
            total_drifts_detected=len(drift_timeline),
            drift_timeline=drift_timeline,
            total_evaluation_time=total_time,
            strategy=self.strategy,
            n_splits=len(splits),
            metadata={
                "primary_metric": self.metric,
                "drift_detector": self.drift_detector_type,
                "gap": self.gap,
            },
        )

    def _initialize_drift_detector(self, reference_data: NDArray[np.float64]) -> None:
        """Initialize drift detector with reference data."""
        detector: (
            KolmogorovSmirnovDriftDetector
            | PopulationStabilityIndexDetector
            | EnsembleDriftDetector
        )
        if self.drift_detector_type == "ks":
            detector = KolmogorovSmirnovDriftDetector()
        elif self.drift_detector_type == "psi":
            detector = PopulationStabilityIndexDetector()
        elif self.drift_detector_type == "ensemble":
            detector = EnsembleDriftDetector()
        else:
            raise ValueError(f"Unknown drift detector: {self.drift_detector_type}")

        detector.fit(reference_data)
        self._drift_detectors["main"] = detector

    def _detect_drift(
        self,
        X_train: NDArray[np.float64],
        X_test: NDArray[np.float64],
        feature_names: list[str] | None,
    ) -> DriftResult:
        """Detect drift between training and test data."""
        if "main" not in self._drift_detectors:
            return DriftResult(
                is_drift=False,
                drift_type=DriftType.DATA_DRIFT,
                severity=DriftSeverity.NONE,
                p_value=1.0,
                test_statistic=0.0,
                threshold=0.05,
            )

        detector = self._drift_detectors["main"]
        return detector.detect(X_test, feature_names)

    def _extract_metric_values(self, performances: list[SplitPerformance]) -> list[float]:
        """Extract primary metric values for degradation analysis."""
        if self.metric == "f1":
            return [p.f1 for p in performances]
        elif self.metric == "accuracy":
            return [p.accuracy for p in performances]
        elif self.metric == "auc":
            return [p.auc_roc or 0.5 for p in performances]
        elif self.metric == "precision":
            return [p.precision for p in performances]
        elif self.metric == "recall":
            return [p.recall for p in performances]
        else:
            return [p.f1 for p in performances]


def create_concept_drift_evaluator(
    strategy: str = "expanding_window",
    n_splits: int = 5,
    **kwargs: Any,
) -> ConceptDriftEvaluator:
    """
    Factory function to create concept drift evaluator.

    Args:
        strategy: Splitting strategy name
        n_splits: Number of splits
        **kwargs: Additional arguments for ConceptDriftEvaluator

    Returns:
        Configured ConceptDriftEvaluator
    """
    strategy_map = {
        "expanding_window": TemporalSplitStrategy.EXPANDING_WINDOW,
        "sliding_window": TemporalSplitStrategy.SLIDING_WINDOW,
        "fixed_origin": TemporalSplitStrategy.FIXED_ORIGIN,
        "walk_forward": TemporalSplitStrategy.WALK_FORWARD,
        "blocked": TemporalSplitStrategy.BLOCKED,
    }

    strat = strategy_map.get(strategy, TemporalSplitStrategy.EXPANDING_WINDOW)

    return ConceptDriftEvaluator(
        n_splits=n_splits,
        strategy=strat,
        **kwargs,
    )


# Exports
__all__ = [
    "ConceptDriftEvaluationResult",
    "ConceptDriftEvaluator",
    "DegradationAnalysis",
    "DegradationAnalyzer",
    "DegradationTrend",
    "SplitPerformance",
    "TemporalSplit",
    "TemporalSplitStrategy",
    "TemporalSplitter",
    "create_concept_drift_evaluator",
]
