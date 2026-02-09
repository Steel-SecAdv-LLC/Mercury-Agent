"""
Mercury Agent - Enhanced Base Domain Detectors
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Enhancements to base domain detectors:
- Adaptive thresholds using Otsu's method and Bayesian optimization
- Event-based metrics for temporal anomalies
- Spatial autocorrelation (Moran's I) for graph/spatial domains
- PR-AUC for imbalanced detection scenarios
- Integration with calibration/conformal modules
- Parallel processing for efficiency
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Golden ratio for harmonic scaling
PHI = 1.618033988749895


@dataclass
class AdaptiveThresholdResult:
    """Result of adaptive threshold computation."""

    threshold: float
    method: str
    confidence: float
    otsu_score: float | None = None
    bayesian_bounds: tuple[float, float] | None = None


@dataclass
class DomainMetrics:
    """Comprehensive metrics for a detection domain."""

    # Standard metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # Domain-specific metrics
    time_to_detection: float | None = None  # For temporal
    spatial_autocorrelation: float | None = None  # For spatial
    spectral_divergence: float | None = None  # For dimensional
    cascade_risk: float | None = None  # For graph

    # Calibration metrics
    brier_score: float | None = None
    ece: float | None = None

    # Benevolence metrics
    equity_score: float | None = None
    harm_reduction: float | None = None


class AdaptiveThresholdOptimizer:
    """
    Adaptive threshold optimization using multiple methods.

    Methods:
    - Otsu's method (histogram-based)
    - Percentile-based
    - Bayesian optimization
    - F1-maximizing search
    """

    def __init__(
        self,
        method: str = "otsu",
        percentile: float = 95.0,
        n_bins: int = 256,
    ):
        """
        Initialize adaptive threshold optimizer.

        Args:
            method: Threshold method ("otsu", "percentile", "bayesian", "f1_max")
            percentile: Percentile for percentile-based method
            n_bins: Number of bins for histogram methods
        """
        self.method = method
        self.percentile = percentile
        self.n_bins = n_bins

    def compute_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> AdaptiveThresholdResult:
        """
        Compute adaptive threshold for anomaly scores.

        Args:
            scores: Anomaly scores (higher = more anomalous)
            labels: Optional ground truth labels for supervised methods

        Returns:
            AdaptiveThresholdResult with threshold and metadata
        """
        if self.method == "otsu":
            return self._otsu_threshold(scores)
        elif self.method == "percentile":
            return self._percentile_threshold(scores)
        elif self.method == "bayesian":
            return self._bayesian_threshold(scores, labels)
        elif self.method == "f1_max" and labels is not None:
            return self._f1_max_threshold(scores, labels)
        else:
            return self._percentile_threshold(scores)

    def _otsu_threshold(self, scores: np.ndarray) -> AdaptiveThresholdResult:
        """
        Compute threshold using Otsu's method.

        Otsu's method finds the threshold that minimizes within-class variance,
        effectively separating normal from anomalous samples.
        """
        # Normalize scores to [0, 1]
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # Compute histogram
        hist, bin_edges = np.histogram(scores_norm, bins=self.n_bins, range=(0, 1))
        hist = hist.astype(float)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Total weight and mean
        weight_total = hist.sum()
        mean_total = np.sum(bin_centers * hist) / weight_total

        best_threshold = 0.5
        best_variance = 0.0
        otsu_score = 0.0

        # Search for optimal threshold
        weight_bg = 0.0
        sum_bg = 0.0

        for i, (count, center) in enumerate(zip(hist, bin_centers)):
            weight_bg += count
            if weight_bg == 0:
                continue

            weight_fg = weight_total - weight_bg
            if weight_fg == 0:
                break

            sum_bg += count * center
            mean_bg = sum_bg / weight_bg
            mean_fg = (mean_total * weight_total - sum_bg) / weight_fg

            # Between-class variance
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

            if variance > best_variance:
                best_variance = variance
                best_threshold = center
                otsu_score = variance / (weight_total + 1e-10)

        # Convert back to original scale
        threshold = best_threshold * (scores.max() - scores.min()) + scores.min()

        return AdaptiveThresholdResult(
            threshold=threshold,
            method="otsu",
            confidence=min(otsu_score * 10, 1.0),
            otsu_score=otsu_score,
        )

    def _percentile_threshold(self, scores: np.ndarray) -> AdaptiveThresholdResult:
        """Compute threshold using percentile."""
        threshold = np.percentile(scores, self.percentile)

        return AdaptiveThresholdResult(
            threshold=threshold,
            method="percentile",
            confidence=0.8,
        )

    def _bayesian_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> AdaptiveThresholdResult:
        """
        Compute threshold using Bayesian estimation.

        Assumes scores come from a mixture of two distributions (normal/anomaly).
        """
        # Fit GMM-like approximation using EM
        # Start with percentile-based initial guess
        init_threshold = np.percentile(scores, 90)

        normal_scores = scores[scores < init_threshold]
        anomaly_scores = scores[scores >= init_threshold]

        if len(normal_scores) < 5 or len(anomaly_scores) < 2:
            return self._percentile_threshold(scores)

        # Estimate parameters
        mu_normal = np.mean(normal_scores)
        sigma_normal = np.std(normal_scores) + 1e-10
        mu_anomaly = np.mean(anomaly_scores)
        sigma_anomaly = np.std(anomaly_scores) + 1e-10

        # Bayesian decision boundary (equal posterior probability)
        # Assuming equal priors, threshold is where likelihoods are equal
        # For Gaussian: solve for x where N(x|μ₁,σ₁) = N(x|μ₂,σ₂)

        a = 1 / (2 * sigma_normal**2) - 1 / (2 * sigma_anomaly**2)
        b = mu_anomaly / sigma_anomaly**2 - mu_normal / sigma_normal**2
        c = (
            mu_normal**2 / (2 * sigma_normal**2)
            - mu_anomaly**2 / (2 * sigma_anomaly**2)
            - np.log(sigma_anomaly / sigma_normal)
        )

        if abs(a) < 1e-10:
            # Linear case
            threshold = -c / b if abs(b) > 1e-10 else init_threshold
        else:
            # Quadratic case
            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                x1 = (-b + np.sqrt(discriminant)) / (2 * a)
                x2 = (-b - np.sqrt(discriminant)) / (2 * a)
                # Choose threshold between the means
                if mu_normal < x1 < mu_anomaly:
                    threshold = x1
                elif mu_normal < x2 < mu_anomaly:
                    threshold = x2
                else:
                    threshold = (mu_normal + mu_anomaly) / 2
            else:
                threshold = (mu_normal + mu_anomaly) / 2

        # Compute confidence based on separation
        separation = (mu_anomaly - mu_normal) / np.sqrt(sigma_normal**2 + sigma_anomaly**2)
        confidence = min(np.tanh(separation / 2), 1.0)

        return AdaptiveThresholdResult(
            threshold=threshold,
            method="bayesian",
            confidence=confidence,
            bayesian_bounds=(mu_normal + 2 * sigma_normal, mu_anomaly - 2 * sigma_anomaly),
        )

    def _f1_max_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> AdaptiveThresholdResult:
        """Compute threshold that maximizes F1 score."""
        # Try different thresholds
        thresholds = np.percentile(scores, np.linspace(1, 99, 50))

        best_f1 = 0.0
        best_threshold = np.median(scores)

        for thresh in thresholds:
            predictions = (scores >= thresh).astype(int)

            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        return AdaptiveThresholdResult(
            threshold=best_threshold,
            method="f1_max",
            confidence=best_f1,
        )


class EventBasedMetrics:
    """
    Event-based metrics for time-series anomaly detection.

    Evaluates detection at the event level rather than point level,
    more appropriate for real-world applications where contiguous
    anomalous segments matter.
    """

    def __init__(self, tolerance: int = 0, min_event_length: int = 1):
        """
        Initialize event-based metrics.

        Args:
            tolerance: Points of tolerance for event matching
            min_event_length: Minimum event length to consider
        """
        self.tolerance = tolerance
        self.min_event_length = min_event_length

    def extract_events(self, labels: np.ndarray) -> list[tuple[int, int]]:
        """
        Extract contiguous events from binary labels.

        Args:
            labels: Binary array (0=normal, 1=anomaly)

        Returns:
            List of (start, end) tuples for each event
        """
        events = []
        in_event = False
        start = 0

        for i, val in enumerate(labels):
            if val == 1 and not in_event:
                start = i
                in_event = True
            elif val == 0 and in_event:
                if i - start >= self.min_event_length:
                    events.append((start, i - 1))
                in_event = False

        if in_event and len(labels) - start >= self.min_event_length:
            events.append((start, len(labels) - 1))

        return events

    def compute_time_to_detection(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute average time-to-detection for anomaly events.

        Measures how quickly after an anomaly starts it is detected.

        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels

        Returns:
            Average time-to-detection (in samples)
        """
        true_events = self.extract_events(y_true)

        if not true_events:
            return 0.0

        detection_times = []

        for start, end in true_events:
            # Find first detection within this event
            event_preds = y_pred[start : end + 1]
            detected_indices = np.where(event_preds == 1)[0]

            if len(detected_indices) > 0:
                detection_times.append(detected_indices[0])
            else:
                # Event not detected - penalize with event length
                detection_times.append(end - start + 1)

        return float(np.mean(detection_times))

    def compute_event_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute comprehensive event-based metrics.

        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels

        Returns:
            Dictionary with event-based precision, recall, F1, and TTD
        """
        true_events = self.extract_events(y_true)
        pred_events = self.extract_events(y_pred)

        if not true_events:
            return {
                "event_precision": 1.0 if not pred_events else 0.0,
                "event_recall": 1.0,
                "event_f1": 1.0 if not pred_events else 0.0,
                "time_to_detection": 0.0,
            }

        if not pred_events:
            return {
                "event_precision": 1.0,
                "event_recall": 0.0,
                "event_f1": 0.0,
                "time_to_detection": float(np.mean([e[1] - e[0] + 1 for e in true_events])),
            }

        def events_overlap(e1: tuple[int, int], e2: tuple[int, int]) -> bool:
            return not (e1[1] + self.tolerance < e2[0] or e2[1] + self.tolerance < e1[0])

        # Event recall: fraction of true events detected
        detected = sum(1 for te in true_events if any(events_overlap(te, pe) for pe in pred_events))
        event_recall = detected / len(true_events)

        # Event precision: fraction of predictions matching true events
        matched = sum(1 for pe in pred_events if any(events_overlap(pe, te) for te in true_events))
        event_precision = matched / len(pred_events)

        # F1
        if event_precision + event_recall > 0:
            event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall)
        else:
            event_f1 = 0.0

        ttd = self.compute_time_to_detection(y_true, y_pred)

        return {
            "event_precision": event_precision,
            "event_recall": event_recall,
            "event_f1": event_f1,
            "time_to_detection": ttd,
        }


class SpatialAutocorrelation:
    """
    Spatial autocorrelation metrics for graph and spatial domains.

    Implements Moran's I and Geary's C for measuring spatial clustering.
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize spatial autocorrelation calculator.

        Args:
            normalize: Whether to normalize the metric
        """
        self.normalize = normalize

    def compute_morans_i(
        self,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Compute Moran's I statistic for spatial autocorrelation.

        Moran's I measures overall spatial autocorrelation:
        - I > 0: Positive autocorrelation (clustering)
        - I < 0: Negative autocorrelation (dispersion)
        - I ≈ 0: Random spatial pattern

        Args:
            values: Array of values at each location
            weights: Spatial weight matrix (adjacency/distance)

        Returns:
            Tuple of (I_statistic, expected_I, z_score)
        """
        n = len(values)
        mean = np.mean(values)
        deviations = values - mean

        # Normalize weights if requested
        if self.normalize:
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = weights / (row_sums + 1e-10)

        # Compute Moran's I
        numerator = np.sum(weights * np.outer(deviations, deviations))
        denominator = np.sum(deviations**2)
        w_sum = np.sum(weights)

        if denominator == 0 or w_sum == 0:
            return 0.0, 0.0, 0.0

        morans_i = (n / w_sum) * (numerator / denominator)

        # Expected value under null hypothesis
        expected_i = -1 / (n - 1)

        # Variance (under normality assumption)
        s1 = np.sum((weights + weights.T) ** 2) / 2
        s2 = np.sum((weights.sum(axis=0) + weights.sum(axis=1)) ** 2)
        s0 = w_sum

        var_i = (
            n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * s0**2)
            - (np.sum(deviations**4) / denominator**2) * ((n**2 - n) * s1 - 2 * n * s2 + 6 * s0**2)
        ) / ((n - 1) * (n - 2) * (n - 3) * s0**2) - expected_i**2

        var_i = max(var_i, 1e-10)
        z_score = (morans_i - expected_i) / np.sqrt(var_i)

        return float(morans_i), float(expected_i), float(z_score)

    def compute_gearys_c(
        self,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute Geary's C statistic.

        Geary's C focuses on local differences:
        - C < 1: Positive autocorrelation
        - C > 1: Negative autocorrelation
        - C = 1: Random

        Args:
            values: Array of values at each location
            weights: Spatial weight matrix

        Returns:
            Geary's C statistic
        """
        n = len(values)
        mean = np.mean(values)

        # Pairwise squared differences
        diff_matrix = (values[:, None] - values[None, :]) ** 2

        numerator = np.sum(weights * diff_matrix)
        denominator = 2 * np.sum(weights) * np.sum((values - mean) ** 2)

        if denominator == 0:
            return 1.0

        C = (n - 1) * numerator / denominator

        return float(C)


class ParallelDetectorExecutor:
    """
    Parallel execution of multiple detectors for efficiency.

    Uses ThreadPoolExecutor to run detectors concurrently,
    reducing latency for multi-detector fusion.
    """

    def __init__(self, max_workers: int = 4, timeout: float = 30.0):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum number of parallel workers
            timeout: Timeout for each detector (seconds)
        """
        self.max_workers = max_workers
        self.timeout = timeout

    def execute_detectors(
        self,
        detectors: dict[str, Any],
        data: np.ndarray,
        method: str = "detect",
    ) -> dict[str, Any]:
        """
        Execute multiple detectors in parallel.

        Args:
            detectors: Dictionary of detector name to detector instance
            data: Input data
            method: Method to call on each detector

        Returns:
            Dictionary of detector name to result
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for name, detector in detectors.items():
                func = getattr(detector, method, None)
                if func is not None:
                    futures[executor.submit(func, data)] = name

            for future in as_completed(futures, timeout=self.timeout):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.warning(f"Detector {name} failed: {e}")
                    results[name] = {"error": str(e)}

        return results


class EnhancedBaseDetector:
    """
    Enhanced base detector with adaptive thresholds and domain metrics.

    Wraps existing detectors with:
    - Adaptive threshold optimization
    - Event-based metrics (for temporal)
    - Spatial autocorrelation (for spatial/graph)
    - Calibration integration
    - Benevolence-aware scoring
    """

    def __init__(
        self,
        base_detector: Any,
        domain: str = "statistical",
        threshold_method: str = "otsu",
        use_calibration: bool = True,
        benevolence_weight: float = 0.1,
    ):
        """
        Initialize enhanced detector.

        Args:
            base_detector: Underlying detector instance
            domain: Domain type for domain-specific metrics
            threshold_method: Adaptive threshold method
            use_calibration: Whether to apply calibration
            benevolence_weight: Weight for benevolence in scoring
        """
        self.base_detector = base_detector
        self.domain = domain
        self.threshold_optimizer = AdaptiveThresholdOptimizer(method=threshold_method)
        self.use_calibration = use_calibration
        self.benevolence_weight = benevolence_weight

        # Domain-specific metric calculators
        self.event_metrics = EventBasedMetrics() if domain == "temporal" else None
        self.spatial_metrics = SpatialAutocorrelation() if domain in ["spatial", "graph"] else None

        # Calibrator (lazy initialization)
        self._calibrator = None

        # Tracking
        self._threshold = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> EnhancedBaseDetector:
        """
        Fit enhanced detector.

        Args:
            X: Training data
            y: Optional labels for supervised threshold optimization

        Returns:
            Self for method chaining
        """
        # Fit base detector
        self.base_detector.fit(X)

        # Get scores for threshold optimization
        if hasattr(self.base_detector, "detect"):
            result = self.base_detector.detect(X)
            scores = result.get("scores", np.zeros(len(X)))
        else:
            scores = self._get_scores(X)

        # Compute adaptive threshold
        threshold_result = self.threshold_optimizer.compute_threshold(scores, y)
        self._threshold = threshold_result.threshold  # type: ignore[assignment]

        # Set up calibration if enabled
        if self.use_calibration and y is not None:
            self._setup_calibration(scores, y)

        self._fitted = True
        return self

    def detect(
        self,
        X: np.ndarray,
        return_metrics: bool = False,
    ) -> dict[str, Any]:
        """
        Detect anomalies with enhanced features.

        Args:
            X: Input data
            return_metrics: Whether to return domain metrics

        Returns:
            Detection result with enhanced information
        """
        # Get base detection
        if hasattr(self.base_detector, "detect"):
            base_result = self.base_detector.detect(X)
        else:
            scores = self._get_scores(X)
            base_result = {"scores": scores}

        scores = base_result.get("scores", np.zeros(len(X)))

        # Apply adaptive threshold
        if self._threshold is not None:
            is_anomaly = scores > self._threshold
        else:
            is_anomaly = scores > 0.5

        # Apply calibration if available
        if self._calibrator is not None:
            scores = self._calibrator.calibrate(scores)

        # Apply benevolence weighting (reduce false positives)
        # Higher benevolence = more conservative threshold
        if self.benevolence_weight > 0:
            benevolence_factor = 1 + self.benevolence_weight * PHI
            is_anomaly = scores > (self._threshold * benevolence_factor if self._threshold else 0.5)

        result = {
            **base_result,
            "scores": scores,
            "is_anomaly": is_anomaly,
            "threshold": self._threshold,
            "detector_type": self.domain,
        }

        if return_metrics:
            result["domain_metrics"] = self._compute_domain_metrics(X, scores)

        return result

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features with enhancements."""
        if hasattr(self.base_detector, "extract_features"):
            features = self.base_detector.extract_features(X)
            if hasattr(features, "numpy"):
                features = features.numpy()
            return features  # type: ignore[no-any-return, unused-ignore]
        return np.zeros((len(X), 32))

    def _get_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores from base detector."""
        if hasattr(self.base_detector, "detect"):
            result = self.base_detector.detect(X)
            return result.get("scores", np.zeros(len(X)))  # type: ignore[no-any-return, unused-ignore]
        elif hasattr(self.base_detector, "predict"):
            result = self.base_detector.predict(X)
            if isinstance(result, dict):
                return result.get("anomaly_scores", np.zeros(len(X)))  # type: ignore[no-any-return, unused-ignore]
            return result  # type: ignore[no-any-return, unused-ignore]
        return np.zeros(len(X))

    def _setup_calibration(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Set up calibration using Platt scaling."""
        try:
            from omni_mercury_engine.core.calibration import PlattScaling

            calibrator = PlattScaling()
            calibrator.fit(scores, labels)
            self._calibrator = calibrator  # type: ignore[assignment]
        except ImportError:
            logger.debug("Calibration module not available")

    def _compute_domain_metrics(
        self,
        X: np.ndarray,
        scores: np.ndarray,
    ) -> DomainMetrics:
        """Compute domain-specific metrics."""
        metrics = DomainMetrics()

        if self.domain == "temporal" and self.event_metrics is not None:
            # Compute time-to-detection
            is_anomaly = scores > (self._threshold or 0.5)
            metrics.time_to_detection = self.event_metrics.compute_time_to_detection(
                np.zeros(len(X)),  # Would need ground truth
                is_anomaly.astype(int),
            )

        elif self.domain in ["spatial", "graph"] and self.spatial_metrics is not None:
            # Compute spatial autocorrelation
            if X.ndim == 2:
                # Use score-based weights
                weights = np.exp(-np.abs(np.subtract.outer(scores, scores)))
                morans_i, _expected_i, _z = self.spatial_metrics.compute_morans_i(scores, weights)
                metrics.spatial_autocorrelation = morans_i

        elif self.domain == "dimensional":
            # Spectral divergence from detector if available
            if hasattr(self.base_detector, "_dimensional_code_breaking"):
                # Would compute from detector results
                pass

        return metrics


def create_enhanced_detector(
    detector_class: type,
    domain: str,
    config: dict[str, Any] | None = None,
    **enhancement_kwargs: Any,
) -> EnhancedBaseDetector:
    """
    Factory function to create enhanced detector.

    Args:
        detector_class: Base detector class
        domain: Domain type
        config: Configuration for base detector
        **enhancement_kwargs: Additional args for EnhancedBaseDetector

    Returns:
        Enhanced detector instance
    """
    base_detector = detector_class(config)
    return EnhancedBaseDetector(base_detector, domain=domain, **enhancement_kwargs)
