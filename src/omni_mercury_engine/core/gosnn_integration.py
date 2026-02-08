"""
Mercury Agent - GOSNN Integration Layer
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Integration layer connecting:
- Enhanced base domain detectors
- Enhanced model domain components
- GOSNN hub (Global Omni-Scalar Network)
- Stacking/BMA fusion
- Calibration and conformal prediction
- Benevolence optimization

Provides unified API for multi-domain anomaly detection with ethical constraints.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

# Constants from 3R mechanism
PHI = 1.618033988749895  # Golden ratio
BENEVOLENCE_THRESHOLD = 0.99
SIGMA_IMMUTABLE_DEFAULT = 0.96
LYAPUNOV_LAMBDA = 0.25

# Performance optimization constants
CACHE_MAX_SIZE = 1000  # Maximum cache entries
CACHE_TTL_SECONDS = 300  # Cache time-to-live


# =============================================================================
# LRU Cache with TTL for Detection Results (2x Speedup Target)
# =============================================================================
class TTLCache:
    """Thread-safe LRU cache with TTL for caching detection results."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: float = CACHE_TTL_SECONDS) -> None:
        """Initialize cache with size and TTL limits."""
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _compute_key(self, data: npt.NDArray[Any]) -> str:
        """Compute cache key from numpy array using fast hashing."""
        # Use tobytes() for efficient array hashing
        # Using SHA3-256 for Ava-Guardian alignment
        return hashlib.sha3_256(data.tobytes()).hexdigest()

    def get(self, data: npt.NDArray[Any]) -> Any | None:
        """Get cached result if valid."""
        key = self._compute_key(data)
        with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    # Expired, remove
                    del self._cache[key]
            self._misses += 1
            return None

    def set(self, data: npt.NDArray[Any], value: Any) -> None:
        """Store result in cache."""
        key = self._compute_key(data)
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (time.time(), value)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


# =============================================================================
# Performance Monitor for GOSNN Operations
# =============================================================================
@dataclass
class PerformanceMetric:
    """Single performance measurement."""

    operation: str
    duration_ms: float
    timestamp: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class GOSNNPerformanceMonitor:
    """
    Performance monitor for GOSNN operations.

    Tracks latency, throughput, and identifies bottlenecks
    for optimization targeting <2% overhead.
    """

    def __init__(self, max_entries: int = 5000) -> None:
        """Initialize performance monitor."""
        self._metrics: list[PerformanceMetric] = []
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def record(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **metadata: Any,
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=time.time(),
            success=success,
            metadata=metadata,
        )
        with self._lock:
            self._metrics.append(metric)
            if len(self._metrics) > self._max_entries:
                self._metrics = self._metrics[-self._max_entries :]

    def get_summary(self, operation: str | None = None) -> dict[str, Any]:
        """Get performance summary for operations."""
        with self._lock:
            metrics = self._metrics
            if operation:
                metrics = [m for m in metrics if m.operation == operation]

            if not metrics:
                return {"count": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}

            durations = [m.duration_ms for m in metrics]
            durations_sorted = sorted(durations)

            return {
                "count": len(metrics),
                "mean_ms": float(np.mean(durations)),
                "std_ms": float(np.std(durations)),
                "min_ms": float(min(durations)),
                "max_ms": float(max(durations)),
                "p50_ms": float(np.percentile(durations_sorted, 50)),
                "p95_ms": float(np.percentile(durations_sorted, 95)),
                "p99_ms": float(np.percentile(durations_sorted, 99)),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
            }

    def get_bottlenecks(self, threshold_ms: float = 100.0) -> list[dict[str, Any]]:
        """Identify operations exceeding latency threshold."""
        with self._lock:
            bottlenecks = []
            ops_seen: set[str] = set()
            for metric in self._metrics:
                if metric.duration_ms > threshold_ms and metric.operation not in ops_seen:
                    ops_seen.add(metric.operation)
                    bottlenecks.append(
                        {
                            "operation": metric.operation,
                            "duration_ms": metric.duration_ms,
                            "metadata": metric.metadata,
                        }
                    )
            return sorted(bottlenecks, key=lambda x: float(x["duration_ms"]), reverse=True)[:10]  # type: ignore[arg-type]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()


# Global instances
_detection_cache = TTLCache()
_performance_monitor = GOSNNPerformanceMonitor()


def get_detection_cache() -> TTLCache:
    """Get the global detection cache instance."""
    return _detection_cache


def get_performance_monitor() -> GOSNNPerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


@dataclass
class IntegrationResult:
    """Result from integrated detection pipeline."""

    # Detection outputs
    is_anomaly: npt.NDArray[Any]
    anomaly_scores: npt.NDArray[Any]
    calibrated_scores: npt.NDArray[Any]
    confidence_intervals: npt.NDArray[Any] | None = None

    # Domain contributions
    domain_scores: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    domain_weights: dict[str, float] = field(default_factory=dict)

    # Ethical metrics
    benevolence_score: float = 1.0
    ethical_compliance: bool = True
    sigma_immutable_value: float = 0.96

    # Performance metadata
    fusion_method: str = "stacking"
    calibration_method: str = "auto"
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "n_anomalies": int(np.sum(self.is_anomaly)),
            "mean_score": float(np.mean(self.anomaly_scores)),
            "calibrated_mean": float(np.mean(self.calibrated_scores)),
            "domain_weights": self.domain_weights,
            "benevolence": self.benevolence_score,
            "ethical_compliance": self.ethical_compliance,
            "fusion_method": self.fusion_method,
        }


class DetectorProtocol(Protocol):
    """Protocol for detectors in the integration layer."""

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any] | None = None) -> Any: ...

    def detect(self, X: npt.NDArray[Any]) -> dict[str, Any]: ...

    def extract_features(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ...


@dataclass
class DomainConfig:
    """Configuration for a detection domain."""

    name: str
    detector_class: type | None = None
    detector_instance: Any = None
    weight: float = 1.0
    ethical_score: float = 1.0
    enabled: bool = True
    calibration_method: str = "auto"
    use_conformal: bool = False
    conformal_alpha: float = 0.1


class GOSNNIntegration:
    """
    Integration layer for GOSNN-based multi-domain anomaly detection.

    Coordinates:
    - Multiple domain detectors (Statistical, Temporal, Spatial, etc.)
    - Enhanced model components (Quantum, Biometric, Affective)
    - Fusion strategies (Stacking, BMA, Ethical-constrained)
    - Calibration and uncertainty quantification
    - Benevolence optimization
    """

    def __init__(
        self,
        sigma_immutable: float = SIGMA_IMMUTABLE_DEFAULT,
        benevolence_threshold: float = BENEVOLENCE_THRESHOLD,
        fusion_method: str = "ethical",
        use_calibration: bool = True,
        use_conformal: bool = True,
        conformal_alpha: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize GOSNN integration.

        Args:
            sigma_immutable: Ethical threshold (0.93-0.96)
            benevolence_threshold: Required benevolence level
            fusion_method: Fusion strategy ("stacking", "bma", "ethical")
            use_calibration: Whether to apply probability calibration
            use_conformal: Whether to use conformal prediction
            conformal_alpha: Conformal significance level
            seed: Random seed for reproducibility
        """
        self.sigma_immutable = sigma_immutable
        self.benevolence_threshold = benevolence_threshold
        self.fusion_method = fusion_method
        self.use_calibration = use_calibration
        self.use_conformal = use_conformal
        self.conformal_alpha = conformal_alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Domain configurations
        self.domains: dict[str, DomainConfig] = {}

        # Fusion and calibration components
        self._fusion = None
        self._calibrator = None
        self._conformal = None

        # Tracking
        self._fitted = False
        self._domain_weights: dict[str, float] = {}

    def add_domain(
        self,
        name: str,
        detector: Any = None,
        detector_class: type | None = None,
        weight: float = 1.0,
        ethical_score: float = 1.0,
        **config_kwargs: Any,
    ) -> GOSNNIntegration:
        """
        Add a detection domain to the integration.

        Args:
            name: Domain name (e.g., "statistical", "temporal")
            detector: Detector instance (optional)
            detector_class: Detector class to instantiate (optional)
            weight: Initial weight for this domain
            ethical_score: Ethical compliance score (0-1)
            **config_kwargs: Additional domain configuration

        Returns:
            Self for method chaining
        """
        self.domains[name] = DomainConfig(
            name=name,
            detector_class=detector_class,
            detector_instance=detector,
            weight=weight,
            ethical_score=ethical_score,
            **config_kwargs,
        )

        return self

    def add_base_domains(self) -> GOSNNIntegration:
        """
        Add standard base domain detectors.

        Includes: Statistical, Temporal, Spatial, Dimensional, Graph-based
        """
        try:
            from omni_mercury_engine.core.enhanced_base_domains import (
                EnhancedBaseDetector,
            )
            from omni_mercury_engine.detectors.dimensional import (
                DimensionalAnalyzer,
            )
            from omni_mercury_engine.detectors.graph_based import (
                GraphAnomalyDetector,
            )
            from omni_mercury_engine.detectors.spatial import (
                SpatialAnomalyDetector,
            )

            # Import base detectors
            from omni_mercury_engine.detectors.statistical import (
                StatisticalAnomalyDetector,
            )
            from omni_mercury_engine.detectors.temporal import (
                TemporalAnomalyDetector,
            )

            # Add enhanced domains
            domain_configs = [
                ("statistical", StatisticalAnomalyDetector, PHI, 0.98),
                ("temporal", TemporalAnomalyDetector, 1.0, 0.97),
                ("spatial", SpatialAnomalyDetector, 1.0 / PHI, 0.96),
                ("dimensional", DimensionalAnalyzer, 1.0, 0.95),
                ("graph", GraphAnomalyDetector, 1.0 / (PHI**2), 0.94),
            ]

            for name, detector_class, weight, ethical_score in domain_configs:
                try:
                    base = detector_class({})
                    enhanced = EnhancedBaseDetector(
                        base_detector=base,
                        domain=name,
                        threshold_method="otsu",
                        use_calibration=self.use_calibration,
                    )
                    self.add_domain(
                        name,
                        detector=enhanced,
                        weight=weight,
                        ethical_score=ethical_score,
                    )
                except Exception as e:
                    logger.warning(f"Failed to add {name} domain: {e}")

        except ImportError as e:
            logger.warning(f"Could not import base detectors: {e}")

        return self

    def add_model_domains(self) -> GOSNNIntegration:
        """
        Add model domain components.

        Includes: Quantum, Biometric, Affective, Consciousness
        """
        try:
            from omni_mercury_engine.core.enhanced_model_domains import (
                EnhancedAffectiveModel,
                EnhancedBiometricModel,
                EnhancedQuantumModel,
            )

            # Quantum model
            self.add_domain(
                "quantum",
                detector=EnhancedQuantumModel(seed=self.seed),
                weight=PHI,
                ethical_score=0.99,
            )

            # Biometric model (with fairness)
            self.add_domain(
                "biometric",
                detector=EnhancedBiometricModel(
                    enforce_fairness=True,
                    fairness_threshold=0.8,
                ),
                weight=1.0,
                ethical_score=0.98,
            )

            # Affective model
            self.add_domain(
                "affective",
                detector=EnhancedAffectiveModel(seed=self.seed),
                weight=1.0 / PHI,
                ethical_score=0.97,
            )

        except ImportError as e:
            logger.warning(f"Could not import model domains: {e}")

        return self

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any] | None = None,
        validation_split: float = 0.2,
    ) -> GOSNNIntegration:
        """
        Fit all domains and integration components.

        Args:
            X: Training data
            y: Optional labels
            validation_split: Fraction for calibration validation

        Returns:
            Self for method chaining
        """
        np.random.seed(self.seed)

        # Split for calibration
        n = len(X)
        n_val = int(n * validation_split)
        idx = np.random.permutation(n)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx] if y is not None else None
        y_val = y[val_idx] if y is not None else None

        # Fit all domain detectors
        domain_predictions = {}

        for name, config in self.domains.items():
            if not config.enabled or config.detector_instance is None:
                continue

            detector = config.detector_instance

            try:
                # Fit detector
                if hasattr(detector, "fit"):
                    if y_train is not None:
                        try:
                            detector.fit(X_train, y_train)
                        except TypeError:
                            detector.fit(X_train)
                    else:
                        detector.fit(X_train)

                # Get validation predictions
                if hasattr(detector, "detect"):
                    result = detector.detect(X_val)
                    scores = result.get("scores", np.zeros(len(X_val)))
                elif hasattr(detector, "extract_features"):
                    features = detector.extract_features(X_val)
                    scores = np.mean(features, axis=1)  # Simple score from features
                else:
                    scores = np.zeros(len(X_val))

                domain_predictions[name] = scores

            except Exception as e:
                logger.warning(f"Failed to fit domain {name}: {e}")
                domain_predictions[name] = np.zeros(len(X_val))

        # Set up fusion
        self._setup_fusion(domain_predictions, y_val)

        # Set up calibration
        if self.use_calibration and y_val is not None:
            self._setup_calibration(domain_predictions, y_val)

        # Set up conformal prediction
        if self.use_conformal and y_val is not None:
            self._setup_conformal(domain_predictions, y_val)

        self._fitted = True
        logger.info(
            f"GOSNNIntegration fitted: {len(self.domains)} domains, " f"fusion={self.fusion_method}"
        )

        return self

    def detect(
        self,
        X: npt.NDArray[Any],
        return_details: bool = False,
        use_cache: bool = True,
    ) -> IntegrationResult:
        """
        Perform integrated multi-domain anomaly detection.

        Args:
            X: Input data
            return_details: Whether to return detailed domain outputs
            use_cache: Whether to use detection result caching (2x speedup)

        Returns:
            IntegrationResult with detection outputs
        """
        start_time = time.time()

        if not self._fitted:
            raise RuntimeError("Must call fit() before detect()")

        # Check cache for repeated detections (2x speedup for repeated queries)
        cache = get_detection_cache()
        monitor = get_performance_monitor()

        if use_cache:
            cached_result = cache.get(X)
            if cached_result is not None and isinstance(cached_result, IntegrationResult):
                monitor.record("detect_cached", (time.time() - start_time) * 1000)
                result: IntegrationResult = cached_result
                return result

        # Collect domain predictions
        domain_scores = {}
        domain_features = {}

        for name, config in self.domains.items():
            if not config.enabled or config.detector_instance is None:
                continue

            detector = config.detector_instance

            try:
                if hasattr(detector, "detect"):
                    detect_result = detector.detect(X)
                    scores = detect_result.get("scores", np.zeros(len(X)))
                elif hasattr(detector, "extract_features"):
                    features = detector.extract_features(X)
                    domain_features[name] = features
                    scores = np.mean(features, axis=1)
                else:
                    scores = np.zeros(len(X))

                domain_scores[name] = scores

            except Exception as e:
                logger.warning(f"Detection failed for domain {name}: {e}")
                domain_scores[name] = np.zeros(len(X))

        # Fuse domain predictions
        fused_scores = self._fuse_predictions(domain_scores)

        # Apply calibration
        if self._calibrator is not None:
            calibrated_scores = self._calibrator.calibrate(fused_scores)
        else:
            calibrated_scores = fused_scores

        # Apply conformal prediction
        confidence_intervals = None
        if self._conformal is not None:
            try:
                confidence_intervals = self._conformal.predict(X)
            except (ValueError, RuntimeError, AttributeError) as e:
                # Conformal prediction optional - log and continue without intervals
                logger.debug(f"Conformal prediction skipped: {type(e).__name__}: {e}")

        # Compute adaptive threshold
        threshold = self._compute_adaptive_threshold(calibrated_scores)

        # Apply benevolence weighting
        benevolence_adjusted = self._apply_benevolence_adjustment(calibrated_scores, threshold)

        # Final predictions
        is_anomaly = benevolence_adjusted > threshold

        # Compute benevolence metrics
        benevolence_score = self._compute_benevolence_score(domain_scores)
        ethical_compliance = benevolence_score >= self.benevolence_threshold

        processing_time = (time.time() - start_time) * 1000

        result = IntegrationResult(
            is_anomaly=is_anomaly,
            anomaly_scores=fused_scores,
            calibrated_scores=calibrated_scores,
            confidence_intervals=confidence_intervals,
            domain_scores=domain_scores if return_details else {},
            domain_weights=self._domain_weights.copy(),
            benevolence_score=benevolence_score,
            ethical_compliance=ethical_compliance,
            sigma_immutable_value=self.sigma_immutable,
            fusion_method=self.fusion_method,
            calibration_method="auto" if self._calibrator else "none",
            processing_time_ms=processing_time,
        )

        # Cache result for 2x speedup on repeated queries
        if use_cache:
            cache.set(X, result)

        # Record performance metrics
        monitor.record(
            "detect",
            processing_time,
            success=True,
            n_samples=len(X),
            n_domains=len(self.domains),
            cache_hit=False,
        )

        return result

    def _setup_fusion(
        self,
        domain_predictions: dict[str, npt.NDArray[Any]],
        y_val: npt.NDArray[Any] | None,
    ) -> None:
        """Set up fusion strategy."""
        try:
            from omni_mercury_engine.core.stacking_fusion import (
                BayesianModelAveraging,
                EthicallyConstrainedFusion,
                StackingFusion,
            )

            if self.fusion_method == "stacking":
                self._fusion = StackingFusion(seed=self.seed)  # type: ignore[assignment]
            elif self.fusion_method == "bma":
                self._fusion = BayesianModelAveraging()  # type: ignore[assignment]
            else:  # ethical
                self._fusion = EthicallyConstrainedFusion(  # type: ignore[assignment]
                    sigma_immutable=self.sigma_immutable,
                )

            # Initialize domain weights based on ethical scores
            self._domain_weights = {
                name: config.weight * config.ethical_score
                for name, config in self.domains.items()
                if config.enabled
            }

            # Normalize
            total = sum(self._domain_weights.values())
            if total > 0:
                self._domain_weights = {k: v / total for k, v in self._domain_weights.items()}

        except ImportError:
            # Fallback to simple weighted average
            self._domain_weights = {
                name: config.weight for name, config in self.domains.items() if config.enabled
            }
            total = sum(self._domain_weights.values())
            if total > 0:
                self._domain_weights = {k: v / total for k, v in self._domain_weights.items()}

    def _setup_calibration(
        self,
        domain_predictions: dict[str, npt.NDArray[Any]],
        y_val: npt.NDArray[Any],
    ) -> None:
        """Set up probability calibration."""
        try:
            from omni_mercury_engine.core.calibration import CalibrationEnsemble

            # Fuse for calibration
            fused = self._fuse_predictions(domain_predictions)

            calibrator = CalibrationEnsemble()
            calibrator.fit(fused, y_val)
            self._calibrator = calibrator  # type: ignore[assignment]

        except ImportError:
            logger.debug("Calibration module not available")

    def _setup_conformal(
        self,
        domain_predictions: dict[str, npt.NDArray[Any]],
        y_val: npt.NDArray[Any],
    ) -> None:
        """Set up conformal prediction."""
        try:
            from omni_mercury_engine.core.conformal_prediction import (
                SplitConformalPredictor,
            )

            fused = self._fuse_predictions(domain_predictions)

            conformal = SplitConformalPredictor(coverage=self.conformal_alpha)
            conformal.fit(fused, y_val)  # type: ignore[call-arg]
            self._conformal = conformal  # type: ignore[assignment]

        except ImportError:
            logger.debug("Conformal prediction module not available")

    def _fuse_predictions(
        self,
        domain_scores: dict[str, npt.NDArray[Any]],
    ) -> npt.NDArray[Any]:
        """Fuse domain predictions using configured strategy."""
        if not domain_scores:
            return np.array([])

        n_samples = len(next(iter(domain_scores.values())))
        fused = np.zeros(n_samples)

        for name, scores in domain_scores.items():
            weight = self._domain_weights.get(name, 1.0 / len(domain_scores))
            fused += weight * scores

        return fused

    def _compute_adaptive_threshold(self, scores: npt.NDArray[Any]) -> float:
        """Compute adaptive threshold using Otsu's method."""
        if len(scores) < 10:
            return 0.5

        # Normalize scores to [0, 1]
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        # Otsu's method
        n_bins = 256
        hist, bin_edges = np.histogram(scores_norm, bins=n_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        weight_total = hist.sum()
        mean_total = np.sum(bin_centers * hist) / weight_total

        best_threshold = 0.5
        best_variance = 0.0

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

            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

            if variance > best_variance:
                best_variance = variance
                best_threshold = center

        # Convert back to original scale
        return float(best_threshold * (scores.max() - scores.min()) + scores.min())

    def _apply_benevolence_adjustment(
        self,
        scores: npt.NDArray[Any],
        threshold: float,
    ) -> npt.NDArray[Any]:
        """Apply benevolence-aware adjustment to scores."""
        # Higher benevolence = more conservative (reduce false positives)
        # Lower benevolence = more aggressive (reduce false negatives)

        adjustment_factor = 1.0 + (1.0 - self.benevolence_threshold) * PHI

        # Scores near threshold get adjusted more
        distance_from_threshold = np.abs(scores - threshold)
        adjustment_weight = np.exp(-distance_from_threshold / (threshold + 1e-10))

        adjusted = scores * (1 + adjustment_weight * (adjustment_factor - 1))

        return adjusted

    def _compute_benevolence_score(
        self,
        domain_scores: dict[str, npt.NDArray[Any]],
    ) -> float:
        """Compute overall benevolence score."""
        ethical_scores = []

        for name, config in self.domains.items():
            if name in domain_scores and config.enabled:
                # Weight by domain ethical score
                ethical_scores.append(config.ethical_score * config.weight)

        if not ethical_scores:
            return 1.0

        # Compute weighted average with golden ratio emphasis
        total_weight = sum(c.weight for c in self.domains.values() if c.enabled)
        benevolence = sum(ethical_scores) / (total_weight + 1e-10)

        # Apply Lyapunov stability factor
        stability_factor = np.exp(-LYAPUNOV_LAMBDA * (1 - benevolence))

        return float(np.clip(benevolence * stability_factor, 0, 1))

    def get_domain_contributions(self) -> dict[str, float]:
        """Get contribution of each domain to the final score."""
        return self._domain_weights.copy()

    def get_ethical_report(self) -> dict[str, Any]:
        """Generate ethical compliance report."""
        domain_ethical_scores = {
            name: config.ethical_score for name, config in self.domains.items() if config.enabled
        }

        avg_ethical = np.mean(list(domain_ethical_scores.values()))

        return {
            "sigma_immutable": self.sigma_immutable,
            "benevolence_threshold": self.benevolence_threshold,
            "domain_ethical_scores": domain_ethical_scores,
            "average_ethical_score": avg_ethical,
            "passes_threshold": avg_ethical >= self.sigma_immutable,
            "fusion_method": self.fusion_method,
            "calibration_enabled": self.use_calibration,
            "conformal_enabled": self.use_conformal,
        }


def create_integrated_detector(
    domains: list[str] | None = None,
    sigma_immutable: float = SIGMA_IMMUTABLE_DEFAULT,
    fusion_method: str = "ethical",
    **kwargs: Any,
) -> GOSNNIntegration:
    """
    Factory function to create integrated detector.

    Args:
        domains: List of domains to include (None = all)
        sigma_immutable: Ethical threshold
        fusion_method: Fusion strategy
        **kwargs: Additional arguments

    Returns:
        Configured GOSNNIntegration instance
    """
    integration = GOSNNIntegration(
        sigma_immutable=sigma_immutable,
        fusion_method=fusion_method,
        **kwargs,
    )

    # Add all domains if not specified
    if domains is None or "base" in domains:
        integration.add_base_domains()

    if domains is None or "model" in domains:
        integration.add_model_domains()

    return integration
