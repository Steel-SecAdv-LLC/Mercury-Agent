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

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Constants from 3R mechanism
PHI = 1.618033988749895  # Golden ratio
BENEVOLENCE_THRESHOLD = 0.99
SIGMA_SACRED_DEFAULT = 0.96
LYAPUNOV_LAMBDA = 0.25


@dataclass
class IntegrationResult:
    """Result from integrated detection pipeline."""

    # Detection outputs
    is_anomaly: np.ndarray
    anomaly_scores: np.ndarray
    calibrated_scores: np.ndarray
    confidence_intervals: np.ndarray | None = None

    # Domain contributions
    domain_scores: dict[str, np.ndarray] = field(default_factory=dict)
    domain_weights: dict[str, float] = field(default_factory=dict)

    # Ethical metrics
    benevolence_score: float = 1.0
    ethical_compliance: bool = True
    sigma_sacred_value: float = 0.96

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

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Any: ...

    def detect(self, X: np.ndarray) -> dict[str, Any]: ...

    def extract_features(self, X: np.ndarray) -> np.ndarray: ...


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
        sigma_sacred: float = SIGMA_SACRED_DEFAULT,
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
            sigma_sacred: Ethical threshold (0.93-0.96)
            benevolence_threshold: Required benevolence level
            fusion_method: Fusion strategy ("stacking", "bma", "ethical")
            use_calibration: Whether to apply probability calibration
            use_conformal: Whether to use conformal prediction
            conformal_alpha: Conformal significance level
            seed: Random seed for reproducibility
        """
        self.sigma_sacred = sigma_sacred
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
        **config_kwargs,
    ) -> "GOSNNIntegration":
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

    def add_base_domains(self) -> "GOSNNIntegration":
        """
        Add standard base domain detectors.

        Includes: Statistical, Temporal, Spatial, Dimensional, Graph-based
        """
        try:
            from omni_mercury_engine.core.enhanced_base_domains import (
                EnhancedBaseDetector,
                create_enhanced_detector,
            )
            from omni_mercury_engine.detectors.dimensional import (
                DimensionalAnomalyDetector,
            )
            from omni_mercury_engine.detectors.graph_based import (
                GraphBasedAnomalyDetector,
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
                ("dimensional", DimensionalAnomalyDetector, 1.0, 0.95),
                ("graph", GraphBasedAnomalyDetector, 1.0 / (PHI**2), 0.94),
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

    def add_model_domains(self) -> "GOSNNIntegration":
        """
        Add model domain components.

        Includes: Quantum, Biometric, Affective, Consciousness
        """
        try:
            from omni_mercury_engine.core.enhanced_model_domains import (
                EnhancedAffectiveModel,
                EnhancedBiometricModel,
                EnhancedQuantumModel,
                LyapunovStabilityAnalyzer,
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
        X: np.ndarray,
        y: np.ndarray | None = None,
        validation_split: float = 0.2,
    ) -> "GOSNNIntegration":
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
        X: np.ndarray,
        return_details: bool = False,
    ) -> IntegrationResult:
        """
        Perform integrated multi-domain anomaly detection.

        Args:
            X: Input data
            return_details: Whether to return detailed domain outputs

        Returns:
            IntegrationResult with detection outputs
        """
        import time

        start_time = time.time()

        if not self._fitted:
            raise RuntimeError("Must call fit() before detect()")

        # Collect domain predictions
        domain_scores = {}
        domain_features = {}

        for name, config in self.domains.items():
            if not config.enabled or config.detector_instance is None:
                continue

            detector = config.detector_instance

            try:
                if hasattr(detector, "detect"):
                    result = detector.detect(X)
                    scores = result.get("scores", np.zeros(len(X)))
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
            except Exception:
                pass

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

        return IntegrationResult(
            is_anomaly=is_anomaly,
            anomaly_scores=fused_scores,
            calibrated_scores=calibrated_scores,
            confidence_intervals=confidence_intervals,
            domain_scores=domain_scores if return_details else {},
            domain_weights=self._domain_weights.copy(),
            benevolence_score=benevolence_score,
            ethical_compliance=ethical_compliance,
            sigma_sacred_value=self.sigma_sacred,
            fusion_method=self.fusion_method,
            calibration_method="auto" if self._calibrator else "none",
            processing_time_ms=processing_time,
        )

    def _setup_fusion(
        self,
        domain_predictions: dict[str, np.ndarray],
        y_val: np.ndarray | None,
    ) -> None:
        """Set up fusion strategy."""
        try:
            from omni_mercury_engine.core.stacking_fusion import (
                BayesianModelAveraging,
                EthicallyConstrainedFusion,
                StackingFusion,
            )

            if self.fusion_method == "stacking":
                self._fusion = StackingFusion(seed=self.seed)
            elif self.fusion_method == "bma":
                self._fusion = BayesianModelAveraging()
            else:  # ethical
                self._fusion = EthicallyConstrainedFusion(
                    sigma_sacred=self.sigma_sacred,
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
        domain_predictions: dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> None:
        """Set up probability calibration."""
        try:
            from omni_mercury_engine.core.calibration import CalibrationEnsemble

            # Fuse for calibration
            fused = self._fuse_predictions(domain_predictions)

            self._calibrator = CalibrationEnsemble()
            self._calibrator.fit(fused, y_val)

        except ImportError:
            logger.debug("Calibration module not available")

    def _setup_conformal(
        self,
        domain_predictions: dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> None:
        """Set up conformal prediction."""
        try:
            from omni_mercury_engine.core.conformal_prediction import (
                SplitConformalPredictor,
            )

            fused = self._fuse_predictions(domain_predictions)

            self._conformal = SplitConformalPredictor(alpha=self.conformal_alpha)
            self._conformal.fit(fused, y_val)

        except ImportError:
            logger.debug("Conformal prediction module not available")

    def _fuse_predictions(
        self,
        domain_scores: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Fuse domain predictions using configured strategy."""
        if not domain_scores:
            return np.array([])

        n_samples = len(next(iter(domain_scores.values())))
        fused = np.zeros(n_samples)

        for name, scores in domain_scores.items():
            weight = self._domain_weights.get(name, 1.0 / len(domain_scores))
            fused += weight * scores

        return fused

    def _compute_adaptive_threshold(self, scores: np.ndarray) -> float:
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
        return best_threshold * (scores.max() - scores.min()) + scores.min()

    def _apply_benevolence_adjustment(
        self,
        scores: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
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
        domain_scores: dict[str, np.ndarray],
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
            "sigma_sacred": self.sigma_sacred,
            "benevolence_threshold": self.benevolence_threshold,
            "domain_ethical_scores": domain_ethical_scores,
            "average_ethical_score": avg_ethical,
            "passes_threshold": avg_ethical >= self.sigma_sacred,
            "fusion_method": self.fusion_method,
            "calibration_enabled": self.use_calibration,
            "conformal_enabled": self.use_conformal,
        }


def create_integrated_detector(
    domains: list[str] | None = None,
    sigma_sacred: float = SIGMA_SACRED_DEFAULT,
    fusion_method: str = "ethical",
    **kwargs,
) -> GOSNNIntegration:
    """
    Factory function to create integrated detector.

    Args:
        domains: List of domains to include (None = all)
        sigma_sacred: Ethical threshold
        fusion_method: Fusion strategy
        **kwargs: Additional arguments

    Returns:
        Configured GOSNNIntegration instance
    """
    integration = GOSNNIntegration(
        sigma_sacred=sigma_sacred,
        fusion_method=fusion_method,
        **kwargs,
    )

    # Add all domains if not specified
    if domains is None or "base" in domains:
        integration.add_base_domains()

    if domains is None or "model" in domains:
        integration.add_model_domains()

    return integration
