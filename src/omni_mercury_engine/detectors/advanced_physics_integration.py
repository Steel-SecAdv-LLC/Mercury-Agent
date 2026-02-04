"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

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
Advanced Physics Integration Module for Mercury Agent.

This module provides unified integration of the advanced physics-inspired
anomaly detection systems with the core Mercury Agent architecture:

1. Integration with 3R Mechanism (Recursion-Resonance-Refactoring)
2. Integration with GOSNN (Global Omni Scalar Network)
3. Unified detector interface for all physics-based detectors
4. Combined anomaly fusion using AAFE (Ava Anomaly Fusion Equation)

The module bridges:
- Spectral Vibration Analysis (frequencies, GNN, phonons)
- Acceleration Dynamics (kinematics, Lyapunov, phase space)
- UI/UX Anomaly Detection (user behavior, engagement)

With the existing Mercury Agent systems:
- Ethical governance via GOSNN scalars
- 3R mechanism for recursive enhancement
- Fusion network for multi-detector combination
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from omni_mercury_engine.core.base import BaseDetector, DetectorMetrics
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.core.three_r.fusion import AnomalyFusionEquation, AnomalyFusionResult
from omni_mercury_engine.core.three_r.engines import RecursionEngine, ResonanceEngine
from omni_mercury_engine.utils.constants import MathematicalConstants

# Import the new advanced detectors
from omni_mercury_engine.detectors.spectral_vibration import (
    SpectralVibrationDetector,
    SpectralVibrationConfig,
    SpectralFeatures,
    VibrationDiagnostic,
    VibrationSignatureType,
)
from omni_mercury_engine.detectors.acceleration_dynamics import (
    AccelerationDynamicsDetector,
    AccelerationDynamicsConfig,
    KinematicFeatures,
    PhaseSpaceFeatures,
    MotionState,
    EnergyState,
)
from omni_mercury_engine.detectors.uiux_anomaly import (
    UIUXAnomalyDetector,
    UIUXConfig,
    UserInteraction,
    InteractionType,
    AnomalyCategory,
    UserBehaviorClass,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PHI = MathematicalConstants.GOLDEN_RATIO
CONVERGENCE_RATE = 0.25


class PhysicsDetectorType(Enum):
    """Types of physics-based detectors."""

    SPECTRAL_VIBRATION = "spectral_vibration"
    ACCELERATION_DYNAMICS = "acceleration_dynamics"
    UIUX_ANOMALY = "uiux_anomaly"
    ALL = "all"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AdvancedPhysicsConfig:
    """Configuration for advanced physics integration.

    Attributes:
        enabled_detectors: Which physics detectors to enable
        spectral_config: Configuration for spectral vibration detector
        dynamics_config: Configuration for acceleration dynamics detector
        uiux_config: Configuration for UI/UX anomaly detector
        use_3r_enhancement: Whether to use 3R mechanism for enhancement
        use_gosnn_scaling: Whether to use GOSNN ethical scaling
        fusion_weights: Custom weights for detector fusion
        recursion_depth: Depth for recursive analysis
        resonance_harmonics: Number of harmonics for resonance analysis
        ethical_compliance_threshold: Threshold for ethical compliance
        threshold: Overall anomaly detection threshold
    """

    enabled_detectors: list[PhysicsDetectorType] = field(
        default_factory=lambda: [PhysicsDetectorType.ALL]
    )
    spectral_config: dict[str, Any] = field(default_factory=dict)
    dynamics_config: dict[str, Any] = field(default_factory=dict)
    uiux_config: dict[str, Any] = field(default_factory=dict)
    use_3r_enhancement: bool = True
    use_gosnn_scaling: bool = True
    fusion_weights: dict[str, float] | None = None
    recursion_depth: int = 3
    resonance_harmonics: int = 8
    ethical_compliance_threshold: float = 0.96
    threshold: float = 0.5


@dataclass
class IntegratedPhysicsResult:
    """Complete result from integrated physics detection.

    Attributes:
        anomaly_score: Combined anomaly score [0, 1]
        is_anomaly: Boolean anomaly flag
        spectral_result: Result from spectral vibration detector
        dynamics_result: Result from acceleration dynamics detector
        uiux_result: Result from UI/UX anomaly detector
        fusion_result: 3R AAFE fusion result
        recursion_score: Score from recursion engine
        resonance_score: Score from resonance engine
        ethical_scaling: Ethical compliance scaling factor
        component_scores: Individual detector scores
        detector_features: Extracted features from all detectors
        recommendations: Combined recommendations
        metadata: Additional metadata
    """

    anomaly_score: float
    is_anomaly: bool
    spectral_result: dict[str, Any] | None
    dynamics_result: dict[str, Any] | None
    uiux_result: dict[str, Any] | None
    fusion_result: AnomalyFusionResult | None
    recursion_score: float
    resonance_score: float
    ethical_scaling: float
    component_scores: dict[str, float]
    detector_features: torch.Tensor
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# GOSNN Integration
# =============================================================================

class PhysicsGOSNNScalars:
    """GOSNN scalar network for physics-based anomaly detection.

    Provides ethical governance scalars specific to the advanced
    physics detection modules.
    """

    # Physics domain scalars
    SPECTRAL_INTEGRITY = 0.95  # Integrity of spectral analysis
    KINEMATIC_PRECISION = 0.94  # Precision of kinematic calculations
    UIUX_FAIRNESS = 0.93  # Fairness in user behavior analysis
    ENERGY_CONSERVATION = 0.96  # Energy conservation principle adherence
    CHAOS_DETECTION_SENSITIVITY = 0.92  # Sensitivity to chaotic behavior
    BOT_DETECTION_FAIRNESS = 0.94  # Fairness in bot detection (avoid false positives)
    VIBRATION_ACCURACY = 0.95  # Accuracy of vibration signature detection

    # Ethical governance scalars
    HUMANITARIAN_IMPACT = 0.97  # Weight humanitarian considerations
    TRANSPARENCY = 0.95  # Transparent decision making
    EXPLAINABILITY = 0.94  # Explainable anomaly detection
    PRIVACY_PROTECTION = 0.96  # Protect user privacy in UIUX analysis
    BIAS_MITIGATION = 0.93  # Mitigate detection biases

    @classmethod
    def get_all_scalars(cls) -> dict[str, float]:
        """Get all GOSNN scalars.

        Returns:
            Dictionary of scalar names to values
        """
        return {
            name: value
            for name, value in vars(cls).items()
            if not name.startswith('_') and isinstance(value, (int, float))
        }

    @classmethod
    def compute_ethical_scaling(
        cls,
        detection_context: dict[str, Any],
    ) -> float:
        """Compute ethical scaling factor based on detection context.

        Args:
            detection_context: Context information about the detection

        Returns:
            Ethical scaling factor [0, 1]
        """
        scalars = cls.get_all_scalars()
        base_score = np.mean(list(scalars.values()))

        # Adjust based on context
        adjustments = []

        # If analyzing user behavior, emphasize privacy
        if detection_context.get("uiux_enabled", False):
            adjustments.append(cls.PRIVACY_PROTECTION)
            adjustments.append(cls.BOT_DETECTION_FAIRNESS)

        # If detecting potential anomalies, emphasize explainability
        if detection_context.get("anomaly_detected", False):
            adjustments.append(cls.EXPLAINABILITY)
            adjustments.append(cls.TRANSPARENCY)

        # If predictive maintenance context, emphasize humanitarian impact
        if detection_context.get("maintenance_context", False):
            adjustments.append(cls.HUMANITARIAN_IMPACT)

        if adjustments:
            context_score = np.mean(adjustments)
            # Blend base and context scores
            final_score = 0.6 * base_score + 0.4 * context_score
        else:
            final_score = base_score

        return float(final_score)


# =============================================================================
# Main Integrated Detector
# =============================================================================

class AdvancedPhysicsIntegratedDetector(BaseDetector):
    """Unified detector integrating all advanced physics-based modules.

    Provides a single interface for:
    - Spectral Vibration Analysis
    - Acceleration Dynamics
    - UI/UX Anomaly Detection

    With full integration into the Mercury Agent architecture:
    - 3R Mechanism for recursive enhancement
    - GOSNN for ethical governance
    - AAFE for anomaly fusion

    Example:
        >>> detector = AdvancedPhysicsIntegratedDetector(config={
        ...     "enabled_detectors": ["spectral_vibration", "acceleration_dynamics"],
        ...     "use_3r_enhancement": True,
        ...     "threshold": 0.6,
        ... })
        >>> detector.fit(training_data)
        >>> result = detector.detect(test_data)
        >>> print(result.anomaly_score, result.fusion_result)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize integrated physics detector.

        Args:
            config: Configuration dictionary. See AdvancedPhysicsConfig.
        """
        super().__init__(config)

        # Parse configuration
        self._physics_config = self._parse_config(config or {})

        # Initialize component detectors
        self._spectral_detector: SpectralVibrationDetector | None = None
        self._dynamics_detector: AccelerationDynamicsDetector | None = None
        self._uiux_detector: UIUXAnomalyDetector | None = None

        self._init_detectors()

        # Initialize 3R components
        self._recursion_engine: RecursionEngine | None = None
        self._resonance_engine: ResonanceEngine | None = None
        self._fusion_equation: AnomalyFusionEquation | None = None

        if self._physics_config.use_3r_enhancement:
            self._init_3r_components()

        # Initialize GOSNN scalars
        self._gosnn = PhysicsGOSNNScalars()

        # Set device
        self.device = torch.device(self.config.get("device", "cpu"))

    def _parse_config(self, config: dict[str, Any]) -> AdvancedPhysicsConfig:
        """Parse configuration dictionary.

        Args:
            config: Raw configuration dictionary

        Returns:
            AdvancedPhysicsConfig object
        """
        enabled = config.get("enabled_detectors", ["all"])
        if isinstance(enabled, list):
            enabled_types = []
            for e in enabled:
                if isinstance(e, str):
                    enabled_types.append(PhysicsDetectorType(e))
                elif isinstance(e, PhysicsDetectorType):
                    enabled_types.append(e)
        else:
            enabled_types = [PhysicsDetectorType.ALL]

        return AdvancedPhysicsConfig(
            enabled_detectors=enabled_types,
            spectral_config=config.get("spectral_config", {}),
            dynamics_config=config.get("dynamics_config", {}),
            uiux_config=config.get("uiux_config", {}),
            use_3r_enhancement=config.get("use_3r_enhancement", True),
            use_gosnn_scaling=config.get("use_gosnn_scaling", True),
            fusion_weights=config.get("fusion_weights"),
            recursion_depth=config.get("recursion_depth", 3),
            resonance_harmonics=config.get("resonance_harmonics", 8),
            ethical_compliance_threshold=config.get("ethical_compliance_threshold", 0.96),
            threshold=config.get("threshold", self.threshold),
        )

    def _init_detectors(self) -> None:
        """Initialize component detectors based on configuration."""
        cfg = self._physics_config
        enabled = cfg.enabled_detectors

        all_enabled = PhysicsDetectorType.ALL in enabled

        if all_enabled or PhysicsDetectorType.SPECTRAL_VIBRATION in enabled:
            spectral_config = {**cfg.spectral_config, "threshold": cfg.threshold}
            self._spectral_detector = SpectralVibrationDetector(spectral_config)

        if all_enabled or PhysicsDetectorType.ACCELERATION_DYNAMICS in enabled:
            dynamics_config = {**cfg.dynamics_config, "threshold": cfg.threshold}
            self._dynamics_detector = AccelerationDynamicsDetector(dynamics_config)

        if all_enabled or PhysicsDetectorType.UIUX_ANOMALY in enabled:
            uiux_config = {**cfg.uiux_config, "threshold": cfg.threshold}
            self._uiux_detector = UIUXAnomalyDetector(uiux_config)

    def _init_3r_components(self) -> None:
        """Initialize 3R mechanism components."""
        cfg = self._physics_config

        self._recursion_engine = RecursionEngine(
            max_depth=cfg.recursion_depth,
            decay_factor=1.0 / PHI,  # Golden ratio decay
        )

        self._resonance_engine = ResonanceEngine(
            num_harmonics=cfg.resonance_harmonics,
            fundamental_weight=PHI,
        )

        self._fusion_equation = AnomalyFusionEquation(
            ethical_compliance_threshold=cfg.ethical_compliance_threshold,
            convergence_rate=CONVERGENCE_RATE,
        )

    def fit(
        self,
        data: np.ndarray | torch.Tensor | dict[str, Any],
        data_type: str = "time_series",
    ) -> AdvancedPhysicsIntegratedDetector:
        """Fit all component detectors on training data.

        Args:
            data: Training data. Format depends on data_type:
                - "time_series": Array for spectral/dynamics analysis
                - "interactions": List of UserInteraction for UI/UX
                - "mixed": Dict with keys for each detector type
            data_type: Type of training data

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data format is invalid.
        """
        if data_type == "time_series":
            # Fit spectral and dynamics detectors
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

            if self._spectral_detector is not None:
                self._spectral_detector.fit(data)

            if self._dynamics_detector is not None:
                self._dynamics_detector.fit(data)

        elif data_type == "interactions":
            # Fit UI/UX detector
            if self._uiux_detector is not None:
                self._uiux_detector.fit(data)  # type: ignore

        elif data_type == "mixed":
            # Fit each detector with appropriate data
            if not isinstance(data, dict):
                raise DetectorException(
                    "Mixed data type requires dict with keys: "
                    "'time_series', 'interactions'"
                )

            if "time_series" in data:
                ts_data = data["time_series"]
                if isinstance(ts_data, torch.Tensor):
                    ts_data = ts_data.cpu().numpy()

                if self._spectral_detector is not None:
                    self._spectral_detector.fit(ts_data)
                if self._dynamics_detector is not None:
                    self._dynamics_detector.fit(ts_data)

            if "interactions" in data and self._uiux_detector is not None:
                self._uiux_detector.fit(data["interactions"])

        else:
            raise DetectorException(
                f"Unknown data_type: {data_type}. "
                "Use 'time_series', 'interactions', or 'mixed'."
            )

        self._is_fitted = True
        logger.info("AdvancedPhysicsIntegratedDetector fitted successfully")

        return self

    def detect(
        self,
        data: np.ndarray | torch.Tensor | dict[str, Any] | list[UserInteraction],
        data_type: str = "time_series",
    ) -> dict[str, Any]:
        """Detect anomalies using all enabled detectors.

        Args:
            data: Input data for detection
            data_type: Type of input data

        Returns:
            Dictionary containing integrated detection results.

        Raises:
            DetectorException: If detector not fitted.
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        # Run component detectors
        spectral_result = None
        dynamics_result = None
        uiux_result = None

        if data_type == "time_series":
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

            if self._spectral_detector is not None and self._spectral_detector.is_fitted():
                spectral_result = self._spectral_detector.detect(data)

            if self._dynamics_detector is not None and self._dynamics_detector.is_fitted():
                dynamics_result = self._dynamics_detector.detect(data)

        elif data_type == "interactions":
            if self._uiux_detector is not None and self._uiux_detector.is_fitted():
                uiux_result = self._uiux_detector.detect(data)  # type: ignore

        elif data_type == "mixed" and isinstance(data, dict):
            if "time_series" in data:
                ts_data = data["time_series"]
                if isinstance(ts_data, torch.Tensor):
                    ts_data = ts_data.cpu().numpy()

                if self._spectral_detector is not None and self._spectral_detector.is_fitted():
                    spectral_result = self._spectral_detector.detect(ts_data)
                if self._dynamics_detector is not None and self._dynamics_detector.is_fitted():
                    dynamics_result = self._dynamics_detector.detect(ts_data)

            if "interactions" in data:
                if self._uiux_detector is not None and self._uiux_detector.is_fitted():
                    uiux_result = self._uiux_detector.detect(data["interactions"])

        # Collect component scores
        component_scores = {}

        if spectral_result is not None:
            component_scores["spectral"] = spectral_result.get("anomaly_score", 0.0)

        if dynamics_result is not None:
            component_scores["dynamics"] = dynamics_result.get("anomaly_score", 0.0)

        if uiux_result is not None:
            component_scores["uiux"] = uiux_result.get("anomaly_score", 0.0)

        # Apply 3R enhancement
        recursion_score = 0.0
        resonance_score = 0.0
        fusion_result = None

        if self._physics_config.use_3r_enhancement and component_scores:
            recursion_score, resonance_score, fusion_result = self._apply_3r_enhancement(
                component_scores,
                spectral_result,
                dynamics_result,
            )

        # Compute GOSNN ethical scaling
        ethical_scaling = 1.0
        if self._physics_config.use_gosnn_scaling:
            detection_context = {
                "uiux_enabled": uiux_result is not None,
                "anomaly_detected": any(s > 0.5 for s in component_scores.values()),
                "maintenance_context": spectral_result is not None,
            }
            ethical_scaling = self._gosnn.compute_ethical_scaling(detection_context)

        # Compute combined anomaly score
        combined_score = self._compute_combined_score(
            component_scores,
            recursion_score,
            resonance_score,
            ethical_scaling,
            fusion_result,
        )

        # Extract combined features
        detector_features = self._extract_combined_features(
            data, data_type, spectral_result, dynamics_result, uiux_result
        )

        # Generate combined recommendations
        recommendations = self._generate_combined_recommendations(
            spectral_result, dynamics_result, uiux_result
        )

        # Auto-calibration
        effective_threshold = self.threshold
        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(np.array([combined_score]))

        is_anomaly = combined_score > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": combined_score,
            "spectral_result": spectral_result,
            "dynamics_result": dynamics_result,
            "uiux_result": uiux_result,
            "fusion_result": fusion_result,
            "recursion_score": recursion_score,
            "resonance_score": resonance_score,
            "ethical_scaling": ethical_scaling,
            "component_scores": component_scores,
            "detector_features": detector_features,
            "recommendations": recommendations,
            "detector_type": "advanced_physics_integrated",
            "threshold": effective_threshold,
            "metadata": {
                "enabled_detectors": [d.value for d in self._physics_config.enabled_detectors],
                "use_3r_enhancement": self._physics_config.use_3r_enhancement,
                "use_gosnn_scaling": self._physics_config.use_gosnn_scaling,
            },
        }

    def extract_features(
        self,
        data: np.ndarray | torch.Tensor | dict[str, Any],
        data_type: str = "time_series",
    ) -> torch.Tensor:
        """Extract features from all enabled detectors.

        Args:
            data: Input data
            data_type: Type of input data

        Returns:
            Combined feature tensor
        """
        # This calls detect internally to get features
        result = self.detect(data, data_type)
        return result["detector_features"]

    def _apply_3r_enhancement(
        self,
        component_scores: dict[str, float],
        spectral_result: dict[str, Any] | None,
        dynamics_result: dict[str, Any] | None,
    ) -> tuple[float, float, AnomalyFusionResult | None]:
        """Apply 3R mechanism enhancement.

        Args:
            component_scores: Scores from component detectors
            spectral_result: Full spectral detection result
            dynamics_result: Full dynamics detection result

        Returns:
            Tuple of (recursion_score, resonance_score, fusion_result)
        """
        recursion_score = 0.0
        resonance_score = 0.0
        fusion_result = None

        if not component_scores:
            return recursion_score, resonance_score, fusion_result

        scores_array = np.array(list(component_scores.values()))

        # Apply recursion engine
        if self._recursion_engine is not None:
            recursion_score = self._recursion_engine.process(scores_array)

        # Apply resonance engine (if we have spectral data)
        if self._resonance_engine is not None and spectral_result is not None:
            spectral_features = spectral_result.get("spectral_features")
            if spectral_features is not None and hasattr(spectral_features, 'power_spectrum'):
                power_spectrum = spectral_features.power_spectrum
                resonance_score = self._resonance_engine.analyze(power_spectrum)
            else:
                resonance_score = self._resonance_engine.analyze(scores_array)

        # Apply AAFE fusion
        if self._fusion_equation is not None:
            # Use component scores as inputs to fusion
            r_score = component_scores.get("spectral", 0.5)
            h_score = resonance_score if resonance_score > 0 else component_scores.get("dynamics", 0.5)
            o_score = component_scores.get("uiux", 0.5)

            fusion_result = self._fusion_equation.compute(
                recursion_score=r_score,
                resonance_score=h_score,
                optimization_score=o_score,
            )

        return recursion_score, resonance_score, fusion_result

    def _compute_combined_score(
        self,
        component_scores: dict[str, float],
        recursion_score: float,
        resonance_score: float,
        ethical_scaling: float,
        fusion_result: AnomalyFusionResult | None,
    ) -> float:
        """Compute combined anomaly score.

        Args:
            component_scores: Individual detector scores
            recursion_score: Score from recursion engine
            resonance_score: Score from resonance engine
            ethical_scaling: GOSNN ethical scaling factor
            fusion_result: AAFE fusion result

        Returns:
            Combined anomaly score [0, 1]
        """
        if not component_scores:
            return 0.0

        # If we have AAFE fusion result, use it
        if fusion_result is not None:
            base_score = fusion_result.final_score
        else:
            # Simple weighted average using golden ratio
            weights = self._physics_config.fusion_weights
            if weights is None:
                # Default golden ratio weighting
                num_scores = len(component_scores)
                phi_weights = [PHI ** (-i) for i in range(num_scores)]
                total_weight = sum(phi_weights)
                weights = {
                    k: phi_weights[i] / total_weight
                    for i, k in enumerate(component_scores.keys())
                }

            base_score = sum(
                weights.get(k, 1.0 / len(component_scores)) * v
                for k, v in component_scores.items()
            )

        # Blend with 3R scores
        if self._physics_config.use_3r_enhancement:
            enhanced_score = (
                0.6 * base_score +
                0.25 * recursion_score +
                0.15 * resonance_score
            )
        else:
            enhanced_score = base_score

        # Apply ethical scaling
        if self._physics_config.use_gosnn_scaling:
            # Ethical scaling reduces false positives by requiring higher confidence
            final_score = enhanced_score * (ethical_scaling ** PHI)
        else:
            final_score = enhanced_score

        return float(np.clip(final_score, 0.0, 1.0))

    def _extract_combined_features(
        self,
        data: Any,
        data_type: str,
        spectral_result: dict[str, Any] | None,
        dynamics_result: dict[str, Any] | None,
        uiux_result: dict[str, Any] | None,
    ) -> torch.Tensor:
        """Extract and combine features from all detectors.

        Args:
            data: Original input data
            data_type: Type of input data
            spectral_result: Spectral detection result
            dynamics_result: Dynamics detection result
            uiux_result: UI/UX detection result

        Returns:
            Combined feature tensor
        """
        feature_parts = []

        # Extract spectral features
        if self._spectral_detector is not None and spectral_result is not None:
            try:
                if data_type in ["time_series", "mixed"]:
                    ts_data = data["time_series"] if isinstance(data, dict) else data
                    spectral_features = self._spectral_detector.extract_features(ts_data)
                    feature_parts.append(spectral_features)
            except Exception as e:
                logger.warning(f"Failed to extract spectral features: {e}")

        # Extract dynamics features
        if self._dynamics_detector is not None and dynamics_result is not None:
            try:
                if data_type in ["time_series", "mixed"]:
                    ts_data = data["time_series"] if isinstance(data, dict) else data
                    dynamics_features = self._dynamics_detector.extract_features(ts_data)
                    feature_parts.append(dynamics_features)
            except Exception as e:
                logger.warning(f"Failed to extract dynamics features: {e}")

        # Extract UI/UX features
        if self._uiux_detector is not None and uiux_result is not None:
            try:
                if data_type in ["interactions", "mixed"]:
                    int_data = data["interactions"] if isinstance(data, dict) else data
                    uiux_features = self._uiux_detector.extract_features(int_data)
                    feature_parts.append(uiux_features)
            except Exception as e:
                logger.warning(f"Failed to extract UI/UX features: {e}")

        if not feature_parts:
            return torch.zeros(1, 64)

        # Concatenate all features
        # Handle different batch sizes by taking first sample
        aligned_features = []
        for feat in feature_parts:
            if feat.dim() == 1:
                aligned_features.append(feat.unsqueeze(0))
            else:
                aligned_features.append(feat[:1])  # Take first sample

        combined = torch.cat(aligned_features, dim=-1)

        return combined

    def _generate_combined_recommendations(
        self,
        spectral_result: dict[str, Any] | None,
        dynamics_result: dict[str, Any] | None,
        uiux_result: dict[str, Any] | None,
    ) -> list[str]:
        """Generate combined recommendations from all detectors.

        Args:
            spectral_result: Spectral detection result
            dynamics_result: Dynamics detection result
            uiux_result: UI/UX detection result

        Returns:
            Combined list of recommendations
        """
        recommendations = []

        # Spectral recommendations
        if spectral_result is not None:
            diagnostic = spectral_result.get("diagnostic")
            if diagnostic is not None and hasattr(diagnostic, 'recommended_action'):
                if diagnostic.recommended_action != "No action required. Continue routine monitoring.":
                    recommendations.append(f"[Spectral] {diagnostic.recommended_action}")

        # Dynamics recommendations
        if dynamics_result is not None:
            descriptions = dynamics_result.get("descriptions", [])
            for desc in descriptions[:3]:  # Limit to 3
                recommendations.append(f"[Dynamics] {desc}")

            if dynamics_result.get("is_chaotic", False):
                recommendations.append("[Dynamics] System shows chaotic behavior - consider stabilization")

        # UI/UX recommendations
        if uiux_result is not None:
            uiux_recs = uiux_result.get("recommendations", [])
            for rec in uiux_recs[:3]:  # Limit to 3
                if rec != "No significant usability issues detected in this session.":
                    recommendations.append(f"[UI/UX] {rec}")

        if not recommendations:
            recommendations.append("No significant issues detected across all analysis domains.")

        return recommendations

    def get_detector_status(self) -> dict[str, bool]:
        """Get status of all component detectors.

        Returns:
            Dictionary mapping detector names to fitted status
        """
        return {
            "spectral_vibration": (
                self._spectral_detector is not None and
                self._spectral_detector.is_fitted()
            ),
            "acceleration_dynamics": (
                self._dynamics_detector is not None and
                self._dynamics_detector.is_fitted()
            ),
            "uiux_anomaly": (
                self._uiux_detector is not None and
                self._uiux_detector.is_fitted()
            ),
        }

    def get_gosnn_scalars(self) -> dict[str, float]:
        """Get all GOSNN scalars.

        Returns:
            Dictionary of scalar names to values
        """
        return self._gosnn.get_all_scalars()


# =============================================================================
# Factory Functions
# =============================================================================

def create_spectral_detector(config: dict[str, Any] | None = None) -> SpectralVibrationDetector:
    """Factory function to create spectral vibration detector.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured SpectralVibrationDetector
    """
    return SpectralVibrationDetector(config)


def create_dynamics_detector(config: dict[str, Any] | None = None) -> AccelerationDynamicsDetector:
    """Factory function to create acceleration dynamics detector.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured AccelerationDynamicsDetector
    """
    return AccelerationDynamicsDetector(config)


def create_uiux_detector(config: dict[str, Any] | None = None) -> UIUXAnomalyDetector:
    """Factory function to create UI/UX anomaly detector.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured UIUXAnomalyDetector
    """
    return UIUXAnomalyDetector(config)


def create_integrated_detector(config: dict[str, Any] | None = None) -> AdvancedPhysicsIntegratedDetector:
    """Factory function to create integrated physics detector.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured AdvancedPhysicsIntegratedDetector
    """
    return AdvancedPhysicsIntegratedDetector(config)
