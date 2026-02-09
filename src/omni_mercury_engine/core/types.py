"""
Mercury Agent
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

Canonical Type Definitions for Mercury Agent

This module provides canonical enum definitions used throughout the Mercury Agent
codebase. These enums consolidate duplicate definitions from various modules to
ensure consistency and reduce code duplication.

Usage:
    from omni_mercury_engine.core.types import (
        CircuitState,
        ThreatLevel,
        EthicalPrinciple,
        FusionStrategy,
        ConfidenceLevel,
        DetectorStatus,
        AnomalyType,
        PrivacyLevel,
    )

Note:
    When adding new shared enums, place them in this module rather than defining
    them locally in feature modules. This promotes code reuse and ensures
    consistent semantics across the codebase.
"""

from __future__ import annotations

from enum import Enum, auto

__all__ = [
    "AnomalyType",
    "CircuitState",
    "ConfidenceLevel",
    "DetectorStatus",
    "EthicalPrinciple",
    "FusionStrategy",
    "PrivacyLevel",
    "ThreatLevel",
]


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================


class CircuitState(Enum):
    """
    States for the circuit breaker pattern.

    The circuit breaker pattern prevents cascading failures by tracking the
    health of external dependencies and temporarily blocking requests when
    failures exceed a threshold.

    States:
        CLOSED: Normal operation - requests flow through normally.
        OPEN: Failure threshold exceeded - requests are rejected immediately.
        HALF_OPEN: Testing recovery - limited requests allowed to test if
                   the service has recovered.

    Example:
        >>> state = CircuitState.CLOSED
        >>> if state == CircuitState.OPEN:
        ...     raise ServiceUnavailableError("Circuit breaker is open")
    """

    CLOSED = auto()  # Normal operation - requests flow through
    OPEN = auto()  # Failing - reject calls immediately
    HALF_OPEN = auto()  # Testing if service has recovered


# =============================================================================
# Security and Threat Assessment
# =============================================================================


class ThreatLevel(Enum):
    """
    Security threat severity levels.

    Unified threat classification combining variants from security modules,
    intelligence fusion systems, and detector components. Provides a
    comprehensive scale from benign to critical threats.

    Levels (in ascending severity):
        NONE: No threat detected - baseline/normal state.
        UNKNOWN: Unable to determine threat level - requires investigation.
        LOW: Minimal risk - routine monitoring sufficient.
        MODERATE: Medium risk - increased monitoring recommended.
        SUBSTANTIAL: Elevated risk - active response may be needed.
        HIGH: Significant risk - immediate attention required.
        SEVERE: Major risk - urgent response required.
        CRITICAL: Extreme risk - immediate action essential.

    Aliases for compatibility:
        - MINIMAL maps to LOW
        - MEDIUM maps to MODERATE
        - ELEVATED maps to SUBSTANTIAL
        - EXTREME maps to CRITICAL
        - SAFE maps to NONE
        - BENIGN maps to NONE
        - ANOMALOUS maps to LOW
        - SUSPICIOUS maps to MODERATE
        - THREAT maps to HIGH

    Example:
        >>> level = ThreatLevel.HIGH
        >>> if level.value >= ThreatLevel.HIGH.value:
        ...     trigger_incident_response()
    """

    # Primary threat levels (ascending severity order)
    NONE = 0  # No threat / Safe / Benign
    UNKNOWN = 1  # Cannot determine threat level
    LOW = 2  # Minimal risk / Anomalous
    MODERATE = 3  # Medium risk / Suspicious
    SUBSTANTIAL = 4  # Elevated risk
    HIGH = 5  # Significant risk / Threat
    SEVERE = 6  # Major risk
    CRITICAL = 7  # Extreme risk / Maximum severity

    # Compatibility aliases (same values as primary levels)
    # These provide alternative semantic names for the same threat levels.
    # Using same value makes them aliases via Python Enum semantics.
    MINIMAL = 2  # Alias for LOW
    MEDIUM = 3  # Alias for MODERATE
    ELEVATED = 4  # Alias for SUBSTANTIAL
    EXTREME = 7  # Alias for CRITICAL
    SAFE = 0  # Alias for NONE
    BENIGN = 0  # Alias for NONE
    ANOMALOUS = 2  # Alias for LOW
    SUSPICIOUS = 3  # Alias for MODERATE
    THREAT = 5  # Alias for HIGH

    def is_actionable(self) -> bool:
        """Return True if threat level requires active response."""
        return self.value >= ThreatLevel.SUBSTANTIAL.value

    def is_critical(self) -> bool:
        """Return True if threat level is critical or severe."""
        return self.value >= ThreatLevel.SEVERE.value


# =============================================================================
# AI Ethics and Governance
# =============================================================================


class EthicalPrinciple(Enum):
    """
    Core ethical principles for AI alignment and governance.

    Comprehensive set of ethical principles combining:
    - The 8 core Mercury principles (Compassion, Evidence, Justice, Altruism,
      Control, Character, Competence, Commitment)
    - Standard AI ethics principles (Beneficence, Non-Maleficence, Autonomy)
    - Operational principles (Transparency, Privacy, Accountability, Safety,
      Fairness, Explainability)

    The 8 Core Mercury Principles:
        COMPASSION: Acting with empathy and care for affected parties.
        EVIDENCE: Decisions grounded in factual, verifiable information.
        JUSTICE: Fair and equitable treatment of all stakeholders.
        ALTRUISM: Prioritizing collective benefit over narrow interests.
        CONTROL: Maintaining human oversight and intervention capability.
        CHARACTER: Consistency in ethical behavior across contexts.
        COMPETENCE: Operating within validated capability boundaries.
        COMMITMENT: Persistent adherence to ethical standards.

    Standard AI Ethics Principles:
        BENEFICENCE: Actively doing good and promoting wellbeing.
        NON_MALEFICENCE: Avoiding causing harm (primum non nocere).
        AUTONOMY: Respecting individual agency and self-determination.

    Operational Principles:
        TRANSPARENCY: Openness about operations and decision-making.
        PRIVACY: Protecting personal and sensitive information.
        ACCOUNTABILITY: Clear responsibility for actions and outcomes.
        SAFETY: Ensuring operations do not cause unacceptable risk.
        FAIRNESS: Avoiding bias and discrimination.
        EXPLAINABILITY: Providing understandable justifications.

    Example:
        >>> principles = [EthicalPrinciple.BENEFICENCE, EthicalPrinciple.SAFETY]
        >>> for p in principles:
        ...     validate_against_principle(action, p)
    """

    # 8 Core Mercury Principles
    COMPASSION = "compassion"
    EVIDENCE = "evidence"
    JUSTICE = "justice"
    ALTRUISM = "altruism"
    CONTROL = "control"
    CHARACTER = "character"
    COMPETENCE = "competence"
    COMMITMENT = "commitment"

    # Standard AI Ethics Principles
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    AUTONOMY = "autonomy"

    # Operational Principles
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"
    ACCOUNTABILITY = "accountability"
    SAFETY = "safety"
    FAIRNESS = "fairness"
    EXPLAINABILITY = "explainability"

    @classmethod
    def core_principles(cls) -> list[EthicalPrinciple]:
        """Return the 8 core Mercury principles."""
        return [
            cls.COMPASSION,
            cls.EVIDENCE,
            cls.JUSTICE,
            cls.ALTRUISM,
            cls.CONTROL,
            cls.CHARACTER,
            cls.COMPETENCE,
            cls.COMMITMENT,
        ]

    @classmethod
    def ai_ethics_principles(cls) -> list[EthicalPrinciple]:
        """Return standard AI ethics principles."""
        return [cls.BENEFICENCE, cls.NON_MALEFICENCE, cls.AUTONOMY]

    @classmethod
    def operational_principles(cls) -> list[EthicalPrinciple]:
        """Return operational governance principles."""
        return [
            cls.TRANSPARENCY,
            cls.PRIVACY,
            cls.ACCOUNTABILITY,
            cls.SAFETY,
            cls.FAIRNESS,
            cls.EXPLAINABILITY,
        ]


# =============================================================================
# Multi-Modal Fusion Strategies
# =============================================================================


class FusionStrategy(Enum):
    """
    Strategies for multi-modal data fusion.

    Fusion strategies define how information from multiple sources, modalities,
    or models are combined to produce unified outputs. Strategies are organized
    by fusion level (early, late, hybrid) and technique.

    Early Fusion (Feature-Level):
        EARLY: Combine features before model processing.
        FEATURE_CONCAT: Concatenate feature vectors directly.
        FEATURE_ATTENTION: Use attention to weight feature combinations.

    Late Fusion (Decision-Level):
        LATE: Combine model outputs/decisions.
        SCORE_AVERAGE: Average prediction scores.
        SCORE_WEIGHTED: Weighted average of scores.
        SCORE_MAX: Take maximum score across models.
        DECISION_VOTING: Majority voting on decisions.
        DECISION_CONFIDENCE: Weight by confidence levels.
        SCORE_LEVEL: Score-level fusion for biometric matching.
        DECISION_LEVEL: Decision-level fusion for biometric matching.

    Hybrid Strategies:
        HYBRID: Combined early and late fusion.
        ATTENTION: Attention-based dynamic fusion.
        HIERARCHICAL: Multi-level hierarchical fusion.
        ADAPTIVE: Context-dependent strategy selection.
        GATED: Gating mechanism for fusion control.
        WEIGHTED_AVERAGE: Weighted combination of outputs.
        CONFIDENCE_WEIGHTED: Weight by prediction confidence.
        QUALITY_WEIGHTED: Weight by input quality metrics.

    Example:
        >>> strategy = FusionStrategy.ADAPTIVE
        >>> fused_result = fuse_modalities(inputs, strategy=strategy)
    """

    # Early Fusion (Feature-Level)
    EARLY = "early"
    FEATURE_CONCAT = "feature_concat"
    FEATURE_ATTENTION = "feature_attention"

    # Late Fusion (Decision-Level)
    LATE = "late"
    SCORE_AVERAGE = "score_average"
    SCORE_WEIGHTED = "score_weighted"
    SCORE_MAX = "score_max"
    DECISION_VOTING = "decision_voting"
    DECISION_CONFIDENCE = "decision_confidence"
    SCORE_LEVEL = "score_level"
    DECISION_LEVEL = "decision_level"

    # Hybrid and Advanced Strategies
    HYBRID = "hybrid"
    ATTENTION = "attention"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    GATED = "gated"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    QUALITY_WEIGHTED = "quality_weighted"

    def is_early_fusion(self) -> bool:
        """Return True if this is an early/feature-level fusion strategy."""
        return self in {
            FusionStrategy.EARLY,
            FusionStrategy.FEATURE_CONCAT,
            FusionStrategy.FEATURE_ATTENTION,
        }

    def is_late_fusion(self) -> bool:
        """Return True if this is a late/decision-level fusion strategy."""
        return self in {
            FusionStrategy.LATE,
            FusionStrategy.SCORE_AVERAGE,
            FusionStrategy.SCORE_WEIGHTED,
            FusionStrategy.SCORE_MAX,
            FusionStrategy.DECISION_VOTING,
            FusionStrategy.DECISION_CONFIDENCE,
            FusionStrategy.SCORE_LEVEL,
            FusionStrategy.DECISION_LEVEL,
        }

    def is_hybrid_fusion(self) -> bool:
        """Return True if this is a hybrid fusion strategy."""
        return self in {
            FusionStrategy.HYBRID,
            FusionStrategy.ATTENTION,
            FusionStrategy.HIERARCHICAL,
            FusionStrategy.ADAPTIVE,
            FusionStrategy.GATED,
            FusionStrategy.WEIGHTED_AVERAGE,
            FusionStrategy.CONFIDENCE_WEIGHTED,
            FusionStrategy.QUALITY_WEIGHTED,
        }


# =============================================================================
# Confidence and Certainty Levels
# =============================================================================


class ConfidenceLevel(Enum):
    """
    Confidence/certainty levels for predictions and assessments.

    Standardized confidence classification for transparency in AI outputs.
    Each level maps to an approximate probability range.

    Levels (with approximate probability ranges):
        VERY_LOW: <30% - High uncertainty, multiple hypotheses possible.
        LOW: 30-50% - Uncertain, verification recommended.
        MEDIUM: 50-70% - Reasonable confidence, typical prediction.
        HIGH: 70-90% - Strong evidence supports conclusion.
        VERY_HIGH: 90-99% - Near-certain, very strong evidence.
        CERTAIN: >99% - Deterministic or near-deterministic.

    Additional Classifications:
        UNCERTAIN: Alternative label for very low confidence.
        MODERATE: Alternative label for medium confidence.

    Example:
        >>> confidence = classify_confidence(0.85)
        >>> assert confidence == ConfidenceLevel.HIGH
        >>> if confidence.requires_verification():
        ...     request_human_review()
    """

    VERY_LOW = "very_low"  # <30% - High uncertainty
    LOW = "low"  # 30-50% - Uncertain
    MEDIUM = "medium"  # 50-70% - Reasonable confidence
    HIGH = "high"  # 70-90% - Strong evidence
    VERY_HIGH = "very_high"  # 90-99% - Near-certain
    CERTAIN = "certain"  # >99% - Deterministic

    # Compatibility aliases
    UNCERTAIN = "uncertain"  # Alias for VERY_LOW
    MODERATE = "moderate"  # Alias for MEDIUM

    @classmethod
    def from_probability(cls, probability: float) -> ConfidenceLevel:
        """
        Classify a probability value into a confidence level.

        Args:
            probability: Float between 0.0 and 1.0

        Returns:
            Appropriate ConfidenceLevel for the probability
        """
        if probability >= 0.99:
            return cls.CERTAIN
        elif probability >= 0.90:
            return cls.VERY_HIGH
        elif probability >= 0.70:
            return cls.HIGH
        elif probability >= 0.50:
            return cls.MEDIUM
        elif probability >= 0.30:
            return cls.LOW
        else:
            return cls.VERY_LOW

    def to_probability_range(self) -> tuple[float, float]:
        """Return the approximate probability range for this confidence level."""
        ranges = {
            ConfidenceLevel.VERY_LOW: (0.0, 0.30),
            ConfidenceLevel.UNCERTAIN: (0.0, 0.30),
            ConfidenceLevel.LOW: (0.30, 0.50),
            ConfidenceLevel.MEDIUM: (0.50, 0.70),
            ConfidenceLevel.MODERATE: (0.50, 0.70),
            ConfidenceLevel.HIGH: (0.70, 0.90),
            ConfidenceLevel.VERY_HIGH: (0.90, 0.99),
            ConfidenceLevel.CERTAIN: (0.99, 1.0),
        }
        return ranges.get(self, (0.0, 1.0))

    def requires_verification(self) -> bool:
        """Return True if this confidence level warrants human verification."""
        return self in {
            ConfidenceLevel.VERY_LOW,
            ConfidenceLevel.UNCERTAIN,
            ConfidenceLevel.LOW,
        }


# =============================================================================
# Detector Operational Status
# =============================================================================


class DetectorStatus(Enum):
    """
    Operational status for anomaly detectors and processing components.

    Tracks the lifecycle state of detector components from initialization
    through operation to completion or error states.

    States:
        IDLE: Detector initialized but not actively processing.
        INITIALIZING: Detector is loading models or configuration.
        RUNNING: Actively processing data and detecting anomalies.
        PAUSED: Temporarily suspended, can resume.
        STOPPING: Gracefully shutting down.
        COMPLETED: Processing finished successfully.
        ERROR: Failed state, requires intervention.
        DEGRADED: Operating with reduced capability.

    Example:
        >>> detector.status = DetectorStatus.RUNNING
        >>> if detector.status == DetectorStatus.ERROR:
        ...     logger.error("Detector failed, triggering recovery")
        ...     detector.restart()
    """

    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"
    DEGRADED = "degraded"

    def is_active(self) -> bool:
        """Return True if detector is actively processing."""
        return self in {
            DetectorStatus.RUNNING,
            DetectorStatus.DEGRADED,
        }

    def is_terminal(self) -> bool:
        """Return True if detector has reached a terminal state."""
        return self in {
            DetectorStatus.COMPLETED,
            DetectorStatus.ERROR,
        }

    def can_transition_to(self, target: DetectorStatus) -> bool:
        """Check if transition to target status is valid."""
        valid_transitions = {
            DetectorStatus.IDLE: {
                DetectorStatus.INITIALIZING,
                DetectorStatus.RUNNING,
            },
            DetectorStatus.INITIALIZING: {
                DetectorStatus.RUNNING,
                DetectorStatus.ERROR,
            },
            DetectorStatus.RUNNING: {
                DetectorStatus.PAUSED,
                DetectorStatus.STOPPING,
                DetectorStatus.COMPLETED,
                DetectorStatus.ERROR,
                DetectorStatus.DEGRADED,
            },
            DetectorStatus.PAUSED: {
                DetectorStatus.RUNNING,
                DetectorStatus.STOPPING,
                DetectorStatus.ERROR,
            },
            DetectorStatus.STOPPING: {
                DetectorStatus.IDLE,
                DetectorStatus.COMPLETED,
                DetectorStatus.ERROR,
            },
            DetectorStatus.DEGRADED: {
                DetectorStatus.RUNNING,
                DetectorStatus.ERROR,
                DetectorStatus.STOPPING,
            },
            DetectorStatus.COMPLETED: {
                DetectorStatus.IDLE,
            },
            DetectorStatus.ERROR: {
                DetectorStatus.IDLE,
            },
        }
        return target in valid_transitions.get(self, set())


# =============================================================================
# Anomaly Classification
# =============================================================================


class AnomalyType(Enum):
    """
    Types of anomalies detected by the system.

    Categorizes anomalies by their characteristics and detection methodology.
    Used to guide response strategies and provide context for alerts.

    Statistical/Temporal Types:
        POINT: Single data point significantly deviates from expected value.
        CONTEXTUAL: Normal value in wrong context (e.g., 90F in winter).
        COLLECTIVE: Group of points that are anomalous together.
        SEASONAL: Deviation from expected seasonal patterns.
        TREND: Unexpected change in underlying trend.

    Domain-Specific Types:
        BEHAVIORAL: Anomaly in user/entity behavior patterns.
        STRUCTURAL: Anomaly in system/data structure.
        TEMPORAL: Time-based anomaly (timing, sequence, duration).
        ETHICAL: Violation of ethical constraints or principles.

    Classification Types:
        NOVEL: Previously unseen pattern (out-of-distribution).
        DRIFT: Gradual change in data distribution.
        UNKNOWN: Cannot categorize - requires investigation.

    Example:
        >>> anomaly = detect_anomaly(data_point)
        >>> if anomaly.type == AnomalyType.COLLECTIVE:
        ...     investigate_correlated_events(anomaly.related_points)
    """

    # Statistical/Temporal Types
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    SEASONAL = "seasonal"
    TREND = "trend"

    # Domain-Specific Types
    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    ETHICAL = "ethical"

    # Classification Types
    NOVEL = "novel"
    DRIFT = "drift"
    UNKNOWN = "unknown"

    def requires_correlation_analysis(self) -> bool:
        """Return True if anomaly type requires analysis of related events."""
        return self in {
            AnomalyType.COLLECTIVE,
            AnomalyType.BEHAVIORAL,
            AnomalyType.DRIFT,
        }

    def is_temporal(self) -> bool:
        """Return True if anomaly has a temporal component."""
        return self in {
            AnomalyType.SEASONAL,
            AnomalyType.TREND,
            AnomalyType.TEMPORAL,
            AnomalyType.DRIFT,
        }


# =============================================================================
# Privacy Protection Levels
# =============================================================================


class PrivacyLevel(Enum):
    """
    Privacy protection levels for data processing and federated learning.

    Defines the level of privacy guarantees applied to data during processing,
    aggregation, and model training operations.

    Levels (in ascending protection order):
        NONE: No privacy protection - raw data exposed.
        BASIC: Basic anonymization (e.g., pseudonymization).
        DIFFERENTIAL_PRIVACY: Mathematical privacy guarantees via noise addition.
        SECURE_AGGREGATION: Cryptographic aggregation without exposing individual inputs.
        SMPC: Secure Multi-Party Computation - strongest computational guarantees.

    Note:
        Higher privacy levels typically incur performance and utility costs.
        Choose the minimum level necessary for compliance and trust requirements.

    Example:
        >>> privacy = PrivacyLevel.DIFFERENTIAL_PRIVACY
        >>> if privacy.requires_noise():
        ...     epsilon = calculate_privacy_budget()
        ...     data = add_laplace_noise(data, epsilon)
    """

    NONE = "none"
    BASIC = "basic"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    SMPC = "secure_multiparty_computation"

    def requires_noise(self) -> bool:
        """Return True if this privacy level requires noise addition."""
        return self == PrivacyLevel.DIFFERENTIAL_PRIVACY

    def requires_encryption(self) -> bool:
        """Return True if this privacy level requires cryptographic operations."""
        return self in {
            PrivacyLevel.SECURE_AGGREGATION,
            PrivacyLevel.SMPC,
        }

    def provides_mathematical_guarantee(self) -> bool:
        """Return True if this privacy level provides formal privacy guarantees."""
        return self in {
            PrivacyLevel.DIFFERENTIAL_PRIVACY,
            PrivacyLevel.SECURE_AGGREGATION,
            PrivacyLevel.SMPC,
        }

    @classmethod
    def for_compliance(cls, regulation: str) -> PrivacyLevel:
        """
        Suggest minimum privacy level for regulatory compliance.

        Args:
            regulation: Regulation identifier (e.g., "GDPR", "HIPAA", "CCPA")

        Returns:
            Minimum recommended PrivacyLevel for compliance
        """
        regulation_requirements = {
            "GDPR": cls.DIFFERENTIAL_PRIVACY,
            "HIPAA": cls.SECURE_AGGREGATION,
            "CCPA": cls.BASIC,
            "SOC2": cls.BASIC,
            "FEDRAMP": cls.SECURE_AGGREGATION,
        }
        return regulation_requirements.get(regulation.upper(), cls.BASIC)
