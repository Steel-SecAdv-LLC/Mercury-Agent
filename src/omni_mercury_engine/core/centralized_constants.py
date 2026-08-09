# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Centralized Constants and Magic Numbers.

Consolidates magic numbers, thresholds, and configuration constants
from across the codebase into a single, documented source of truth.

This module addresses P2 magic number extraction issues by:
1. Documenting the origin and purpose of each constant
2. Providing domain-specific configuration groups
3. Enabling environment variable overrides
4. Providing type-safe access via dataclasses
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

# ==============================================================================
# MATHEMATICAL CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class MathConstants:
    """Mathematical constants used throughout the system."""

    # Golden Ratio (φ) - Used in OAE weights, harmonic analysis
    # Origin: core/three_r_mechanism.py, global_omni_scalar_network.py
    GOLDEN_RATIO: float = 1.618033988749895

    # Euler's number
    # Origin: Various statistical computations
    EULER: float = 2.718281828459045

    # Pi
    PI: float = 3.141592653589793

    # Square root of 2 - Used in quantum normalization
    SQRT_2: float = 1.4142135623730951

    # Epsilon for numerical stability
    # Origin: Multiple files for division safety
    EPSILON: float = 1e-8
    EPSILON_LARGE: float = 1e-6
    EPSILON_SMALL: float = 1e-10


MATH = MathConstants()

# ==============================================================================
# LYAPUNOV STABILITY CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class LyapunovConstants:
    """Lyapunov stability framework constants."""

    # Controls exponential decay rate V(S_t) <= ε * e^(-λt) for OAE fusion
    # score stability. Intentionally faster (0.25) than LAMBDA_DECAY (0.18)
    # in double_helix_engine.py, which controls evolutionary adaptation speed.
    # The convergence bound must be tighter than the adaptation rate for
    # the system to stabilize.
    # See also: core/double_helix_engine.py:LAMBDA_DECAY = 0.18
    LAMBDA_CONVERGENCE: float = 0.25

    # Initial bound (ε)
    # Origin: core/three_r/fusion.py
    EPSILON_INITIAL: float = 1.0

    # Stability verification window size
    # Origin: core/three_r/fusion.py:174
    STABILITY_WINDOW: int = 10

    # Minimum convergence rate ratio for stability
    # Origin: core/three_r_mechanism.py:302
    MIN_CONVERGENCE_RATIO: float = 0.5


LYAPUNOV = LyapunovConstants()

# ==============================================================================
# ETHICAL GOVERNANCE CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class EthicalConstants:
    """Ethical governance thresholds and constants."""

    # Sigma Immutable - Default ethical compliance threshold
    # Origin: global_omni_scalar_network.py, three_r/types.py
    # Purpose: Minimum ethical score for system operations
    #
    # NOTE - two distinct sigma_Immutable thresholds exist BY DESIGN, and
    # this class is the single authoritative source for both:
    #   * SIGMA_IMMUTABLE_DEFAULT (0.96) - the GOSNN gating default used by
    #     get_sigma_immutable_threshold() for domain-tuned ethical gating
    #     (stricter; ~10-15% false-positive reduction).
    #   * SIGMA_IMMUTABLE_TRAINED_THRESHOLD (0.93) - the trained
    #     sigma_Immutable network's calibrated decision threshold: the
    #     ethical-band lower bound of the training labelling rule in
    #     scripts/train_sigma_immutable.py.  Used by SigmaImmutableGate /
    #     EthicalGate defaults and by the deterministic
    #     CRITICAL_ETHICAL_FLOOR derived from it.
    # They are not a drift bug; 0.93 is what the network was trained
    # against, 0.96 is the stricter operational gating point layered on
    # top.  README "Mathematical Rigor" documents both.
    SIGMA_IMMUTABLE_DEFAULT: float = 0.96

    # Sigma Immutable - Trained-network decision threshold (see NOTE above).
    # Origin: scripts/train_sigma_immutable.py ethical-band lower bound.
    # Numerically equal to SIGMA_IMMUTABLE_MEDICAL but semantically distinct:
    # changing the medical calibration must not silently move the trained
    # network's decision threshold, so the two are declared separately.
    SIGMA_IMMUTABLE_TRAINED_THRESHOLD: float = 0.93

    # Sigma Immutable - Medical domain (lower for medical urgency)
    # Origin: Domain-specific calibration
    SIGMA_IMMUTABLE_MEDICAL: float = 0.93

    # Sigma Immutable - Infrastructure (highest for critical systems)
    # Origin: Domain-specific calibration
    SIGMA_IMMUTABLE_INFRASTRUCTURE: float = 0.995

    # Sigma Immutable - Humanitarian
    # Origin: Domain-specific calibration
    SIGMA_IMMUTABLE_HUMANITARIAN: float = 0.95

    # Configured magnitude of the GOSNN ``omnibenevolence`` omni-scalar.
    #
    # This is a SCALAR VALUE, not a pass-bar. It used to be
    # ``BENEVOLENCE_IMMUTABLE`` and was read in two incompatible ways: as the
    # omni-scalar's setting (correct) and as a >= 0.99 decision-boundary
    # threshold every action had to clear (deleted). The threshold reading is
    # gone: benevolence is advisory everywhere, and the enforced
    # decision-boundary control is the harm-uplift gate
    # (``cognitive.decision_gate.enforce_decision_boundary``, see
    # ``docs/HARM_POLICY.md``). The rename exists so the two readings can never
    # be confused again.
    # Origin: ethical_alignment_engine.py
    OMNIBENEVOLENCE_SCALAR: float = 0.99

    # Minimum ethical alignment score
    # Origin: ethical_governor.py
    ETHICAL_MINIMUM: float = 0.6

    # Sigma Directive weighted score threshold
    # Origin: ethical_governor.py:129
    SIGMA_DIRECTIVE_THRESHOLD: float = 0.8

    # Demographic parity bias detection threshold
    # Origin: ethical_governor.py:351
    BIAS_DETECTION_THRESHOLD: float = 0.1

    # Pattern detection geometric threshold
    # Origin: ethical_alignment_engine.py
    PATTERN_DETECTION_THRESHOLD: float = 0.6

    # Ruleset version — used by the benevolence-decision cache
    # (cognitive/benevolence_cache.py) as part of the cache key. Bumping
    # this constant atomically invalidates every cached decision so a
    # change to the ethical ruleset cannot be served from a stale cache.
    # Format: monotonically increasing integer; bump on any semantic change
    # to scoring weights, principle definitions, or threshold floors.
    # v2: benevolence scorer gains morphological (char-trigram) harm matching + a
    # severity x irreversibility damping (fail-closed), so cached v1 verdicts must
    # invalidate.
    # v3: benevolence scorer gains a curated euphemism/paraphrase harm lexicon
    # (meaning-level: "put him down") + an optional pluggable harm classifier, both
    # fail-closed (can only RAISE harm); cached v2 verdicts must invalidate.
    # v4: benevolence scorer gains the two-axis (hazard-domain x operational-
    # intent) weapons/mass-casualty uplift gate (assess_weapons_uplift in
    # cognitive/ethical_bounding.py, see docs/HARM_POLICY.md) -- a blocking
    # disposition now raises PHYSICAL/SOCIETAL harm and hard-vetoes
    # is_permissible; cached v3 verdicts must invalidate.
    RULESET_VERSION: int = 4


ETHICAL = EthicalConstants()

# ==============================================================================
# SOFT SIGMOID BENEVOLENCE WEIGHTING (Phase 3)
#
# Disambiguation — neither of Mercury's two benevolence mechanisms is a gate:
#   * ADVISORY score: ``BenevolenceScorer.score_action``
#     (cognitive/ethical_bounding.py) reports a benevolence float and flags it
#     against an advisory reporting threshold. It approves nothing and blocks
#     nothing. The mandatory decision-boundary control is the harm-uplift gate
#     (``cognitive.decision_gate.enforce_decision_boundary`` /
#     ``assess_weapons_uplift``, see docs/HARM_POLICY.md), which raises
#     EthicalConstraintViolationError with ``check="harm_uplift"``.
#   * SOFT weighting: ``sigmoid_benevolence_gate`` below — a smooth η(b)
#     multiplier used inside score fusion (core/three_r/fusion.py) where a
#     hard step would create a discontinuity in the fused score.  Fusion
#     weighting only; it cannot approve or veto an action.
# ==============================================================================


@dataclass(frozen=True)
class BenevolenceDomainProfile:
    """Soft sigmoid benevolence weighting parameters for a specific domain.

    Inside score *fusion*, the smooth sigmoid curve is used instead of the
    hard benevolence threshold (≥ 0.99), which remains enforced separately
    at every decision boundary by ``BenevolenceScorer.enforce``:

        η(b) = 1 / (1 + exp(-k · (b - b₀)))

    Where:
        b₀ = inflection point (domain-specific)
        k  = steepness parameter (domain-specific)

    This eliminates the discontinuity at the threshold boundary while
    maintaining strict gating behavior for high-stakes domains.

    Provenance: Logistic function (Verhulst, 1845). Parameters calibrated
    per domain based on operational risk tolerance.
    """

    b0: float  # Inflection point
    k: float  # Steepness
    label: str = ""  # Human-readable domain name


@dataclass(frozen=True)
class BenevolenceGateConstants:
    """Domain-specific sigmoid benevolence gate profiles.

    All domain fallbacks derive from a SINGLE sigmoid function:
        η(b) = 1 / (1 + exp(-k · (b - b₀)))

    No separate hardcoded fallback values anywhere — one equation, one behavior.
    """

    # Domain profiles: (b₀, k, label)
    # b₀ = inflection point, k = steepness
    MEDICAL: BenevolenceDomainProfile = BenevolenceDomainProfile(b0=0.93, k=30.0, label="Medical")
    SECURITY: BenevolenceDomainProfile = BenevolenceDomainProfile(b0=0.95, k=25.0, label="Security")
    ENVIRONMENTAL: BenevolenceDomainProfile = BenevolenceDomainProfile(
        b0=0.90, k=20.0, label="Environmental"
    )
    HUMANITARIAN: BenevolenceDomainProfile = BenevolenceDomainProfile(
        b0=0.92, k=35.0, label="Humanitarian"
    )
    INFRASTRUCTURE: BenevolenceDomainProfile = BenevolenceDomainProfile(
        b0=0.94, k=25.0, label="Infrastructure"
    )
    DEFAULT: BenevolenceDomainProfile = BenevolenceDomainProfile(b0=0.93, k=25.0, label="Default")


BENEVOLENCE_GATE = BenevolenceGateConstants()


def sigmoid_benevolence_gate(
    benevolence_score: float,
    domain: str = "default",
) -> float:
    """Compute the SOFT sigmoid benevolence weighting value.

    Smooth fusion-weighting term (see the disambiguation note above):
        η(b) = 1 / (1 + exp(-k · (b - b₀)))

    There is **no hard ``≥ 0.99`` benevolence threshold** for this to be the
    smooth counterpart of. That pass-bar was deleted: it scored a fixed string
    the engine wrote for itself, so it refused benign work for having plain
    vocabulary and admitted anything phrased positively.
    ``BenevolenceScorer.enforce`` no longer applies it. The enforced control at
    every decision boundary is the fail-closed harm-uplift gate in
    ``cognitive/decision_gate.py``; benevolence is advisory.

    Args:
        benevolence_score: Raw benevolence score in [0, 1].
        domain: Domain name for profile selection.

    Returns:
        Gate value in (0, 1). Values near 1.0 indicate full ethical
        compliance; values near 0.0 indicate ethical violation.

    Provenance:
        Logistic function (Verhulst, 1845). Domain profiles from
        operational risk analysis.
    """
    domain_lower = domain.lower()
    profiles = {
        "medical": BENEVOLENCE_GATE.MEDICAL,
        "security": BENEVOLENCE_GATE.SECURITY,
        "environmental": BENEVOLENCE_GATE.ENVIRONMENTAL,
        "humanitarian": BENEVOLENCE_GATE.HUMANITARIAN,
        "infrastructure": BENEVOLENCE_GATE.INFRASTRUCTURE,
    }
    profile = profiles.get(domain_lower, BENEVOLENCE_GATE.DEFAULT)

    # Clip input to prevent overflow in exp()
    exponent = -profile.k * (benevolence_score - profile.b0)
    exponent = max(-500.0, min(500.0, exponent))  # Prevent overflow

    return 1.0 / (1.0 + math.exp(exponent))


# ==============================================================================
# DOMAIN-ADAPTIVE HARMONIC FREQUENCIES (Phase 3)
# ==============================================================================


@dataclass(frozen=True)
class DomainHarmonicConstants:
    """Domain-specific fundamental frequencies for harmonic analysis.

    Replaces the universal Schumann resonance (7.83 Hz) with
    domain-appropriate frequencies. For domains with unknown
    fundamentals (None), adaptive spectral peak detection is used.

    Provenance:
        - Environmental: Schumann resonances (Schumann, 1952)
        - Medical: HRV frequency bands (Task Force, 1996)
        - Infrastructure: Power grid + structural resonance
        - Space: Solar cycle + orbital mechanics
    """

    ENVIRONMENTAL: tuple[float, ...] = (7.83, 14.3, 20.8, 27.3, 33.8)
    MEDICAL: tuple[float, ...] = (0.04, 0.15, 0.4, 1.0, 40.0)
    INFRASTRUCTURE: tuple[float, ...] = (50.0, 60.0, 0.1, 0.01)
    SPACE: tuple[float, ...] = (0.001, 0.01, 0.1, 11.0)
    HUMANITARIAN: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    # Security and Financial use adaptive detection (no predefined fundamentals)


DOMAIN_HARMONICS = DomainHarmonicConstants()


def get_domain_fundamentals(domain: str) -> tuple[float, ...] | None:
    """Get fundamental frequencies for a domain.

    Args:
        domain: Domain name.

    Returns:
        Tuple of fundamental frequencies in Hz, or None if adaptive
        detection should be used (MUSIC/ESPRIT algorithm).
    """
    domain_lower = domain.lower()
    mapping: dict[str, tuple[float, ...] | None] = {
        "environmental": DOMAIN_HARMONICS.ENVIRONMENTAL,
        "medical": DOMAIN_HARMONICS.MEDICAL,
        "infrastructure": DOMAIN_HARMONICS.INFRASTRUCTURE,
        "space": DOMAIN_HARMONICS.SPACE,
        "security": None,  # Auto-detect via MUSIC/ESPRIT
        "financial": None,  # Auto-detect via MUSIC/ESPRIT
        "humanitarian": DOMAIN_HARMONICS.HUMANITARIAN,
    }
    return mapping.get(domain_lower, DOMAIN_HARMONICS.ENVIRONMENTAL)


# ==============================================================================
# RECURSION CONVERGENCE BOUNDS (Phase 3)
# ==============================================================================


@dataclass(frozen=True)
class RecursionConvergenceConstants:
    """Convergence bounds for recursive computations.

    Implements Banach contraction mapping constraints:
        d(R(x), R(y)) ≤ α · d(x, y) with α < 1

    This guarantees:
        - Unique fixed point existence
        - Geometric convergence
        - Computable error bounds: err ≤ α^d · ‖x₀ - R(x₀)‖ / (1 - α)

    Provenance: Banach fixed-point theorem (Banach, 1922).
    """

    # Maximum contraction factor (alpha_max)
    # Constraining alpha < 1 guarantees convergence
    ALPHA_MAX: float = 0.95

    # Recommended operating range for alpha
    ALPHA_MIN_RECOMMENDED: float = 0.5
    ALPHA_MAX_RECOMMENDED: float = 0.85

    # Maximum recursion depth before forced termination
    MAX_DEPTH: int = 50

    # Convergence tolerance — stop when change < this
    CONVERGENCE_TOLERANCE: float = 1e-6

    # Contraction violation threshold — halt if exceeded
    CONTRACTION_VIOLATION_THRESHOLD: float = 1.0


RECURSION = RecursionConvergenceConstants()

# ==============================================================================
# ANOMALY DETECTION CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class AnomalyDetectionConstants:
    """Anomaly detection thresholds and parameters."""

    # Default classification threshold
    # Origin: neurosymbolic_hub.py:810, score_calibration.py
    DEFAULT_THRESHOLD: float = 0.5

    # Maximum threshold cap
    # Origin: score_calibration.py
    MAX_THRESHOLD_CAP: float = 0.95

    # Minimum threshold floor
    # Origin: score_calibration.py
    MIN_THRESHOLD_FLOOR: float = 0.01

    # Default contamination estimate
    # Origin: Various detectors
    DEFAULT_CONTAMINATION: float = 0.05

    # Z-score threshold for outlier detection
    # Origin: api/server.py, statistical detectors
    ZSCORE_DEFAULT_THRESHOLD: float = 3.0

    # IQR multiplier for outlier detection
    # Origin: feature_pipeline.py, statistical modules
    IQR_MULTIPLIER: float = 1.5

    # Percentile for outlier detection
    # Origin: Various modules
    OUTLIER_PERCENTILE: float = 95.0

    # MAD (Median Absolute Deviation) multiplier
    # Origin: score_calibration.py
    MAD_MULTIPLIER: float = 3.0


ANOMALY = AnomalyDetectionConstants()

# ==============================================================================
# FUSION WEIGHTS CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class FusionConstants:
    """Fusion and weighting constants."""

    # OAE default weights — canonical PHI:1:1 derivation (the single source of
    # truth; matches core/three_r/fusion.py:OmniAvaEquation, ml/three_r_attention.py,
    # and tools/oae_weight_certifier.py).  Recursion carries the φ-weighted share;
    # Harmonic and Optimization receive equal unit shares.
    # w_R: Recursion, w_H: Harmonic, w_O: Optimization
    # phi_sum = φ + 2 ≈ 3.6180  (NOT φ + 1 + 1/φ — an earlier draft used the
    # latter and silently drifted to (0.5, 0.309, 0.191); oae_weight_certifier
    # is the gate against a recurrence.)
    OAE_WEIGHT_R: float = 0.447214  # φ / (φ + 2)
    OAE_WEIGHT_H: float = 0.276393  # 1 / (φ + 2)
    OAE_WEIGHT_O: float = 0.276393  # 1 / (φ + 2)
    # Backward compatibility
    AAFE_WEIGHT_R: float = OAE_WEIGHT_R
    AAFE_WEIGHT_H: float = OAE_WEIGHT_H
    AAFE_WEIGHT_O: float = OAE_WEIGHT_O

    # Neural-symbolic fusion weights
    # Origin: neurosymbolic_fusion.py, neurosymbolic_hub.py
    NEURAL_WEIGHT: float = 0.6
    SYMBOLIC_WEIGHT: float = 0.4

    # Attention fusion parameters
    # Origin: core/fusion.py
    ATTENTION_HEADS_DEFAULT: int = 8
    ATTENTION_HIDDEN_DIM: int = 128
    ATTENTION_DROPOUT: float = 0.1

    # Ensemble averaging weight decay
    # Origin: Various ensemble modules
    ENSEMBLE_DECAY: float = 0.9


FUSION = FusionConstants()

# ==============================================================================
# CONFIDENCE BANDS
# ==============================================================================


@dataclass(frozen=True)
class ConfidenceConstants:
    """Confidence classification thresholds."""

    # High confidence - definitive classifications
    # Origin: truth_decipher.py, various classifiers
    HIGH: float = 0.9

    # Medium confidence - probable classifications
    MEDIUM: float = 0.7

    # Low confidence - possible classifications
    LOW: float = 0.5

    # Minimum confidence for actions
    MINIMUM_ACTIONABLE: float = 0.3


CONFIDENCE = ConfidenceConstants()

# ==============================================================================
# DOMAIN-SPECIFIC CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class MedicalDomainConstants:
    """Medical domain-specific constants."""

    # Vital sign normal ranges (adult)
    HEART_RATE_MIN: float = 60.0
    HEART_RATE_MAX: float = 100.0
    SYSTOLIC_BP_MIN: float = 90.0
    SYSTOLIC_BP_MAX: float = 140.0
    DIASTOLIC_BP_MIN: float = 60.0
    DIASTOLIC_BP_MAX: float = 90.0
    RESPIRATORY_RATE_MIN: float = 12.0
    RESPIRATORY_RATE_MAX: float = 20.0
    OXYGEN_SATURATION_MIN: float = 95.0
    OXYGEN_SATURATION_MAX: float = 100.0
    TEMPERATURE_MIN: float = 36.1  # Celsius
    TEMPERATURE_MAX: float = 37.8  # Celsius
    MAP_MIN: float = 70.0  # Mean Arterial Pressure
    MAP_MAX: float = 105.0

    # SOFA score weights
    SOFA_RESPIRATORY: float = 0.20
    SOFA_COAGULATION: float = 0.15
    SOFA_LIVER: float = 0.15
    SOFA_CARDIOVASCULAR: float = 0.20
    SOFA_CNS: float = 0.15
    SOFA_RENAL: float = 0.15

    # Alert fatigue prevention window (seconds)
    ALERT_FATIGUE_WINDOW: int = 300


MEDICAL = MedicalDomainConstants()


@dataclass(frozen=True)
class FinancialDomainConstants:
    """Financial domain-specific constants."""

    # Benford's Law expected first digit distribution
    # P(d) = log10(1 + 1/d) for d in 1-9
    BENFORD_1: float = 0.301
    BENFORD_2: float = 0.176
    BENFORD_3: float = 0.125
    BENFORD_4: float = 0.097
    BENFORD_5: float = 0.079
    BENFORD_6: float = 0.067
    BENFORD_7: float = 0.058
    BENFORD_8: float = 0.051
    BENFORD_9: float = 0.046

    # Transaction velocity windows
    VELOCITY_WINDOW_SHORT: int = 10
    VELOCITY_WINDOW_MEDIUM: int = 50
    VELOCITY_WINDOW_LONG: int = 100

    # Seasonality periods (in transaction counts or days)
    SEASONALITY_DAILY: int = 1
    SEASONALITY_WEEKLY: int = 7
    SEASONALITY_MONTHLY: int = 30
    SEASONALITY_YEARLY: int = 365


FINANCIAL = FinancialDomainConstants()


@dataclass(frozen=True)
class InfrastructureConstants:
    """Infrastructure domain-specific constants."""

    # SCADA process variable correlation threshold
    CORRELATION_THRESHOLD: float = 0.7

    # Lagged correlation analysis windows
    LAG_WINDOW_1: int = 1
    LAG_WINDOW_5: int = 5
    LAG_WINDOW_10: int = 10
    LAG_WINDOW_30: int = 30

    # Alarm threshold multiplier (standard deviations)
    ALARM_THRESHOLD_MULTIPLIER: float = 2.0

    # Control loop stability thresholds
    OSCILLATION_INDEX_THRESHOLD: float = 0.3
    STABILITY_SCORE_THRESHOLD: float = 0.1


INFRASTRUCTURE = InfrastructureConstants()

# ==============================================================================
# NEURAL NETWORK CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class NeuralNetConstants:
    """Neural network architecture constants."""

    # Transformer defaults
    # Origin: cognitive/neural_memory_layer.py
    TRANSFORMER_D_MODEL: int = 128
    TRANSFORMER_N_HEADS: int = 8
    TRANSFORMER_D_FF: int = 512
    TRANSFORMER_N_LAYERS: int = 6
    TRANSFORMER_DROPOUT: float = 0.1

    # LSTM defaults
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.2

    # CNN defaults
    CNN_BASE_CHANNELS: int = 64
    CNN_KERNEL_SIZE: int = 3
    CNN_POOL_SIZE: int = 2

    # Training defaults
    LEARNING_RATE_DEFAULT: float = 0.001
    WEIGHT_DECAY_DEFAULT: float = 0.0001
    BATCH_SIZE_DEFAULT: int = 32
    MAX_EPOCHS_DEFAULT: int = 100
    EARLY_STOPPING_PATIENCE: int = 10


NEURAL = NeuralNetConstants()

# ==============================================================================
# CALIBRATION CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class CalibrationConstants:
    """Score calibration constants."""

    # Platt scaling defaults
    PLATT_MAX_ITER: int = 100
    PLATT_TOLERANCE: float = 1e-6

    # Isotonic regression
    ISOTONIC_MIN_SAMPLES: int = 10

    # Threshold optimization
    THRESHOLD_GRID_SIZE: int = 50
    THRESHOLD_CV_FOLDS: int = 5

    # Gaussian Mixture Model
    GMM_N_COMPONENTS: int = 2
    GMM_MAX_ITER: int = 100

    # Adaptive threshold adjustment
    ADAPTIVE_HISTORY_WEIGHT: float = 0.3
    ADAPTIVE_WINDOW_SIZE: int = 1000


CALIBRATION = CalibrationConstants()

# ==============================================================================
# API AND VALIDATION CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class APIConstants:
    """API and validation constants."""

    # Size limits
    MAX_DATA_POINTS: int = 100000
    MAX_FEATURES: int = 1000
    MAX_STRING_LENGTH: int = 256
    MAX_ARRAY_DEPTH: int = 3

    # Value limits
    MIN_VALUE: float = -1e15
    MAX_VALUE: float = 1e15
    MAX_NAN_RATIO: float = 0.1
    MAX_INF_RATIO: float = 0.01

    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20

    # Timeouts (seconds)
    REQUEST_TIMEOUT: int = 30
    DETECTION_TIMEOUT: int = 60


API = APIConstants()

# ==============================================================================
# SLIDING WINDOW CONSTANTS
# ==============================================================================


@dataclass(frozen=True)
class SlidingWindowConstants:
    """Sliding window normalization constants."""

    # Default window sizes
    WINDOW_SIZE_SMALL: int = 50
    WINDOW_SIZE_MEDIUM: int = 100
    WINDOW_SIZE_LARGE: int = 500

    # Minimum samples for statistics
    MIN_SAMPLES: int = 10

    # Exponential decay factor
    DECAY_FACTOR: float = 0.99

    # Update frequency
    UPDATE_FREQUENCY: int = 1


SLIDING_WINDOW = SlidingWindowConstants()

# ==============================================================================
# UTILITY CLASSES
# ==============================================================================


class ConstantRegistry:
    """Registry for accessing and overriding constants.

    Allows environment variable overrides for all constants.
    """

    _overrides: dict[str, Any] = {}

    @classmethod
    def get(cls, group: str, name: str, default: Any = None) -> Any:
        """Get a constant value with environment override support.

        Environment variable format: MERCURY_{GROUP}_{NAME}

        Args:
            group: Constant group (e.g., "ETHICAL", "ANOMALY")
            name: Constant name (e.g., "SIGMA_IMMUTABLE_DEFAULT")
            default: Default value if not found

        Returns:
            Constant value
        """
        # Check environment override
        env_key = f"MERCURY_{group.upper()}_{name.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return cls._parse_value(env_value, default)

        # Check programmatic override
        override_key = f"{group}.{name}"
        if override_key in cls._overrides:
            return cls._overrides[override_key]

        return default

    @classmethod
    def set(cls, group: str, name: str, value: Any) -> None:
        """Set a programmatic override for a constant.

        Args:
            group: Constant group
            name: Constant name
            value: New value
        """
        cls._overrides[f"{group}.{name}"] = value

    @classmethod
    def _parse_value(cls, value: str, reference: Any) -> Any:
        """Parse string value based on reference type."""
        if isinstance(reference, bool):
            return value.lower() in ("true", "1", "yes")
        elif isinstance(reference, int):
            return int(value)
        elif isinstance(reference, float):
            return float(value)
        return value


def get_domain_constants(domain: str) -> dict[str, Any]:
    """Get all constants for a specific domain.

    Args:
        domain: Domain name (medical, financial, infrastructure)

    Returns:
        Dictionary of domain-specific constants
    """
    domain = domain.lower()

    if domain == "medical":
        return {
            "sigma_immutable": ETHICAL.SIGMA_IMMUTABLE_MEDICAL,
            "vital_ranges": {
                "heart_rate": (MEDICAL.HEART_RATE_MIN, MEDICAL.HEART_RATE_MAX),
                "systolic_bp": (MEDICAL.SYSTOLIC_BP_MIN, MEDICAL.SYSTOLIC_BP_MAX),
                "diastolic_bp": (MEDICAL.DIASTOLIC_BP_MIN, MEDICAL.DIASTOLIC_BP_MAX),
                "respiratory_rate": (MEDICAL.RESPIRATORY_RATE_MIN, MEDICAL.RESPIRATORY_RATE_MAX),
                "oxygen_saturation": (MEDICAL.OXYGEN_SATURATION_MIN, MEDICAL.OXYGEN_SATURATION_MAX),
                "temperature": (MEDICAL.TEMPERATURE_MIN, MEDICAL.TEMPERATURE_MAX),
            },
            "sofa_weights": {
                "respiratory": MEDICAL.SOFA_RESPIRATORY,
                "coagulation": MEDICAL.SOFA_COAGULATION,
                "liver": MEDICAL.SOFA_LIVER,
                "cardiovascular": MEDICAL.SOFA_CARDIOVASCULAR,
                "cns": MEDICAL.SOFA_CNS,
                "renal": MEDICAL.SOFA_RENAL,
            },
        }

    elif domain == "financial":
        return {
            "sigma_immutable": ETHICAL.SIGMA_IMMUTABLE_DEFAULT,
            "benford_distribution": [
                FINANCIAL.BENFORD_1,
                FINANCIAL.BENFORD_2,
                FINANCIAL.BENFORD_3,
                FINANCIAL.BENFORD_4,
                FINANCIAL.BENFORD_5,
                FINANCIAL.BENFORD_6,
                FINANCIAL.BENFORD_7,
                FINANCIAL.BENFORD_8,
                FINANCIAL.BENFORD_9,
            ],
            "velocity_windows": [
                FINANCIAL.VELOCITY_WINDOW_SHORT,
                FINANCIAL.VELOCITY_WINDOW_MEDIUM,
                FINANCIAL.VELOCITY_WINDOW_LONG,
            ],
            "seasonality_periods": [
                FINANCIAL.SEASONALITY_WEEKLY,
                FINANCIAL.SEASONALITY_MONTHLY,
                FINANCIAL.SEASONALITY_YEARLY,
            ],
        }

    elif domain == "infrastructure":
        return {
            "sigma_immutable": ETHICAL.SIGMA_IMMUTABLE_INFRASTRUCTURE,
            "correlation_threshold": INFRASTRUCTURE.CORRELATION_THRESHOLD,
            "lag_windows": [
                INFRASTRUCTURE.LAG_WINDOW_1,
                INFRASTRUCTURE.LAG_WINDOW_5,
                INFRASTRUCTURE.LAG_WINDOW_10,
                INFRASTRUCTURE.LAG_WINDOW_30,
            ],
            "alarm_threshold_multiplier": INFRASTRUCTURE.ALARM_THRESHOLD_MULTIPLIER,
        }

    else:
        return {
            "sigma_immutable": ETHICAL.SIGMA_IMMUTABLE_DEFAULT,
            "default_threshold": ANOMALY.DEFAULT_THRESHOLD,
            "contamination": ANOMALY.DEFAULT_CONTAMINATION,
        }


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "ANOMALY",
    "API",
    "BENEVOLENCE_GATE",
    "CALIBRATION",
    "CONFIDENCE",
    "DOMAIN_HARMONICS",
    "ETHICAL",
    "FINANCIAL",
    "FUSION",
    "INFRASTRUCTURE",
    "LYAPUNOV",
    "MATH",
    "MEDICAL",
    "NEURAL",
    "RECURSION",
    "SLIDING_WINDOW",
    "APIConstants",
    "AnomalyDetectionConstants",
    "BenevolenceDomainProfile",
    "BenevolenceGateConstants",
    "CalibrationConstants",
    "ConfidenceConstants",
    "ConstantRegistry",
    "DomainHarmonicConstants",
    "EthicalConstants",
    "FinancialDomainConstants",
    "FusionConstants",
    "InfrastructureConstants",
    "LyapunovConstants",
    "MathConstants",
    "MedicalDomainConstants",
    "NeuralNetConstants",
    "RecursionConvergenceConstants",
    "SlidingWindowConstants",
    "get_domain_constants",
    "get_domain_fundamentals",
    "sigmoid_benevolence_gate",
]
