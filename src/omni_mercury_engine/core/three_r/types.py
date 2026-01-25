"""
Mercury Agent - 3R Mechanism Type Definitions
Copyright (C) 2025 Steel Security Advisory LLC

Type definitions, enums, and dataclasses for the 3R Mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AnomalyDetectionMethod(Enum):
    """Methods for detecting code anomalies."""

    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    MULTI_VARIATE = "multi_variate"


class IssueType(Enum):
    """Types of engineering issues in code."""

    BUG = "bug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CODE_QUALITY = "code_quality"
    COMPLEXITY = "complexity"
    LOGIC = "logic"


class IssueSeverity(Enum):
    """Severity levels for code issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EvolutionStrategy(Enum):
    """Evolution strategies for adaptive code improvement."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


# Golden ratio constant for AVA Anomaly Fusion Equation (AAFE)
GOLDEN_RATIO_CONSTANT: float = 1.618033988749895

# Convergence rate parameter (Lyapunov decay rate)
CONVERGENCE_RATE_PARAMETER: float = 0.25


@dataclass
class AnomalyFusionResult:
    """Result of AVA Anomaly Fusion Equation (AAFE) computation with neural verification.

    The AVA Anomaly Fusion Equation (AAFE) provides unified scoring for precision dominance:
    A = (w_R * R(x) + w_H * H(omega) + w_O * O(theta)) * η_Ethical^Φ

    Where:
        - R(x): Recursion component (hierarchical feature extraction)
        - H(omega): Resonance/Harmonic component (frequency-domain analysis)
        - O(theta): Refactoring/Optimization component (adaptive enhancement)
        - η_Ethical: Ethical compliance threshold (0.93-0.96)
        - Φ: Golden ratio constant (1.618) for harmonic scaling
        - w_R, w_H, w_O: Learned fusion weights that sum to 1.0

    Attributes:
        fusion_score: Final A(x) score combining all components
        recursion_score: R(x) component value
        resonance_score: H(omega) component value (harmonic synergy)
        optimization_score: O(theta) component value
        ethical_compliance_threshold: Ethical compliance score used (η_Ethical)
        fusion_weights: Dictionary of learned weights {w_R, w_H, w_O}
        lyapunov_bound: Upper bound on convergence
        convergence_rate: Estimated convergence rate
        neural_anomaly_score: Neural network anomaly score
        dual_verified: True if both traditional and neural scores agree
    """

    fusion_score: float
    recursion_score: float
    resonance_score: float
    optimization_score: float
    ethical_compliance_threshold: float
    fusion_weights: dict[str, float]
    lyapunov_bound: float
    convergence_rate: float = CONVERGENCE_RATE_PARAMETER
    neural_anomaly_score: float | None = None
    dual_verified: bool = False


@dataclass
class RefactoringConfig:
    """Configuration for automatic refactoring operations."""

    max_recursion_depth: int = 10
    complexity_threshold: int = 15
    max_function_length: int = 50
    max_nested_depth: int = 4
    enable_auto_fix: bool = True
    evolution_strategy: EvolutionStrategy = EvolutionStrategy.MODERATE
    preserve_semantics: bool = True
    enable_type_inference: bool = True
    target_metrics: dict[str, float] = field(
        default_factory=lambda: {
            "cyclomatic_complexity": 10.0,
            "cognitive_complexity": 15.0,
            "maintainability_index": 70.0,
            "lines_of_code": 100.0,
        }
    )


@dataclass
class CodeIssue:
    """Represents a detected code issue."""

    issue_type: IssueType
    severity: IssueSeverity
    line_number: int
    column: int
    message: str
    suggestion: str
    confidence: float
    context: str = ""


@dataclass
class RefactoringResult:
    """Result of a refactoring operation."""

    success: bool
    original_code: str
    refactored_code: str
    issues_fixed: list[CodeIssue]
    complexity_before: float
    complexity_after: float
    improvement_score: float
    warnings: list[str] = field(default_factory=list)
