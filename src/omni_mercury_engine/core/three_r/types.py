"""
Mercury Agent - 3R Mechanism Type Definitions

Copyright (C) 2025 Steel Security Advisors LLC

Type definitions, enums, and dataclasses for the 3R Mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# P2: Import from centralized constants
from omni_mercury_engine.core.centralized_constants import LYAPUNOV, MATH


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


# P2: Golden ratio constant for Omni-Ava Equation (OAE)
# Now references centralized constant
GOLDEN_RATIO_CONSTANT: float = MATH.GOLDEN_RATIO

# P2: Convergence rate parameter (Lyapunov decay rate)
# Now references centralized constant
CONVERGENCE_RATE_PARAMETER: float = LYAPUNOV.LAMBDA_CONVERGENCE

# Mathematical constants (derived from golden ratio)
_GOLDEN_RATIO_CONJUGATE: float = 1.0 / MATH.GOLDEN_RATIO  # 0.618...
_CATALAN: float = 0.9159655941772190151
_EULER_MASCHERONI: float = 0.5772156649015328606
_FEIGENBAUM_DELTA: float = 4.6692016091029906719
_OMEGA: float = 0.5671432904097838730


@dataclass
class AnomalyFusionResult:
    """Result of Omni-Ava Equation (OAE) computation with neural verification.

    The Omni-Ava Equation (OAE) provides unified scoring for precision dominance:
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
    """
    Configuration for automatic refactoring operations.

    Mathematical constants are sourced from the centralized MathematicalConstants module for
    precision and consistency.
    """

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

    apply_refactorings: bool = False
    create_backup: bool = True
    require_confirmation: bool = True
    backup_suffix: str = ".bak"
    max_complexity_threshold: int = 10
    max_nesting_threshold: int = 4

    enable_caching: bool = True
    enable_harmonics: bool = True
    enable_quantum_paths: bool = True
    enable_pattern_resonance: bool = True
    enable_neurosymbolic: bool = True
    quantum_num_paths: int = 1
    enable_parallel_processing: bool = False
    enable_resonance_feedback: bool = False
    resonance_feedback_depth: int = 3
    enable_multiverse_optimization: bool = False

    golden_ratio: float = _GOLDEN_RATIO_CONJUGATE
    catalan_constant: float = _CATALAN
    euler_mascheroni: float = _EULER_MASCHERONI
    feigenbaum_delta: float = _FEIGENBAUM_DELTA
    omega_constant: float = _OMEGA

    enable_spherical_harmonics: bool = False
    spherical_harmonic_degree: int = 4
    enable_rotation_invariance: bool = False

    enable_quantum_superposition: bool = False
    superposition_paths: int = 3

    enable_federated_learning: bool = False
    federated_num_clients: int = 5
    federated_learning_rate: float = 0.001
    federated_local_epochs: int = 5
    federated_aggregation: str = "fedavg"

    enable_symbolic_reasoning: bool = False
    symbolic_temporal_logic: bool = True
    symbolic_graph_based: bool = True
    symbolic_explainability_threshold: float = 0.7

    enable_info_geometry: bool = False
    info_geom_distance_metric: str = "fisher_rao"
    info_geom_manifold_dim: int = 10
    info_geom_approximation: str = "closed_form"

    enable_quantum_kernels: bool = False
    quantum_kernel_type: str = "quantum_inspired"
    quantum_num_qubits: int = 4
    quantum_entanglement_depth: int = 2
    quantum_gamma: float = 1.0

    enable_novel_class_discovery: bool = False
    ncd_enable_mebin: bool = True
    ncd_num_clusters: int = 5
    ncd_low_semantics_mode: bool = True
    ncd_non_prominence_mode: bool = True

    enable_multivariate_ts: bool = False
    mvts_window_size: int = 100
    mvts_lstm_hidden_dim: int = 64
    mvts_temporal_conv_filters: int = 32
    mvts_graph_conv_layers: int = 2

    enable_chaos_creativity: bool = False
    chaos_creativity_intensity: float = 0.1
    chaos_creativity_num_hypotheses: int = 10

    enable_chaos_optimization: bool = False
    chaos_population_size: int = 30
    chaos_max_iterations: int = 100
    chaos_map_type: str = "logistic"
    chaos_alpha: float = 0.8
    chaos_beta: float = 0.2


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
