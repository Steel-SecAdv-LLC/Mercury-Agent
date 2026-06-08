# Copyright (C) 2025 Steel Security Advisors LLC
"""3R Mechanism Package."""

from omni_mercury_engine.core.three_r.engines import (
    RecursionEngine,
    ResonanceEngine,
)
from omni_mercury_engine.core.three_r.fusion import (
    AAFEWeightOptimizer,
    AnomalyFusionEquation,
    DomainAdaptiveAAFEWeights,
    DomainAdaptiveOAEWeights,
    OAEWeightOptimizer,
    OmniAvaEquation,
)
from omni_mercury_engine.core.three_r.learnable_fusion import (
    Learnable3RConfig,
    Learnable3REngine,
    Learnable3RResult,
)
from omni_mercury_engine.core.three_r.types import (
    CONVERGENCE_RATE_PARAMETER,
    GOLDEN_RATIO_CONSTANT,
    AnomalyDetectionMethod,
    AnomalyFusionResult,
    CodeIssue,
    EvolutionStrategy,
    IssueSeverity,
    IssueType,
    RefactoringConfig,
    RefactoringResult,
)

__all__ = [
    "CONVERGENCE_RATE_PARAMETER",
    "GOLDEN_RATIO_CONSTANT",
    "AAFEWeightOptimizer",
    "AnomalyDetectionMethod",
    "AnomalyFusionEquation",
    "AnomalyFusionResult",
    "CodeIssue",
    "DomainAdaptiveAAFEWeights",
    "DomainAdaptiveOAEWeights",
    "EvolutionStrategy",
    "IssueSeverity",
    "IssueType",
    "Learnable3RConfig",
    "Learnable3REngine",
    "Learnable3RResult",
    "OAEWeightOptimizer",
    "OmniAvaEquation",
    "RecursionEngine",
    "RefactoringConfig",
    "RefactoringResult",
    "ResonanceEngine",
]
