"""
Mercury Agent - 3R Mechanism Package
Copyright (C) 2025 Steel Security Advisory LLC

The Three-R (Recursion-Resonance-Refactoring) Mechanism provides a unified
framework for adaptive anomaly detection and code quality enhancement.

This package has been split from the monolithic three_r_mechanism.py for
improved maintainability:

- types.py: Enums, dataclasses, and type definitions
- engines.py: RecursionEngine and ResonanceEngine
- fusion.py: AnomalyFusionEquation and AAFEWeightOptimizer

The main orchestrator (ThreeRMechanism) remains in the parent module
for backward compatibility.

Example:
    >>> from omni_mercury_engine.core.three_r import (
    ...     RecursionEngine,
    ...     ResonanceEngine,
    ...     AnomalyFusionEquation,
    ... )
    >>> recursion = RecursionEngine(max_depth=5)
    >>> resonance = ResonanceEngine(sampling_rate=1.0)
    >>> fusion = AnomalyFusionEquation(ethical_compliance_threshold=0.96)
"""

from typing import Any
from omni_mercury_engine.core.three_r.engines import (
    RecursionEngine,
    ResonanceEngine,
)
from omni_mercury_engine.core.three_r.fusion import (
    AAFEWeightOptimizer,
    AnomalyFusionEquation,
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
    "EvolutionStrategy",
    "IssueSeverity",
    "IssueType",
    "RecursionEngine",
    "RefactoringConfig",
    "RefactoringResult",
    "ResonanceEngine",
]
