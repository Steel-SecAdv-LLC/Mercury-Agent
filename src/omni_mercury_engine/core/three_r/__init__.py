# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""3R Mechanism Package.

The Three-R (Recursion-Resonance-Refactoring) Mechanism provides a unified
framework for adaptive anomaly detection and code quality enhancement.

This package has been split from the monolithic three_r_mechanism.py for
improved maintainability:

- types.py: Enums, dataclasses, and type definitions
- engines.py: RecursionEngine and ResonanceEngine
- fusion.py: OmniAvaEquation and OAEWeightOptimizer

The main orchestrator (ThreeRMechanism) remains in the parent module
for backward compatibility.

Example:
    >>> from omni_mercury_engine.core.three_r import (
    ...     RecursionEngine,
    ...     ResonanceEngine,
    ...     OmniAvaEquation,
    ... )
    >>> recursion = RecursionEngine(max_depth=5)
    >>> resonance = ResonanceEngine(sampling_rate=1.0)
    >>> fusion = OmniAvaEquation(ethical_compliance_threshold=0.96)
"""

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
