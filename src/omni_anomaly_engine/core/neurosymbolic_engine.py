"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Backward compatibility shim for neurosymbolic engine module.

Note:
    This module has been split into two locations:

    For anomaly detection with LTN and symbolic reasoning:
        from omni_anomaly_engine.models.neurosymbolic import NeurosymbolicEngine

    For AST-based code analysis:
        from omni_anomaly_engine.core.code_analysis import CodeAnalysisEngine
"""

import warnings

# Re-export from code_analysis (the AST-focused implementation that was here)
from omni_anomaly_engine.core.code_analysis import (
    CodeAnalysisConfig,
    CodeAnalysisEngine,
    NeurosymbolicConfig,
    NeurosymbolicEngine,
    ReadinessLevel,
    TrainingPhase,
)

__all__ = [
    "NeurosymbolicEngine",
    "NeurosymbolicConfig",
    "TrainingPhase",
    "CodeAnalysisEngine",
    "CodeAnalysisConfig",
    "ReadinessLevel",
]

# Issue deprecation warning on import
warnings.warn(
    "omni_anomaly_engine.core.neurosymbolic_engine is deprecated. "
    "Use omni_anomaly_engine.core.code_analysis for AST analysis or "
    "omni_anomaly_engine.models.neurosymbolic for LTN-based detection.",
    DeprecationWarning,
    stacklevel=2,
)
