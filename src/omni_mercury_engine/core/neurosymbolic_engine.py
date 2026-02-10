"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Backward compatibility shim for neurosymbolic engine module.

Note:
    This module has been split into two locations:

    For anomaly detection with LTN and symbolic reasoning:
        from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine

    For AST-based code analysis:
        from omni_mercury_engine.core.code_analysis import CodeAnalysisEngine

    To suppress this deprecation warning, set the environment variable:
        MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS=1
"""

from __future__ import annotations

import os
import warnings

# Re-export from code_analysis (the AST-focused implementation that was here)
from omni_mercury_engine.core.code_analysis import (
    CodeAnalysisConfig,
    CodeAnalysisEngine,
    NeurosymbolicConfig,
    NeurosymbolicEngine,
    ReadinessLevel,
    TrainingPhase,
)

__all__ = [
    "CodeAnalysisConfig",
    "CodeAnalysisEngine",
    "NeurosymbolicConfig",
    "NeurosymbolicEngine",
    "ReadinessLevel",
    "TrainingPhase",
]


class NeurosymbolicEngineDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning for neurosymbolic_engine module.

    This warning is issued when importing from the deprecated
    omni_mercury_engine.core.neurosymbolic_engine module.

    To suppress this warning:
        - Set MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS=1 environment variable
        - Use warnings.filterwarnings('ignore', category=NeurosymbolicEngineDeprecationWarning)
        - Import from the new locations directly
    """

    pass


def _emit_deprecation_warning() -> None:
    """Emit deprecation warning if not suppressed."""
    if os.environ.get("MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return

    warnings.warn(
        "omni_mercury_engine.core.neurosymbolic_engine is deprecated. "
        "Use omni_mercury_engine.core.code_analysis for AST analysis or "
        "omni_mercury_engine.models.neurosymbolic for LTN-based detection. "
        "Set MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS=1 to suppress this warning.",
        NeurosymbolicEngineDeprecationWarning,
        stacklevel=3,
    )


_emit_deprecation_warning()
