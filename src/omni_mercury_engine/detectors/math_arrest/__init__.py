# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Anomaly Math Arrest: 21-probe mathematically-independent equation ensemble."""

from omni_mercury_engine.detectors.math_arrest.arrest import (
    PROBE_PRESETS,
    AnomalyMathArrest,
)
from omni_mercury_engine.detectors.math_arrest.base_probe import (
    CATALAN_G,
    EPSILON,
    MIN_SAMPLES,
    PHI,
    BaseEquationProbe,
    ProbeResult,
)

__all__ = [
    "CATALAN_G",
    "EPSILON",
    "MIN_SAMPLES",
    "PHI",
    "PROBE_PRESETS",
    "AnomalyMathArrest",
    "BaseEquationProbe",
    "ProbeResult",
]
