# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Anomaly Math Revolver: 21-probe mathematically-independent equation ensemble."""

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    CATALAN_G,
    EPSILON,
    MIN_SAMPLES,
    PHI,
    BaseEquationProbe,
    ProbeResult,
)
from omni_mercury_engine.detectors.math_revolver.revolver import AnomalyMathRevolver

__all__ = [
    "CATALAN_G",
    "EPSILON",
    "MIN_SAMPLES",
    "PHI",
    "AnomalyMathRevolver",
    "BaseEquationProbe",
    "ProbeResult",
]
