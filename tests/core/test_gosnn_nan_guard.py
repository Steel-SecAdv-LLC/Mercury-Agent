# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.global_omni_scalar_network import EthicalGate


def test_ethical_gate_nan_guard() -> None:
    """Partial NaN vector must produce a finite ethical score."""
    gate = EthicalGate(input_dim=8)
    vec = np.array([1.0, np.nan, 0.5, np.nan, 0.8, 0.2, np.nan, 0.9])
    passes, score = gate.evaluate(vec)
    assert np.isfinite(score), f"Non-finite score: {score}"


def test_ethical_gate_all_nan() -> None:
    """All-NaN vector must produce a finite ethical score."""
    gate = EthicalGate(input_dim=4)
    vec = np.full(4, np.nan)
    passes, score = gate.evaluate(vec)
    assert np.isfinite(score), f"Non-finite score: {score}"
