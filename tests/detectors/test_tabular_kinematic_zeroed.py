"""Test that KinematicScore weight is zero on tabular data with adaptive CV."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def test_tabular_kinematic_zeroed() -> None:
    """KinematicScore weight must be 0.0 on tabular data; others adaptive."""
    det = MercuryAnomalyDetector(auto_validate=True)
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)  # tabular: no temporal structure
    det.fit(X)
    weights = det._adaptive_weights
    assert weights[1] == 0.0, f"KinematicScore must be zero on tabular, got {weights[1]}"
    assert abs(weights[0] + weights[2] - 1.0) < 1e-6, (
        f"Resonance + InfoGeo must sum to 1, got {weights[0] + weights[2]}"
    )
    assert weights[0] > 0, f"Resonance weight must be positive, got {weights[0]}"
    assert weights[2] > 0, f"InfoGeo weight must be positive, got {weights[2]}"


def test_tabular_adaptive_not_hardcoded() -> None:
    """Tabular weights should reflect CV results, not hardcoded 0.50/0.50."""
    det = MercuryAnomalyDetector(auto_validate=True)
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    det.fit(X)
    assert det._weight_source == "unsupervised_adaptive_tabular", (
        f"Expected adaptive tabular source, got {det._weight_source}"
    )
