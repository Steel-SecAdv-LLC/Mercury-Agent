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


def test_tabular_kinematic_zeroed_and_adaptive() -> None:
    """KinematicScore must be 0.0 on tabular; Resonance+InfoGeo must be adaptive."""
    np.random.seed(42)
    det = MercuryAnomalyDetector(auto_validate=True)
    X = np.random.randn(300, 10)
    det.fit(X)

    w = det._last_adaptive_weights
    assert w[1] == 0.0, f"KinematicScore must be 0 on tabular data; got {w[1]}"
    assert abs(w[0] + w[2] - 1.0) < 1e-6, f"Res+InfoGeo must sum to 1.0; got {w[0]+w[2]}"
    assert w[0] > 0 and w[2] > 0, f"Both weights must be > 0; got {w}"
    # Weights must NOT be hardcoded 0.50/0.50
    assert not (abs(w[0] - 0.50) < 1e-9 and abs(w[2] - 0.50) < 1e-9), \
        "Weights appear hardcoded to 0.50/0.50 — adaptive CV is required"
