"""
Tests for the three-way ensemble integration (Mercury + AMA + SpectralDomainSound).

SPDX-License-Identifier: GPL-3.0-only
"""

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def test_ama_wired_into_mercury() -> None:
    """AnomalyMathArrest must be fitted and active inside MercuryAnomalyDetector."""
    det = MercuryAnomalyDetector(auto_validate=True)
    X = np.random.RandomState(42).randn(300, 5)
    det.fit(X)
    assert det._ama_detector is not None, "AMA must be fitted"
    assert det._ama_detector._is_fitted, "AMA must be in fitted state"
    result = det.detect(X)
    assert result["ama_active"] is True
    assert 0.30 <= result["mercury_weight"] <= 0.70
    assert 0.30 <= result["ama_weight"] <= 0.70
    assert abs(result["mercury_weight"] + result["ama_weight"] - 1.0) < 1e-6


def test_fusion_weights_are_data_driven() -> None:
    """Fusion weights must change with different training datasets, proving
    they are derived from CV AUC and not hardcoded defaults."""
    rng = np.random.RandomState(0)

    # Dataset A: Gaussian — Resonance/InfoGeo should dominate Mercury side
    X_a = rng.randn(300, 5)
    det_a = MercuryAnomalyDetector(auto_validate=False)
    det_a.fit(X_a)

    # Dataset B: Uniform — different distributional shape
    X_b = rng.uniform(-3, 3, size=(300, 5))
    det_b = MercuryAnomalyDetector(auto_validate=False)
    det_b.fit(X_b)

    w_a = (det_a._mercury_weight, det_a._ama_weight)
    w_b = (det_b._mercury_weight, det_b._ama_weight)

    # Weights must not be identical across structurally different datasets.
    # Tolerance of 1e-3 allows for near-ties on simple synthetic data.
    assert not (
        abs(w_a[0] - w_b[0]) < 1e-3 and abs(w_a[1] - w_b[1]) < 1e-3
    ), (
        f"Fusion weights are identical across different datasets: "
        f"A={w_a}, B={w_b}. This indicates hardcoded defaults, "
        f"not adaptive CV."
    )


def test_enable_ama_false_disables_ama() -> None:
    """enable_ama=False must prevent AMA from being fitted."""
    det = MercuryAnomalyDetector(auto_validate=False, enable_ama=False)
    X = np.random.RandomState(7).randn(200, 4)
    det.fit(X)
    assert det._ama_detector is None, "AMA must be None when enable_ama=False"
    result = det.detect(X)
    assert result["ama_active"] is False
