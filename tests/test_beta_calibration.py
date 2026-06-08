"""Stage 2 R1/R4: Beta-MCA calibration properties + exact-reducing accept-gate.

Pins the guarantees the calibration thesis rests on:

* the monotone beta map preserves AUROC **exactly** (I3-free);
* it lowers Brier and ECE vs the uncalibrated scores on a miscalibrated set;
* unfitted / degenerate -> identity passthrough (exact reduction);
* the accept-gate never ships a Brier/ECE regression.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.calibration import (
    BetaCalibration,
    _IdentityCalibration,
    compute_ece,
    fit_accept_gated_mca,
)
from omni_mercury_engine.ml.mercury_ml import brier_score_loss, roc_auc_score


def _miscalibrated(seed: int = 0, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Monotone-but-miscalibrated scores: true p = s**3 (over-confident at top)."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < s**3).astype(int)
    return s, y


def test_beta_preserves_auroc_exactly() -> None:
    s, y = _miscalibrated(1)
    beta = BetaCalibration().fit(s, y)
    p = beta.calibrate(s)
    assert beta._fitted
    # Strictly-monotone map -> identical rank order -> identical AUROC.
    assert roc_auc_score(y, p) == roc_auc_score(y, s)


def test_beta_lowers_brier_and_ece_vs_identity() -> None:
    s, y = _miscalibrated(2)
    beta = BetaCalibration().fit(s, y)
    p = beta.calibrate(s)
    assert brier_score_loss(y, p) < brier_score_loss(y, s)
    assert compute_ece(y, p) < compute_ece(y, s)


def test_beta_unfitted_is_identity_passthrough() -> None:
    s = np.array([0.1, 0.4, 0.55, 0.9, 0.99])
    out = BetaCalibration().calibrate(s)
    assert np.array_equal(out, s)  # default-off / unfitted == exact reduction


def test_beta_degenerate_calibration_stays_identity() -> None:
    s = np.linspace(0.1, 0.9, 20)
    y = np.zeros(20, dtype=int)  # one class only
    beta = BetaCalibration().fit(s, y)
    assert not beta._fitted
    assert np.array_equal(beta.calibrate(s), s)


def test_accept_gate_never_ships_a_regression() -> None:
    s, y = _miscalibrated(3)
    cal, accepted = fit_accept_gated_mca(s, y)
    assert accepted and isinstance(cal, BetaCalibration)
    p = cal.calibrate(s)
    # The gate guarantees no Brier regression on the calibration set.
    assert brier_score_loss(y, p) <= brier_score_loss(y, s)
    assert compute_ece(y, p) <= compute_ece(y, s) + 1e-3
    # AUROC still exactly tied.
    assert roc_auc_score(y, p) == roc_auc_score(y, s)


def test_accept_gate_degenerate_falls_back_to_identity() -> None:
    s = np.linspace(0.1, 0.9, 20)
    y = np.zeros(20, dtype=int)
    cal, accepted = fit_accept_gated_mca(s, y)
    assert not accepted and isinstance(cal, _IdentityCalibration)
    assert np.array_equal(cal.calibrate(s), s)


def _labelled(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(120, 4))
    anom = rng.normal(3.0, 1.0, size=(40, 4))
    X = np.vstack([normal, anom])
    y = np.concatenate([np.zeros(len(normal), int), np.ones(len(anom), int)])
    return X, y


def test_detector_calibration_map_default_off_is_byte_exact() -> None:
    """Default calibration_map='identity' -> detect output byte-identical; no MCA key."""
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    X, _ = _labelled(0)
    default = MercuryAnomalyDetector().fit(X).detect(X)
    off = MercuryAnomalyDetector({"calibration_map": "identity"}).fit(X).detect(X)
    assert "calibrated_probabilities" not in default
    np.testing.assert_array_equal(default["scores"], off["scores"])
    np.testing.assert_array_equal(default["is_anomaly"], off["is_anomaly"])


def test_detector_mca_is_additive_and_preserves_auroc() -> None:
    """calibration_map='mca' adds calibrated_probabilities; scores/verdict untouched."""
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    X, y = _labelled(1)
    idx = np.arange(len(X))
    cal = np.concatenate([idx[:90], idx[120:150]])
    off = MercuryAnomalyDetector().fit_with_calibration_subset(X, cal, y[cal]).detect(X)
    on = (
        MercuryAnomalyDetector({"calibration_map": "mca"})
        .fit_with_calibration_subset(X, cal, y[cal])
        .detect(X)
    )
    assert "calibrated_probabilities" not in off
    assert "calibrated_probabilities" in on
    # Exact-reducing on the existing outputs: scores / verdict unchanged.
    np.testing.assert_array_equal(np.asarray(on["scores"]), np.asarray(off["scores"]))
    np.testing.assert_array_equal(np.asarray(on["is_anomaly"]), np.asarray(off["is_anomaly"]))
    # Calibrated probabilities preserve AUROC exactly (monotone map).
    p = np.asarray(on["calibrated_probabilities"], dtype=np.float64)
    s = np.asarray(on["scores"], dtype=np.float64)
    assert roc_auc_score(y, p) == roc_auc_score(y, s)
