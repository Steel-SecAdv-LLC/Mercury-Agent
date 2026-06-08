"""Item 4: conformal operating point is opt-in and exact-reducing when off."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def _labelled(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(120, 4))
    anom = rng.normal(3.0, 1.0, size=(40, 4))
    X = np.vstack([normal, anom])
    y = np.concatenate([np.zeros(len(normal), int), np.ones(len(anom), int)])
    return X, y


def test_anomaly_score_threshold_matches_class1_quantile() -> None:
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0.0, 0.4, 200), rng.uniform(0.6, 1.0, 60)])
    y = np.concatenate([np.zeros(200, int), np.ones(60, int)])
    clf = BinaryConformalClassifier(coverage=0.9, seed=42).fit(p, y)
    tau = clf.anomaly_score_threshold()
    assert 0.0 <= tau <= 1.0
    assert_allclose(tau, 1.0 - clf._thresholds[1])


def test_default_off_is_byte_exact_reduction() -> None:
    """Default config == explicit-off: identical threshold, scores, verdict."""
    X, y = _labelled()
    idx = np.arange(len(X))
    cal = np.concatenate([idx[:90], idx[120:150]])  # labelled calibration subset

    default = MercuryAnomalyDetector().fit_with_calibration_subset(X, cal, y[cal])
    off = MercuryAnomalyDetector({"conformal_operating_point": False}).fit_with_calibration_subset(
        X, cal, y[cal]
    )

    assert default._calibration_method != "conformal_lac"
    assert default._supervised_threshold == off._supervised_threshold
    da, oa = default.detect(X), off.detect(X)
    assert_allclose(da["scores"], oa["scores"], rtol=0.0, atol=0.0)
    assert np.array_equal(da["is_anomaly"], oa["is_anomaly"])


def test_opt_in_uses_conformal_operating_point() -> None:
    """With the flag on, the supervised threshold is the conformal quantile."""
    X, y = _labelled()
    idx = np.arange(len(X))
    cal = np.concatenate([idx[:90], idx[120:150]])

    off = MercuryAnomalyDetector().fit_with_calibration_subset(X, cal, y[cal])
    on = MercuryAnomalyDetector({"conformal_operating_point": True}).fit_with_calibration_subset(
        X, cal, y[cal]
    )

    assert on._calibration_method == "conformal_lac"
    # The conformal operating point differs from the Youden/F1 operating point.
    assert on._supervised_threshold != off._supervised_threshold
    # And it equals the standalone conformal classifier's threshold on the same
    # calibration scores (ties the runtime path to the measurement protocol).
    s_cal = np.asarray(on.detect(X[cal])["scores"], dtype=np.float64)
    clf = BinaryConformalClassifier(coverage=0.90, seed=42).fit(s_cal, y[cal])
    assert_allclose(on._supervised_threshold, clf.anomaly_score_threshold())
