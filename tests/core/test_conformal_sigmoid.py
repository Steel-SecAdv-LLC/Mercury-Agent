# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import warnings

import numpy as np


def test_sigmoid_overflow_guard() -> None:
    """Extreme decision values must produce finite sigmoid scores without RuntimeWarning."""
    from omni_mercury_engine.core.conformal_prediction import ConformalAnomalyDetector

    class _FakeDecisionDetector:
        """Detector stub that returns extreme decision_function values."""

        def fit(self, X, y=None):
            return self

        def decision_function(self, X):
            return np.array([1e4, -1e4, 0.0, 500, -500, 1e8, -1e8])

    det = ConformalAnomalyDetector(base_detector=_FakeDecisionDetector())

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        scores = det._get_anomaly_scores(np.empty((7, 1)))

    assert np.all(np.isfinite(scores)), f"Non-finite scores: {scores}"
    assert np.all((scores >= 0.0) & (scores <= 1.0)), f"Scores out of [0,1]: {scores}"
