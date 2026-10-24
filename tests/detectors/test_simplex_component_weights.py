# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Labelled-path component weights learned on the probability simplex."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.ml.mercury_ml import roc_auc_score


def _labelled(seed: int, n: int = 400, n_anom: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(n - n_anom, 4))
    anomalies = rng.normal(3.0, 1.0, (n_anom, 4))
    X = np.vstack([normal, anomalies])
    y = np.r_[np.zeros(n - n_anom, dtype=int), np.ones(n_anom, dtype=int)]
    return X, y


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight_learning"):
        StatisticalAnomalyDetector({"weight_learning": "softmax"})


def test_simplex_weights_lie_on_the_simplex() -> None:
    X, y = _labelled(0)
    det = StatisticalAnomalyDetector().fit(X)
    components = det._component_score_matrix(X)
    margin = det._compute_adaptive_weights(X, y, components=components)
    learned = det._learn_simplex_weights(components, y, initial=margin)
    assert learned is not None
    assert learned.shape == (3,)
    assert np.all(learned >= 0.0)
    assert float(learned.sum()) == pytest.approx(1.0)


def test_simplex_does_not_lower_the_training_surrogate() -> None:
    X, y = _labelled(1)
    det = StatisticalAnomalyDetector().fit(X)
    C = det._component_score_matrix(X)
    margin = det._compute_adaptive_weights(X, y, components=C)
    learned = det._learn_simplex_weights(C, y, initial=margin)
    assert learned is not None
    assert roc_auc_score(y, C @ learned) >= roc_auc_score(y, C @ margin) - 1e-9


def test_fit_with_labels_uses_simplex_by_default() -> None:
    X, y = _labelled(2)
    det = StatisticalAnomalyDetector().fit_with_labels(X, y)
    assert det._weight_source == "simplex"
    margin = StatisticalAnomalyDetector({"weight_learning": "margin"}).fit_with_labels(X, y)
    assert margin._weight_source == "adaptive"


def test_too_few_labels_keep_the_margin_weights() -> None:
    X, y = _labelled(3, n_anom=3)
    det = StatisticalAnomalyDetector().fit(X)
    C = det._component_score_matrix(X)
    assert det._learn_simplex_weights(C, y, initial=np.full(3, 1 / 3)) is None
    weights = det._supervised_weights(X, y)
    assert det._weight_source == "adaptive"
    np.testing.assert_allclose(weights, det._compute_adaptive_weights(X, y))


def test_no_signal_fallback_is_not_refined(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _labelled(4)
    det = StatisticalAnomalyDetector().fit(X)
    monkeypatch.setattr(det, "_component_separation", lambda scores, labels: 0.5)
    weights = det._supervised_weights(X, y)
    assert det._weight_source == "fallback_default"
    np.testing.assert_allclose(weights, [0.40, 0.30, 0.30])


def test_simplex_learning_is_deterministic() -> None:
    X, y = _labelled(5, n=1200, n_anom=200)  # 200 x 1000 pairs: subsampled path
    a = StatisticalAnomalyDetector().fit_with_labels(X, y)
    b = StatisticalAnomalyDetector().fit_with_labels(X, y)
    np.testing.assert_array_equal(a._adaptive_weights, b._adaptive_weights)
