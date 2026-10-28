# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measured discrimination floors for the rebuilt base detectors.

The planted-anomaly protocol that drove the 2026-10 rebuild: correlated
training data, then scattered point mean shifts, variance bursts,
variance-matched correlation breaks, and a contiguous level shift. The
floors sit safely under the measured 5-seed means so seeds cannot flake
them, and they pin the specific regressions the rebuild fixed:

- temporal scored ~chance (0.54) on scattered 3-sigma point anomalies and
  on level shifts (0.53), because fit() stored statistics detect() never
  read, both channels scored against the contaminated batch, and the
  windowed trend adapted INTO shifts. Measured after: mean 0.889,
  var 0.964, shift 0.982.
- dimensional scored 0.625 on variance-matched correlation breaks, the one
  anomaly family a PCA detector exists for, because its residuals were not
  eigenvalue-normalized. Measured after (mahalanobis_blend 0.7): corr
  0.828 against the exact-Mahalanobis ceiling 0.901 for this protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector
from omni_mercury_engine.ml.mercury_ml import roc_auc_score

D = 8
SEEDS = (1, 2, 3, 4, 5)


def _scenarios(seed: int) -> tuple[np.ndarray[Any, Any], dict[str, tuple[Any, Any]]]:
    r = np.random.default_rng(seed)
    mix = r.normal(size=(D, D)) * 0.3 + np.eye(D)
    train = r.normal(size=(400, D)) @ mix
    y = np.zeros(120)
    idx = r.choice(120, 30, replace=False)
    y[idx] = 1
    base = r.normal(size=(120, D)) @ mix
    mean = base.copy()
    mean[idx.astype(int)] += 3.0
    var = base.copy()
    var[idx.astype(int)] *= 3.0
    uncorr = r.normal(size=(30, D)) * train.std(0, keepdims=True)
    corr = base.copy()
    corr[idx.astype(int)] = uncorr
    y_shift = np.zeros(120)
    y_shift[60:90] = 1
    shift = base.copy()
    shift[60:90] += 2.0
    return train, {
        "mean": (mean, y),
        "var": (var, y),
        "corr": (corr, y),
        "shift": (shift, y_shift),
    }


def _mean_auc(detector_cls: type, scenario: str) -> float:
    values = []
    for seed in SEEDS:
        train, scenarios = _scenarios(seed)
        detector = detector_cls()
        detector.fit(train)
        x, y = scenarios[scenario]
        scores = np.asarray(detector.detect(x)["scores"], dtype=float).ravel()
        assert np.all(np.isfinite(scores))
        values.append(roc_auc_score(y, scores))
    return float(np.mean(values))


@pytest.mark.parametrize(
    ("scenario", "floor"),
    [("mean", 0.84), ("var", 0.92), ("shift", 0.95)],
)
def test_temporal_discrimination_floors(scenario: str, floor: float) -> None:
    auc = _mean_auc(TemporalAnomalyDetector, scenario)
    assert auc >= floor, f"temporal {scenario} ROC-AUC {auc:.3f} fell below {floor}"


@pytest.mark.parametrize(
    ("scenario", "floor"),
    [("mean", 0.99), ("var", 0.97), ("corr", 0.78), ("shift", 0.99)],
)
def test_dimensional_discrimination_floors(scenario: str, floor: float) -> None:
    auc = _mean_auc(DimensionalAnalyzer, scenario)
    assert auc >= floor, f"dimensional {scenario} ROC-AUC {auc:.3f} fell below {floor}"


def test_statistical_sanity_on_the_same_protocol() -> None:
    """The protocol itself is sound: the strongest base detector aces it."""
    assert _mean_auc(MercuryAnomalyDetector, "mean") >= 0.99


def test_temporal_fit_state_is_read_by_detect() -> None:
    """fit() statistics must change detect() output (they used to be dead state)."""
    rng = np.random.default_rng(0)
    quiet = rng.normal(0.0, 1.0, size=(300, 4))
    loud = rng.normal(0.0, 10.0, size=(300, 4))
    batch = rng.normal(0.0, 1.0, size=(60, 4))
    batch[7] += 6.0
    scores_quiet = TemporalAnomalyDetector().fit(quiet).detect(batch)["scores"]
    scores_loud = TemporalAnomalyDetector().fit(loud).detect(batch)["scores"]
    # Fitted on quiet data the 6-sigma spike is glaring; fitted on loud data
    # it is ordinary. Identical outputs would mean fit is decorative again.
    assert scores_quiet[7] > scores_loud[7] + 0.1


def test_temporal_scores_the_opening_rows() -> None:
    """Rows earlier than window_size used to score a hard 0."""
    rng = np.random.default_rng(1)
    detector = TemporalAnomalyDetector().fit(rng.normal(size=(300, 4)))
    batch = rng.normal(size=(40, 4))
    batch[2] += 6.0  # inside the first window
    scores = detector.detect(batch)["scores"]
    assert scores[2] > np.median(scores) + 0.1


def test_temporal_legacy_state_still_detects() -> None:
    """A checkpoint without the per-feature references falls back cleanly."""
    rng = np.random.default_rng(2)
    fitted = TemporalAnomalyDetector().fit(rng.normal(size=(300, 4)))
    state = fitted.get_fitted_state()
    assert state is not None
    for key in ("feature_mean", "feature_std", "diff_mean", "diff_std"):
        state.pop(key)
    legacy = TemporalAnomalyDetector()
    legacy.set_fitted_state(state)
    result = legacy.detect(rng.normal(size=(60, 4)))
    assert np.all(np.isfinite(result["scores"]))


def test_dimensional_legacy_state_keeps_pre_upgrade_scores() -> None:
    """A basis without the spectrum disables the Mahalanobis channel exactly."""
    rng = np.random.default_rng(3)
    fitted = DimensionalAnalyzer().fit(rng.normal(size=(300, 6)))
    state = fitted.get_fitted_state()
    assert state is not None
    assert state["pca_explained_variance"] is not None
    state["pca_full_components"] = None
    state["pca_explained_variance"] = None
    legacy = DimensionalAnalyzer()
    legacy.set_fitted_state(state)
    assert legacy._mahalanobis_scores(rng.normal(size=(5, 6))) is None
    result = legacy.detect(rng.normal(size=(30, 6)))
    assert np.all(np.isfinite(result["scores"]))


def test_dimensional_fitted_state_round_trips_the_spectrum() -> None:
    rng = np.random.default_rng(4)
    fitted = DimensionalAnalyzer().fit(rng.normal(size=(300, 6)))
    state = fitted.get_fitted_state()
    assert state is not None
    restored = DimensionalAnalyzer()
    restored.set_fitted_state(state)
    probe = rng.normal(size=(20, 6))
    a = fitted._mahalanobis_scores(probe)
    b = restored._mahalanobis_scores(probe)
    assert a is not None and b is not None
    np.testing.assert_allclose(a, b)
