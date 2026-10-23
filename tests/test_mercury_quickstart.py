# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The quickstart's evaluation protocols, checked without the network.

The end-to-end run on the real DTU AIS data (and its deterministic
ROC-AUCs) is pinned by the Examples Parity lane; these tests pin what the
README states about the protocols: the cross-fit scores every track once,
each with a model that never saw it, and no label reaches any fit.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "mercury_quickstart.py"


@pytest.fixture(scope="module")
def quickstart() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mercury_quickstart", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cross_fit_scores_each_track_once_from_a_model_that_never_saw_it(
    quickstart: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = np.arange(53.0).reshape(-1, 1)  # row i holds the value i
    calls: list[tuple[set[float], set[float]]] = []

    def recording_fit(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        calls.append((set(X_train[:, 0]), set(X_test[:, 0])))
        return X_test[:, 0] * 10.0

    monkeypatch.setattr(quickstart, "fit_and_score", recording_fit)
    scores = quickstart.label_free_cross_fit(features, n_folds=5, seed=42)

    assert len(calls) == 5
    held_out = [test for _, test in calls]
    assert set().union(*held_out) == set(features[:, 0])
    assert sum(len(test) for test in held_out) == len(features)  # each track once
    for fit_rows, test in calls:
        assert not fit_rows & test  # never scored by a model that saw it
    # Scores land on their own rows.
    assert np.array_equal(scores, features[:, 0] * 10.0)


def test_cross_fit_takes_no_labels(quickstart: ModuleType) -> None:
    """The signature itself: features in, scores out; labels cannot reach a fit."""
    params = inspect.signature(quickstart.label_free_cross_fit).parameters
    assert list(params) == ["features", "n_folds", "seed"]


def test_bootstrap_interval_is_deterministic_and_brackets_the_auc(
    quickstart: ModuleType,
) -> None:
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(200, dtype=int), np.ones(25, dtype=int)]
    scores = np.r_[rng.normal(0, 1, 200), rng.normal(1.5, 1, 25)]
    low, high = quickstart.bootstrap_auc_ci(y, scores)
    assert (low, high) == quickstart.bootstrap_auc_ci(y, scores)
    auc = quickstart.roc_auc_score(y, scores)
    assert low < auc < high
    assert 0.0 <= low < high <= 1.0


def test_published_baselines_are_the_papers_all_track_figures(quickstart: ModuleType) -> None:
    """JMSE 11(11):2085 Figure 12a; the 0.79 is the stationary-removed variant."""
    figures = {name: value for name, value, _ in quickstart.PUBLISHED_BASELINES}
    assert figures == {
        "VRNN (GeoTrackNet arch.)": 0.87,
        "Two-step clustering": 0.73,
        "RVAE": 0.72,
    }
