# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The foundation-extra adapters on the recorded NAB files, measured and pinned.

``matrix_profile`` (STUMPY discords) and ``foundation_ensemble`` (whose default
configuration has the matrix profile as its only member) need no fitting: the
whole of each file is scored with ``detect_anomalies``, the probationary period
is ignored when ranking and when scoring the alarms, and the mean AUROC against
NAB's anomaly windows and NAB's standard-profile score of the alarms are
pinned.  A discord is a subsequence with no near match, and a NAB anomaly
window is a long episode that repeats within itself, so the profile ranks the
road-traffic file's windows *below* chance (0.07); raw ``stumpy.stump`` gives
the same ranking, so that is the method on this data, not the adapter.

The numbers are the measured behaviour on this four-file subset, not published
NAB results.  Seeds are fixed, runs are repeatable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("stumpy")

from omni_mercury_engine.models.foundation.ensemble import FoundationEnsemble
from omni_mercury_engine.models.foundation.matrix_profile import MatrixProfileDetector
from tests.recorded_data import auroc, nab_corpus, nab_standard_score

AUROC_TOLERANCE = 0.05

#: entry -> (mean AUROC, NAB standard score) on the four files.
MEASURED = {
    "matrix_profile": (0.356, -82.0),
    "foundation_ensemble": (0.356, -82.0),
}
_CLASSES: dict[str, Any] = {
    "matrix_profile": MatrixProfileDetector,
    "foundation_ensemble": FoundationEnsemble,
}


def measure(name: str) -> tuple[float, float]:
    """Mean AUROC over the files and NAB standard score of the alarms."""
    aurocs: list[float] = []
    alarms: dict[str, np.ndarray[Any, Any]] = {}
    for series in nab_corpus():
        result = _CLASSES[name]().detect_anomalies(series.values.astype(np.float64))
        scores = np.asarray(result["scores"], dtype=float).reshape(-1)
        flags = np.asarray(result["is_anomaly"], dtype=bool).reshape(-1)
        assert scores.shape == series.values.shape, name
        later = slice(series.n_probation, None)
        positive = series.positive[later]
        aurocs.append(auroc(scores[later][positive], scores[later][~positive]))
        alarms[series.name] = np.concatenate([np.zeros(series.n_probation, bool), flags[later]])
    return float(np.mean(aurocs)), nab_standard_score(alarms)


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_ranking_and_alarms_match_the_measurement(name: str) -> None:
    pinned_auc, pinned_nab = MEASURED[name]
    mean_auc, nab = measure(name)
    assert mean_auc == pytest.approx(pinned_auc, abs=AUROC_TOLERANCE), name
    assert nab == pytest.approx(pinned_nab, abs=max(7.0, 0.1 * abs(pinned_nab))), name


def test_the_default_ensemble_is_its_one_member() -> None:
    series = nab_corpus()[0].values.astype(np.float64)
    ensemble = np.asarray(FoundationEnsemble().detect_anomalies(series)["scores"])
    member = np.asarray(MatrixProfileDetector().detect_anomalies(series)["scores"])
    assert np.array_equal(ensemble.reshape(-1), member.reshape(-1))
