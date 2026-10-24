# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Series detectors on real, labelled NAB data, measured and pinned.

Each detector is fitted on a file's probationary period (NAB's first 15%) and
scores the rest of four recorded NAB files (``tests/fixtures/nab``: AWS
CloudWatch, ad-exchange, a known-cause server fault, road traffic).  Two things
are measured against NAB's own anomaly windows:

* ranking: the mean AUROC of the per-sample score (windows are wide, so this is
  a blunt measure; 0.5 is chance);
* decision: NAB's standard-profile score of the detector's own ``is_anomaly``
  (0 for raising no alarm, 100 for a perfect detector, negative when false
  alarms cost more than the detected windows earn).

``deeplog_sequence`` was pinned here at AUROC 0.534 / NAB -780.3 and is no
longer: it is a categorical next-key model over integer log-template keys
(manifest modality ``event``), and a continuous series gives it no grammar to
learn.  It is measured on recorded HDFS log templates in
``test_recorded_hdfs_log_templates.py`` instead.

The numbers are the measured behaviour on this four-file subset, not published
NAB results and not comparable to them.  They are pinned so a change that
degrades a detector, or that improves one, fails here and must be recorded: a
detector measured at chance is pinned at chance.  Seeds are fixed, runs are
repeatable.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from tests.core.test_detector_manifest_seams import FAST_CONFIG
from tests.recorded_data import LabelledSeries, auroc, nab_corpus, nab_standard_score

ENTRIES = {e.name: e for e in DETECTOR_MANIFEST}

#: entry -> (mean AUROC, NAB standard score) measured on the four files.
MEASURED: dict[str, tuple[float, float]] = {
    "deep_svdd": (0.659, -30.0),
    "diffusion_ad": (0.661, -32.2),
    "digital_twin": (0.607, 53.4),
    "dimensional": (0.466, 7.0),
    "echo_state": (0.699, -368.3),
    "energy_based": (0.671, 27.4),
    "extended_statistical": (0.624, 31.4),
    "gaussian_process": (0.563, 52.9),
    "hawkes": (0.533, 46.1),
    "imm": (0.510, 43.6),
    "lstm_autoencoder": (0.829, -88.5),
    "page_hinkley": (0.534, 26.6),
    "particle_filter": (0.582, 39.6),
    "spectral_residual": (0.489, 45.3),
    "spiking": (0.583, -318.7),
    "spot_evt": (0.503, 19.5),
    "sprt": (0.494, 30.2),
    "srcnn": (0.683, 12.2),
    "survival": (0.596, 48.4),
    "temporal": (0.539, 56.6),
}
AUROC_TOLERANCE = 0.05


def _run(name: str) -> tuple[float, float]:
    """Mean AUROC over the files and NAB standard score of the alarms."""
    entry = ENTRIES[name]
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    aucs: list[float] = []
    alarms: dict[str, np.ndarray[Any, Any]] = {}
    for series in nab_corpus():
        scores, flags = _score(cls, name, series)
        positive = series.positive[series.n_probation :]
        aucs.append(auroc(scores[positive], scores[~positive]))
        alarms[series.name] = np.concatenate([np.zeros(series.n_probation, bool), flags])
    return float(np.mean(aucs)), nab_standard_score(alarms)


def _score(
    cls: Any, name: str, series: LabelledSeries
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    torch.manual_seed(0)
    np.random.seed(0)
    detector = cls(FAST_CONFIG[name]) if name in FAST_CONFIG else cls()
    detector.fit(series.values[: series.n_probation])
    result = detector.detect(series.values[series.n_probation :])
    scores = np.asarray(result["scores"], dtype=float)
    flags = np.asarray(result["is_anomaly"], dtype=bool)
    expected = series.values.size - series.n_probation
    assert scores.shape == (expected,) and flags.shape == (expected,), name
    assert np.isfinite(scores).all(), f"{name}: non-finite score on recorded data"
    assert scores.min() >= 0.0, f"{name}: negative score"
    if isinstance(detector, BaseDetector):  # the base contract: scores in [0, 1]
        assert scores.max() <= 1.0, f"{name}: score outside [0, 1]"
    return scores, flags


@pytest.fixture(scope="module")
def measured() -> dict[str, tuple[float, float]]:
    return {}


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_ranking_and_alarms_match_the_measurement(
    name: str, measured: dict[str, tuple[float, float]]
) -> None:
    pinned_auc, pinned_nab = MEASURED[name]
    mean_auc, nab = _run(name)
    assert mean_auc == pytest.approx(pinned_auc, abs=AUROC_TOLERANCE), (
        f"{name}: mean AUROC {mean_auc:.3f} vs measured {pinned_auc:.3f}; "
        "record the change if it is real"
    )
    assert nab == pytest.approx(pinned_nab, abs=max(7.0, 0.1 * abs(pinned_nab))), (
        f"{name}: NAB standard score {nab:.1f} vs measured {pinned_nab:.1f}; "
        "record the change if it is real"
    )


def test_every_pinned_detector_is_a_manifest_entry_without_other_recorded_data() -> None:
    assert set(MEASURED) <= set(ENTRIES)
