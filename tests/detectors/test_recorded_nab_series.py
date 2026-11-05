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
  alarms cost more than the detected windows earn);
* false alarms: the mean, over the files, of the share of NAB-normal points
  (outside every anomaly window) that ``is_anomaly`` flags.

``spiking``'s 0.142 is one file: ``rogue_agent_key_hold`` sits at exactly
zero for 13% of its probationary period and 59% of the scored part, and the
detector is driven by the magnitude of the deviation from the training mean
(zero is 1.27 standard deviations below it), so the idle stretch reads as a
sustained excursion (0.528 of that file's normal points).  It is a property of
the input drive, not of the cut, and is pinned rather than tuned away.

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

#: entry -> (mean AUROC, NAB standard score, mean normal-point false-alarm
#: rate) measured on the four files.
MEASURED: dict[str, tuple[float, float, float]] = {
    "deep_svdd": (0.659, -30.0, 0.061),
    "diffusion_ad": (0.661, -32.2, 0.054),
    "digital_twin": (0.607, 53.4, 0.014),
    # Was (0.533, 48.7, 0.001) under batch min-max normalisation and a fixed
    # 0.5 cut, whose false-alarm rate depended on the batch: 0.001 on these
    # 1,000-row series, 0.30 of Gaussian rows in 10-row batches. Each row is
    # now normalised against the training range and cut at the training
    # rows' 0.95 conformal quantile, the 5% budget of the other detectors;
    # on these files that alarms more, and NAB's standard profile scores it
    # lower. The ranking is unchanged.
    "dimensional": (0.537, 22.7, 0.025),
    # Was (0.699, -368.3) at a fixed readout ridge of 1e-4, which flagged
    # 0.188 of the normal points (0.533 on rogue_agent_key_hold): the
    # 101-parameter readout's out-of-fold error exceeded the trivial
    # predictor's. The ridge is now chosen by out-of-fold error.
    "echo_state": (0.629, 44.5, 0.012),
    "energy_based": (0.671, 27.4, 0.026),
    "extended_statistical": (0.624, 31.4, 0.026),
    "gaussian_process": (0.563, 52.9, 0.013),
    "hawkes": (0.533, 46.1, 0.011),
    "imm": (0.510, 43.6, 0.016),
    "lstm_autoencoder": (0.829, 3.1, 0.038),
    # The 21-probe equation ensemble, reachable from this corpus for the first
    # time (docs/DORMANCY_LEDGER.md 2b): second-best ranking here, behind only
    # lstm_autoencoder, at a 0.022 false-alarm rate on the conformal cut.
    "math_arrest": (0.758, 40.4, 0.021),
    "page_hinkley": (0.534, 26.6, 0.008),
    "particle_filter": (0.582, 39.6, 0.017),
    "spectral_residual": (0.489, 45.3, 0.010),
    "spiking": (0.583, -318.7, 0.142),
    "spot_evt": (0.503, 19.5, 0.003),
    "sprt": (0.494, 30.2, 0.005),
    "srcnn": (0.683, 12.2, 0.001),
    # The default operating point (fit-time split-conformal cut, alpha 0.05).
    # The batch-relative cut it replaced scored the same ranking -130.5, with
    # a false-alarm rate of 0.120 (0.407 on the AWS CloudWatch file).
    "statistical": (0.570, 30.7, 0.021),
    "survival": (0.596, 48.4, 0.012),
    "temporal": (0.643, 46.0, 0.008),
}
AUROC_TOLERANCE = 0.05


def _run(name: str) -> tuple[float, float, float]:
    """Mean AUROC, NAB standard score and mean normal-point false-alarm rate."""
    entry = ENTRIES[name]
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    aucs: list[float] = []
    false_alarm_rates: list[float] = []
    alarms: dict[str, np.ndarray[Any, Any]] = {}
    for series in nab_corpus():
        scores, flags = _score(cls, name, series)
        positive = series.positive[series.n_probation :]
        aucs.append(auroc(scores[positive], scores[~positive]))
        false_alarm_rates.append(float(flags[~positive].mean()))
        alarms[series.name] = np.concatenate([np.zeros(series.n_probation, bool), flags])
    return float(np.mean(aucs)), nab_standard_score(alarms), float(np.mean(false_alarm_rates))


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
def measured() -> dict[str, tuple[float, float, float]]:
    return {}


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_ranking_and_alarms_match_the_measurement(
    name: str, measured: dict[str, tuple[float, float, float]]
) -> None:
    pinned_auc, pinned_nab, pinned_fpr = MEASURED[name]
    mean_auc, nab, fpr = _run(name)
    assert mean_auc == pytest.approx(pinned_auc, abs=AUROC_TOLERANCE), (
        f"{name}: mean AUROC {mean_auc:.3f} vs measured {pinned_auc:.3f}; "
        "record the change if it is real"
    )
    assert nab == pytest.approx(pinned_nab, abs=max(7.0, 0.1 * abs(pinned_nab))), (
        f"{name}: NAB standard score {nab:.1f} vs measured {pinned_nab:.1f}; "
        "record the change if it is real"
    )
    assert fpr == pytest.approx(pinned_fpr, abs=max(0.01, 0.25 * pinned_fpr)), (
        f"{name}: normal-point false-alarm rate {fpr:.3f} vs measured {pinned_fpr:.3f}; "
        "record the change if it is real"
    )


def test_every_pinned_detector_is_a_manifest_entry_without_other_recorded_data() -> None:
    assert set(MEASURED) <= set(ENTRIES)
