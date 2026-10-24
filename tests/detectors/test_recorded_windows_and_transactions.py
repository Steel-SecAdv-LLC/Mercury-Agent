# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Window and transaction detectors on recorded data, measured and pinned.

* ``acceleration_dynamics`` scores whole signal windows.  The four recorded NAB
  files are cut into 64-sample windows (stride 32); a window is positive when
  it overlaps one of NAB's anomaly windows.  The detector is fitted on the
  windows inside the probationary period (NAB's first 15%), inputs standardised
  by their mean and spread, and the AUROC of its window scores over the later
  windows is pinned per file.
* ``frequent_pattern`` mines association rules over binary transactions.  No
  labelled transaction corpus is recorded here, so the ADBench tables are
  binarised at the training median (item present when the value is above it),
  the protocol of ``test_recorded_adbench_tabular``.  Magnitude is lost in that
  encoding, and the detector measures at chance on all five tables; that is
  what is pinned, and it is not a finding about transaction data.

The numbers are the measured behaviour on these files, not published results.
Seeds are fixed, runs are repeatable.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.detectors.acceleration_dynamics import AccelerationDynamicsDetector
from omni_mercury_engine.detectors.frequent_pattern import FrequentPatternDetector
from tests.recorded_data import (
    ADBENCH_NAMES,
    adbench,
    auroc,
    nab_corpus,
    stratified_split,
)

AUROC_TOLERANCE = 0.05
WINDOW = 64

#: NAB file (``nab_corpus`` order) -> window AUROC of ``acceleration_dynamics``.
MEASURED_WINDOWS = (0.927, 0.675, 0.471, 0.844)
#: ADBench table (``ADBENCH_NAMES`` order) -> AUROC of ``frequent_pattern``.
MEASURED_TRANSACTIONS = (0.490, 0.413, 0.591, 0.498, 0.443)


def measure_windows() -> tuple[float, ...]:
    """Window AUROC of ``acceleration_dynamics`` on each recorded NAB file."""
    values = []
    for series in nab_corpus():
        starts = np.arange(0, len(series.values) - WINDOW + 1, WINDOW // 2)
        windows = np.stack([series.values[a : a + WINDOW] for a in starts])
        positive = np.array([series.positive[a : a + WINDOW].any() for a in starts])
        later = starts >= series.n_probation
        reference = windows[starts + WINDOW <= series.n_probation]
        mean, spread = reference.mean(), reference.std() or 1.0
        torch.manual_seed(0)
        np.random.seed(0)
        detector = AccelerationDynamicsDetector().fit((reference - mean) / spread)
        scores = np.asarray(detector.detect((windows[later] - mean) / spread)["scores"], float)
        values.append(auroc(scores[positive[later]], scores[~positive[later]]))
    return tuple(values)


def measure_transactions() -> tuple[float, ...]:
    """Mean held-out AUROC of ``frequent_pattern`` on each binarised ADBench table."""
    values = []
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        per_seed = []
        for seed in (0, 1, 2):
            train, test = stratified_split(labels, seed)
            median = np.median(features[train], axis=0)
            detector = FrequentPatternDetector().fit((features[train] > median).astype(float))
            scores = np.asarray(detector.detect((features[test] > median).astype(float))["scores"])
            per_seed.append(auroc(scores[labels[test] == 1], scores[labels[test] == 0]))
        values.append(float(np.mean(per_seed)))
    return tuple(values)


def test_acceleration_dynamics_window_auroc_matches_the_measurement() -> None:
    for series, got, pinned in zip(nab_corpus(), measure_windows(), MEASURED_WINDOWS, strict=True):
        assert got == pytest.approx(pinned, abs=AUROC_TOLERANCE), series.name


def test_frequent_pattern_auroc_matches_the_measurement() -> None:
    for table, got, pinned in zip(
        ADBENCH_NAMES, measure_transactions(), MEASURED_TRANSACTIONS, strict=True
    ):
        assert got == pytest.approx(pinned, abs=AUROC_TOLERANCE), table
