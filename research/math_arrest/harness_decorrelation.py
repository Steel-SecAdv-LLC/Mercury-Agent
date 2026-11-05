# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does the probe decorrelator's down-weighting actually buy?

``CorrelationAwareDecorrelator`` down-weights members of a correlated probe
cluster. That had never been measured either way. This harness isolates it
through the shipped fusion with everything else held fixed, on both recorded
corpora, and reports ranking AUROC *and* the operational triple -- because the
two disagree, and only one of them charges for a false alarm.

Reproduces the table in ``PhiWeightedFusion``'s docstring and the
``math_arrest`` row of ``tests/detectors/test_recorded_nab_series.py``.

Run from the repository root::

    python research/math_arrest/harness_decorrelation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from tests.recorded_data import (
    ADBENCH_NAMES,
    adbench,
    auroc,
    nab_corpus,
    stratified_split,
)

#: Mirrors ``tests/detectors/test_recorded_math_arrest.py``.
SEEDS = (0, 1, 2)
WEIGHTINGS = ("uniform", "confidence", "phi_affinity")


def _flatten(detector: AnomalyMathArrest) -> None:
    """Force every multiplier to 1.0, leaving the measurement in place.

    The decorrelator still calibrates, so ``effective_probe_count`` and
    ``redundant_pairs`` are unchanged; only the weighting effect is removed.
    """
    decorrelator = detector._decorrelator
    decorrelator._weight_multipliers = dict.fromkeys(decorrelator._weight_multipliers, 1.0)


def nab_auroc(weighting: str, *, applied: bool) -> float:
    """Mean AUROC over the four recorded NAB files."""
    out = []
    for series in nab_corpus():
        split = series.n_probation
        positive = series.positive[split:]
        detector = AnomalyMathArrest(weighting=weighting)
        detector.fit(series.values[:split])
        if not applied:
            _flatten(detector)
        scores = np.asarray(detector.detect(series.values[split:]), dtype=float).ravel()
        out.append(auroc(scores[positive], scores[~positive]))
    return float(np.mean(out))


def adbench_auroc(weighting: str, *, applied: bool) -> float:
    """Mean AUROC over five ADBench tables x three stratified seeds."""
    per_table = []
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        values = []
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            detector = AnomalyMathArrest(weighting=weighting)
            detector.fit((features[train] - mean) / spread)
            if not applied:
                _flatten(detector)
            scores = np.asarray(
                detector.detect((features[test] - mean) / spread), dtype=float
            ).ravel()
            held = labels[test]
            values.append(auroc(scores[held == 1], scores[held == 0]))
        per_table.append(float(np.mean(values)))
    return float(np.mean(per_table))


def main() -> None:
    """Print the AUROC comparison and the conclusion it does not support."""
    print("Ranking AUROC -- higher is better\n")
    print(
        f"{'weighting':16s} {'NAB applied':>12s} {'NAB measured':>13s}"
        f" {'ADB applied':>12s} {'ADB measured':>13s}"
    )
    for weighting in WEIGHTINGS:
        print(
            f"{weighting:16s} {nab_auroc(weighting, applied=True):12.4f}"
            f" {nab_auroc(weighting, applied=False):13.4f}"
            f" {adbench_auroc(weighting, applied=True):12.4f}"
            f" {adbench_auroc(weighting, applied=False):13.4f}"
        )
    print(
        "\nAUROC prefers dropping the multipliers in all six cells. The operational\n"
        "triple does not: measured through tests/detectors/test_recorded_nab_series.py,\n"
        "applying them scores AUROC 0.758 / NAB 40.4 / alarm rate 0.021, against\n"
        "0.770 / 19.6 / 0.034 without. AUROC does not charge for a false alarm; the\n"
        "NAB standard score does, and the alarm rate shows why -- suppressing a\n"
        "redundant cluster is what keeps the ensemble quiet. The multipliers stay."
    )


if __name__ == "__main__":
    main()
