# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tabular detectors on real, labelled ADBench tables, measured and pinned.

Five recorded tables (``tests/fixtures/adbench``: Pima diabetes, cardio, breast
cancer Wisconsin, WBC, thyroid) with their published anomaly labels.  Each
detector is fitted on the training part of a stratified 70/30 split with inputs
standardised on the training rows (the protocol of
``benchmarks/dormant_module_revival.py``) and the mean held-out AUROC over three
seeds is pinned.  0.5 is chance.

``topological`` and ``biometric`` (its tabular route) are set-level methods (one
verdict per window), so they are measured the way it is used: fitted on half of the normal rows, then asked to
rank 30 windows of 25 held-out normal rows against 30 windows of 25 anomalous
rows by the window's anomaly score.  WBC has 10 anomalies, too few for windows,
and is left out for it.

The numbers are the measured behaviour on these tables, not published ADBench
results.  They are pinned so a change that degrades a detector, or improves
one, fails here and must be recorded.  Seeds are fixed, runs are repeatable.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from omni_mercury_engine.detectors.topological_analysis import TopologicalAnomalyDetector
from omni_mercury_engine.models.biometric import BiometricAnomalyModel
from tests.core.test_detector_manifest_seams import FAST_CONFIG
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, stratified_split

if TYPE_CHECKING:
    from collections.abc import Callable

ENTRIES = {e.name: e for e in DETECTOR_MANIFEST}
SEEDS = (0, 1, 2)
AUROC_TOLERANCE = 0.05

#: entry -> held-out AUROC per table, in ``ADBENCH_NAMES`` order.
MEASURED: dict[str, tuple[float, ...]] = {
    "adaptive_anomaly": (0.670, 0.883, 0.971, 1.000, 0.958),
    "info_geometry": (0.656, 0.890, 0.964, 0.975, 0.936),
    "spatial": (0.587, 0.566, 0.564, 0.982, 0.835),
    "directive": (0.685, 0.925, 0.988, 0.954, 0.956),
    "realtime_threat": (0.675, 0.907, 0.986, 0.994, 0.972),
}

#: topological: window AUROC per table (``pima_real``, ``cardio``, ``breastw``, ``thyroid``).
MEASURED_WINDOWS: tuple[float, ...] = (0.599, 0.961, 0.999, 1.000)

#: biometric (tabular route): window AUROC per table, same order.
MEASURED_BIOMETRIC_WINDOWS: tuple[float, ...] = (0.997, 1.000, 1.000, 1.000)


def _scores(name: str, detector: Any, rows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The per-row anomaly score each seam returns (the contract differs by entry)."""
    if name == "adaptive_anomaly":
        return np.asarray(detector.detect(rows).scores, dtype=float)
    if name == "info_geometry":
        return np.asarray(detector.predict(rows)["anomaly_scores"], dtype=float)
    if name == "realtime_threat":
        return np.asarray(detector.predict(rows)["scores"], dtype=float)
    return np.asarray(detector.detect(rows)["scores"], dtype=float)


def measure_tabular(name: str, table: str) -> float:
    """Mean held-out AUROC of one manifest entry on one table."""
    entry = ENTRIES[name]
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    features, labels = adbench(table)
    values = []
    for seed in SEEDS:
        train, test = stratified_split(labels, seed)
        mean = features[train].mean(axis=0)
        spread = features[train].std(axis=0)
        spread[spread < 1e-8] = 1.0
        fit_rows = (features[train] - mean) / spread
        test_rows = (features[test] - mean) / spread
        torch.manual_seed(seed)
        np.random.seed(seed)
        detector = cls(FAST_CONFIG[name]) if name in FAST_CONFIG else cls()
        detector.fit(fit_rows)
        scores = _scores(name, detector, test_rows)
        values.append(auroc(scores[labels[test] == 1], scores[labels[test] == 0]))
    return float(np.mean(values))


def measure_windows(
    table: str, make_scorer: Callable[[np.ndarray[Any, Any], int], Callable[..., float]]
) -> float:
    """Mean AUROC of a set-level detector's window score, anomalous against normal windows.

    ``make_scorer(reference_rows, seed)`` returns the function that scores one
    window of rows; 30 windows of 25 held-out normal rows are ranked against 30
    windows of 25 anomalous rows, over three seeds.
    """
    size, windows = 25, 30
    features, labels = adbench(table)
    values = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        normal = rng.permutation(np.where(labels == 0)[0])
        anomalous = np.where(labels == 1)[0]
        reference, held = normal[: len(normal) // 2], normal[len(normal) // 2 :]
        mean = features[reference].mean(axis=0)
        spread = features[reference].std(axis=0)
        spread[spread < 1e-8] = 1.0
        rows = (features - mean) / spread
        score = make_scorer(rows[reference], seed)
        negative = [score(rows[rng.choice(held, size, replace=False)]) for _ in range(windows)]
        positive = [score(rows[rng.choice(anomalous, size, replace=False)]) for _ in range(windows)]
        values.append(auroc(np.array(positive), np.array(negative)))
    return float(np.mean(values))


def topological_scorer(reference: np.ndarray[Any, Any], seed: int) -> Callable[..., float]:
    """``topological`` fitted on the reference (capped at 80 rows)."""
    detector = TopologicalAnomalyDetector(n_reference_samples=80, seed=seed).fit(reference)
    return lambda window: float(detector.score(window)["anomaly_score"])


def biometric_scorer(reference: np.ndarray[Any, Any], seed: int) -> Callable[..., float]:
    """``biometric``'s tabular route: one harmonic spectral-deviation score per matrix."""
    del reference, seed
    model = BiometricAnomalyModel()
    return lambda window: float(np.asarray(model.predict(window)["anomaly_scores"]).reshape(-1)[0])


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_held_out_auroc_matches_the_measurement(name: str) -> None:
    measured = tuple(measure_tabular(name, table) for table in ADBENCH_NAMES)
    for table, got, pinned in zip(ADBENCH_NAMES, measured, MEASURED[name], strict=True):
        assert got == pytest.approx(pinned, abs=AUROC_TOLERANCE), f"{name} on {table}"


WINDOW_TABLES = ("pima_real", "cardio", "breastw", "thyroid")


def test_topological_window_auroc_matches_the_measurement() -> None:
    for table, pinned in zip(WINDOW_TABLES, MEASURED_WINDOWS, strict=True):
        got = measure_windows(table, topological_scorer)
        assert got == pytest.approx(pinned, abs=AUROC_TOLERANCE), table


def test_biometric_window_auroc_matches_the_measurement() -> None:
    for table, pinned in zip(WINDOW_TABLES, MEASURED_BIOMETRIC_WINDOWS, strict=True):
        got = measure_windows(table, biometric_scorer)
        assert got == pytest.approx(pinned, abs=AUROC_TOLERANCE), table


def test_topological_fit_cost_is_bounded_by_the_cap() -> None:
    """3772 thyroid rows fitted in 167 s when calibration ignored the cap."""
    import time

    features, _ = adbench("thyroid")
    rows = (features - features.mean(axis=0)) / np.where(
        features.std(axis=0) > 1e-8, features.std(axis=0), 1.0
    )
    start = time.perf_counter()
    TopologicalAnomalyDetector(n_reference_samples=80, seed=0).fit(rows)
    assert time.perf_counter() - start < 20.0


def test_every_pinned_entry_is_in_the_manifest() -> None:
    assert set(MEASURED) | {"topological"} <= set(ENTRIES)
