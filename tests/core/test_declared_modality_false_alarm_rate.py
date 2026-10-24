# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""No manifest entry may flood on normal data of the modality it declares.

``DETECTOR_MANIFEST`` carries a hand-written ``modality`` per entry, and
``DetectorRegistry.extract_all_features`` routes data by it. Nothing checked
the declaration against the detector's own input handling, so five entries
declared a modality their ``fit`` contradicts and the filter -- built to
prevent exactly that -- delivered data they could not interpret. None of them
raised; each returned a confident wrong verdict.

``tests/detectors/test_stationary_false_alarm_rate.py`` catches that for
``series_1d`` by driving every such entry on a stationary process and pinning
what it flags where nothing is anomalous. A wrong declaration shows up there
as a flood, which is how ``rca`` was found: declared ``series_1d``, a
200-sample series became one row of 200 "nodes", every node's training spread
collapsed to the ``1e-9`` floor, and it flagged 100% of a held-out half.

This module extends that principle to ``matrix_2d``, the other modality with
a population of entries behind the registry seam. The payload is independent
Gaussian rows, so nothing in the held-out half is anomalous and whatever an
entry flags there is a false alarm. The seam tests next door assert only that
a detector *returned* something; this asserts the verdict means something.
"""

from __future__ import annotations

import importlib
from functools import cache
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

SEEDS = (7, 11, 23, 101, 999)
#: Rows per half. 120 clears ``physics_integrated``'s conformal calibration,
#: which needs ``_CALIBRATION_STRIDE * _MIN_CALIBRATION_SIGNALS`` = 76 signals.
ROWS = 120
COLUMNS = 8
TOLERANCE = 0.03

#: Per-row entry -> share of the held-out normal rows it flags.
MEASURED_RATE: dict[str, float] = {
    # Above a 0.05 nominal but bounded. A security detector biased to
    # fail closed is the defensible side to err on.
    "realtime_threat": 0.1067,
    "kmeans_distance": 0.0867,
    "adversarial_autoencoder": 0.0900,
    "copod": 0.0717,
    # At or under contract.
    "directive": 0.0600,
    "rca": 0.0467,
    "spatial": 0.0433,
}

#: Set-level entry -> share of the seeds whose single verdict came back
#: anomalous. Only 0.0, 0.2, ... 1.0 can occur, so this is not a rate.
MEASURED_VERDICT: dict[str, float] = {
    "physics_integrated": 0.0,
}

#: ``matrix_2d`` entries this gate cannot drive, and why: each returns a
#: ranking or a diagnostic rather than an ``is_anomaly`` verdict, so there is
#: no false alarm to count. They are covered by their own tests.
#: Constructor arguments for entries that need them to be reproducible.
#: ``AdversarialAutoencoderDetector`` draws its network weights and picks its
#: splits from ``default_rng(seed)``, which no global seed reaches, so without
#: an explicit seed the measurement moves between runs (0.1567 then 0.1367 on
#: the same command). Its own ``seed`` documents ``None`` as "uses OS
#: entropy", so pinning it here is the caller's job, not a defect.
ENTRY_KWARGS: dict[str, dict[str, Any]] = {
    "adversarial_autoencoder": {"seed": 0},
}

NO_VERDICT_SEAM = frozenset(
    {
        "adaptive_anomaly",
        "info_geometry",
        "topological",
    }
)


def _candidates() -> list[Any]:
    """Manifest entries this gate drives: ``matrix_2d``, registry seam."""
    return [
        entry
        for entry in DETECTOR_MANIFEST
        if entry.modality == "matrix_2d" and "registry" in entry.seams
    ]


def normal_rows(seed: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """A fit half and a held-out half of independent Gaussian rows."""
    rng = np.random.default_rng(seed)
    return (
        np.ascontiguousarray(rng.normal(size=(ROWS, COLUMNS))),
        np.ascontiguousarray(rng.normal(size=(ROWS, COLUMNS))),
    )


@cache
def _verdicts(name: str) -> tuple[np.ndarray[Any, Any], ...]:
    """One held-out verdict array per seed, from a fit on the other half.

    Cached: four tests read the same verdicts and a fit here is not cheap.
    """
    entry = next(e for e in DETECTOR_MANIFEST if e.name == name)
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    out = []
    for seed in SEEDS:
        fit_rows, held_out = normal_rows(seed)
        # Trained entries (e.g. adversarial_autoencoder) draw weights, so the
        # measurement is only reproducible with both RNGs pinned -- the same
        # thing tests/detectors/test_stationary_false_alarm_rate.py does.
        torch.manual_seed(0)
        np.random.seed(0)
        detector = cls(**ENTRY_KWARGS.get(name, {}))
        detector.fit(fit_rows)
        score = getattr(detector, "detect", None) or detector.predict
        result = score(held_out)
        out.append(np.asarray(result["is_anomaly"], dtype=bool).reshape(-1))
    return tuple(out)


@pytest.mark.parametrize("name", sorted(MEASURED_RATE))
def test_the_per_row_false_alarm_rate_is_what_was_measured(name: str) -> None:
    """Pinned so a change that worsens -- or improves -- one fails here."""
    rate = float(np.mean([v.mean() for v in _verdicts(name)]))

    assert rate == pytest.approx(MEASURED_RATE[name], abs=TOLERANCE), (
        f"{name}: flags {rate:.4f} of held-out normal rows against the measured "
        f"{MEASURED_RATE[name]:.4f}; record the change if it is real"
    )


@pytest.mark.parametrize("name", sorted(MEASURED_VERDICT))
def test_the_set_level_verdict_share_is_what_was_measured(name: str) -> None:
    share = float(np.mean([bool(v.all()) for v in _verdicts(name)]))

    assert share == pytest.approx(MEASURED_VERDICT[name], abs=0.21), (
        f"{name}: {share:.1f} of the seeds called entirely normal rows anomalous "
        f"against the measured {MEASURED_VERDICT[name]:.1f}"
    )


def test_the_verdict_arity_is_what_each_group_claims() -> None:
    """A per-row entry must return one verdict per row, a set-level entry one.

    Keeps an entry from drifting between the groups unnoticed, where a
    "rate" would silently become one Boolean or the reverse.
    """
    for name in MEASURED_RATE:
        for verdict in _verdicts(name):
            assert verdict.size == ROWS, f"{name} is pinned per-row but returned {verdict.size}"
    for name in MEASURED_VERDICT:
        for verdict in _verdicts(name):
            assert verdict.size == 1, f"{name} is pinned set-level but returned {verdict.size}"


def test_every_drivable_matrix_entry_is_pinned() -> None:
    """A new ``matrix_2d`` entry cannot join the registry seam unmeasured.

    It is pinned here, or listed in ``NO_VERDICT_SEAM`` with the reason. The
    point of the gate is that no entry reaches the registry's modality filter
    without someone having looked at what it does to normal data.
    """
    pinned = set(MEASURED_RATE) | set(MEASURED_VERDICT) | NO_VERDICT_SEAM
    driven = {entry.name for entry in _candidates()}

    assert driven <= pinned, f"unmeasured matrix_2d entries: {sorted(driven - pinned)}"
    assert (
        pinned <= driven
    ), f"pinned entries that are no longer matrix_2d: {sorted(pinned - driven)}"


def test_no_excluded_entry_silently_gained_a_verdict() -> None:
    """An excluded entry that starts emitting ``is_anomaly`` must be pinned."""
    for name in sorted(NO_VERDICT_SEAM):
        entry = next(e for e in DETECTOR_MANIFEST if e.name == name)
        cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
        fit_rows, held_out = normal_rows(SEEDS[0])
        detector = cls(**ENTRY_KWARGS.get(name, {}))
        detector.fit(fit_rows)
        score = getattr(detector, "detect", None) or getattr(detector, "predict", None)
        if score is None:
            continue
        result = score(held_out)
        assert not (isinstance(result, dict) and result.get("is_anomaly") is not None), (
            f"{name} now returns an is_anomaly verdict; move it out of "
            "NO_VERDICT_SEAM and pin its false-alarm rate"
        )
