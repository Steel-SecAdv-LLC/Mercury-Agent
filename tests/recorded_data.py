# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recorded data for detector tests: NAB series, ADBench tables, CWRU bearing vibration, CHB-MIT EEG, HDFS log templates.

Everything here is read from ``tests/fixtures`` (recorded by
``scripts/record_nab_fixtures.py``, ``scripts/record_adbench_fixtures.py``,
``scripts/record_cwru_fixtures.py``, ``scripts/record_eeg_fixtures.py`` and
``scripts/record_hdfs_log_fixtures.py``); nothing is generated. NAB labels a *window* around each real anomaly, so a point
is positive when its timestamp lies inside one of the file's windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.evaluation.nab_scoring import (
    NABStream,
    SweepTable,
    load_nab_corpus,
    probationary_length,
    score_corpus,
    sweep_table,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NAB_CATEGORIES = ("realAWSCloudwatch", "realAdExchange", "realKnownCause", "realTraffic")
ADBENCH_NAMES = ("pima_real", "cardio", "breastw", "wbc", "thyroid")


@dataclass(frozen=True)
class LabelledSeries:
    """One NAB file: the series, the per-point window label and the probation split."""

    name: str
    values: np.ndarray[Any, Any]
    positive: np.ndarray[Any, Any]
    n_probation: int
    table: SweepTable


@cache
def nab_corpus() -> tuple[LabelledSeries, ...]:
    """The four recorded NAB files, each with its windows and sweep table."""
    series = []
    for stream in load_nab_corpus(FIXTURES / "nab", NAB_CATEGORIES):
        series.append(_labelled(stream))
    return tuple(series)


def _labelled(stream: NABStream) -> LabelledSeries:
    positive = np.zeros(stream.values.size, dtype=bool)
    for first, last in stream.windows:
        positive[first : last + 1] = True
    return LabelledSeries(
        name=stream.name,
        values=stream.values,
        positive=positive,
        n_probation=int(probationary_length(stream.values.size)),
        table=sweep_table(stream.values.size, stream.windows, "standard"),
    )


def nab_standard_score(alarms: dict[str, np.ndarray[Any, Any]]) -> float:
    """NAB's normalized standard-profile score: 0 for no alarms, 100 for a perfect detector."""
    tables = {s.name: s.table for s in nab_corpus()}
    return float(score_corpus(alarms, tables)["score"])


def adbench(name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """An ADBench table ``(X, y)``, ``y == 1`` marking the anomalies."""
    with np.load(FIXTURES / "adbench" / f"{name}.npz", allow_pickle=False) as archive:
        return np.asarray(archive["X"], dtype=np.float64), np.asarray(archive["y"], dtype=np.int64)


def auroc(positive_scores: np.ndarray[Any, Any], negative_scores: np.ndarray[Any, Any]) -> float:
    """Probability a random positive outscores a random negative (ties count half).

    Computed from mid-ranks of the pooled scores (the Mann-Whitney U identity).
    """
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUROC needs at least one positive and one negative score")
    pooled = np.concatenate([pos, neg])
    order = np.argsort(pooled, kind="stable")
    ranks = np.empty(pooled.size)
    start = 0
    while start < pooled.size:
        stop = start
        while stop + 1 < pooled.size and pooled[order[stop + 1]] == pooled[order[start]]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def stratified_split(
    labels: np.ndarray[Any, Any], seed: int, held_out: float = 0.3
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Train and held-out row indices, each class split alike (``benchmarks/dormant_module_revival.py``)."""
    rng = np.random.RandomState(seed)
    held: list[int] = []
    for cls in (0, 1):
        rows = np.where(labels == cls)[0]
        rng.shuffle(rows)
        held.extend(rows[: max(1, int(len(rows) * held_out))])
    mask = np.zeros(len(labels), dtype=bool)
    mask[sorted(held)] = True
    return np.where(~mask)[0], np.where(mask)[0]


def cwru(name: str) -> np.ndarray[Any, Any]:
    """A recorded CWRU drive-end vibration segment (12 kHz, in g) as float64.

    ``name`` is one of ``normal_0hp``, ``normal_1hp``, ``inner_race``, ``ball``,
    ``outer_race``; the faults are 0.007-inch at 0 hp.
    """
    with np.load(FIXTURES / "cwru" / f"{name}.npz", allow_pickle=False) as archive:
        return np.asarray(archive["x"], dtype=np.float64)


def hdfs_log_templates() -> tuple[np.ndarray[Any, Any], list[str]]:
    """Drain-parsed HDFS event-template keys ``(keys, event_ids)``.

    2000 real Hadoop HDFS log lines in file order, parsed into 14 event
    templates by Drain (LogHub's ``HDFS_2k`` sample); ``keys[i]`` indexes
    ``event_ids``.  Not anomaly-labelled -- LogHub's block-level labels ship
    with the full ``HDFS_1`` release -- so this supports a normal-regime
    false-alarm measurement and an injected-unseen-template check, not an
    AUROC.
    """
    with np.load(FIXTURES / "loghub" / "hdfs_templates.npz", allow_pickle=False) as archive:
        return (
            np.asarray(archive["keys"], dtype=np.int64),
            [str(event) for event in archive["event_ids"]],
        )


def eeg(record: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Recorded CHB-MIT scalp EEG (FP1-F7, 256 Hz, microvolts): ``(background, seizure)``.

    ``record`` is ``chb01_03`` or ``chb01_04``; ``background`` is the 300 s that
    end 120 s before the annotated seizure and ``seizure`` is the seizure itself.
    """
    with np.load(FIXTURES / "eeg" / f"{record}.npz", allow_pickle=False) as archive:
        return (
            np.asarray(archive["background"], dtype=np.float64),
            np.asarray(archive["seizure"], dtype=np.float64),
        )
