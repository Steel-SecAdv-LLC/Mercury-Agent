# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Numenta Anomaly Benchmark (NAB) score, reimplemented exactly.

NAB (Lavin & Ahmad, 2015, arXiv:1510.03336; Ahmad et al., 2017,
Neurocomputing 262:134-147) scores a streaming detector by *when* it alarms,
not only whether: each labelled anomaly window is credited once, for its
earliest alarm, on a scaled sigmoid of how early in the window the alarm fell;
a window without an alarm costs ``fnWeight``; and every alarm outside a window
costs ``fpWeight`` times a sigmoid of its distance past the previous window.
The first 15% of each file (at most 750 points) is a probationary period that
is not scored. Corpus scores are normalized so the null detector (no alarms)
scores 0 and the perfect detector (an alarm at each window's first scorable
point) scores 100.

This module follows NAB's reference implementation (``nab/sweeper.py`` and
``config/profiles.json`` at commit ``ea702d75``) line for line, and its tests
hold it equal to that implementation on the real NAB corpus. It reads the
corpus in NAB's own layout (``data/<category>/<file>.csv`` with ``timestamp,
value`` columns, ``labels/combined_windows.json``).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "NAB_REAL_CATEGORIES",
    "PROFILES",
    "CostProfile",
    "NABResult",
    "NABStream",
    "SweepTable",
    "load_nab_corpus",
    "normalized_score",
    "probationary_length",
    "scaled_sigmoid",
    "score_alarms",
    "score_corpus",
    "sweep_table",
]

#: The five NAB categories of real (not synthetic) data.
NAB_REAL_CATEGORIES = (
    "realAWSCloudwatch",
    "realAdExchange",
    "realKnownCause",
    "realTraffic",
    "realTweets",
)


@dataclass(frozen=True)
class CostProfile:
    """One NAB application profile (``config/profiles.json``).

    Attributes:
        tp: Weight of a detected window (``tpWeight``).
        fp: Weight of a false alarm (``fpWeight``).
        fn: Cost of a missed window (``fnWeight``).
    """

    tp: float
    fp: float
    fn: float


#: NAB's three application profiles.
PROFILES: dict[str, CostProfile] = {
    "standard": CostProfile(tp=1.0, fp=0.11, fn=1.0),
    "reward_low_FP_rate": CostProfile(tp=1.0, fp=0.22, fn=1.0),
    "reward_low_FN_rate": CostProfile(tp=1.0, fp=0.11, fn=2.0),
}


def scaled_sigmoid(relative_position: float) -> float:
    """The ``scaledSigmoid`` of NAB: ~0.987 at a window's left edge, 0 at its right.

    Args:
        relative_position: -1 at the window's left edge, 0 at its right edge;
            positive values are distances past a window (false positives).

    Returns:
        ``2 * sigmoid(-5 x) - 1``, or -1 beyond 3.
    """
    if relative_position > 3.0:
        return -1.0
    return 2.0 / (1.0 + math.exp(5.0 * relative_position)) - 1.0


def probationary_length(n_points: int, probation_percent: float = 0.15) -> float:
    """Points at the start of a file that NAB does not score.

    Returns ``min(floor(probation_percent * n), probation_percent * 5000)``,
    kept a float exactly as NAB compares it (``i >= length``).
    """
    return min(math.floor(probation_percent * n_points), probation_percent * 5000)


@dataclass(frozen=True)
class NABStream:
    """One NAB data file with its labelled anomaly windows.

    Attributes:
        name: ``<category>/<file>.csv``, NAB's key for the file.
        timestamps: ``datetime64`` per point.
        values: The series.
        windows: ``(first, last)`` point indices of each window, inclusive,
            sorted and disjoint.
    """

    name: str
    timestamps: np.ndarray[Any, Any]
    values: np.ndarray[Any, Any]
    windows: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SweepTable:
    """What an alarm at each point of a file would score.

    Attributes:
        weighted: NAB's weighted sweep score per point (positive inside a
            window, negative outside).
        window_id: The window a point lies in, or -1.
        scorable: Whether the point is past the probationary period.
        n_windows: Windows NAB scores in this file (those with at least one
            scorable point).
        profile: The cost profile the table was built for.
        scored_spans: ``(first, last)`` scorable point indices of each scored
            window, inclusive.
        perfect: The raw score of the perfect detector on this file.
    """

    weighted: np.ndarray[Any, np.dtype[np.float64]]
    window_id: np.ndarray[Any, np.dtype[np.int64]]
    scorable: np.ndarray[Any, np.dtype[np.bool_]]
    n_windows: int
    profile: CostProfile
    scored_spans: tuple[tuple[int, int], ...] = ()
    perfect: float = 0.0


@dataclass(frozen=True)
class NABResult:
    """A file's (or corpus's) raw NAB score and counts.

    Attributes:
        raw: The raw score.
        detected: Windows with at least one alarm.
        missed: Windows without one.
        false_alarms: Scorable alarms outside every window.
    """

    raw: float
    detected: int
    missed: int
    false_alarms: int


def sweep_table(
    n_points: int,
    windows: Sequence[tuple[int, int]],
    profile: CostProfile | str = "standard",
    probation_percent: float = 0.15,
) -> SweepTable:
    """The ``Sweeper.calcSweepScore`` of NAB, as a per-point table.

    Args:
        n_points: Length of the file.
        windows: ``(first, last)`` inclusive point indices, sorted and disjoint.
        profile: A :class:`CostProfile` or a :data:`PROFILES` name.
        probation_percent: NAB's probationary fraction.

    Returns:
        The table :func:`score_alarms` scores alarms against.

    Raises:
        ValueError: Windows out of range, unsorted, overlapping, or one point
            wide (NAB divides by ``width - 1`` past a window).
    """
    cost = PROFILES[profile] if isinstance(profile, str) else profile
    previous_last = -1
    for first, last in windows:
        if not 0 <= first <= last < n_points or first <= previous_last:
            raise ValueError(f"windows must be sorted, disjoint and within [0, {n_points})")
        if last == first:
            raise ValueError("NAB windows must span at least two points")
        previous_last = last
    max_tp = scaled_sigmoid(-1.0)
    weighted = np.empty(n_points, dtype=np.float64)
    window_id = np.full(n_points, -1, dtype=np.int64)
    index = np.arange(n_points)
    prev_right: int | None = None
    prev_width = 0.0
    cursor = 0
    for w, (first, last) in enumerate((*windows, (n_points, n_points))):
        # Outside: from the cursor up to this window's first point.
        gap = index[cursor:first]
        if prev_right is None:
            weighted[cursor:first] = -1.0 * cost.fp
        else:
            positions = np.abs(prev_right - gap) / (prev_width - 1.0)
            weighted[cursor:first] = [scaled_sigmoid(p) * cost.fp for p in positions]
        if first >= n_points:
            break
        width = float(last - first + 1)
        inside = index[first : last + 1]
        positions = -(last - inside + 1) / width
        weighted[first : last + 1] = [scaled_sigmoid(p) * cost.tp / max_tp for p in positions]
        window_id[first : last + 1] = w
        prev_right, prev_width, cursor = last, width, last + 1
    scorable = index >= probationary_length(n_points, probation_percent)
    first_scorable = math.ceil(probationary_length(n_points, probation_percent))
    spans = []
    for first, last in windows:
        if max(first, first_scorable) <= last:
            spans.append((max(first, first_scorable), last))
    # The perfect detector alarms at each scored window's first scorable point.
    perfect = sum(max(-cost.fn, float(weighted[first])) for first, _ in spans)
    return SweepTable(weighted, window_id, scorable, len(spans), cost, tuple(spans), perfect)


def score_alarms(alarms: np.ndarray[Any, Any], table: SweepTable) -> NABResult:
    """Score one file's alarms (NAB's ``calcScoreByThreshold`` at one threshold).

    Args:
        alarms: Boolean per point.
        table: The file's :func:`sweep_table`.

    Returns:
        The raw score and counts.

    Raises:
        ValueError: ``alarms`` of the wrong length.
    """
    alarms = np.asarray(alarms, dtype=bool)
    if alarms.shape != table.weighted.shape:
        raise ValueError(f"expected {table.weighted.size} alarms, got {alarms.size}")
    active = alarms & table.scorable
    outside = active & (table.window_id < 0)
    false_alarm_cost = float(table.weighted[outside].sum())
    window_total, detected = 0.0, 0
    for first, last in table.scored_spans:
        hits = active[first : last + 1]
        if hits.any():
            window_total += max(
                -table.profile.fn, float(table.weighted[first : last + 1][hits].max())
            )
            detected += 1
        else:
            window_total -= table.profile.fn
    return NABResult(
        raw=window_total + false_alarm_cost,
        detected=detected,
        missed=table.n_windows - detected,
        false_alarms=int(outside.sum()),
    )


def normalized_score(raw: float, null: float, perfect: float) -> float:
    """Normalize as NAB does: 0 for the null detector, 100 for the perfect one."""
    return 100.0 * (raw - null) / (perfect - null)


def score_corpus(
    alarms_by_stream: Mapping[str, np.ndarray[Any, Any]],
    tables: Mapping[str, SweepTable],
) -> dict[str, Any]:
    """Score a set of files together, as NAB scores a detector over its corpus.

    Args:
        alarms_by_stream: Alarms per file name.
        tables: Sweep tables per file name (the same names).

    Returns:
        ``score`` (normalized), ``raw``, ``null``, ``perfect``, ``detected``,
        ``missed``, ``false_alarms`` and ``per_stream`` raw results.

    Raises:
        KeyError: A file without a table.
    """
    raw = null = perfect = 0.0
    detected = missed = false_alarms = 0
    per_stream: dict[str, NABResult] = {}
    for name, alarms in alarms_by_stream.items():
        table = tables[name]
        result = score_alarms(alarms, table)
        per_stream[name] = result
        raw += result.raw
        detected += result.detected
        missed += result.missed
        false_alarms += result.false_alarms
        null += -table.profile.fn * table.n_windows
        perfect += table.perfect
    return {
        "score": normalized_score(raw, null, perfect),
        "raw": raw,
        "null": null,
        "perfect": perfect,
        "detected": detected,
        "missed": missed,
        "false_alarms": false_alarms,
        "per_stream": per_stream,
    }


def _read_csv(path: Path) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    timestamps = np.array([row["timestamp"] for row in rows], dtype="datetime64[us]")
    values = np.array([float(row["value"]) for row in rows], dtype=np.float64)
    return timestamps, values


def load_nab_corpus(
    root: str | Path,
    categories: Iterable[str] = NAB_REAL_CATEGORIES,
) -> list[NABStream]:
    """Read NAB files and their windows from a checkout or download of NAB.

    Args:
        root: Directory holding ``data/`` and ``labels/combined_windows.json``;
            a data file may also be stored gzip-compressed (``<file>.csv.gz``).
        categories: Categories to read (default: the five real ones).

    Returns:
        One :class:`NABStream` per file, sorted by name.

    Raises:
        FileNotFoundError: The labels file or a category directory is missing.
        ValueError: A window edge that is not one of its file's timestamps.
    """
    root = Path(root)
    labels = json.loads((root / "labels" / "combined_windows.json").read_text(encoding="utf-8"))
    streams = []
    for category in categories:
        folder = root / "data" / category
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted([*folder.glob("*.csv"), *folder.glob("*.csv.gz")]):
            name = f"{category}/{path.name.removesuffix('.gz')}"
            timestamps, values = _read_csv(path)
            lookup = {t: i for i, t in enumerate(timestamps.tolist())}
            windows = []
            for start, end in labels.get(name, []):
                edges = np.array([start, end], dtype="datetime64[us]").tolist()
                if edges[0] not in lookup or edges[1] not in lookup:
                    raise ValueError(f"{name}: window edge {start} / {end} is not a timestamp")
                windows.append((lookup[edges[0]], lookup[edges[1]]))
            streams.append(NABStream(name, timestamps, values, tuple(windows)))
    return sorted(streams, key=lambda s: s.name)
