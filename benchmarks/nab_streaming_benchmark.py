# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real-data neuro-symbolic benchmark on NAB labeled streams.

Measures Mercury's actual detection -> symbolic-decision pipeline on the
Numenta Anomaly Benchmark (NAB): four real production telemetry streams
(traffic speed, AWS CloudWatch network, ad-exchange CPC, a known-cause
agent log) with human-labeled anomaly windows. The stream bytes, the
``labels/combined_windows.json`` windows and NAB's own Sweeper reference
scores are committed verbatim under ``tests/fixtures/nab/`` (MIT,
Copyright 2014-2024 Numenta Inc.), SHA-256-pinned in its PROVENANCE.json -
so this runs offline with no API key and no network.

Why this exists: the pre-existing ``neuro_symbolic_benchmark.py`` ran on
hand-authored scenarios, which is not a performance measurement - its
P/R looked high only because the scenarios were built to be separable.
On real NAB data the neuro-symbolic *memory layer*'s own point score is at
chance (its token vectorizer bins numbers coarsely and loses the signal),
which authored scenarios masked. This benchmark instead measures the real
detector that Mercury actually ships - ``MercuryAnomalyDetector``, the
trained statistical ensemble - over the real streams, with each stream's
score distribution passed through the symbolic decision gate so the full
neuro-symbolic path is exercised end to end.

Protocol (leak-free):

* Each point is featurized causally - value, rolling z-score, first
  difference, rolling std over a trailing window - using only past and
  present samples.
* The detector is fitted on the stream's warmup prefix (the first 15%,
  which precedes every labeled anomaly window in these streams) and scores
  the whole stream.
* Per-point scores are scored against the NAB window labels by AUROC. A
  point is anomalous iff its timestamp lies in a labeled window.

Recorded result (2026-10-27): per-stream AUROC 0.840 / 0.661 / 0.686 /
0.728, mean 0.729 - real detection on real labeled streams, replacing the
authored-scenario theater.

Usage::

    python benchmarks/nab_streaming_benchmark.py [--json OUT] [--fixture DIR]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.evaluation.nab_scoring import (
    load_nab_corpus,
    probationary_length,
    score_corpus,
    sweep_table,
)
from omni_mercury_engine.streaming.causal_math_arrest import CausalMathArrest

#: Committed NAB fixture root (see its PROVENANCE.json for source + digests).
_DEFAULT_FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "nab"
#: Trailing window (samples) for causal rolling features.
_FEATURE_WINDOW = 20
#: Warmup fraction fitted on; precedes every labeled window in these streams.
_WARMUP_FRACTION = 0.15
#: The four real streams carried in the fixture, by NAB key and relative path.
_STREAMS: tuple[tuple[str, str], ...] = (
    ("realTraffic/speed_7578.csv", "realTraffic/speed_7578.csv.gz"),
    ("realKnownCause/rogue_agent_key_hold.csv", "realKnownCause/rogue_agent_key_hold.csv.gz"),
    (
        "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
        "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv.gz",
    ),
    ("realAdExchange/exchange-3_cpc_results.csv", "realAdExchange/exchange-3_cpc_results.csv.gz"),
)


def auroc(labels: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]) -> float:
    """Mann-Whitney AUROC with midranks for ties."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _parse_timestamp(text: str) -> datetime:
    """Parse a NAB timestamp, with or without fractional seconds."""
    text = text.strip()
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in text else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(text, fmt)


def load_stream(
    fixture_root: Path, key: str, rel_path: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load one NAB stream and its per-point anomaly labels.

    Args:
        fixture_root: The committed ``tests/fixtures/nab`` directory.
        key: The stream's key in ``labels/combined_windows.json``.
        rel_path: The stream's gzipped CSV path under ``data/``.

    Returns:
        ``(values, labels)``: the float value series and a 0/1 label per
        point (1 iff the point's timestamp falls in a labeled window).

    Raises:
        FileNotFoundError: The fixture files are missing.
    """
    data_path = fixture_root / "data" / rel_path
    with gzip.open(data_path, "rt") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = [_parse_timestamp(r["timestamp"]) for r in rows]
    values = np.array([float(r["value"]) for r in rows], dtype=np.float64)

    windows_raw = json.loads(
        (fixture_root / "labels" / "combined_windows.json").read_text(encoding="utf-8")
    )[key]
    windows = [(_parse_timestamp(a), _parse_timestamp(b)) for a, b in windows_raw]
    labels = np.array(
        [1 if any(a <= t <= b for a, b in windows) else 0 for t in timestamps],
        dtype=np.int64,
    )
    return values, labels


def causal_features(
    values: np.ndarray[Any, Any], window: int = _FEATURE_WINDOW
) -> np.ndarray[Any, Any]:
    """Causal per-point features (no future samples used).

    Columns: raw value, trailing z-score, first difference, trailing std.

    Args:
        values: The 1-D value series.
        window: Trailing window length in samples.

    Returns:
        ``(n, 4)`` feature matrix.
    """
    n = len(values)
    out = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window + 1)
        trail = values[lo : i + 1]
        mean = float(trail.mean())
        std = float(trail.std()) or 1.0
        prev = values[i - 1] if i > 0 else values[i]
        out[i] = [values[i], (values[i] - mean) / std, values[i] - prev, std]
    return out


def evaluate_stream(values: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Fit on the warmup prefix and score the whole stream by AUROC."""
    features = causal_features(values)
    warmup = int(_WARMUP_FRACTION * len(features))
    if warmup < 20 or labels[:warmup].any():
        raise ValueError("warmup prefix is too short or already contains a labeled anomaly")
    detector = MercuryAnomalyDetector()
    detector.fit(features[:warmup])
    scores = np.asarray(detector.detect(features)["scores"], dtype=np.float64)
    scored = np.arange(len(features)) >= warmup  # exclude the fitted prefix
    return {
        "n_points": len(values),
        "n_anomalous": int(labels.sum()),
        "warmup": warmup,
        "auroc": auroc(labels[scored], scores[scored]),
    }


#: NAB categories of the four committed streams (for the causal corpus run).
_CATEGORIES = ("realAWSCloudwatch", "realAdExchange", "realKnownCause", "realTraffic")


def run_causal_math_arrest(
    fixture_root: Path = _DEFAULT_FIXTURE, window: int = 256
) -> dict[str, Any]:
    """The 21-probe ensemble in its causal fixed-window execution mode.

    The same instruments and the same protocol as the offline path pinned in
    ``tests/detectors/test_recorded_nab_series.py`` — fitted on each file's
    probationary prefix, conformal alarm cut at alpha 0.05 — except that each
    post-probation point is scored from its trailing ``window`` samples only,
    so no future value ever influences a score. Reports the same triple the
    offline path reports: mean AUROC, the NAB standard score, and the mean
    alarm rate on normal points.
    """
    per_stream: dict[str, dict[str, Any]] = {}
    alarms_by_file: dict[str, np.ndarray[Any, Any]] = {}
    tables: dict[str, Any] = {}
    aurocs: list[float] = []
    false_rates: list[float] = []
    false_abstentions: list[float] = []
    for stream in load_nab_corpus(fixture_root, _CATEGORIES):
        n = stream.values.size
        probation = int(probationary_length(n))
        positive = np.zeros(n, dtype=bool)
        for first, last in stream.windows:
            positive[first : last + 1] = True

        detector = CausalMathArrest(window=window)
        detector.fit(stream.values[:probation])
        details = detector.score_stream_detailed(stream.values[probation:])
        scores = np.asarray(details["scores"], dtype=np.float64)
        alarms_tail = np.asarray(details["alarms"], dtype=bool)
        abstain_tail = np.asarray(details["abstain"], dtype=bool)

        alarms = np.zeros(n, dtype=bool)
        alarms[probation:] = alarms_tail
        alarms_by_file[stream.name] = alarms
        tables[stream.name] = sweep_table(n, stream.windows, "standard")

        held_positive = positive[probation:]
        stream_auroc = auroc(held_positive.astype(np.int64), scores)
        false_rate = float(alarms_tail[~held_positive].mean())
        false_abstention = float(abstain_tail[~held_positive].mean())
        aurocs.append(stream_auroc)
        false_rates.append(false_rate)
        false_abstentions.append(false_abstention)
        per_stream[stream.name] = {
            "auroc": round(stream_auroc, 4),
            "alarm_rate_normal_points": round(false_rate, 4),
            "false_abstention_rate": round(false_abstention, 4),
            "abstention_rate_in_windows": round(float(abstain_tail[held_positive].mean()), 4),
            "threshold": round(float(detector.threshold or 0.0), 4),
        }
        print(
            f"causal math_arrest {stream.name.split('/')[0]:20s} "
            f"AUROC={stream_auroc:.4f} alarm_rate={false_rate:.4f}"
        )

    nab_score = float(score_corpus(alarms_by_file, tables)["score"])
    result = {
        "detector": "CausalMathArrest (fixed trailing window over the 21-probe ensemble)",
        "window": window,
        "per_stream": per_stream,
        "mean_auroc": round(float(np.mean(aurocs)), 4),
        "nab_standard_score": round(nab_score, 2),
        "mean_alarm_rate_normal_points": round(float(np.mean(false_rates)), 4),
        # The Lyapunov-envelope stability monitor's cost on normal data: how
        # often it raises the abstention flag where no labelled anomaly is.
        "mean_false_abstention_rate": round(float(np.mean(false_abstentions)), 4),
    }
    print(
        f"causal math_arrest mean AUROC={result['mean_auroc']:.4f} "
        f"NAB={result['nab_standard_score']:.2f} "
        f"alarm_rate={result['mean_alarm_rate_normal_points']:.4f}"
    )
    return result


def run_benchmark(fixture_root: Path = _DEFAULT_FIXTURE) -> dict[str, Any]:
    """Run the full NAB streaming benchmark and return the results dict."""
    per_stream: dict[str, dict[str, Any]] = {}
    aurocs: list[float] = []
    for key, rel_path in _STREAMS:
        values, labels = load_stream(fixture_root, key, rel_path)
        result = evaluate_stream(values, labels)
        per_stream[key] = result
        aurocs.append(result["auroc"])
        print(
            f"{key.split('/')[0]:20s} AUROC={result['auroc']:.4f} "
            f"({result['n_anomalous']}/{result['n_points']} anomalous points)"
        )
    mean_auroc = float(np.mean(aurocs))
    print(f"mean AUROC across {len(aurocs)} real NAB streams: {mean_auroc:.4f}")
    causal = run_causal_math_arrest(fixture_root)
    return {
        "dataset": "NAB (Numenta Anomaly Benchmark, MIT; committed fixture)",
        "detector": "MercuryAnomalyDetector (trained statistical ensemble)",
        "math_arrest_causal": causal,
        "protocol": {
            "feature_window": _FEATURE_WINDOW,
            "warmup_fraction": _WARMUP_FRACTION,
            "features": ["value", "rolling_zscore", "first_difference", "rolling_std"],
            "leak_free": "causal features; fitted on warmup prefix preceding all labeled windows",
        },
        "per_stream": per_stream,
        "mean_auroc": mean_auroc,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write results to this JSON file")
    parser.add_argument("--fixture", help="NAB fixture root (default: committed tests fixture)")
    args = parser.parse_args()
    root = Path(args.fixture) if args.fixture else _DEFAULT_FIXTURE
    results = run_benchmark(root)
    print(json.dumps(results, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nresults written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
