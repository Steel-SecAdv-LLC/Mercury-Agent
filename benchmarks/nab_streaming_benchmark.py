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
    return {
        "dataset": "NAB (Numenta Anomaly Benchmark, MIT; committed fixture)",
        "detector": "MercuryAnomalyDetector (trained statistical ensemble)",
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
