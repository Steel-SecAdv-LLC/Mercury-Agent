# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""NAB competitive streaming benchmark: Mercury streaming tier vs baselines.

Runs Mercury's :class:`StreamingScoreEnsemble` (``consensus`` and ``average``
combiners over the 1-D tier members, the exact unsupervised protocol of
``benchmarks/detection_tier_benchmark.py``) head-to-head against external
baselines executed in the SAME harness on ALL FIVE real NAB categories
(``realKnownCause``, ``realAWSCloudwatch``, ``realTraffic``,
``realAdExchange``, ``realTweets`` -- synthetic ``artificial*`` sets are
excluded).

Same-harness baselines (identical warm-up, identical series, identical
metrics -- no baseline is quoted, all are run):

* ``iforest_windowed`` -- PyOD IsolationForest on sliding-window embeddings
  (window 32, library defaults, fixed seed), fit on the warm-up windows.
* ``ewma_zscore`` -- a causal EWMA mean/variance z-score detector
  (alpha 0.05, fixed for every stream; the classic streaming baseline).
* ``random`` -- seeded uniform scores: the floor any real detector must beat.
* ``perfect`` -- the labels replayed as scores: the harness ceiling (AUC 1.0
  by construction; validates the metric plumbing, not a competitor).

Protocol per stream (identical for every method): fit/initialise on the
initial 15% warm-up window (min 200 points), then score the WHOLE series
causally; long series are anomaly-preservingly cropped to 6000 points
(see ``detection_tier_benchmark._crop_to_anomaly``). Metrics:

* point-wise ROC-AUC via the Mann-Whitney rank identity
  (``detection_tier_benchmark._roc_auc``), and
* a NAB-style *window detection rate*: ground-truth anomaly windows are the
  contiguous label==1 runs; every method gets the same alarm budget (its
  top-q scores, q = the stream's labelled anomaly rate) and a window counts
  as detected when at least one alarm lands inside it. ``alarm_precision``
  is the fraction of alarm points inside any window.

NAB's own published scoreboard is *quoted* in the output metadata for context
(citation included) but those numbers are NAB scores from NAB's own scorer --
NOT comparable to the same-harness AUC / detection-rate measurements here,
and they are kept in a clearly separated section.

Usage::

    python benchmarks/nab_competitive.py                 # all 58 real streams
    python benchmarks/nab_competitive.py --max-files 5   # smoke run

Output:
    benchmarks/nab_competitive_results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import numpy as np

import detection_tier_benchmark as dtb

SEED = 42
WINDOW = 32  # sliding-window length for the windowed IForest baseline (fixed, untuned)
EWMA_ALPHA = 0.05  # EWMA smoothing (fixed, untuned, same for every stream)
OUTPUT_PATH = _HERE / "nab_competitive_results.json"

#: All five real NAB categories (the synthetic ``artificial*`` sets stay out).
NAB_REAL_CATEGORIES: tuple[str, ...] = (
    "realKnownCause",
    "realAWSCloudwatch",
    "realTraffic",
    "realAdExchange",
    "realTweets",
)

#: Method registry order (report order). Mercury first, then baselines,
#: then the random/perfect references.
METHOD_ORDER: tuple[str, ...] = (
    "mercury_consensus",
    "mercury_average",
    "iforest_windowed",
    "ewma_zscore",
    "random",
    "perfect",
)

#: NAB's published scoreboard (standard profile), QUOTED for context only.
#: These are NAB scores computed by NAB's own scorer over its full corpus --
#: a different metric on a different protocol, NOT comparable to the
#: same-harness AUC / window-detection-rate measurements in this file.
#: Source: https://github.com/numenta/NAB#scoreboard (retrieved 2026-07-10).
#: Citation: Ahmad, Lavin, Purdy & Agha (2017), "Unsupervised real-time
#: anomaly detection for streaming data", Neurocomputing 262:134-147.
PUBLISHED_NAB_SCOREBOARD_STANDARD: dict[str, Any] = {
    "Perfect": 100.0,
    "ARTime": 74.9,
    "Numenta HTM": "70.5-69.7",
    "CAD OSE": 69.9,
    "earthgecko Skyline": 58.2,
    "KNN CAD": 58.0,
    "Relative Entropy": 54.6,
    "Random Cut Forest": 51.7,
    "Twitter ADVec v1.0.0": 47.1,
    "Windowed Gaussian": 39.6,
    "Etsy Skyline": 35.7,
    "Bayesian Changepoint": 17.7,
    "EXPoSE": 16.4,
    "Random": 11.0,
    "Null": 0.0,
}


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Baseline detectors (same harness: fit on warm-up, score whole series)
# ---------------------------------------------------------------------------


def _window_embed(series: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window embedding: row ``i`` = the ``window`` points ending at ``i``.

    Left edge is padded with the first value so every point gets a row and the
    scores align 1:1 with the series (causal: row ``i`` never sees ``t > i``).
    """
    padded = np.concatenate([np.full(window - 1, series[0], dtype=np.float64), series])
    idx = np.arange(series.size)[:, None] + np.arange(window)[None, :]
    return padded[idx]


def _iforest_windowed_scores(series: np.ndarray, warmup: int, seed: int) -> np.ndarray:
    """PyOD IsolationForest on sliding windows: fit on warm-up rows, score all."""
    from pyod.models.iforest import IForest

    X = _window_embed(np.asarray(series, dtype=np.float64), WINDOW)
    model = IForest(random_state=seed)
    model.fit(X[:warmup])
    return np.asarray(model.decision_function(X), dtype=np.float64).ravel()


def _ewma_zscore_scores(series: np.ndarray, warmup: int) -> np.ndarray:
    """Causal EWMA z-score: |x_t - ewma| / ewm_std with fixed alpha.

    Mean/variance are initialised from the warm-up window and updated only
    with past points, so the score at ``t`` never sees ``x_{t'>t-1}``.
    """
    x = np.asarray(series, dtype=np.float64)
    mean = float(np.mean(x[:warmup]))
    var = float(np.var(x[:warmup])) + 1e-12
    scores = np.zeros(x.size, dtype=np.float64)
    a = EWMA_ALPHA
    for t in range(x.size):
        scores[t] = abs(x[t] - mean) / np.sqrt(var)
        delta = x[t] - mean
        mean += a * delta
        var = (1.0 - a) * (var + a * delta * delta)
    return scores


def _mercury_ensemble_scores(
    series: np.ndarray, warmup: int, seed: int, method: str
) -> np.ndarray:
    """Mercury StreamingScoreEnsemble: fit members on warm-up, score everything.

    Exactly the unsupervised protocol of
    ``detection_tier_benchmark._evaluate_unsupervised_ensemble``.
    """
    from omni_mercury_engine.detectors.detection_tier import (
        StreamingScoreEnsemble,
        build_tier_detectors,
    )

    detectors = build_tier_detectors(list(dtb.MEMBER_DETECTORS))
    ensemble = StreamingScoreEnsemble(detectors, method=method, seed=seed)
    ensemble.fit(series[:warmup])
    return np.asarray(ensemble.score(series), dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _label_windows(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous label==1 runs as half-open ``[start, end)`` windows."""
    labels = np.asarray(labels, dtype=np.int64).ravel()
    padded = np.concatenate([[0], labels, [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, edges.size, 2)]


def window_detection_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """NAB-style window detection at a budget-matched alarm rate.

    The alarm threshold is the ``(1 - anomaly_rate)`` quantile of the method's
    own scores, so every method fires the same number of alarms (the labelled
    anomaly budget) and no method gets an oracle threshold.

    Returns:
        ``window_detection_rate`` (fraction of ground-truth windows containing
        at least one alarm), ``n_windows``, and ``alarm_precision`` (fraction
        of alarm points inside any window).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    windows = _label_windows(labels)
    rate = float(labels.mean())
    if not windows or rate <= 0.0:
        return {"window_detection_rate": 0.0, "n_windows": 0, "alarm_precision": 0.0}
    threshold = float(np.quantile(scores, 1.0 - rate))
    alarms = scores > threshold
    detected = sum(1 for lo, hi in windows if bool(alarms[lo:hi].any()))
    n_alarms = int(alarms.sum())
    in_window = int((alarms & (labels == 1)).sum())
    return {
        "window_detection_rate": detected / len(windows),
        "n_windows": len(windows),
        "alarm_precision": (in_window / n_alarms) if n_alarms else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-stream evaluation
# ---------------------------------------------------------------------------


def evaluate_stream(
    name: str, series: np.ndarray, labels: np.ndarray, *, seed: int = SEED
) -> dict[str, Any]:
    """Run every method on one NAB stream under the shared protocol."""
    warmup = dtb._warmup_len(series.size)
    rng = np.random.RandomState(seed)

    runners: dict[str, Any] = {
        "mercury_consensus": lambda: _mercury_ensemble_scores(series, warmup, seed, "consensus"),
        "mercury_average": lambda: _mercury_ensemble_scores(series, warmup, seed, "average"),
        "iforest_windowed": lambda: _iforest_windowed_scores(series, warmup, seed),
        "ewma_zscore": lambda: _ewma_zscore_scores(series, warmup),
        "random": lambda: rng.uniform(size=series.size),
        "perfect": lambda: labels.astype(np.float64),
    }

    methods: dict[str, dict[str, Any]] = {}
    for method, runner in runners.items():
        try:
            t0 = time.perf_counter()
            scores = runner()
            elapsed = time.perf_counter() - t0
            entry: dict[str, Any] = {
                "roc_auc": round(dtb._roc_auc(scores, labels), 6),
                "seconds": round(elapsed, 4),
            }
            entry.update(
                {k: round(float(v), 6) for k, v in window_detection_metrics(scores, labels).items()}
            )
            methods[method] = entry
        except Exception as exc:
            methods[method] = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "name": name,
        "n": int(series.size),
        "anomaly_rate": round(float(labels.mean()), 6),
        "warmup": int(warmup),
        "methods": methods,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize(streams: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/median per method + Mercury wins/losses vs each baseline."""
    per_method: dict[str, Any] = {}
    for m in METHOD_ORDER:
        aucs, dets = [], []
        errors = []
        for s in streams:
            r = s["methods"].get(m, {})
            if "error" in r:
                errors.append({"stream": s["name"], "error": r["error"]})
            elif "roc_auc" in r:
                aucs.append(r["roc_auc"])
                dets.append(r["window_detection_rate"])
        per_method[m] = {
            "n_streams": len(aucs),
            "mean_auc": round(float(np.mean(aucs)), 6) if aucs else None,
            "median_auc": round(float(np.median(aucs)), 6) if aucs else None,
            "mean_window_detection_rate": round(float(np.mean(dets)), 6) if dets else None,
            "errors": errors,
        }

    # Per-category means for the Mercury headline method.
    per_category: dict[str, Any] = {}
    for s in streams:
        cat = s["name"].split("/", 1)[0]
        per_category.setdefault(cat, {"n": 0, "_aucs": {m: [] for m in METHOD_ORDER}})
        per_category[cat]["n"] += 1
        for m in METHOD_ORDER:
            r = s["methods"].get(m, {})
            if "roc_auc" in r:
                per_category[cat]["_aucs"][m].append(r["roc_auc"])
    for cat, entry in per_category.items():
        entry["mean_auc"] = {
            m: (round(float(np.mean(v)), 6) if v else None) for m, v in entry["_aucs"].items()
        }
        del entry["_aucs"]

    # Head-to-head: each Mercury combiner vs each real baseline (the
    # random/perfect references are context, not competitors, but are
    # included so the table is complete).
    head_to_head: dict[str, Any] = {}
    for mm in ("mercury_consensus", "mercury_average"):
        vs: dict[str, Any] = {}
        for bm in METHOD_ORDER:
            if bm in ("mercury_consensus", "mercury_average"):
                continue
            wins = losses = ties = 0
            loss_list: list[dict[str, Any]] = []
            for s in streams:
                a = s["methods"].get(mm, {})
                b = s["methods"].get(bm, {})
                if "roc_auc" not in a or "roc_auc" not in b:
                    continue
                d = a["roc_auc"] - b["roc_auc"]
                if d > 0:
                    wins += 1
                elif d < 0:
                    losses += 1
                    loss_list.append({"stream": s["name"], "auc_margin": round(-d, 6)})
                else:
                    ties += 1
            loss_list.sort(key=lambda e: -e["auc_margin"])
            vs[bm] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "worst_losses": loss_list[:5],
            }
        head_to_head[mm] = vs

    return {
        "n_streams": len(streams),
        "per_method": per_method,
        "per_category": per_category,
        "head_to_head": head_to_head,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    *,
    seed: int = SEED,
    max_files: int | None = None,
    categories: tuple[str, ...] = NAB_REAL_CATEGORIES,
) -> dict[str, Any]:
    """Run the NAB competitive benchmark and return the results dict."""
    print("=" * 78)
    print("NAB competitive benchmark -- Mercury streaming tier vs same-harness baselines")
    print(f"  categories={list(categories)}  seed={seed}")
    print("=" * 78)

    datasets = dtb.load_nab_series(categories, max_files=max_files)
    t_start = time.perf_counter()
    streams: list[dict[str, Any]] = []
    for name, series, labels in datasets:
        row = evaluate_stream(name, series, labels, seed=seed)
        streams.append(row)
        aucs = {
            m: r.get("roc_auc") for m, r in row["methods"].items() if "roc_auc" in r
        }
        best = max(aucs, key=lambda k: aucs[k]) if aucs else None
        print(
            f"  [{name}] n={row['n']} rate={row['anomaly_rate']:.3f} best={best} "
            + "  ".join(f"{m}={v:.3f}" for m, v in aucs.items())
        )
    runtime_s = time.perf_counter() - t_start

    summary = summarize(streams)

    return {
        "metadata": {
            "benchmark": "nab_competitive (Mercury streaming tier vs same-harness baselines)",
            "source": "NAB (Numenta Anomaly Benchmark) -- ALL five real categories",
            "loader": "omni_mercury_engine.datasets.timeseries.NABLoader",
            "categories": list(categories),
            "license": "AGPL-3.0",
            "protocol": (
                "unsupervised streaming: every method is fitted/initialised on the initial "
                "15% warm-up window (min 200 points) and scores the whole series; long "
                "series anomaly-preservingly cropped to 6000 points; point-wise Mann-"
                "Whitney ROC-AUC + budget-matched NAB-style window detection rate; same "
                "series, warm-up, and metrics for every method"
            ),
            "members": list(dtb.MEMBER_DETECTORS),
            "baseline_config": {
                "iforest_windowed": {"window": WINDOW, "pyod_defaults": True, "seed": seed},
                "ewma_zscore": {"alpha": EWMA_ALPHA},
                "random": {"seeded_uniform": True},
                "perfect": {"labels_as_scores": True, "role": "harness ceiling, not a competitor"},
            },
            "seed": seed,
            "max_len": dtb._MAX_LEN,
            "warmup_frac": dtb._WARMUP_FRAC,
            "git_commit": _git_commit(),
            "command_line": " ".join(sys.argv),
            "timestamp": datetime.now(UTC).isoformat(),
            "runtime_seconds": round(runtime_s, 1),
            "published_nab_scoreboard_standard_profile": {
                "NOT_same_harness": (
                    "QUOTED for context only. These are NAB scores from NAB's own scorer "
                    "over its full corpus -- a different metric and protocol; do not "
                    "compare them numerically to the AUC / detection-rate measurements "
                    "in this file."
                ),
                "source": "https://github.com/numenta/NAB#scoreboard (retrieved 2026-07-10)",
                "citation": (
                    "Ahmad, Lavin, Purdy & Agha (2017). Unsupervised real-time anomaly "
                    "detection for streaming data. Neurocomputing 262:134-147."
                ),
                "scores": PUBLISHED_NAB_SCOREBOARD_STANDARD,
            },
        },
        "summary": summary,
        "per_stream": streams,
    }


def _print_summary(results: dict[str, Any]) -> None:
    """Print the aggregate table."""
    summary = results["summary"]
    print("\n" + "=" * 78)
    print(f"{'method':<20} {'n':>3} {'mean AUC':>9} {'med AUC':>9} {'win-det rate':>13}")
    print("-" * 78)
    for m in METHOD_ORDER:
        s = summary["per_method"][m]
        if s["mean_auc"] is None:
            print(f"{m:<20} {s['n_streams']:>3}  (no successful streams)")
            continue
        print(
            f"{m:<20} {s['n_streams']:>3} {s['mean_auc']:>9.4f} {s['median_auc']:>9.4f} "
            f"{s['mean_window_detection_rate']:>13.4f}"
        )
    print("-" * 78)
    for mm, vs in summary["head_to_head"].items():
        for bm, rec in vs.items():
            print(f"{mm} vs {bm}: {rec['wins']}W/{rec['losses']}L/{rec['ties']}T")


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-files", type=int, default=None, help="cap streams (smoke runs)")
    ap.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    results = run(seed=args.seed, max_files=args.max_files)
    _print_summary(results)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
