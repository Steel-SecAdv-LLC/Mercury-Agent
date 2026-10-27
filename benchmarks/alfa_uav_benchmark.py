# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ALFA UAV fault-detection benchmark - real injected faults, recorded protocol.

Measures Mercury on the ALFA dataset (DOI 10.1184/R1/12707963): 46 real
fixed-wing flights, 36 with a mid-flight injected actuator fault (engine,
aileron, rudder, elevator) whose injection instant is recorded by the
onboard ``failure_status`` topics - event ground truth, not thresholds.

Protocol (leak-free, deployable):

* Per flight, :class:`MercuryAnomalyDetector` is fitted on the flight's
  first 30 s (every fault is injected at >= 49 s) and scores the rest.
* The stuck-actuator statistic
  (:mod:`omni_mercury_engine.detectors.drone.actuator_health`) runs over
  the five servo PWM channels against the 30 s baseline window's std.
* Decision-level OR fusion: each score's threshold is the 99.5th
  percentile of the pooled post-baseline ticks of the ten NO-FAILURE
  flights only. Fault flights never influence any threshold.

Recorded result (2026-10-27, 4 Hz grid, window 32):

* per-tick AUROC - ensemble: engines 0.906, rudder 0.994, elevator 0.924,
  aileron 0.601; stuck statistic: aileron 0.907 (the classes complement).
* OR fusion: 35/36 fault flights detected, joint normal FPR 0.68%,
  median detection delay 8.0 s (p90 8.4 s) - the window length dominates.

Usage::

    python benchmarks/alfa_uav_benchmark.py [--data-dir PATH] [--json OUT]

``--data-dir`` points at an extracted ALFA ``processed/`` root; without it
the loader downloads the pinned archive (272 MB).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.uav_flight import ALFASequence, ALFAUAVLoader
from omni_mercury_engine.detectors.drone.actuator_health import (
    DEFAULT_STUCK_WINDOW,
    stuck_actuator_scores,
)
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

#: Per-flight baseline window (seconds). Every ALFA fault is injected at
#: >= 49 s into its flight, so the first 30 s are always pre-fault.
BASELINE_S = 30.0
#: Pooled-normal percentile for each alarm's threshold (0.5% per-tick FPR).
THRESHOLD_PERCENTILE = 99.5
#: Columns of the servo PWM channels in ALFAUAVLoader.FEATURE_NAMES.
RC_SLICE = slice(22, 27)


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


def score_flight(
    sequence: ALFASequence,
) -> (
    tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    | None
):
    """Score one flight with both alarms against its own 30 s baseline.

    Returns:
        ``(times, ensemble_scores, stuck_scores, labels)`` for the
        post-baseline ticks, or None when the flight is too short for the
        protocol (under ~40 baseline ticks or nothing after the baseline).

    Raises:
        ValueError: Ground truth overlaps the baseline window (would leak).
    """
    cut = int(np.searchsorted(sequence.times_s, sequence.times_s[0] + BASELINE_S))
    if cut < 40 or cut >= len(sequence.labels):
        return None
    if sequence.labels[:cut].any():
        raise ValueError(f"{sequence.name}: fault inside the baseline window")
    detector = MercuryAnomalyDetector()
    detector.fit(sequence.features[:cut])
    ensemble = np.asarray(detector.detect(sequence.features[cut:])["scores"], dtype=np.float64)
    pwm = sequence.features[:, RC_SLICE]
    stuck = stuck_actuator_scores(
        pwm, baseline_std=pwm[:cut].std(axis=0), window=DEFAULT_STUCK_WINDOW
    )[cut:]
    return sequence.times_s[cut:], ensemble, stuck, sequence.labels[cut:]


def run_benchmark(data_dir: str | None) -> dict[str, Any]:
    """Run the full recorded protocol and return the results dictionary."""
    preprocessing: dict[str, Any] = {}
    if data_dir:
        preprocessing["local_dir"] = data_dir
    config = DatasetConfig(name="alfa_uav", preprocessing=preprocessing)
    loader = ALFAUAVLoader(config)
    loader.download()
    sequences = loader.load_sequences()
    normal = [s for s in sequences if not s.fault_surfaces]
    faulted = [s for s in sequences if s.fault_surfaces]
    print(f"ALFA: {len(normal)} no-failure flights, {len(faulted)} fault flights")

    skipped: list[str] = []
    normal_scored = []
    for s in normal:
        scored = score_flight(s)
        if scored is None:
            skipped.append(s.name)
        else:
            normal_scored.append(scored)
    pooled_ens = np.concatenate([s[1] for s in normal_scored])
    pooled_stuck = np.concatenate([s[2] for s in normal_scored])
    thr_ens = float(np.percentile(pooled_ens, THRESHOLD_PERCENTILE))
    thr_stuck = float(np.percentile(pooled_stuck, THRESHOLD_PERCENTILE))
    joint_fpr = float(((pooled_ens >= thr_ens) | (pooled_stuck >= thr_stuck)).mean())

    by_type: dict[str, dict[str, list[np.ndarray[Any, Any]]]] = defaultdict(
        lambda: {"labels": [], "ens": [], "stuck": []}
    )
    delays: list[float] = []
    missed: list[str] = []
    for sequence in faulted:
        scored = score_flight(sequence)
        if scored is None:
            skipped.append(sequence.name)
            continue
        times, ens, stuck, labels = scored
        if not labels.any():
            # Fault window entirely outside the scored ticks: np.argmax
            # would silently return 0 and count a pre-fault alarm as a
            # detection. No such flight exists in the current corpus.
            skipped.append(f"{sequence.name} (no scored fault tick)")
            continue
        key = "+".join(sequence.fault_surfaces)
        by_type[key]["labels"].append(labels)
        by_type[key]["ens"].append(ens)
        by_type[key]["stuck"].append(stuck)
        injection = int(np.argmax(labels == 1))
        alarms = np.nonzero((ens >= thr_ens) | (stuck >= thr_stuck))[0]
        alarms = alarms[alarms >= injection]
        if len(alarms):
            delays.append(float(times[alarms[0]] - times[injection]))
        else:
            missed.append(sequence.name)

    per_type: dict[str, dict[str, float]] = {}
    all_labels = [np.concatenate(v["labels"]) for v in by_type.values()]
    all_ens = [np.concatenate(v["ens"]) for v in by_type.values()]
    all_stuck = [np.concatenate(v["stuck"]) for v in by_type.values()]
    for key, parts in sorted(by_type.items()):
        labels = np.concatenate(parts["labels"])
        per_type[key] = {
            "n_flights": len(parts["labels"]),
            "auroc_ensemble": auroc(labels, np.concatenate(parts["ens"])),
            "auroc_stuck": auroc(labels, np.concatenate(parts["stuck"])),
        }
    delay_arr = np.asarray(delays, dtype=np.float64)
    results: dict[str, Any] = {
        "dataset": "ALFA (DOI 10.1184/R1/12707963)",
        "protocol": {
            "baseline_s": BASELINE_S,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "stuck_window": DEFAULT_STUCK_WINDOW,
            "sample_hz": loader.sample_hz,
        },
        "thresholds": {"ensemble": thr_ens, "stuck": thr_stuck},
        "joint_normal_fpr": joint_fpr,
        "auroc_overall_ensemble": auroc(np.concatenate(all_labels), np.concatenate(all_ens)),
        "auroc_overall_stuck": auroc(np.concatenate(all_labels), np.concatenate(all_stuck)),
        "per_fault_type": per_type,
        "detected_flights": len(delays),
        "missed_flights": missed,
        "skipped_too_short": skipped,
        "delay_median_s": float(np.median(delay_arr)) if len(delay_arr) else None,
        "delay_p90_s": float(np.percentile(delay_arr, 90)) if len(delay_arr) else None,
    }
    return results


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="extracted ALFA processed/ root")
    parser.add_argument("--json", help="write results to this JSON file")
    args = parser.parse_args()
    results = run_benchmark(args.data_dir)
    print(json.dumps(results, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nresults written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
