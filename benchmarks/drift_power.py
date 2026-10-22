#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure what the drift ensemble flags, with and without the principal-axis test.

Per-feature KS and PSI compare one column at a time, so in correlated data a
shift along a low-variance direction moves every column by a fraction of its
spread. Each scenario draws a reference from a correlated Gaussian
(``corr = rho ** |i - j|``), then windows of ``window`` rows from the same
Gaussian shifted along its weakest principal axis by ``shift`` standard
deviations of that axis (0 = no drift, which gives the false-alarm rate).
Reported: the share of windows ``EnsembleDriftDetector`` flags, without
(``principal_axes=False``) and with the principal-axis test.

Usage::

    python benchmarks/drift_power.py                # writes benchmarks/results/drift_power.json
    python benchmarks/drift_power.py --trials 200 --references 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "drift_power.json"

# (features, correlation, reference rows, window rows)
SCENARIOS = [
    (8, 0.9, 2000, 64),
    (8, 0.9, 2000, 200),
    (8, 0.9, 2000, 1000),
    (32, 0.9, 2000, 200),
    (8, 0.0, 2000, 200),
    (128, 0.9, 1000, 300),
    (128, 0.95, 400, 300),
]
SHIFTS = [0.0, 0.5, 1.0, 2.0]


def scenario(
    features: int, rho: float, n_ref: int, window: int, trials: int, references: int, seed: int
) -> dict[str, object]:
    """Flag rates for one design across the shifts, over independent references.

    One reference sample sets where its detector's thresholds sit, so a single
    draw can move every rate in the cell; ``trials`` windows are split across
    ``references`` independent references to average that out.
    """
    from omni_mercury_engine.detectors.drift import EnsembleDriftDetector

    cov = rho ** np.abs(np.subtract.outer(np.arange(features), np.arange(features)))
    chol = np.linalg.cholesky(cov)
    values, vectors = np.linalg.eigh(cov)
    rng = np.random.default_rng(seed + features * 1000 + window + n_ref)
    per_reference = max(1, trials // references)
    flagged = {shift: np.zeros(2) for shift in SHIFTS}
    for _ in range(references):
        reference = rng.normal(size=(n_ref, features)) @ chol.T
        detectors = (
            EnsembleDriftDetector(principal_axes=False).fit(reference),
            EnsembleDriftDetector().fit(reference),
        )
        for shift in SHIFTS:
            offset = shift * np.sqrt(values[0]) * vectors[:, 0]
            for _ in range(per_reference):
                current = rng.normal(size=(window, features)) @ chol.T + offset
                flagged[shift] += [d.detect(current).is_drift for d in detectors]
    windows = per_reference * references
    return {
        "features": features,
        "correlation": rho,
        "reference_rows": n_ref,
        "window_rows": window,
        "windows_per_cell": windows,
        "max_single_feature_shift_sd_at_1sd": float(
            np.abs(np.sqrt(values[0]) * vectors[:, 0]).max()
        ),
        "flagged_pct_by_shift_sd": {
            str(shift): {
                "without_principal_axes_pct": 100 * counts[0] / windows,
                "with_principal_axes_pct": 100 * counts[1] / windows,
            }
            for shift, counts in flagged.items()
        },
    }


SPLIT_CHECKS = [(64, 0.9, 500, 200), (128, 0.9, 1000, 300), (128, 0.95, 400, 300)]


def split_check(
    features: int, rho: float, n_ref: int, window: int, trials: int, references: int, seed: int
) -> dict[str, object]:
    """False alarms with no drift: axes and distribution from the same rows vs split.

    ``PrincipalAxisDriftDetector`` learns its axes on half the reference and takes
    the distribution from the other half. Learning both from the same rows lets the
    axes follow the reference's own noise, so fresh windows look drifted.
    """
    from scipy import stats

    from omni_mercury_engine.detectors.drift import PrincipalAxisDriftDetector

    cov = rho ** np.abs(np.subtract.outer(np.arange(features), np.arange(features)))
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed + features * 1000 + window + n_ref + 1)
    per_reference = max(1, trials // references)
    flagged = np.zeros(2)
    for _ in range(references):
        reference = rng.normal(size=(n_ref, features)) @ chol.T
        split = PrincipalAxisDriftDetector().fit(reference)
        center = reference.mean(axis=0)
        axes = np.linalg.svd(reference - center, full_matrices=False)[2].T
        same_rows = (reference - center) @ axes
        threshold = split.p_value_threshold / features
        for _ in range(per_reference):
            current = rng.normal(size=(window, features)) @ chol.T
            scores = (current - center) @ axes
            smallest = min(
                stats.ks_2samp(same_rows[:, k], scores[:, k]).pvalue for k in range(features)
            )
            flagged += [smallest < threshold, split.detect(current).is_drift]
    windows = per_reference * references
    return {
        "features": features,
        "correlation": rho,
        "reference_rows": n_ref,
        "window_rows": window,
        "windows": windows,
        "false_alarm_pct_same_rows": 100 * flagged[0] / windows,
        "false_alarm_pct_split": 100 * flagged[1] / windows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=100, help="windows per cell")
    parser.add_argument("--references", type=int, default=5, help="independent references")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    logging.disable(logging.WARNING)
    started = time.perf_counter()
    results = [scenario(*design, args.trials, args.references, args.seed) for design in SCENARIOS]
    splits = [
        split_check(*design, args.trials, args.references, args.seed) for design in SPLIT_CHECKS
    ]
    document = {
        "schema": "mercury.benchmarks.drift_power/v1",
        "design": {
            "trials": args.trials,
            "references": args.references,
            "seed": args.seed,
            "shifts_sd": SHIFTS,
        },
        "scenarios": results,
        "split_check": splits,
        "host": {
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"{'d':>4} {'rho':>4} {'ref':>5} {'win':>5} {'shift':>6} | {'without':>8} {'with':>6}  (% of windows flagged)"
    )
    for r in results:
        for shift, rate in r["flagged_pct_by_shift_sd"].items():  # type: ignore[attr-defined]
            print(
                f"{r['features']:>4} {r['correlation']:>4} {r['reference_rows']:>5} {r['window_rows']:>5} "
                f"{shift:>6} | {rate['without_principal_axes_pct']:8.1f} {rate['with_principal_axes_pct']:6.1f}"
            )
    print(
        "\nfalse alarms with no drift, axes learned on the same rows vs on a split of the reference:"
    )
    for r in splits:
        print(
            f"{r['features']:>4} {r['correlation']:>4} {r['reference_rows']:>5} {r['window_rows']:>5} | "
            f"same rows {r['false_alarm_pct_same_rows']:5.1f}%   split {r['false_alarm_pct_split']:5.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
