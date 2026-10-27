# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""UCI Robot Execution Failures benchmark - real manipulator faults, tiny-n protocol.

Measures Mercury on UCI dataset 138 (``datasets/robot_failures.py``): 463
real manipulator executions, each 15 force/torque samples, with the
observed task outcome as event ground truth.

The corpus is small (20-44 nominal runs per problem) and failure-heavy
(55-79%), so a single split would be noise. Protocol:

* per learning problem, over ``N_SPLITS`` seeded repetitions: fit
  :class:`MercuryAnomalyDetector` on a random half of the **nominal** runs
  only, score the held-out nominals plus every failure run, and take the
  AUROC;
* report mean +/- std over repetitions, per problem and pooled (the pooled
  figure weights problems by their scored sizes).

Semi-supervised novelty detection on genuine labels - nothing is tuned on
the failures.

Usage::

    python benchmarks/robot_failures_benchmark.py [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.robot_failures import RobotExecutionFailuresLoader
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

N_SPLITS = 20


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


def evaluate_problem(
    features: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], n_splits: int = N_SPLITS
) -> dict[str, Any]:
    """Repeated-split novelty evaluation of one learning problem."""
    nominal_idx = np.nonzero(labels == 0)[0]
    failure_idx = np.nonzero(labels == 1)[0]
    aurocs: list[float] = []
    for seed in range(n_splits):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(nominal_idx)
        n_fit = max(len(nominal_idx) // 2, 5)
        fit_idx, held_idx = shuffled[:n_fit], shuffled[n_fit:]
        if len(held_idx) == 0:
            continue
        detector = MercuryAnomalyDetector()
        detector.fit(features[fit_idx])
        test_idx = np.concatenate([held_idx, failure_idx])
        scores = np.asarray(detector.detect(features[test_idx])["scores"], dtype=np.float64)
        aurocs.append(auroc(labels[test_idx], scores))
    arr = np.asarray(aurocs, dtype=np.float64)
    return {
        "n_nominal": len(nominal_idx),
        "n_failure": len(failure_idx),
        "n_splits": len(arr),
        "auroc_mean": float(arr.mean()),
        "auroc_std": float(arr.std()),
    }


def run_benchmark() -> dict[str, Any]:
    """Run the full recorded protocol and return the results dictionary."""
    # All five labelings are measured per problem; the pooled figure skips
    # lp3, which relabels lp2's 47 executions (identical force/torque rows)
    # and would double-count them.
    loader = RobotExecutionFailuresLoader(
        DatasetConfig(
            name="robot_failures",
            preprocessing={"problems": RobotExecutionFailuresLoader.PROBLEMS},
        )
    )
    loader.download()
    per_problem: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0
    weight = 0
    for problem, (features, labels, _names) in loader.load_problem_arrays().items():
        result = evaluate_problem(features, labels)
        per_problem[problem] = result
        print(
            f"{problem}: AUROC {result['auroc_mean']:.4f} +/- {result['auroc_std']:.4f} "
            f"({result['n_nominal']} nominal / {result['n_failure']} failures, "
            f"{result['n_splits']} splits)"
        )
        if problem == "lp3":
            continue
        scored = result["n_failure"] + result["n_nominal"] - result["n_nominal"] // 2
        weighted_sum += result["auroc_mean"] * scored
        weight += scored
    pooled = weighted_sum / weight if weight else float("nan")
    print(f"pooled (size-weighted, lp3 excluded as a relabeling of lp2): {pooled:.4f}")
    return {
        "dataset": "UCI Robot Execution Failures (138, DOI 10.24432/C5M89N)",
        "protocol": {
            "n_splits": N_SPLITS,
            "fit": "half of the nominal runs per problem",
            "score": "held-out nominals + all failures",
        },
        "per_problem": per_problem,
        "pooled_auroc": pooled,
        "pooled_note": "lp3 excluded from pooling: it relabels lp2's 47 executions",
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write results to this JSON file")
    args = parser.parse_args()
    results = run_benchmark()
    print(json.dumps(results, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nresults written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
