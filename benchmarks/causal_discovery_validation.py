# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Causal-discovery validation: does the dormant ``causal_discovery`` engine recover known causal structure?

`causal_discovery.py` was orphaned and, judged by the anomaly-AUC lens, looked
un-revivable -- it emits a causal *graph*, not a per-sample anomaly score. That
is the wrong metric. The transparent test for a constraint-based causal-discovery
algorithm is **structural recovery against a known ground-truth DAG**: generate a
linear-Gaussian structural equation model with a known graph, sample from it,
run discovery, and compare the recovered skeleton to the truth.

This is the non-AUC measurement framework the dormancy ledger calls for. It is
self-contained (synthetic ground truth, no network) and reports standard causal-
discovery metrics: skeleton precision / recall / F1 and the structural Hamming
distance (SHD), against a random-graph chance baseline so "recovery" means beating
chance, not merely being non-zero.

Pre-registered bar: the engine is a *validated* causal tool if mean skeleton
F1 >= 0.60 across the difficulty grid and clearly beats the chance baseline.

Usage::

    python -m benchmarks.causal_discovery_validation \\
        --n-graphs 5 --out artifacts/causal_discovery_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Difficulty grid: (n_variables, n_samples). Recovery should be strong when
# samples are plentiful and degrade as variables grow / samples shrink.
GRID = [(6, 1000), (6, 200), (10, 1000), (10, 300)]
_DENSITY = 0.3
_RECOVERY_F1 = 0.60


def _random_dag(n: int, density: float, rng: np.random.Generator) -> np.ndarray[Any, Any]:
    """Random DAG as an upper-triangular boolean adjacency (i -> j for i < j)."""
    a = rng.random((n, n)) < density
    return np.triu(a, k=1)


def _sample_sem(
    adjacency: np.ndarray[Any, Any], n_samples: int, rng: np.random.Generator
) -> np.ndarray[Any, Any]:
    """Sample a standardised linear-Gaussian SEM with the given DAG."""
    n = adjacency.shape[0]
    weights = (
        adjacency
        * rng.uniform(0.8, 2.0, adjacency.shape)
        * rng.choice([-1.0, 1.0], adjacency.shape)
    )
    x = np.zeros((n_samples, n))
    for j in range(n):  # nodes are in topological order (parents are i < j)
        parents = np.where(adjacency[:, j])[0]
        x[:, j] = x[:, parents] @ weights[parents, j] + rng.normal(0, 1.0, n_samples)
    std = x.std(0)
    std[std < 1e-8] = 1.0
    # np.asarray pins the return to a concrete ndarray (numpy stubs type the
    # mean/broadcast arithmetic as Any under some versions -> no-any-return).
    return np.asarray((x - x.mean(0)) / std)


def _skeleton(adjacency: np.ndarray[Any, Any]) -> set[frozenset[int]]:
    n = adjacency.shape[0]
    return {frozenset((i, j)) for i in range(n) for j in range(n) if adjacency[i, j]}


def _prf(true: set[frozenset[int]], pred: set[frozenset[int]]) -> tuple[float, float, float, int]:
    tp = len(true & pred)
    fp = len(pred - true)
    fn = len(true - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, fp + fn


def _chance_f1(n: int, true: set[frozenset[int]], n_pred: int, rng: np.random.Generator) -> float:
    """Expected F1 of a random graph predicting ``n_pred`` edges (averaged)."""
    all_pairs = [frozenset((i, j)) for i in range(n) for j in range(i + 1, n)]
    if not all_pairs or n_pred == 0:
        return 0.0
    f1s = []
    for _ in range(20):
        idx = rng.choice(len(all_pairs), size=min(n_pred, len(all_pairs)), replace=False)
        pred = {all_pairs[i] for i in idx}
        f1s.append(_prf(true, pred)[2])
    return float(np.mean(f1s))


def evaluate(n_vars: int, n_samples: int, n_graphs: int) -> dict[str, Any]:
    from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine

    f1s: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    shds: list[int] = []
    chance_f1s: list[float] = []
    for g in range(n_graphs):
        rng = np.random.default_rng(1000 * n_vars + n_samples + g)
        adjacency = _random_dag(n_vars, _DENSITY, rng)
        true = _skeleton(adjacency)
        if not true:
            continue
        data = _sample_sem(adjacency, n_samples, rng)
        engine = CausalDiscoveryEngine(significance_level=0.05, enable_temporal=False, seed=g)
        graph = engine.discover_structure(data, variable_names=[f"X{i}" for i in range(n_vars)])
        pred = {frozenset((int(e.source[1:]), int(e.target[1:]))) for e in graph.edges}
        precision, recall, f1, shd = _prf(true, pred)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        shds.append(shd)
        chance_f1s.append(_chance_f1(n_vars, true, len(pred), rng))
    return {
        "n_vars": n_vars,
        "n_samples": n_samples,
        "mean_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "mean_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "mean_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "mean_shd": float(np.mean(shds)) if shds else float("nan"),
        "chance_f1": float(np.mean(chance_f1s)) if chance_f1s else float("nan"),
        "n_graphs": len(f1s),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-graphs", type=int, default=5)
    parser.add_argument("--out", default="artifacts/causal_discovery_validation.json", type=str)
    args = parser.parse_args()

    print("Causal-discovery validation (skeleton recovery vs known SEM ground truth)")
    print("-" * 80)
    results = [evaluate(n, s, args.n_graphs) for (n, s) in GRID]
    for r in results:
        print(
            f"  n_vars={r['n_vars']:<3} n={r['n_samples']:<5} "
            f"F1={r['mean_f1']:.3f} (chance {r['chance_f1']:.3f})  "
            f"P={r['mean_precision']:.3f} R={r['mean_recall']:.3f} SHD={r['mean_shd']:.2f}"
        )

    mean_f1 = float(np.nanmean([r["mean_f1"] for r in results]))
    mean_chance = float(np.nanmean([r["chance_f1"] for r in results]))
    passed = bool(mean_f1 >= _RECOVERY_F1 and mean_f1 > 2 * mean_chance)
    verdict = {
        "mean_skeleton_f1": mean_f1,
        "mean_chance_f1": mean_chance,
        "passed": passed,
        "verdict": (
            "VALIDATED -- causal_discovery recovers known structure well above chance; "
            "revive as a measured causal tool"
            if passed
            else "WEAK -- recovery does not clear the bar; keep dormant"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"results": results, "verdict": verdict}, indent=2, sort_keys=True)
    )
    print("-" * 80)
    print(f"mean skeleton F1 = {mean_f1:.3f} (chance {mean_chance:.3f})")
    print(f"VERDICT: {verdict['verdict']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
