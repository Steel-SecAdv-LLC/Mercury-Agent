# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""K-means-distance detector: does the revived dormant clusterer ADD to the fusion ensemble?

``benchmarks/dormant_module_revival.py`` established that the dormant
``KMeansClusterer`` (revived as ``detectors.kmeans_distance.KMeansDistanceDetector``)
carries real *standalone* anomaly signal on ADBench (mean AUC ~0.86). Standalone
signal is necessary but not sufficient: the live ensemble already ships a
distance/density detector (``spatial``), so the detector earns a place in the
*default* set only if it improves the **fused** ROC-AUC. This is the marginal
ablation: each (dataset, seed) trains the fusion model from the same split and
initialisation with and without the k-means detector, differing only in that
one feature group.

Pre-registered gate (same conservative noise floor as the neuro-symbolic
ablation): add to the default ensemble iff mean ΔAUC > +0.002 with a majority of
seeds agreeing. The bar is not moved to manufacture a pass.

Usage::

    python -m benchmarks.kmeans_ensemble_marginal \\
        --datasets cardio thyroid breastw WBC Pima --seeds 0 1 2 \\
        --out artifacts/kmeans_ensemble_marginal.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATASETS = ["cardio", "thyroid", "breastw", "WBC", "Pima"]
DEFAULT_SEEDS = [0, 1, 2]
_AUC_MEANINGFUL = 0.002


def _fused_auc(X_tr, y_tr, X_te, y_te, seed: int, add_kmeans: bool) -> float:
    import torch

    from omni_mercury_engine.detectors.kmeans_distance import KMeansDistanceDetector
    from omni_mercury_engine.engine import OmniMercuryEngine
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    if add_kmeans:
        engine.detectors["kmeans_distance"] = KMeansDistanceDetector()
    engine.fit_fusion(X_tr, y_tr, epochs=20, batch_size=64, early_stopping_patience=15)
    return float(roc_auc_score(y_te, engine.score_fusion(X_te)))


def run_dataset(name: str, seeds: list[int]) -> dict[str, Any] | None:
    from benchmarks.neurosymbolic_ablation import _load_dataset, _stratified_indices

    try:
        X, y = _load_dataset(name)
    except Exception as exc:
        print(f"  {name:<10} SKIP (load failed: {exc})")
        return None

    deltas: list[float] = []
    base_aucs: list[float] = []
    km_aucs: list[float] = []
    better = 0
    for seed in seeds:
        rng = np.random.RandomState(seed)
        te = _stratified_indices(y, 0.3, rng)
        mask = np.zeros(len(y), dtype=bool)
        mask[te] = True
        tr = np.where(~mask)[0]
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        a_base = _fused_auc(X[tr], y[tr], X[te], y[te], seed, add_kmeans=False)
        a_km = _fused_auc(X[tr], y[tr], X[te], y[te], seed, add_kmeans=True)
        base_aucs.append(a_base)
        km_aucs.append(a_km)
        deltas.append(a_km - a_base)
        better += int(a_km >= a_base)
    if not deltas:
        return None
    print(
        f"  {name:<10} base {np.mean(base_aucs):.4f} -> +kmeans {np.mean(km_aucs):.4f} "
        f"(dAUC={np.mean(deltas):+.4f})  [{better}/{len(deltas)} seeds >=]"
    )
    return {
        "dataset": name,
        "delta_auc_mean": float(np.mean(deltas)),
        "base_auc_mean": float(np.mean(base_aucs)),
        "kmeans_auc_mean": float(np.mean(km_aucs)),
        "seeds_better": better,
        "n_seeds": len(deltas),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", default="artifacts/kmeans_ensemble_marginal.json", type=str)
    args = parser.parse_args()

    print("K-means-distance ensemble-marginal ablation (fused AUC, real ADBench labels)")
    print("-" * 80)
    results = [r for r in (run_dataset(n, args.seeds) for n in args.datasets) if r]
    if not results:
        print("INTEGRITY FAILURE: no dataset measured (network unavailable?).")
        return 1

    mean_delta = float(np.mean([r["delta_auc_mean"] for r in results]))
    seed_agree = float(np.mean([r["seeds_better"] / max(1, r["n_seeds"]) for r in results]))
    passed = bool(mean_delta > _AUC_MEANINGFUL and seed_agree >= 0.5)
    verdict = {
        "mean_delta_auc": mean_delta,
        "seed_agreement": seed_agree,
        "passed": passed,
        "verdict": (
            "ADD -- k-means detector improves the fused ensemble; enable by default"
            if passed
            else "HOLD -- no fused improvement beyond noise; keep optional (redundant with spatial)"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"results": results, "verdict": verdict}, indent=2, sort_keys=True)
    )
    print("-" * 80)
    print(f"mean dAUC = {mean_delta:+.4f}   seed agreement = {seed_agree:.2f}")
    print(f"VERDICT: {verdict['verdict']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
