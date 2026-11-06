# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the 21-probe math-arrest instrument ADD to the live fusion ensemble?

``docs/DORMANCY_LEDGER.md`` §2b measured ``AnomalyMathArrest`` on real labels,
repaired four defects, and wired it into ``DETECTOR_MANIFEST`` behind a
``BaseDetector`` adapter -- then explicitly withheld a default-set slot:

    A **default-set slot is still not granted**: that needs a marginal-gain
    ablation against the live ensemble, which this entry does not claim.
    Reachable and measured is not the same as promoted.

This is that ablation. It is the same shape as
``benchmarks/kmeans_ensemble_marginal.py``, which settled the same question for
the revived k-means detector, so the two decisions are comparable: each
(dataset, seed) trains the engine's fusion from the same split and the same
initialisation with and without the detector, differing only in that one
feature group.

Pre-registered gate, unchanged from the k-means ablation so the bar is not
moved to manufacture an outcome: **add to the default ensemble iff mean
ΔAUC > +0.002 with a majority of seeds agreeing.**

Two things this harness reports that the k-means one did not, because both
turned out to matter here:

* **The baseline per table.** The trained ``OmniFusionModel`` already reaches
  0.99+ on several recorded tables, so a marginal-gain test has almost no
  headroom there. A Δ near zero against a 0.998 baseline is a ceiling, not a
  verdict on the detector, and the two must not be confused.
* **The feature group's width and degeneracy.** Every probe is a series
  equation and ``BaseEquationProbe._to_1d`` reduces a matrix to its per-row
  mean, so the obvious failure mode is a feature group that is constant on
  tabular input. Measured, it is not: 21 columns with a median 542 distinct
  values over 548 rows and one constant column. The ablation is therefore
  measuring fusion, not a dead group -- which is worth establishing rather
  than assuming.

Reads the recorded ADBench fixtures (``tests/fixtures/adbench``) rather than
the network loader, so the run is offline and reproducible.

It does need the engine, and therefore the AMA/PQC native backend: Mercury
fails closed at import without it (``_pqc_gate``), which is why this ablation
was deferred when §2b was written. See ``docs/INSTALLATION.md`` 'Post-Quantum
Cryptography backend' for the build; the sovereignty boundary means there is no
third-party substitute to fall back on.

Usage::

    python benchmarks/math_arrest_ensemble_marginal.py
    python benchmarks/math_arrest_ensemble_marginal.py --tables cardio --seeds 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: The noise floor the k-means ablation pre-registered. Not moved.
AUC_MEANINGFUL = 0.002


def _fused_auc(
    train_rows: Any,
    train_labels: Any,
    test_rows: Any,
    test_labels: Any,
    seed: int,
    *,
    add_math_arrest: bool,
) -> float:
    """Train the engine's fusion on one split; return held-out fused AUROC."""
    import torch

    from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector
    from omni_mercury_engine.engine import OmniMercuryEngine
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    if add_math_arrest:
        engine.register_detector("math_arrest", MathArrestDetector())
    engine.fit_fusion(
        train_rows, train_labels, epochs=20, batch_size=64, early_stopping_patience=15
    )
    return float(roc_auc_score(test_labels, engine.score_fusion(test_rows)))


def _feature_group_shape(train_rows: Any, test_rows: Any) -> dict[str, Any]:
    """Width and degeneracy of the feature group on tabular input."""
    from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector

    detector = MathArrestDetector()
    detector.fit(train_rows)
    features = detector.extract_features(test_rows)
    distinct = [len(np.unique(np.round(features[:, j], 9))) for j in range(features.shape[1])]
    return {
        "n_rows": int(features.shape[0]),
        "n_columns": int(features.shape[1]),
        "constant_columns": int(sum(1 for d in distinct if d == 1)),
        "median_distinct_values": int(np.median(distinct)),
    }


def run_table(name: str, seeds: list[int]) -> dict[str, Any] | None:
    """Ablate one recorded table over *seeds*."""
    from tests.recorded_data import adbench, stratified_split

    features, labels = adbench(name)
    deltas: list[float] = []
    base: list[float] = []
    with_probe: list[float] = []
    better = 0
    group: dict[str, Any] = {}

    for seed in seeds:
        train_idx, test_idx = stratified_split(labels, seed)
        if len(np.unique(labels[test_idx])) < 2 or len(np.unique(labels[train_idx])) < 2:
            continue
        train_rows, test_rows = features[train_idx], features[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        if not group:
            group = _feature_group_shape(train_rows, test_rows)

        without = _fused_auc(
            train_rows, train_labels, test_rows, test_labels, seed, add_math_arrest=False
        )
        added = _fused_auc(
            train_rows, train_labels, test_rows, test_labels, seed, add_math_arrest=True
        )
        base.append(without)
        with_probe.append(added)
        deltas.append(added - without)
        better += int(added >= without)

    if not deltas:
        return None
    print(
        f"  {name:<11s} base {np.mean(base):.4f} -> +math_arrest {np.mean(with_probe):.4f} "
        f"(dAUC={np.mean(deltas):+.4f})  [{better}/{len(deltas)} seeds >=]  "
        f"group {group.get('n_columns')} cols, {group.get('constant_columns')} constant"
    )
    return {
        "table": name,
        "delta_auc_mean": float(np.mean(deltas)),
        "delta_auc_per_seed": [float(d) for d in deltas],
        "base_auc_mean": float(np.mean(base)),
        "math_arrest_auc_mean": float(np.mean(with_probe)),
        "seeds_better": better,
        "n_seeds": len(deltas),
        "feature_group": group,
    }


def main() -> int:
    """Run the ablation and print the pre-registered verdict."""
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    from tests.recorded_data import ADBENCH_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", default=list(ADBENCH_NAMES))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default="artifacts/math_arrest_ensemble_marginal.json")
    args = parser.parse_args()

    print("math_arrest ensemble-marginal ablation (fused AUROC, recorded ADBench labels)")
    print("-" * 100)
    results = [r for r in (run_table(t, args.seeds) for t in args.tables) if r]
    if not results:
        print("INTEGRITY FAILURE: no table measured.")
        return 1

    mean_delta = float(np.mean([r["delta_auc_mean"] for r in results]))
    seed_agreement = float(np.mean([r["seeds_better"] / max(1, r["n_seeds"]) for r in results]))
    passed = bool(mean_delta > AUC_MEANINGFUL and seed_agreement >= 0.5)
    verdict = {
        "mean_delta_auc": mean_delta,
        "seed_agreement": seed_agreement,
        "bar": f"mean dAUC > +{AUC_MEANINGFUL} and seed agreement >= 0.5",
        "passed": passed,
        "verdict": (
            "ADD -- math_arrest improves the fused ensemble; enable by default"
            if passed
            else "HOLD -- no fused improvement beyond the noise floor; keep "
            "reachable-but-not-default"
        ),
    }

    print("-" * 100)
    print(f"mean dAUC = {mean_delta:+.4f}   seed agreement = {seed_agreement:.2f}")
    print(f"bar       = {verdict['bar']}")
    print(f"VERDICT   : {verdict['verdict']}")
    print()
    print("per-table baselines, so a ceiling is visible rather than read as a verdict:")
    for r in results:
        headroom = 1.0 - r["base_auc_mean"]
        print(
            f"  {r['table']:<11s} base {r['base_auc_mean']:.4f}  headroom {headroom:.4f}  "
            f"dAUC {r['delta_auc_mean']:+.4f}  per-seed "
            f"{[round(d, 4) for d in r['delta_auc_per_seed']]}"
        )

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"results": results, "verdict": verdict}, indent=2, sort_keys=True)
    )
    print(f"\nreport -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
