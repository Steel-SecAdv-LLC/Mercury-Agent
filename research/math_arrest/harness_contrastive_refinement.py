# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contrastive probe refinement, measured on the recorded tables.

``AnomalyMathArrest.refine_contrastive`` re-weights probes through the
decorrelator's multipliers using the fused score's own discordant pairs on a
labelled calibration fold. This harness measures what that buys: per recorded
ADBench table (three seeds, 49/21/30 fit/calibration/test), the held-out
AUROC of the unrefined ensemble against refinements at eta 0.25 / 0.5 / 1.0,
with mean rank across configurations (competitive tie convention), plus the
calibration-fold discordance before and after each adjustment.

Results go to ``artifacts/math_arrest_contrastive_refinement.json``;
``tests/detectors/test_contrastive_refinement.py`` pins the mechanism and the
measured verdict.

Run from the repository root::

    python research/math_arrest/harness_contrastive_refinement.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from research.math_arrest.harness_probe_redundancy import _git_commit
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, stratified_split

ARTIFACT_PATH = _ROOT / "artifacts" / "math_arrest_contrastive_refinement.json"
SEEDS = (0, 1, 2)
ETAS = (0.0, 0.25, 0.5, 1.0)


def _mean_rank(per_config: dict[str, dict[str, float]]) -> dict[str, float]:
    configs = sorted(per_config)
    tables = sorted(next(iter(per_config.values())))
    totals = dict.fromkeys(configs, 0.0)
    for table in tables:
        values = np.array([per_config[c][table] for c in configs])
        order = np.argsort(-values, kind="stable")
        ranks = np.empty(len(configs))
        i = 0
        while i < len(configs):
            j = i
            while j + 1 < len(configs) and values[order[j + 1]] == values[order[i]]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        for k, config in enumerate(configs):
            totals[config] += float(ranks[k])
    return {c: round(totals[c] / len(tables), 4) for c in configs}


def main() -> int:
    per_seed: dict[str, dict[str, list[float]]] = {}
    discordance: dict[str, list[float]] = {"before": [], "after_best": []}

    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for seed in SEEDS:
            train_all, test = stratified_split(labels, seed)
            fit_local, cal_local = stratified_split(labels[train_all], seed)
            fit_rows, cal_rows = train_all[fit_local], train_all[cal_local]
            mean = features[fit_rows].mean(axis=0)
            spread = features[fit_rows].std(axis=0)
            spread[spread < 1e-8] = 1.0

            def z(rows: np.ndarray) -> np.ndarray:
                return (features[rows] - mean) / spread

            y_test = labels[test]
            for eta in ETAS:
                detector = AnomalyMathArrest()
                detector.fit(z(fit_rows))
                report = detector.refine_contrastive(z(cal_rows), labels[cal_rows], eta=eta)
                fused = np.asarray(detector.detect(z(test)), dtype=np.float64).ravel()
                value = auroc(fused[y_test == 1], fused[y_test == 0])
                per_seed.setdefault(f"eta_{eta}", {}).setdefault(table, []).append(value)
                if eta == 0.0:
                    discordance["before"].append(report["discordant_before"])
                if eta == 0.5:
                    discordance["after_best"].append(report["discordant_after"])

    per_config = {
        config: {table: round(float(np.mean(v)), 4) for table, v in tables.items()}
        for config, tables in per_seed.items()
    }
    ranks = _mean_rank(per_config)
    summary = {
        config: {
            "per_table_auc": tables,
            "mean_auc": round(float(np.mean(list(tables.values()))), 4),
            "mean_rank": ranks[config],
        }
        for config, tables in per_config.items()
    }
    baseline_rank = summary["eta_0.0"]["mean_rank"]
    improving = [
        config
        for config, cell in summary.items()
        if config != "eta_0.0" and cell["mean_rank"] < baseline_rank
    ]
    artifact = {
        "schema": "mercury.research.math_arrest_contrastive_refinement/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "config": {"tables": list(ADBENCH_NAMES), "seeds": list(SEEDS), "etas": list(ETAS)},
        "results": summary,
        "calibration_discordance": {
            "mean_before": round(float(np.mean(discordance["before"])), 4),
            "mean_after_eta_0.5": round(float(np.mean(discordance["after_best"])), 4),
        },
        "improving_etas": improving,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"{'config':10s} {'mean AUC':>9s} {'mean rank':>10s}")
    for config, cell in sorted(summary.items(), key=lambda kv: kv[1]["mean_rank"]):
        print(f"{config:10s} {cell['mean_auc']:9.4f} {cell['mean_rank']:10.2f}")
    print(
        f"calibration discordance: {artifact['calibration_discordance']['mean_before']:.4f} "
        f"-> {artifact['calibration_discordance']['mean_after_eta_0.5']:.4f} (eta 0.5)"
    )
    print(f"improving etas: {improving or 'none'}")
    print(f"wrote {ARTIFACT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
