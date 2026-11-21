# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Entropy-constrained re-selection of the R/K/IG fusion weights.

The fusion-mode ablation measured the shipped adaptive
Resonance/Kinematic/InfoGeometry weights against plain average, the fixed
40/30/30 prior and the single best component — and the single best component
(a fully peaked, zero-entropy weight vector) beat the adaptive fusion on the
recorded tables. That is exactly the collapse an entropy constraint exists to
examine: this harness sweeps the whole weight simplex (step 0.05) over the
three component scores, computes each vector's normalised entropy
``H(w)/ln 3``, and selects the best mean-rank configuration **subject to the
entropy band** ``[0.30, 0.95]`` — excluding both the collapse onto a single
component and the diffuse near-uniform extreme.

Component scores are produced once per (table, seed) by the fitted
``MercuryAnomalyDetector``'s own component methods, so every candidate weight
vector reweights identical internal streams; nothing external enters.
Results go to ``artifacts/fusion_weight_entropy.json``;
``tests/benchmarks/test_weight_entropy.py`` pins the selection rule and the
measured verdict.

Run from the repository root::

    python research/ensembles/harness_weight_entropy.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.math_arrest.harness_probe_redundancy import _git_commit
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, stratified_split

ARTIFACT_PATH = _ROOT / "artifacts" / "fusion_weight_entropy.json"
SEEDS = (0, 1, 2)
GRID_STEP = 0.05
#: Normalised-entropy band: below 0.30 the vector has collapsed onto one
#: component; above 0.95 it is indistinguishable from uniform.
ENTROPY_BAND = (0.30, 0.95)


def normalized_entropy(weights: np.ndarray[Any, Any]) -> float:
    """Shannon entropy of a weight vector, normalised to [0, 1] by ln(k)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 0.0]
    if w.size <= 1:
        return 0.0
    w = w / w.sum()
    return float(-(w * np.log(w)).sum() / np.log(len(weights)))


def simplex_grid(step: float = GRID_STEP) -> list[np.ndarray[Any, Any]]:
    """All 3-simplex weight vectors at the given resolution."""
    ticks = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    out = []
    for a in ticks:
        for b in ticks:
            c = 1.0 - a - b
            if c >= -1e-9:
                out.append(np.array([a, b, max(c, 0.0)]))
    return out


def component_scores() -> dict[str, dict[str, Any]]:
    """R/K/IG component scores on every (table, seed) held-out fold."""
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    out: dict[str, dict[str, Any]] = {}
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            detector = MercuryAnomalyDetector()
            detector.fit((features[train] - mean) / spread)
            x_test = (features[test] - mean) / spread
            out[f"{table}/seed{seed}"] = {
                "table": table,
                "components": np.column_stack(
                    [
                        detector._compute_resonance_score(x_test),
                        detector._compute_kinematic_score(x_test),
                        detector._compute_info_geometry_score(x_test),
                    ]
                ),
                "adaptive_weights": detector._adaptive_weights.copy(),
                "labels": labels[test],
            }
    return out


def evaluate(
    weights_by_run: dict[str, np.ndarray[Any, Any]], data: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """Per-table mean AUROC of a (possibly per-run) weight assignment."""
    per_table: dict[str, list[float]] = {}
    for run, record in data.items():
        w = weights_by_run[run]
        fused = record["components"] @ (w / max(w.sum(), 1e-12))
        y = record["labels"]
        per_table.setdefault(record["table"], []).append(auroc(fused[y == 1], fused[y == 0]))
    return {table: round(float(np.mean(v)), 4) for table, v in per_table.items()}


def mean_rank(per_config: dict[str, dict[str, float]]) -> dict[str, float]:
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


#: The standing adoption bar: a mean gain above +0.003 with no table below -0.002.
ADOPTION_MEAN_GAIN = 0.003
ADOPTION_WORST_TABLE = -0.002


def leave_one_table_out(
    per_config: dict[str, dict[str, float]], in_band: list[str]
) -> dict[str, Any]:
    """Does a weight vector chosen on four tables beat the shipped weights on the fifth?

    The in-band winner above is selected on the same tables it is scored on,
    which is why it was recorded and not adopted.  Here each table is held
    out in turn: the vector with the best mean rank on the other four is
    selected, and scored on the one it never saw, beside the shipped adaptive
    weights on that same table.
    """
    tables = sorted(per_config["shipped_adaptive"])
    folds: dict[str, dict[str, Any]] = {}
    for held in tables:
        seen = [table for table in tables if table != held]
        ranks = mean_rank({key: {t: per_config[key][t] for t in seen} for key in in_band})
        chosen = min(in_band, key=lambda key: ranks[key])
        shipped = per_config["shipped_adaptive"][held]
        folds[held] = {
            "selected_on_the_other_four": chosen,
            "auc": per_config[chosen][held],
            "shipped_adaptive_auc": shipped,
            "delta": round(per_config[chosen][held] - shipped, 4),
        }
    deltas = [fold["delta"] for fold in folds.values()]
    mean_delta, worst = float(np.mean(deltas)), float(min(deltas))
    return {
        "folds": folds,
        "mean_delta": round(mean_delta, 4),
        "worst_table_delta": round(worst, 4),
        "bar": {"mean_gain_above": ADOPTION_MEAN_GAIN, "no_table_below": ADOPTION_WORST_TABLE},
        "bar_met": mean_delta > ADOPTION_MEAN_GAIN and worst >= ADOPTION_WORST_TABLE,
    }


def main() -> int:
    data = component_scores()
    grid = simplex_grid()

    per_config: dict[str, dict[str, float]] = {}
    entropy_of: dict[str, float] = {}
    for w in grid:
        key = f"w_{w[0]:.2f}_{w[1]:.2f}_{w[2]:.2f}"
        per_config[key] = evaluate(dict.fromkeys(data, w), data)
        entropy_of[key] = round(normalized_entropy(w), 4)

    # References: the shipped adaptive weights (per run), uniform, 40/30/30.
    per_config["shipped_adaptive"] = evaluate(
        {run: record["adaptive_weights"] for run, record in data.items()}, data
    )
    entropy_of["shipped_adaptive"] = round(
        float(
            np.mean([normalized_entropy(record["adaptive_weights"]) for record in data.values()])
        ),
        4,
    )
    ranks = mean_rank(per_config)

    def summary(key: str) -> dict[str, Any]:
        return {
            "per_table_auc": per_config[key],
            "mean_auc": round(float(np.mean(list(per_config[key].values()))), 4),
            "mean_rank": ranks[key],
            "normalized_entropy": entropy_of[key],
        }

    in_band = [
        key
        for key in per_config
        if key != "shipped_adaptive" and ENTROPY_BAND[0] <= entropy_of[key] <= ENTROPY_BAND[1]
    ]
    selected = min(in_band, key=lambda key: ranks[key])
    unconstrained = min(
        (key for key in per_config if key != "shipped_adaptive"), key=lambda key: ranks[key]
    )

    artifact: dict[str, Any] = {
        "schema": "mercury.research.fusion_weight_entropy/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "config": {
            "tables": list(ADBENCH_NAMES),
            "seeds": list(SEEDS),
            "grid_step": GRID_STEP,
            "entropy_band": list(ENTROPY_BAND),
            "n_grid": len(grid),
        },
        "selected_in_band": {"key": selected, **summary(selected)},
        "best_unconstrained": {"key": unconstrained, **summary(unconstrained)},
        "references": {
            "shipped_adaptive": summary("shipped_adaptive"),
            "uniform": (
                summary("w_0.35_0.35_0.30")
                if "w_0.35_0.35_0.30" in per_config
                else summary("w_0.30_0.35_0.35")
            ),
            "fixed_403030": summary("w_0.40_0.30_0.30"),
        },
        "improves_on_shipped": ranks[selected] < ranks["shipped_adaptive"],
        "leave_one_table_out": leave_one_table_out(per_config, in_band),
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"grid: {len(grid)} vectors, {len(in_band)} inside entropy band {ENTROPY_BAND}")
    rows: list[tuple[str, dict[str, Any]]] = [
        ("selected (in band)", artifact["selected_in_band"]),
        ("best unconstrained", artifact["best_unconstrained"]),
        ("shipped adaptive", {"key": "-", **summary("shipped_adaptive")}),
        ("fixed 40/30/30", {"key": "-", **summary("w_0.40_0.30_0.30")}),
    ]
    for label, cell in rows:
        print(
            f"  {label:20s} {cell['key']:18s} mean AUC {cell['mean_auc']:.4f} "
            f"rank {cell['mean_rank']:8.2f} H {cell['normalized_entropy']:.3f}"
        )
    print(f"improves on shipped adaptive: {artifact['improves_on_shipped']}")
    held_out = artifact["leave_one_table_out"]
    for table, fold in held_out["folds"].items():
        print(
            f"  held out {table:10s} chose {fold['selected_on_the_other_four']:18s} "
            f"AUC {fold['auc']:.4f} vs shipped {fold['shipped_adaptive_auc']:.4f} "
            f"delta {fold['delta']:+.4f}"
        )
    print(
        f"leave-one-table-out: mean delta {held_out['mean_delta']:+.4f}, worst table "
        f"{held_out['worst_table_delta']:+.4f}, bar met: {held_out['bar_met']}"
    )
    print(f"wrote {ARTIFACT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
