# Copyright (C) 2025 Steel Security Advisors LLC
"""Rule-graph sweep: does richer symbolic structure beat the minimal 2-rule."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATASETS = ["cardio", "thyroid", "WBC"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FRACTIONS = [0.1, 0.25, 0.5]
GRAPHS = ["consensus", "consensus_salience"]
_AUC_MEANINGFUL = 0.002


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--fractions", nargs="*", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--out", default="artifacts/symbolic_rulegraph_sweep.json", type=str)
    args = parser.parse_args()

    from benchmarks.neurosymbolic_ablation import _load_dataset, _stratified_indices, _train_eval

    print("Symbolic rule-graph sweep (adaptive constraint, same-cell consensus vs +salience)")
    print(f"datasets={args.datasets} seeds={args.seeds} fractions={args.fractions}")
    print("-" * 88)

    deltas: dict[str, list[float]] = {g: [] for g in GRAPHS}
    agree: dict[str, int] = dict.fromkeys(GRAPHS, 0)
    n_cells = 0
    rows: list[dict[str, Any]] = []

    for name in args.datasets:
        try:
            X, y = _load_dataset(name)
        except Exception as exc:
            print(f"  {name:<10} SKIP (load failed: {exc})")
            continue
        for frac in args.fractions:
            for seed in args.seeds:
                rng = np.random.RandomState(seed)
                te = _stratified_indices(y, 0.3, rng)
                mask = np.zeros(len(y), dtype=bool)
                mask[te] = True
                pool = np.where(~mask)[0]
                sub = _stratified_indices(y[pool], frac, np.random.RandomState(seed + 1))
                tr = pool[sub]
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                a_neu, _, _ = _train_eval(X[tr], y[tr], X[te], y[te], 0.0, 20, seed)
                cell = {"dataset": name, "frac": frac, "seed": seed, "neural": a_neu}
                for g in GRAPHS:
                    a_g, _, _ = _train_eval(
                        X[tr], y[tr], X[te], y[te], "adaptive", 20, seed, "product", g
                    )
                    deltas[g].append(a_g - a_neu)
                    agree[g] += int(a_g >= a_neu)
                    cell[g] = a_g
                n_cells += 1
                rows.append(cell)
                print(
                    f"  {name:<8} frac={frac:<4} seed={seed}  neu={a_neu:.4f}  "
                    + "  ".join(
                        f"{g.split('_')[-1][:4]}={cell[g]:.4f}(d={cell[g]-a_neu:+.4f})"
                        for g in GRAPHS
                    )
                )

    if n_cells == 0:
        print("INTEGRITY FAILURE: no cell measured (network unavailable?).")
        return 1

    summary = {
        g: {
            "mean_delta_auc": float(np.mean(deltas[g])),
            "seed_agreement": float(agree[g] / max(1, n_cells)),
        }
        for g in GRAPHS
    }
    base = summary["consensus"]["mean_delta_auc"]
    rich = summary["consensus_salience"]["mean_delta_auc"]
    rich_wins = bool(
        rich > base + _AUC_MEANINGFUL and summary["consensus_salience"]["seed_agreement"] >= 0.5
    )
    verdict = {
        "summary": summary,
        "salience_beats_consensus": rich_wins,
        "recommended_default": "consensus_salience" if rich_wins else "consensus",
        "verdict": (
            "SWITCH default to consensus_salience (richer rules help on real labels)"
            if rich_wins
            else "KEEP consensus default (the salience rule does not beat it beyond noise)"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"cells": rows, "verdict": verdict}, indent=2, sort_keys=True)
    )
    print("-" * 88)
    for g in GRAPHS:
        print(
            f"  {g:<20} mean ΔAUC={summary[g]['mean_delta_auc']:+.4f}  "
            f"seed_agree={summary[g]['seed_agreement']:.2f}"
        )
    print(f"VERDICT: {verdict['verdict']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
