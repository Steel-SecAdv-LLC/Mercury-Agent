# Copyright (C) 2025 Steel Security Advisors LLC
"""Symbolic-constraint semantics sweep: does a *crisp* implication residuum."""

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
SEMANTICS = ["product", "lukasiewicz", "godel"]
_AUC_MEANINGFUL = 0.002


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--fractions", nargs="*", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--out", default="artifacts/symbolic_semantics_sweep.json", type=str)
    args = parser.parse_args()

    from benchmarks.neurosymbolic_ablation import _load_dataset, _stratified_indices, _train_eval

    print("Symbolic semantics sweep (adaptive constraint, same-cell crisp vs fuzzy)")
    print(f"datasets={args.datasets} seeds={args.seeds} fractions={args.fractions}")
    print("-" * 88)

    # deltas[sem] = list of (adaptive_sem - neural) AUC across cells; agree[sem]
    # counts cells where the semantics beat or tied neural.
    deltas: dict[str, list[float]] = {s: [] for s in SEMANTICS}
    agree: dict[str, int] = dict.fromkeys(SEMANTICS, 0)
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
                for sem in SEMANTICS:
                    a_sem, _, _ = _train_eval(X[tr], y[tr], X[te], y[te], "adaptive", 20, seed, sem)
                    d = a_sem - a_neu
                    deltas[sem].append(d)
                    agree[sem] += int(a_sem >= a_neu)
                    cell[sem] = a_sem
                n_cells += 1
                rows.append(cell)
                print(
                    f"  {name:<8} frac={frac:<4} seed={seed}  neu={a_neu:.4f}  "
                    + "  ".join(f"{s[:4]}={cell[s]:.4f}(d={cell[s]-a_neu:+.4f})" for s in SEMANTICS)
                )

    if n_cells == 0:
        print("INTEGRITY FAILURE: no cell measured (network unavailable?).")
        return 1

    summary = {
        s: {
            "mean_delta_auc": float(np.mean(deltas[s])),
            "seed_agreement": float(agree[s] / max(1, n_cells)),
        }
        for s in SEMANTICS
    }
    prod = summary["product"]["mean_delta_auc"]
    best_crisp = max(("lukasiewicz", "godel"), key=lambda s: summary[s]["mean_delta_auc"])
    crisp_wins = bool(
        summary[best_crisp]["mean_delta_auc"] > prod + _AUC_MEANINGFUL
        and summary[best_crisp]["seed_agreement"] >= 0.5
    )
    verdict = {
        "summary": summary,
        "best_crisp": best_crisp,
        "crisp_beats_product": crisp_wins,
        "recommended_default": best_crisp if crisp_wins else "product",
        "verdict": (
            f"SWITCH default semantics to {best_crisp} (beats product on real labels)"
            if crisp_wins
            else "KEEP product/reichenbach default (crisp does not beat it beyond noise)"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"cells": rows, "verdict": verdict}, indent=2, sort_keys=True)
    )
    print("-" * 88)
    for s in SEMANTICS:
        print(
            f"  {s:<12} mean ΔAUC={summary[s]['mean_delta_auc']:+.4f}  "
            f"seed_agree={summary[s]['seed_agreement']:.2f}"
        )
    print(f"VERDICT: {verdict['verdict']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
