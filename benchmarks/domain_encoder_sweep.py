# Copyright (C) 2025 Steel Security Advisors LLC
"""Differentiable domain-encoder DESIGN-SPACE sweep (WS-B follow-on).

PR #262 ran one differentiable-encoder design and recorded a sub-threshold
mean low-data ΔAUC (+0.0048) → QUARANTINE. The mandate this round: a quarantine
may stand only on *covered* search, not unexplored search. This harness sweeps
the design space and **stratifies by dataset family and data size**, so the
default-off verdict rests on an exhausted grid, not a single configuration.

Design axes swept (the "fusion points / kernel widths / normalization" called
out by the mandate), all through the real wired fusion path
(``engine.fit_fusion(domain_encoder=True, domain_encoder_config=...)``):

* **fusion points** -- which physics encoders are present: each alone
  (spectral / kinematic / fisher), leave-one-out, and the full stack;
* **kernel widths** -- the KinematicEncoder finite-difference conv widths
  (narrow ``(2,3)`` vs default ``(2,3,4)`` vs wide ``(2,3,4,5,6)``);
* **normalization** -- spectral magnitude transform (log1p vs sqrt) and an
  optional LayerNorm before the projection.

Two integrity controls, both from this round:

* **confound guard** -- every (config, family) cell is audited for the
  inverted-ranking confound (``evaluation.ablation_guard``); a KEEP on a
  collapsed-baseline cell is forced to QUARANTINE. This is essential here: on the
  imbalanced hard family the *baseline* fusion arm frequently inverts (AUC<0.5),
  so an unguarded delta would be fake.
* **family stratification** -- datasets are split into ``hard`` (low baseline
  AUC) and ``ceiling`` (saturated) so we can see whether any modest signal is
  conditionally real on hard/low-data rather than washed-out on average.

The baseline arm (``domain_encoder=False``) is identical across every encoder
design, so it is computed once per (dataset, seed, fraction) and reused.

Usage::

    python benchmarks/domain_encoder_sweep.py --seeds 0 1 2 --out artifacts/domain_encoder_sweep.json
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
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "src"))

from benchmarks.domain_encoder_ablation import (
    _f1_at_best_threshold,
    _load_dataset,
    _stratified_split,
)
from omni_mercury_engine.engine import OmniMercuryEngine
from omni_mercury_engine.evaluation.ablation_guard import check_ablation_confound
from omni_mercury_engine.ml.mercury_ml import roc_auc_score

# Dataset families (by baseline difficulty). Hard = low-AUC/imbalanced where any
# learnable-encoder gain is most plausible; ceiling = saturated, no headroom.
HARD = ["Pima", "glass"]
CEILING = ["cardio", "thyroid"]
DEFAULT_DATASETS = HARD + CEILING
FAMILY = {**dict.fromkeys(HARD, "hard"), **dict.fromkeys(CEILING, "ceiling")}

DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FRACTIONS = [0.25, 1.0]
LOW_DATA = 0.25
_AUC_MEANINGFUL = 0.002

# The design grid. None == baseline encoder (full stack, default everything).
DESIGN_GRID: dict[str, dict[str, Any]] = {
    "full_default": {},
    "spectral_only": {"domains": ("spectral",)},
    "kinematic_only": {"domains": ("kinematic",)},
    "fisher_only": {"domains": ("fisher",)},
    "no_kinematic": {"domains": ("spectral", "fisher")},
    "no_spectral": {"domains": ("kinematic", "fisher")},
    "narrow_kernels": {"encoder_kwargs": {"kinematic": {"kernel_widths": (2, 3)}}},
    "wide_kernels": {"encoder_kwargs": {"kinematic": {"kernel_widths": (2, 3, 4, 5, 6)}}},
    "sqrt_spectral": {"encoder_kwargs": {"spectral": {"magnitude_transform": "sqrt"}}},
    "layernorm": {"normalize": True},
}


def _fit_eval(
    x_tr: np.ndarray[Any, Any],
    y_tr: np.ndarray[Any, Any],
    x_te: np.ndarray[Any, Any],
    y_te: np.ndarray[Any, Any],
    epochs: int,
    seed: int,
    *,
    domain_encoder: bool,
    config: dict[str, Any] | None,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        engine.fit_fusion(
            x_tr,
            y_tr,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            early_stopping_patience=epochs,
            domain_encoder=domain_encoder,
            domain_encoder_config=config,
        )
        scores = engine.score_fusion(x_te)
    return float(roc_auc_score(y_te, scores)), _f1_at_best_threshold(y_te, scores)


def run_sweep(
    datasets: list[str], seeds: list[int], fractions: list[float], epochs: int
) -> dict[str, Any]:
    # 1. baselines: domain_encoder=False, once per (dataset, seed, fraction).
    baseline: dict[tuple[str, float, int], float] = {}
    prepared: dict[str, Any] = {}
    for name in datasets:
        x, y = _load_dataset(name)
        pool, test_idx = _stratified_split(y, 0.7, np.random.RandomState(0))
        x_te, y_te = x[test_idx], y[test_idx]
        prepared[name] = (x, y, pool, x_te, y_te)
        for frac in fractions:
            for seed in seeds:
                sub = _stratified_split(y[pool], frac, np.random.RandomState(seed + 1))[0]
                tr = pool[sub]
                b_auc, _ = _fit_eval(
                    x[tr], y[tr], x_te, y_te, epochs, seed, domain_encoder=False, config=None
                )
                baseline[(name, frac, seed)] = b_auc
        print(f"  baselines done: {name}", flush=True)

    # 2. encoder arms per design config, paired against the cached baselines.
    cells: list[dict[str, Any]] = []
    for cfg_name, cfg in DESIGN_GRID.items():
        for name in datasets:
            x, y, pool, x_te, y_te = prepared[name]
            for frac in fractions:
                b_aucs, e_aucs = [], []
                for seed in seeds:
                    sub = _stratified_split(y[pool], frac, np.random.RandomState(seed + 1))[0]
                    tr = pool[sub]
                    e_auc, _ = _fit_eval(
                        x[tr],
                        y[tr],
                        x_te,
                        y_te,
                        epochs,
                        seed,
                        domain_encoder=True,
                        config=cfg or None,
                    )
                    b_aucs.append(baseline[(name, frac, seed)])
                    e_aucs.append(e_auc)
                confound = check_ablation_confound(b_aucs, e_aucs, max_degenerate_fraction=0.0)
                cells.append(
                    {
                        "config": cfg_name,
                        "dataset": name,
                        "family": FAMILY[name],
                        "fraction": frac,
                        "baseline_aucs": [round(a, 4) for a in b_aucs],
                        "encoder_aucs": [round(a, 4) for a in e_aucs],
                        "delta_auc": float(np.mean(e_aucs) - np.mean(b_aucs)),
                        "confounded": confound.confounded,
                    }
                )
        print(f"  config done: {cfg_name}", flush=True)
    return {"cells": cells}


def stratified_verdict(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Stratify by config x family x data-size; a cell counts only if confound-free."""
    strata: dict[str, dict[str, Any]] = {}
    best_clean_delta = -1.0
    best_clean_where = None
    any_clean_keep = False
    for c in cells:
        size = "low_data" if c["fraction"] == LOW_DATA else "full_data"
        key = f"{c['family']}/{size}"
        s = strata.setdefault(
            key, {"clean_deltas": [], "confounded_deltas": [], "n_cells": 0, "n_confounded": 0}
        )
        s["n_cells"] += 1
        if c["confounded"]:
            s["n_confounded"] += 1
            s["confounded_deltas"].append(c["delta_auc"])
        else:
            s["clean_deltas"].append(c["delta_auc"])
            if c["delta_auc"] > best_clean_delta:
                best_clean_delta = c["delta_auc"]
                best_clean_where = f"{c['config']} on {c['dataset']} ({key})"
            if c["delta_auc"] > _AUC_MEANINGFUL:
                any_clean_keep = True
    summary = {}
    for key, s in sorted(strata.items()):
        cd = s["clean_deltas"]
        summary[key] = {
            "n_cells": s["n_cells"],
            "n_confounded": s["n_confounded"],
            "mean_clean_delta_auc": float(np.mean(cd)) if cd else None,
            "max_clean_delta_auc": float(np.max(cd)) if cd else None,
            "n_clean_cells_above_threshold": int(sum(d > _AUC_MEANINGFUL for d in cd)),
        }
    return {
        "auc_meaningful_threshold": _AUC_MEANINGFUL,
        "by_family_and_size": summary,
        "best_confound_free_delta": best_clean_delta,
        "best_confound_free_where": best_clean_where,
        "any_confound_free_cell_clears_threshold": any_clean_keep,
        "verdict": (
            "QUARANTINE (covered search) -- across the full design grid x family x "
            "data-size, no confound-free configuration produces a robust above-noise "
            "improvement; the differentiable encoder stays off by default. The earlier "
            "modest low-data signal is confined to confounded (inverted-baseline) cells "
            "or is sub-threshold."
            if not any_clean_keep
            else "INVESTIGATE -- a confound-free cell clears the noise threshold; review "
            "before promoting (must also be stable across seeds/datasets)."
        ),
    }


def main() -> int:
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--fractions", nargs="*", type=float, default=DEFAULT_FRACTIONS)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--out", default="artifacts/domain_encoder_sweep.json")
    args = ap.parse_args()

    print(
        f"Domain-encoder DESIGN SWEEP: {len(DESIGN_GRID)} configs x {args.datasets} "
        f"seeds={args.seeds} fractions={args.fractions} epochs={args.epochs}",
        flush=True,
    )
    swept = run_sweep(args.datasets, args.seeds, args.fractions, args.epochs)
    verdict = stratified_verdict(swept["cells"])
    artifact = {
        "metadata": {
            "purpose": "WS-B differentiable domain-encoder design-space sweep + family stratification",
            "dataset_source": "https://github.com/Minqi824/ADBench (MIT)",
            "families": {"hard": HARD, "ceiling": CEILING},
            "design_grid": list(DESIGN_GRID),
            "seeds": args.seeds,
            "fractions": args.fractions,
            "epochs": args.epochs,
            "metric": "ROC-AUC (mercury_ml, no sklearn)",
            "confound_guard": "evaluation.ablation_guard (inverted-ranking, strict)",
        },
        "cells": swept["cells"],
        "verdict": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nVERDICT: {verdict['verdict']}")
    print(
        f"  best confound-free ΔAUC = {verdict['best_confound_free_delta']:+.4f} "
        f"({verdict['best_confound_free_where']})"
    )
    for key, s in verdict["by_family_and_size"].items():
        print(
            f"  {key:<16} cells={s['n_cells']} confounded={s['n_confounded']} "
            f"mean_clean_dAUC={s['mean_clean_delta_auc']}"
        )
    print(f"  artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
