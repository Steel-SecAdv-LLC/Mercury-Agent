# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""DB spectral-divergence ablation: does giving the dead term real semantics improve the live detector on real labels?

The 2026-06-11 native-acceleration pass disclosed that the
``DimensionalAnalyzer`` DB term's spectral-divergence component is
identically zero (single-row signatures are empty), so 50% of that score's
weight has always been a constant. The operator approved real semantics —
per-row feature-axis power spectra compared against the fit-time mean
training spectrum — **gated by this pre-registered ablation**, exactly the
discipline applied to every other change of shipped scores (cf.
``benchmarks/kmeans_ensemble_marginal.py``).

Design (paired, fail-closed):

* Real ADBench labels only; 5 datasets x 3 seeds; stratified 30% held-out.
* Per (dataset, seed): fit two ``DimensionalAnalyzer`` instances on the same
  train split under identical global seeds (the autoencoder lane follows the
  global RNG, so OFF/ON are exactly paired) — one with the legacy zero term
  (``db_spectral_divergence=False``, the pre-2026-06-11 default), one with
  the real semantics enabled.
* Measure held-out ROC-AUC of the full detector score (what ships) and of
  the DB term alone (mechanism context).

Pre-registered verdict bar (set before running):

* ``ENABLE_DEFAULT`` only if mean paired detector ΔAUC (ON − OFF) >= +0.002
  (the repo's established noise floor) AND the per-run sign agreement is
  >= 2/3. Otherwise ``KEEP_OPT_IN`` (legacy zero term stays the default —
  a redundant or harmful term does not get to move production numbers).
  Measured 2026-06-11: the gate CLEARED (mean dAUC +0.071, agreement
  0.93; see ``artifacts/db_spectral_ablation.json``), so the shipped
  default is now ``True``; this harness remains the re-runnable record.

Usage::

    python -m benchmarks.db_spectral_ablation
    python -m benchmarks.db_spectral_ablation \\
        --datasets cardio thyroid breastw WBC Pima --seeds 0 1 2 \\
        --out artifacts/db_spectral_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from typing import Any

import numpy as np

DEFAULT_DATASETS = ["cardio", "thyroid", "breastw", "WBC", "Pima"]
DEFAULT_SEEDS = [0, 1, 2]

# Pre-registered bars (see module docstring).
_DELTA_AUC_BAR = 0.002
_AGREEMENT_BAR = 2.0 / 3.0


def _load_dataset(name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": name}))
    loader.download()
    data = loader.load()
    return np.asarray(data[0], dtype=np.float64), np.asarray(data[1]).astype(int).ravel()


def _stratified(
    y: np.ndarray[Any, Any], frac: float, rng: np.random.RandomState
) -> np.ndarray[Any, Any]:
    keep: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        keep.extend(idx[: max(1, int(len(idx) * frac))].tolist())
    return np.array(sorted(keep))


def _fit_and_score(
    enable_term: bool,
    seed: int,
    X_tr: np.ndarray[Any, Any],
    X_te: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Fit one analyzer under pinned global seeds; return (scores, db_term)."""
    import torch

    from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer

    np.random.seed(seed)
    torch.manual_seed(seed)
    analyzer = DimensionalAnalyzer({"db_spectral_divergence": enable_term})
    analyzer.fit(X_tr)
    scores = np.asarray(analyzer.detect(X_te)["scores"], dtype=np.float64)
    db_term = np.asarray(analyzer._dimensional_code_breaking(X_te), dtype=np.float64)
    return scores, db_term


def run_dataset_seed(name: str, seed: int) -> dict[str, Any] | None:
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    X, y = _load_dataset(name)
    rng = np.random.RandomState(seed)
    te = _stratified(y, 0.3, rng)
    mask = np.zeros(len(y), dtype=bool)
    mask[te] = True
    tr = np.where(~mask)[0]
    if len(np.unique(y[te])) < 2:
        return None
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd < 1e-8] = 1.0
    X_tr, X_te = (X[tr] - mu) / sd, (X[te] - mu) / sd

    scores_off, db_off = _fit_and_score(False, seed, X_tr, X_te)
    scores_on, db_on = _fit_and_score(True, seed, X_tr, X_te)

    auc_off = float(roc_auc_score(y[te], scores_off))
    auc_on = float(roc_auc_score(y[te], scores_on))
    return {
        "dataset": name,
        "seed": seed,
        "auc_off": auc_off,
        "auc_on": auc_on,
        "delta_auc": auc_on - auc_off,
        "db_term_auc_off": float(roc_auc_score(y[te], db_off)),
        "db_term_auc_on": float(roc_auc_score(y[te], db_on)),
        "n_test": len(te),
    }


def derive_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = np.array([r["delta_auc"] for r in results], dtype=float)
    mean_delta = float(deltas.mean())
    agreement = float(np.mean(deltas > 0))
    enable_default = bool(mean_delta >= _DELTA_AUC_BAR and agreement >= _AGREEMENT_BAR)
    return {
        "mean_delta_auc": mean_delta,
        "seed_agreement": agreement,
        "mean_auc_off": float(np.mean([r["auc_off"] for r in results])),
        "mean_auc_on": float(np.mean([r["auc_on"] for r in results])),
        "mean_db_term_auc_off": float(np.mean([r["db_term_auc_off"] for r in results])),
        "mean_db_term_auc_on": float(np.mean([r["db_term_auc_on"] for r in results])),
        "n_runs": len(results),
        "enable_default": enable_default,
        "verdict": (
            "ENABLE_DEFAULT -- real spectral divergence improves the shipped detector "
            "beyond the noise floor with seed agreement; flip the default"
            if enable_default
            else "KEEP_OPT_IN -- real semantics measured; gain does not clear the "
            "pre-registered bar, so the legacy default (zero term) ships unchanged "
            "and the flag remains available for spectral-structured domains"
        ),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", default="artifacts/db_spectral_ablation.json", type=str)
    args = parser.parse_args()

    print("DB spectral-divergence ablation (paired OFF/ON on real ADBench labels)")
    print(f"datasets={args.datasets}  seeds={args.seeds}")
    print("-" * 84)

    results: list[dict[str, Any]] = []
    for name in args.datasets:
        for seed in args.seeds:
            try:
                row = run_dataset_seed(name, seed)
            except Exception as exc:
                print(f"  {name} seed={seed}: FAILED ({type(exc).__name__}: {exc})")
                continue
            if row is None:
                continue
            results.append(row)
            print(
                f"  {name:<9} seed={seed}  AUC off/on={row['auc_off']:.3f}/"
                f"{row['auc_on']:.3f}  dAUC={row['delta_auc']:+.4f}  "
                f"db-term off/on={row['db_term_auc_off']:.3f}/{row['db_term_auc_on']:.3f}"
            )

    if not results:
        print("INTEGRITY FAILURE: nothing could be measured (network unavailable?).")
        return 1

    verdict = derive_verdict(results)
    print("-" * 84)
    print(
        f"mean dAUC={verdict['mean_delta_auc']:+.4f}  "
        f"agreement={verdict['seed_agreement']:.2f}  "
        f"db-term AUC off/on={verdict['mean_db_term_auc_off']:.3f}/"
        f"{verdict['mean_db_term_auc_on']:.3f}"
    )
    print(f"VERDICT: {verdict['verdict']}")

    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "verdict": verdict}, indent=2, sort_keys=True))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
