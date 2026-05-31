"""
Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3

Neuro-symbolic ablation: does symbolic co-training beat neural-only?

This is the anti-theater gate for the differentiable symbolic-constraint LTN
(:class:`omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule`).
It compares two *otherwise identical* training runs on **real, genuinely
labelled** ADBench datasets (NeurIPS 2022 ground truth):

* ``neural``    -- ``engine.fit_fusion(..., symbolic_weight=0.0)``
* ``symbolic``  -- ``engine.fit_fusion(..., symbolic_weight=lambda)``

For each (dataset, seed, train-fraction) both conditions see the same split and
the same network initialisation, so the only difference is the symbolic loss.
We measure three things the gate accepts as evidence:

1. **AUC up**            -- ROC-AUC on the held-out test set.
2. **False-positives down** -- false-positive rate at a fixed 90% recall.
3. **Sample-efficiency up** -- AUC gain concentrated in the low-data regime
   (train fractions 0.1 / 0.25), where an unsupervised consensus prior should
   help most.

Ablation integrity (non-negotiable): metrics are computed on real held-out
labels only. If ADBench cannot be downloaded the run reports that plainly and
exits non-zero -- it never fabricates or simulates a pass.

Usage::

    python -m benchmarks.neurosymbolic_ablation
    python -m benchmarks.neurosymbolic_ablation \\
        --datasets breastw cardio thyroid --seeds 0 1 2 \\
        --fractions 0.1 0.25 0.5 1.0 --lam 0.1 --epochs 20 \\
        --out artifacts/neurosymbolic_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)

# Genuinely-labelled ADBench datasets (ground-truth anomaly labels), small
# enough to train quickly on CPU. Must stay in lockstep with the de-leaked
# headline set in benchmarks/fusion_raw_benchmark.py -- no statistically /
# heuristically labelled source is admitted, so every metric here is honest.
DEFAULT_DATASETS = ["breastw", "cardio", "thyroid", "WBC", "Pima"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
LOW_DATA_FRACTIONS = (0.1, 0.25)

# Verdict thresholds (conservative; an improvement must clear noise to count).
_AUC_MEANINGFUL = 0.002
_SAMPLE_EFF_MEANINGFUL = 0.005


def fpr_at_recall(
    y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any], target_recall: float = 0.9
) -> float:
    """False-positive rate at the lowest threshold reaching ``target_recall``.

    Sweeps every score as a candidate threshold and returns the FPR at the
    operating point whose true-positive rate (recall) first reaches
    ``target_recall``. This is the "how many false alarms to catch 90% of real
    anomalies" number -- the quantity the precision/false-positive rule targets.

    Args:
        y_true: Binary ground-truth labels ``(n,)`` (1 = anomaly).
        y_score: Anomaly scores ``(n,)`` (higher = more anomalous).
        target_recall: Recall the operating point must achieve (default 0.9).

    Returns:
        FPR in ``[0, 1]`` at that operating point, or ``nan`` if a class is
        absent (FPR/recall undefined).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / n_pos
    reached = int(np.searchsorted(recall, target_recall, side="left"))
    reached = min(reached, len(recall) - 1)
    return float(fp[reached] / n_neg)


@dataclass
class Condition:
    """Aggregated metrics for one training condition at one train-fraction."""

    auc_mean: float
    auc_std: float
    fpr_mean: float
    fpr_std: float
    aucs: list[float] = field(default_factory=list)
    fprs: list[float] = field(default_factory=list)


@dataclass
class FractionResult:
    """Paired neural vs symbolic result at one (dataset, train-fraction)."""

    fraction: float
    neural: Condition
    symbolic: Condition
    delta_auc_mean: float  # symbolic - neural (positive = AUC improved)
    delta_fpr_mean: float  # neural - symbolic (positive = false-positives down)
    seeds_auc_better: int  # of n_seeds, how many had symbolic AUC >= neural
    n_seeds: int


def _condition(aucs: list[float], fprs: list[float]) -> Condition:
    valid_auc = [a for a in aucs if not np.isnan(a)]
    valid_fpr = [f for f in fprs if not np.isnan(f)]
    return Condition(
        auc_mean=float(np.mean(valid_auc)) if valid_auc else float("nan"),
        auc_std=float(np.std(valid_auc)) if valid_auc else float("nan"),
        fpr_mean=float(np.mean(valid_fpr)) if valid_fpr else float("nan"),
        fpr_std=float(np.std(valid_fpr)) if valid_fpr else float("nan"),
        aucs=aucs,
        fprs=fprs,
    )


def _load_dataset(name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": name}))
    loader.download()
    data = loader.load()
    X = np.asarray(data[0], dtype=np.float32)
    y = np.asarray(data[1]).astype(int).ravel()
    return X, y


def _stratified_indices(
    y: np.ndarray[Any, Any], frac: float, rng: np.random.RandomState
) -> np.ndarray[Any, Any]:
    """Stratified subsample keeping at least one sample of each present class."""
    keep: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        cut = max(1, int(round(len(cls_idx) * frac)))
        keep.extend(cls_idx[:cut].tolist())
    return np.array(sorted(keep))


def _train_eval(
    X_tr: np.ndarray[Any, Any],
    y_tr: np.ndarray[Any, Any],
    X_te: np.ndarray[Any, Any],
    y_te: np.ndarray[Any, Any],
    lam: float,
    epochs: int,
    seed: int,
) -> tuple[float, float]:
    """Train one fusion model and return ``(auc, fpr_at_90_recall)`` on test."""
    import torch

    from omni_mercury_engine.engine import OmniMercuryEngine
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    # Seed both RNGs so neural and symbolic conditions share initialisation,
    # train/val split and detector fits -- the only difference is ``lam``.
    torch.manual_seed(seed)
    np.random.seed(seed)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(
        X_tr, y_tr, epochs=epochs, batch_size=64, early_stopping_patience=15, symbolic_weight=lam
    )
    probs = engine.score_fusion(X_te)
    return float(roc_auc_score(y_te, probs)), fpr_at_recall(y_te, probs)


def run_dataset(
    name: str, seeds: list[int], fractions: list[float], lam: float, epochs: int
) -> dict[str, Any] | None:
    """Run the paired ablation for one dataset across seeds and fractions."""
    try:
        X, y = _load_dataset(name)
    except Exception as exc:
        print(f"  {name:<12} SKIP (load failed: {exc})")
        return None

    n_pos = int((y == 1).sum())
    if n_pos < 4 or len(np.unique(y)) < 2:
        print(f"  {name:<12} SKIP (too few anomalies for an honest split)")
        return None

    fraction_results: list[FractionResult] = []
    for frac in fractions:
        n_seeds = 0
        neu_auc: list[float] = []
        neu_fpr: list[float] = []
        sym_auc: list[float] = []
        sym_fpr: list[float] = []
        auc_better = 0
        for seed in seeds:
            # Fixed stratified test split per (dataset, seed); subsample train.
            rng = np.random.RandomState(seed)
            test_idx = _stratified_indices(y, 0.3, rng)
            test_mask = np.zeros(len(y), dtype=bool)
            test_mask[test_idx] = True
            train_pool = np.where(~test_mask)[0]
            sub = _stratified_indices(y[train_pool], frac, np.random.RandomState(seed + 1))
            train_idx = train_pool[sub]
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue

            a_n, f_n = _train_eval(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx], 0.0, epochs, seed
            )
            a_s, f_s = _train_eval(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx], lam, epochs, seed
            )
            neu_auc.append(a_n)
            neu_fpr.append(f_n)
            sym_auc.append(a_s)
            sym_fpr.append(f_s)
            auc_better += int(a_s >= a_n)
            n_seeds += 1

        if n_seeds == 0:
            continue
        neural = _condition(neu_auc, neu_fpr)
        symbolic = _condition(sym_auc, sym_fpr)
        fraction_results.append(
            FractionResult(
                fraction=frac,
                neural=neural,
                symbolic=symbolic,
                delta_auc_mean=symbolic.auc_mean - neural.auc_mean,
                delta_fpr_mean=neural.fpr_mean - symbolic.fpr_mean,
                seeds_auc_better=auc_better,
                n_seeds=n_seeds,
            )
        )
        print(
            f"  {name:<12} frac={frac:<4} "
            f"AUC {neural.auc_mean:.4f} -> {symbolic.auc_mean:.4f} "
            f"(d={symbolic.auc_mean - neural.auc_mean:+.4f})  "
            f"FPR@90 {neural.fpr_mean:.4f} -> {symbolic.fpr_mean:.4f} "
            f"(d={neural.fpr_mean - symbolic.fpr_mean:+.4f})  "
            f"[{auc_better}/{n_seeds} seeds AUC>=]"
        )

    if not fraction_results:
        return None
    return {"dataset": name, "fractions": [asdict(fr) for fr in fraction_results]}


def derive_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a transparent keep/quarantine verdict from the measured deltas.

    No result is hard-coded: each gate is a simple, conservative aggregate of
    the paired deltas. The constraint is recommended for default-on only if at
    least one gate clears its noise threshold with a majority of seeds agreeing.
    """
    from omni_mercury_engine.evaluation.ablation_guard import (
        check_ablation_confound,
        confound_free_or_quarantine,
    )

    full_auc_deltas: list[float] = []
    full_fpr_deltas: list[float] = []
    low_auc_deltas: list[float] = []
    full_seed_agree: list[float] = []
    all_neural_aucs: list[float] = []
    all_symbolic_aucs: list[float] = []

    for ds in results:
        for fr in ds["fractions"]:
            all_neural_aucs.extend(fr.get("neural", {}).get("aucs", []))
            all_symbolic_aucs.extend(fr.get("symbolic", {}).get("aucs", []))
            if fr["fraction"] >= 1.0:
                full_auc_deltas.append(fr["delta_auc_mean"])
                if not np.isnan(fr["delta_fpr_mean"]):
                    full_fpr_deltas.append(fr["delta_fpr_mean"])
                full_seed_agree.append(fr["seeds_auc_better"] / max(1, fr["n_seeds"]))
            if fr["fraction"] in LOW_DATA_FRACTIONS:
                low_auc_deltas.append(fr["delta_auc_mean"])

    mean_full_auc = float(np.mean(full_auc_deltas)) if full_auc_deltas else float("nan")
    mean_full_fpr = float(np.mean(full_fpr_deltas)) if full_fpr_deltas else float("nan")
    mean_low_auc = float(np.mean(low_auc_deltas)) if low_auc_deltas else float("nan")
    seed_agree = float(np.mean(full_seed_agree)) if full_seed_agree else float("nan")

    gate_auc = mean_full_auc > _AUC_MEANINGFUL and seed_agree >= 0.5
    gate_fp = mean_full_fpr > 0.0 and seed_agree >= 0.5
    gate_sample_eff = (
        mean_low_auc > _SAMPLE_EFF_MEANINGFUL
        and not np.isnan(mean_full_auc)
        and mean_low_auc > mean_full_auc
    )
    raw_passed = bool(gate_auc or gate_fp or gate_sample_eff)

    # Confound guard: reject a KEEP built on a collapsed (inverted-ranking) arm.
    confound = check_ablation_confound(
        all_neural_aucs, all_symbolic_aucs, max_degenerate_fraction=0.2
    )
    passed, note = confound_free_or_quarantine(raw_passed, confound)

    return {
        "mean_delta_auc_full_data": mean_full_auc,
        "mean_delta_fpr_full_data": mean_full_fpr,
        "mean_delta_auc_low_data": mean_low_auc,
        "seed_agreement_full_data": seed_agree,
        "gate_auc_up": bool(gate_auc),
        "gate_false_positives_down": bool(gate_fp),
        "gate_sample_efficiency_up": bool(gate_sample_eff),
        "raw_passed": raw_passed,
        "confound": confound.as_dict(),
        "passed": passed,
        "verdict": (
            "KEEP -- enable symbolic co-training by default"
            if passed
            else (
                note
                if confound.confounded
                else "QUARANTINE -- keep symbolic_weight=0 default; no measured improvement"
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--fractions", nargs="*", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--lam", type=float, default=0.1, help="symbolic_weight lambda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("artifacts/neurosymbolic_ablation.json"))
    args = parser.parse_args()

    print("Neuro-symbolic ablation (neural-only vs neural+symbolic, real ADBench labels)")
    print(
        f"lambda={args.lam}  seeds={args.seeds}  fractions={args.fractions}  epochs={args.epochs}"
    )
    print("-" * 88)

    results: list[dict[str, Any]] = []
    for name in args.datasets:
        out = run_dataset(name, args.seeds, args.fractions, args.lam, args.epochs)
        if out is not None:
            results.append(out)

    print("-" * 88)
    if not results:
        print(
            "ABLATION INTEGRITY FAILURE: no genuinely-labelled dataset could be "
            "measured (network unavailable?). Not reporting a verdict on absent data."
        )
        return 1

    verdict = derive_verdict(results)
    report = {
        "config": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "fractions": args.fractions,
            "lambda": args.lam,
            "epochs": args.epochs,
        },
        "results": results,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"datasets measured: {len(results)}")
    print(f"mean dAUC (full data):  {verdict['mean_delta_auc_full_data']:+.4f}")
    print(f"mean dFPR@90 reduction: {verdict['mean_delta_fpr_full_data']:+.4f}")
    print(f"mean dAUC (low data):   {verdict['mean_delta_auc_low_data']:+.4f}")
    print(
        "gates: "
        f"AUC_up={verdict['gate_auc_up']}  "
        f"FP_down={verdict['gate_false_positives_down']}  "
        f"sample_eff={verdict['gate_sample_efficiency_up']}"
    )
    print(f"VERDICT: {verdict['verdict']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
