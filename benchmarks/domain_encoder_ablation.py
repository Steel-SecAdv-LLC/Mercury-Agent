# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Differentiable domain-encoder ablation (WS-B / Target 2).

Faithful, paired comparison on real ADBench labels of the **actual wired
feature** -- the opt-in differentiable domain encoder in the production fusion
path:

* ``baseline`` -- ``engine.fit_fusion(X, y, domain_encoder=False)`` (the
  default supervised fusion net);
* ``encoder``  -- ``engine.fit_fusion(X, y, domain_encoder=True)`` -- the same
  fusion net plus the jointly-trained :class:`DomainEncoderStack`
  (FFT-spectral + finite-difference-kinematic + Fisher/entropy nn.Modules).

For each (dataset, seed, train-fraction) both arms see the identical split, the
identical fusion machinery, and the identical optimiser budget; the only
difference is whether the differentiable domain encoder is wired in. Both arms
are fully supervised through the real fusion path, so the comparison is fair
(no supervised-vs-unsupervised or dimensionality confound) and
``delta_auc = encoder - baseline`` directly measures the value of the feature.

This replaces an earlier self-contained design (a tiny head on the production
3 component scores, and a frozen-vs-learnable encoder proxy). Both proxies were
confounded: on imbalanced datasets a small head on weak/random features
converged to *inverted* rankings (AUC < 0.5), inflating the delta to a
meaningless +0.4-0.9. The fusion net is a robust supervised learner, so the
wired-path comparison is the transparent one.

The verdict uses the same conservative noise thresholds as
``neurosymbolic_ablation.py``: an improvement must clear noise on a majority of
seeds; otherwise the default stays off (quarantined).

Usage::

    python benchmarks/domain_encoder_ablation.py \\
        --datasets cardio Pima thyroid --seeds 0 1 2 \\
        --fractions 0.25 1.0 --epochs 25 \\
        --out artifacts/domain_encoder_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.engine import OmniMercuryEngine
from omni_mercury_engine.evaluation.ablation_guard import (
    check_ablation_confound,
    confound_free_or_quarantine,
)
from omni_mercury_engine.ml.mercury_ml import roc_auc_score

DEFAULT_DATASETS = ["cardio", "Pima", "thyroid"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FRACTIONS = [0.25, 1.0]
LOW_DATA_FRACTIONS = (0.25,)

# Conservative verdict thresholds (mirrors neurosymbolic_ablation.py): an
# improvement must clear noise to count as real.
_AUC_MEANINGFUL = 0.002


def _f1_at_best_threshold(y_true: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]) -> float:
    """Best-threshold (oracle) F1 over a 101-point sweep -- no sklearn."""
    best = 0.0
    if float(y_true.sum()) == 0:
        return 0.0
    for thr in np.linspace(0.0, 1.0, 101):
        pred = (scores > thr).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        denom = 2 * tp + fp + fn
        if denom > 0:
            best = max(best, 2 * tp / denom)
    return best


def _eval(
    x_tr: np.ndarray[Any, Any],
    y_tr: np.ndarray[Any, Any],
    x_te: np.ndarray[Any, Any],
    y_te: np.ndarray[Any, Any],
    epochs: int,
    seed: int,
    domain_encoder: bool,
) -> tuple[float, float]:
    """Train one arm through the real fusion path; return (test AUC, oracle-F1)."""
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
        )
        scores = engine.score_fusion(x_te)
    return float(roc_auc_score(y_te, scores)), _f1_at_best_threshold(y_te, scores)


@dataclass
class FractionResult:
    fraction: float
    n_train: int
    baseline_auc: float
    encoder_auc: float
    delta_auc: float
    baseline_f1: float
    encoder_f1: float
    seeds_encoder_wins: int
    n_seeds: int
    baseline_seed_aucs: list[float] = field(default_factory=list)
    encoder_seed_aucs: list[float] = field(default_factory=list)


def _load_dataset(name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    cfg = DatasetConfig(name=f"adbench-{name}", preprocessing={"dataset": name})
    loader = ADBenchLoader(cfg)
    loader.download()
    x, y = loader._load_raw()
    return np.nan_to_num(np.asarray(x, dtype=np.float32)), (np.asarray(y) > 0).astype(int)


def _stratified_split(
    y: np.ndarray[Any, Any], train_frac: float, rng: np.random.RandomState
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    tr: list[int] = []
    te: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = max(1, round(len(idx) * train_frac))
        tr.extend(idx[:cut].tolist())
        te.extend(idx[cut:].tolist())
    return np.array(tr), np.array(te)


def run_dataset(name: str, seeds: list[int], fractions: list[float], epochs: int) -> dict[str, Any]:
    x, y = _load_dataset(name)
    pool_tr, test_idx = _stratified_split(y, 0.7, np.random.RandomState(0))
    x_te, y_te = x[test_idx], y[test_idx]

    fraction_results: list[FractionResult] = []
    for frac in fractions:
        base_aucs, enc_aucs, base_f1s, enc_f1s = [], [], [], []
        wins = 0
        n_train_used = 0
        for seed in seeds:
            sub = _stratified_split(y[pool_tr], frac, np.random.RandomState(seed + 1))[0]
            tr_idx = pool_tr[sub]
            x_tr, y_tr = x[tr_idx], y[tr_idx]
            n_train_used = len(tr_idx)
            b_auc, b_f1 = _eval(x_tr, y_tr, x_te, y_te, epochs, seed, domain_encoder=False)
            e_auc, e_f1 = _eval(x_tr, y_tr, x_te, y_te, epochs, seed, domain_encoder=True)
            base_aucs.append(b_auc)
            enc_aucs.append(e_auc)
            base_f1s.append(b_f1)
            enc_f1s.append(e_f1)
            if e_auc - b_auc > _AUC_MEANINGFUL:
                wins += 1
        b_mean = float(np.mean(base_aucs))
        e_mean = float(np.mean(enc_aucs))
        fr = FractionResult(
            fraction=frac,
            n_train=n_train_used,
            baseline_auc=b_mean,
            encoder_auc=e_mean,
            delta_auc=e_mean - b_mean,
            baseline_f1=float(np.mean(base_f1s)),
            encoder_f1=float(np.mean(enc_f1s)),
            seeds_encoder_wins=wins,
            n_seeds=len(seeds),
            baseline_seed_aucs=[float(a) for a in base_aucs],
            encoder_seed_aucs=[float(a) for a in enc_aucs],
        )
        fraction_results.append(fr)
        print(
            f"  {name:<10} frac={frac:<5} n_tr={fr.n_train:<5} "
            f"baseline_auc={b_mean:.4f} encoder_auc={e_mean:.4f} "
            f"dAUC={fr.delta_auc:+.4f} wins={wins}/{len(seeds)}",
            flush=True,
        )
    return {"dataset": name, "fractions": [asdict(fr) for fr in fraction_results]}


def derive_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Transparent KEEP/QUARANTINE verdict from measured deltas, guarded against
    the inverted-ranking confound that produced PR #262's spurious +0.48."""
    full_deltas, low_deltas = [], []
    all_base_aucs: list[float] = []
    all_enc_aucs: list[float] = []
    agree = 0
    n_low = 0
    for ds in results:
        for fr in ds["fractions"]:
            all_base_aucs.extend(fr.get("baseline_seed_aucs", []))
            all_enc_aucs.extend(fr.get("encoder_seed_aucs", []))
            if fr["fraction"] >= 1.0:
                full_deltas.append(fr["delta_auc"])
            if fr["fraction"] in LOW_DATA_FRACTIONS:
                low_deltas.append(fr["delta_auc"])
                n_low += 1
                if fr["seeds_encoder_wins"] >= (fr["n_seeds"] + 1) // 2:
                    agree += 1
    mean_full = float(np.mean(full_deltas)) if full_deltas else 0.0
    mean_low = float(np.mean(low_deltas)) if low_deltas else 0.0
    seed_agree = agree / n_low if n_low else 0.0
    raw_cleared = mean_low > _AUC_MEANINGFUL and seed_agree >= 0.5

    # Confound guard: a KEEP built on a collapsed (inverted-ranking) arm is not
    # real. Tolerate one noisy seed across the whole sweep, but flag systematic
    # inversion -- exactly the artifact that faked the +0.48 in PR #262.
    confound = check_ablation_confound(all_base_aucs, all_enc_aucs, max_degenerate_fraction=0.2)
    cleared, note = confound_free_or_quarantine(raw_cleared, confound)
    return {
        "mean_full_data_delta_auc": mean_full,
        "mean_low_data_delta_auc": mean_low,
        "low_data_seed_agreement": seed_agree,
        "auc_meaningful_threshold": _AUC_MEANINGFUL,
        "raw_cleared_bar": raw_cleared,
        "confound": confound.as_dict(),
        "cleared_bar": cleared,
        "verdict": (
            (
                "KEEP -- the differentiable domain encoder improves the fusion path on real "
                "labels, clearing the conservative bar"
            )
            if cleared
            else (
                note
                if confound.confounded
                else "QUARANTINE -- keep domain_encoder=False default; wiring the differentiable "
                "encoder into the fusion path does not clear the conservative bar"
            )
        ),
    }


def main() -> int:
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--fractions", nargs="*", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", default="artifacts/domain_encoder_ablation.json")
    args = parser.parse_args()

    print(
        f"Domain-encoder ablation (fusion without vs with encoder): datasets={args.datasets} "
        f"seeds={args.seeds} fractions={args.fractions} epochs={args.epochs}",
        flush=True,
    )
    results = [run_dataset(n, args.seeds, args.fractions, args.epochs) for n in args.datasets]
    verdict = derive_verdict(results)
    artifact = {
        "metadata": {
            "purpose": "WS-B differentiable domain-encoder wired into the fusion path: ablation",
            "dataset_source": "https://github.com/Minqi824/ADBench",
            "dataset_license": "MIT",
            "seeds": args.seeds,
            "fractions": args.fractions,
            "epochs": args.epochs,
            "metric": "ROC-AUC (mercury_ml, no sklearn); oracle-F1 101-pt sweep",
            "arms": {
                "baseline": "engine.fit_fusion(domain_encoder=False)",
                "encoder": "engine.fit_fusion(domain_encoder=True) -- + DomainEncoderStack",
            },
            "isolation": "identical split/fusion/optimiser; only domain_encoder flag differs",
        },
        "results": results,
        "verdict": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nVERDICT: {verdict['verdict']}")
    print(
        f"  mean full-data dAUC={verdict['mean_full_data_delta_auc']:+.4f}  "
        f"mean low-data dAUC={verdict['mean_low_data_delta_auc']:+.4f}  "
        f"low-data seed agreement={verdict['low_data_seed_agreement']:.2f}"
    )
    print(f"  artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
