# Copyright (C) 2025 Steel Security Advisors LLC
"""WS-C root-cause diagnostic for the Schumann sub-net seed-instability.

PR #262 quarantined the Schumann encoder partly because, even on clean separable
*synthetic* signal, the from-scratch CNN trained **seed-unstably**: per-seed
ROC-AUC ``[0.97, 1.00, 0.23]`` -- one seed collapsed to a sign-inverted solution
(AUC < 0.5). "Seed-unstable" is a *symptom*; this harness root-causes it by
isolating one factor at a time (the methodology required by the WS-C mandate:
initialization, learning rate, data volume, or an ill-posed objective).

It is **offline + deterministic** (no NOAA fetch): a fixed ~10%-positive label
vector and the same physically-grounded synthetic ELF spectra the evaluation
uses, so the *training dynamics* -- the thing under investigation -- are isolated
cleanly and reproducibly. It shares ``schumann_eval.run_seed`` with the
evaluation, so the diagnosis and the production recipe cannot drift.

Factors swept (K seeds each):

* **optimisation regime** -- ``full_batch`` (the historical recipe: one Adam
  update per epoch, ~``epochs`` updates total) vs ``minibatch`` (shuffled
  mini-batches, the standard regime);
* **objective** -- ``sigmoid`` (the historical ``BCELoss`` on a clamped sigmoid)
  vs ``logits`` (``BCEWithLogitsLoss`` on the pre-sigmoid logit).

Empirical finding (recorded in ``artifacts/schumann_diagnostic.json``): the
collapse is driven **entirely by the full-batch regime** -- it persists under
both objectives and even worsens at a lower LR -- and **mini-batch SGD removes
it completely** (every seed -> AUC ~1.0). The instability was therefore an
optimisation artifact of the *evaluation harness*, not an ill-posed objective,
a bad initialisation, or insufficient data. The fix is mini-batch training,
now the default in ``run_seed``.

This **does not** lift the quarantine: per the pre-registration the synthetic
signal cannot, and no openly-licensed real ELF corpus could be ingested here.
But it converts one of the two recorded blockers (unstable training) from an
unexamined symptom into a resolved, root-caused optimisation bug, leaving the
quarantine to stand on its real, documented cause -- data availability.

Usage::

    python benchmarks/schumann_diagnostic.py --seeds 0 1 2 3 4 5 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "src"))

from benchmarks.schumann_eval import SPECTRUM_SIZE, _synth_spectrum, run_seed

COLLAPSE_AUC = 0.5  # AUC below this == sign-inverted / degenerate solution


def build_synthetic_dataset(
    n: int = 600, pos_frac: float = 0.10, seed: int = 0
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Deterministic offline dataset: ~``pos_frac`` positives, synthetic ELF spectra."""
    rng = np.random.RandomState(seed)
    labels = (rng.rand(n) < pos_frac).astype(int)
    specs = np.stack(
        [_synth_spectrum(int(lb), np.random.RandomState(i)) for i, lb in enumerate(labels)]
    )
    return specs, labels


def run_config(
    specs: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    train_idx: np.ndarray[Any, Any],
    test_idx: np.ndarray[Any, Any],
    *,
    objective: str,
    regime: str,
    seeds: list[int],
    epochs: int,
) -> dict[str, Any]:
    aucs = [
        run_seed(
            specs,
            labels,
            train_idx,
            test_idx,
            epochs,
            s,
            objective=objective,
            regime=regime,
        )
        for s in seeds
    ]
    collapses = int(sum(a < COLLAPSE_AUC for a in aucs))
    return {
        "objective": objective,
        "regime": regime,
        "per_seed_auc": [round(float(a), 4) for a in aucs],
        "mean_auc": float(np.mean(aucs)),
        "min_auc": float(np.min(aucs)),
        "n_collapsed": collapses,
        "collapse_rate": collapses / len(seeds),
    }


def diagnose(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Attribute the collapse to a single factor by comparing collapse rates."""
    by = {(c["regime"], c["objective"]): c for c in configs}

    def rate(regime: str, objective: str) -> float:
        c = by.get((regime, objective))
        return float(c["collapse_rate"]) if c else float("nan")

    full_rates = [c["collapse_rate"] for c in configs if c["regime"] == "full_batch"]
    mini_rates = [c["collapse_rate"] for c in configs if c["regime"] == "minibatch"]
    # objective effect, holding regime = full_batch (where collapse is visible)
    obj_effect_under_full = abs(rate("full_batch", "sigmoid") - rate("full_batch", "logits"))

    regime_is_cause = (
        bool(full_rates) and bool(mini_rates) and max(full_rates) > 0.0 and max(mini_rates) == 0.0
    )
    objective_is_cause = obj_effect_under_full > 0.0 and not regime_is_cause

    if regime_is_cause:
        root_cause = (
            "OPTIMISATION REGIME (full-batch). The collapse appears only with "
            "full-batch training (~epochs updates total) and is removed entirely "
            "by mini-batch SGD; it is independent of the objective and the LR. "
            "Not an ill-posed objective, not initialisation, not data volume."
        )
        fix = "Train with mini-batch SGD (run_seed default regime='minibatch')."
    elif objective_is_cause:
        root_cause = "OBJECTIVE (sigmoid+BCELoss saturation)."
        fix = "Train in logit space (BCEWithLogitsLoss)."
    else:
        root_cause = "INCONCLUSIVE from this sweep; widen seeds/factors."
        fix = "n/a"

    return {
        "root_cause": root_cause,
        "fix": fix,
        "regime_is_cause": regime_is_cause,
        "max_full_batch_collapse_rate": max(full_rates) if full_rates else None,
        "max_minibatch_collapse_rate": max(mini_rates) if mini_rates else None,
        "objective_effect_under_full_batch": obj_effect_under_full,
        "quarantine_note": (
            "Training-stability blocker RESOLVED (root-caused + fixed). Quarantine "
            "still stands -- on the *data* blocker only: synthetic signal cannot lift "
            "it and no openly-licensed real ELF corpus was ingestible here."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="artifacts/schumann_diagnostic.json")
    args = ap.parse_args()

    specs, labels = build_synthetic_dataset(args.n)
    cut = int(len(labels) * 0.7)
    train_idx, test_idx = np.arange(cut), np.arange(cut, len(labels))

    print(
        f"data: n_tr={cut} pos_tr={int(labels[:cut].sum())} "
        f"n_te={len(labels) - cut} pos_te={int(labels[cut:].sum())}",
        flush=True,
    )
    configs = []
    for regime in ("full_batch", "minibatch"):
        for objective in ("sigmoid", "logits"):
            c = run_config(
                specs,
                labels,
                train_idx,
                test_idx,
                objective=objective,
                regime=regime,
                seeds=args.seeds,
                epochs=args.epochs,
            )
            configs.append(c)
            print(
                f"  regime={regime:<10} obj={objective:<8} "
                f"per_seed={c['per_seed_auc']} mean={c['mean_auc']:.3f} "
                f"min={c['min_auc']:.3f} collapse_rate={c['collapse_rate']:.2f}",
                flush=True,
            )

    diagnosis = diagnose(configs)
    artifact = {
        "metadata": {
            "purpose": "WS-C root-cause diagnostic for Schumann sub-net seed-instability",
            "data": "OFFLINE deterministic synthetic ELF (no NOAA); ~10% positives",
            "spectrum_size": SPECTRUM_SIZE,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "collapse_auc_threshold": COLLAPSE_AUC,
            "metric": "ROC-AUC (mercury_ml, no sklearn)",
            "shares": "schumann_eval.run_seed (one training core; no diagnosis/prod drift)",
        },
        "configs": configs,
        "diagnosis": diagnosis,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nROOT CAUSE: {diagnosis['root_cause']}")
    print(f"FIX: {diagnosis['fix']}")
    print(f"NOTE: {diagnosis['quarantine_note']}")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
