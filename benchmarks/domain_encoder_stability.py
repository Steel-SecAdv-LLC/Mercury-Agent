# Copyright (C) 2025 Steel Security Advisors LLC
"""WS-B stability re-run: settle the design sweep's INVESTIGATE flag.

`benchmarks/domain_encoder_sweep.py` (3 seeds) returned **INVESTIGATE** — a
confound-free cell cleared the noise threshold (best +0.097, `wide_kernels` on
`glass`, hard/low-data). Per the mandate ("a quarantine may stand only on covered
search; unearned pessimism is as much a defect as unearned optimism"), that flag
must be run to ground rather than dismissed.

This harness re-runs the two strongest designs (`full_default`, `wide_kernels`)
at **8 seeds** on the hard family, **adding a well-powered hard/imbalanced set**
(`annthyroid`, ~160 test positives) so the verdict does not hinge on `glass`'s
3-positive test split. It reports, per (dataset, config): mean/std ΔAUC over the
8 seeds and the confound flag (baseline inversion count). A trigger that was real
would survive more seeds and reappear on a well-powered set; a small-sample or
baseline-collapse artifact regresses to noise.

Usage::

    python benchmarks/domain_encoder_stability.py --out artifacts/domain_encoder_stability.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "src"))

from benchmarks.domain_encoder_ablation import _load_dataset, _stratified_split
from benchmarks.domain_encoder_sweep import _fit_eval
from omni_mercury_engine.evaluation.ablation_guard import check_ablation_confound

CONFIGS: dict[str, dict[str, Any]] = {
    "full_default": {},
    "wide_kernels": {"encoder_kwargs": {"kinematic": {"kernel_widths": (2, 3, 4, 5, 6)}}},
}
DEFAULT_DATASETS = ["glass", "Pima", "annthyroid"]  # 3 / 80 / ~160 test positives
DEFAULT_SEEDS = list(range(8))
FRACTION = 0.25
_AUC_MEANINGFUL = 0.002


def run(datasets: list[str], seeds: list[int], epochs: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for name in datasets:
        x, y = _load_dataset(name)
        pool, test = _stratified_split(y, 0.7, np.random.RandomState(0))
        x_te, y_te = x[test], y[test]
        base = {}
        for s in seeds:
            sub = _stratified_split(y[pool], FRACTION, np.random.RandomState(s + 1))[0]
            tr = pool[sub]
            base[s] = _fit_eval(
                x[tr], y[tr], x_te, y_te, epochs, s, domain_encoder=False, config=None
            )[0]
        for cfg_name, cfg in CONFIGS.items():
            enc = []
            for s in seeds:
                sub = _stratified_split(y[pool], FRACTION, np.random.RandomState(s + 1))[0]
                tr = pool[sub]
                enc.append(
                    _fit_eval(
                        x[tr], y[tr], x_te, y_te, epochs, s, domain_encoder=True, config=cfg or None
                    )[0]
                )
            b = [base[s] for s in seeds]
            d = np.array(enc) - np.array(b)
            confound = check_ablation_confound(b, enc, max_degenerate_fraction=0.0)
            results.append(
                {
                    "dataset": name,
                    "test_positives": int(y_te.sum()),
                    "config": cfg_name,
                    "mean_delta_auc": float(d.mean()),
                    "std_delta_auc": float(d.std()),
                    "min_delta_auc": float(d.min()),
                    "max_delta_auc": float(d.max()),
                    "confounded": confound.confounded,
                    "baseline_inversions": confound.n_degenerate_baseline,
                    "n_seeds": len(seeds),
                }
            )
            print(
                f"  {name:<11}({int(y_te.sum())} pos) {cfg_name:<14} "
                f"dAUC={d.mean():+.4f}±{d.std():.4f} confounded={confound.confounded}",
                flush=True,
            )
    return {"results": results}


def verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """A design 'survives' only if it is confound-free AND mean-above-noise AND
    its mean exceeds its own std (not noise) on a well-powered set (>=50 pos)."""
    survivors = [
        r
        for r in results
        if (not r["confounded"])
        and r["mean_delta_auc"] > _AUC_MEANINGFUL
        and r["mean_delta_auc"] > r["std_delta_auc"]
        and r["test_positives"] >= 50
    ]
    return {
        "auc_meaningful_threshold": _AUC_MEANINGFUL,
        "survivors": [f"{r['config']} on {r['dataset']}" for r in survivors],
        "verdict": (
            "QUARANTINE -- INVESTIGATE flag resolved: no design survives the 8-seed, "
            "well-powered re-run (the sweep's best cell was small-sample / "
            "baseline-collapse, not a robust encoder gain). domain_encoder=False stays "
            "the default."
            if not survivors
            else "PROMOTE-CANDIDATE -- a design survived; recommend conditional enablement "
            "and a wider confirmation."
        ),
    }


def main() -> int:
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--out", default="artifacts/domain_encoder_stability.json")
    args = ap.parse_args()

    print(
        f"WS-B stability re-run: configs={list(CONFIGS)} datasets={args.datasets} "
        f"seeds={args.seeds} (resolving the sweep's INVESTIGATE flag)",
        flush=True,
    )
    swept = run(args.datasets, args.seeds, args.epochs)
    v = verdict(swept["results"])
    artifact = {
        "metadata": {
            "purpose": "WS-B: resolve the domain-encoder design-sweep INVESTIGATE flag",
            "dataset_source": "https://github.com/Minqi824/ADBench (MIT)",
            "configs": {k: (v or "full default") for k, v in CONFIGS.items()},
            "seeds": args.seeds,
            "fraction": FRACTION,
            "epochs": args.epochs,
            "metric": "ROC-AUC (mercury_ml, no sklearn)",
        },
        "results": swept["results"],
        "verdict": v,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"\nVERDICT: {v['verdict']}")
    print(f"  survivors: {v['survivors'] or 'none'}")
    print(f"  artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
