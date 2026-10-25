# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Held-out ROC-AUC of the Sigma directive detector on real ADBench labels.

Same protocol as ``benchmarks/advanced_detector_auc.py``: a stratified 30%
test split per seed, features standardised on the training split, the
detector fitted unsupervised on the training split and scored on the test
split with its default configuration. Ranking only.

Run it on two trees to compare a change, for example the parent of the
commit that introduced the absolute EOA scale against its child::

    python -m benchmarks.directive_signal --output after.json
    git worktree add ../before <commit>^ && cd ../before && \\
        python -m benchmarks.directive_signal --output ../before.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.dormant_module_revival import _load_dataset, _stratified

DEFAULT_DATASETS = ["Pima", "breastw", "WBC", "cardio", "thyroid"]
DEFAULT_SEEDS = [0, 1, 2]
TEST_FRACTION = 0.3


def directive_scores(
    X_tr: np.ndarray[Any, Any], X_te: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Fit the default directive detector on the training split; score the test split."""
    from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

    detector = SigmaDirectiveDetector()
    detector.fit(X_tr)
    return np.asarray(detector.detect(X_te)["scores"], dtype=np.float64)


def run(datasets: list[str], seeds: list[int]) -> dict[str, Any]:
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    per_dataset: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        X, y = _load_dataset(dataset)
        aucs: list[float] = []
        for seed in seeds:
            test = _stratified(y, TEST_FRACTION, np.random.RandomState(seed))
            if len(np.unique(y[test])) < 2:
                continue
            is_test = np.zeros(len(y), dtype=bool)
            is_test[test] = True
            train = np.where(~is_test)[0]
            mean, std = X[train].mean(0), X[train].std(0)
            std[std < 1e-8] = 1.0
            X_tr = ((X[train] - mean) / std).astype(np.float64)
            X_te = ((X[test] - mean) / std).astype(np.float64)
            scores = directive_scores(X_tr, X_te)
            aucs.append(float(roc_auc_score(y[test], scores)))
        per_dataset[dataset] = {
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "n_seeds": len(aucs),
        }
    means = [v["mean_auc"] for v in per_dataset.values() if np.isfinite(v["mean_auc"])]
    return {
        "datasets": per_dataset,
        "mean_auc": float(np.mean(means)) if means else float("nan"),
        "seeds": seeds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, help="write the JSON record here")
    args = parser.parse_args(argv)

    record = run(args.datasets, args.seeds)
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
