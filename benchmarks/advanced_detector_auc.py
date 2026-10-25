# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Held-out ROC-AUC of the advanced tabular detectors on real ADBench labels.

Same protocol as ``benchmarks/dormant_module_revival.py``: a stratified 30%
test split per seed, features standardised on the training split, detectors
fitted unsupervised on the training split and scored on the test split. The
shipped ``lof_reference`` is the baseline. Ranking only: the detectors' alarm
cuts (a training-score percentile) are not evaluated here.

``MultiScaleTransformerDetector`` is a sequence model and is not measured:
ADBench rows are not sequences.

Usage::

    python -m benchmarks.advanced_detector_auc
    python -m benchmarks.advanced_detector_auc --datasets Pima cardio --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.dormant_module_revival import _load_dataset, _score_lof_reference, _stratified

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_DATASETS = ["Pima", "breastw", "WBC", "cardio", "thyroid"]
DEFAULT_SEEDS = [0, 1, 2]
EPOCHS = 30


def _score_copod(
    X_tr: np.ndarray[Any, Any], y_tr: np.ndarray[Any, Any], X_te: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    from omni_mercury_engine.detectors.advanced.copod_detector import COPODDetector

    return np.asarray(COPODDetector().fit(X_tr).predict(X_te))


def _score_adversarial_ae(
    X_tr: np.ndarray[Any, Any], y_tr: np.ndarray[Any, Any], X_te: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    from omni_mercury_engine.detectors.advanced.adversarial_ae import (
        AdversarialAutoencoderDetector,
    )

    detector = AdversarialAutoencoderDetector(input_dim=X_tr.shape[1], epochs=EPOCHS, seed=0)
    return np.asarray(detector.fit(X_tr).predict(X_te))


def _score_contrastive(
    X_tr: np.ndarray[Any, Any], y_tr: np.ndarray[Any, Any], X_te: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    from omni_mercury_engine.detectors.advanced.contrastive_detector import (
        ContrastiveLearningDetector,
    )

    detector = ContrastiveLearningDetector(input_dim=X_tr.shape[1], epochs=EPOCHS, seed=0)
    return np.asarray(detector.fit(X_tr).predict(X_te))


SCORERS: dict[str, Callable[..., np.ndarray[Any, Any]]] = {
    "copod": _score_copod,
    "adversarial_ae": _score_adversarial_ae,
    "contrastive": _score_contrastive,
    "lof_reference": _score_lof_reference,
}


def measure(datasets: list[str], seeds: list[int]) -> dict[str, dict[str, float]]:
    """Mean held-out AUC per scorer and dataset."""
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    results: dict[str, dict[str, float]] = {name: {} for name in SCORERS}
    for dataset in datasets:
        X, y = _load_dataset(dataset)
        for name, scorer in SCORERS.items():
            aucs = []
            for seed in seeds:
                test = _stratified(y, 0.3, np.random.RandomState(seed))
                is_test = np.zeros(len(y), dtype=bool)
                is_test[test] = True
                train = np.where(~is_test)[0]
                if len(np.unique(y[test])) < 2:
                    continue
                mean, std = X[train].mean(0), X[train].std(0)
                std[std < 1e-8] = 1.0
                X_tr = ((X[train] - mean) / std).astype(np.float64)
                X_te = ((X[test] - mean) / std).astype(np.float64)
                scores = scorer(X_tr, y[train], X_te)
                aucs.append(float(roc_auc_score(y[test], scores)))
            if aucs:
                results[name][dataset] = float(np.mean(aucs))
    return results


def main() -> None:
    """Print one row per scorer: per-dataset mean AUC and the mean over datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    results = measure(args.datasets, args.seeds)
    for name, per_dataset in results.items():
        cells = "  ".join(f"{ds}={auc:.3f}" for ds, auc in per_dataset.items())
        mean = float(np.mean(list(per_dataset.values()))) if per_dataset else float("nan")
        print(f"{name:15s} mean={mean:.3f}  {cells}")


if __name__ == "__main__":
    main()
