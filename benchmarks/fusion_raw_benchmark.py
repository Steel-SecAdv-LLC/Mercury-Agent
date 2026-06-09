# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Raw fusion training-path benchmark.

Demonstrates the headline claim for the "expose the real training path"
work: ``OmniMercuryEngine.fit_fusion(X, y)`` trained directly on *raw*
genuinely-labelled features reproduces ~0.90 median ROC-AUC, with no manual
feature-dict assembly.

For each ADBench dataset (real ground-truth labels, NeurIPS 2022) this:
  1. loads raw (X, y),
  2. splits train/test deterministically,
  3. trains via the raw path ``engine.fit_fusion(X_train, y_train)``,
  4. scores the held-out test set with ``engine.score_fusion`` and reports AUC.

Requires network access to download the ADBench NPZ files (cached after the
first run). Run with::

    python -m benchmarks.fusion_raw_benchmark
    python -m benchmarks.fusion_raw_benchmark --datasets thyroid WBC cardio
"""

from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np

# Genuinely-labelled ADBench datasets (ground-truth anomaly labels), small
# enough to train quickly on CPU. Excludes any heuristically/threshold-labelled
# source so the reported AUC is honest (see loaders de-leak work).
DEFAULT_DATASETS = ["thyroid", "WBC", "cardio", "Pima", "breastw", "Ionosphere"]


def _run_one(name: str, epochs: int, seed: int) -> float | None:
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig
    from omni_mercury_engine.engine import OmniMercuryEngine
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    try:
        loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": name}))
        loader.download()
        data = loader.load()
        X = np.asarray(data[0], dtype=np.float32)
        y = np.asarray(data[1]).astype(int).ravel()
    except Exception as exc:
        print(f"  {name:<14} SKIP (load failed: {exc})")
        return None

    # Stratified 70/30 split so rare-anomaly datasets keep both classes in
    # train and test (avoids single-class splits inflating skip counts).
    rng = np.random.RandomState(seed)
    tr_idx: list[int] = []
    te_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        cut = max(1, int(len(cls_idx) * 0.7))
        tr_idx.extend(cls_idx[:cut].tolist())
        te_idx.extend(cls_idx[cut:].tolist())
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)
    X_tr, y_tr, X_te, y_te = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]

    import torch

    torch.manual_seed(seed)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
        print(f"  {name:<14} SKIP (single-class split; raise sample count or reseed)")
        return None

    from omni_mercury_engine.core.calibration import compute_ece

    fit_metrics = engine.fit_fusion(
        X_tr, y_tr, epochs=epochs, batch_size=64, early_stopping_patience=15
    )

    # Calibrated probabilities (default) and raw (calibrator bypassed) to show
    # temperature scaling improves ECE while preserving AUC exactly (#3).
    probs_cal = engine.score_fusion(X_te)
    calibrator = engine._fusion_calibrator
    engine._fusion_calibrator = None
    probs_raw = engine.score_fusion(X_te)
    engine._fusion_calibrator = calibrator

    auc = roc_auc_score(y_te, probs_cal)
    temp = fit_metrics.get("temperature", 1.0)
    ece_raw = compute_ece(y_te, probs_raw)
    ece_cal = compute_ece(y_te, probs_cal)
    print(
        f"  {name:<14} AUC={auc:.4f}  T={temp:.3f}  ECE {ece_raw:.4f}->{ece_cal:.4f}  "
        f"(n={len(X)}, anom={y.mean():.3%}, dim={X.shape[1]})"
    )
    return auc


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Raw fusion training-path benchmark (engine.fit_fusion on raw ADBench input)")
    print("-" * 72)
    aucs: list[float] = []
    for name in args.datasets:
        auc = _run_one(name, args.epochs, args.seed)
        if auc is not None and not np.isnan(auc):
            aucs.append(auc)

    print("-" * 72)
    if aucs:
        print(
            f"datasets measured: {len(aucs)}   "
            f"median AUC: {np.median(aucs):.4f}   mean AUC: {np.mean(aucs):.4f}"
        )
    else:
        print("No datasets measured (network unavailable?).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
