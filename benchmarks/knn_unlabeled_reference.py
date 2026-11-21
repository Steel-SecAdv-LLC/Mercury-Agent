# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the kNN stream's fusion weight when the reference is unlabelled.

``MercuryAnomalyDetector`` fuses a kNN density stream into its ensemble at a
weight set from the feature width (up to 0.85 at low width). That weight was
measured the way ``mercury_benchmark.py`` measures everything: the detector is
fitted on rows known to be normal. This script measures the other way the
detector is used: fitted on, and scoring, the same unlabelled rows (the CLI and
API auto-fit, the MCP tool, the governed-fusion suite).

For each ADBench set (capped at 6,000 rows, class-stratified, standardised) the
detector is fitted on all rows and scores them, at kNN weights 0, 0.25, 0.5 and
the dimensional weight. The result backs
``MercuryAnomalyDetector._KNN_WEIGHT_UNLABELED_CAP`` and is pinned by
``tests/detectors/test_knn_unlabeled_reference.py``.

Usage::

    python benchmarks/knn_unlabeled_reference.py

Output: ``benchmarks/results/knn_unlabeled_reference.json``. Needs the ADBench
files (downloaded and cached by the dataset loader).
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "results" / "knn_unlabeled_reference.json"

#: kNN fusion weights measured; ``None`` is the dimensional weight.
WEIGHTS: tuple[float | None, ...] = (0.0, 0.25, 0.5, None)
ROW_CAP = 6000


def measure() -> dict[str, Any]:
    """Return per-dataset AUROC at each weight and the summary over datasets."""
    sys.path.insert(0, str(_HERE))
    import mercury_benchmark as mb

    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
    from omni_mercury_engine.ml.mercury_ml import StandardScaler, roc_auc_score

    per_dataset: dict[str, Any] = {}
    for entry in mb._load_adbench():
        if "error" in entry:
            continue
        features, labels = entry["X"], entry["y"]
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        features, labels = mb._cap_stratified(features, labels, ROW_CAP)
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
        features = StandardScaler().fit_transform(features)
        aucs = []
        for weight in WEIGHTS:
            config = None if weight is None else {"knn_stream_weight": weight}
            detector = MercuryAnomalyDetector(config=config).fit(features)
            scores = np.asarray(detector.detect(features)["scores"]).reshape(-1)
            aucs.append(round(float(roc_auc_score(labels, scores)), 6))
        per_dataset[entry["name"]] = {
            "n": len(labels),
            "n_features": int(features.shape[1]),
            "contamination": round(float(labels.mean()), 6),
            "auroc": aucs,
        }
        print(f"{entry['name']:12s} " + " ".join(f"{auc:.4f}" for auc in aucs), flush=True)

    table = np.array([row["auroc"] for row in per_dataset.values()])
    regret = table.max(axis=1, keepdims=True) - table
    names = ["0", "0.25", "0.5", "dimensional"]
    return {
        "protocol": "fit on and score the same unlabelled rows; ADBench, cap 6000, standardised",
        "weights": names,
        "n_datasets": len(per_dataset),
        "mean_auroc": dict(zip(names, np.round(table.mean(axis=0), 4).tolist(), strict=True)),
        "median_auroc": dict(
            zip(names, np.round(np.median(table, axis=0), 4).tolist(), strict=True)
        ),
        "mean_regret": dict(zip(names, np.round(regret.mean(axis=0), 4).tolist(), strict=True)),
        "max_regret": dict(zip(names, np.round(regret.max(axis=0), 4).tolist(), strict=True)),
        "below_chance": dict(zip(names, (table < 0.5).sum(axis=0).tolist(), strict=True)),
        "per_dataset": per_dataset,
    }


def main() -> int:
    """Measure and write the artifact."""
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    result = measure()
    _OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key in ("mean_auroc", "median_auroc", "mean_regret", "max_regret", "below_chance"):
        print(key, result[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
