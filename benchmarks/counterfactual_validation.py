# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Validate detection counterfactuals on real labeled detections, all methods.

Protocol: fit :class:`MercuryAnomalyDetector` on a real ADBench tabular
anomaly dataset, take its TRUE-POSITIVE detections (flagged AND labeled
anomalous — real detections of real anomalies), and generate a
counterfactual for each with EVERY search method. Correctness and
minimality are the module's own re-scored guarantees; this benchmark
measures how often each method produces a verified flip and how sparse /
close the result is.

Usage::

    PYTHONPATH=src python benchmarks/counterfactual_validation.py \
        --out benchmarks/counterfactual_validation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_HERE.parent / "src"))

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.explainability.detection_counterfactuals import (
    explain_detection_counterfactual,
    make_statistical_score_fn,
)

logger = logging.getLogger("counterfactual_validation")

#: Pre-registered: fixed before any results were observed. WBC is a small
#: real medical dataset (378x30) so the batch re-scoring stays tractable
#: across five methods.
DATASET = "WBC"
METHODS = ("wachter", "dice", "growing_spheres", "prototype", "genetic")

#: Bounded per-method search budgets: every candidate evaluation is a full
#: batch re-score through the real detector, so unbounded defaults are
#: minutes-per-explanation. These budgets are part of the pre-registered
#: protocol and recorded in the results.
METHOD_BUDGETS: dict[str, dict[str, Any]] = {
    "wachter": {"max_iterations": 120},
    "dice": {"max_iterations": 60},
    "growing_spheres": {"n_samples": 200, "step_size": 0.25, "max_iterations": 40},
    "prototype": {},
    "genetic": {"population_size": 40, "max_generations": 30},
}
MAX_EXPLANATIONS_PER_METHOD = 8
SEED = 0


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - git-less environments
        return "unknown"


def _load_dataset() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load the real labeled dataset (network on first use, NPZ-cached after)."""
    config = DatasetConfig(name=f"adbench-{DATASET}", preprocessing={"dataset": DATASET})
    loader = ADBenchLoader(config)
    loader.download()
    X, y = loader._load_raw()
    return np.asarray(X, dtype=np.float64), (np.asarray(y) > 0).astype(int)


def main() -> int:
    """Run the validation and write the committed results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Results JSON path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    X, y = _load_dataset()
    detector = MercuryAnomalyDetector()
    detector.fit(X[y == 0])
    detection = detector.detect(X)
    scores = np.asarray(detection["scores"], dtype=np.float64)
    flags = np.asarray(detection["is_anomaly"]).astype(bool)
    threshold = float(detection["threshold"])

    true_positives = np.flatnonzero(flags & (y == 1))
    if true_positives.size == 0:
        raise RuntimeError("no true-positive detections; validation is impossible")
    # Strongest detections first — the hardest points to flip.
    true_positives = true_positives[np.argsort(-scores[true_positives])]
    chosen = true_positives[:MAX_EXPLANATIONS_PER_METHOD]
    logger.info(
        "dataset %s: %d samples, %d TP detections, explaining %d per method",
        DATASET,
        X.shape[0],
        true_positives.size,
        chosen.size,
    )

    per_method: dict[str, Any] = {}
    for method in METHODS:
        flips = 0
        minimal = 0
        sparsities: list[int] = []
        distances: list[float] = []
        t0 = time.perf_counter()
        for idx in chosen:
            score_fn = make_statistical_score_fn(detector, X, int(idx))
            kwargs: dict[str, Any] = dict(METHOD_BUDGETS.get(method, {}))
            if method == "prototype":
                kwargs.update({"training_data": X, "training_labels": y})
            cf = explain_detection_counterfactual(
                score_fn,
                X[int(idx)],
                threshold=threshold,
                method=method,
                seed=SEED,
                n_restarts=2,
                max_pair_evals=60,
                **kwargs,
            )
            if cf.flipped:
                flips += 1
                sparsities.append(len(cf.changed_features))
                distances.append(float(cf.distance))
                if cf.minimal:
                    minimal += 1
        wall = time.perf_counter() - t0
        per_method[method] = {
            "explanations": int(chosen.size),
            "flip_rate": flips / chosen.size,
            "minimality_verified_rate": (minimal / flips) if flips else 0.0,
            "mean_sparsity": float(np.mean(sparsities)) if sparsities else None,
            "mean_distance": float(np.mean(distances)) if distances else None,
            "wall_seconds": round(wall, 2),
        }
        logger.info("%s: %s", method, per_method[method])

    results = {
        "protocol": (
            "fit MercuryAnomalyDetector on the normal split of a real ADBench "
            "dataset; explain its strongest true-positive detections with every "
            "method; flip/minimality are re-scored through the real detect() path"
        ),
        "dataset": DATASET,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_true_positive_detections": int(true_positives.size),
        "explanations_per_method": int(chosen.size),
        "seed": SEED,
        "threshold": threshold,
        "method_budgets": METHOD_BUDGETS,
        "n_restarts": 2,
        "per_method": per_method,
        "provenance": {"commit": _git_commit()},
    }
    out = args.out or (_HERE / "counterfactual_validation_results.json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("results written: %s", out)

    weak = [m for m, r in per_method.items() if r["flip_rate"] < 0.9]
    if weak:
        logger.warning("methods under the 0.9 flip-rate bar: %s", weak)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
