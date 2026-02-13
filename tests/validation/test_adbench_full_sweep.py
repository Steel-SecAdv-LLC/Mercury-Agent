"""
Mercury Agent - Comprehensive ADBench benchmark across 16 datasets.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+

Expands ADBench coverage from 4 to 16 datasets using the ADBench
repository (NeurIPS 2022). Tests the statistical anomaly detector
with threshold optimization on each dataset.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.detectors.threshold_calibrator import find_optimal_threshold

logger = logging.getLogger(__name__)

LIVE_DATA_ENABLED = os.getenv("MERCURY_RUN_LIVE_DATA", "").lower() in ("true", "1", "yes")

# Primary test datasets (16 from ADBench repository)
PRIMARY_DATASETS = [
    "cardio",
    "thyroid",
    "mammography",
    "breastw",
    "Ionosphere",
    "Pima",
    "satellite",
    "shuttle",
    "wine",
    "glass",
    "musk",
    "arrhythmia",
    "optdigits",
    "pendigits",
    "vertebral",
    "WBC",
]

# Minimum AUC threshold (generous for diverse datasets)
MIN_AUC = 0.55


def _benchmark_dataset(dataset_name: str) -> dict[str, Any]:
    """Run benchmark on a single ADBench dataset."""
    config = DatasetConfig(
        name=f"adbench-{dataset_name}",
        preprocessing={"dataset": dataset_name},
    )
    loader = ADBenchLoader(config)
    loader.download()
    X, y = loader._load_raw()
    X = loader.preprocess(X)

    # Split: 70/30 with fixed seed
    n = len(X)
    n_train = int(n * 0.7)
    indices = np.random.RandomState(42).permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_test = y[test_idx]

    # Train and score
    detector = StatisticalAnomalyDetector()
    detector.fit(X_train)
    result = detector.detect(X_test)
    scores = result["scores"]

    # Metrics
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = 0.0

    threshold = find_optimal_threshold(scores, y_test)
    predictions = (scores >= threshold).astype(int)
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    accuracy = float(accuracy_score(y_test, predictions))

    return {
        "dataset": dataset_name,
        "detector": "statistical",
        "auc": auc,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": threshold,
        "n_samples": len(X_test),
        "n_features": int(X_test.shape[1]),
        "anomaly_ratio": float(y_test.mean()),
    }


@pytest.mark.skipif(not LIVE_DATA_ENABLED, reason="MERCURY_RUN_LIVE_DATA not set")
@pytest.mark.parametrize("dataset_name", PRIMARY_DATASETS)
def test_adbench_statistical_detector(dataset_name: str) -> None:
    """Test statistical anomaly detection on ADBench datasets."""
    result = _benchmark_dataset(dataset_name)

    logger.info(
        "ADBench %s: AUC=%.3f, F1=%.3f, Acc=%.3f (threshold=%.3f, n=%d, d=%d)",
        result["dataset"],
        result["auc"],
        result["f1"],
        result["accuracy"],
        result["threshold"],
        result["n_samples"],
        result["n_features"],
    )

    assert result["auc"] >= MIN_AUC, (
        f"AUC {result['auc']:.3f} < {MIN_AUC} for {dataset_name}"
    )


@pytest.mark.skipif(not LIVE_DATA_ENABLED, reason="MERCURY_RUN_LIVE_DATA not set")
def test_adbench_comprehensive_report() -> None:
    """Generate comprehensive benchmark report across all 16 datasets."""
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for dataset_name in PRIMARY_DATASETS:
        try:
            result = _benchmark_dataset(dataset_name)
            results.append(result)
        except Exception as e:
            logger.warning("Failed to benchmark %s: %s", dataset_name, e)
            failures.append(f"{dataset_name}: {e}")

    # Compute summary statistics
    if results:
        aucs = [r["auc"] for r in results]
        f1s = [r["f1"] for r in results]
        above_target = sum(1 for a in aucs if a >= 0.70)

        from datetime import datetime

        report = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "datasets_tested": len(results),
            "datasets_failed": len(failures),
            "detectors": ["statistical"],
            "summary": {
                "mean_auc": float(np.mean(aucs)),
                "median_auc": float(np.median(aucs)),
                "mean_f1": float(np.mean(f1s)),
                "datasets_above_target": above_target,
                "target_auc": 0.70,
            },
            "results": results,
            "failures": failures,
        }

        # Save report
        output_path = Path("benchmarks/adbench_comprehensive_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            "ADBench sweep: %d datasets, mean AUC=%.3f, mean F1=%.3f, %d above target",
            len(results),
            report["summary"]["mean_auc"],
            report["summary"]["mean_f1"],
            above_target,
        )

    assert len(results) >= len(PRIMARY_DATASETS) // 2, (
        f"Need at least {len(PRIMARY_DATASETS) // 2} successful benchmarks, got {len(results)}"
    )
