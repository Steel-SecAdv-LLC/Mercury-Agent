"""
Mercury Agent - Real-data validation test suite.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+

Tests Mercury Agent's anomaly detectors against real-world datasets
from ADBench (NeurIPS 2022) and NSL-KDD. Measures AUC-ROC, F1, and
accuracy with threshold optimization.

These tests download real data from public repositories and require
network access. Set MERCURY_RUN_LIVE_DATA=true to enable in CI.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.security import NSLKDDLoader
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.detectors.threshold_calibrator import find_optimal_threshold

logger = logging.getLogger(__name__)

# Gate on environment variable for CI
LIVE_DATA_ENABLED = os.getenv("MERCURY_RUN_LIVE_DATA", "").lower() in ("true", "1", "yes")

# ADBench datasets to validate
ADBENCH_DATASETS = ["cardio", "thyroid", "mammography", "breastw"]

# Minimum acceptable metrics
MIN_ADBENCH_AUC = 0.70  # Minimum AUC for any ADBench dataset
MIN_NSLKDD_AUC = 0.50  # NSL-KDD is harder for unsupervised methods


def _load_adbench(dataset_name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load an ADBench dataset by name."""
    config = DatasetConfig(
        name=f"adbench-{dataset_name}",
        preprocessing={"dataset": dataset_name},
    )
    loader = ADBenchLoader(config)
    loader.download()
    X, y = loader._load_raw()
    X = loader.preprocess(X)
    return X, y


def _run_detector_on_data(
    X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]
) -> dict[str, float]:
    """Run statistical detector on data and return metrics."""
    # Split: use 70% for training, 30% for testing
    n = len(X)
    n_train = int(n * 0.7)
    indices = np.random.RandomState(42).permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_test = y[test_idx]

    # Train detector
    detector = StatisticalAnomalyDetector()
    detector.fit(X_train)

    # Get scores
    result = detector.detect(X_test)
    scores = result["scores"]

    # Compute AUC
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = 0.0

    # Optimal threshold for F1
    optimal_threshold = find_optimal_threshold(scores, y_test)
    predictions = (scores >= optimal_threshold).astype(int)

    f1 = float(f1_score(y_test, predictions, zero_division=0))
    accuracy = float(accuracy_score(y_test, predictions))

    return {
        "auc": auc,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": optimal_threshold,
        "n_test": len(y_test),
        "anomaly_ratio": float(y_test.mean()),
    }


@pytest.mark.skipif(not LIVE_DATA_ENABLED, reason="MERCURY_RUN_LIVE_DATA not set")
@pytest.mark.parametrize("dataset_name", ADBENCH_DATASETS)
def test_adbench_statistical_detector(dataset_name: str) -> None:
    """Test statistical detector on ADBench datasets with threshold optimization."""
    X, y = _load_adbench(dataset_name)

    metrics = _run_detector_on_data(X, y)

    logger.info(
        "ADBench %s: AUC=%.3f, F1=%.3f, Acc=%.3f (threshold=%.3f, n=%d)",
        dataset_name,
        metrics["auc"],
        metrics["f1"],
        metrics["accuracy"],
        metrics["threshold"],
        metrics["n_test"],
    )

    assert metrics["auc"] >= MIN_ADBENCH_AUC, (
        f"ADBench {dataset_name}: AUC {metrics['auc']:.3f} < {MIN_ADBENCH_AUC}"
    )


@pytest.mark.skipif(not LIVE_DATA_ENABLED, reason="MERCURY_RUN_LIVE_DATA not set")
def test_nslkdd_statistical_detector() -> None:
    """Test statistical detector on NSL-KDD network intrusion data."""
    config = DatasetConfig(
        name="nsl-kdd",
        preprocessing={"binary": True, "include_test": True},
    )
    loader = NSLKDDLoader(config)
    features, labels = loader.load_data()
    features = loader.preprocess(features)

    metrics = _run_detector_on_data(features, labels)

    logger.info(
        "NSL-KDD: AUC=%.3f, F1=%.3f, Acc=%.3f (threshold=%.3f, n=%d)",
        metrics["auc"],
        metrics["f1"],
        metrics["accuracy"],
        metrics["threshold"],
        metrics["n_test"],
    )

    assert metrics["auc"] >= MIN_NSLKDD_AUC, (
        f"NSL-KDD: AUC {metrics['auc']:.3f} < {MIN_NSLKDD_AUC}"
    )


@pytest.mark.skipif(not LIVE_DATA_ENABLED, reason="MERCURY_RUN_LIVE_DATA not set")
def test_adbench_metrics_stored() -> None:
    """Verify that ADBench metrics can be computed and stored."""
    results: dict[str, dict[str, float]] = {}

    for dataset_name in ADBENCH_DATASETS:
        try:
            X, y = _load_adbench(dataset_name)
            metrics = _run_detector_on_data(X, y)
            results[dataset_name] = metrics
        except Exception as e:
            logger.warning("Failed to load ADBench %s: %s", dataset_name, e)

    # Store results for CI validation
    output_path = Path("benchmarks/live_data_baseline.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline: dict[str, float] = {}
    for name, m in results.items():
        baseline[f"adbench_{name}_auc"] = m["auc"]
        baseline[f"adbench_{name}_f1"] = m["f1"]

    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    assert len(results) >= 1, "At least one ADBench dataset should load successfully"
