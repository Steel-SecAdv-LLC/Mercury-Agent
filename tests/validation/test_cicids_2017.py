"""
Mercury Agent - CICIDS-2017 network intrusion detection benchmark.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+

Tests Mercury Agent detectors against the CICIDS-2017 dataset,
a modern network intrusion detection dataset with real network traffic.
Complements NSL-KDD with more recent attack patterns.

Reference: Sharafaldin I, Lashkari AH, Ghorbani AA.
    Toward Generating a New Intrusion Detection Dataset and Intrusion
    Traffic Characterization. ICISSP 2018.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.security import CICIDSLoader
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.detectors.threshold_calibrator import find_optimal_threshold

logger = logging.getLogger(__name__)

LIVE_DATA_ENABLED = os.getenv("MERCURY_RUN_LIVE_DATA", "").lower() in ("true", "1", "yes")

# Minimum acceptable metrics for CICIDS-2017
MIN_CICIDS_AUC = 0.55  # Network data is challenging for unsupervised methods


def _load_cicids() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Load CICIDS-2017 dataset."""
    config = DatasetConfig(
        name="cicids",
        preprocessing={"binary": True, "source": "huggingface"},
    )
    loader = CICIDSLoader(config)
    features, labels = loader.load_data()
    features = loader.preprocess(features)
    return features, labels


@pytest.mark.skipif(not LIVE_DATA_ENABLED, reason="MERCURY_RUN_LIVE_DATA not set")
def test_cicids_2017_statistical() -> None:
    """CICIDS-2017 with statistical detector."""
    features, labels = _load_cicids()

    # Use subset for efficiency (full dataset is 2.8M records)
    n = min(len(features), 50000)
    indices = np.random.RandomState(42).permutation(len(features))[:n]
    X, y = features[indices], labels[indices]

    # Split 70/30
    n_train = int(n * 0.7)
    X_train, X_test = X[:n_train], X[n_train:]
    y_test = y[n_train:]

    # Train and detect
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

    logger.info(
        "CICIDS-2017 (Statistical): AUC=%.3f, F1=%.3f, Acc=%.3f (threshold=%.3f, n=%d)",
        auc,
        f1,
        accuracy,
        threshold,
        len(y_test),
    )

    assert auc >= MIN_CICIDS_AUC, f"CICIDS AUC {auc:.3f} < {MIN_CICIDS_AUC}"


@pytest.mark.skipif(not LIVE_DATA_ENABLED, reason="MERCURY_RUN_LIVE_DATA not set")
def test_cicids_2017_auto_calibrated() -> None:
    """CICIDS-2017 with auto-calibrated statistical detector."""
    features, labels = _load_cicids()

    # Use subset
    n = min(len(features), 50000)
    indices = np.random.RandomState(42).permutation(len(features))[:n]
    X, y = features[indices], labels[indices]

    # Split 70/30
    n_train = int(n * 0.7)
    X_train, X_test = X[:n_train], X[n_train:]
    y_test = y[n_train:]

    # Train with auto-calibration enabled
    detector = StatisticalAnomalyDetector(config={"auto_calibrate": True})
    detector.fit(X_train)
    result = detector.detect(X_test)
    scores = result["scores"]

    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = 0.0

    # With auto-calibration, F1 should be better
    predictions = result["is_anomaly"].astype(int)
    f1 = float(f1_score(y_test, predictions, zero_division=0))

    logger.info(
        "CICIDS-2017 (Auto-Calibrated): AUC=%.3f, F1=%.3f (n=%d)",
        auc,
        f1,
        len(y_test),
    )

    assert auc >= MIN_CICIDS_AUC, f"CICIDS (auto-cal) AUC {auc:.3f} < {MIN_CICIDS_AUC}"
