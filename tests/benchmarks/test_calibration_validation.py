"""
Mercury Agent - Tests for Calibration Validation Harness
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Tests for benchmarks/calibration_validation.py covering all three
math debt validations (MD-011, MD-003, MD-005) on synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ and benchmarks/ are on the path
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "benchmarks"))
sys.path.insert(0, str(_project_root))

sklearn = pytest.importorskip("sklearn")

from calibration_validation import (  # type: ignore[import-not-found]
    run_calibration_validation,
    run_conformal_coverage,
    run_fusion_weight_analysis,
)


def _make_synthetic_data(
    n_samples: int = 200,
    n_features: int = 5,
    anomaly_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic train/cal/test data with separable anomalies."""
    rng = np.random.RandomState(seed)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Normal data: centred at origin
    X_normal = rng.randn(n_normal, n_features) * 1.0
    # Anomalies: shifted mean + higher variance
    X_anomaly = rng.randn(n_anomalies, n_features) * 2.0 + 3.0

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)]).astype(int)

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split: 60% train, 20% cal, 20% test
    n_train = int(n_samples * 0.6)
    n_cal = int(n_samples * 0.2)

    X_train, y_train = X[:n_train], y[:n_train]
    X_cal = X[n_train : n_train + n_cal]
    X_test, y_test = X[n_train + n_cal :], y[n_train + n_cal :]

    return X_train, y_train, X_cal, X_test, y_test


class TestRunCalibrationValidation:
    """Tests for run_calibration_validation (MD-011)."""

    def test_returns_expected_keys(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_calibration_validation("synthetic", X_train, y_train, X_test, y_test)

        expected_keys = {
            "calibrated_threshold",
            "calibration_method",
            "adaptive_weights",
            "weight_source",
            "calibrated_f1",
            "uncalibrated_f1",
            "calibrated_precision",
            "calibrated_recall",
            "delta_f1",
            "auc",
        }
        assert expected_keys.issubset(
            result.keys()
        ), f"Missing keys: {expected_keys - result.keys()}"

    def test_f1_values_in_range(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_calibration_validation("synthetic", X_train, y_train, X_test, y_test)

        assert 0.0 <= result["calibrated_f1"] <= 1.0
        assert 0.0 <= result["uncalibrated_f1"] <= 1.0
        assert 0.0 <= result["calibrated_precision"] <= 1.0
        assert 0.0 <= result["calibrated_recall"] <= 1.0

    def test_delta_f1_consistency(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_calibration_validation("synthetic", X_train, y_train, X_test, y_test)

        expected_delta = result["calibrated_f1"] - result["uncalibrated_f1"]
        assert abs(result["delta_f1"] - expected_delta) < 1e-10

    def test_threshold_is_positive(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_calibration_validation("synthetic", X_train, y_train, X_test, y_test)

        assert isinstance(result["calibrated_threshold"], float)
        assert 0.0 <= result["calibrated_threshold"] <= 1.0

    def test_auc_is_valid(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_calibration_validation("synthetic", X_train, y_train, X_test, y_test)

        assert 0.0 <= result["auc"] <= 1.0 or np.isnan(result["auc"])


class TestRunConformalCoverage:
    """Tests for run_conformal_coverage (MD-005)."""

    def test_returns_coverage_results(self) -> None:
        X_train, y_train, X_cal, X_test, y_test = _make_synthetic_data()
        result = run_conformal_coverage("synthetic", X_train, y_train, X_cal, X_test, y_test)

        assert "coverage_results" in result
        assert len(result["coverage_results"]) == 3  # 90%, 95%, 99%

    def test_coverage_values_in_range(self) -> None:
        X_train, y_train, X_cal, X_test, y_test = _make_synthetic_data()
        result = run_conformal_coverage("synthetic", X_train, y_train, X_cal, X_test, y_test)

        for cov in result["coverage_results"]:
            if "error" not in cov:
                assert 0.0 <= cov["empirical_coverage"] <= 1.0
                assert cov["target_coverage"] in [0.90, 0.95, 0.99]
                assert cov["coverage_gap"] >= 0.0

    def test_class_coverage_present(self) -> None:
        X_train, y_train, X_cal, X_test, y_test = _make_synthetic_data()
        result = run_conformal_coverage("synthetic", X_train, y_train, X_cal, X_test, y_test)

        for cov in result["coverage_results"]:
            if "error" not in cov:
                assert "class_coverage" in cov
                # Both classes should be represented
                assert 0 in cov["class_coverage"] or 1 in cov["class_coverage"]


class TestRunFusionWeightAnalysis:
    """Tests for run_fusion_weight_analysis (MD-003)."""

    def test_returns_expected_keys(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_fusion_weight_analysis("synthetic", X_train, y_train, X_test, y_test)

        expected_keys = {
            "strategy_used",
            "adaptive_weights",
            "weight_source",
            "f1_at_learned_weights",
            "notes",
        }
        assert expected_keys.issubset(
            result.keys()
        ), f"Missing keys: {expected_keys - result.keys()}"

    def test_adaptive_weights_valid(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_fusion_weight_analysis("synthetic", X_train, y_train, X_test, y_test)

        if result["adaptive_weights"] is not None:
            weights = result["adaptive_weights"]
            assert len(weights) == 3
            assert all(0.0 <= w <= 1.0 for w in weights)
            assert abs(sum(weights) - 1.0) < 1e-6

    def test_f1_in_range(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_fusion_weight_analysis("synthetic", X_train, y_train, X_test, y_test)

        f1 = result["f1_at_learned_weights"]
        assert 0.0 <= f1 <= 1.0 or np.isnan(f1)

    def test_strategy_is_documented(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_fusion_weight_analysis("synthetic", X_train, y_train, X_test, y_test)

        assert result["strategy_used"] in {
            "neurosymbolic_hub",
            "statistical_adaptive_weights",
        }
