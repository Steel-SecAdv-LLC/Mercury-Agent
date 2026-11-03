# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Calibration Validation Harness.

Tests for benchmarks/calibration_validation.py covering all three
math debt validations (MD-011, MD-003, MD-005) on synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# Ensure src/ and benchmarks/ are on the path
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "benchmarks"))
sys.path.insert(0, str(_project_root))

import calibration_validation as cv_module
from calibration_validation import (
    _best_f1_threshold,
    _f1_from_weights,
    _heldout_scores,
    run_calibration_validation,
    run_conformal_coverage,
    run_fusion_weight_analysis,
    run_fusion_weight_cv,
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


class TestFusionWeightCrossValidation:
    """MD-003 cross-validation scores every fold's weights on the rows it held out.

    The loop used to split each fold into training and held-out rows, fit the
    L-BFGS-B weights on the training rows, and never touch the held-out rows
    again: it averaged fold optima without validating any of them (flake8
    7.5.0 reported the two unused variables). Each fold now scores its
    optimum, the default weights and its own adaptive weights on its held-out
    rows at a cut chosen on its training rows, and flags an optimum the
    optimiser's box decided.
    """

    def test_every_fold_is_scored_on_its_held_out_rows(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_fusion_weight_cv("synthetic", X_train, y_train, X_test, y_test, n_splits=3)
        held = result["cv_heldout"]
        assert held["n_folds"] == 3 and len(held["folds"]) == 3
        # The held-out rows of the three folds partition the training rows.
        assert sum(f["n_val"] for f in held["folds"]) == len(X_train)
        assert sum(f["n_val_anomalies"] for f in held["folds"]) == int(y_train.sum())
        for fold in held["folds"]:
            for scheme in ("optimal", "default", "adaptive"):
                score = fold[scheme]
                assert 0.0 <= score["f1"] <= 1.0
                assert np.isfinite(score["bce"]) and score["bce"] > 0.0
        for scheme in ("optimal", "default", "adaptive"):
            mean = np.mean([f[scheme]["f1"] for f in held["folds"]])
            assert held[f"f1_{scheme}_mean"] == pytest.approx(mean)
        assert held["delta_f1_optimal_vs_default"] == pytest.approx(
            held["f1_optimal_mean"] - held["f1_default_mean"]
        )

    def test_the_reported_optimum_is_still_the_mean_of_the_fold_optima(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        result = run_fusion_weight_cv("synthetic", X_train, y_train, X_test, y_test)
        folds = np.array([f["optimal_weights"] for f in result["cv_heldout"]["folds"]])
        mean = folds.mean(axis=0)
        assert result["mean_optimal_weights"] == pytest.approx((mean / mean.sum()).tolist())
        assert result["std_optimal_weights"] == pytest.approx(folds.std(axis=0).tolist())

    def test_bound_pinned_optima_are_counted(self) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        held = run_fusion_weight_cv("synthetic", X_train, y_train, X_test, y_test)["cv_heldout"]
        flagged = sum(1 for f in held["folds"] if f["optimum_on_bounds"])
        assert held["folds_with_optimum_on_bounds"] == flagged
        for fold in held["folds"]:
            if fold["optimum_on_bounds"]:
                # Normalised from raw weights that all sit on 0.05 or 0.95.
                w = np.array(fold["optimal_weights"])
                assert min(w) / max(w) == pytest.approx(0.05 / 0.95, rel=1e-4) or np.allclose(
                    w, w[0]
                )

    def test_fold_adaptive_weights_never_see_held_out_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        X_train, y_train, _, X_test, y_test = _make_synthetic_data()
        seen: list[int] = []
        original: Callable[..., np.ndarray] = (
            cv_module.MercuryAnomalyDetector._compute_adaptive_weights
        )

        def recording(
            self: object,
            X: np.ndarray,
            labels: np.ndarray,
            *,
            components: np.ndarray | None = None,
        ) -> np.ndarray:
            seen.append(len(labels))
            return original(self, X, labels, components=components)

        monkeypatch.setattr(
            cv_module.MercuryAnomalyDetector, "_compute_adaptive_weights", recording
        )
        result = run_fusion_weight_cv("synthetic", X_train, y_train, X_test, y_test, n_splits=3)
        fold_train_sizes = [len(X_train) - f["n_val"] for f in result["cv_heldout"]["folds"]]
        # One whole-training call (the reported adaptive weights), then one per
        # fold on that fold's training rows only.
        assert seen[0] == len(X_train)
        assert seen[1:] == fold_train_sizes

    def test_the_held_out_cut_comes_from_the_training_rows(self) -> None:
        w = np.array([1.0, 0.0, 0.0])
        # A row is flagged when its fused score is strictly above the cut. On
        # the training rows the F1-best cut is the highest normal, 0.2. The
        # held-out normals sit at 0.3 and 0.65, above that cut, so a cut taken
        # from the training rows flags them: 2 true and 2 false positives.
        sm_train = np.array([[0.1, 0, 0], [0.2, 0, 0], [0.6, 0, 0], [0.9, 0, 0]])
        y_tr = np.array([0, 0, 1, 1])
        sm_val = np.array([[0.3, 0, 0], [0.65, 0, 0], [0.8, 0, 0], [0.95, 0, 0]])
        y_va = np.array([0, 0, 1, 1])
        cut = _best_f1_threshold(w, sm_train, y_tr)
        assert cut == pytest.approx(0.2)
        held = _heldout_scores(w, sm_train, y_tr, sm_val, y_va)
        assert held["threshold"] == pytest.approx(cut)
        assert held["f1"] == pytest.approx(2 / 3)
        # The oracle cut, chosen on the held-out rows themselves, separates
        # them perfectly: the figure the old procedure would have reported.
        assert _f1_from_weights(w, sm_val, y_va) == pytest.approx(1.0)

    def test_the_oracle_figure_is_unchanged(self) -> None:
        rng = np.random.RandomState(7)
        sm = rng.rand(80, 3)
        y = (sm[:, 0] + 0.3 * rng.rand(80) > 0.7).astype(int)
        w = np.array([0.4, 0.3, 0.3])
        fused = np.clip(sm @ (w / w.sum()), 0.0, 1.0)
        from omni_mercury_engine.ml.mercury_ml import f1_score

        brute = max(
            float(f1_score(y, (fused > t).astype(int), zero_division=0)) for t in np.unique(fused)
        )
        assert _f1_from_weights(w, sm, y) == pytest.approx(brute)
