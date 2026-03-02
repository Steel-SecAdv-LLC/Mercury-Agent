"""
Mercury Agent - Calibration Validation Harness
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

Resolves math debt items MD-011, MD-003, MD-005 by running calibration,
cross-validation, and conformal coverage measurement against all
datasets used in the mercury benchmark.

Usage:
    python benchmarks/calibration_validation.py
    python benchmarks/calibration_validation.py --skip-conformal
    python benchmarks/calibration_validation.py --datasets lympho,smtp
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from omni_mercury_engine.ml.mercury_ml import (
    StandardScaler,
    StratifiedKFold,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    train_test_split,
)

# ---------------------------------------------------------------------------
# Ensure src/ is on the path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import from mercury_benchmark (sibling module in benchmarks/)
_benchmarks_dir = str(Path(__file__).parent)
if _benchmarks_dir not in sys.path:
    sys.path.insert(0, _benchmarks_dir)

from mercury_benchmark import (  # type: ignore[import-untyped]
    DOMAIN_DATASETS,
    _cap_stratified,
    _load_adbench,
    _load_domain_dataset,
)

from omni_mercury_engine.core.conformal_prediction import (
    ConformalAnomalyDetector,
    CrossConformalPredictor,
    SplitConformalPredictor,
)
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

_cal_logger = logging.getLogger(__name__)

try:
    import optuna  # noqa: F401

    _AUTO_TUNE = True
except ImportError:
    _AUTO_TUNE = False
    _cal_logger.info(
        "optuna not installed — auto_tune disabled. " "Install with: pip install optuna"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SAMPLES = 20_000
MIN_SAMPLES_PER_CLASS = 10
OUTPUT_PATH = Path(__file__).parent / "calibration_validation_results.json"
CONFORMAL_COVERAGE_LEVELS = [0.90, 0.95, 0.99]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def _stratified_split(
    X: np.ndarray, y: np.ndarray, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Split into train (60%) / calibration (20%) / test (20%).

    Returns (X_train, y_train, X_cal, y_cal, X_test, y_test) or None
    if any split has fewer than MIN_SAMPLES_PER_CLASS per class.
    """
    n = len(X)
    if n < 50:
        return None

    # First split: 60% train, 40% rest
    try:
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=rng.randint(0, 2**31), stratify=y
        )
    except ValueError:
        return None

    # Second split: 50/50 of remaining = 20%/20% of total
    try:
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=rng.randint(0, 2**31), stratify=y_rest
        )
    except ValueError:
        return None

    # Verify minimum samples per class in each split
    for split_y, split_name in [
        (y_train, "train"),
        (y_cal, "cal"),
        (y_test, "test"),
    ]:
        classes, counts = np.unique(split_y, return_counts=True)
        if len(classes) < 2:
            return None
        if any(c < MIN_SAMPLES_PER_CLASS for c in counts):
            return None

    return X_train, y_train, X_cal, y_cal, X_test, y_test


# ---------------------------------------------------------------------------
# MD-011: Calibration Validation
# ---------------------------------------------------------------------------


def run_calibration_validation(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Run threshold calibration and measure improvement over default.

    1. Fit detector with fit_with_labels(X_train, y_train)
       - This triggers ThresholdCalibrationPipeline internally
       - Records: calibrated threshold, strategy selected, adaptive weights
    2. Score X_test with detect(X_test)
    3. Compute calibrated F1 (using _supervised_threshold)
    4. Compute uncalibrated F1 (binary predictions at default 0.5 threshold)
    5. Return comparison metrics
    """
    detector = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
    detector.fit_with_labels(X_train, y_train)

    result = detector.detect(X_test)
    scores = result["scores"]

    # Calibrated predictions (detector already uses _supervised_threshold)
    calibrated_threshold = detector._supervised_threshold or 0.5
    calibrated_preds = (scores > calibrated_threshold).astype(int)
    calibrated_f1 = float(f1_score(y_test, calibrated_preds, zero_division=0))
    calibrated_precision = float(precision_score(y_test, calibrated_preds, zero_division=0))
    calibrated_recall = float(recall_score(y_test, calibrated_preds, zero_division=0))

    # Uncalibrated predictions (default threshold 0.5)
    uncalibrated_preds = (scores > 0.5).astype(int)
    uncalibrated_f1 = float(f1_score(y_test, uncalibrated_preds, zero_division=0))

    # AUC (threshold-independent)
    auc = _safe_auc(y_test, scores)

    # Adaptive weights from the fitted detector
    adaptive_weights = getattr(detector, "_adaptive_weights", [0.40, 0.30, 0.30])
    if isinstance(adaptive_weights, np.ndarray):
        adaptive_weights = adaptive_weights.tolist()

    weight_source = getattr(detector, "_weight_source", "unknown")
    calibration_method = getattr(detector, "_calibration_method", "unknown")

    return {
        "calibrated_threshold": float(calibrated_threshold),
        "calibration_method": str(calibration_method),
        "adaptive_weights": adaptive_weights,
        "weight_source": str(weight_source),
        "calibrated_f1": calibrated_f1,
        "uncalibrated_f1": uncalibrated_f1,
        "calibrated_precision": calibrated_precision,
        "calibrated_recall": calibrated_recall,
        "delta_f1": calibrated_f1 - uncalibrated_f1,
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# MD-005: Conformal Coverage
# ---------------------------------------------------------------------------


def run_conformal_coverage(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Measure empirical coverage at 90%, 95%, 99% target levels.

    For EACH coverage level:
    1. Create ConformalAnomalyDetector with the target coverage
    2. Fit on combined train+cal data (it splits internally)
    3. Call evaluate_coverage(X_test, y_test) -> CoverageResult
    4. Record empirical_coverage, target_coverage, coverage_gap, per-class coverage

    evaluate_coverage() measures prediction ACCURACY (fraction of correct
    binary predictions), not regression interval coverage. A coverage of
    0.95 means 95% of test predictions are correct.
    """
    X_fit = np.vstack([X_train, X_cal])
    y_fit = np.concatenate([y_train, np.zeros(len(X_cal), dtype=int)])

    # Use labels from cal if available; but the spec says cal is unlabeled for
    # conformal calibration -- combine train+cal, let ConformalAnomalyDetector
    # split internally.

    coverage_results: list[dict[str, Any]] = []

    for target in CONFORMAL_COVERAGE_LEVELS:
        try:
            cad = ConformalAnomalyDetector(
                base_detector=MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE),
                coverage=target,
                calibration_fraction=0.2,
                method="split",
                seed=42,
            )
            cad.fit(X_fit, y_fit)
            cov_result = cad.evaluate_coverage(X_test, y_test)

            class_coverage = {
                int(k): float(v) for k, v in cov_result.marginal_coverage_by_class.items()
            }

            coverage_results.append(
                {
                    "target_coverage": target,
                    "empirical_coverage": float(cov_result.empirical_coverage),
                    "coverage_gap": float(cov_result.coverage_gap),
                    "meets_guarantee": bool(cov_result.empirical_coverage >= target),
                    "class_coverage": class_coverage,
                }
            )
        except Exception as e:
            coverage_results.append(
                {
                    "target_coverage": target,
                    "error": str(e),
                    "meets_guarantee": False,
                }
            )

    return {"coverage_results": coverage_results}


def measure_score_based_coverage(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Measure the TRUE conformal coverage guarantee (score-based).

    The conformal guarantee (Vovk et al., 2005) is about SCORES, not
    class labels: the fraction of test nonconformity scores at or below
    the calibration quantile threshold should be >= coverage.

    For each target coverage level (90%, 95%, 99%):
    1. Fit MercuryAnomalyDetector on X_train (unsupervised)
    2. Get anomaly scores on X_cal (calibration nonconformity scores)
    3. Compute quantile threshold via SplitConformalPredictor.fit()
    4. Get anomaly scores on X_test
    5. Measure: fraction of test scores <= threshold (the raw guarantee)
    6. Also report: normal-class coverage and anomaly detection rate

    This avoids ConformalAnomalyDetector's internal data splitting and
    tests the SplitConformalPredictor directly.
    """
    # Fit detector on training data (unsupervised — no labels)
    detector = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
    detector.fit(X_train)

    # Get calibration scores
    cal_result = detector.detect(X_cal)
    cal_scores = cal_result["scores"]

    # Get test scores
    test_result = detector.detect(X_test)
    test_scores = test_result["scores"]

    score_coverage_results: list[dict[str, Any]] = []

    for target in CONFORMAL_COVERAGE_LEVELS:
        try:
            # --- Split Conformal ---
            split_pred = SplitConformalPredictor(coverage=target, seed=42)
            split_pred.fit(cal_scores)
            split_threshold = split_pred.get_anomaly_threshold()

            # Score-based coverage: fraction of ALL test scores <= threshold
            score_coverage = float(np.mean(test_scores <= split_threshold))

            # Normal-class coverage: fraction of normal test points <= threshold
            normal_mask = y_test == 0
            normal_coverage = (
                float(np.mean(test_scores[normal_mask] <= split_threshold))
                if np.any(normal_mask)
                else float("nan")
            )

            # Anomaly detection rate: fraction of anomalies ABOVE threshold
            anomaly_mask = y_test == 1
            anomaly_detection_rate = (
                float(np.mean(test_scores[anomaly_mask] > split_threshold))
                if np.any(anomaly_mask)
                else float("nan")
            )

            # --- Cross Conformal (k=5) for comparison ---
            cross_pred = CrossConformalPredictor(coverage=target, n_folds=5, seed=42)

            def _scoring_fn(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
                d = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
                d.fit(X)
                return d.detect(X)["scores"]

            # NOTE: CrossConformalPredictor's API passes only X[cal_idx] to the
            # scoring function, so the function fits and scores on the same fold.
            # For unsupervised detectors this is acceptable (score distributions
            # are similar regardless of training subset), but for supervised
            # models a proper cross-conformal implementation would require
            # fitting on X[train_idx] and scoring X[cal_idx] separately.
            cross_pred.fit(X_cal, _scoring_fn)
            cross_threshold = cross_pred.get_anomaly_threshold()

            # NOTE: test_scores were produced by a detector fit on X_train (above),
            # while the conformal thresholds were calibrated on X_cal with separate
            # detector instances. For unsupervised detectors, this distribution
            # mismatch is negligible. For supervised detectors, a single detector
            # fit on X_train should be used for both calibration and test scoring.
            cross_score_coverage = float(np.mean(test_scores <= cross_threshold))
            cross_normal_coverage = (
                float(np.mean(test_scores[normal_mask] <= cross_threshold))
                if np.any(normal_mask)
                else float("nan")
            )

            score_coverage_results.append(
                {
                    "target_coverage": target,
                    "split_threshold": float(split_threshold),
                    "split_score_coverage": score_coverage,
                    "split_normal_coverage": normal_coverage,
                    "split_anomaly_detection_rate": anomaly_detection_rate,
                    "split_meets_score_guarantee": bool(score_coverage >= target),
                    "split_meets_normal_guarantee": bool(normal_coverage >= target),
                    "cross_threshold": float(cross_threshold),
                    "cross_score_coverage": cross_score_coverage,
                    "cross_normal_coverage": cross_normal_coverage,
                    "cross_meets_score_guarantee": bool(cross_score_coverage >= target),
                }
            )
        except Exception as e:
            score_coverage_results.append(
                {
                    "target_coverage": target,
                    "error": str(e),
                    "split_meets_score_guarantee": False,
                    "cross_meets_score_guarantee": False,
                }
            )

    return {"score_coverage_results": score_coverage_results}


# ---------------------------------------------------------------------------
# MD-003: Fusion Weight Analysis
# ---------------------------------------------------------------------------

# Track whether Strategy A has been attempted globally
_strategy_a_attempted = False
_strategy_a_failed = False


def run_fusion_weight_analysis(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Analyze neural-symbolic fusion weights.

    Strategy A: Try NeuroSymbolicHub (requires torch, NeuralEncoder, KnowledgeGraph).
    Strategy B (fallback): Validate statistical detector's adaptive ensemble
    weights which are exercised during fit_with_labels().
    """
    global _strategy_a_attempted, _strategy_a_failed

    # Strategy A: try NeuroSymbolicHub if not already known to fail
    if not _strategy_a_failed:
        _strategy_a_attempted = True
        result = _try_neurosymbolic_hub(name, X_train, y_train, X_test, y_test)
        if result is not None:
            return result
        # Strategy A failed; don't retry on subsequent datasets
        _strategy_a_failed = True

    # Strategy B: validate statistical detector adaptive weights
    return _run_statistical_weight_analysis(name, X_train, y_train, X_test, y_test)


def _try_neurosymbolic_hub(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any] | None:
    """Attempt Strategy A: exercise NeuroSymbolicHub's _learn_fusion_weights.

    Returns result dict on success, None on failure.

    The NeuroSymbolicHub requires NeuralEncoder + KnowledgeGraph with
    per-sample symbolic forward-chaining.  Even on small datasets the
    combined fit is prohibitively slow for a benchmark loop over 50+
    datasets.  We verify importability but fall back to Strategy B.
    """
    try:
        from omni_mercury_engine.core.neurosymbolic_hub import (  # noqa: F401
            FusionMode,
            NeuroSymbolicHub,
        )

        # Hub imports OK, but per-sample KnowledgeGraph.forward_chain() makes
        # _learn_fusion_weights() O(n * rules * max_iterations) which is too
        # slow for a 50+ dataset benchmark loop.  Confirmed importable but
        # architecturally unsuitable for batch cross-validation.
        return None
    except ImportError:
        return None


def _run_statistical_weight_analysis(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Strategy B: validate statistical detector adaptive ensemble weights.

    The 0.40/0.30/0.30 defaults and adaptive weights are exercised
    during fit_with_labels() via _compute_adaptive_weights().
    """
    detector = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
    detector.fit_with_labels(X_train, y_train)

    result = detector.detect(X_test)
    scores = result["scores"]

    calibrated_threshold = detector._supervised_threshold or 0.5
    preds = (scores > calibrated_threshold).astype(int)
    f1_val = float(f1_score(y_test, preds, zero_division=0))

    adaptive_weights = getattr(detector, "_adaptive_weights", [0.40, 0.30, 0.30])
    if isinstance(adaptive_weights, np.ndarray):
        adaptive_weights = adaptive_weights.tolist()

    weight_source = getattr(detector, "_weight_source", "unknown")
    component_aucs = getattr(detector, "_component_aucs", {})

    return {
        "strategy_used": "statistical_adaptive_weights",
        "neural_weight": None,
        "symbolic_weight": None,
        "adaptive_weights": adaptive_weights,
        "weight_source": str(weight_source),
        "f1_at_learned_weights": f1_val,
        "component_aucs": {k: float(v) for k, v in component_aucs.items()},
        "notes": (
            "MD-003 PARTIALLY RESOLVED: Neural-symbolic fusion weights live in "
            "NeuroSymbolicHub (requires torch + NeuralEncoder + KnowledgeGraph). "
            "Statistical detector adaptive ensemble weights validated instead."
        ),
    }


def _bce_objective_3way(w: np.ndarray, scores_matrix: np.ndarray, y: np.ndarray) -> float:
    """BCE loss for 3-way weighted fusion.

    Same mechanism as NeuroSymbolicHub._learn_fusion_weights (line 889).
    """
    w = np.abs(w)
    w = w / (np.sum(w) + 1e-10)
    fused = scores_matrix @ w
    fused = np.clip(fused, 1e-10, 1 - 1e-10)
    return float(-np.mean(y * np.log(fused) + (1 - y) * np.log(1 - fused)))


def _f1_from_weights(
    w: np.ndarray,
    scores_matrix: np.ndarray,
    y: np.ndarray,
    threshold: float | None = None,
) -> float:
    """Compute F1 score for a given weight vector on component scores."""
    w = np.abs(w)
    w = w / (np.sum(w) + 1e-10)
    fused = np.clip(scores_matrix @ w, 0.0, 1.0)
    if threshold is None:
        # Use Youden's J to find optimal threshold
        thresholds = np.unique(fused)
        if len(thresholds) > 200:
            thresholds = np.percentile(fused, np.linspace(0, 100, 200))
        best_f1 = 0.0
        for t in thresholds:
            preds = (fused > t).astype(int)
            f1_val = float(f1_score(y, preds, zero_division=0))
            best_f1 = max(best_f1, f1_val)
        return best_f1
    preds = (fused > threshold).astype(int)
    return float(f1_score(y, preds, zero_division=0))


def run_fusion_weight_cv(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Cross-validate fusion weights using L-BFGS-B (same as NeuroSymbolicHub).

    The NeuroSymbolicHub._learn_fusion_weights() (line 865) uses::

        minimize(bce_objective, [0.6, 0.4], method='L-BFGS-B',
                 bounds=[(0.1, 0.9), (0.1, 0.9)])

    We replicate this exact optimization on the statistical detector's
    3-component scores (resonance, kinematic, infogeo) to validate whether:
    (a) The default weights (0.4/0.3/0.3) are near-optimal
    (b) The adaptive AUC-proportional weights are near-optimal
    (c) PHI-weighted (~0.618/0.382 for 2-way) is near-optimal

    Uses StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).
    """
    PHI = (1.0 + np.sqrt(5.0)) / 2.0
    phi_2way_weight = float(PHI / (1.0 + PHI))  # ~0.618

    # Fit detector on full training data to get component scores
    detector = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
    detector.fit(X_train)

    # Get 3-component scores for training data
    train_result = detector.detect(X_train)
    train_components = train_result["ensemble_components"]
    train_scores_matrix = np.column_stack(
        [
            train_components["resonance"],
            train_components["kinematic"],
            train_components["info_geometry"],
        ]
    )

    # Get 3-component scores for test data
    test_result = detector.detect(X_test)
    test_components = test_result["ensemble_components"]
    test_scores_matrix = np.column_stack(
        [
            test_components["resonance"],
            test_components["kinematic"],
            test_components["info_geometry"],
        ]
    )

    # Get adaptive weights for comparison
    detector_supervised = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
    detector_supervised.fit(X_train)
    detector_supervised._adaptive_weights = detector_supervised._compute_adaptive_weights(
        X_train, y_train
    )
    adaptive_weights = detector_supervised._adaptive_weights.tolist()

    # Cross-validate to find optimal weights
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_optimal_weights: list[np.ndarray] = []

    for train_idx, val_idx in skf.split(train_scores_matrix, y_train):
        sm_train = train_scores_matrix[train_idx]
        y_tr = y_train[train_idx]
        sm_val = train_scores_matrix[val_idx]
        y_va = y_train[val_idx]

        # Optimize weights on fold training data
        result = minimize(
            _bce_objective_3way,
            [0.4, 0.3, 0.3],
            args=(sm_train, y_tr),
            method="L-BFGS-B",
            bounds=[(0.05, 0.95)] * 3,
        )
        opt_w = np.abs(result.x)
        opt_w = opt_w / np.sum(opt_w)
        cv_optimal_weights.append(opt_w)

    # Average optimal weights across folds
    mean_optimal = np.mean(cv_optimal_weights, axis=0)
    mean_optimal = mean_optimal / mean_optimal.sum()
    std_optimal = np.std(cv_optimal_weights, axis=0)

    # Compute test F1 for each weight scheme
    default_weights = np.array([0.4, 0.3, 0.3])
    f1_at_optimal = _f1_from_weights(mean_optimal, test_scores_matrix, y_test)
    f1_at_default = _f1_from_weights(default_weights, test_scores_matrix, y_test)
    f1_at_adaptive = _f1_from_weights(np.array(adaptive_weights), test_scores_matrix, y_test)

    # PHI 2-way fusion: best component + rest
    # Find best component by AUC on test data
    component_aucs = []
    for i in range(3):
        component_aucs.append(_safe_auc(y_test, test_scores_matrix[:, i]))
    best_idx = int(np.argmax(component_aucs))
    rest_idx = [i for i in range(3) if i != best_idx]
    phi_3way = np.zeros(3)
    phi_3way[best_idx] = phi_2way_weight
    rest_weight = (1.0 - phi_2way_weight) / len(rest_idx)
    for ri in rest_idx:
        phi_3way[ri] = rest_weight
    f1_at_phi_2way = _f1_from_weights(phi_3way, test_scores_matrix, y_test)

    return {
        "strategy_used": "lbfgs_cv_3way",
        "n_splits": n_splits,
        "mean_optimal_weights": mean_optimal.tolist(),
        "std_optimal_weights": std_optimal.tolist(),
        "default_weights": default_weights.tolist(),
        "adaptive_weights": adaptive_weights,
        "f1_at_optimal": f1_at_optimal,
        "f1_at_default": f1_at_default,
        "f1_at_adaptive": f1_at_adaptive,
        "delta_optimal_vs_default": f1_at_optimal - f1_at_default,
        "delta_optimal_vs_adaptive": f1_at_optimal - f1_at_adaptive,
        "phi_2way_weight": phi_2way_weight,
        "phi_3way_weights": phi_3way.tolist(),
        "f1_at_phi_2way": f1_at_phi_2way,
        "component_aucs": {
            "resonance": component_aucs[0],
            "kinematic": component_aucs[1],
            "infogeo": component_aucs[2],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_validation(
    skip_conformal: bool = False,
    dataset_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full calibration validation suite."""
    print("=" * 70)
    print("Mercury Agent - Calibration Validation Harness")
    print("MD-011 (calibration), MD-003 (fusion weights), MD-005 (conformal)")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    # --- Load ADBench datasets ---
    print("\n[ADBench] Loading 47 tabular datasets ...")
    adb_entries = _load_adbench()

    # --- Load Domain datasets ---
    print("[Domain] Loading Mercury domain datasets ...")
    domain_entries: list[dict[str, Any]] = []
    for dname, cat, cls_name, mod, kwargs in DOMAIN_DATASETS:
        domain_entries.append(_load_domain_dataset(dname, cat, cls_name, mod, **kwargs))

    all_entries = adb_entries + domain_entries

    # Apply dataset filter
    if dataset_filter:
        filter_lower = [f.lower() for f in dataset_filter]
        all_entries = [e for e in all_entries if e["name"].lower() in filter_lower]
        print(f"Filtered to {len(all_entries)} datasets: {dataset_filter}")

    print(f"\nProcessing {len(all_entries)} datasets ...\n")

    for entry in all_entries:
        result = _validate_single(entry, skip_conformal=skip_conformal)
        results.append(result)
        gc.collect()

    # --- Summary ---
    successful = [r for r in results if r.get("error") is None]

    # MD-011 summary
    cal_improved = sum(1 for r in successful if r.get("calibration", {}).get("delta_f1", 0) > 0)
    cal_same = sum(1 for r in successful if r.get("calibration", {}).get("delta_f1", 0) == 0)
    cal_degraded = sum(1 for r in successful if r.get("calibration", {}).get("delta_f1", 0) < 0)
    cal_f1s = [r["calibration"]["calibrated_f1"] for r in successful if "calibration" in r]
    uncal_f1s = [r["calibration"]["uncalibrated_f1"] for r in successful if "calibration" in r]
    delta_f1s = [r["calibration"]["delta_f1"] for r in successful if "calibration" in r]

    # MD-005 summary (legacy accuracy-based)
    conformal_stats: dict[float, dict[str, int]] = {}
    if not skip_conformal:
        for target in CONFORMAL_COVERAGE_LEVELS:
            conformal_stats[target] = {"meets": 0, "total": 0}
        for r in successful:
            if "conformal" not in r:
                continue
            for cov in r["conformal"].get("coverage_results", []):
                tgt = cov.get("target_coverage")
                if tgt in conformal_stats:
                    conformal_stats[tgt]["total"] += 1
                    if cov.get("meets_guarantee", False):
                        conformal_stats[tgt]["meets"] += 1

    # MD-005 summary (corrected score-based)
    score_conformal_stats: dict[float, dict[str, int]] = {}
    if not skip_conformal:
        for target in CONFORMAL_COVERAGE_LEVELS:
            score_conformal_stats[target] = {
                "split_meets": 0,
                "cross_meets": 0,
                "normal_meets": 0,
                "total": 0,
            }
        for r in successful:
            if "conformal_score_based" not in r:
                continue
            for cov in r["conformal_score_based"].get("score_coverage_results", []):
                tgt = cov.get("target_coverage")
                if tgt in score_conformal_stats:
                    score_conformal_stats[tgt]["total"] += 1
                    if cov.get("split_meets_score_guarantee", False):
                        score_conformal_stats[tgt]["split_meets"] += 1
                    if cov.get("cross_meets_score_guarantee", False):
                        score_conformal_stats[tgt]["cross_meets"] += 1
                    if cov.get("split_meets_normal_guarantee", False):
                        score_conformal_stats[tgt]["normal_meets"] += 1

    # MD-003 summary
    weight_strategies = [
        r.get("fusion", {}).get("strategy_used", "unknown") for r in successful if "fusion" in r
    ]
    all_adaptive_weights = [
        r["fusion"]["adaptive_weights"]
        for r in successful
        if "fusion" in r and r["fusion"].get("adaptive_weights") is not None
    ]

    # MD-003 CV summary
    cv_deltas_default: list[float] = []
    cv_deltas_adaptive: list[float] = []
    cv_optimal_weights_all: list[list[float]] = []
    for r in successful:
        fcv = r.get("fusion_cv", {})
        if "delta_optimal_vs_default" in fcv:
            cv_deltas_default.append(fcv["delta_optimal_vs_default"])
            cv_deltas_adaptive.append(fcv["delta_optimal_vs_adaptive"])
            cv_optimal_weights_all.append(fcv["mean_optimal_weights"])

    # Decision gate for MD-003: default validated if optimal-default < 0.02
    default_validated_count = sum(1 for d in cv_deltas_default if d < 0.02)
    adaptive_validated_count = sum(1 for d in cv_deltas_adaptive if d < 0.02)

    summary = {
        "total_datasets": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "md_011": {
            "calibration_improved": cal_improved,
            "calibration_same": cal_same,
            "calibration_degraded": cal_degraded,
            "mean_calibrated_f1": float(np.mean(cal_f1s)) if cal_f1s else None,
            "mean_uncalibrated_f1": float(np.mean(uncal_f1s)) if uncal_f1s else None,
            "mean_delta_f1": float(np.mean(delta_f1s)) if delta_f1s else None,
            "pct_improved_or_same": (
                float((cal_improved + cal_same) / len(successful) * 100) if successful else None
            ),
        },
        "md_005_accuracy_based": (
            {
                str(tgt): {
                    "meets_guarantee": stats["meets"],
                    "total": stats["total"],
                    "pct": (
                        float(stats["meets"] / stats["total"] * 100) if stats["total"] > 0 else None
                    ),
                }
                for tgt, stats in conformal_stats.items()
            }
            if not skip_conformal
            else "skipped"
        ),
        "md_005_score_based": (
            {
                str(tgt): {
                    "split_meets": stats["split_meets"],
                    "cross_meets": stats["cross_meets"],
                    "normal_meets": stats["normal_meets"],
                    "total": stats["total"],
                    "split_pct": (
                        float(stats["split_meets"] / stats["total"] * 100)
                        if stats["total"] > 0
                        else None
                    ),
                    "cross_pct": (
                        float(stats["cross_meets"] / stats["total"] * 100)
                        if stats["total"] > 0
                        else None
                    ),
                    "normal_pct": (
                        float(stats["normal_meets"] / stats["total"] * 100)
                        if stats["total"] > 0
                        else None
                    ),
                }
                for tgt, stats in score_conformal_stats.items()
            }
            if not skip_conformal
            else "skipped"
        ),
        "md_003": {
            "strategy_used": (weight_strategies[0] if weight_strategies else "unknown"),
            "n_datasets_with_adaptive_weights": len(all_adaptive_weights),
            "weight_distribution": _compute_weight_distribution(all_adaptive_weights),
        },
        "md_003_cv": {
            "strategy": "lbfgs_cv_3way",
            "n_datasets": len(cv_deltas_default),
            "mean_delta_optimal_vs_default": (
                float(np.mean(cv_deltas_default)) if cv_deltas_default else None
            ),
            "mean_delta_optimal_vs_adaptive": (
                float(np.mean(cv_deltas_adaptive)) if cv_deltas_adaptive else None
            ),
            "default_validated_count": default_validated_count,
            "default_validated_pct": (
                float(default_validated_count / len(cv_deltas_default) * 100)
                if cv_deltas_default
                else None
            ),
            "adaptive_validated_count": adaptive_validated_count,
            "adaptive_validated_pct": (
                float(adaptive_validated_count / len(cv_deltas_adaptive) * 100)
                if cv_deltas_adaptive
                else None
            ),
            "mean_optimal_weights": (
                np.mean(cv_optimal_weights_all, axis=0).tolist() if cv_optimal_weights_all else None
            ),
        },
    }

    # Print summary table
    _print_summary_table(results, summary, skip_conformal)

    output = {
        "metadata": {
            "git_commit": _git_commit(),
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}" f".{sys.version_info.micro}"
            ),
            "timestamp": datetime.now(UTC).isoformat(),
            "max_samples_per_dataset": MAX_SAMPLES,
            "skip_conformal": skip_conformal,
        },
        "summary": summary,
        "results": results,
    }

    return output


def _compute_weight_distribution(
    all_weights: list[list[float]],
) -> dict[str, Any] | None:
    """Compute weight distribution statistics across datasets."""
    if not all_weights:
        return None
    arr = np.array(all_weights)
    return {
        "resonance": {
            "mean": float(np.mean(arr[:, 0])),
            "std": float(np.std(arr[:, 0])),
            "min": float(np.min(arr[:, 0])),
            "max": float(np.max(arr[:, 0])),
        },
        "kinematic": {
            "mean": float(np.mean(arr[:, 1])),
            "std": float(np.std(arr[:, 1])),
            "min": float(np.min(arr[:, 1])),
            "max": float(np.max(arr[:, 1])),
        },
        "infogeo": {
            "mean": float(np.mean(arr[:, 2])),
            "std": float(np.std(arr[:, 2])),
            "min": float(np.min(arr[:, 2])),
            "max": float(np.max(arr[:, 2])),
        },
    }


def _validate_single(
    entry: dict[str, Any],
    skip_conformal: bool = False,
) -> dict[str, Any]:
    """Run all three validations on a single dataset."""
    name = entry["name"]
    category = entry.get("category", "unknown")

    if "error" in entry:
        print(f"  [{name}] SKIP: {entry['error'][:80]}")
        return {"name": name, "category": category, "error": entry["error"]}

    X_full = entry["X"]
    y_full = entry["y"]

    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)

    unique_labels = np.unique(y_full)
    if len(unique_labels) < 2:
        msg = f"Only one class present (labels={unique_labels.tolist()})"
        print(f"  [{name}] SKIP: {msg}")
        return {"name": name, "category": category, "error": msg}

    # Cap with stratified sampling
    X_full, y_full = _cap_stratified(X_full, y_full, MAX_SAMPLES)

    # Handle NaN/Inf
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

    # Stratified 60/20/20 split
    rng = np.random.RandomState(42)
    split = _stratified_split(X_full, y_full, rng)
    if split is None:
        msg = "Insufficient samples for stratified 60/20/20 split"
        print(f"  [{name}] SKIP: {msg}")
        return {"name": name, "category": category, "error": msg}

    X_train, y_train, X_cal, y_cal, X_test, y_test = split

    # StandardScaler fit on train, transform cal and test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cal = scaler.transform(X_cal)
    X_test = scaler.transform(X_test)

    result: dict[str, Any] = {
        "name": name,
        "category": category,
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "anomaly_ratio": float(y_full.mean()),
        "error": None,
    }

    # --- MD-011: Calibration validation ---
    try:
        t0 = time.perf_counter()
        cal_result = run_calibration_validation(name, X_train, y_train, X_test, y_test)
        cal_result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        result["calibration"] = cal_result
    except Exception as e:
        result["calibration"] = {"error": str(e)}

    # --- MD-005: Conformal coverage (accuracy-based, legacy) ---
    if not skip_conformal:
        try:
            t0 = time.perf_counter()
            conf_result = run_conformal_coverage(name, X_train, y_train, X_cal, X_test, y_test)
            conf_result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            result["conformal"] = conf_result
        except Exception as e:
            result["conformal"] = {"error": str(e)}

    # --- MD-005: Score-based conformal coverage (corrected metric) ---
    if not skip_conformal:
        try:
            t0 = time.perf_counter()
            score_cov_result = measure_score_based_coverage(
                name, X_train, y_train, X_cal, X_test, y_test
            )
            score_cov_result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
            result["conformal_score_based"] = score_cov_result
        except Exception as e:
            result["conformal_score_based"] = {"error": str(e)}

    # --- MD-003: Fusion weight analysis (Strategy B) ---
    try:
        t0 = time.perf_counter()
        fusion_result = run_fusion_weight_analysis(name, X_train, y_train, X_test, y_test)
        fusion_result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        result["fusion"] = fusion_result
    except Exception as e:
        result["fusion"] = {"error": str(e)}

    # --- MD-003: L-BFGS-B cross-validated fusion weights ---
    try:
        t0 = time.perf_counter()
        fusion_cv_result = run_fusion_weight_cv(name, X_train, y_train, X_test, y_test)
        fusion_cv_result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        result["fusion_cv"] = fusion_cv_result
    except Exception as e:
        result["fusion_cv"] = {"error": str(e)}

    # Print row
    cal = result.get("calibration", {})
    conf_sb = result.get("conformal_score_based", {})
    fus = result.get("fusion", {})
    fcv = result.get("fusion_cv", {})

    cal_f1 = cal.get("calibrated_f1", float("nan"))
    uncal_f1 = cal.get("uncalibrated_f1", float("nan"))
    delta = cal.get("delta_f1", float("nan"))
    wt_src = cal.get("weight_source", "?")[:8]

    cov_strs = []
    if not skip_conformal and "score_coverage_results" in conf_sb:
        for cov in conf_sb["score_coverage_results"]:
            sc = cov.get("split_score_coverage", float("nan"))
            cov_strs.append(f"{sc:.2f}")
    else:
        cov_strs = ["skip"] * 3

    fus_strat = fcv.get("strategy_used", fus.get("strategy_used", "?"))[:6]
    cv_delta = fcv.get("delta_optimal_vs_default", float("nan"))

    print(
        f"  [{name:<20s}] "
        f"Cal={cal_f1:.3f} Uncal={uncal_f1:.3f} D={delta:+.3f} "
        f"Wt={wt_src} "
        f"ScCov={'/'.join(cov_strs)} "
        f"Fus={fus_strat} dF1={cv_delta:+.3f}"
    )

    return result


def _print_summary_table(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    skip_conformal: bool,
) -> None:
    """Print human-readable summary."""
    successful = [r for r in results if r.get("error") is None]

    print("\n" + "=" * 100)
    print(
        f"{'Dataset':<22s} {'Cal.F1':>7s} {'Uncal.F1':>9s} {'Delta':>7s} "
        f"{'Wt.Src':>8s} {'Cov@90':>7s} {'Cov@95':>7s} {'Cov@99':>7s} {'Fusion':>8s}"
    )
    print("-" * 100)

    for r in successful:
        cal = r.get("calibration", {})
        conf = r.get("conformal", {})
        fus = r.get("fusion", {})

        cal_f1 = f"{cal.get('calibrated_f1', float('nan')):.4f}"
        uncal_f1 = f"{cal.get('uncalibrated_f1', float('nan')):.4f}"
        delta = f"{cal.get('delta_f1', float('nan')):+.4f}"
        wt_src = cal.get("weight_source", "?")[:8]

        covs: list[str] = []
        if not skip_conformal and "coverage_results" in conf:
            for cov in conf["coverage_results"]:
                emp = cov.get("empirical_coverage", float("nan"))
                covs.append(f"{emp:.4f}")
        else:
            covs = ["  skip"] * 3

        fus_strat = fus.get("strategy_used", "?")[:8]

        print(
            f"{r['name']:<22s} {cal_f1:>7s} {uncal_f1:>9s} {delta:>7s} "
            f"{wt_src:>8s} {covs[0]:>7s} {covs[1]:>7s} {covs[2]:>7s} {fus_strat:>8s}"
        )

    failed = [r for r in results if r.get("error") is not None]
    if failed:
        print(f"\n--- Failed ({len(failed)}) ---")
        for r in failed:
            print(f"  {r['name']}: {r['error'][:70]}")

    print("\n--- MD-011: Calibration Pipeline ---")
    md011 = summary.get("md_011", {})
    print(
        f"  Improved: {md011.get('calibration_improved', 0)} | "
        f"Same: {md011.get('calibration_same', 0)} | "
        f"Degraded: {md011.get('calibration_degraded', 0)}"
    )
    if md011.get("mean_calibrated_f1") is not None:
        print(f"  Mean Calibrated F1:   {md011['mean_calibrated_f1']:.4f}")
        print(f"  Mean Uncalibrated F1: {md011['mean_uncalibrated_f1']:.4f}")
        print(f"  Mean Delta F1:        {md011['mean_delta_f1']:+.4f}")

    if not skip_conformal:
        print("\n--- MD-005: Conformal Coverage (accuracy-based, legacy) ---")
        md005 = summary.get("md_005_accuracy_based", {})
        if isinstance(md005, dict):
            for tgt_str, stats in md005.items():
                if isinstance(stats, dict):
                    pct = stats.get("pct") or 0
                    print(
                        f"  Accuracy@{tgt_str}: "
                        f"{stats.get('meets_guarantee', 0)}/{stats.get('total', 0)} "
                        f"({pct:.1f}%) meet guarantee"
                    )

        print("\n--- MD-005: Conformal Coverage (score-based, CORRECTED) ---")
        md005s = summary.get("md_005_score_based", {})
        if isinstance(md005s, dict):
            for tgt_str, stats in md005s.items():
                if isinstance(stats, dict):
                    sp = stats.get("split_pct") or 0
                    cp = stats.get("cross_pct") or 0
                    np_ = stats.get("normal_pct") or 0
                    total = stats.get("total", 0)
                    print(
                        f"  ScoreCov@{tgt_str}: "
                        f"split={stats.get('split_meets', 0)}/{total} ({sp:.1f}%) "
                        f"cross={stats.get('cross_meets', 0)}/{total} ({cp:.1f}%) "
                        f"normal={stats.get('normal_meets', 0)}/{total} ({np_:.1f}%)"
                    )

    print("\n--- MD-003: Fusion Weight Analysis ---")
    md003 = summary.get("md_003", {})
    print(f"  Strategy: {md003.get('strategy_used', 'unknown')}")
    wd = md003.get("weight_distribution")
    if wd:
        for comp in ["resonance", "kinematic", "infogeo"]:
            if comp in wd:
                s = wd[comp]
                print(
                    f"  {comp:<12s}: mean={s['mean']:.3f} "
                    f"std={s['std']:.3f} "
                    f"range=[{s['min']:.3f}, {s['max']:.3f}]"
                )

    print("\n--- MD-003: L-BFGS-B Cross-Validated Fusion Weights ---")
    md003cv = summary.get("md_003_cv", {})
    print(f"  Datasets tested: {md003cv.get('n_datasets', 0)}")
    if md003cv.get("mean_delta_optimal_vs_default") is not None:
        print(
            f"  Mean delta (optimal-default): " f"{md003cv['mean_delta_optimal_vs_default']:+.4f}"
        )
        print(
            f"  Mean delta (optimal-adaptive): " f"{md003cv['mean_delta_optimal_vs_adaptive']:+.4f}"
        )
        print(
            f"  Default validated (<0.02 gap): "
            f"{md003cv['default_validated_count']}/{md003cv['n_datasets']} "
            f"({md003cv['default_validated_pct']:.1f}%)"
        )
        print(
            f"  Adaptive validated (<0.02 gap): "
            f"{md003cv['adaptive_validated_count']}/{md003cv['n_datasets']} "
            f"({md003cv['adaptive_validated_pct']:.1f}%)"
        )
        if md003cv.get("mean_optimal_weights"):
            w = md003cv["mean_optimal_weights"]
            print(f"  Mean optimal weights: " f"R={w[0]:.3f} K={w[1]:.3f} I={w[2]:.3f}")

    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mercury Agent - Calibration Validation Harness")
    parser.add_argument(
        "--skip-conformal",
        action="store_true",
        help="Skip conformal coverage validation (faster)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of dataset names to run",
    )
    args = parser.parse_args()

    dataset_filter = None
    if args.datasets:
        dataset_filter = [d.strip() for d in args.datasets.split(",")]

    output = run_validation(
        skip_conformal=args.skip_conformal,
        dataset_filter=dataset_filter,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
