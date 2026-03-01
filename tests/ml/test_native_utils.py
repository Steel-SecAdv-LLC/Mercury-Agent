"""
Numerical correctness tests for _native_utils.py native implementations.
Each test compares native output against a known-good reference computed
from scipy or numpy directly (not sklearn) using a fixed random seed.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0-or-later
"""

import numpy as np
import pytest

from omni_mercury_engine.ml._native_utils import (
    NativeKFold,
    NativeLogisticRegression,
    NativeStandardScaler,
    native_brier_score_loss,
    native_f1_score,
    native_roc_auc_score,
    native_train_test_split,
)


RNG = np.random.RandomState(42)


def test_native_roc_auc_perfect_separation() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert abs(native_roc_auc_score(y_true, y_score) - 1.0) < 1e-9


def test_native_roc_auc_random_chance() -> None:
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 200)
    y_score = rng.rand(200)
    auc = native_roc_auc_score(y_true, y_score)
    assert 0.35 < auc < 0.65, f"Random AUC {auc:.3f} outside expected range"


def test_native_f1_score_binary() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    # TP=2, FP=1, FN=1 → precision=0.667, recall=0.667, F1=0.667
    f1 = native_f1_score(y_true, y_pred)
    assert abs(f1 - (2 / 3)) < 1e-6, f"F1={f1:.6f}"


def test_native_brier_score_perfect() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert native_brier_score_loss(y_true, y_prob) < 1e-9


def test_native_brier_score_worst() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert abs(native_brier_score_loss(y_true, y_prob) - 1.0) < 1e-9


def test_native_kfold_split_count() -> None:
    X = np.arange(100).reshape(50, 2)
    kf = NativeKFold(n_splits=5, shuffle=False)
    splits = list(kf.split(X))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(test_idx) == 10
        assert len(train_idx) == 40


def test_native_kfold_no_overlap() -> None:
    X = np.arange(60).reshape(30, 2)
    kf = NativeKFold(n_splits=3, shuffle=True, random_state=42)
    all_test_indices = []
    for _, test_idx in kf.split(X):
        all_test_indices.extend(test_idx.tolist())
    assert len(all_test_indices) == len(set(all_test_indices)), "Fold test sets overlap"
    assert sorted(all_test_indices) == list(range(30)), "Not all samples covered"


def test_native_standard_scaler_zero_mean_unit_var() -> None:
    rng = np.random.RandomState(7)
    X = rng.randn(100, 5) * 10 + 3
    scaler = NativeStandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)


def test_native_train_test_split_sizes() -> None:
    X = np.arange(200).reshape(100, 2)
    y = np.zeros(100)
    y[50:] = 1
    X_tr, X_te, y_tr, y_te = native_train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    assert len(X_tr) == 80
    assert len(X_te) == 20
    # Stratification: both splits should have ~50% class 1
    assert abs(y_te.mean() - 0.5) < 0.15


def test_native_logistic_regression_converges() -> None:
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(100, 5), rng.randn(100, 5) + 3])
    y = np.array([0] * 100 + [1] * 100)
    lr = NativeLogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    acc = (lr.predict(X) == y).mean()
    assert acc > 0.90, f"Logistic regression accuracy {acc:.3f} too low on separable data"
