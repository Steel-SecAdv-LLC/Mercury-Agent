"""Backward-compatibility shim — all Mercury ML primitives live in mercury_ml.

This module re-exports every public name under the legacy ``native_*`` /
``Native*`` aliases so that existing import statements continue to work.
New code should import directly from ``omni_mercury_engine.ml.mercury_ml``.

Copyright (C) 2025 Steel Security Advisors LLC (GPL-3.0-only)
"""

from __future__ import annotations

# Re-export everything from the canonical module under both old and new names.
from omni_mercury_engine.ml.mercury_ml import (
    PCA,
    EllipticEnvelope,
    GaussianMixture,
    GradientBoostingClassifier,
    IsotonicRegression,
    KFold,
    KMeans,
    LabelEncoder,
    LocalOutlierFactor,
    LogisticRegression,
    NearestNeighbors,
    OneClassSVM,
    PassiveAggressiveClassifier,
    SGDClassifier,
    StandardScaler,
    StratifiedKFold,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    calibration_curve,
    clone,
    confusion_matrix,
    cross_val_predict,
    cross_val_score,
    f1_score,
    log_loss,
    mutual_info_classif,
    precision_score,
    recall_score,
    roc_auc_score,
    train_test_split,
)

# Legacy aliases (Native* / native_* prefixed names)
NativeStandardScaler = StandardScaler
NativeLabelEncoder = LabelEncoder
NativeKFold = KFold
NativeStratifiedKFold = StratifiedKFold
NativeLogisticRegression = LogisticRegression
NativeSGDClassifier = SGDClassifier
NativePassiveAggressiveClassifier = PassiveAggressiveClassifier
NativeGradientBoostingClassifier = GradientBoostingClassifier
NativePCA = PCA
NativeKMeans = KMeans
NativeGaussianMixture = GaussianMixture
NativeIsotonicRegression = IsotonicRegression
NativeNearestNeighbors = NearestNeighbors
NativeOneClassSVM = OneClassSVM
NativeLocalOutlierFactor = LocalOutlierFactor
NativeEllipticEnvelope = EllipticEnvelope

native_roc_auc_score = roc_auc_score
native_average_precision_score = average_precision_score
native_accuracy_score = accuracy_score
native_precision_score = precision_score
native_recall_score = recall_score
native_f1_score = f1_score
native_log_loss = log_loss
native_brier_score_loss = brier_score_loss
native_confusion_matrix = confusion_matrix
native_train_test_split = train_test_split
native_calibration_curve = calibration_curve
native_mutual_info_classif = mutual_info_classif
native_cross_val_predict = cross_val_predict
native_cross_val_score = cross_val_score
native_clone = clone
