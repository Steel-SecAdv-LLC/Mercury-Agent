# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Core anomaly detection metrics.

Implements standard metrics used in anomaly detection literature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label as connected_components

from omni_mercury_engine.core.config import ThresholdConfig

logger = logging.getLogger(__name__)

# Centralized thresholds for consistent behavior
_thresholds = ThresholdConfig()


def _to_numpy(arr: Any) -> np.ndarray[Any, Any]:
    """Convert array-like to numpy."""
    if hasattr(arr, "cpu"):  # torch tensor
        # Typed intermediate: arr is Any, so bind to an ndarray-typed name to
        # drop the no-any-return ignore.
        converted: np.ndarray[Any, Any] = np.asarray(arr.cpu().numpy())
        return converted
    return np.asarray(arr)


def compute_auroc(
    y_true: np.ndarray[Any, Any],
    y_score: np.ndarray[Any, Any],
) -> float:
    """Compute Area Under ROC Curve.

    Args:
        y_true: Binary labels [N]
        y_score: Anomaly scores [N]

    Returns:
        AUROC score in [0, 1]. Returns 0.5 for undefined cases
        (e.g., all labels are the same class).
    """
    y_true = _to_numpy(y_true).flatten()
    y_score = _to_numpy(y_score).flatten()

    # Handle edge case: all labels are the same class
    # AUROC is undefined in this case; return 0.5 (random classifier baseline)
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return 0.5

    try:
        from omni_mercury_engine.ml.mercury_ml import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        # Manual computation
        return _manual_auroc(y_true, y_score)
    except ValueError:
        # sklearn raises ValueError for edge cases; fall back to manual
        return _manual_auroc(y_true, y_score)


def _manual_auroc(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    """Manual AUROC computation without sklearn."""
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]

    # Compute AUC via Wilcoxon-Mann-Whitney statistic
    ranks = np.arange(1, len(y_true) + 1)
    pos_ranks = ranks[y_true_sorted == 1].sum()

    auc = (pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_auprc(
    y_true: np.ndarray[Any, Any],
    y_score: np.ndarray[Any, Any],
) -> float:
    """Compute Area Under Precision-Recall Curve.

    Args:
        y_true: Binary labels [N]
        y_score: Anomaly scores [N]

    Returns:
        AUPRC score in [0, 1]
    """
    y_true = _to_numpy(y_true).flatten()
    y_score = _to_numpy(y_score).flatten()

    try:
        from omni_mercury_engine.ml.mercury_ml import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except ImportError:
        # Manual computation
        return _manual_auprc(y_true, y_score)


def _manual_auprc(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    """Manual AUPRC computation."""
    # Sort by score descending
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]

    # Compute precision at each recall level
    tp_cumsum = np.cumsum(y_true_sorted)
    fp_cumsum = np.cumsum(1 - y_true_sorted)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / y_true.sum()

    # Compute area under curve
    recall_diff = np.diff(recall, prepend=0)
    auprc = (precision * recall_diff).sum()

    return float(auprc)


def compute_f1_max(
    y_true: np.ndarray[Any, Any],
    y_score: np.ndarray[Any, Any],
    n_thresholds: int = 100,
) -> tuple[float, float]:
    """Compute maximum F1 score across thresholds.

    .. warning::
        **In-sample / diagnostic.** The threshold is tuned on the same
        ``(y_true, y_score)`` it is scored against, so ``max_f1`` is an
        optimistic upper bound, not a transparent operating point.  For leakage-free
        reporting tune on validation and report on test via
        :func:`omni_mercury_engine.evaluation.metrics.fit_threshold` /
        :meth:`AnomalyMetrics.compute_all` with ``tune_on="val"``.

    Args:
        y_true: Binary labels [N]
        y_score: Anomaly scores [N]
        n_thresholds: Number of thresholds to try

    Returns:
        Tuple of (max_f1, optimal_threshold)
    """
    y_true = _to_numpy(y_true).flatten()
    y_score = _to_numpy(y_score).flatten()

    thresholds = np.linspace(y_score.min(), y_score.max(), n_thresholds)

    best_f1 = 0.0
    best_threshold = _thresholds.anomaly_default

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

    return float(best_f1), float(best_threshold)


def compute_optimal_threshold(
    y_true: np.ndarray[Any, Any],
    y_score: np.ndarray[Any, Any],
    metric: str = "f1",
) -> float:
    """Compute optimal threshold for given metric.

    .. warning::
        **In-sample.** The threshold is chosen on the supplied data; to avoid
        leakage when the same data is then scored, select on a validation split
        and apply to a disjoint test split (see
        :func:`omni_mercury_engine.evaluation.metrics.fit_threshold`).

    Args:
        y_true: Binary labels
        y_score: Anomaly scores
        metric: Metric to optimize ('f1', 'accuracy', 'youden')

    Returns:
        Optimal threshold value
    """
    y_true = _to_numpy(y_true).flatten()
    y_score = _to_numpy(y_score).flatten()

    if metric == "f1":
        _, threshold = compute_f1_max(y_true, y_score)
        return threshold

    thresholds = np.linspace(y_score.min(), y_score.max(), 100)
    best_score = 0.0
    best_threshold = _thresholds.anomaly_default

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        if metric == "accuracy":
            score = (y_pred == y_true).mean()
        elif metric == "youden":
            # Youden's J = sensitivity + specificity - 1
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()

            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return float(best_threshold)


def compute_pixel_auroc(
    y_true: np.ndarray[Any, Any],
    y_score: np.ndarray[Any, Any],
) -> float:
    """Compute pixel-level AUROC for anomaly localization.

    Args:
        y_true: Ground truth masks [N, H, W] or [H, W]
        y_score: Anomaly score maps [N, H, W] or [H, W]

    Returns:
        Pixel-level AUROC
    """
    y_true = _to_numpy(y_true).flatten()
    y_score = _to_numpy(y_score).flatten()

    return compute_auroc(y_true, y_score)


def compute_pro(
    y_true: np.ndarray[Any, Any],
    y_score: np.ndarray[Any, Any],
    integration_limit: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """Compute Per-Region Overlap (PRO) metric.

    PRO measures localization quality by computing overlap between
    predicted and ground truth regions for each connected component.

    Args:
        y_true: Ground truth masks [N, H, W]
        y_score: Anomaly score maps [N, H, W]
        integration_limit: FPR integration limit
        num_thresholds: Number of thresholds for curve

    Returns:
        PRO score (normalized AUC under PRO curve). Returns 1.0 for
        perfect localization where predictions exactly match ground truth.
    """
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, ...]
        y_score = y_score[np.newaxis, ...]

    # Handle perfect localization edge case:
    # If scores perfectly match masks (binary 0/1), return 1.0
    # This is the ideal case where the detector perfectly localizes anomalies
    score_is_binary = np.allclose(y_score, y_score.astype(bool).astype(float))
    if score_is_binary and np.allclose(y_score, y_true):
        return 1.0

    # Compute thresholds
    thresholds = np.linspace(y_score.min(), y_score.max(), num_thresholds)

    # Compute FPR and PRO for each threshold
    fprs = []
    pros = []

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(float)

        # Compute FPR
        fp = ((y_pred > 0) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fpr = fp / max(fp + tn, 1)
        fprs.append(fpr)

        # Compute PRO (average overlap for each ground truth region)
        overlaps = []
        for i in range(len(y_true)):
            mask = y_true[i]
            pred = y_pred[i]

            if mask.sum() == 0:
                continue

            # Find connected components
            labeled, num_features = connected_components(mask)

            for region_id in range(1, num_features + 1):
                region_mask = labeled == region_id
                region_pred = pred * region_mask

                # Overlap = intersection / region size
                overlap = region_pred.sum() / region_mask.sum()
                overlaps.append(overlap)

        if overlaps:
            pros.append(np.mean(overlaps))
        else:
            pros.append(0.0)

    # Convert to arrays and sort by FPR
    fprs = np.array(fprs)  # type: ignore[assignment, unused-ignore]
    pros = np.array(pros)  # type: ignore[assignment, unused-ignore]
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    # Integrate up to limit
    valid = fprs <= integration_limit
    if valid.sum() < 2:
        # If we have high PRO at FPR=0, that's still good localization
        # Return the PRO value at the lowest FPR
        if len(pros) > 0 and pros[0] > 0.8:
            return float(pros[0])
        return 0.0

    fprs_valid = fprs[valid]
    pros_valid = pros[valid]

    # Normalize FPR to [0, 1] within integration limit
    fprs_norm = fprs_valid / integration_limit

    # Compute AUC (trapezoid in NumPy 2.0+)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined, unused-ignore]
    pro_auc = _trapz(pros_valid, fprs_norm)

    return float(pro_auc)


@dataclass
class MetricResult:
    """Container for metric results."""

    name: str
    value: float
    threshold: float | None = None


class AnomalyMetrics:
    """Unified anomaly detection metrics calculator.

    Example:
        >>> metrics = AnomalyMetrics()
        >>> results = metrics.compute_all(y_true, y_score, y_pred)
        >>> print(results['auroc'], results['f1_max'])
    """

    @staticmethod
    def compute_all(
        y_true: np.ndarray[Any, Any],
        y_score: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any] | None = None,
        masks_true: np.ndarray[Any, Any] | None = None,
        masks_score: np.ndarray[Any, Any] | None = None,
        *,
        tune_on: str = "in_sample",
        val_frac: float = 0.2,
        test_frac: float = 0.4,
        random_state: int = 0,
    ) -> dict[str, float]:
        """Compute all standard metrics.

        Args:
            y_true: Ground truth labels [N]
            y_score: Anomaly scores [N]
            y_pred: Binary predictions (optional)
            masks_true: Ground truth masks for localization [N, H, W]
            masks_score: Predicted anomaly maps [N, H, W]
            tune_on: ``"in_sample"`` (default, legacy) tunes and reports the
                threshold on the same data -- ``f1_max``/``optimal_threshold``
                are an optimistic upper bound.  ``"val"`` tunes the threshold on
                a held-out validation split and reports the threshold-dependent
                metrics (``f1_max``, accuracy/precision/recall) on a disjoint
                test split, with AUROC/AUPRC also computed on test.  Falls back
                to in-sample (with a warning) when the data is too small or
                single-class to split.
            val_frac: Validation fraction for ``tune_on="val"``.
            test_frac: Test fraction for ``tune_on="val"``.
            random_state: Split seed for ``tune_on="val"``.

        Returns:
            Dict of metric names to values

        Raises:
            ValueError: If ``tune_on`` is not ``"in_sample"`` or ``"val"``.
        """
        if tune_on not in ("in_sample", "val"):
            raise ValueError(f"tune_on must be 'in_sample' or 'val', got {tune_on!r}")

        results: dict[str, float] = {}

        if tune_on == "val":
            return AnomalyMetrics._compute_all_split(
                y_true,
                y_score,
                y_pred=y_pred,
                masks_true=masks_true,
                masks_score=masks_score,
                val_frac=val_frac,
                test_frac=test_frac,
                random_state=random_state,
            )

        # Image-level metrics
        results["auroc"] = compute_auroc(y_true, y_score)
        results["auprc"] = compute_auprc(y_true, y_score)

        f1_max, threshold = compute_f1_max(y_true, y_score)
        results["f1_max"] = f1_max
        results["optimal_threshold"] = threshold

        # Compute accuracy if predictions provided
        if y_pred is not None:
            y_pred = _to_numpy(y_pred).flatten()
            y_true_np = _to_numpy(y_true).flatten()
            results["accuracy"] = float((y_pred == y_true_np).mean())

            tp = ((y_pred == 1) & (y_true_np == 1)).sum()
            fp = ((y_pred == 1) & (y_true_np == 0)).sum()
            fn = ((y_pred == 0) & (y_true_np == 1)).sum()

            results["precision"] = float(tp / max(tp + fp, 1))
            results["recall"] = float(tp / max(tp + fn, 1))

        # Pixel-level metrics if masks provided
        if masks_true is not None and masks_score is not None:
            results["pixel_auroc"] = compute_pixel_auroc(masks_true, masks_score)
            results["pro"] = compute_pro(masks_true, masks_score)

        return results

    @staticmethod
    def _compute_all_split(
        y_true: np.ndarray[Any, Any],
        y_score: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any] | None = None,
        masks_true: np.ndarray[Any, Any] | None = None,
        masks_score: np.ndarray[Any, Any] | None = None,
        *,
        val_frac: float = 0.2,
        test_frac: float = 0.4,
        random_state: int = 0,
    ) -> dict[str, float]:
        """Leakage-free ``compute_all``: tune threshold on val, report on test.

        Threshold-dependent metrics (``f1``/``f1_max``, accuracy/precision/recall)
        are evaluated on a disjoint test split using a threshold tuned on
        validation. Pixel-level (mask-based) metrics are computed on the SAME
        test split when the mask arrays are per-sample aligned (leading dim equals
        the sample count); if they are not aligned to the samples, they fall back
        to the full mask input.
        """
        from omni_mercury_engine.evaluation.metrics import fit_threshold, split_three_way

        yt_all = _to_numpy(y_true).flatten().astype(int)
        ys_all = _to_numpy(y_score).flatten()
        n = len(yt_all)

        _, val_idx, test_idx = split_three_way(
            n, yt_all, val_frac=val_frac, test_frac=test_frac, random_state=random_state
        )
        feasible = (
            len(val_idx) > 0
            and len(test_idx) > 0
            and len(np.unique(yt_all[val_idx])) >= 2
            and len(np.unique(yt_all[test_idx])) >= 2
        )
        if not feasible:
            logger.warning(
                "compute_all(tune_on='val'): split infeasible for N=%d; falling "
                "back to in-sample metrics (optimistic upper bound).",
                n,
            )
            return AnomalyMetrics.compute_all(
                y_true, y_score, y_pred=y_pred, masks_true=masks_true, masks_score=masks_score
            )

        threshold = fit_threshold(yt_all[val_idx], ys_all[val_idx])
        yt, ys = yt_all[test_idx], ys_all[test_idx]
        y_pred_test = (ys >= threshold).astype(int)

        results: dict[str, float] = {}
        results["auroc"] = compute_auroc(yt, ys)
        results["auprc"] = compute_auprc(yt, ys)

        tp = ((y_pred_test == 1) & (yt == 1)).sum()
        fp = ((y_pred_test == 1) & (yt == 0)).sum()
        fn = ((y_pred_test == 0) & (yt == 1)).sum()
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        op_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # This is the operating-point F1 at the val-tuned threshold on the test
        # split -- NOT an oracle max-over-thresholds (computing that on test
        # would re-introduce the threshold leakage this split exists to remove).
        # Exposed as "f1" (transparent name); "f1_max" is kept as a backward-compatible
        # alias for consumers of the in-sample API, with the same value.
        results["f1"] = op_f1
        results["f1_max"] = op_f1
        results["optimal_threshold"] = float(threshold)
        results["accuracy"] = float((y_pred_test == yt).mean())
        results["precision"] = precision
        results["recall"] = recall

        if masks_true is not None and masks_score is not None:
            # Keep pixel-level metrics on the SAME disjoint test split as the
            # sample-level metrics; scoring them on the full (train+val+test)
            # masks would leak optimistic pixels and break the transparent-split
            # contract. Convert to NumPy BEFORE applying the split index: the
            # mask args may be torch tensors, lists, or other arraylikes that
            # do not support NumPy advanced indexing with ``test_idx`` (a NumPy
            # int array). ``compute_pixel_auroc``/``compute_pro`` flatten via
            # ``_to_numpy`` anyway, so this only hoists the conversion. Index
            # only when the mask arrays are per-sample aligned (leading
            # dim == sample count).
            mt_np = _to_numpy(masks_true)
            ms_np = _to_numpy(masks_score)
            mt = mt_np[test_idx] if len(mt_np) == n else mt_np
            ms = ms_np[test_idx] if len(ms_np) == n else ms_np
            results["pixel_auroc"] = compute_pixel_auroc(mt, ms)
            results["pro"] = compute_pro(mt, ms)

        return results

    @staticmethod
    def compute_per_category(
        y_true: np.ndarray[Any, Any],
        y_score: np.ndarray[Any, Any],
        categories: list[str],
        *,
        tune_on: str = "in_sample",
    ) -> dict[str, dict[str, Any]]:
        """Compute metrics per category.

        Args:
            y_true: Ground truth labels [N]
            y_score: Anomaly scores [N]
            categories: Category for each sample [N]
            tune_on: Threshold-selection policy forwarded to
                :meth:`compute_all` per category (``"in_sample"`` or ``"val"``).

        Returns:
            Dict mapping category to metric dict
        """
        y_true = _to_numpy(y_true).flatten()
        y_score = _to_numpy(y_score).flatten()

        unique_categories = list(set(categories))
        results: dict[str, dict[str, Any]] = {}

        for cat in unique_categories:
            mask = np.array([c == cat for c in categories])
            if mask.sum() == 0:
                continue

            cat_true = y_true[mask]
            cat_score = y_score[mask]

            # Skip if all same class
            if len(np.unique(cat_true)) < 2:
                results[cat] = {"auroc": 0.5, "note": "single_class"}
                continue

            results[cat] = AnomalyMetrics.compute_all(cat_true, cat_score, tune_on=tune_on)

        return results
