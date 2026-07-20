# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Clinical discrimination + calibration metrics for medical predictor scores.

The medical subsystem emits probability-like ``risk_score`` / sub-risk values
(:mod:`omni_mercury_engine.medical.cardiology`, ``critical_care``, ``abms``,
etc.), but until now nothing measured whether those scores actually *separate*
outcomes or whether their magnitude is *trustworthy*. This module supplies the
clinical measurement layer requested for Phase 2:

* **Discrimination** -- AUROC, AUPRC, sensitivity (recall), specificity, PPV,
  NPV, F1, balanced accuracy at an operating threshold.
* **Calibration** -- Brier score, Expected/Maximum Calibration Error (ECE/MCE)
  and a binned reliability curve.
* **Uncertainty** -- a bootstrap confidence interval on AUROC so a small cohort
  never reads as more certain than it is.

Every metric reuses Mercury's existing, audited primitives rather than a new
implementation:

* AUROC -> :func:`omni_mercury_engine.evaluation.metrics.compute_auc_roc`
* ECE / MCE -> :func:`omni_mercury_engine.core.calibration.compute_ece` /
  :func:`~omni_mercury_engine.core.calibration.compute_mce`
* AUPRC -> :func:`omni_mercury_engine.ml.mercury_ml.average_precision_score`

The module is pure ``numpy`` (a Mercury core dependency); it imports no medical
predictor, so it loads anywhere and stays a leaf in the import graph.

.. note::

   Threshold-free metrics (AUROC/AUPRC) are rank-based and run on the raw score.
   Probability-dependent metrics (Brier/ECE/MCE/reliability) require a score in
   ``[0, 1]``; scores are clipped to that range before those metrics, so a
   predictor whose magnitude drifts out of range is measured conservatively
   rather than crashing the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.core.calibration import compute_ece, compute_mce
from omni_mercury_engine.evaluation.metrics import compute_auc_roc

__all__ = [
    "ClinicalMetricReport",
    "ReliabilityBin",
    "bootstrap_auroc_ci",
    "confusion_at_threshold",
    "evaluate_clinical_scores",
    "npv",
    "ppv",
    "reliability_curve",
    "sensitivity",
    "specificity",
    "youden_threshold",
]


def _as_1d(name: str, values: Any) -> np.ndarray[Any, Any]:
    """Coerce ``values`` to a finite 1-D float array or raise ``ValueError``."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _binary_labels(values: Any) -> np.ndarray[Any, Any]:
    """Coerce ``values`` to a binary ``{0, 1}`` integer array or raise."""
    arr = _as_1d("y_true", values)
    uniq = set(np.unique(arr).tolist())
    if not uniq <= {0.0, 1.0}:
        raise ValueError(f"y_true must be binary 0/1, got labels {sorted(uniq)}")
    return arr.astype(int)


def _probabilities(values: Any) -> np.ndarray[Any, Any]:
    """Coerce scores to probabilities in ``[0, 1]`` (clipped, not rejected)."""
    return np.clip(_as_1d("y_score", values), 0.0, 1.0)


def confusion_at_threshold(
    y_true: Any, y_score: Any, threshold: float
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` for ``y_score >= threshold``.

    Args:
        y_true: Binary ground-truth labels (1 = positive/event).
        y_score: Predicted scores (higher = more likely positive).
        threshold: Decision threshold; a score ``>= threshold`` predicts positive.

    Returns:
        Tuple of counts ``(true_pos, false_pos, true_neg, false_neg)``.
    """
    y = _binary_labels(y_true)
    s = _as_1d("y_score", y_score)
    if y.shape != s.shape:
        raise ValueError(f"length mismatch: y_true {y.shape} vs y_score {s.shape}")
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return tp, fp, tn, fn


def sensitivity(y_true: Any, y_score: Any, threshold: float) -> float:
    """Sensitivity / recall ``TP / (TP + FN)`` at ``threshold`` (1.0 if no positives)."""
    tp, _fp, _tn, fn = confusion_at_threshold(y_true, y_score, threshold)
    denom = tp + fn
    return float(tp / denom) if denom else 1.0


def specificity(y_true: Any, y_score: Any, threshold: float) -> float:
    """Specificity ``TN / (TN + FP)`` at ``threshold`` (1.0 if no negatives)."""
    _tp, fp, tn, _fn = confusion_at_threshold(y_true, y_score, threshold)
    denom = tn + fp
    return float(tn / denom) if denom else 1.0


def ppv(y_true: Any, y_score: Any, threshold: float) -> float:
    """Positive predictive value ``TP / (TP + FP)`` (``nan`` if nothing flagged)."""
    tp, fp, _tn, _fn = confusion_at_threshold(y_true, y_score, threshold)
    denom = tp + fp
    return float(tp / denom) if denom else float("nan")


def npv(y_true: Any, y_score: Any, threshold: float) -> float:
    """Negative predictive value ``TN / (TN + FN)`` (``nan`` if nothing cleared)."""
    _tp, _fp, tn, fn = confusion_at_threshold(y_true, y_score, threshold)
    denom = tn + fn
    return float(tn / denom) if denom else float("nan")


def youden_threshold(y_true: Any, y_score: Any) -> float:
    """Return the score threshold maximising Youden's J (sensitivity + specificity - 1).

    Candidate thresholds are the sorted unique scores; the midpoint between the
    two scores that bracket the optimum is returned so the operating point is
    not glued to an observed value. Falls back to ``0.5`` when a threshold
    cannot be chosen (degenerate single-class input).

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores.

    Returns:
        The J-maximising decision threshold.
    """
    y = _binary_labels(y_true)
    s = _as_1d("y_score", y_score)
    if y.shape != s.shape:
        raise ValueError(f"length mismatch: y_true {y.shape} vs y_score {s.shape}")
    if len(set(y.tolist())) < 2:
        return 0.5
    order = np.argsort(-s)
    s_sorted = s[order]
    y_sorted = y[order]
    p = int(np.sum(y == 1))
    n = int(np.sum(y == 0))
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / p
    fpr = fp / n
    j = tpr - fpr
    best = int(np.argmax(j))
    # Threshold just below the best score admits exactly that prefix as positive.
    thr = float(s_sorted[best])
    if best + 1 < len(s_sorted):
        thr = (thr + float(s_sorted[best + 1])) / 2.0
    return thr


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability (calibration) curve.

    Attributes:
        lower: Inclusive lower probability edge of the bin.
        upper: Upper probability edge of the bin.
        count: Number of samples whose predicted probability fell in the bin.
        mean_confidence: Mean predicted probability of the samples in the bin.
        empirical_rate: Observed positive rate of the samples in the bin.
    """

    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the bin."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "empirical_rate": self.empirical_rate,
        }


def reliability_curve(y_true: Any, y_score: Any, n_bins: int = 10) -> list[ReliabilityBin]:
    """Compute a binned reliability curve (empirical rate vs mean confidence).

    Uses equal-width probability bins over ``[0, 1]``. Empty bins are omitted so
    the caller never plots a spurious point. The last bin is right-inclusive.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted probabilities (clipped to ``[0, 1]``).
        n_bins: Number of equal-width bins.

    Returns:
        The non-empty reliability bins, ordered by probability.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y = _binary_labels(y_true)
    p = _probabilities(y_score)
    if y.shape != p.shape:
        raise ValueError(f"length mismatch: y_true {y.shape} vs y_score {p.shape}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=float(lo),
                upper=float(hi),
                count=count,
                mean_confidence=float(np.mean(p[mask])),
                empirical_rate=float(np.mean(y[mask])),
            )
        )
    return bins


def bootstrap_auroc_ci(
    y_true: Any,
    y_score: Any,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for AUROC.

    Resamples ``(y_true, y_score)`` pairs with replacement ``n_boot`` times and
    returns the ``[alpha/2, 1 - alpha/2]`` percentiles of the AUROC distribution.
    Resamples that collapse to a single class are skipped (AUROC undefined). If
    fewer than 2% of resamples are usable the interval degrades to
    ``(0.5, 0.5)`` rather than reporting false precision.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores.
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided significance level (0.05 -> 95% CI).
        seed: Seed for the resampling RNG (reproducibility).

    Returns:
        ``(ci_low, ci_high)`` percentile bounds on AUROC.
    """
    y = _binary_labels(y_true)
    s = _as_1d("y_score", y_score)
    if y.shape != s.shape:
        raise ValueError(f"length mismatch: y_true {y.shape} vs y_score {s.shape}")
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y[idx]
        if len(set(yb.tolist())) < 2:
            continue
        aucs.append(compute_auc_roc(yb, s[idx]))
    if len(aucs) < max(1, n_boot // 50):
        return 0.5, 0.5
    lo = float(np.percentile(aucs, 100 * (alpha / 2)))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


@dataclass
class ClinicalMetricReport:
    """Discrimination + calibration measurement for one clinical score.

    Attributes:
        n: Number of scored cases.
        n_positive: Number of positive (event) cases.
        prevalence: Positive-class base rate ``n_positive / n``.
        threshold: Operating threshold used for the confusion-matrix metrics.
        auroc: Area under the ROC curve (rank discrimination).
        auroc_ci_low: Lower bootstrap CI bound on AUROC.
        auroc_ci_high: Upper bootstrap CI bound on AUROC.
        auprc: Area under the precision-recall curve (average precision).
        sensitivity: Recall / true-positive rate at ``threshold``.
        specificity: True-negative rate at ``threshold``.
        ppv: Positive predictive value at ``threshold``.
        npv: Negative predictive value at ``threshold``.
        f1: F1 score at ``threshold``.
        balanced_accuracy: Mean of sensitivity and specificity.
        brier: Brier score of the (clipped) probabilities.
        ece: Expected calibration error.
        mce: Maximum calibration error.
        reliability: Non-empty reliability-curve bins.
        coverage: Optional conformal coverage report (populated by the harness).
    """

    n: int
    n_positive: int
    prevalence: float
    threshold: float
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    auprc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    balanced_accuracy: float
    brier: float
    ece: float
    mce: float
    reliability: list[ReliabilityBin] = field(default_factory=list)
    coverage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the full report."""
        return {
            "n": self.n,
            "n_positive": self.n_positive,
            "prevalence": self.prevalence,
            "threshold": self.threshold,
            "auroc": self.auroc,
            "auroc_ci_low": self.auroc_ci_low,
            "auroc_ci_high": self.auroc_ci_high,
            "auprc": self.auprc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "brier": self.brier,
            "ece": self.ece,
            "mce": self.mce,
            "reliability": [b.to_dict() for b in self.reliability],
            "coverage": self.coverage,
        }


def _auprc(y: np.ndarray[Any, Any], s: np.ndarray[Any, Any]) -> float:
    """Average precision (AUPRC), reusing mercury_ml with a numpy fallback."""
    if len(set(y.tolist())) < 2:
        return float(np.mean(y))
    try:
        from omni_mercury_engine.ml.mercury_ml import average_precision_score

        return float(average_precision_score(y, s))
    except Exception:  # pragma: no cover - defensive fallback
        order = np.argsort(-s)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(int(np.sum(y == 1)), 1)
        rec_prev = np.concatenate([[0.0], recall[:-1]])
        return float(np.sum((recall - rec_prev) * precision))


def evaluate_clinical_scores(
    y_true: Any,
    y_score: Any,
    *,
    threshold: float | None = None,
    n_bins: int = 10,
    bootstrap: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> ClinicalMetricReport:
    """Measure discrimination + calibration of a clinical score against outcomes.

    Args:
        y_true: Binary ground-truth outcomes (1 = event).
        y_score: Predicted scores/probabilities (higher = more likely event).
        threshold: Operating threshold for confusion-matrix metrics. When
            ``None`` the Youden-J-optimal threshold is chosen from the data.
        n_bins: Bin count for ECE/MCE and the reliability curve.
        bootstrap: Whether to compute a bootstrap AUROC confidence interval.
        n_boot: Bootstrap resample count (used only when ``bootstrap``).
        seed: Seed for the bootstrap RNG.

    Returns:
        A fully-populated :class:`ClinicalMetricReport` (``coverage`` unset).
    """
    y = _binary_labels(y_true)
    s = _as_1d("y_score", y_score)
    if y.shape != s.shape:
        raise ValueError(f"length mismatch: y_true {y.shape} vs y_score {s.shape}")
    p = np.clip(s, 0.0, 1.0)

    thr = youden_threshold(y, s) if threshold is None else float(threshold)
    tp, fp, tn, fn = confusion_at_threshold(y, s, thr)
    sens = float(tp / (tp + fn)) if (tp + fn) else 1.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 1.0
    prec = float(tp / (tp + fp)) if (tp + fp) else float("nan")
    npv_v = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    f1 = (
        float(2 * prec * sens / (prec + sens))
        if (not np.isnan(prec)) and (prec + sens) > 0
        else 0.0
    )

    auroc = compute_auc_roc(y, s)
    if bootstrap:
        ci_low, ci_high = bootstrap_auroc_ci(y, s, n_boot=n_boot, seed=seed)
    else:
        ci_low, ci_high = auroc, auroc

    brier = float(np.mean((p - y) ** 2))

    return ClinicalMetricReport(
        n=len(y),
        n_positive=int(np.sum(y == 1)),
        prevalence=float(np.mean(y)),
        threshold=thr,
        auroc=auroc,
        auroc_ci_low=ci_low,
        auroc_ci_high=ci_high,
        auprc=_auprc(y, s),
        sensitivity=sens,
        specificity=spec,
        ppv=prec,
        npv=npv_v,
        f1=f1,
        balanced_accuracy=float((sens + spec) / 2.0),
        brier=brier,
        ece=float(compute_ece(y, p, n_bins)),
        mce=float(compute_mce(y, p, n_bins)),
        reliability=reliability_curve(y, p, n_bins),
    )
