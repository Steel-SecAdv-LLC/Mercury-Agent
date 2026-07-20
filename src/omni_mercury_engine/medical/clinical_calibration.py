# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply Mercury's probability calibrators to medical predictor scores.

Mercury already ships a conformal stack, a Venn-Abers predictor, point
calibrators (isotonic / Platt / Beta / temperature) and a Bayesian confidence
calibrator -- but, as flagged for Phase 2, *none were applied to medical
outputs*. This module closes that gap: it wraps each calibrator behind one
``fit`` / ``transform`` interface, fits it on a held-out calibration split of a
clinical score, and measures the reliability change (ECE / Brier / reliability
curve) plus a distribution-free conformal coverage report on the test split.

Calibrators wired here:

* ``venn_abers`` -> :class:`omni_mercury_engine.core.conformal_prediction.VennAbersCalibrator`
* ``isotonic`` / ``platt`` / ``beta`` / ``temperature`` ->
  :mod:`omni_mercury_engine.core.calibration`
* ``bayesian`` -> a Beta-Binomial histogram calibrator (:class:`BayesianBinningCalibrator`)
  built on the same Beta posterior-mean rule as
  :class:`~omni_mercury_engine.agentic.bayesian_calibrator.BayesianConfidenceCalibrator`.

Conformal *coverage* (the fraction of cases whose prediction set contains the
true label, guaranteed >= the target) is measured with the existing
:class:`~omni_mercury_engine.core.conformal_prediction.BinaryConformalClassifier`.

All calibration is *fail-safe*: a calibrator that cannot fit (e.g. a
single-class calibration split) degrades to identity and is reported as
``degraded`` rather than raising, so the harness never fabricates a calibrated
number it did not actually compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from omni_mercury_engine.medical.clinical_metrics import (
    ClinicalMetricReport,
    evaluate_clinical_scores,
)

__all__ = [
    "SUPPORTED_METHODS",
    "BayesianBinningCalibrator",
    "CalibrationComparison",
    "CalibratorProtocol",
    "calibrate_and_evaluate",
    "compare_calibrators",
    "conformal_coverage_report",
    "fit_calibrator",
]

#: Calibration methods this module can fit against a medical score.
SUPPORTED_METHODS: tuple[str, ...] = (
    "venn_abers",
    "isotonic",
    "platt",
    "beta",
    "temperature",
    "bayesian",
)


class CalibratorProtocol(Protocol):
    """Uniform calibrator surface: fit on (scores, labels), transform scores."""

    fitted: bool

    def fit(self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> Any:
        """Fit the calibrator on a calibration split."""
        ...

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map raw scores to calibrated probabilities in ``[0, 1]``."""
        ...


class BayesianBinningCalibrator:
    """Beta-Binomial histogram calibrator (a Bayesian calibrator for scores).

    Each equal-width score bin gets a Beta(``alpha``, ``beta``) prior over its
    true positive rate; the calibrated probability for a bin is the posterior
    mean ``(alpha + k) / (alpha + beta + n)`` where ``k`` positives fall in the
    bin out of ``n``. This is the histogram analogue of the familiarity-weighted
    Beta posterior used by
    :class:`~omni_mercury_engine.agentic.bayesian_calibrator.BayesianConfidenceCalibrator`:
    a sparsely-populated bin is shrunk toward the prior mean instead of trusting
    a noisy empirical rate. With the default uniform Beta(1, 1) prior an empty
    bin calibrates to 0.5 (maximum ignorance).

    Args:
        n_bins: Number of equal-width bins over ``[0, 1]``.
        alpha: Beta prior positive pseudo-count.
        beta: Beta prior negative pseudo-count.
    """

    def __init__(self, n_bins: int = 10, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Initialize the calibrator."""
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        self.n_bins = n_bins
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fitted = False
        self._bin_prob: np.ndarray[Any, Any] = np.full(n_bins, 0.5)
        self._edges: np.ndarray[Any, Any] = np.linspace(0.0, 1.0, n_bins + 1)

    def _bin_index(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map scores in ``[0, 1]`` to bin indices in ``[0, n_bins)``."""
        idx = np.clip(np.floor(scores * self.n_bins).astype(int), 0, self.n_bins - 1)
        return idx

    def fit(
        self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
    ) -> BayesianBinningCalibrator:
        """Fit per-bin Beta posteriors on a calibration split.

        Args:
            scores: Calibration scores (clipped to ``[0, 1]``).
            labels: Binary calibration labels.

        Returns:
            Self for chaining.
        """
        p = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
        y = np.asarray(labels, dtype=float).ravel()
        idx = self._bin_index(p)
        for b in range(self.n_bins):
            mask = idx == b
            n = int(np.sum(mask))
            k = float(np.sum(y[mask])) if n else 0.0
            self._bin_prob[b] = (self.alpha + k) / (self.alpha + self.beta + n)
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map scores to their bin's posterior-mean probability."""
        p = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
        return self._bin_prob[self._bin_index(p)]


class _CoreCalibratorAdapter:
    """Adapt a :mod:`core.calibration` point calibrator to fit/transform."""

    def __init__(self, method: str) -> None:
        """Build the underlying calibrator for ``method``."""
        from omni_mercury_engine.core import calibration as core_cal

        self.method = method
        self.fitted = False
        if method == "isotonic":
            self._cal: Any = core_cal.IsotonicCalibration()
        elif method == "platt":
            self._cal = core_cal.PlattScaling()
        elif method == "beta":
            self._cal = core_cal.BetaCalibration()
        elif method == "temperature":
            self._cal = core_cal.TemperatureScaling()
        else:  # pragma: no cover - guarded by fit_calibrator
            raise ValueError(f"unsupported core calibrator: {method}")

    def fit(self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> Any:
        """Fit the wrapped calibrator (``y_prob``, ``y_true`` order)."""
        p = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
        y = np.asarray(labels, dtype=float).ravel()
        self._cal.fit(p, y)
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Calibrate scores, clipped back into ``[0, 1]``."""
        p = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
        return np.clip(np.asarray(self._cal.calibrate(p), dtype=float).ravel(), 0.0, 1.0)


class _VennAbersAdapter:
    """Adapt :class:`VennAbersCalibrator` to the fit/transform surface."""

    def __init__(self, seed: int = 42) -> None:
        """Build the underlying Venn-Abers predictor."""
        from omni_mercury_engine.core.conformal_prediction import VennAbersCalibrator

        self._cal = VennAbersCalibrator(seed=seed)
        self.fitted = False

    def fit(self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> Any:
        """Precompute Venn-Abers corners on the calibration split."""
        p = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
        y = np.asarray(labels, dtype=float).ravel()
        self._cal.fit(p, y)
        self.fitted = bool(getattr(self._cal, "_fitted", False))
        return self

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return merged Venn-Abers probabilities."""
        p = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
        return np.clip(np.asarray(self._cal.predict_proba(p), dtype=float).ravel(), 0.0, 1.0)


class _IdentityCalibrator:
    """Degraded fallback: pass scores through unchanged."""

    def __init__(self) -> None:
        """Initialize the identity calibrator (never 'fitted')."""
        self.fitted = False

    def fit(self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> Any:
        """No-op fit; identity is always available but never counts as fitted."""
        return self

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the (clipped) scores unchanged."""
        return np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)


def fit_calibrator(
    method: str,
    scores_cal: np.ndarray[Any, Any],
    labels_cal: np.ndarray[Any, Any],
    *,
    n_bins: int = 10,
    seed: int = 42,
) -> Any:
    """Fit a named calibrator, degrading to identity if it cannot fit.

    Args:
        method: One of :data:`SUPPORTED_METHODS`.
        scores_cal: Calibration-split scores.
        labels_cal: Calibration-split binary labels.
        n_bins: Bin count for the Bayesian histogram calibrator.
        seed: Seed for the Venn-Abers calibrator.

    Returns:
        A fitted calibrator exposing ``fit``/``transform``/``fitted``. When the
        underlying method cannot fit (e.g. single-class split), an identity
        calibrator with ``fitted == False`` is returned.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {SUPPORTED_METHODS}")
    scores = np.clip(np.asarray(scores_cal, dtype=float).ravel(), 0.0, 1.0)
    labels = np.asarray(labels_cal, dtype=float).ravel()
    if len(set(labels.tolist())) < 2:
        return _IdentityCalibrator()
    try:
        cal: Any
        if method == "venn_abers":
            cal = _VennAbersAdapter(seed=seed)
        elif method == "bayesian":
            cal = BayesianBinningCalibrator(n_bins=n_bins)
        else:
            cal = _CoreCalibratorAdapter(method)
        cal.fit(scores, labels)
        if not getattr(cal, "fitted", False):
            return _IdentityCalibrator()
        return cal
    except Exception:  # pragma: no cover - defensive: never crash the harness
        return _IdentityCalibrator()


def conformal_coverage_report(
    probs_cal: np.ndarray[Any, Any],
    labels_cal: np.ndarray[Any, Any],
    probs_test: np.ndarray[Any, Any],
    labels_test: np.ndarray[Any, Any],
    *,
    coverage: float = 0.9,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure distribution-free conformal coverage on the test split.

    Fits a :class:`BinaryConformalClassifier` (LAC / Mondrian, reusing the
    conformal stack's finite-sample quantile) on the calibration probabilities
    and reports empirical coverage, per-class coverage, mean set size and
    abstain/empty rates on the test split.

    Args:
        probs_cal: Calibrated probabilities on the calibration split.
        labels_cal: Calibration-split binary labels.
        probs_test: Calibrated probabilities on the test split.
        labels_test: Test-split binary labels.
        coverage: Target per-class coverage (e.g. 0.9).
        seed: Seed forwarded to the conformal predictors.

    Returns:
        The ``coverage_report`` mapping, or ``{"available": False, ...}`` when
        the calibration split is single-class (coverage undefined).
    """
    from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier

    y_cal = np.asarray(labels_cal, dtype=float).ravel()
    if len(set(y_cal.tolist())) < 2:
        return {"available": False, "reason": "single-class calibration split"}
    clf = BinaryConformalClassifier(coverage=coverage, seed=seed)
    clf.fit(np.asarray(probs_cal, dtype=float).ravel(), y_cal.astype(int))
    report = clf.coverage_report(
        np.asarray(probs_test, dtype=float).ravel(),
        np.asarray(labels_test, dtype=float).ravel().astype(int),
    )
    report["available"] = True
    return report


@dataclass
class CalibrationComparison:
    """Before/after reliability for one calibration method on a clinical score.

    Attributes:
        method: Calibration method name.
        fitted: Whether the calibrator actually fit (``False`` => identity).
        report_uncalibrated: Test-split metrics on the raw score.
        report_calibrated: Test-split metrics on the calibrated probability.
        ece_reduction: ``ece_before - ece_after`` (positive = improvement).
        brier_reduction: ``brier_before - brier_after``.
        coverage: Conformal coverage report on the calibrated probabilities.
    """

    method: str
    fitted: bool
    report_uncalibrated: ClinicalMetricReport
    report_calibrated: ClinicalMetricReport
    ece_reduction: float
    brier_reduction: float
    coverage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the comparison."""
        return {
            "method": self.method,
            "fitted": self.fitted,
            "report_uncalibrated": self.report_uncalibrated.to_dict(),
            "report_calibrated": self.report_calibrated.to_dict(),
            "ece_reduction": self.ece_reduction,
            "brier_reduction": self.brier_reduction,
            "coverage": self.coverage,
        }


def calibrate_and_evaluate(
    scores_cal: np.ndarray[Any, Any],
    labels_cal: np.ndarray[Any, Any],
    scores_test: np.ndarray[Any, Any],
    labels_test: np.ndarray[Any, Any],
    *,
    method: str,
    coverage: float = 0.9,
    n_bins: int = 10,
    seed: int = 0,
) -> CalibrationComparison:
    """Fit one calibrator and measure the reliability change on the test split.

    Args:
        scores_cal: Calibration-split raw scores.
        labels_cal: Calibration-split labels.
        scores_test: Test-split raw scores.
        labels_test: Test-split labels.
        method: Calibration method (:data:`SUPPORTED_METHODS`).
        coverage: Target conformal coverage.
        n_bins: Bin count for ECE/MCE and the Bayesian calibrator.
        seed: Seed for calibrator + bootstrap reproducibility.

    Returns:
        A populated :class:`CalibrationComparison`.
    """
    cal = fit_calibrator(method, scores_cal, labels_cal, n_bins=n_bins, seed=seed or 42)
    p_test_raw = np.clip(np.asarray(scores_test, dtype=float).ravel(), 0.0, 1.0)
    p_test_cal = cal.transform(scores_test)

    report_raw = evaluate_clinical_scores(labels_test, p_test_raw, n_bins=n_bins, seed=seed)
    report_cal = evaluate_clinical_scores(labels_test, p_test_cal, n_bins=n_bins, seed=seed)

    cov = conformal_coverage_report(
        cal.transform(scores_cal),
        labels_cal,
        p_test_cal,
        labels_test,
        coverage=coverage,
        seed=seed or 42,
    )
    report_cal.coverage = cov

    return CalibrationComparison(
        method=method,
        fitted=bool(getattr(cal, "fitted", False)),
        report_uncalibrated=report_raw,
        report_calibrated=report_cal,
        ece_reduction=float(report_raw.ece - report_cal.ece),
        brier_reduction=float(report_raw.brier - report_cal.brier),
        coverage=cov,
    )


def compare_calibrators(
    scores_cal: np.ndarray[Any, Any],
    labels_cal: np.ndarray[Any, Any],
    scores_test: np.ndarray[Any, Any],
    labels_test: np.ndarray[Any, Any],
    *,
    methods: tuple[str, ...] = SUPPORTED_METHODS,
    coverage: float = 0.9,
    n_bins: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Sweep every calibrator and pick the one with the lowest test ECE.

    Args:
        scores_cal: Calibration-split scores.
        labels_cal: Calibration-split labels.
        scores_test: Test-split scores.
        labels_test: Test-split labels.
        methods: Calibration methods to compare.
        coverage: Target conformal coverage.
        n_bins: Bin count for calibration metrics.
        seed: Seed for reproducibility.

    Returns:
        Mapping with ``best_method``, the per-method
        :class:`CalibrationComparison` dicts, and the uncalibrated baseline ECE.
    """
    comparisons: dict[str, CalibrationComparison] = {}
    for method in methods:
        comparisons[method] = calibrate_and_evaluate(
            scores_cal,
            labels_cal,
            scores_test,
            labels_test,
            method=method,
            coverage=coverage,
            n_bins=n_bins,
            seed=seed,
        )
    baseline_ece = next(iter(comparisons.values())).report_uncalibrated.ece
    fitted = {m: c for m, c in comparisons.items() if c.fitted}
    pool = fitted or comparisons
    best_method = min(pool, key=lambda m: pool[m].report_calibrated.ece)
    return {
        "best_method": best_method,
        "baseline_ece": baseline_ece,
        "best_ece": comparisons[best_method].report_calibrated.ece,
        "comparisons": {m: c.to_dict() for m, c in comparisons.items()},
    }
