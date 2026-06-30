# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The single confidence-calibration routing point.

Confidence numbers in Mercury were historically heuristic -- ``0.5 + |margin|``
in the decider, ``exp(-phi * total)`` in the uncertainty quantifier, dozens of
``min(0.95, 0.5 + k*len(...))`` across modules -- none calibrated against actual
accuracy. This module is the one place a raw score becomes a *calibrated*
probability, fit with an honest, cross-validated estimate of the map it actually
deploys, and reported with ECE/Brier so the number means what it says.

It is a thin, opinionated wrapper over the calibrators already in
:mod:`omni_mercury_engine.core.calibration` (Platt / isotonic / temperature via
:class:`CalibrationEnsemble`, plus the accept-gated monotone map). The contract
it adds on top is a **cross-validated accept-gate**:

1. **Measure the deployed map, not a proxy.** The map that ships is refit on
   *all* the data. Its calibration is estimated out-of-fold (k-fold
   cross-validation): every row gets a calibrated prediction from a map fit on
   the *other* folds, and ECE/Brier are computed on those out-of-fold (OOF)
   predictions. This is an honest estimate of the deployed map's generalization
   -- not the metrics of a one-off 70/30 holdout that is then thrown away (the
   previous behaviour, where the reported number described a map that was never
   shipped).
2. **Accept only on a *significant* improvement.** A point estimate of "Brier
   went down" is noise at small n. The gate bootstraps a one-sided confidence
   interval on the per-row Brier delta (calibrated minus raw) and accepts only
   when the whole interval sits below zero. A few-sample register can no longer
   accept (or reject) calibration essentially at random.
3. **Deterministic by default.** The seed defaults to a fixed value, so the
   verdict of this contract is reproducible run to run; an operator can still
   override it.

When the data cannot support an honest OOF estimate (too few samples, a single
class, or a single-sample minority class that cannot appear on both sides of a
fold) the routing point stays identity (raw scores passed through, flagged
uncalibrated). A confidence that cannot be *shown* to be better calibrated than
the raw score is never silently "calibrated".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from omni_mercury_engine.core.calibration import (
    CalibrationEnsemble,
    IsotonicCalibration,
    PlattScaling,
    StrictIsotonicCalibration,
    TemperatureScaling,
    compute_ece,
)

logger = logging.getLogger(__name__)

# Deterministic-by-default seed for the calibration contract. ``None`` would make
# the OOF split -- and therefore the accept/reject verdict -- change run to run,
# a quiet inconsistency for something whose whole purpose is "the number means
# what it says". Callers can still override per instance.
_DEFAULT_SEED = 0

# Minimum samples below which no honest cross-validated estimate is attempted.
_MIN_SAMPLES = 8
# Default number of CV folds (capped by the minority-class count so every fold
# carries both classes on each side).
_DEFAULT_FOLDS = 5


@dataclass
class ConfidenceReport:
    """Cross-validated calibration quality for a fitted :class:`CalibratedConfidence`.

    ``brier``/``ece`` are measured **out-of-fold** -- an honest estimate of the
    deployed (refit-on-all-data) map's generalization, not the metrics of a
    discarded holdout. When ``held_out`` is ``False`` no honest out-of-sample
    estimate was possible (too few samples, a single class, or a single-sample
    minority class); in that case ``accepted`` is always ``False`` and the
    routing point stays identity -- an in-sample fit is never accepted, since its
    no-regression gate would reward overfitting.
    """

    n: int
    method: str
    accepted: bool
    brier_raw: float
    brier_cal: float
    ece_raw: float
    ece_cal: float
    held_out: bool = True
    note: str = ""
    # How the metrics above were obtained:
    #   "cv_oof"       -- cross-validated out-of-fold (the honest default)
    #   "insufficient" -- no out-of-sample estimate possible (stayed identity)
    eval_protocol: str = "cv_oof"
    n_folds: int = 0
    # One-sided bootstrap CI on the per-row Brier delta (brier_cal - brier_raw).
    # A negative ``ci_high`` means the calibrated map is *significantly* better.
    brier_delta_ci_low: float = 0.0
    brier_delta_ci_high: float = 0.0
    accepted_significant: bool = False

    @property
    def brier_improvement(self) -> float:
        """Absolute Brier reduction (raw - calibrated); >0 is better."""
        return self.brier_raw - self.brier_cal

    @property
    def ece_improvement(self) -> float:
        """Absolute ECE reduction (raw - calibrated); >0 is better."""
        return self.ece_raw - self.ece_cal

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view for provenance/telemetry."""
        return {
            "n": self.n,
            "method": self.method,
            "accepted": self.accepted,
            "brier_raw": self.brier_raw,
            "brier_cal": self.brier_cal,
            "brier_improvement": self.brier_improvement,
            "ece_raw": self.ece_raw,
            "ece_cal": self.ece_cal,
            "ece_improvement": self.ece_improvement,
            "held_out": self.held_out,
            "note": self.note,
            "eval_protocol": self.eval_protocol,
            "n_folds": self.n_folds,
            "brier_delta_ci": [self.brier_delta_ci_low, self.brier_delta_ci_high],
            "accepted_significant": self.accepted_significant,
        }


_CALIBRATOR_FACTORIES = {
    "platt": PlattScaling,
    "isotonic": IsotonicCalibration,
    "strict_isotonic": StrictIsotonicCalibration,
    "temperature": TemperatureScaling,
    "auto": CalibrationEnsemble,
}


class CalibratedConfidence:
    """Route a raw score to a calibrated probability, fit with a CV accept-gate.

    Usage::

        cc = CalibratedConfidence(method="auto")
        report = cc.fit(scores, labels)        # scores in [0,1], labels in {0,1}
        p = cc.transform(new_scores)           # calibrated P(positive)
        cc.report.to_dict()                    # measured (out-of-fold) ECE/Brier

    Until ``fit`` is called -- or when the cross-validated fit does not
    *significantly* beat raw scores -- ``transform`` is the identity (clipped to
    ``[0, 1]``) and :attr:`is_calibrated` is ``False``, so callers can stay
    honest about whether a coverage/accuracy guarantee actually backs the number.
    """

    def __init__(
        self,
        method: str = "auto",
        *,
        eval_fraction: float = 0.3,
        ece_tol: float = 1e-3,
        seed: int | None = _DEFAULT_SEED,
        n_folds: int = _DEFAULT_FOLDS,
        accept_alpha: float = 0.05,
        n_bootstrap: int = 1000,
    ) -> None:
        """Initialize the routing point.

        Args:
            method: Calibrator family -- ``"auto"`` (Brier-selected ensemble),
                ``"platt"``, ``"isotonic"``, ``"strict_isotonic"`` or
                ``"temperature"``.
            eval_fraction: Retained for backward compatibility only. The contract
                now uses k-fold cross-validation, not a single holdout fraction;
                this argument is ignored.
            ece_tol: Slack allowed on the ECE no-regression check.
            seed: Seed for the CV split and the acceptance bootstrap. Defaults to
                a fixed value so the accept/reject verdict is reproducible.
            n_folds: Target number of CV folds (capped by the minority-class
                count so every fold carries both classes).
            accept_alpha: One-sided significance level for the bootstrap accept
                gate (default 0.05 -> 95% one-sided).
            n_bootstrap: Bootstrap resamples for the Brier-delta CI.
        """
        if method not in _CALIBRATOR_FACTORIES:
            raise ValueError(
                f"Unknown calibration method {method!r}; choose one of "
                f"{sorted(_CALIBRATOR_FACTORIES)}"
            )
        self.method = method
        self.eval_fraction = float(eval_fraction)  # legacy, ignored (see docstring)
        self.ece_tol = float(ece_tol)
        self.seed = seed
        self.n_folds = int(n_folds)
        self.accept_alpha = float(accept_alpha)
        self.n_bootstrap = int(n_bootstrap)
        self._rng = np.random.default_rng(seed)
        self._calibrator: Any = None
        self._accepted: bool = False
        self._report: ConfidenceReport | None = None

    # -- properties --------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """Whether a fitted, accepted calibrator backs ``transform``."""
        return self._accepted and self._calibrator is not None

    @property
    def report(self) -> ConfidenceReport | None:
        """The most recent :class:`ConfidenceReport`, or ``None`` before fit."""
        return self._report

    # -- internals ---------------------------------------------------------

    def _new_calibrator(self) -> Any:
        factory = _CALIBRATOR_FACTORIES[self.method]
        # The ``auto`` ensemble holds its own RNG for the internal model-selection
        # split; seed it so the OOF predictions -- and therefore the accept/reject
        # verdict -- are reproducible. The parametric calibrators are deterministic
        # given their data and take no seed.
        if factory is CalibrationEnsemble:
            return factory(seed=self.seed if self.seed is not None else _DEFAULT_SEED)
        return factory()

    def _effective_folds(self, y: np.ndarray[Any, Any]) -> int:
        """Number of CV folds that keeps both classes on each side of every fold.

        Round-robin stratified folding gives each fold ~``min_class / k`` of the
        minority class; capping ``k`` at the minority count guarantees at least
        one minority sample per fold. ``k < 2`` means no honest out-of-sample
        split exists (e.g. a single-sample minority class).
        """
        _, counts = np.unique(y, return_counts=True)
        min_class = int(counts.min())
        return int(min(self.n_folds, min_class))

    def _oof_predictions(
        self, s: np.ndarray[Any, Any], y: np.ndarray[Any, Any], k: int
    ) -> np.ndarray[Any, Any] | None:
        """Cross-validated out-of-fold calibrated predictions, or ``None``.

        Each row's prediction comes from a calibrator fit on the *other* folds,
        so the resulting ECE/Brier estimate the deployed (refit-on-all) map
        without the optimism of in-sample evaluation. Returns ``None`` if any
        training fold is single-class (the calibrator cannot fit).
        """
        from omni_mercury_engine.ml.mercury_ml import StratifiedKFold

        seed = self.seed if self.seed is not None else _DEFAULT_SEED
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        p_oof = np.full(s.shape[0], np.nan, dtype=float)
        for train_idx, test_idx in skf.split(s.reshape(-1, 1), y):
            if len(np.unique(y[train_idx])) < 2 or test_idx.size == 0:
                return None
            cal = self._new_calibrator()
            cal.fit(s[train_idx], y[train_idx])
            pred = np.asarray(cal.calibrate(s[test_idx]), dtype=float).reshape(-1)
            p_oof[test_idx] = np.clip(pred, 0.0, 1.0)
        if np.isnan(p_oof).any():
            return None
        return p_oof

    def _brier_delta_ci(
        self, s: np.ndarray[Any, Any], y: np.ndarray[Any, Any], p_oof: np.ndarray[Any, Any]
    ) -> tuple[float, float]:
        """One-sided bootstrap CI on the per-row Brier delta (cal - raw).

        The per-row squared-error delta ``(p_oof - y)^2 - (s - y)^2`` has mean
        equal to ``brier_cal - brier_raw``. Bootstrapping its mean gives a CI on
        the improvement; we return ``(low, high)`` at ``accept_alpha`` (two-sided
        bounds, of which ``high`` drives the one-sided accept test).
        """
        delta = (p_oof - y) ** 2 - (s - y) ** 2
        n = delta.shape[0]
        boot = self._rng.integers(0, n, size=(self.n_bootstrap, n))
        means = delta[boot].mean(axis=1)
        lo = float(np.quantile(means, self.accept_alpha))
        hi = float(np.quantile(means, 1.0 - self.accept_alpha))
        return lo, hi

    def _identity_report(
        self,
        *,
        n: int,
        s: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        note: str,
        eval_protocol: str,
    ) -> ConfidenceReport:
        """Stay identity (uncalibrated) and report the raw, in-sample metrics."""
        from omni_mercury_engine.ml.mercury_ml import brier_score_loss

        measurable = bool(n and len(np.unique(y)) > 1)
        brier_raw = float(brier_score_loss(y, s)) if measurable else 0.0
        ece_raw = compute_ece(y, s) if n else 0.0
        self._calibrator = None
        self._accepted = False
        self._report = ConfidenceReport(
            n=n,
            method=self.method,
            accepted=False,
            brier_raw=brier_raw,
            brier_cal=brier_raw,
            ece_raw=ece_raw,
            ece_cal=ece_raw,
            held_out=False,
            note=note,
            eval_protocol=eval_protocol,
            n_folds=0,
        )
        return self._report

    # -- fit / transform ---------------------------------------------------

    def fit(self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> ConfidenceReport:
        """Fit with a cross-validated accept-gate and decide acceptance.

        Args:
            scores: Uncalibrated probability-like scores in ``[0, 1]``.
            labels: Binary ground-truth labels in ``{0, 1}``.

        Returns:
            A :class:`ConfidenceReport` with measured (out-of-fold) raw-vs-
            calibrated ECE/Brier, a bootstrap CI on the Brier improvement, and
            the accept decision.
        """
        s = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(labels, dtype=float).reshape(-1)
        n = s.size

        from omni_mercury_engine.ml.mercury_ml import brier_score_loss

        # Degenerate data: cannot calibrate, stay identity (honest).
        if n < _MIN_SAMPLES or len(np.unique(y)) < 2 or s.size != y.size:
            return self._identity_report(
                n=n,
                s=s,
                y=y,
                note="insufficient or single-class data; staying uncalibrated (identity)",
                eval_protocol="insufficient",
            )

        # A genuine out-of-sample estimate is REQUIRED for acceptance. If the data
        # is too imbalanced to fold (a single-sample minority class cannot appear
        # on both sides of a split) we stay identity and say so -- an in-sample
        # fit would let the no-regression gate reward overfitting (module
        # contract).
        k = self._effective_folds(y)
        if k < 2:
            report = self._identity_report(
                n=n,
                s=s,
                y=y,
                note=(
                    "data too imbalanced for a held-out split (single-sample "
                    "class); staying uncalibrated (identity) -- never accept an "
                    "in-sample fit"
                ),
                eval_protocol="insufficient",
            )
            logger.info(
                "CalibratedConfidence[%s] fit on n=%d: no out-of-sample split possible, "
                "staying identity (Brier %.4f, ECE %.4f)",
                self.method,
                n,
                report.brier_raw,
                report.ece_raw,
            )
            return report

        p_oof = self._oof_predictions(s, y, k)
        if p_oof is None:
            return self._identity_report(
                n=n,
                s=s,
                y=y,
                note=(
                    "a cross-validation fold was single-class; staying "
                    "uncalibrated (identity) -- never accept an in-sample fit"
                ),
                eval_protocol="insufficient",
            )

        brier_raw = float(brier_score_loss(y, s))
        brier_cal = float(brier_score_loss(y, p_oof))
        ece_raw = compute_ece(y, s)
        ece_cal = compute_ece(y, p_oof)
        ci_low, ci_high = self._brier_delta_ci(s, y, p_oof)

        # Accept only on a SIGNIFICANT out-of-fold improvement: the whole
        # one-sided Brier-delta CI sits below zero (calibrated beats raw beyond
        # sampling noise) AND ECE does not regress past tolerance. A point Brier
        # tie/win at small n is no longer enough -- the bootstrap CI is wide there
        # and the gate abstains rather than accepting noise.
        significant = bool(ci_high < 0.0)
        accepted = bool(significant and ece_cal <= ece_raw + self.ece_tol)

        if accepted:
            # Refit on all data for deployment. The OOF metrics above already
            # estimate THIS map's generalization, so the number we report
            # describes the map we actually ship.
            deploy = self._new_calibrator()
            deploy.fit(s, y)
            self._calibrator = deploy
            note = (
                "calibrated (cross-validated OOF improvement significant; "
                "deployed map refit on all data)"
            )
        else:
            self._calibrator = None
            note = (
                "no significant cross-validated improvement "
                f"(Brier delta CI [{ci_low:.4f}, {ci_high:.4f}]); staying identity"
            )

        self._accepted = accepted
        self._report = ConfidenceReport(
            n=n,
            method=self.method,
            accepted=accepted,
            brier_raw=brier_raw,
            brier_cal=brier_cal if accepted else brier_raw,
            ece_raw=ece_raw,
            ece_cal=ece_cal if accepted else ece_raw,
            held_out=True,
            note=note,
            eval_protocol="cv_oof",
            n_folds=k,
            brier_delta_ci_low=ci_low,
            brier_delta_ci_high=ci_high,
            accepted_significant=significant,
        )
        logger.info(
            "CalibratedConfidence[%s] fit on n=%d (%d-fold OOF): accepted=%s, "
            "Brier %.4f->%.4f (delta CI hi %.4f), ECE %.4f->%.4f",
            self.method,
            n,
            k,
            accepted,
            brier_raw,
            self._report.brier_cal,
            ci_high,
            ece_raw,
            self._report.ece_cal,
        )
        return self._report

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map raw scores to calibrated probabilities (identity if uncalibrated)."""
        s = np.asarray(scores, dtype=float).reshape(-1)
        if not self.is_calibrated:
            return np.asarray(np.clip(s, 0.0, 1.0))
        calibrated = np.asarray(self._calibrator.calibrate(s), dtype=float).reshape(-1)
        return np.asarray(np.clip(calibrated, 0.0, 1.0))

    def transform_one(self, score: float) -> float:
        """Calibrate a single scalar score."""
        return float(self.transform(np.array([score]))[0])

    def __call__(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Alias for :meth:`transform`."""
        return self.transform(scores)


__all__ = ["CalibratedConfidence", "ConfidenceReport"]
