# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The single confidence-calibration routing point.

Confidence numbers in Mercury were historically heuristic -- ``0.5 + |margin|``
in the decider, ``exp(-phi * total)`` in the uncertainty quantifier, dozens of
``min(0.95, 0.5 + k*len(...))`` across modules -- none calibrated against actual
accuracy. This module is the one place a raw score becomes a *calibrated*
probability, fit on a held-out split and reported with ECE/Brier so the number
means what it says.

It is a thin, opinionated wrapper over the calibrators already in
:mod:`omni_mercury_engine.core.calibration` (Platt / isotonic / temperature via
:class:`CalibrationEnsemble`, plus the accept-gated monotone map). The key
contract it adds on top is the **R4 accept-gate**: the fitted map is used only
if it does not regress Brier *and* ECE on a held-out evaluation split; otherwise
the routing point falls back to identity (raw scores passed through, flagged
uncalibrated). A confidence that cannot be shown to be better calibrated than
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


@dataclass
class ConfidenceReport:
    """Held-out calibration quality for a fitted :class:`CalibratedConfidence`.

    ``brier``/``ece`` are measured on a held-out evaluation split (or in-sample,
    flagged via ``held_out=False``, when the data was too small to split).
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
        }


_CALIBRATOR_FACTORIES = {
    "platt": PlattScaling,
    "isotonic": IsotonicCalibration,
    "strict_isotonic": StrictIsotonicCalibration,
    "temperature": TemperatureScaling,
    "auto": CalibrationEnsemble,
}


class CalibratedConfidence:
    """Route a raw score to a calibrated probability, fit on a held-out split.

    Usage::

        cc = CalibratedConfidence(method="auto")
        report = cc.fit(scores, labels)        # scores in [0,1], labels in {0,1}
        p = cc.transform(new_scores)           # calibrated P(positive)
        cc.report.to_dict()                    # measured ECE/Brier

    Until ``fit`` is called -- or when the fit does not beat raw scores on the
    held-out split -- ``transform`` is the identity (clipped to ``[0, 1]``) and
    :attr:`is_calibrated` is ``False``, so callers can stay honest about whether
    a coverage/accuracy guarantee actually backs the number.
    """

    def __init__(
        self,
        method: str = "auto",
        *,
        eval_fraction: float = 0.3,
        ece_tol: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        """Initialize the routing point.

        Args:
            method: Calibrator family -- ``"auto"`` (Brier-selected ensemble),
                ``"platt"``, ``"isotonic"``, ``"strict_isotonic"`` or
                ``"temperature"``.
            eval_fraction: Fraction held out (stratified) to measure ECE/Brier
                and decide acceptance.
            ece_tol: Slack allowed on the ECE no-regression check.
            seed: Seed for the held-out split shuffle.
        """
        if method not in _CALIBRATOR_FACTORIES:
            raise ValueError(
                f"Unknown calibration method {method!r}; choose one of "
                f"{sorted(_CALIBRATOR_FACTORIES)}"
            )
        self.method = method
        self.eval_fraction = float(eval_fraction)
        self.ece_tol = float(ece_tol)
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

    # -- fit / transform ---------------------------------------------------

    def _stratified_split(
        self, n: int, y: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Stratified (fit, eval) index split keeping both classes in each side."""
        idx = np.arange(n)
        fit_parts, eval_parts = [], []
        for cls in np.unique(y):
            cls_idx = idx[y == cls]
            self._rng.shuffle(cls_idx)
            n_eval = int(round(len(cls_idx) * self.eval_fraction))
            n_eval = min(max(n_eval, 1), len(cls_idx) - 1)  # keep >=1 each side
            eval_parts.append(cls_idx[:n_eval])
            fit_parts.append(cls_idx[n_eval:])
        return np.concatenate(fit_parts), np.concatenate(eval_parts)

    def _new_calibrator(self) -> Any:
        return _CALIBRATOR_FACTORIES[self.method]()

    def fit(
        self, scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
    ) -> ConfidenceReport:
        """Fit the calibrator on a held-out split and decide acceptance.

        Args:
            scores: Uncalibrated probability-like scores in ``[0, 1]``.
            labels: Binary ground-truth labels in ``{0, 1}``.

        Returns:
            A :class:`ConfidenceReport` with measured raw-vs-calibrated
            ECE/Brier and the accept decision.
        """
        s = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(labels, dtype=float).reshape(-1)
        n = s.size

        from omni_mercury_engine.ml.mercury_ml import brier_score_loss

        # Degenerate data: cannot calibrate, stay identity (honest).
        if n < 8 or len(np.unique(y)) < 2 or s.size != y.size:
            self._calibrator = None
            self._accepted = False
            self._report = ConfidenceReport(
                n=n,
                method=self.method,
                accepted=False,
                brier_raw=float(brier_score_loss(y, s)) if n and len(np.unique(y)) > 1 else 0.0,
                brier_cal=float(brier_score_loss(y, s)) if n and len(np.unique(y)) > 1 else 0.0,
                ece_raw=compute_ece(y, s) if n else 0.0,
                ece_cal=compute_ece(y, s) if n else 0.0,
                held_out=False,
                note="insufficient or single-class data; staying uncalibrated (identity)",
            )
            return self._report

        # Try a held-out evaluation; fall back to in-sample if the split would
        # leave a side single-class.
        fit_idx, eval_idx = self._stratified_split(n, y)
        held_out = bool(
            len(eval_idx) > 0
            and len(np.unique(y[eval_idx])) >= 2
            and len(np.unique(y[fit_idx])) >= 2
        )
        if not held_out:
            fit_idx = np.arange(n)
            eval_idx = np.arange(n)

        cal = self._new_calibrator()
        cal.fit(s[fit_idx], y[fit_idx])

        s_eval, y_eval = s[eval_idx], y[eval_idx]
        p_eval = np.asarray(cal.calibrate(s_eval), dtype=float).reshape(-1)

        brier_raw = float(brier_score_loss(y_eval, s_eval))
        brier_cal = float(brier_score_loss(y_eval, p_eval))
        ece_raw = compute_ece(y_eval, s_eval)
        ece_cal = compute_ece(y_eval, p_eval)

        accepted = bool(brier_cal <= brier_raw + 1e-12 and ece_cal <= ece_raw + self.ece_tol)

        if accepted:
            # Refit on all data for deployment now that the map is trusted.
            deploy = self._new_calibrator()
            deploy.fit(s, y)
            self._calibrator = deploy
            note = "calibrated (accepted on held-out split)" if held_out else (
                "calibrated (in-sample; data too small to hold out)"
            )
        else:
            self._calibrator = None
            note = "calibration regressed Brier/ECE on held-out split; staying identity"

        self._accepted = accepted
        self._report = ConfidenceReport(
            n=n,
            method=self.method,
            accepted=accepted,
            brier_raw=brier_raw,
            brier_cal=brier_cal if accepted else brier_raw,
            ece_raw=ece_raw,
            ece_cal=ece_cal if accepted else ece_raw,
            held_out=held_out,
            note=note,
        )
        logger.info(
            "CalibratedConfidence[%s] fit on n=%d: accepted=%s, Brier %.4f->%.4f, ECE %.4f->%.4f",
            self.method, n, accepted, brier_raw, self._report.brier_cal, ece_raw, self._report.ece_cal,
        )
        return self._report

    def transform(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map raw scores to calibrated probabilities (identity if uncalibrated)."""
        s = np.asarray(scores, dtype=float).reshape(-1)
        if not self.is_calibrated:
            return np.clip(s, 0.0, 1.0)
        return np.clip(np.asarray(self._calibrator.calibrate(s), dtype=float).reshape(-1), 0.0, 1.0)

    def transform_one(self, score: float) -> float:
        """Calibrate a single scalar score."""
        return float(self.transform(np.array([score]))[0])

    def __call__(self, scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Alias for :meth:`transform`."""
        return self.transform(scores)


__all__ = ["CalibratedConfidence", "ConfidenceReport"]
