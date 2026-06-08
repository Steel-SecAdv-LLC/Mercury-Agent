"""Decision-curve analysis (Vickers & Elkin 2006) + single operating-point pathway.

Mercury Agent - Decision Curve Analysis (Stage 3, R2).
Copyright (C) 2025 Steel Security Advisors LLC.

Decision-curve analysis (Vickers & Elkin 2006) on calibrated probabilities, plus
the single, explicit operating-point pathway for the governed fusion substrate.

Net benefit at threshold ``t`` (treat iff ``p >= t``):

    NB(t) = TP/n - FP/n * t/(1 - t)

with the ``treat-all`` envelope ``prevalence - (1 - prevalence) * t/(1 - t)`` and
the ``treat-none`` envelope ``0``.  ``t/(1 - t)`` is the cost exchange rate, so a
threshold *is* a cost ratio: the cost-driven **Bayes threshold** is

    t* = c / (c + b)

where ``c`` is the harm of a false positive and ``b`` the benefit of a true
positive (equivalently, ``t*/(1 - t*) = c / b``).

**Single operating-point pathway (reconciliation with Item 4).**  There is ONE
shipped operating point: the MCA-calibrated probability thresholded at the
cost-driven Bayes ``t*``.  The conformal / Venn-Abers layer is a distribution-free
*coverage guarantee* (a recall floor) layered on top -- **not** a second,
competing threshold.  ``reconciled_operating_point`` returns this single decision
together with the conformal recall-floor diagnostic, so the two never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def bayes_threshold(cost_fp: float, benefit_tp: float) -> float:
    """Cost-driven Bayes operating threshold ``t* = c / (c + b)``.

    Args:
        cost_fp: harm of a false positive (``c``).
        benefit_tp: benefit of a true positive (``b``); for the
            missed-detection-catastrophic regime, ``b >> c`` -> small ``t*``.
    """
    denom = cost_fp + benefit_tp
    if denom <= 0:
        return 0.5
    return float(cost_fp / denom)


def net_benefit(y_true: np.ndarray[Any, Any], y_prob: np.ndarray[Any, Any], t: float) -> float:
    """Net benefit of treating ``p >= t`` at threshold ``t`` in (0, 1)."""
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    n = len(y)
    if n == 0 or not (0.0 < t < 1.0):
        return float("nan")
    treat = p >= t
    tp = float(np.sum(treat & (y == 1))) / n
    fp = float(np.sum(treat & (y == 0))) / n
    return float(tp - fp * (t / (1.0 - t)))


def low_threshold_prior(thresholds: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Per-domain threshold prior ``pi(t) ~ 1/t`` (low t up-weighted).

    Up-weights low thresholds heavily, encoding the missed-detection-catastrophic
    regime (a miss is far costlier than a false alarm).
    """
    t = np.asarray(thresholds, dtype=float).ravel()
    w = 1.0 / np.clip(t, 1e-9, None)
    prior: np.ndarray[Any, Any] = w / np.sum(w)
    return prior


@dataclass
class DecisionCurve:
    """A decision curve: net benefit of the model vs the treat-all/none envelopes."""

    thresholds: np.ndarray[Any, Any]
    model: np.ndarray[Any, Any]
    treat_all: np.ndarray[Any, Any]
    treat_none: np.ndarray[Any, Any]
    prevalence: float

    def prior_weighted_net_benefit(self, prior: np.ndarray[Any, Any] | None = None) -> float:
        """Threshold-prior integral of the model net benefit (low t up-weighted)."""
        pri = low_threshold_prior(self.thresholds) if prior is None else np.asarray(prior, float)
        pri = pri / np.clip(pri.sum(), 1e-12, None)
        return float(np.sum(pri * self.model))


def decision_curve(
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    thresholds: np.ndarray[Any, Any] | None = None,
) -> DecisionCurve:
    """Compute the decision curve (model NB + treat-all / treat-none envelopes)."""
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    ts = np.linspace(0.02, 0.98, 25) if thresholds is None else np.asarray(thresholds, float)
    prev = float(np.mean(y == 1)) if len(y) else float("nan")
    model = np.array([net_benefit(y, p, float(t)) for t in ts])
    odds = ts / np.clip(1.0 - ts, 1e-9, None)
    treat_all = prev - (1.0 - prev) * odds
    treat_none = np.zeros_like(ts)
    return DecisionCurve(ts, model, treat_all, treat_none, prev)


@dataclass
class OperatingPoint:
    """The single reconciled operating point + the conformal coverage diagnostic."""

    bayes_threshold: float
    decision: np.ndarray[Any, Any]  # p >= t* (the shipped verdict)
    net_benefit_at_t_star: float
    conformal_recall_floor: float | None  # coverage guarantee diagnostic, not a 2nd threshold


def reconciled_operating_point(
    y_prob: np.ndarray[Any, Any],
    cost_fp: float,
    benefit_tp: float,
    *,
    y_true: np.ndarray[Any, Any] | None = None,
    conformal_coverage: float | None = None,
) -> OperatingPoint:
    """The ONE operating-point pathway: MCA prob -> cost-driven Bayes ``t*``.

    The decision is ``p >= t*``.  ``conformal_coverage`` (if given) is reported as
    a *recall-floor diagnostic* -- the distribution-free coverage guarantee that
    layers on top of, and never competes with, the Bayes threshold.
    """
    p = np.asarray(y_prob, dtype=float).ravel()
    t_star = bayes_threshold(cost_fp, benefit_tp)
    decision = p >= t_star
    nb = net_benefit(y_true, p, t_star) if y_true is not None else float("nan")
    return OperatingPoint(
        bayes_threshold=t_star,
        decision=decision,
        net_benefit_at_t_star=nb,
        conformal_recall_floor=conformal_coverage,
    )
