# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sequential hypothesis tests as streaming detectors: Wald's SPRT and Page-Hinkley.

Two classical sequential tests exposed through the
:class:`~omni_mercury_engine.core.base.BaseDetector` contract:

* :class:`SPRTDetector` -- Wald's Sequential Probability Ratio Test (Wald 1945,
  "Sequential Tests of Statistical Hypotheses", Annals of Mathematical
  Statistics 16(2):117-186) in its Gaussian mean-shift form.  The
  log-likelihood ratio of ``H1: N(mu1, sigma^2)`` against ``H0: N(mu0,
  sigma^2)`` accumulates per sample and is compared against the exact Wald
  boundaries ``A = (1 - beta) / alpha`` and ``B = beta / (1 - alpha)`` (in log
  form).  ``fit`` estimates ``mu0``/``sigma`` from reference data; ``detect``
  runs the SPRT over the series, restarting after each decision, and scores
  samples with a bounded monotone transform of the running LLR that equals
  0.5 exactly at the H1 boundary, so the default operating point flags
  precisely the samples where the test decides H1.

* :class:`PageHinkleyDetector` -- the Page-Hinkley test (Page 1954,
  "Continuous Inspection Schemes", Biometrika 41(1/2):100-115):
  ``m_t = sum_{i<=t}(x_i - x_bar - delta)`` and
  ``PH_t = m_t - min_{s<=t} m_s``, alarming when ``PH_t > lambda``, with both
  one-sided variants (upward and downward mean shifts).  ``fit`` estimates the
  reference mean ``x_bar``; ``delta`` (the tolerance) and ``lambda`` (the
  alarm threshold) are caller-configured operating values.

Both detectors depend only on NumPy, keep their learned state as plain floats
(so the shared ``BaseDetector.save``/``load`` persistence applies unchanged),
and are registered as opt-in BASE detectors on the ``series_1d`` modality.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors._calibration import (
    bound_finite_config,
    finite_features,
    finite_scores,
)
from omni_mercury_engine.detectors.provenance import DecisionSource

if TYPE_CHECKING:
    import torch

__all__ = ["PageHinkleyDetector", "SPRTDetector"]

#: ``ln 2`` -- places the Page-Hinkley score exactly at 0.5 when ``PH = lambda``.
_LN2 = float(np.log(2.0))


def _to_1d_f64(detector: Any, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
    """Coerce numpy/torch input to a finite 1-D float64 series (NaN policy applied)."""
    detach = getattr(data, "detach", None)
    if callable(detach):
        data = detach().cpu().numpy()
    return bound_finite_config(detector, np.asarray(data, dtype=np.float64)).ravel()


def _page_hinkley_scan(
    deviations: np.ndarray[Any, Any], delta: float, lambda_threshold: float
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    """Both one-sided Page-Hinkley trajectories, restarted after each alarm.

    ``m_t = sum(x_i - delta)`` with ``PH_t = m_t - min_{s<=t} m_s`` grows
    monotonically once a shift is sustained, so without a restart one change
    point alarms at that sample *and at every sample after it*: a cumulative
    test with no reset reports the same change for the rest of the series.  The
    Page-Hinkley change detector therefore clears its accumulators on the
    sample following an alarm, the same discipline
    :class:`SPRTDetector` already applies after each Wald decision.

    Because the reference mean is fixed by ``fit`` rather than tracked, a shift
    that persists re-alarms roughly every ``lambda / (shift - delta)`` samples
    instead of exactly once: a fixed-reference CUSUM reports that the series is
    still away from the reference it was fitted on.

    Args:
        deviations: Series minus the reference mean, in reference standard
            deviations.
        delta: Per-sample tolerance.
        lambda_threshold: Alarm threshold on the PH statistic.

    Returns:
        ``(m_increase, ph_increase, m_decrease, ph_decrease, ph)``, each of the
        same length as ``deviations``; ``ph`` is the per-sample maximum of the
        two one-sided statistics.
    """
    n = deviations.size
    m_inc = np.empty(n, dtype=np.float64)
    m_dec = np.empty(n, dtype=np.float64)
    ph_inc = np.empty(n, dtype=np.float64)
    ph_dec = np.empty(n, dtype=np.float64)
    run_inc = run_dec = 0.0
    # The running minima start unbounded so that, exactly as
    # ``np.minimum.accumulate`` does over a whole series, the first sample of
    # every run is its own minimum and PH is 0 there.
    min_inc = min_dec = math.inf
    restart = False
    for i in range(n):
        if restart:
            run_inc = run_dec = 0.0
            min_inc = min_dec = math.inf
            restart = False
        deviation = float(deviations[i])
        run_inc += deviation - delta
        run_dec += -deviation - delta
        min_inc = min(min_inc, run_inc)
        min_dec = min(min_dec, run_dec)
        m_inc[i] = run_inc
        m_dec[i] = run_dec
        ph_inc[i] = run_inc - min_inc
        ph_dec[i] = run_dec - min_dec
        if max(ph_inc[i], ph_dec[i]) > lambda_threshold:
            restart = True
    return m_inc, ph_inc, m_dec, ph_dec, np.maximum(ph_inc, ph_dec)


class SPRTDetector(BaseDetector):
    """Wald Sequential Probability Ratio Test for a Gaussian mean shift.

    Tests ``H0: x ~ N(mu0, sigma^2)`` against ``H1: x ~ N(mu1, sigma^2)``
    sample by sample.  The per-sample log-likelihood-ratio increment is
    ``(mu1 - mu0) / sigma^2 * (x - (mu0 + mu1) / 2)``; the accumulated LLR is
    compared against the exact Wald boundaries ``log A = log((1-beta)/alpha)``
    (decide H1) and ``log B = log(beta/(1-alpha))`` (decide H0).  ``detect``
    restarts the test after each decision so a long series yields a sequence
    of decisions, and scores every sample with the logistic transform of the
    running LLR's distance to the H1 boundary, ``sigmoid(LLR - log A)`` -- a
    bounded monotone map equal to 0.5 exactly at ``log A``.  At the default
    threshold (0.5) the flagged samples are exactly those where the test
    decides H1; a score of 0.5 at LLR 0 would instead flag every sample with
    any evidence toward H1, far more often than ``alpha``.
    """

    decision_source = DecisionSource(
        kind="published",
        description=(
            "Decisions are made at Wald's exact SPRT boundaries A = (1-beta)/alpha "
            "and B = beta/(1-alpha) on the accumulated Gaussian mean-shift "
            "log-likelihood ratio; the error rates alpha/beta (and the effect size "
            "defining mu1) are caller-configured operating values, not published "
            "constants. The per-sample anomaly score is sigmoid(LLR - log A), which "
            "equals 0.5 exactly at the H1 boundary, so the default operating "
            "point (0.5) flags exactly the samples where the test decides H1."
        ),
        citation=(
            "Wald (1945), 'Sequential Tests of Statistical Hypotheses', "
            "Annals of Mathematical Statistics 16(2):117-186"
        ),
    )

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.05,
        effect_size: float = 1.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SPRT detector.

        Args:
            alpha: Type-I error bound (probability of deciding H1 under H0).
                Must satisfy ``0 < alpha < 1``.
            beta: Type-II error bound (probability of deciding H0 under H1).
                Must satisfy ``0 < beta < 1``.
            effect_size: Mean shift under H1, in reference standard deviations:
                ``mu1 = mu0 + effect_size * sigma``. Must be non-zero (a zero
                shift makes H0 and H1 identical and the LLR identically zero).
            config: Optional ``BaseDetector`` config.

        Raises:
            ValueError: If ``alpha``/``beta`` are outside ``(0, 1)``, their sum
                is >= 1 (the Wald boundaries would invert), or ``effect_size``
                is zero.
        """
        super().__init__(config)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if alpha + beta >= 1.0:
            raise ValueError(
                f"alpha + beta must be < 1 for the Wald boundaries to be ordered, "
                f"got alpha={alpha}, beta={beta}"
            )
        if effect_size == 0.0:
            raise ValueError("effect_size must be non-zero (H1 must differ from H0)")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.effect_size = float(effect_size)
        # Exact Wald boundaries in log form: decide H1 at/above log_a, H0 at/below log_b.
        self.log_boundary_h1 = math.log((1.0 - self.beta) / self.alpha)
        self.log_boundary_h0 = math.log(self.beta / (1.0 - self.alpha))
        # Reference-regime parameters; refined from data in ``fit``.
        self._mu0 = 0.0
        self._sigma = 1.0
        self._mu1 = self.effect_size

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SPRTDetector:
        """Estimate ``mu0`` and ``sigma`` from reference (normal-regime) data.

        Args:
            data: Reference series (flattened to 1-D). Its mean becomes
                ``mu0`` and its standard deviation ``sigma``; ``mu1`` is set
                to ``mu0 + effect_size * sigma``.

        Returns:
            ``self``.
        """
        series = _to_1d_f64(self, data)
        if series.size >= 2:
            self._mu0 = float(np.mean(series))
            self._sigma = max(float(np.std(series)), 1e-9)
            self._mu1 = self._mu0 + self.effect_size * self._sigma
        self._is_fitted = True
        return self

    def _llr_increments(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Per-sample Gaussian mean-shift log-likelihood-ratio increments."""
        slope = (self._mu1 - self._mu0) / (self._sigma**2)
        midpoint = 0.5 * (self._mu0 + self._mu1)
        return slope * (series - midpoint)

    def run_sprt(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Run one SPRT (no restart) over ``data`` until a decision or exhaustion.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            Dict with ``decision`` (``"h0"``/``"h1"``/``"continue"``), ``llr``
            (the LLR trajectory up to and including the deciding sample),
            ``n_samples`` (samples consumed), ``log_boundary_h1``,
            ``log_boundary_h0``, ``alpha`` and ``beta``.
        """
        series = _to_1d_f64(self, data)
        increments = self._llr_increments(series)
        llr = 0.0
        trajectory: list[float] = []
        decision = "continue"
        for value in increments:
            llr += float(value)
            trajectory.append(llr)
            if llr >= self.log_boundary_h1:
                decision = "h1"
                break
            if llr <= self.log_boundary_h0:
                decision = "h0"
                break
        return {
            "decision": decision,
            "llr": trajectory,
            "n_samples": len(trajectory),
            "log_boundary_h1": self.log_boundary_h1,
            "log_boundary_h0": self.log_boundary_h0,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def _restarting_sprt(
        self, series: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], list[dict[str, Any]]]:
        """SPRT with restart after each decision.

        Returns:
            ``(llr_trajectory, decisions)`` where ``llr_trajectory[t]`` is the
            running LLR of the active test after sample ``t`` and ``decisions``
            lists ``{"index", "decision", "llr"}`` for each boundary crossing.
        """
        increments = self._llr_increments(series)
        trajectory = np.zeros(series.size, dtype=np.float64)
        decisions: list[dict[str, Any]] = []
        llr = 0.0
        for t in range(series.size):
            llr += float(increments[t])
            trajectory[t] = llr
            if llr >= self.log_boundary_h1:
                decisions.append({"index": t, "decision": "h1", "llr": llr})
                llr = 0.0
            elif llr <= self.log_boundary_h0:
                decisions.append({"index": t, "decision": "h0", "llr": llr})
                llr = 0.0
        return trajectory, decisions

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: ``sigmoid(LLR - log A)`` on the running LLR.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 scores in ``(0, 1)``.
        """
        series = _to_1d_f64(self, data)
        trajectory, _ = self._restarting_sprt(series)
        scores = self._squash(trajectory - self.log_boundary_h1)
        return finite_features(scores, detector=self.name).reshape(-1, 1)

    @staticmethod
    def _squash(llr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Bounded, strictly increasing transform: the logistic function.

        Callers pass the LLR's offset from the H1 boundary, so 0 (the
        boundary itself) maps to 0.5.
        """
        out = np.empty_like(llr)
        pos = llr >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-llr[pos]))
        exp_neg = np.exp(llr[~pos])
        out[~pos] = exp_neg / (1.0 + exp_neg)
        return out

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Run the restarting SPRT over ``data`` and score every sample.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            Detection dict: ``scores`` are ``sigmoid(LLR - log A)`` on the
            running LLR (0.5 exactly at the H1 boundary), ``anomaly_score``
            their maximum, ``is_anomaly`` the
            per-sample threshold verdicts, and ``metadata`` carries the LLR
            trajectory, every boundary decision (with restart indices), the
            final segment's state (``"h0"``/``"h1"``/``"continue"``), the exact
            Wald log-boundaries, and ``alpha``/``beta``.
        """
        series = _to_1d_f64(self, data)
        trajectory, decisions = self._restarting_sprt(series)
        scores = finite_scores(
            self._squash(trajectory - self.log_boundary_h1), detector=self.name
        ).astype(np.float32)
        if decisions and decisions[-1]["index"] == series.size - 1:
            final_decision = str(decisions[-1]["decision"])
        else:
            final_decision = "continue"
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
            "metadata": {
                "llr": trajectory.tolist(),
                "decisions": decisions,
                "final_decision": final_decision,
                "log_boundary_h1": self.log_boundary_h1,
                "log_boundary_h0": self.log_boundary_h0,
                "alpha": self.alpha,
                "beta": self.beta,
                "mu0": self._mu0,
                "mu1": self._mu1,
                "sigma": self._sigma,
            },
        }


class PageHinkleyDetector(BaseDetector):
    """Page-Hinkley cumulative test for a sustained mean shift.

    Implements the scheme of Page (1954): with reference mean ``x_bar``
    (estimated by ``fit``) and tolerance ``delta``, the upward statistic is
    ``m_t = sum_{i<=t}(x_i - x_bar - delta)`` and
    ``PH_t = m_t - min_{s<=t} m_s``; an alarm is raised when
    ``PH_t > lambda``.  The downward variant mirrors it
    (``m_t = sum_{i<=t}(x_bar - x_i - delta)``), so both one-sided shifts are
    watched.  Once ``fit`` has seen the reference series, ``x_i`` is measured in
    units of its standard deviation, so ``delta`` and ``lambda`` mean the same
    thing for a series in any unit (an unfitted detector keeps the series'
    own unit).  Both accumulators restart on the sample after an alarm -- the
    same discipline :class:`SPRTDetector` applies after each Wald decision --
    so a sustained shift re-alarms roughly every ``lambda / (shift - delta)``
    samples against the fitted reference instead of at every sample from its
    onset onward.  The per-sample anomaly score is ``1 - 2^(-PH_t / lambda)`` on
    the larger of the two statistics -- a bounded monotone transform equal to
    0.5 exactly at the alarm threshold.
    """

    decision_source = DecisionSource(
        kind="published",
        description=(
            "The alarm rule is the Page-Hinkley cumulative test: PH_t = m_t - "
            "min_{s<=t} m_s with m_t = sum(x_i - x_bar - delta), alarming when "
            "PH_t exceeds lambda, run in both one-sided variants and restarted "
            "on the sample after each alarm; the tolerance "
            "delta and threshold lambda are caller-configured operating values, "
            "not published constants. Scores map PH_t through the bounded "
            "monotone transform 1 - 2^(-PH/lambda), which equals 0.5 exactly at "
            "the alarm boundary. After fit() the series is measured in units of "
            "the reference standard deviation, so the same delta and lambda "
            "apply whatever the series' unit."
        ),
        citation=("Page (1954), 'Continuous Inspection Schemes', Biometrika 41(1/2):100-115"),
    )

    def __init__(
        self,
        delta: float = 0.005,
        lambda_threshold: float = 50.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Page-Hinkley detector.

        Args:
            delta: Tolerance (magnitude of mean drift ignored per sample), in
                reference standard deviations once fitted. Must be >= 0.
            lambda_threshold: Alarm threshold ``lambda`` on the PH statistic,
                in reference standard deviations once fitted. Must be > 0.
            config: Optional ``BaseDetector`` config.

        Raises:
            ValueError: If ``delta`` is negative or ``lambda_threshold`` is
                not strictly positive.
        """
        super().__init__(config)
        if delta < 0.0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        if lambda_threshold <= 0.0:
            raise ValueError(f"lambda_threshold must be > 0, got {lambda_threshold}")
        self.delta = float(delta)
        self.lambda_threshold = float(lambda_threshold)
        # Reference mean and spread; refined from data in ``fit``.
        self._xbar = 0.0
        self._scale = 1.0

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> PageHinkleyDetector:
        """Estimate the reference mean ``x_bar`` and spread from normal-regime data.

        Args:
            data: Reference series (flattened to 1-D); its mean becomes
                ``x_bar`` and its standard deviation the unit ``delta`` and
                ``lambda`` are measured in (1 for a constant or one-sample
                series).

        Returns:
            ``self``.
        """
        series = _to_1d_f64(self, data)
        if series.size >= 1:
            self._xbar = float(np.mean(series))
            spread = float(np.std(series)) if series.size >= 2 else 0.0
            self._scale = spread if spread > 1e-12 else 1.0
        self._is_fitted = True
        return self

    def page_hinkley_statistics(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Both one-sided Page-Hinkley statistics over ``data``.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            Dict with ``m_increase``/``ph_increase`` (upward shift),
            ``m_decrease``/``ph_decrease`` (downward shift), ``ph`` (the
            per-sample maximum of the two sides), ``alarms`` (indices where
            ``ph > lambda``), ``x_bar``, ``scale`` (the unit of the statistics),
            ``delta`` and ``lambda``.

            The statistics restart on the sample after an alarm, so a
            sustained shift re-alarms at intervals of about
            ``lambda / (shift - delta)`` samples rather than at every sample
            from its onset to the end of the series.
        """
        series = _to_1d_f64(self, data)
        deviations = (series - self._xbar) / self._scale
        m_inc, ph_inc, m_dec, ph_dec, ph = _page_hinkley_scan(
            deviations, self.delta, self.lambda_threshold
        )
        return {
            "m_increase": m_inc,
            "ph_increase": ph_inc,
            "m_decrease": m_dec,
            "ph_decrease": ph_dec,
            "ph": ph,
            "alarms": np.flatnonzero(ph > self.lambda_threshold).tolist(),
            "x_bar": self._xbar,
            "scale": self._scale,
            "delta": self.delta,
            "lambda": self.lambda_threshold,
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the squashed Page-Hinkley statistic.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 scores in ``[0, 1)``.
        """
        stats = self.page_hinkley_statistics(data)
        scores = 1.0 - np.exp(-stats["ph"] * _LN2 / self.lambda_threshold)
        return finite_features(scores, detector=self.name).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Run both one-sided Page-Hinkley tests and score every sample.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            Detection dict: ``scores`` are ``1 - 2^(-PH_t / lambda)`` on the
            larger one-sided statistic (0.5 exactly at the alarm boundary),
            ``anomaly_score`` their maximum, ``is_anomaly`` the per-sample
            threshold verdicts, and ``metadata`` carries both one-sided PH
            trajectories, the alarm indices (``PH_t > lambda``), and the
            operating values ``delta``/``lambda``/``x_bar``.
        """
        stats = self.page_hinkley_statistics(data)
        raw = 1.0 - np.exp(-stats["ph"] * _LN2 / self.lambda_threshold)
        scores = finite_scores(raw, detector=self.name).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
            "metadata": {
                "ph_increase": stats["ph_increase"].tolist(),
                "ph_decrease": stats["ph_decrease"].tolist(),
                "alarms": stats["alarms"],
                "delta": self.delta,
                "lambda": self.lambda_threshold,
                "x_bar": self._xbar,
                "scale": self._scale,
            },
        }
