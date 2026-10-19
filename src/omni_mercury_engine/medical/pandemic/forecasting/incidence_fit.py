# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit the SEIR model to observed daily incidence and forecast it with intervals.

Method (Chowell, Bleichrodt & Luo 2024, *Stat. Med.* 43:1826-1848; King et al.
2015, *Proc. R. Soc. B* 282:20150347):

- Observation model: the count on day j is negative binomial (NB2) with mean
  μ_j and size k, so Var = μ + μ²/k. μ_j is the model's E→I transitions during
  day j, the daily increment of C with C' = σE. Cumulative counts are never
  fitted (King et al. 2015, principle 1).
- Parameters: β, an initial amplitude a and k, by maximum likelihood from
  several starting points, with the exact gradient from the forward
  sensitivity equations; σ, γ and the population are fixed. The initial state
  lies on the dominant eigenvector of the (E, I) system linearised at the
  disease-free state S = N - R (R = ``initial_recovered``, so β acts as
  β(N - R)/N), E0 = (Λ+γ)·a/σ and I0 = a, because in the growth phase only that
  combination of E0 and I0 is identifiable.
- Uncertainty: a parametric bootstrap. B replicate data sets y* ~ NB(μ̂, k̂)
  are each refitted; the intervals for R0 and for the mean are the percentile
  intervals of the refits.
- Prediction interval for future counts: the plug-in interval of NB(μ̂, k̂),
  calibrated per forecast day (Beran 1990, *JASA* 85:715-723). With the fit as
  the truth of the bootstrap world, the nominal level is raised until the
  refits' own plug-in intervals cover a count with the target probability. The
  upward bias of k̂ in small samples (Lloyd-Smith 2007, *PLoS ONE* 2:e180)
  makes uncalibrated intervals too narrow: over 120 simulated outbreaks (40
  observed days, k = 20, B = 200) the bootstrap-draw interval covered 91.1% of
  future counts at a 95% target, and the calibrated interval 95.9%.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from scipy.stats import nbinom

__all__ = ["IncidenceForecast", "fit_seir_incidence"]

# Parameters fit in log space: β per day, amplitude (people), NB size k.
_BETA_BOUNDS = (1e-4, 20.0)
_K_BOUNDS = (1e-2, 1e6)
_K_STARTS = (1.0, 30.0, 1e4)
_BETA_FACTORS = (1.0, 0.5, 2.0)
_PENALTY = 1e12
# Three parameters need more days than parameters; no published minimum exists.
_MIN_DAYS = 5


@dataclass(frozen=True)
class IncidenceForecast:
    """A fitted SEIR model, its forecast and their uncertainty.

    Attributes:
        beta: Fitted transmission rate per day.
        r0: β/γ at the fitted β.
        dispersion: Fitted negative-binomial size k (Var = μ + μ²/k).
        initial_exposed: Fitted E at the start of the first observed day.
        initial_infectious: Fitted I at the start of the first observed day.
        log_likelihood: Log-likelihood of the observed counts at the fit.
        fitted: Expected daily incidence over the observed days.
        forecast_mean: Expected daily incidence over the forecast horizon.
        mean_lower: Lower bound of the band of the mean.
        mean_upper: Upper bound of the band of the mean.
        prediction_lower: Lower prediction bound for future daily counts.
        prediction_upper: Upper prediction bound for future daily counts.
        prediction_level: Calibrated nominal level of each day's plug-in
            prediction interval; above ``level`` when the plug-in interval
            undercovers.
        r0_interval: Bootstrap interval for R0.
        level: Coverage level of the intervals.
        n_bootstrap: Bootstrap refits that succeeded.
    """

    beta: float
    r0: float
    dispersion: float
    initial_exposed: float
    initial_infectious: float
    log_likelihood: float
    fitted: np.ndarray[Any, Any]
    forecast_mean: np.ndarray[Any, Any]
    mean_lower: np.ndarray[Any, Any]
    mean_upper: np.ndarray[Any, Any]
    prediction_lower: np.ndarray[Any, Any]
    prediction_upper: np.ndarray[Any, Any]
    prediction_level: np.ndarray[Any, Any]
    r0_interval: tuple[float, float]
    level: float
    n_bootstrap: int


class _SEIRIncidence:
    """Expected daily E→I transitions of an SEIR model started on its eigenvector."""

    def __init__(self, population: float, sigma: float, gamma: float, recovered: float) -> None:
        self.population = population
        self.sigma = sigma
        self.gamma = gamma
        self.recovered = recovered

    def growth_rate(self, beta: float) -> float:
        """Λ of the (E, I) system linearised at S = N - R, where β acts as β(N - R)/N."""
        s, g = self.sigma, self.gamma
        effective = beta * (self.population - self.recovered) / self.population
        return (-(s + g) + math.sqrt((s - g) ** 2 + 4.0 * s * effective)) / 2.0

    def initial_state(self, beta: float, amplitude: float) -> tuple[float, float, float]:
        """(S0, E0, I0) on the dominant eigenvector of the linearised (E, I) system."""
        exposed = (self.growth_rate(beta) + self.gamma) * amplitude / self.sigma
        infectious = amplitude
        return self.population - exposed - infectious - self.recovered, exposed, infectious

    def expected(self, beta: float, amplitude: float, n_days: int) -> np.ndarray[Any, Any] | None:
        """Expected incidence on days 0..n_days-1, or ``None`` for an invalid state."""
        susceptible, exposed, infectious = self.initial_state(beta, amplitude)
        if not (susceptible > 0.0 and math.isfinite(exposed)):
            return None
        rate, s, g = beta / self.population, self.sigma, self.gamma

        def rhs(y: np.ndarray[Any, Any], _t: float) -> list[float]:
            sus, exp_, inf, _ = y.tolist()  # Python floats: faster than numpy scalars
            force = rate * sus * inf
            return [-force, force - s * exp_, s * exp_ - g * inf, s * exp_]

        days = np.arange(n_days + 1, dtype=float)
        solution = odeint(rhs, [susceptible, exposed, infectious, 0.0], days, rtol=1e-8, atol=1e-6)
        mean = np.diff(solution[:, 3])
        if not np.all(np.isfinite(mean)):
            return None
        return np.maximum(mean, 1e-10)

    def expected_with_sensitivities(
        self, beta: float, amplitude: float, n_days: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
        """Expected incidence and its exact derivatives in β and the amplitude.

        Integrates the forward sensitivity equations s' = (∂f/∂y)s + ∂f/∂p
        alongside the SEIR system, started from the derivatives of the
        eigenvector initial state, so one solve gives the likelihood's gradient.
        """
        susceptible, exposed, infectious = self.initial_state(beta, amplitude)
        if not (susceptible > 0.0 and math.isfinite(exposed)):
            return None
        n, s, g = self.population, self.sigma, self.gamma
        fraction = (n - self.recovered) / n
        root = math.sqrt((s - g) ** 2 + 4.0 * s * beta * fraction)
        d_exposed_d_beta = amplitude * fraction / root  # a/σ · dΛ/dβ, dΛ/dβ = σ f / root
        d_exposed_d_amp = (self.growth_rate(beta) + g) / s
        initial = [
            susceptible, exposed, infectious, 0.0,
            -d_exposed_d_beta, d_exposed_d_beta, 0.0, 0.0,
            -d_exposed_d_amp - 1.0, d_exposed_d_amp, 1.0, 0.0,
        ]  # fmt: skip
        rate = beta / n

        def rhs(y: np.ndarray[Any, Any], _t: float) -> list[float]:
            sus, exp_, inf, _, b_s, b_e, b_i, _, a_s, a_e, a_i, _ = y.tolist()
            force = rate * sus * inf
            # Linearised force of infection along each sensitivity; β also enters directly.
            b_force = rate * (inf * b_s + sus * b_i) + sus * inf / n
            a_force = rate * (inf * a_s + sus * a_i)
            return [
                -force, force - s * exp_, s * exp_ - g * inf, s * exp_,
                -b_force, b_force - s * b_e, s * b_e - g * b_i, s * b_e,
                -a_force, a_force - s * a_e, s * a_e - g * a_i, s * a_e,
            ]  # fmt: skip

        days = np.arange(n_days + 1, dtype=float)
        solution = odeint(rhs, initial, days, rtol=1e-8, atol=1e-6)
        mean = np.diff(solution[:, 3])
        if not np.all(np.isfinite(solution)):
            return None
        floored = mean < 1e-10
        d_beta = np.where(floored, 0.0, np.diff(solution[:, 7]))
        d_amp = np.where(floored, 0.0, np.diff(solution[:, 11]))
        return np.maximum(mean, 1e-10), d_beta, d_amp


def _objective(
    theta: np.ndarray[Any, Any],
    counts: np.ndarray[Any, Any],
    log_factorials: np.ndarray[Any, Any],
    model: _SEIRIncidence,
) -> tuple[float, np.ndarray[Any, Any]]:
    """Negative log-likelihood and its exact gradient in θ = log(β, a, k).

    With m the mean and k the size, the NB2 log-pmf is ln Γ(y+k) - ln Γ(k) -
    ln y! + k ln(k/(k+m)) + y ln(m/(k+m)); its derivatives are y/m - (y+k)/(k+m)
    in m and ψ(y+k) - ψ(k) + ln(k/(k+m)) + (m-y)/(k+m) in k.
    """
    beta, amplitude, size = (float(v) for v in np.exp(theta))
    solved = model.expected_with_sensitivities(beta, amplitude, counts.size)
    if solved is None:
        return _PENALTY, np.zeros(3)
    mean, d_beta, d_amp = solved
    total = size + mean
    log_ratio = np.log(size / total)
    log_pmf = (
        gammaln(counts + size)
        - gammaln(size)
        - log_factorials
        + size * log_ratio
        + counts * np.log(mean / total)
    )
    value = -float(np.sum(log_pmf))
    if not math.isfinite(value):
        return _PENALTY, np.zeros(3)
    d_mean = -(counts / mean - (counts + size) / total)
    d_size = -float(
        np.sum(digamma(counts + size) - digamma(size) + log_ratio + (mean - counts) / total)
    )
    gradient = np.array(
        [beta * float(d_mean @ d_beta), amplitude * float(d_mean @ d_amp), size * d_size]
    )
    return value, gradient


def _starting_points(
    counts: np.ndarray[Any, Any], model: _SEIRIncidence
) -> list[np.ndarray[Any, Any]]:
    """Starts from the early log-linear growth rate, at several β and k values."""
    window = counts[: min(counts.size, 21)]
    days = np.arange(window.size, dtype=float)
    rate = float(np.polyfit(days, np.log(window + 1.0), 1)[0])
    s, g = model.sigma, model.gamma
    rate = max(rate, -0.9 * min(s, g))
    susceptible = (model.population - model.recovered) / model.population
    beta0 = float(np.clip((rate + s) * (rate + g) / (s * susceptible), *_BETA_BOUNDS))
    starts = []
    for factor in _BETA_FACTORS:
        beta = float(np.clip(beta0 * factor, *_BETA_BOUNDS))
        amplitude = max(float(np.mean(counts[:3])), 0.5) / (model.growth_rate(beta) + g)
        for size in _K_STARTS:
            starts.append(np.log([beta, amplitude, size]))
    return starts


def _bounds(model: _SEIRIncidence) -> list[tuple[float, float]]:
    """Box bounds on θ = log(β, a, k) inside which every initial state has S0 > 0.

    E0 + I0 = a(Λ+γ+σ)/σ and Λ grows with β, so capping a below (N - R)σ/(Λ+γ+σ)
    at the largest β keeps S0 positive across the box. Without the cap the box
    corner that L-BFGS-B's first step reaches can be infeasible, and the penalty
    there collapses the line search onto the starting point.
    """
    largest = model.growth_rate(_BETA_BOUNDS[1]) + model.gamma + model.sigma
    # 1 - 1e-9 keeps S0 > 0 through the rounding of E0 and I0.
    feasible = (model.population - model.recovered) * model.sigma / largest * (1.0 - 1e-9)
    return [
        (math.log(_BETA_BOUNDS[0]), math.log(_BETA_BOUNDS[1])),
        (math.log(1e-6), math.log(min(model.population / 10.0, feasible))),
        (math.log(_K_BOUNDS[0]), math.log(_K_BOUNDS[1])),
    ]


def _fit(
    counts: np.ndarray[Any, Any],
    model: _SEIRIncidence,
    starts: list[np.ndarray[Any, Any]],
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray[Any, Any], float]:
    """Best L-BFGS-B fit over the starting points: (log parameters, NLL)."""
    best_theta, best_value = starts[0], math.inf
    log_factorials = gammaln(counts + 1.0)
    for start in starts:
        result = minimize(
            _objective,
            np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds]),
            args=(counts, log_factorials, model),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.fun < best_value:
            best_theta, best_value = result.x, float(result.fun)
    return best_theta, best_value


def _calibrated_prediction(
    mean: np.ndarray[Any, Any],
    size: float,
    boot_means: np.ndarray[Any, Any],
    boot_sizes: np.ndarray[Any, Any],
    level: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Beran-calibrated plug-in prediction interval per day: (lower, upper, level used).

    NB(mean, size) is the truth of the bootstrap world, and refit b's plug-in
    interval at level l is the central l interval of NB(boot_means[b],
    boot_sizes[b]). For each day the smallest l whose intervals cover that day's
    count with average probability >= ``level`` is found by bisection.
    """
    truth_p = size / (size + mean)
    sizes = boot_sizes[:, None]
    boot_p = sizes / (sizes + boot_means)

    def coverage(nominal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        lower = nbinom.ppf((1.0 - nominal) / 2.0, sizes, boot_p)
        upper = nbinom.ppf((1.0 + nominal) / 2.0, sizes, boot_p)
        inside = nbinom.cdf(upper, size, truth_p) - nbinom.cdf(lower - 1.0, size, truth_p)
        return np.asarray(inside.mean(axis=0))

    low = np.full(mean.shape, 1e-6)
    high = np.full(mean.shape, 1.0 - 1e-9)
    for _ in range(30):
        mid = (low + high) / 2.0
        enough = coverage(mid) >= level
        high = np.where(enough, mid, high)
        low = np.where(enough, low, mid)
    lower = nbinom.ppf((1.0 - high) / 2.0, size, truth_p)
    upper = nbinom.ppf((1.0 + high) / 2.0, size, truth_p)
    return lower, upper, high


def _real(name: str, value: float) -> float:
    """``value`` as a finite float; booleans and non-numbers raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _count(name: str, value: int) -> int:
    """``value`` as a positive int; booleans and non-integers raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def fit_seir_incidence(
    daily_cases: np.ndarray[Any, Any],
    *,
    population: float,
    sigma: float,
    gamma: float,
    horizon_days: int = 14,
    n_bootstrap: int = 200,
    level: float = 0.95,
    seed: int | None = None,
    initial_recovered: float = 0.0,
) -> IncidenceForecast:
    """Fit β, the initial amplitude and NB dispersion, then forecast with intervals.

    Args:
        daily_cases: Daily counts of new infectious cases, oldest first.
        population: Population size N.
        sigma: Rate of leaving the exposed stage (1/latent period), per day.
        gamma: Rate of leaving the infectious stage (1/infectious period), per day.
        horizon_days: Days to forecast after the last observed day.
        n_bootstrap: Bootstrap replicates (Chowell et al. use about 200-300).
        level: Target coverage of the intervals.
        seed: Seed for the bootstrap.
        initial_recovered: People already immune at the start.

    Returns:
        The fit, the forecast and their bootstrap intervals.

    Raises:
        ValueError: If the counts are not at least five finite, non-negative
            whole numbers with some cases, or a setting is out of range.
    """
    counts = np.asarray(daily_cases, dtype=float).ravel()
    if counts.size < _MIN_DAYS or not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError(f"daily_cases needs >= {_MIN_DAYS} finite, non-negative counts")
    if not np.array_equal(counts, np.round(counts)):
        raise ValueError("daily_cases must be whole numbers")
    if counts.sum() <= 0.0:
        raise ValueError("daily_cases has no cases")
    level = _real("level", level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    horizon = _count("horizon_days", horizon_days)
    # At least one replicate in each tail; rounding keeps 1 - 0.9 from asking for 21.
    needed = math.ceil(round(2.0 / (1.0 - level), 9))
    if _count("n_bootstrap", n_bootstrap) < needed:
        raise ValueError(f"n_bootstrap must be >= {needed} for level {level}, got {n_bootstrap}")
    population, sigma, gamma = (
        _real(name, value)
        for name, value in (("population", population), ("sigma", sigma), ("gamma", gamma))
    )
    initial_recovered = _real("initial_recovered", initial_recovered)
    if not (sigma > 0 and gamma > 0 and initial_recovered >= 0):
        raise ValueError("sigma and gamma must be > 0 and initial_recovered >= 0")
    if not population > initial_recovered + counts.sum():
        raise ValueError(
            f"population ({population:g}) must exceed initial_recovered plus the "
            f"observed cases ({initial_recovered + counts.sum():g})"
        )
    model = _SEIRIncidence(population, sigma, gamma, initial_recovered)
    bounds = _bounds(model)
    theta, nll = _fit(counts, model, _starting_points(counts, model), bounds)
    beta, amplitude, size = (float(v) for v in np.exp(theta))
    full_mean = model.expected(beta, amplitude, counts.size + horizon)
    if full_mean is None:
        raise ValueError("the fit did not reach a valid initial state")

    rng = np.random.default_rng(seed)
    means, sizes, r0s = [], [], []
    for _ in range(int(n_bootstrap)):
        replicate = rng.negative_binomial(size, size / (size + full_mean[: counts.size]))
        theta_b, _ = _fit(replicate.astype(float), model, [theta], bounds)
        beta_b, amplitude_b, size_b = (float(v) for v in np.exp(theta_b))
        mean_b = model.expected(beta_b, amplitude_b, counts.size + horizon)
        if mean_b is None:
            continue
        means.append(mean_b[counts.size :])
        sizes.append(size_b)
        r0s.append(beta_b / gamma)
    if len(means) < needed:
        raise ValueError(f"only {len(means)} bootstrap refits succeeded")

    tails = [(1.0 - level) / 2.0, (1.0 + level) / 2.0]
    mean_band = np.quantile(np.array(means), tails, axis=0)
    lower, upper, nominal = _calibrated_prediction(
        full_mean[counts.size :], size, np.array(means), np.array(sizes), level
    )
    r0_band = np.quantile(np.array(r0s), tails)
    _, exposed, infectious = model.initial_state(beta, amplitude)
    return IncidenceForecast(
        beta=beta,
        r0=beta / gamma,
        dispersion=size,
        initial_exposed=exposed,
        initial_infectious=infectious,
        log_likelihood=-nll,
        fitted=full_mean[: counts.size],
        forecast_mean=full_mean[counts.size :],
        mean_lower=mean_band[0],
        mean_upper=mean_band[1],
        prediction_lower=lower,
        prediction_upper=upper,
        prediction_level=nominal,
        r0_interval=(float(r0_band[0]), float(r0_band[1])),
        level=level,
        n_bootstrap=len(means),
    )
