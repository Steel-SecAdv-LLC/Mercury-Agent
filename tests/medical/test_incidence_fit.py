# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``medical.pandemic.forecasting.incidence_fit``: SEIR fitted to daily incidence.

Outbreaks are simulated from the model itself with negative-binomial noise, so
the true β, R0 and dispersion are known. Bootstrap sizes are kept small for
speed; the coverage of the intervals is measured outside the unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import nbinom

from omni_mercury_engine.medical.pandemic.forecasting import (
    EpidemicForecaster,
    IncidenceForecast,
    fit_seir_incidence,
)
from omni_mercury_engine.medical.pandemic.forecasting.incidence_fit import (
    _bounds,
    _calibrated_prediction,
    _fit as _optimise,
    _objective,
    _SEIRIncidence,
    _starting_points,
)

_N, _SIGMA, _GAMMA = 1e6, 0.2, 0.1
_BETA, _AMPLITUDE = 0.3, 20.0  # R0 = 3
_DAYS, _HORIZON = 40, 14


def _expected() -> np.ndarray:
    mean = _SEIRIncidence(_N, _SIGMA, _GAMMA, 0.0).expected(_BETA, _AMPLITUDE, _DAYS + _HORIZON)
    assert mean is not None
    return mean


def _outbreak(seed: int, size: float = 20.0) -> np.ndarray:
    mean = _expected()
    return np.random.default_rng(seed).negative_binomial(size, size / (size + mean))


def _fit(cases: np.ndarray, seed: int = 0) -> IncidenceForecast:
    return fit_seir_incidence(
        cases[:_DAYS],
        population=_N,
        sigma=_SIGMA,
        gamma=_GAMMA,
        horizon_days=_HORIZON,
        n_bootstrap=30,
        level=0.9,
        seed=seed,
    )


@pytest.fixture(scope="module")
def forecast() -> IncidenceForecast:
    return _fit(_outbreak(1))


@pytest.mark.parametrize("recovered", [0.0, 0.5 * _N])
def test_gradient_is_exact(recovered: float) -> None:
    """The sensitivity-equation gradient equals central differences of the likelihood."""
    model = _SEIRIncidence(_N, _SIGMA, _GAMMA, recovered)
    counts = _outbreak(5)[:_DAYS].astype(float)
    log_factorials = gammaln(counts + 1.0)
    theta = np.log([0.31, 18.0, 25.0])
    _, gradient = _objective(theta, counts, log_factorials, model)
    step = 1e-5
    numeric = [
        (_objective(theta + e, counts, log_factorials, model)[0]
         - _objective(theta - e, counts, log_factorials, model)[0]) / (2 * step)
        for e in np.eye(3) * step
    ]  # fmt: skip
    assert gradient == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_likelihood_is_the_negative_binomial() -> None:
    """The closed-form log-pmf equals scipy's on the same mean, and both solves agree."""
    model = _SEIRIncidence(_N, _SIGMA, _GAMMA, 0.0)
    counts = _outbreak(6)[:_DAYS].astype(float)
    solved = model.expected_with_sensitivities(0.3, 20.0, _DAYS)
    plain = model.expected(0.3, 20.0, _DAYS)
    assert solved is not None and plain is not None
    assert solved[0] == pytest.approx(plain, rel=1e-6)
    expected = -np.sum(nbinom.logpmf(counts, 15.0, 15.0 / (15.0 + solved[0])))
    value, _ = _objective(np.log([0.3, 20.0, 15.0]), counts, gammaln(counts + 1.0), model)
    assert value == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("sigma", "gamma", "recovered"),
    [(0.2, 0.1, 0.0), (0.1, 0.5, 0.0), (1.0, 0.05, 0.0), (0.2, 0.1, 0.95 * _N)],
)
def test_every_point_of_the_search_box_is_a_valid_state(
    sigma: float, gamma: float, recovered: float
) -> None:
    """S0 falls as β and a rise, so S0 > 0 at the box's corner holds everywhere in it."""
    model = _SEIRIncidence(_N, sigma, gamma, recovered)
    (_, log_beta), (_, log_amplitude), _ = _bounds(model)
    susceptible, _, _ = model.initial_state(np.exp(log_beta), np.exp(log_amplitude))
    assert susceptible > 0.0


def test_refit_started_at_the_dispersion_bound_reaches_the_optimum() -> None:
    """A near-Poisson bootstrap refit started at the fit, with k at its upper bound.

    With a capped only at N/10, L-BFGS-B's first step reached an infeasible box
    corner, the penalty there shrank the line search to nothing, and the refit
    stopped at its start (negative log-likelihood 126.49; the optimum is 125.15).
    """
    model = _SEIRIncidence(_N, _SIGMA, _GAMMA, 0.0)
    mean = model.expected(_BETA, _AMPLITUDE, _DAYS)
    assert mean is not None
    counts = np.random.default_rng(15).negative_binomial(1e5, 1e5 / (1e5 + mean)).astype(float)
    bounds = _bounds(model)
    theta, _ = _optimise(counts, model, _starting_points(counts, model), bounds)
    beta, amplitude, size = np.exp(theta)
    assert size == pytest.approx(1e6)
    fitted = model.expected(beta, amplitude, _DAYS)
    assert fitted is not None
    rng = np.random.default_rng(1)
    for _ in range(150):
        replicate = rng.negative_binomial(size, size / (size + fitted)).astype(float)
    start, _ = _objective(theta, replicate, gammaln(replicate + 1.0), model)
    _, refit = _optimise(replicate, model, [theta], bounds)
    assert refit < start - 1.0


def test_noise_free_incidence_recovers_beta() -> None:
    fit = _fit(np.round(_expected()))
    assert fit.beta == pytest.approx(_BETA, rel=0.01)
    assert fit.r0 == pytest.approx(3.0, rel=0.01)
    assert fit.initial_infectious == pytest.approx(_AMPLITUDE, rel=0.05)


def test_noisy_outbreak_recovers_r0(forecast: IncidenceForecast) -> None:
    """Within 3 sd of the estimate's spread over 120 simulated outbreaks (sd 4%).

    The interval's coverage of the truth is measured over many outbreaks
    outside the unit tests; one outbreak cannot test it.
    """
    assert forecast.r0 == pytest.approx(3.0, rel=0.12)
    low, high = forecast.r0_interval
    assert low < forecast.r0 < high


def test_initial_immunity_enters_the_initial_state() -> None:
    """Fails when the eigenvector ignores immunity (R0 3.047, interval above 3)."""
    model = _SEIRIncidence(_N, _SIGMA, _GAMMA, 0.5 * _N)
    mean = model.expected(_BETA, _AMPLITUDE, _DAYS)
    assert mean is not None
    assert np.allclose(mean[1:] / mean[:-1], mean[1] / mean[0], rtol=1e-3)  # growth from day 0
    fit = fit_seir_incidence(
        np.round(mean),
        population=_N,
        sigma=_SIGMA,
        gamma=_GAMMA,
        n_bootstrap=30,
        level=0.9,
        seed=0,
        initial_recovered=0.5 * _N,
    )
    assert fit.r0 == pytest.approx(3.0, rel=0.01)
    assert fit.r0_interval[0] <= 3.0 <= fit.r0_interval[1]


def test_intervals_are_ordered_and_nested(forecast: IncidenceForecast) -> None:
    assert forecast.forecast_mean.shape == forecast.prediction_upper.shape == (_HORIZON,)
    assert forecast.fitted.shape == (_DAYS,)
    assert np.all(forecast.mean_lower <= forecast.mean_upper)
    assert np.all(forecast.prediction_lower <= forecast.mean_lower + 1.0)
    assert np.all(forecast.prediction_upper >= forecast.mean_upper - 1.0)
    assert np.all(
        forecast.prediction_upper - forecast.prediction_lower
        >= forecast.mean_upper - forecast.mean_lower
    )
    assert forecast.level == 0.9 and forecast.n_bootstrap == 30
    assert forecast.prediction_level.shape == (_HORIZON,)
    assert np.all((forecast.prediction_level > 0.0) & (forecast.prediction_level < 1.0))


def test_mean_band_is_calibrated_not_plain_percentile(forecast: IncidenceForecast) -> None:
    """The band of the mean carries its per-day calibrated nominal level.

    Fails on the previous tree: the band was the raw percentile of the
    bootstrap refits (measured 86-92% coverage at a 95% target) and exposed
    no ``mean_nominal``; it is now read at a prepivoted nominal level over
    refits drawn at the bias-corrected dispersion, never below the target.
    """
    assert forecast.mean_nominal is not None
    assert forecast.mean_nominal.shape == (_HORIZON,)
    assert np.all(forecast.mean_nominal >= forecast.level)
    assert np.all(forecast.mean_nominal <= 1.0)
    # The calibrated band contains the fitted forecast path itself.
    assert np.all(forecast.mean_lower <= forecast.forecast_mean)
    assert np.all(forecast.forecast_mean <= forecast.mean_upper)


def test_mean_band_is_seed_reproducible() -> None:
    first, second = _fit(_outbreak(3), seed=7), _fit(_outbreak(3), seed=7)
    np.testing.assert_array_equal(first.mean_lower, second.mean_lower)
    np.testing.assert_array_equal(first.mean_upper, second.mean_upper)
    assert first.mean_nominal is not None and second.mean_nominal is not None
    np.testing.assert_array_equal(first.mean_nominal, second.mean_nominal)


def _coverage(lower: np.ndarray, upper: np.ndarray, mean: np.ndarray, size: float) -> np.ndarray:
    p = size / (size + mean)
    return np.asarray(nbinom.cdf(upper, size, p) - nbinom.cdf(lower - 1.0, size, p))


def test_calibration_without_estimation_error_is_the_plug_in_interval() -> None:
    """Refits equal to the truth: the calibrated interval is the plug-in one, and covers."""
    mean, size = np.array([5.0, 50.0, 500.0]), 20.0
    lower, upper, nominal = _calibrated_prediction(
        mean, size, np.tile(mean, (40, 1)), np.full(40, size), 0.95
    )
    p = size / (size + mean)
    assert np.all(_coverage(lower, upper, mean, size) >= 0.95)
    assert np.all(lower >= nbinom.ppf(0.025, size, p))
    assert np.all(upper <= nbinom.ppf(0.975, size, p))
    assert np.all(nominal <= 0.95)


def test_calibration_widens_when_refits_understate_dispersion() -> None:
    """Refits with k twice the truth's give too-narrow plug-in intervals; calibration widens them."""
    mean, size = np.array([50.0, 500.0]), 20.0
    lower, upper, nominal = _calibrated_prediction(
        mean, size, np.tile(mean, (40, 1)), np.full(40, 2.0 * size), 0.95
    )
    assert np.all(nominal > 0.95)
    assert np.all(_coverage(lower, upper, mean, size) >= 0.95)


def test_calibration_resolves_levels_near_one() -> None:
    """Near-Poisson refits of a k = 5.5e4 truth need level 1 - 1.7e-8, short of the 1 - 1e-9 cap.

    Twenty bisection steps (2^-20 in level) stopped at the cap, 8% wider.
    """
    mean, size = np.array([4e5]), 5.5e4
    lower, upper, nominal = _calibrated_prediction(
        mean, size, np.tile(mean, (40, 1)), np.full(40, 1e9), 0.95
    )
    assert 1e-8 < 1.0 - nominal[0] < 1e-7
    assert _coverage(lower, upper, mean, size)[0] >= 0.95


def test_an_unreachable_target_falls_back_to_a_covering_interval() -> None:
    """A day whose needed nominal exceeds the 1 - 1e-9 cap still covers the target.

    Regression: with bootstrap means collapsed toward 0 while the fitted
    truth sits at 50, no nominal level inside the search range reaches 95%
    coverage; bisection stopped at the cap and returned the capped nominal
    as if achieved, with an interval whose true coverage was ~0.
    """
    mean, size = np.array([50.0]), 10.0
    lower, upper, nominal = _calibrated_prediction(
        mean, size, np.full((40, 1), 0.01), np.full(40, size), 0.95
    )
    assert _coverage(lower, upper, mean, size)[0] >= 0.95
    assert nominal[0] == pytest.approx(0.95)
    assert lower[0] == 0.0


def test_forecast_continues_the_fitted_curve(forecast: IncidenceForecast) -> None:
    """The mean is one trajectory: the forecast starts where the fit ends."""
    step = forecast.fitted[-1] / forecast.fitted[-2]
    assert forecast.forecast_mean[0] / forecast.fitted[-1] == pytest.approx(step, rel=0.05)


def test_overdispersion_is_estimated() -> None:
    """k = 2 is estimated closely; a large k is weakly identified, so only the order is checked."""
    clumped = _fit(_outbreak(2, size=2.0)).dispersion
    near_poisson = _fit(_outbreak(2, size=1e4)).dispersion
    assert 1.0 < clumped < 5.0
    assert near_poisson > 10.0 * clumped


def test_seeded_fit_is_reproducible() -> None:
    cases = _outbreak(3)
    a, b = _fit(cases, seed=7), _fit(cases, seed=7)
    assert a.r0_interval == b.r0_interval
    assert np.array_equal(a.prediction_upper, b.prediction_upper)


def test_forecaster_fits_with_its_own_rates() -> None:
    forecaster = EpidemicForecaster(
        {"beta": 0.5, "sigma": _SIGMA, "gamma": _GAMMA, "population": _N}
    )
    fit = forecaster.fit_to_incidence(
        _outbreak(4)[:_DAYS], horizon_days=_HORIZON, n_bootstrap=30, level=0.9, seed=0
    )
    assert fit.r0 == pytest.approx(3.0, rel=0.1)
    assert forecaster.beta == 0.5


@pytest.mark.parametrize(
    ("cases", "kwargs", "match"),
    [
        ([1, 2, 3, 4], {}, ">= 5"),
        ([1, 2, -3, 4, 5], {}, "non-negative"),
        ([1, 2, np.nan, 4, 5], {}, "finite"),
        ([1, 2.5, 3, 4, 5], {}, "whole numbers"),
        ([0, 0, 0, 0, 0], {}, "no cases"),
        ([1, 2, 3, 4, 5], {"level": 1.0}, "level"),
        ([1, 2, 3, 4, 5], {"horizon_days": 0}, "horizon_days"),
        ([1, 2, 3, 4, 5], {"horizon_days": 1.5}, "horizon_days"),
        ([1, 2, 3, 4, 5], {"n_bootstrap": 19, "level": 0.9}, "n_bootstrap must be >= 20"),
        ([1, 2, 3, 4, 5], {"sigma": 0.0}, "sigma"),
        ([1, 2, 3, 4, 5], {"sigma": float("inf")}, "sigma"),
        ([1, 2, 100_000.4, 4, 5], {}, "whole numbers"),
        ([300, 300, 300, 300, 300], {"population": 1000}, "must exceed"),
        ([1, 2, 3, 4, 5], {"population": 10, "initial_recovered": 0}, "must exceed"),
        ([1, 2, 3, 4, 5], {"population": True}, "population"),
        ([1, 2, 3, 4, 5], {"n_bootstrap": float("inf")}, "n_bootstrap"),
        ([1, 2, 3, 4, 5], {"level": "0.9"}, "level"),
    ],
)
def test_unusable_input_is_rejected(
    cases: list[float], kwargs: dict[str, object], match: str
) -> None:
    options: dict[str, object] = {"population": _N, "sigma": _SIGMA, "gamma": _GAMMA}
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        fit_seir_incidence(np.array(cases, dtype=float), **options)  # type: ignore[arg-type]
