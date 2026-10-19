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
from scipy.stats import nbinom

from omni_mercury_engine.medical.pandemic.forecasting import (
    EpidemicForecaster,
    IncidenceForecast,
    fit_seir_incidence,
)
from omni_mercury_engine.medical.pandemic.forecasting.incidence_fit import (
    _calibrated_prediction,
    _SEIRIncidence,
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
