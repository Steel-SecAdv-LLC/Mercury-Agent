# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``medical.pandemic.forecasting.EpidemicForecaster``: R0, threshold and impact.

The released forecaster reported R0 = β/(σ+γ) (1.67 for the default rates whose
true R0 is 5), a "chaos" score that depended on population size and flagged
sub-threshold epidemics, deaths from peak prevalence instead of cumulative
infections, and peak days off a 365-point grid over 365 days. The tests marked
as such fail on the released code.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.medical.pandemic.forecasting.epidemic_model import EpidemicForecaster


def _final_size_r0(forecaster: EpidemicForecaster) -> float:
    """R0 implied by the simulated final size: ln(s∞)/(s∞ - 1) for S0 ≈ N."""
    s_end = forecaster.forecast_pandemic().seir_trajectory["susceptible"][-1]
    s_inf = s_end / forecaster.population
    return float(-np.log(s_inf) / (1.0 - s_inf))


@pytest.mark.parametrize("beta", [0.15, 0.25, 0.5])
def test_r0_is_beta_over_gamma(beta: float) -> None:
    """Fails on the released β/(σ+γ)."""
    forecaster = EpidemicForecaster({"beta": beta, "forecast_days": 1000})
    r0 = forecaster.forecast_pandemic().r0_estimate
    assert r0 == pytest.approx(beta / 0.1)
    assert r0 == pytest.approx(_final_size_r0(forecaster), rel=1e-3)


def test_supercritical_outbreak_is_not_called_subcritical() -> None:
    """Fails on the released code (R0 0.833, "dies out", while 89% are infected)."""
    result = EpidemicForecaster({"beta": 0.25}).forecast_pandemic()
    assert result.outbreak_detected
    assert result.r0_estimate > 1.0
    assert result.bifurcation_detected


def test_threshold_is_the_epidemic_threshold() -> None:
    """Fails on the released score, which flagged a sub-threshold epidemic."""
    below = EpidemicForecaster({"beta": 0.09}).forecast_pandemic()
    above = EpidemicForecaster({"beta": 0.11}).forecast_pandemic()
    assert below.chaos_score < 0.0 and not below.bifurcation_detected
    assert above.chaos_score > 0.0 and above.bifurcation_detected


@pytest.mark.parametrize("population", [1_000, 1_000_000, 100_000_000])
def test_threshold_score_does_not_depend_on_population(population: int) -> None:
    """Fails on the released score (2.10 at 1e3, capped 5.0 at 1e6)."""
    forecaster = EpidemicForecaster({"population": population})
    score = forecaster.forecast_pandemic(initial_infected=0, initial_exposed=1).chaos_score
    assert score == pytest.approx(0.170156, abs=1e-3)


def test_threshold_score_is_the_early_growth_rate() -> None:
    result = EpidemicForecaster().forecast_pandemic()
    infected = result.seir_trajectory["infected"][10:40]
    slope = np.polyfit(np.arange(infected.size), np.log(infected), 1)[0]
    assert result.chaos_score == pytest.approx(slope, rel=0.02)


def test_grid_is_daily() -> None:
    """Fails on the released 365-point grid over 365 days."""
    result = EpidemicForecaster({"forecast_days": 200}).forecast_pandemic()
    days = result.seir_trajectory["time_days"]
    assert days[0] == 0.0 and days[-1] == 200.0 and np.all(np.diff(days) == 1.0)
    assert days[result.peak_day] == result.peak_day


def test_conservation() -> None:
    trajectory = EpidemicForecaster().forecast_pandemic().seir_trajectory
    total = sum(trajectory[k] for k in ("susceptible", "exposed", "infected", "recovered"))
    assert np.max(np.abs(total - 1_000_000)) < 1e-3


def test_impact_counts_cumulative_infections() -> None:
    """Fails on the released impact (1% of *peak prevalence*, 3,050 deaths)."""
    forecaster = EpidemicForecaster()
    result = forecaster.forecast_pandemic()
    impact = result.humanitarian_impact
    s_end = result.seir_trajectory["susceptible"][-1]
    assert impact["total_infections"] == round(forecaster.population - s_end)
    assert impact["estimated_deaths"] == int(impact["total_infections"] * 0.01)
    assert impact["economic_impact_usd"] == impact["total_infections"] * 50000.0
    assert impact["assumptions"]["infection_fatality_ratio"] == 0.01
    assert impact["estimated_deaths"] > int(result.peak_infections * 0.01)


def test_impact_ratios_are_configurable() -> None:
    impact = (
        EpidemicForecaster({"infection_fatality_ratio": 0.002, "cost_per_infection_usd": 0.0})
        .forecast_pandemic()
        .humanitarian_impact
    )
    assert impact["estimated_deaths"] == int(impact["total_infections"] * 0.002)
    assert impact["economic_impact_usd"] == 0.0


def test_r_from_incidence_recovers_r0() -> None:
    forecaster = EpidemicForecaster({"population": 100_000_000})
    trajectory = forecaster.forecast_pandemic(initial_infected=100, initial_exposed=0)
    incidence = forecaster.sigma * trajectory.seir_trajectory["exposed"][20:50]
    estimate = forecaster.estimate_r0_from_incidence(np.random.default_rng(0).poisson(incidence))
    assert estimate == pytest.approx(5.0, rel=0.05)


def test_r_from_incidence_below_one_for_a_declining_series() -> None:
    forecaster = EpidemicForecaster()
    cases = 1000.0 * np.exp(-0.02 * np.arange(30))
    assert forecaster.estimate_r0_from_incidence(cases) < 1.0


@pytest.mark.parametrize("cases", [[1, 2], [0, 0, 0], [1, -1, 3], [1, np.nan, 3], [0, 0, 5]])
def test_r_from_incidence_rejects_unusable_series(cases: list[float]) -> None:
    with pytest.raises(ValueError):
        EpidemicForecaster().estimate_r0_from_incidence(np.array(cases, dtype=float))


def test_intervention_restores_beta_and_counts_cumulative_lives() -> None:
    forecaster = EpidemicForecaster()
    effect = forecaster.simulate_intervention_effect("vaccination", 0.8)
    assert forecaster.beta == 0.5
    assert 0.0 < effect["peak_reduction"] < 1.0
    assert effect["lives_saved_estimate"] > 0


@pytest.mark.parametrize(
    "config",
    [
        {"gamma": 0.0},
        {"sigma": -1.0},
        {"beta": float("nan")},
        {"population": 0},
        {"forecast_days": 0},
        {"infection_fatality_ratio": 1.5},
        {"cost_per_infection_usd": -1},
    ],
)
def test_invalid_configuration_is_rejected(config: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        EpidemicForecaster(config)


def test_invalid_initial_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        EpidemicForecaster({"population": 100}).forecast_pandemic(initial_infected=200)


def test_intervention_never_raises_transmission() -> None:
    """Fails on the released 0.5 floor, which raised R0 = 0.4 to 0.5."""
    effect = EpidemicForecaster({"beta": 0.04}).simulate_intervention_effect("quarantine", 0.5)
    assert effect["peak_reduction"] >= 0.0


@pytest.mark.parametrize("effectiveness", [-1.0, 1.5])
def test_intervention_effectiveness_is_validated(effectiveness: float) -> None:
    with pytest.raises(ValueError, match="effectiveness"):
        EpidemicForecaster().simulate_intervention_effect("vaccination", effectiveness)


def test_no_rapid_intervention_below_threshold() -> None:
    """Fails on the released recommender ("peak expected in 8 days" at R0 0.5)."""
    result = EpidemicForecaster({"beta": 0.05}).forecast_pandemic()
    assert not any("Rapid intervention" in step for step in result.recommended_interventions)


@pytest.mark.parametrize(
    "config",
    [
        {"beta": None},
        {"beta": "0.5"},
        {"forecast_days": float("inf")},
        {"population": float("inf")},
        {"forecast_days": 10.5},
        {"sigma": True},
    ],
)
def test_non_numeric_or_infinite_settings_are_value_errors(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EpidemicForecaster(config)
