# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Epidemic Forecasting using an SEIR Model with an Epidemic-Threshold Test.

Implements the Susceptible-Exposed-Infected-Recovered (SEIR) compartmental
model without vital dynamics.

Mathematical Foundation:
- SEIR Dynamics: dS/dt = -β*S*I/N
                  dE/dt = β*S*I/N - σ*E
                  dI/dt = σ*E - γ*I
                  dR/dt = γ*I
- R0 = ρ(FV⁻¹) = β/γ (next-generation matrix; van den Driessche & Watmough 2002).
  Every exposed person becomes infectious, so σ does not enter R0.
- Threshold Λ: the dominant eigenvalue of the infection subsystem linearised at
  the disease-free state, Λ = (-(σ+γ) + √((σ-γ)² + 4σβ·S0/N)) / 2 per day. The
  disease-free state loses stability (a transcritical bifurcation) exactly when
  Λ > 0, i.e. when R0·S0/N > 1, and Λ is the early exponential growth rate.
- R from incidence: R = (1 + r/σ)(1 + r/γ) for an observed growth rate r
  (Wallinga & Lipsitch 2007, eq. 3.2), with r from a Poisson log-linear fit.

References:
- Kermack & McKendrick (1927); Anderson & May (1991)
- van den Driessche & Watmough (2002), Math. Biosci. 180:29-48
- Wallinga & Lipsitch (2007), Proc. R. Soc. B 274:599-604
- Obadia, Haneef & Boëlle (2012), BMC Med. Inform. Decis. Mak. 12:147 (R0 package)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import odeint


@dataclass
class PandemicForecast:
    """Result from pandemic forecasting."""

    outbreak_detected: bool
    r0_estimate: float
    peak_infections: int
    peak_day: int

    seir_trajectory: dict[str, np.ndarray[Any, Any]]
    chaos_score: float
    bifurcation_detected: bool

    humanitarian_impact: dict[str, Any]
    recommended_interventions: list[str]


class EpidemicForecaster:
    """SEIR-Based Pandemic Forecaster (Medical Interdiction).

    Forecasts disease spread from configured SEIR rates and tests whether the
    outbreak is above the epidemic threshold.

    Features:
    - SEIR compartmental model integration on a daily grid
    - R0 = β/γ, and R estimated from observed daily incidence
    - Epidemic-threshold test: ``chaos_score`` is Λ, the early growth rate per
      day; ``bifurcation_detected`` is ``Λ > chaos_threshold`` (default 0, the
      R_eff = 1 threshold)
    - Impact from cumulative infections and configurable ratios
    - Intervention recommendation (vaccination, quarantine, treatment)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize epidemic forecaster.

        Args:
            config: Configuration dict with SEIR rates (``beta``, ``sigma``,
                ``gamma`` per day), ``population``, ``forecast_days``,
                ``chaos_threshold`` (per day) and the impact assumptions
                ``infection_fatality_ratio``, ``hospitalization_ratio`` and
                ``cost_per_infection_usd``.

        Raises:
            ValueError: If a rate, the population or the horizon is out of range.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        self.beta = _finite(self.config, "beta", 0.5, lower=0.0)
        self.sigma = _finite(self.config, "sigma", 0.2, lower=0.0, strict=True)
        self.gamma = _finite(self.config, "gamma", 0.1, lower=0.0, strict=True)

        self.population = _finite(self.config, "population", 1000000, lower=0.0, strict=True)
        days = _finite(self.config, "forecast_days", 365, lower=1.0)
        if not days.is_integer():
            raise ValueError(f"forecast_days must be a whole number, got {days}")
        self.forecast_days = int(days)

        # Λ per day; 0.0 is the epidemic threshold (the released score was unitless).
        self.chaos_threshold = _finite(self.config, "chaos_threshold", 0.0)

        # Illustrative planning assumptions, not properties of any pathogen.
        self.infection_fatality_ratio = _finite(
            self.config, "infection_fatality_ratio", 0.01, lower=0.0, upper=1.0
        )
        self.hospitalization_ratio = _finite(
            self.config, "hospitalization_ratio", 0.05, lower=0.0, upper=1.0
        )
        self.cost_per_infection_usd = _finite(
            self.config, "cost_per_infection_usd", 50000.0, lower=0.0
        )

        self.logger.info(
            f"EpidemicForecaster initialized (β={self.beta}, σ={self.sigma}, γ={self.gamma})"
        )

    def forecast_pandemic(
        self,
        initial_infected: int = 10,
        initial_exposed: int = 50,
        initial_recovered: int = 0,
    ) -> PandemicForecast:
        """Forecast pandemic progression using SEIR model.

        Args:
            initial_infected: Initial infected population
            initial_exposed: Initial exposed population
            initial_recovered: Initial recovered population

        Returns:
            PandemicForecast with trajectory and recommendations

        Raises:
            ValueError: If an initial count is negative or they exceed the population.
        """
        initial = (initial_infected, initial_exposed, initial_recovered)
        if min(initial) < 0 or sum(initial) > self.population:
            raise ValueError(
                f"initial counts must be >= 0 and sum to at most the population, got {initial}"
            )
        initial_susceptible = (
            self.population - initial_infected - initial_exposed - initial_recovered
        )

        y0 = [initial_susceptible, initial_exposed, initial_infected, initial_recovered]

        # One point per day, so an index on the grid is a day number.
        t = np.arange(self.forecast_days + 1, dtype=float)

        solution = odeint(self._seir_derivatives, y0, t)

        susceptible, exposed, infected, recovered = solution.T

        r0 = self._estimate_r0()

        peak_infections = int(np.max(infected))
        peak_day = int(np.argmax(infected))
        total_infections = round(float(self.population - susceptible[-1] - initial_recovered))

        outbreak_detected = peak_infections > (0.01 * self.population)

        chaos_score = self._growth_rate(initial_susceptible / self.population)
        bifurcation_detected = chaos_score > self.chaos_threshold

        seir_trajectory = {
            "susceptible": susceptible,
            "exposed": exposed,
            "infected": infected,
            "recovered": recovered,
            "time_days": t,
        }

        humanitarian_impact = self._assess_pandemic_impact(peak_infections, r0, total_infections)

        interventions = self._recommend_interventions(
            r0, peak_infections, peak_day, bifurcation_detected
        )

        result = PandemicForecast(
            outbreak_detected=outbreak_detected,
            r0_estimate=r0,
            peak_infections=peak_infections,
            peak_day=peak_day,
            seir_trajectory=seir_trajectory,
            chaos_score=chaos_score,
            bifurcation_detected=bifurcation_detected,
            humanitarian_impact=humanitarian_impact,
            recommended_interventions=interventions,
        )

        if outbreak_detected:
            self.logger.warning(
                f"Pandemic outbreak detected "
                f"(R0={r0:.2f}, peak={peak_infections:,} on day {peak_day})"
            )

        if bifurcation_detected and chaos_score > 0.0:
            self.logger.warning(
                f"Above the epidemic threshold (Λ={chaos_score:.3f}/day, "
                f"doubling time {np.log(2.0) / chaos_score:.1f} days)"
            )

        return result

    def _seir_derivatives(self, y: list[float], t: float) -> list[float]:
        """Compute SEIR model derivatives.

        Args:
            y: [S, E, I, R] state vector
            t: Time

        Returns:
            [dS/dt, dE/dt, dI/dt, dR/dt] derivatives
        """
        susceptible, exposed, infected, _recovered = y
        total_pop = self.population

        dS_dt = -self.beta * susceptible * infected / total_pop
        dE_dt = self.beta * susceptible * infected / total_pop - self.sigma * exposed
        dI_dt = self.sigma * exposed - self.gamma * infected
        dR_dt = self.gamma * infected

        return [dS_dt, dE_dt, dI_dt, dR_dt]

    def _estimate_r0(self) -> float:
        """Basic reproduction number R0 = β/γ.

        R0 > 1: Epidemic growth
        R0 < 1: Disease dies out

        Next-generation matrix (van den Driessche & Watmough 2002) for the
        infected compartments (E, I): F = [[0, β], [0, 0]],
        V = [[σ, 0], [-σ, γ]], so FV⁻¹ = [[β/γ, β/γ], [0, 0]] and
        R0 = ρ(FV⁻¹) = β/γ.
        """
        return float(self.beta / self.gamma)

    def _growth_rate(self, susceptible_fraction: float) -> float:
        """Dominant eigenvalue Λ of the linearised (E, I) subsystem, per day.

        The Jacobian at the disease-free state with S/N = s is
        [[-σ, β·s], [σ, -γ]]. Its largest eigenvalue is positive exactly when
        β·s/γ > 1, so ``Λ > 0`` marks the epidemic threshold, and Λ is the
        early exponential growth rate of E and I.
        """
        b, s, g = self.beta * susceptible_fraction, self.sigma, self.gamma
        return float((-(s + g) + np.sqrt((s - g) ** 2 + 4.0 * s * b)) / 2.0)

    def estimate_r0_from_incidence(self, daily_cases: np.ndarray[Any, Any]) -> float:
        """Estimate R from daily incidence in the exponential-growth phase.

        Fits ``log E[cases_t] = a + r·t`` by Poisson regression (iteratively
        reweighted least squares) and converts the growth rate with
        R = (1 + r/σ)(1 + r/γ) (Wallinga & Lipsitch 2007, eq. 3.2), using this
        forecaster's latent (1/σ) and infectious (1/γ) periods. Pass only the
        growth phase: once depletion of susceptibles slows growth, the estimate
        is biased low.

        Args:
            daily_cases: Non-negative daily case counts, oldest first.

        Returns:
            The estimated reproduction number.

        Raises:
            ValueError: If fewer than three days, a negative or non-finite count,
                no cases, or a growth rate at or below ``-min(σ, γ)``.
        """
        cases = np.asarray(daily_cases, dtype=float).ravel()
        if cases.size < 3 or not np.all(np.isfinite(cases)) or np.any(cases < 0.0):
            raise ValueError("daily_cases needs >= 3 finite, non-negative counts")
        if cases.sum() <= 0.0:
            raise ValueError("daily_cases has no cases")

        t = np.arange(cases.size, dtype=float)
        design = np.column_stack([np.ones_like(t), t])
        coef = np.array([np.log(cases.mean()), 0.0])
        converged = False
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for _ in range(100):
                mu = np.exp(design @ coef)
                # IRLS step for the Poisson log link: weights mu, working response z.
                z = design @ coef + (cases - mu) / mu
                weighted = design * mu[:, None]
                try:
                    step = np.linalg.solve(design.T @ weighted, weighted.T @ z) - coef
                except np.linalg.LinAlgError:
                    break
                if not np.all(np.isfinite(step)):
                    break
                coef = coef + step
                if np.max(np.abs(step)) < 1e-10:
                    converged = True
                    break
        if not converged:
            raise ValueError("the Poisson growth fit did not converge (too few non-zero days?)")
        growth_rate = float(coef[1])
        if growth_rate <= -min(self.sigma, self.gamma):
            raise ValueError(f"growth rate {growth_rate:.4f}/day is outside the model's range")
        return float((1.0 + growth_rate / self.sigma) * (1.0 + growth_rate / self.gamma))

    def _assess_pandemic_impact(
        self, peak_infections: int, r0: float, total_infections: int
    ) -> dict[str, Any]:
        """Assess humanitarian impact of pandemic.

        Deaths, hospitalisations and cost scale with cumulative infections over
        the horizon (prevalence at the peak drives only the healthcare strain).
        The ratios are the configured planning assumptions, reported under
        ``assumptions``; they are not estimates for any particular pathogen.
        """
        estimated_deaths = int(total_infections * self.infection_fatality_ratio)
        estimated_hospitalizations = int(total_infections * self.hospitalization_ratio)

        economic_impact = total_infections * self.cost_per_infection_usd

        vulnerable_populations = []
        if r0 > 3.0:
            vulnerable_populations.extend(
                ["Elderly (65+)", "Immunocompromised", "Healthcare workers"]
            )
        elif r0 > 2.0:
            vulnerable_populations.extend(["Elderly (65+)", "Immunocompromised"])
        elif r0 > 1.5:
            vulnerable_populations.append("Elderly (65+)")

        impact = {
            "total_infections": total_infections,
            "estimated_deaths": estimated_deaths,
            "estimated_hospitalizations": estimated_hospitalizations,
            "economic_impact_usd": economic_impact,
            "vulnerable_populations": vulnerable_populations,
            "healthcare_system_strain": (
                "CRITICAL"
                if peak_infections > 0.05 * self.population
                else "MODERATE" if peak_infections > 0.01 * self.population else "LOW"
            ),
            "assumptions": {
                "infection_fatality_ratio": self.infection_fatality_ratio,
                "hospitalization_ratio": self.hospitalization_ratio,
                "cost_per_infection_usd": self.cost_per_infection_usd,
            },
        }

        return impact

    def _recommend_interventions(
        self, r0: float, peak_infections: int, peak_day: int, bifurcation_detected: bool
    ) -> list[str]:
        """Recommend pandemic intervention strategies (Medical Interdiction).

        Interventions: vaccination, quarantine, social distancing, treatment,
                       healthcare surge capacity, public health messaging.
        """
        interventions = []

        if r0 > 3.0:
            interventions.append("URGENT: Implement strict quarantine and lockdown measures")
            interventions.append("Activate emergency vaccine production and distribution")
            interventions.append("Deploy national guard for logistics support")

        if r0 > 2.0:
            interventions.append("Implement social distancing protocols")
            interventions.append("Mandate mask usage in public spaces")
            interventions.append("Accelerate vaccine rollout to high-risk populations")

        if r0 > 1.5:
            interventions.append("Increase testing and contact tracing capacity")
            interventions.append("Prepare healthcare surge capacity")
            interventions.append("Launch public health awareness campaigns")

        if peak_infections > 0.05 * self.population:
            interventions.append("Activate field hospitals and emergency medical facilities")
            interventions.append("Request federal medical resource assistance (FEMA, CDC)")

        if bifurcation_detected:
            interventions.append(
                "Above the epidemic threshold (R_eff > 1): sustained spread expected "
                "without intervention"
            )
            if peak_infections > 0.01 * self.population:
                interventions.append("Activate international coordination (WHO, CDC, ECDC)")

        if bifurcation_detected and peak_day < 90:
            interventions.append(f"Rapid intervention required - peak expected in {peak_day} days")

        if not interventions:
            interventions.append("Continue monitoring - no immediate interventions required")

        return interventions

    def simulate_intervention_effect(
        self, intervention_type: str, effectiveness: float = 0.5
    ) -> dict[str, float]:
        """Simulate effect of intervention on pandemic trajectory.

        Args:
            intervention_type: 'vaccination', 'quarantine', 'social_distancing', 'treatment'
            effectiveness: Intervention effectiveness (0-1)

        Returns:
            Dict with r0_reduction, peak_reduction, delay_days, lives_saved_estimate

        Raises:
            ValueError: If ``effectiveness`` is outside [0, 1].
        """
        if not 0.0 <= effectiveness <= 1.0:
            raise ValueError(f"effectiveness must be in [0, 1], got {effectiveness}")
        r0_original = self._estimate_r0()

        if intervention_type == "vaccination":
            r0_reduction = effectiveness * 0.7
        elif intervention_type == "quarantine":
            r0_reduction = effectiveness * 0.6
        elif intervention_type == "social_distancing":
            r0_reduction = effectiveness * 0.4
        elif intervention_type == "treatment":
            r0_reduction = effectiveness * 0.2
        else:
            r0_reduction = 0.0

        # The 0.5 floor keeps residual transmission; it never raises R0.
        r0_new = max(r0_original * (1 - r0_reduction), min(0.5, r0_original))

        baseline_forecast = self.forecast_pandemic()
        baseline_peak = baseline_forecast.peak_infections

        original_beta = self.beta
        try:
            # R0 = β/γ, so scaling β scales R0 by the same factor.
            self.beta = original_beta * (r0_new / r0_original) if r0_original > 0 else 0.0
            intervention_forecast = self.forecast_pandemic()
        finally:
            self.beta = original_beta
        intervention_peak = intervention_forecast.peak_infections

        peak_reduction = (
            (baseline_peak - intervention_peak) / baseline_peak if baseline_peak > 0 else 0.0
        )

        delay_days = intervention_forecast.peak_day - baseline_forecast.peak_day

        infections_averted = (
            baseline_forecast.humanitarian_impact["total_infections"]
            - intervention_forecast.humanitarian_impact["total_infections"]
        )

        return {
            "r0_reduction": r0_reduction,
            "peak_reduction": peak_reduction,
            "delay_days": float(delay_days),
            "lives_saved_estimate": int(infections_averted * self.infection_fatality_ratio),
        }


def _finite(
    config: dict[str, Any],
    name: str,
    default: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    strict: bool = False,
) -> float:
    """A finite numeric setting within bounds (``strict`` excludes ``lower``)."""
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    in_range = np.isfinite(number)
    if lower is not None:
        in_range = in_range and (number > lower if strict else number >= lower)
    if upper is not None:
        in_range = in_range and number <= upper
    if not in_range:
        raise ValueError(f"{name} is out of range: {value!r}")
    return number
