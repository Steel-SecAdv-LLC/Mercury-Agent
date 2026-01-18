"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


"""
Epidemic Forecasting using SEIR Model with Chaos Detection

Implements Susceptible-Exposed-Infected-Recovered (SEIR) compartmental
model with chaos Λ detection for bifurcation analysis.

Mathematical Foundation:
- SEIR Dynamics: dS/dt = -β*S*I/N
                  dE/dt = β*S*I/N - σ*E
                  dI/dt = σ*E - γ*I
                  dR/dt = γ*I
- Chaos Λ: Detects bifurcations in epidemic curves (loyalty → compromise → exfiltration
          analogy adapted for viral spread: contained → outbreak → pandemic)
- R0 Threshold: Basic reproduction number R0 = β/(σ+γ)

References:
- Chaos detection: ethical_config.py (omni_chaos_lambda_bifurcation)
- Epidemiological literature: Kermack-McKendrick (1927), Anderson-May (1991)
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import odeint


@dataclass
class PandemicForecast:
    """Result from pandemic forecasting"""

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
    """
    SEIR-Based Pandemic Forecaster (Medical Interdiction)

    Forecasts disease spread dynamics with chaos detection for
    early warning of outbreak→pandemic transitions.

    Features:
    - SEIR compartmental model integration
    - Chaos Λ bifurcation detection
    - R0 estimation for transmissibility
    - Humanitarian impact assessment
    - Intervention recommendation (vaccination, quarantine, treatment)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize epidemic forecaster.

        Args:
            config: Configuration dict with SEIR parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        self.beta = self.config.get("beta", 0.5)
        self.sigma = self.config.get("sigma", 0.2)
        self.gamma = self.config.get("gamma", 0.1)

        self.population = self.config.get("population", 1000000)
        self.forecast_days = self.config.get("forecast_days", 365)

        self.chaos_threshold = self.config.get("chaos_threshold", 1.28)

        self.logger.info(
            f"EpidemicForecaster initialized (β={self.beta}, σ={self.sigma}, γ={self.gamma})"
        )

    def forecast_pandemic(
        self,
        initial_infected: int = 10,
        initial_exposed: int = 50,
        initial_recovered: int = 0,
    ) -> PandemicForecast:
        """
        Forecast pandemic progression using SEIR model.

        Args:
            initial_infected: Initial infected population
            initial_exposed: Initial exposed population
            initial_recovered: Initial recovered population

        Returns:
            PandemicForecast with trajectory and recommendations
        """
        initial_susceptible = (
            self.population - initial_infected - initial_exposed - initial_recovered
        )

        y0 = [initial_susceptible, initial_exposed, initial_infected, initial_recovered]

        t = np.linspace(0, self.forecast_days, self.forecast_days)

        solution = odeint(self._seir_derivatives, y0, t)

        susceptible, exposed, infected, recovered = solution.T

        r0 = self._estimate_r0()

        peak_infections = int(np.max(infected))
        peak_day = int(np.argmax(infected))

        outbreak_detected = peak_infections > (0.01 * self.population)

        chaos_score = self._detect_chaos(infected)
        bifurcation_detected = chaos_score > self.chaos_threshold

        seir_trajectory = {
            "susceptible": susceptible,
            "exposed": exposed,
            "infected": infected,
            "recovered": recovered,
            "time_days": t,
        }

        humanitarian_impact = self._assess_pandemic_impact(peak_infections, r0)

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

        if bifurcation_detected:
            self.logger.error(
                f"Critical bifurcation detected (chaos={chaos_score:.3f}) - "
                "transition to pandemic imminent"
            )

        return result

    def _seir_derivatives(self, y: list[float], t: float) -> list[float]:
        """
        Compute SEIR model derivatives.

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
        """
        Estimate basic reproduction number R0 = β/(σ+γ).

        R0 > 1: Epidemic growth
        R0 = 1: Endemic equilibrium
        R0 < 1: Disease dies out

        Proof:
        Next-generation matrix method (Diekmann et al. 1990):
        R0 = ρ(FV^-1) where F is infection matrix, V is transition matrix.
        For SEIR: R0 simplifies to β/(σ+γ).
        """
        r0 = self.beta / (self.sigma + self.gamma)
        return float(r0)

    def _detect_chaos(self, infected_trajectory: np.ndarray[Any, Any]) -> float:
        """
        Detect chaos/bifurcations in infected trajectory.

        Uses variance and derivative analysis to identify transitions:
        - Contained outbreak (stable, low chaos)
        - Active outbreak (moderate chaos)
        - Pandemic (high chaos, exponential growth)

        Analogy to CI: loyalty → compromise → exfiltration
        Pandemic: contained → outbreak → pandemic

        Returns:
            Chaos score (higher = more chaotic/bifurcating)
        """
        if len(infected_trajectory) < 2:
            return 0.0

        derivatives = np.diff(infected_trajectory)

        second_derivatives = np.diff(derivatives)

        variance = np.var(second_derivatives)

        exponential_growth = np.sum(derivatives > 0) / len(derivatives)

        chaos_score = variance * 0.5 + exponential_growth * 0.5

        chaos_score = min(chaos_score * 10.0, 5.0)

        return float(chaos_score)

    def _assess_pandemic_impact(self, peak_infections: int, r0: float) -> dict[str, Any]:
        """
        Assess humanitarian impact of pandemic.

        Simulated estimates for research purposes.
        """
        mortality_rate = 0.01
        hospitalization_rate = 0.05

        estimated_deaths = int(peak_infections * mortality_rate)
        estimated_hospitalizations = int(peak_infections * hospitalization_rate)

        economic_impact = peak_infections * 50000

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
            "estimated_deaths": estimated_deaths,
            "estimated_hospitalizations": estimated_hospitalizations,
            "economic_impact_usd": economic_impact,
            "vulnerable_populations": vulnerable_populations,
            "healthcare_system_strain": (
                "CRITICAL"
                if peak_infections > 0.05 * self.population
                else "MODERATE" if peak_infections > 0.01 * self.population else "LOW"
            ),
        }

        return impact

    def _recommend_interventions(
        self, r0: float, peak_infections: int, peak_day: int, bifurcation_detected: bool
    ) -> list[str]:
        """
        Recommend pandemic intervention strategies (Medical Interdiction).

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
                "CRITICAL BIFURCATION: Pandemic transition imminent - escalate all interventions"
            )
            interventions.append("Activate international coordination (WHO, CDC, ECDC)")

        if peak_day < 90:
            interventions.append(f"Rapid intervention required - peak expected in {peak_day} days")

        if not interventions:
            interventions.append("Continue monitoring - no immediate interventions required")

        return interventions

    def simulate_intervention_effect(
        self, intervention_type: str, effectiveness: float = 0.5
    ) -> dict[str, float]:
        """
        Simulate effect of intervention on pandemic trajectory.

        Args:
            intervention_type: 'vaccination', 'quarantine', 'social_distancing', 'treatment'
            effectiveness: Intervention effectiveness (0-1)

        Returns:
            Dict with r0_reduction, peak_reduction, delay_days
        """
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

        r0_new = max(r0_original * (1 - r0_reduction), 0.5)

        baseline_forecast = self.forecast_pandemic()
        baseline_peak = baseline_forecast.peak_infections

        self.beta = self.beta * (r0_new / r0_original)
        intervention_forecast = self.forecast_pandemic()
        intervention_peak = intervention_forecast.peak_infections

        self.beta = self.beta * (r0_original / r0_new)

        peak_reduction = (baseline_peak - intervention_peak) / baseline_peak

        delay_days = intervention_forecast.peak_day - baseline_forecast.peak_day

        return {
            "r0_reduction": r0_reduction,
            "peak_reduction": peak_reduction,
            "delay_days": float(delay_days),
            "lives_saved_estimate": int((baseline_peak - intervention_peak) * 0.01),
        }
