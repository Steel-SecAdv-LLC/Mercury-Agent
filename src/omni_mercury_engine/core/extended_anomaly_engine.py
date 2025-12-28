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
Extended Mercury Agent ♱ with 14-Engine Integration
Production-ready anomaly detection with 3R mechanism
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class EngineConfig:
    enable_fusion: bool = True
    enable_3r_mechanism: bool = True
    max_recursion_depth: int = 5
    resonance_sampling_rate: float = 1.0
    device: str = "cpu"
    batch_size: int = 32
    enable_security: bool = True
    enable_evolution: bool = True


class EvolutionStrategy(Enum):
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    HILL_CLIMBING = "hill_climbing"
    SIMULATED_ANNEALING = "simulated_annealing"


class EvolutionEngine:
    def __init__(
        self,
        state_dim: int = 100,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        rng: DeterministicRNG | None = None,
    ):
        self.state_dim = state_dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self._rng = rng or get_global_rng()
        self.population = self._initialize_population()
        self.best_fitness = -np.inf
        self.best_solution = None
        self.generation_count = 0

    def _initialize_population(self) -> np.ndarray[Any, Any]:
        return self._rng.randn(self.population_size, self.state_dim) * 0.1

    def evaluate_fitness(
        self, individual: np.ndarray[Any, Any], fitness_fn: Callable[[np.ndarray[Any, Any]], float]
    ) -> float:
        return float(fitness_fn(individual))

    def evolve_generation(
        self, fitness_fn: Callable[[np.ndarray[Any, Any]], float]
    ) -> dict[str, Any]:
        fitness_scores = np.array(
            [self.evaluate_fitness(ind, fitness_fn) for ind in self.population]
        )

        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_solution = self.population[best_idx].copy()

        selected = self._selection(fitness_scores)
        offspring = self._crossover(selected)
        offspring = self._mutation(offspring)

        self.population = offspring
        self.generation_count += 1

        return {
            "generation": self.generation_count,
            "best_fitness": self.best_fitness,
            "mean_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
        }

    def _selection(self, fitness_scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        selected = []
        for _ in range(self.population_size):
            tournament_idx = self._rng.choice(self.population_size, size=3, replace=False)
            tournament_fitness = fitness_scores[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx])
        return np.array(selected)

    def _crossover(self, selected: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

            if self._rng.rand() < self.crossover_rate:
                crossover_point = self._rng.randint(1, self.state_dim)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        return np.array(offspring[: self.population_size])

    def _mutation(self, offspring: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        for i in range(len(offspring)):
            if self._rng.rand() < self.mutation_rate:
                mutation_idx = self._rng.randint(0, self.state_dim)
                offspring[i, mutation_idx] += self._rng.randn() * 0.1
        return offspring


class SecurityEngine:
    def __init__(self) -> None:
        self.threat_patterns = self._load_threat_patterns()
        self.rate_limit_window = 60
        self.rate_limit_max = 60
        self.request_history: dict[str, list[float]] = {}

    def _load_threat_patterns(self) -> dict[str, list[str]]:
        return {
            "sql_injection": [
                r"(?i)(union.*select|select.*from|insert.*into|delete.*from)",
                r"(?i)(drop.*table|exec\s*\(|execute\s*\()",
                r"(?i)(\bor\b.*=.*|'\s*or\s*'1'\s*=\s*'1)",
            ],
            "xss": [
                r"(?i)(<script|javascript:|onerror=|onload=)",
                r"(?i)(<iframe|<object|<embed)",
                r"(?i)(eval\s*\(|alert\s*\()",
            ],
            "path_traversal": [r"\.\.\/|\.\.\\", r"(?i)(\/etc\/passwd|\/etc\/shadow)"],
        }

    def detect_threats(self, input_data: str) -> dict[str, Any]:
        import re

        threats = []

        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data):
                    threats.append({"type": threat_type, "pattern": pattern, "severity": "high"})

        return {"is_threat": len(threats) > 0, "threats": threats, "num_threats": len(threats)}

    def check_rate_limit(self, identifier: str) -> bool:
        import time

        current_time = time.time()

        if identifier not in self.request_history:
            self.request_history[identifier] = []

        recent_requests = [
            t for t in self.request_history[identifier] if current_time - t < self.rate_limit_window
        ]

        self.request_history[identifier] = recent_requests

        if len(recent_requests) >= self.rate_limit_max:
            return False

        self.request_history[identifier].append(current_time)
        return True


class IntegrationEngine:
    def __init__(self) -> None:
        self.integrations: dict[str, dict[str, Any]] = {}

    def register_integration(
        self,
        integration_id: str,
        endpoint_url: str,
        auth_type: str = "api_key",
        rate_limit: int = 1000,
    ) -> None:
        self.integrations[integration_id] = {
            "endpoint_url": endpoint_url,
            "auth_type": auth_type,
            "rate_limit": rate_limit,
            "status": "active",
        }
        logging.info(f"Registered integration: {integration_id}")

    def make_request(
        self,
        integration_id: str,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if integration_id not in self.integrations:
            return {"error": f"Unknown integration: {integration_id}"}

        return {
            "integration_id": integration_id,
            "method": method,
            "endpoint": endpoint,
            "status": "simulated",
            "message": "Integration request simulated (no actual HTTP call)",
        }


class OmniMercury:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

        if self.config.enable_3r_mechanism:
            from .three_r_mechanism import ThreeRMechanism

            self.three_r: ThreeRMechanism | None = ThreeRMechanism(
                max_recursion_depth=self.config.max_recursion_depth,
                sampling_rate=self.config.resonance_sampling_rate,
            )
        else:
            self.three_r = None

        self.evolution_engine: EvolutionEngine | None = (
            EvolutionEngine() if self.config.enable_evolution else None
        )

        self.security_engine: SecurityEngine | None = (
            SecurityEngine() if self.config.enable_security else None
        )

        self.integration_engine: IntegrationEngine = IntegrationEngine()

        logging.info("Mercury Agent ♱ initialized with full integration")

    def detect_anomaly(
        self, data: np.ndarray[Any, Any], use_3r_enhancement: bool = True
    ) -> dict[str, Any]:
        enhanced_data = data

        if use_3r_enhancement and self.config.enable_3r_mechanism and self.three_r:
            enhanced_data = self.three_r.enhance_features(data)

        anomaly_score = self._compute_anomaly_score(enhanced_data)

        is_anomaly = anomaly_score > 0.5

        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "data_shape": data.shape,
            "enhanced_shape": enhanced_data.shape if use_3r_enhancement else data.shape,
            "3r_applied": use_3r_enhancement and self.three_r is not None,
        }

        if self.config.enable_3r_mechanism and self.three_r and len(data) > 10:
            resonance_result = self.three_r.detect_with_resonance(data.flatten())
            result["resonance_analysis"] = resonance_result

        return result

    def _compute_anomaly_score(self, data: np.ndarray[Any, Any]) -> float:
        if data.size == 0:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        z_scores = np.abs((data - mean) / std)
        max_z_score = np.max(z_scores)

        anomaly_score = 1.0 / (1.0 + np.exp(-0.5 * (max_z_score - 3.0)))

        return float(anomaly_score)

    def validate_input_security(self, input_data: str) -> dict[str, Any]:
        if not self.config.enable_security or not self.security_engine:
            return {"secure": True, "message": "Security checks disabled"}

        return self.security_engine.detect_threats(input_data)

    def evolve_detector(
        self, fitness_fn: Callable[[np.ndarray[Any, Any]], float], num_generations: int = 10
    ) -> dict[str, Any]:
        if not self.config.enable_evolution or not self.evolution_engine:
            return {"error": "Evolution disabled"}

        results = []
        for _ in range(num_generations):
            gen_result = self.evolution_engine.evolve_generation(fitness_fn)
            results.append(gen_result)

        return {
            "num_generations": num_generations,
            "final_best_fitness": self.evolution_engine.best_fitness,
            "best_solution": self.evolution_engine.best_solution,
            "generation_history": results,
        }
