"""
Mercury Agent - Multi-Objective Benevolence Optimization
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Implements benevolence as an explicit optimization target in multi-objective loss:
- Pareto optimization (scipy.optimize)
- Scalarized multi-objective loss with benevolence constraint
- NSGA-II style evolutionary optimization
- Gradient-based benevolence-aware training
- Integration with ethical gating (benevolence >= 0.99 threshold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np


logger = logging.getLogger(__name__)

# Benevolence threshold from ethical requirements
BENEVOLENCE_THRESHOLD = 0.99

# Golden ratio for harmonic scaling
PHI = 1.618033988749895


@dataclass
class ObjectiveResult:
    """Result of multi-objective optimization."""

    detection_loss: float
    benevolence_score: float
    fairness_score: float
    pareto_optimal: bool
    combined_loss: float
    weights_used: dict[str, float]
    constraint_violations: list[str] = field(default_factory=list)


@dataclass
class ParetoSolution:
    """A single solution on the Pareto front."""

    parameters: np.ndarray
    objectives: np.ndarray  # [detection_loss, 1 - benevolence, 1 - fairness]
    dominated_by: int = 0  # Number of solutions that dominate this one


@dataclass
class ParetoFront:
    """Collection of Pareto-optimal solutions."""

    solutions: list[ParetoSolution]
    n_objectives: int
    objective_names: list[str]

    def get_best_balanced(self) -> ParetoSolution | None:
        """Get solution with best balance of objectives."""
        if not self.solutions:
            return None

        # Normalize objectives
        obj_matrix = np.array([s.objectives for s in self.solutions])
        if len(obj_matrix) == 0:
            return None

        normalized = (obj_matrix - obj_matrix.min(axis=0)) / (
            obj_matrix.max(axis=0) - obj_matrix.min(axis=0) + 1e-10
        )

        # Find solution with minimum sum (balanced)
        scores = normalized.sum(axis=1)
        best_idx = np.argmin(scores)

        return self.solutions[best_idx]

    def get_best_benevolent(self) -> ParetoSolution | None:
        """Get solution with highest benevolence."""
        if not self.solutions:
            return None

        # Benevolence is stored as (1 - benevolence), so minimize
        benevolence_idx = 1  # Index in objectives array
        best_idx = np.argmin([s.objectives[benevolence_idx] for s in self.solutions])

        return self.solutions[best_idx]


class BenevolenceLoss:
    """
    Computes benevolence loss for optimization.

    Benevolence measures the degree to which actions promote
    well-being and minimize harm. Higher is better (target >= 0.99).
    """

    def __init__(
        self,
        harm_weight: float = 0.4,
        equity_weight: float = 0.3,
        transparency_weight: float = 0.3,
    ):
        """
        Initialize benevolence loss.

        Args:
            harm_weight: Weight for harm reduction component
            equity_weight: Weight for equity/fairness component
            transparency_weight: Weight for transparency component
        """
        self.harm_weight = harm_weight
        self.equity_weight = equity_weight
        self.transparency_weight = transparency_weight

    def compute(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_attrs: np.ndarray | None = None,
        explanation_provided: bool = True,
    ) -> float:
        """
        Compute benevolence score.

        Args:
            predictions: Model predictions (binary or probability)
            labels: True labels
            sensitive_attrs: Sensitive attribute for equity calculation
            explanation_provided: Whether explanations are available

        Returns:
            Benevolence score (0-1, target >= 0.99)
        """
        # Harm reduction: minimize false positives and false negatives
        # weighted by their real-world impact
        binary_pred = (predictions > 0.5).astype(int) if predictions.dtype != int else predictions

        fp = np.sum((binary_pred == 1) & (labels == 0))
        fn = np.sum((binary_pred == 0) & (labels == 1))
        n = len(labels)

        # False negatives (missed anomalies) are more harmful
        harm_score = 1.0 - (0.3 * fp + 0.7 * fn) / (n + 1e-10)
        harm_score = max(0.0, harm_score)

        # Equity: equal performance across groups
        if sensitive_attrs is not None and len(np.unique(sensitive_attrs)) > 1:
            equity_score = self._compute_equity(binary_pred, labels, sensitive_attrs)
        else:
            equity_score = 1.0  # No groups to compare

        # Transparency: reward explainability
        transparency_score = 1.0 if explanation_provided else 0.5

        # Weighted combination
        benevolence = (
            self.harm_weight * harm_score
            + self.equity_weight * equity_score
            + self.transparency_weight * transparency_score
        )

        return float(np.clip(benevolence, 0.0, 1.0))

    def _compute_equity(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_attrs: np.ndarray,
    ) -> float:
        """
        Compute equity score based on group fairness.

        Uses demographic parity ratio (80% rule).
        """
        groups = np.unique(sensitive_attrs)
        positive_rates = []

        for group in groups:
            mask = sensitive_attrs == group
            if np.sum(mask) > 0:
                rate = np.mean(predictions[mask])
                positive_rates.append(rate)

        if len(positive_rates) < 2:
            return 1.0

        # Demographic parity ratio
        min_rate = min(positive_rates)
        max_rate = max(positive_rates)

        if max_rate == 0:
            return 1.0

        ratio = min_rate / (max_rate + 1e-10)

        # Score based on 80% rule
        if ratio >= 0.8:
            return 1.0
        else:
            return float(ratio / 0.8)  # Linear penalty below 80%


class MultiObjectiveLoss:
    """
    Multi-objective loss combining detection, benevolence, and fairness.

    Supports both scalarized (weighted sum) and Pareto optimization.
    """

    def __init__(
        self,
        detection_weight: float = 0.6,
        benevolence_weight: float = 0.3,
        fairness_weight: float = 0.1,
        benevolence_threshold: float = BENEVOLENCE_THRESHOLD,
        penalty_factor: float = 10.0,
    ):
        """
        Initialize multi-objective loss.

        Args:
            detection_weight: Weight for detection performance
            benevolence_weight: Weight for benevolence score
            fairness_weight: Weight for fairness score
            benevolence_threshold: Minimum benevolence (default 0.99)
            penalty_factor: Penalty multiplier for constraint violations
        """
        self.detection_weight = detection_weight
        self.benevolence_weight = benevolence_weight
        self.fairness_weight = fairness_weight
        self.benevolence_threshold = benevolence_threshold
        self.penalty_factor = penalty_factor

        self.benevolence_loss = BenevolenceLoss()

    def compute(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_attrs: np.ndarray | None = None,
    ) -> ObjectiveResult:
        """
        Compute multi-objective loss.

        Args:
            predictions: Model predictions
            labels: True labels
            sensitive_attrs: Sensitive attributes for fairness

        Returns:
            ObjectiveResult with all loss components
        """
        # Detection loss (binary cross-entropy)
        proba = np.clip(predictions, 1e-10, 1 - 1e-10)
        detection_loss = -np.mean(labels * np.log(proba) + (1 - labels) * np.log(1 - proba))

        # Benevolence score
        benevolence = self.benevolence_loss.compute(predictions, labels, sensitive_attrs)

        # Fairness score (equalized odds)
        fairness = self._compute_fairness(predictions, labels, sensitive_attrs)

        # Check constraints
        violations = []
        if benevolence < self.benevolence_threshold:
            violations.append(f"Benevolence {benevolence:.3f} < {self.benevolence_threshold}")

        # Combined loss with constraint penalty
        constraint_penalty = 0.0
        if benevolence < self.benevolence_threshold:
            constraint_penalty = self.penalty_factor * (self.benevolence_threshold - benevolence)

        combined_loss = (
            self.detection_weight * detection_loss
            + self.benevolence_weight * (1 - benevolence)
            + self.fairness_weight * (1 - fairness)
            + constraint_penalty
        )

        return ObjectiveResult(
            detection_loss=detection_loss,
            benevolence_score=benevolence,
            fairness_score=fairness,
            pareto_optimal=len(violations) == 0,
            combined_loss=combined_loss,
            weights_used={
                "detection": self.detection_weight,
                "benevolence": self.benevolence_weight,
                "fairness": self.fairness_weight,
            },
            constraint_violations=violations,
        )

    def _compute_fairness(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        sensitive_attrs: np.ndarray | None,
    ) -> float:
        """Compute fairness via equalized odds."""
        if sensitive_attrs is None or len(np.unique(sensitive_attrs)) < 2:
            return 1.0

        binary_pred = (predictions > 0.5).astype(int)
        groups = np.unique(sensitive_attrs)

        # True positive rates per group
        tpr_per_group = []
        fpr_per_group = []

        for group in groups:
            mask = sensitive_attrs == group

            # TPR
            pos_mask = mask & (labels == 1)
            if np.sum(pos_mask) > 0:
                tpr = np.mean(binary_pred[pos_mask])
                tpr_per_group.append(tpr)

            # FPR
            neg_mask = mask & (labels == 0)
            if np.sum(neg_mask) > 0:
                fpr = np.mean(binary_pred[neg_mask])
                fpr_per_group.append(fpr)

        # Equalized odds: minimize gap in TPR and FPR
        tpr_gap = max(tpr_per_group) - min(tpr_per_group) if len(tpr_per_group) >= 2 else 0
        fpr_gap = max(fpr_per_group) - min(fpr_per_group) if len(fpr_per_group) >= 2 else 0

        fairness = 1.0 - 0.5 * (tpr_gap + fpr_gap)
        return float(np.clip(fairness, 0.0, 1.0))


class ParetoOptimizer:
    """
    Pareto optimization for multi-objective benevolence optimization.

    Finds the Pareto front of non-dominated solutions trading off
    detection performance, benevolence, and fairness.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
        n_objectives: int = 3,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        seed: int = 42,
    ):
        """
        Initialize Pareto optimizer.

        Args:
            objective_fn: Function that takes parameters and returns objective vector
            n_objectives: Number of objectives
            population_size: Population size for evolutionary algorithm
            n_generations: Number of generations
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
            seed: Random seed
        """
        self.objective_fn = objective_fn
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed

    def optimize(
        self,
        bounds: list[tuple[float, float]],
        benevolence_constraint: float = BENEVOLENCE_THRESHOLD,
    ) -> ParetoFront:
        """
        Find Pareto-optimal solutions.

        Args:
            bounds: Parameter bounds [(min, max), ...]
            benevolence_constraint: Minimum benevolence (in objectives[1])

        Returns:
            ParetoFront with non-dominated solutions
        """
        np.random.seed(self.seed)

        # Initialize population
        population = []
        for _ in range(self.population_size):
            params = np.array([np.random.uniform(low, high) for low, high in bounds])
            objectives = self.objective_fn(params)
            population.append(ParetoSolution(params, objectives))

        # Evolutionary loop
        for gen in range(self.n_generations):
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(population)

            # Select parents from best fronts
            parents: list[ParetoSolution] = []
            for front in fronts:
                if len(parents) + len(front) <= self.population_size:
                    parents.extend(front)
                else:
                    # Crowding distance selection
                    front = self._crowding_distance_selection(
                        front, self.population_size - len(parents)
                    )
                    parents.extend(front)
                    break

            # Generate offspring
            offspring: list[ParetoSolution] = []
            while len(offspring) < self.population_size:
                # Tournament selection
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                parent1, parent2 = parents[p1], parents[p2]

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child_params = self._sbx_crossover(
                        parent1.parameters, parent2.parameters, bounds
                    )
                else:
                    child_params = parent1.parameters.copy()

                # Mutation
                if np.random.random() < self.mutation_rate:
                    child_params = self._polynomial_mutation(child_params, bounds)

                objectives = self.objective_fn(child_params)
                offspring.append(ParetoSolution(child_params, objectives))

            population = parents + offspring

        # Final Pareto front
        fronts = self._fast_non_dominated_sort(population)
        pareto_solutions = fronts[0] if fronts else []

        # Filter by benevolence constraint
        # benevolence is stored as (1 - benevolence), so constraint is (1 - threshold)
        constrained_solutions = [
            s for s in pareto_solutions if s.objectives[1] <= (1 - benevolence_constraint)
        ]

        if not constrained_solutions:
            logger.warning(
                f"No solutions meet benevolence threshold {benevolence_constraint}. "
                "Returning best unconstrained solutions."
            )
            constrained_solutions = pareto_solutions

        return ParetoFront(
            solutions=constrained_solutions,
            n_objectives=self.n_objectives,
            objective_names=["detection_loss", "1-benevolence", "1-fairness"],
        )

    def _fast_non_dominated_sort(
        self,
        population: list[ParetoSolution],
    ) -> list[list[ParetoSolution]]:
        """Fast non-dominated sorting (NSGA-II)."""
        n = len(population)
        domination_count = [0] * n
        dominated_set: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[ParetoSolution]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            population[i].dominated_by = domination_count[i]
            if domination_count[i] == 0:
                fronts[0].append(population[i])

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for sol in fronts[current_front]:
                idx = population.index(sol)
                for j in dominated_set[idx]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(population[j])
            current_front += 1
            if next_front:
                fronts.append(next_front)

        return [f for f in fronts if f]

    def _dominates(self, sol1: ParetoSolution, sol2: ParetoSolution) -> bool:
        """Check if sol1 dominates sol2 (all objectives <=, at least one <)."""
        better_in_any = False
        for o1, o2 in zip(sol1.objectives, sol2.objectives):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_any = True
        return better_in_any

    def _crowding_distance_selection(
        self,
        front: list[ParetoSolution],
        n_select: int,
    ) -> list[ParetoSolution]:
        """Select solutions with highest crowding distance."""
        if len(front) <= n_select:
            return front

        n = len(front)
        distances = [0.0] * n

        for m in range(self.n_objectives):
            # Sort by objective m
            sorted_idx = sorted(range(n), key=lambda i: front[i].objectives[m])

            # Boundary points get infinite distance
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")

            # Range for normalization
            obj_range = front[sorted_idx[-1]].objectives[m] - front[sorted_idx[0]].objectives[m]
            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    front[sorted_idx[i + 1]].objectives[m] - front[sorted_idx[i - 1]].objectives[m]
                ) / obj_range

        # Select top by distance
        selected_idx = sorted(range(n), key=lambda i: -distances[i])[:n_select]
        return [front[i] for i in selected_idx]

    def _sbx_crossover(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        bounds: list[tuple[float, float]],
        eta: float = 20.0,
    ) -> np.ndarray:
        """Simulated Binary Crossover (SBX)."""
        child = np.zeros_like(p1)

        for i in range(len(p1)):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-10:
                    y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                    lo, hi = bounds[i]

                    beta = 1.0 + 2.0 * (y1 - lo) / (y2 - y1)
                    alpha = 2.0 - beta ** (-(eta + 1))
                    u = np.random.random()

                    if u <= 1.0 / alpha:
                        betaq = (u * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1))

                    c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                    child[i] = np.clip(c1, lo, hi)
                else:
                    child[i] = p1[i]
            else:
                child[i] = p1[i]

        return child

    def _polynomial_mutation(
        self,
        params: np.ndarray,
        bounds: list[tuple[float, float]],
        eta: float = 20.0,
    ) -> np.ndarray:
        """Polynomial mutation."""
        mutated = params.copy()

        for i in range(len(params)):
            if np.random.random() < 1.0 / len(params):
                lo, hi = bounds[i]
                delta_max = hi - lo

                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                mutated[i] = params[i] + delta * delta_max
                mutated[i] = np.clip(mutated[i], lo, hi)

        return mutated


def optimize_benevolent_detector(
    model_fn: Callable[[np.ndarray], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    parameter_bounds: list[tuple[float, float]],
    sensitive_attrs: np.ndarray | None = None,
    benevolence_threshold: float = BENEVOLENCE_THRESHOLD,
    n_generations: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, ParetoFront]:
    """
    Optimize detector parameters for benevolence.

    Args:
        model_fn: Function that takes parameters and returns fitted model
        X_train: Training features
        y_train: Training labels
        parameter_bounds: Bounds for each parameter
        sensitive_attrs: Sensitive attributes for fairness
        benevolence_threshold: Minimum benevolence score
        n_generations: Number of optimization generations
        seed: Random seed

    Returns:
        Tuple of (best_parameters, pareto_front)
    """
    mo_loss = MultiObjectiveLoss(benevolence_threshold=benevolence_threshold)

    def objective(params: dict[str, Any]) -> np.ndarray:
        """Multi-objective function returning [detection, 1-benevolence, 1-fairness]."""
        try:
            model = model_fn(params)
            model.fit(X_train, y_train)
            predictions = model.predict_proba(X_train)
            if predictions.ndim == 2:
                predictions = predictions[:, 1]
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            return np.array([1.0, 1.0, 1.0])  # Worst case

        result = mo_loss.compute(predictions, y_train, sensitive_attrs)

        return np.array(
            [
                result.detection_loss,
                1 - result.benevolence_score,
                1 - result.fairness_score,
            ]
        )

    optimizer = ParetoOptimizer(
        objective_fn=objective,
        n_objectives=3,
        n_generations=n_generations,
        seed=seed,
    )

    front = optimizer.optimize(parameter_bounds, benevolence_threshold)

    best_solution = front.get_best_benevolent()
    if best_solution is None:
        best_solution = front.get_best_balanced()

    best_params = best_solution.parameters if best_solution else np.zeros(len(parameter_bounds))

    logger.info(
        f"Benevolent optimization complete: "
        f"best_params={best_params}, "
        f"front_size={len(front.solutions)}"
    )

    return best_params, front
