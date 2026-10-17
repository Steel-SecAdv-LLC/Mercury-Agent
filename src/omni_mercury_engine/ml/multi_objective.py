# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""NSGA-II multi-objective optimisation over bounded continuous parameters.

Deb, Pratap, Agarwal & Meyarivan (2002), "A fast and elitist multiobjective
genetic algorithm: NSGA-II", IEEE TEC 6(2). Every objective is minimised.

One generation:

1. Binary tournament on (non-domination rank, crowding distance) picks parents.
2. Simulated binary crossover (SBX, distribution index ``eta_c``) produces two
   children per pair; bounded polynomial mutation (``eta_m``) perturbs each
   gene with probability ``mutation_rate`` (default ``1 / n_params``).
3. Parents and offspring are merged (elitism), fast non-dominated sorting
   ranks the ``2N`` pool, and the next population is filled front by front,
   truncating the last front by crowding distance.

The optimiser is deterministic for a given ``seed``: the generator is reset at
the start of every :meth:`NSGA2Optimizer.optimize` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["NSGA2Optimizer", "ParetoFront", "ParetoSolution"]


@dataclass
class ParetoSolution:
    """One evaluated individual.

    Attributes:
        parameters: Parameter vector inside the optimisation bounds.
        objectives: Objective vector (every component is minimised).
        rank: Non-domination rank (0 = first front).
        crowding: Crowding distance within its front (``inf`` at the edges).
        dominated_by: Number of individuals that dominated this one when it
            was last sorted.
    """

    parameters: np.ndarray[Any, Any]
    objectives: np.ndarray[Any, Any]
    rank: int = 0
    crowding: float = 0.0
    dominated_by: int = 0


@dataclass
class ParetoFront:
    """The non-dominated set returned by :meth:`NSGA2Optimizer.optimize`.

    Attributes:
        solutions: Non-dominated solutions, duplicates removed.
        n_objectives: Length of every objective vector.
        objective_names: Human-readable name per objective.
    """

    solutions: list[ParetoSolution]
    n_objectives: int
    objective_names: list[str] = field(default_factory=list)

    def objective_matrix(self) -> np.ndarray[Any, Any]:
        """Return the ``(n_solutions, n_objectives)`` objective matrix."""
        if not self.solutions:
            return np.empty((0, self.n_objectives))
        return np.vstack([s.objectives for s in self.solutions])

    def best_for(self, objective_index: int) -> ParetoSolution | None:
        """Return the solution minimising one objective (ties: lowest sum)."""
        if not self.solutions:
            return None
        matrix = self.objective_matrix()
        order = np.lexsort((matrix.sum(axis=1), matrix[:, objective_index]))
        return self.solutions[int(order[0])]

    def knee(self) -> ParetoSolution | None:
        """Return the most balanced solution.

        Objectives are min-max normalised over the front and the solution
        with the smallest normalised sum (closest to the utopia point in the
        L1 sense) is returned.
        """
        if not self.solutions:
            return None
        matrix = self.objective_matrix()
        span = matrix.max(axis=0) - matrix.min(axis=0)
        span = np.where(span > 0, span, 1.0)
        normalised = (matrix - matrix.min(axis=0)) / span
        return self.solutions[int(np.argmin(normalised.sum(axis=1)))]

    def get_best_balanced(self) -> ParetoSolution | None:
        """Alias of :meth:`knee` kept for callers of the historical API."""
        return self.knee()


def dominates(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> bool:
    """Return whether objective vector ``a`` Pareto-dominates ``b``."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray[Any, Any]) -> list[list[int]]:
    """Partition row indices of ``objectives`` into non-domination fronts.

    Args:
        objectives: ``(n, m)`` matrix; every column is minimised.

    Returns:
        Fronts as lists of row indices, best front first.
    """
    n = objectives.shape[0]
    if n == 0:
        return []
    le = np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2)
    lt = np.any(objectives[:, None, :] < objectives[None, :, :], axis=2)
    dom = le & lt  # dom[i, j]: i dominates j
    domination_count = dom.sum(axis=0).astype(int)
    fronts: list[list[int]] = []
    current = [int(i) for i in np.flatnonzero(domination_count == 0)]
    remaining = domination_count.copy()
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for i in current:
            for j in np.flatnonzero(dom[i]):
                remaining[j] -= 1
                if remaining[j] == 0:
                    nxt.append(int(j))
        current = nxt
    return fronts


def crowding_distance(objectives: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Crowding distance of each row within one front.

    Boundary solutions of every objective get ``inf`` so they are always
    retained; interior distances are normalised by each objective's range.
    """
    n, m = objectives.shape
    distance = np.zeros(n)
    if n <= 2:
        distance[:] = np.inf
        return distance
    for k in range(m):
        order = np.argsort(objectives[:, k], kind="stable")
        values = objectives[order, k]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        if not (np.isfinite(values[0]) and np.isfinite(values[-1])):
            continue  # a front of invalid (+inf) points has no spread to measure
        span = values[-1] - values[0]
        if span <= 0:
            continue
        distance[order[1:-1]] += (values[2:] - values[:-2]) / span
    return distance


class NSGA2Optimizer:
    """Elitist non-dominated sorting genetic algorithm (NSGA-II).

    Args:
        objective_fn: Maps a parameter vector to an objective vector of
            length ``n_objectives``; every objective is minimised.
        n_objectives: Number of objectives.
        population_size: Individuals per generation (rounded up to even).
        n_generations: Generations after the initial population.
        crossover_rate: Probability a parent pair is recombined by SBX.
        mutation_rate: Per-gene mutation probability; ``None`` uses
            ``1 / n_params``.
        eta_c: SBX distribution index (larger = children nearer parents).
        eta_m: Polynomial-mutation distribution index.
        seed: Seed of the per-run generator.
        objective_names: Optional names reported on the returned front.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any] | Sequence[float]],
        n_objectives: int,
        population_size: int = 60,
        n_generations: int = 80,
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,
        eta_c: float = 15.0,
        eta_m: float = 20.0,
        seed: int = 0,
        objective_names: Sequence[str] | None = None,
    ) -> None:
        """Initialise the optimiser."""
        if n_objectives < 1:
            raise ValueError("n_objectives must be >= 1")
        if population_size < 4:
            raise ValueError("population_size must be >= 4")
        if n_generations < 0:
            raise ValueError("n_generations must be >= 0")
        self.objective_fn = objective_fn
        self.n_objectives = int(n_objectives)
        self.population_size = int(population_size + (population_size % 2))
        self.n_generations = int(n_generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = mutation_rate
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.seed = int(seed)
        self.objective_names = list(objective_names or [f"f{i}" for i in range(n_objectives)])
        self._rng: np.random.Generator = np.random.default_rng(self.seed)

    def _evaluate(self, params: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        values = np.asarray(self.objective_fn(params), dtype=np.float64).reshape(-1)
        if values.shape[0] != self.n_objectives:
            raise ValueError(
                f"objective_fn returned {values.shape[0]} values, expected {self.n_objectives}"
            )
        # A non-finite objective means the evaluation failed: the whole
        # point is invalid (every objective +inf), so it can never be
        # non-dominated through its remaining, finite objectives.
        if not np.all(np.isfinite(values)):
            return np.full(self.n_objectives, np.inf)
        return values

    def _tournament(self, ranks: np.ndarray[Any, Any], crowd: np.ndarray[Any, Any]) -> int:
        i, j = self._rng.integers(0, len(ranks), size=2)
        if ranks[i] != ranks[j]:
            return int(i if ranks[i] < ranks[j] else j)
        if crowd[i] != crowd[j]:
            return int(i if crowd[i] > crowd[j] else j)
        return int(i if self._rng.random() < 0.5 else j)

    def _sbx_spread(self, u: float, beta: float) -> float:
        """Bounded-SBX spread factor for one side of the parent interval."""
        alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
        if u <= 1.0 / alpha:
            return float((u * alpha) ** (1.0 / (self.eta_c + 1.0)))
        return float((1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1.0)))

    def _sbx(
        self,
        p1: np.ndarray[Any, Any],
        p2: np.ndarray[Any, Any],
        lo: np.ndarray[Any, Any],
        hi: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        c1, c2 = p1.copy(), p2.copy()
        if self._rng.random() > self.crossover_rate:
            return c1, c2
        for k in range(len(p1)):
            if self._rng.random() > 0.5 or abs(p1[k] - p2[k]) < 1e-14 or hi[k] <= lo[k]:
                continue
            y1, y2 = (p1[k], p2[k]) if p1[k] < p2[k] else (p2[k], p1[k])
            span = y2 - y1
            u = float(self._rng.random())
            betaq_lo = self._sbx_spread(u, 1.0 + 2.0 * (y1 - lo[k]) / span)
            betaq_hi = self._sbx_spread(u, 1.0 + 2.0 * (hi[k] - y2) / span)
            child_lo = float(np.clip(0.5 * ((y1 + y2) - betaq_lo * span), lo[k], hi[k]))
            child_hi = float(np.clip(0.5 * ((y1 + y2) + betaq_hi * span), lo[k], hi[k]))
            if self._rng.random() < 0.5:
                c1[k], c2[k] = child_lo, child_hi
            else:
                c1[k], c2[k] = child_hi, child_lo
        return c1, c2

    def _mutate(
        self, x: np.ndarray[Any, Any], lo: np.ndarray[Any, Any], hi: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        rate = self.mutation_rate if self.mutation_rate is not None else 1.0 / max(1, len(x))
        out = x.copy()
        for k in range(len(x)):
            span = hi[k] - lo[k]
            if span <= 0 or self._rng.random() >= rate:
                continue
            d1 = (out[k] - lo[k]) / span
            d2 = (hi[k] - out[k]) / span
            u = self._rng.random()
            power = 1.0 / (self.eta_m + 1.0)
            if u < 0.5:
                val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - d1) ** (self.eta_m + 1.0)
                dq = val**power - 1.0
            else:
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - d2) ** (self.eta_m + 1.0)
                dq = 1.0 - val**power
            out[k] = float(np.clip(out[k] + dq * span, lo[k], hi[k]))
        return out

    @staticmethod
    def _rank_and_crowd(
        objectives: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[list[int]]]:
        fronts = fast_non_dominated_sort(objectives)
        ranks = np.zeros(objectives.shape[0], dtype=int)
        crowd = np.zeros(objectives.shape[0])
        for r, front in enumerate(fronts):
            ranks[front] = r
            crowd[front] = crowding_distance(objectives[front])
        return ranks, crowd, fronts

    def optimize(self, bounds: Sequence[tuple[float, float]]) -> ParetoFront:
        """Run NSGA-II and return the final non-dominated front.

        Args:
            bounds: ``(low, high)`` per parameter.

        Returns:
            The first front of the final population (duplicates removed).
        """
        if not bounds:
            raise ValueError("bounds must contain at least one (low, high) pair")
        lo = np.array([float(b[0]) for b in bounds])
        hi = np.array([float(b[1]) for b in bounds])
        if np.any(hi < lo):
            raise ValueError("every bound must satisfy low <= high")
        self._rng = np.random.default_rng(self.seed)
        n = self.population_size

        params = lo + self._rng.random((n, len(lo))) * (hi - lo)
        objs = np.vstack([self._evaluate(p) for p in params])
        ranks, crowd, _ = self._rank_and_crowd(objs)

        for _ in range(self.n_generations):
            children: list[np.ndarray[Any, Any]] = []
            while len(children) < n:
                a = params[self._tournament(ranks, crowd)]
                b = params[self._tournament(ranks, crowd)]
                c1, c2 = self._sbx(a, b, lo, hi)
                children.append(self._mutate(c1, lo, hi))
                children.append(self._mutate(c2, lo, hi))
            child_params = np.vstack(children[:n])
            child_objs = np.vstack([self._evaluate(p) for p in child_params])

            pool_params = np.vstack([params, child_params])
            pool_objs = np.vstack([objs, child_objs])
            _, pool_crowd, fronts = self._rank_and_crowd(pool_objs)
            keep: list[int] = []
            for front in fronts:
                if len(keep) + len(front) <= n:
                    keep.extend(front)
                    continue
                order = sorted(front, key=lambda i: -pool_crowd[i])
                keep.extend(order[: n - len(keep)])
                break
            params, objs = pool_params[keep], pool_objs[keep]
            ranks, crowd, _ = self._rank_and_crowd(objs)

        first = np.flatnonzero(ranks == 0)
        valid = np.all(np.isfinite(objs), axis=1)
        if valid[first].any():
            first = first[valid[first]]
        seen: set[bytes] = set()
        solutions: list[ParetoSolution] = []
        for i in first:
            key = np.round(params[i], 12).tobytes() + np.round(objs[i], 12).tobytes()
            if key in seen:
                continue
            seen.add(key)
            solutions.append(
                ParetoSolution(
                    parameters=params[i].copy(),
                    objectives=objs[i].copy(),
                    rank=0,
                    crowding=float(crowd[i]),
                )
            )
        return ParetoFront(
            solutions=solutions,
            n_objectives=self.n_objectives,
            objective_names=list(self.objective_names),
        )
