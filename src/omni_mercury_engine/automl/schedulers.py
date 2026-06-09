# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Trial Schedulers for AutoML.

Implements Hyperband and ASHA for efficient hyperparameter optimization.

References:
- Li et al. (2018): Hyperband: A Novel Bandit-Based Approach
- Li et al. (2020): A System for Massively Parallel Hyperparameter Tuning
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SchedulerDecision(Enum):
    """Decision made by scheduler for a trial."""

    CONTINUE = auto()
    PAUSE = auto()
    STOP = auto()


@dataclass
class TrialInfo:
    """Information about a trial."""

    trial_id: str
    config: dict[str, Any]
    budget: float
    metric: float | None = None
    iteration: int = 0
    status: str = "running"


class TrialScheduler(ABC):
    """Base class for trial schedulers."""

    @abstractmethod
    def on_trial_result(
        self,
        trial_id: str,
        result: dict[str, Any],
    ) -> SchedulerDecision:
        """Called when a trial reports results."""
        pass

    @abstractmethod
    def on_trial_complete(self, trial_id: str) -> None:
        """Called when a trial completes."""
        pass

    @abstractmethod
    def get_next_budget(self, trial_id: str) -> float:
        """Get the next budget allocation for a trial."""
        pass


class MedianStoppingScheduler(TrialScheduler):
    """Median stopping rule scheduler.

    Stops trials performing below the median of completed trials.
    """

    def __init__(
        self,
        grace_period: int = 5,
        min_trials: int = 3,
    ) -> None:
        """Initialize median stopping scheduler."""
        self._grace_period = grace_period
        self._min_trials = min_trials
        self._trial_history: dict[str, list[float]] = {}
        self._completed_curves: list[list[float]] = []

    def on_trial_result(
        self,
        trial_id: str,
        result: dict[str, Any],
    ) -> SchedulerDecision:
        """Check if trial should continue."""
        metric = result.get("metric", result.get("loss", 0.0))
        iteration = result.get("iteration", 0)

        if trial_id not in self._trial_history:
            self._trial_history[trial_id] = []

        self._trial_history[trial_id].append(metric)

        if iteration < self._grace_period:
            return SchedulerDecision.CONTINUE

        if len(self._completed_curves) < self._min_trials:
            return SchedulerDecision.CONTINUE

        medians = []
        for curve in self._completed_curves:
            if len(curve) > iteration:
                medians.append(curve[iteration])

        if not medians:
            return SchedulerDecision.CONTINUE

        median_value = np.median(medians)

        if metric > median_value:
            return SchedulerDecision.STOP

        return SchedulerDecision.CONTINUE

    def on_trial_complete(self, trial_id: str) -> None:
        """Record completed trial."""
        if trial_id in self._trial_history:
            self._completed_curves.append(self._trial_history[trial_id])

    def get_next_budget(self, trial_id: str) -> float:
        """Get next budget (always 1 for this scheduler)."""
        return 1.0


class HyperbandScheduler(TrialScheduler):
    """Hyperband scheduler for efficient hyperparameter optimization.

    Implements successive halving with different budget allocations.
    """

    def __init__(
        self,
        max_budget: float = 81,
        reduction_factor: float = 3,
        min_budget: float = 1,
    ) -> None:
        """Initialize Hyperband scheduler.

        Args:
            max_budget: Maximum budget (e.g., epochs, iterations)
            reduction_factor: Reduction factor (eta in paper)
            min_budget: Minimum budget
        """
        self._max_budget = max_budget
        self._eta = reduction_factor
        self._min_budget = min_budget

        self._s_max = int(np.floor(np.log(max_budget / min_budget) / np.log(reduction_factor)))
        self._B = (self._s_max + 1) * max_budget

        self._brackets: list[HyperbandBracket] = []
        self._current_bracket = 0
        self._trial_to_bracket: dict[str, int] = {}
        self._trial_budgets: dict[str, float] = {}

        self._initialize_brackets()

    def _initialize_brackets(self) -> None:
        """Initialize all brackets."""
        for s in range(self._s_max, -1, -1):
            n = int(np.ceil((self._B / self._max_budget) * (self._eta**s) / (s + 1)))
            r = self._max_budget * (self._eta ** (-s))

            bracket = HyperbandBracket(
                s=s,
                n_configs=n,
                budget=r,
                eta=self._eta,
                max_budget=self._max_budget,
            )
            self._brackets.append(bracket)

    def get_initial_configs(self, n_configs: int) -> list[tuple[int, float]]:
        """Get initial configurations with budgets.

        Returns list of (bracket_idx, budget) tuples.
        """
        configs = []

        for bracket_idx, bracket in enumerate(self._brackets):
            for _ in range(bracket.n_configs):
                configs.append((bracket_idx, bracket.budget))

        return configs[:n_configs]

    def register_trial(self, trial_id: str, bracket_idx: int, budget: float) -> None:
        """Register a trial with its bracket."""
        self._trial_to_bracket[trial_id] = bracket_idx
        self._trial_budgets[trial_id] = budget

    def on_trial_result(
        self,
        trial_id: str,
        result: dict[str, Any],
    ) -> SchedulerDecision:
        """Process trial result."""
        bracket_idx = self._trial_to_bracket.get(trial_id)
        if bracket_idx is None:
            return SchedulerDecision.CONTINUE

        bracket = self._brackets[bracket_idx]
        metric = result.get("metric", result.get("loss", float("inf")))
        current_budget = result.get("budget", self._trial_budgets.get(trial_id, 0))

        bracket.add_result(trial_id, metric, current_budget)

        if bracket.should_promote(trial_id):
            next_budget = bracket.get_next_budget(trial_id)
            if next_budget is not None:
                self._trial_budgets[trial_id] = next_budget
                return SchedulerDecision.CONTINUE
            else:
                return SchedulerDecision.STOP
        elif bracket.should_stop(trial_id):
            return SchedulerDecision.STOP

        return SchedulerDecision.CONTINUE

    def on_trial_complete(self, trial_id: str) -> None:
        """Handle trial completion."""
        bracket_idx = self._trial_to_bracket.get(trial_id)
        if bracket_idx is not None:
            self._brackets[bracket_idx].mark_complete(trial_id)

    def get_next_budget(self, trial_id: str) -> float:
        """Get next budget for trial."""
        return self._trial_budgets.get(trial_id, self._min_budget)


@dataclass
class HyperbandBracket:
    """A single bracket in Hyperband."""

    s: int
    n_configs: int
    budget: float
    eta: float
    max_budget: float

    results: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    rung_results: dict[int, dict[str, float]] = field(default_factory=dict)
    current_rung: int = 0
    _completed: set[str] = field(default_factory=set)

    def add_result(self, trial_id: str, metric: float, budget: float) -> None:
        """Add a result for a trial."""
        if trial_id not in self.results:
            self.results[trial_id] = []
        self.results[trial_id].append((budget, metric))

        rung = self._budget_to_rung(budget)
        if rung not in self.rung_results:
            self.rung_results[rung] = {}
        self.rung_results[rung][trial_id] = metric

    def _budget_to_rung(self, budget: float) -> int:
        """Convert budget to rung number."""
        return int(np.log(budget / self.budget) / np.log(self.eta))

    def should_promote(self, trial_id: str) -> bool:
        """Check if trial should be promoted to next rung."""
        if trial_id not in self.results:
            return False

        latest_budget = self.results[trial_id][-1][0]
        rung = self._budget_to_rung(latest_budget)

        rung_results = self.rung_results.get(rung, {})
        if len(rung_results) < self.n_configs / (self.eta**rung):
            return False

        sorted_trials = sorted(rung_results.items(), key=lambda x: x[1])
        n_promote = max(1, int(len(sorted_trials) / self.eta))

        promoted_ids = [t[0] for t in sorted_trials[:n_promote]]
        return trial_id in promoted_ids and trial_id not in self._completed

    def should_stop(self, trial_id: str) -> bool:
        """Check if trial should be stopped."""
        if trial_id not in self.results:
            return False

        latest_budget = self.results[trial_id][-1][0]
        return latest_budget >= self.max_budget

    def get_next_budget(self, trial_id: str) -> float | None:
        """Get next budget for trial."""
        if trial_id not in self.results:
            return self.budget

        latest_budget = self.results[trial_id][-1][0]
        next_budget = latest_budget * self.eta

        if next_budget > self.max_budget:
            return None

        return next_budget

    def mark_complete(self, trial_id: str) -> None:
        """Mark trial as complete.

        Adds trial to completed set so it won't be re-scheduled.
        """
        self._completed.add(trial_id)


class ASHAScheduler(TrialScheduler):
    """Asynchronous Successive Halving Algorithm (ASHA).

    Provides asynchronous early stopping with successive halving.
    """

    def __init__(
        self,
        max_budget: float = 100,
        reduction_factor: float = 4,
        min_budget: float = 1,
        grace_period: int = 1,
    ) -> None:
        """Initialize ASHA scheduler.

        Args:
            max_budget: Maximum budget
            reduction_factor: Halving rate
            min_budget: Minimum budget
            grace_period: Minimum iterations before stopping
        """
        self._max_budget = max_budget
        self._eta = reduction_factor
        self._min_budget = min_budget
        self._grace_period = grace_period

        self._rungs: dict[int, list[tuple[str, float]]] = {}
        self._trial_budgets: dict[str, float] = {}
        self._trial_rungs: dict[str, int] = {}

        self._rung_budgets = self._compute_rung_budgets()

    def _compute_rung_budgets(self) -> list[float]:
        """Compute budget at each rung."""
        budgets = []
        budget = self._min_budget

        while budget <= self._max_budget:
            budgets.append(budget)
            budget *= self._eta

        return budgets

    def _get_rung(self, budget: float) -> int:
        """Get rung index for a budget."""
        for i, rung_budget in enumerate(self._rung_budgets):
            if budget <= rung_budget:
                return i
        return len(self._rung_budgets) - 1

    def on_trial_result(
        self,
        trial_id: str,
        result: dict[str, Any],
    ) -> SchedulerDecision:
        """Process trial result asynchronously."""
        metric = result.get("metric", result.get("loss", float("inf")))
        budget = result.get("budget", result.get("iteration", 1))

        rung = self._get_rung(budget)
        self._trial_rungs[trial_id] = rung
        self._trial_budgets[trial_id] = budget

        if rung not in self._rungs:
            self._rungs[rung] = []
        self._rungs[rung].append((trial_id, metric))

        if budget < self._grace_period:
            return SchedulerDecision.CONTINUE

        if self._should_promote(trial_id, rung, metric):
            next_budget = self._get_next_budget(rung)
            if next_budget is not None and next_budget <= self._max_budget:
                self._trial_budgets[trial_id] = next_budget
                return SchedulerDecision.CONTINUE
            else:
                return SchedulerDecision.STOP

        return SchedulerDecision.STOP

    def _should_promote(
        self,
        trial_id: str,
        rung: int,
        metric: float,
    ) -> bool:
        """Check if trial should be promoted."""
        rung_results = self._rungs.get(rung, [])

        if len(rung_results) < 2:
            return True

        sorted_results = sorted(rung_results, key=lambda x: x[1])
        n_promote = max(1, int(len(sorted_results) / self._eta))

        promoted = [r[0] for r in sorted_results[:n_promote]]
        return trial_id in promoted

    def _get_next_budget(self, current_rung: int) -> float | None:
        """Get budget for next rung."""
        if current_rung + 1 < len(self._rung_budgets):
            return self._rung_budgets[current_rung + 1]
        return None

    def on_trial_complete(self, trial_id: str) -> None:
        """Handle trial completion.

        Cleans up per-trial tracking state.
        """
        self._trial_budgets.pop(trial_id, None)

    def get_next_budget(self, trial_id: str) -> float:
        """Get current budget for trial."""
        return self._trial_budgets.get(trial_id, self._min_budget)

    def get_initial_budget(self) -> float:
        """Get initial budget for new trials."""
        return self._min_budget
