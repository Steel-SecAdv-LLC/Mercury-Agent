# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Curiosity-driven novelty scoring.

This module previously bundled a "cognitive evolution engine" of
self-play simulation, rule mutation, chain-of-thought reasoning,
counterfactual simulation and theory-of-mind components whose internals
were decorative -- canned insight strings, a hypothesis hardcoded to a
fixed sentence, and a novelty score anchored to the constant ``0.7`` plus
length/size bonuses. None of it was wired into the runtime, and its
outputs carried no measured signal.

Those decorative components were removed. What remains is a single,
genuine :class:`CuriosityEngine` that scores novelty as a *measured*
statistical distance of an observation from the distribution of
observations it has already seen (an online diagonal-Mahalanobis
distance), so a "novelty score" reflects how unusual an input actually is
rather than a hand-tuned constant. It is wired into
:class:`~omni_mercury_engine.cognitive.orchestrator.CognitiveOrchestrator`
to score detected anomalies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """Outcome of scoring one observation for novelty.

    Attributes:
        exploration_id: Monotonic id for this exploration.
        target: Free-text label for what was explored (e.g. ``"anomaly:cyber"``).
        novelty_score: Novelty in ``[0, 1]``. A monotone function of the measured
            standardized distance from the observed distribution; ``0.5`` during
            warm-up (fewer than two prior observations, so no variance estimate
            exists yet -- an honestly *undetermined* score, not a measured one).
        is_novel: ``novelty_score >= novelty_threshold``.
        standardized_distance: The raw RMS standardized deviation the score is
            derived from (``0.0`` during warm-up). A diagnostic, unbounded value.
        n_observations: How many observations have informed the running estimate
            (including this one).
        timestamp: Wall-clock creation time.
    """

    exploration_id: str
    target: str
    novelty_score: float
    is_novel: bool
    standardized_distance: float
    n_observations: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for attaching to a detection result."""
        return {
            "exploration_id": self.exploration_id,
            "target": self.target,
            "novelty_score": self.novelty_score,
            "is_novel": self.is_novel,
            "standardized_distance": self.standardized_distance,
            "n_observations": self.n_observations,
        }


class CuriosityEngine:
    """Score observations by how far they are from the seen distribution.

    Maintains an online (Welford) estimate of the per-feature mean and variance
    of every numeric observation vector it has been given, and scores a new
    observation by its RMS standardized deviation from that running
    distribution -- a diagonal Mahalanobis distance. High deviation means the
    observation is unlike what has been seen, i.e. novel. The score has no
    hand-tuned anchor: it is ``1 - exp(-distance)``, a monotone map of a measured
    quantity into ``[0, 1)``.

    Warm-up: variance is undefined until two observations exist, so the first
    one or two explorations return the neutral score ``0.5`` (``is_novel`` False)
    and disclose the small ``n_observations`` rather than fabricating novelty.

    The running estimate resets if the observation dimensionality changes, since
    distances across different feature spaces are not comparable.
    """

    _WARMUP_NOVELTY = 0.5

    def __init__(self, novelty_threshold: float = 0.7) -> None:
        """Initialize.

        Args:
            novelty_threshold: Score at or above which an observation is flagged
                ``is_novel``.
        """
        self.novelty_threshold = novelty_threshold
        self._exploration_counter = 0
        self._count = 0
        self._mean: np.ndarray[Any, Any] | None = None
        self._m2: np.ndarray[Any, Any] | None = None  # running sum of squared deviations
        logger.info("CuriosityEngine initialised (novelty_threshold=%s)", novelty_threshold)

    @property
    def observations_seen(self) -> int:
        """Number of observations folded into the running estimate."""
        return self._count

    @staticmethod
    def _to_vector(data: Any) -> np.ndarray[Any, Any] | None:
        """Coerce supported inputs to a 1-D float vector, or ``None``.

        Accepts a mapping of scalar values (bools excluded), or any
        array-like. Returns ``None`` when no numeric content is present.

        Mapping keys are read in sorted order, not insertion order, so two
        equivalent dicts built with different key orderings yield the same
        feature vector -- otherwise the Mahalanobis novelty score would depend
        on how the caller happened to construct the observation dict.
        """
        if data is None:
            return None
        if isinstance(data, dict):
            values = [
                float(data[k])
                for k in sorted(data)
                if isinstance(data[k], (int, float, np.number)) and not isinstance(data[k], bool)
            ]
            return np.asarray(values, dtype=float) if values else None
        try:
            arr = np.asarray(data, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        return arr if arr.size else None

    def _variance(self) -> np.ndarray[Any, Any] | None:
        if self._m2 is None or self._count < 2:
            return None
        return self._m2 / (self._count - 1)

    def _score(self, x: np.ndarray[Any, Any]) -> tuple[float, float]:
        """Return ``(novelty, standardized_distance)`` for ``x`` pre-update."""
        variance = self._variance()
        if self._mean is None or variance is None:
            return self._WARMUP_NOVELTY, 0.0
        std = np.sqrt(variance)
        # Only dimensions with real variance can be standardized. A constant
        # feature has std ~ 0; dividing by it would blow the whole score to 1.0
        # on any deviation and let one degenerate dimension dominate. Such
        # dimensions are excluded from the distance instead. If every dimension
        # is constant, novelty is undetermined (neutral warm-up score).
        valid = std > 1e-12
        if not np.any(valid):
            return self._WARMUP_NOVELTY, 0.0
        z = (x[valid] - self._mean[valid]) / std[valid]
        distance = float(np.sqrt(np.mean(np.square(z))))
        novelty = float(1.0 - np.exp(-distance))
        return novelty, distance

    def _update(self, x: np.ndarray[Any, Any]) -> None:
        """Fold ``x`` into the online mean/variance estimate (Welford)."""
        self._count += 1
        if self._mean is None or self._m2 is None:
            self._mean = np.zeros_like(x)
            self._m2 = np.zeros_like(x)
        delta = x - self._mean
        self._mean = self._mean + delta / self._count
        self._m2 = self._m2 + delta * (x - self._mean)

    def explore(self, target: str, data: Any = None) -> ExplorationResult:
        """Score one observation for novelty and fold it into the estimate.

        Args:
            target: Free-text label for the exploration.
            data: The observation -- a mapping of scalars or an array-like.
                When it carries no numeric content, novelty is reported as
                ``0.0`` (nothing to measure) and the estimate is unchanged.

        Returns:
            An :class:`ExplorationResult`.
        """
        self._exploration_counter += 1
        exploration_id = f"explore_{self._exploration_counter:06d}"

        x = self._to_vector(data)
        if x is None:
            return ExplorationResult(
                exploration_id=exploration_id,
                target=target,
                novelty_score=0.0,
                is_novel=False,
                standardized_distance=0.0,
                n_observations=self._count,
            )

        # Distances are only comparable within one feature space; reset if the
        # dimensionality changes.
        if self._mean is not None and x.shape != self._mean.shape:
            self._count = 0
            self._mean = None
            self._m2 = None

        novelty, distance = self._score(x)
        self._update(x)

        return ExplorationResult(
            exploration_id=exploration_id,
            target=target,
            novelty_score=novelty,
            is_novel=novelty >= self.novelty_threshold,
            standardized_distance=distance,
            n_observations=self._count,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Return live counters for orchestrator statistics roll-up."""
        return {
            "explorations_performed": self._exploration_counter,
            "observations_seen": self._count,
            "novelty_threshold": self.novelty_threshold,
        }


__all__ = ["CuriosityEngine", "ExplorationResult"]
