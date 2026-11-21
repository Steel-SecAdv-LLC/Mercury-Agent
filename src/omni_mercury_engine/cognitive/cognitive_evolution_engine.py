# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.measurement import MeasurementEvidence

logger = logging.getLogger(__name__)

#: A feature whose reference standard deviation is at or below this carries no
#: scale to standardize by and is left out of the distance.
MIN_REFERENCE_STD = 1e-12


#: Reference distances a rank verdict needs: with fewer than 20 the smallest
#: attainable p-value, 1/(n+1), is above 0.05 and no observation could be
#: called novel at that level.
MIN_CALIBRATION = 20


def rank_p_value(distance: float, calibration: Any) -> float:
    """The share of reference distances at least as large as ``distance`` (rank p-value).

    ``(1 + #{c >= distance}) / (n + 1)``: distribution-free, and at most
    ``alpha`` for no more than a fraction ``alpha`` of observations that are
    exchangeable with the reference.

    Args:
        distance: The observation's standardized distance.
        calibration: The reference distances.

    Returns:
        The p-value in ``(0, 1]``; ``1.0`` when there is no reference.
    """
    scores = np.asarray(calibration, dtype=float).reshape(-1)
    return float((1 + int(np.count_nonzero(scores >= distance))) / (scores.size + 1))


@dataclass
class ExplorationResult:
    """Outcome of scoring one observation for novelty.

    Attributes:
        exploration_id: Monotonic id for this exploration.
        target: Free-text label for what was explored (e.g. ``"anomaly:cyber"``).
        novelty_score: Novelty in ``[0, 1]``. A monotone function of the measured
            standardized distance from the observed distribution; ``0.5`` during
            warm-up (fewer than two prior observations, so no variance estimate
            exists yet -- a transparently *undetermined* score, not a measured one).
        is_novel: ``measured and p_value <= alpha``: the observation's distance
            ranks in the top ``alpha`` of the distances the reference
            observations themselves had.  An unmeasured score is never novel.
        standardized_distance: The raw RMS standardized deviation the score is
            derived from (``0.0`` during warm-up). A diagnostic, unbounded value.
        n_observations: How many observations have informed the running estimate
            (including this one).
        evidence: What proves the verdict was measured: how many reference
            distances it was ranked among, the digest of the reference mean,
            scale and distances, the p-value and the level it was compared
            with.  ``None`` during warm-up (fewer than
            :data:`MIN_CALIBRATION` reference distances), for an all-constant
            reference, or an unusable observation.  Required, with no
            default: ``measured`` is derived from it and cannot be declared
            (it used to be a boolean that defaulted to ``True``).
        p_value: The rank p-value of the distance among the reference
            distances; ``None`` when unmeasured.
        alpha: The level the p-value was compared with.
        observation: The scored vector, so the score can be recomputed
            without this engine. ``None`` when nothing was scored.
        reference_mean: The reference mean the vector was scored against
            (before it was folded in). ``None`` when unmeasured.
        reference_std: The reference standard deviation, likewise.
        reference_n: Observations behind that reference.
        calibration: The reference distances the p-value is a rank among.
        timestamp: Wall-clock creation time.
    """

    exploration_id: str
    target: str
    novelty_score: float
    is_novel: bool
    standardized_distance: float
    n_observations: int
    evidence: MeasurementEvidence | None
    p_value: float | None = None
    alpha: float | None = None
    observation: tuple[float, ...] | None = None
    reference_mean: tuple[float, ...] | None = None
    reference_std: tuple[float, ...] | None = None
    reference_n: int = 0
    calibration: tuple[float, ...] | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def measured(self) -> bool:
        """Whether the score came from a real variance estimate: it carries the evidence."""
        return self.evidence is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view for attaching to a detection result."""
        return {
            "exploration_id": self.exploration_id,
            "target": self.target,
            "novelty_score": self.novelty_score,
            "is_novel": self.is_novel,
            "standardized_distance": self.standardized_distance,
            "n_observations": self.n_observations,
            "measured": self.measured,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "evidence": self.evidence.to_dict() if self.evidence is not None else None,
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

    The VERDICT is a rank test, not a cut on that score. Each absorbed
    observation's distance against the reference as it then stood is kept
    (the last ``calibration_window`` of them), and a new observation is novel
    when its distance ranks in the top ``alpha`` of those: a p-value of
    ``(1 + #{reference distances >= distance}) / (n + 1)`` at or below
    ``alpha``. The rule is distribution-free, so the share of ordinary
    observations called novel is at most ``alpha`` by construction. The cut it
    replaces (score >= 0.7) had no such property: measured on the recorded
    benchmark tables it called 14.5% of ordinary held-out rows novel; this one
    calls 3.9% (``artifacts/novelty_discovery.json``).

    Warm-up: variance is undefined until two observations exist (the neutral
    score ``0.5``), and the verdict is undetermined until
    :data:`MIN_CALIBRATION` reference distances exist. An undetermined verdict
    is never novel and carries no evidence, so it cannot be read as measured.

    The running estimate resets if the observation dimensionality changes, since
    distances across different feature spaces are not comparable.
    """

    _WARMUP_NOVELTY = 0.5

    def __init__(
        self,
        novelty_threshold: float = 0.7,
        *,
        alpha: float = 0.05,
        calibration_window: int = 256,
    ) -> None:
        """Initialize.

        Args:
            novelty_threshold: A cut on the score, kept for callers that pass
                it and reported in the statistics.  It no longer decides
                ``is_novel``: measured on the recorded benchmark tables, a
                score at or above 0.7 called 14.5% of ordinary held-out rows
                novel (``artifacts/novelty_discovery.json``).
            alpha: The level of the verdict: an observation is novel when its
                distance ranks in the top ``alpha`` of the reference
                distances, so at most that fraction of ordinary observations
                is called novel.
            calibration_window: How many of the most recent reference
                distances the rank is taken among.

        Raises:
            ValueError: If ``alpha`` is not in ``(0, 1)`` or the window is
                smaller than :data:`MIN_CALIBRATION`.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if calibration_window < MIN_CALIBRATION:
            raise ValueError(
                f"calibration_window must be at least {MIN_CALIBRATION}, got {calibration_window}"
            )
        self.novelty_threshold = novelty_threshold
        self.alpha = float(alpha)
        self._exploration_counter = 0
        self._count = 0
        self._mean: np.ndarray[Any, Any] | None = None
        self._m2: np.ndarray[Any, Any] | None = None  # running sum of squared deviations
        # The standardized distance each absorbed observation had against the
        # reference as it stood when it arrived: what a new distance is ranked among.
        self._calibration: deque[float] = deque(maxlen=calibration_window)
        logger.info("CuriosityEngine initialised (alpha=%s)", alpha)

    @property
    def observations_seen(self) -> int:
        """Number of observations folded into the running estimate."""
        return self._count

    @staticmethod
    def _to_vector(data: Any) -> np.ndarray[Any, Any] | None:
        """Coerce supported inputs to a 1-D float vector, or ``None``.

        Accepts a mapping of scalar values (bools excluded), or any
        array-like. Returns ``None`` when there is no usable numeric content.

        Mapping keys are read in sorted order, not insertion order, so two
        equivalent dicts built with different key orderings yield the same
        feature vector -- otherwise the Mahalanobis novelty score would depend
        on how the caller happened to construct the observation dict.

        Non-finite inputs (NaN/inf, e.g. a missing sensor value or the
        ``mean`` of an empty batch) are rejected and yield ``None``: they would
        otherwise emit a NaN novelty score into the result (invalid JSON) and
        permanently poison the running mean/variance estimate.
        """
        if data is None:
            return None
        if isinstance(data, dict):
            values = [
                float(data[k])
                for k in sorted(data)
                if isinstance(data[k], (int, float, np.number)) and not isinstance(data[k], bool)
            ]
            if not values:
                return None
            arr = np.asarray(values, dtype=float)
        else:
            try:
                arr = np.asarray(data, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return None
            if arr.size == 0:
                return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr

    def _variance(self) -> np.ndarray[Any, Any] | None:
        if self._m2 is None or self._count < 2:
            return None
        return self._m2 / (self._count - 1)

    def _score(self, x: np.ndarray[Any, Any]) -> tuple[float, float, bool]:
        """Return ``(novelty, standardized_distance, measured)`` for ``x`` pre-update.

        ``measured`` is ``False`` when no variance estimate exists yet (warm-up,
        or every dimension constant): the neutral score returned in that case is
        explicitly *undetermined*, and callers must not treat it as a measured
        novelty (in particular, it must never satisfy a novelty threshold).
        """
        variance = self._variance()
        if self._mean is None or variance is None:
            return self._WARMUP_NOVELTY, 0.0, False
        std = np.sqrt(variance)
        # Only dimensions with real variance can be standardized. A constant
        # feature has std ~ 0; dividing by it would blow the whole score to 1.0
        # on any deviation and let one degenerate dimension dominate. Such
        # dimensions are excluded from the distance instead. If every dimension
        # is constant, novelty is undetermined (neutral warm-up score).
        valid = std > MIN_REFERENCE_STD
        if not np.any(valid):
            return self._WARMUP_NOVELTY, 0.0, False
        z = (x[valid] - self._mean[valid]) / std[valid]
        distance = float(np.sqrt(np.mean(np.square(z))))
        novelty = float(1.0 - np.exp(-distance))
        return novelty, distance, True

    def _update(self, x: np.ndarray[Any, Any]) -> None:
        """Fold ``x`` into the online mean/variance estimate (Welford)."""
        self._count += 1
        if self._mean is None or self._m2 is None:
            self._mean = np.zeros_like(x)
            self._m2 = np.zeros_like(x)
        delta = x - self._mean
        self._mean = self._mean + delta / self._count
        self._m2 = self._m2 + delta * (x - self._mean)

    def _result(
        self,
        exploration_id: str,
        target: str,
        x: np.ndarray[Any, Any] | None,
        scored: tuple[float, float, bool] | None,
        n_observations: int,
    ) -> ExplorationResult:
        """One result, with the evidence and the numbers behind a measured score.

        Must be called BEFORE ``x`` is folded in: the reference it records is
        the one the score was computed against.
        """
        if x is None or scored is None:
            return ExplorationResult(
                exploration_id=exploration_id,
                target=target,
                novelty_score=0.0,
                is_novel=False,
                standardized_distance=0.0,
                n_observations=n_observations,
                evidence=None,
            )
        score, distance, has_distance = scored
        variance = self._variance()
        calibration = np.asarray(self._calibration, dtype=float)
        if (
            not has_distance
            or self._mean is None
            or variance is None
            or calibration.size < MIN_CALIBRATION
        ):
            # No distance, or too few reference distances to rank it among:
            # the verdict is undetermined, and says so by carrying no evidence.
            return ExplorationResult(
                exploration_id=exploration_id,
                target=target,
                novelty_score=score,
                is_novel=False,
                standardized_distance=distance,
                n_observations=n_observations,
                evidence=None,
                observation=tuple(float(v) for v in x),
            )
        std = np.sqrt(variance)
        p_value = rank_p_value(distance, calibration)
        return ExplorationResult(
            exploration_id=exploration_id,
            target=target,
            novelty_score=score,
            is_novel=p_value <= self.alpha,
            standardized_distance=distance,
            n_observations=n_observations,
            evidence=MeasurementEvidence.from_samples(
                "conformal_rank",
                self._mean,
                std,
                calibration,
                n=int(calibration.size),
                statistic=p_value,
                threshold=self.alpha,
            ),
            p_value=p_value,
            alpha=self.alpha,
            observation=tuple(float(v) for v in x),
            reference_mean=tuple(float(v) for v in self._mean),
            reference_std=tuple(float(v) for v in std),
            reference_n=self._count,
            calibration=tuple(float(v) for v in calibration),
        )

    def novelty(self, target: str, data: Any = None) -> ExplorationResult:
        """Score one observation for novelty WITHOUT folding it in (read-only).

        :meth:`explore` scores and then absorbs the observation, so a stream
        of scored rows drifts the estimate toward what it is scoring — right
        for exploration, wrong for measurement.  This is the measurement
        form: the same score, the estimate untouched.  Used by the novelty
        discovery harness (``research/cognitive/harness_novelty_discovery.py``)
        and by any caller that must score held-out rows against a fixed,
        fitted distribution.

        Args:
            target: Free-text label for the exploration.
            data: The observation — a mapping of scalars or an array-like.

        Returns:
            An :class:`ExplorationResult`; ``n_observations`` reports the
            unchanged count.
        """
        self._exploration_counter += 1
        exploration_id = f"novelty_{self._exploration_counter:06d}"
        x = self._to_vector(data)
        if x is None or (self._mean is not None and x.shape != self._mean.shape):
            return self._result(exploration_id, target, None, None, self._count)
        return self._result(exploration_id, target, x, self._score(x), self._count)

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
            return self._result(exploration_id, target, None, None, self._count)

        # Distances are only comparable within one feature space; reset if the
        # dimensionality changes.
        if self._mean is not None and x.shape != self._mean.shape:
            self._count = 0
            self._mean = None
            self._m2 = None
            self._calibration.clear()

        # Build the result against the reference as it stands, THEN fold the
        # observation in: the reference it records is the one it was scored on.
        scored = self._score(x)
        result = self._result(exploration_id, target, x, scored, self._count + 1)
        if scored[2]:
            # This observation's own distance joins the reference distances
            # only after it has been ranked among the earlier ones.
            self._calibration.append(scored[1])
        self._update(x)
        return result

    def get_statistics(self) -> dict[str, Any]:
        """Return live counters for orchestrator statistics roll-up."""
        return {
            "explorations_performed": self._exploration_counter,
            "observations_seen": self._count,
            "novelty_threshold": self.novelty_threshold,
            "alpha": self.alpha,
            "calibration_distances": len(self._calibration),
        }


__all__ = [
    "MIN_CALIBRATION",
    "MIN_REFERENCE_STD",
    "CuriosityEngine",
    "ExplorationResult",
    "rank_p_value",
]
