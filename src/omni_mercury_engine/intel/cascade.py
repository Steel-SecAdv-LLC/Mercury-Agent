# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Confidence cascade: calibrated-uncertainty routing between a cheap and heavy path.

Most items are easy. Paying for a heavy reasoning model on all of them wastes
compute and latency; refusing to ever call it drops accuracy on the hard tail.
The cascade spends the heavy path only where it pays: it always runs the **cheap
template path** first, measures how *uncertain* that answer is on a **calibrated**
scale, and escalates to the **heavy model path** only when the uncertainty (or
the self-consistency disagreement) crosses a threshold.

The router is deterministic and fully instrumented:

* :class:`ConfidenceCascadeRouter` routes each item, applying an optional
  :class:`~omni_mercury_engine.core.calibration`-style calibrator to the cheap
  path's score so the routing threshold is on a *calibrated* probability, and
  folding in the self-consistency disagreement (:mod:`.self_consistency`) so a
  split cheap-path vote escalates even when its point probability looks
  confident.
* :class:`CascadeInstrumentation` accumulates per-path counts, compute cost, and
  latency, and :meth:`CascadeInstrumentation.report` renders the cost-vs-savings
  summary (the ``ci/confidence-cascade`` cost report).

Cost and latency are measured through injected ``clock`` / cost hooks (defaulting
to :func:`time.perf_counter`) so tests are deterministic and a caller can plug in
real token-cost accounting.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RoutePath(Enum):
    """Which path served an item."""

    CHEAP = "cheap"
    HEAVY = "heavy"


@dataclass(frozen=True)
class PathResult:
    """A path's prediction for one item.

    Attributes:
        answer: The path's answer (label/text/any).
        prob: The path's confidence for the positive class in ``[0, 1]``. For the
            cheap path this is calibrated by the router before routing when a
            calibrator is supplied.
        disagreement: Optional self-consistency disagreement in ``[0, 1]`` (0 when
            the path draws a single deterministic answer).
    """

    answer: Any
    prob: float
    disagreement: float = 0.0


#: A path maps an item to a :class:`PathResult`.
Path = Callable[[Any], PathResult]


@dataclass(frozen=True)
class CascadeConfig:
    """Routing thresholds and per-path cost weights.

    Attributes:
        low_uncertainty: Route to the cheap path only when the combined
            calibrated uncertainty is at or below this (the confident-cheap band).
        high_uncertainty: At or above this the item is unambiguously escalated;
            the band in between is also escalated (heavy) but tagged ``borderline``
            so the report can show the medium tier distinctly.
        cheap_cost: Compute-cost units charged for a cheap-path call.
        heavy_cost: Compute-cost units charged for a heavy-path call.
        disagreement_weight: How much the cheap path's disagreement contributes to
            the combined uncertainty (``0`` ignores it, ``1`` lets a fully-split
            vote escalate on its own). Clamped to ``[0, 1]``.
    """

    low_uncertainty: float = 0.30
    high_uncertainty: float = 0.60
    cheap_cost: float = 1.0
    heavy_cost: float = 20.0
    disagreement_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the threshold ordering and cost sanity (fail loud at build)."""
        if not 0.0 <= self.low_uncertainty <= self.high_uncertainty <= 1.0:
            raise ValueError(
                "require 0 <= low_uncertainty <= high_uncertainty <= 1; got "
                f"low={self.low_uncertainty}, high={self.high_uncertainty}"
            )
        if self.cheap_cost < 0 or self.heavy_cost < 0:
            raise ValueError("costs must be non-negative")


def point_uncertainty(prob: float) -> float:
    """Distance-to-boundary uncertainty of a probability: ``1 - 2*|p - 0.5|``.

    ``0`` at a fully-confident ``p in {0, 1}``, ``1`` at the maximally-uncertain
    ``p = 0.5``. This is the calibrated-probability half of the routing signal.
    """
    p = float(np.clip(prob, 0.0, 1.0))
    return 1.0 - 2.0 * abs(p - 0.5)


@dataclass
class CascadeInstrumentation:
    """Accumulates per-path counts, compute cost, and latency for a run."""

    n_cheap: int = 0
    n_heavy: int = 0
    cheap_cost: float = 0.0
    heavy_cost: float = 0.0
    cheap_latency: float = 0.0
    heavy_latency: float = 0.0
    #: Cost the all-heavy baseline (heavy on every item) would have charged.
    baseline_heavy_cost: float = 0.0
    _heavy_unit_cost: float = field(default=0.0, repr=False)

    @property
    def total(self) -> int:
        """Total items routed."""
        return self.n_cheap + self.n_heavy

    @property
    def total_cost(self) -> float:
        """Total compute cost actually spent by the cascade."""
        return self.cheap_cost + self.heavy_cost

    @property
    def total_latency(self) -> float:
        """Total measured latency across all routed items."""
        return self.cheap_latency + self.heavy_latency

    def report(self) -> dict[str, float]:
        """Render the cost/latency/savings summary.

        ``compute_saved_fraction`` is the fraction of the all-heavy baseline cost
        the cascade avoided -- the stream's declared value metric. It is ``0``
        when nothing was routed.
        """
        total = self.total or 1
        baseline = self.baseline_heavy_cost or (self._heavy_unit_cost * self.total)
        saved = (baseline - self.total_cost) / baseline if baseline > 0 else 0.0
        return {
            "n_items": self.total,
            "n_cheap": self.n_cheap,
            "n_heavy": self.n_heavy,
            "cheap_fraction": self.n_cheap / total,
            "heavy_fraction": self.n_heavy / total,
            "total_cost": self.total_cost,
            "baseline_heavy_cost": baseline,
            "compute_saved_fraction": max(0.0, saved),
            "total_latency": self.total_latency,
            "mean_latency": self.total_latency / total,
        }


@dataclass(frozen=True)
class RoutedOutcome:
    """The record of routing one item."""

    path: RoutePath
    result: PathResult
    uncertainty: float
    cost: float
    latency: float
    tier: str  # "confident_cheap" | "borderline" | "high_uncertainty"
    escalated_by: str | None  # None | "uncertainty" | "disagreement"


class ConfidenceCascadeRouter:
    """Routes items cheap-first, escalating to the heavy path on calibrated uncertainty."""

    def __init__(
        self,
        cheap: Path,
        heavy: Path,
        config: CascadeConfig | None = None,
        *,
        calibrator: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Initialize the router.

        Args:
            cheap: The cheap template path (always run first).
            heavy: The heavy model path (run only on escalation).
            config: Thresholds + cost weights (defaults to :class:`CascadeConfig`).
            calibrator: Optional object exposing ``calibrate(np.ndarray) ->
                np.ndarray``; when given, the cheap path's ``prob`` is calibrated
                before the routing threshold is applied, so routing is on a
                *calibrated* probability.
            clock: Monotonic clock for latency (injected for deterministic tests).
        """
        self.cheap = cheap
        self.heavy = heavy
        self.config = config or CascadeConfig()
        self.calibrator = calibrator
        self.clock = clock
        self.instrumentation = CascadeInstrumentation(_heavy_unit_cost=self.config.heavy_cost)

    def _calibrate(self, prob: float) -> float:
        """Apply the calibrator (if any) to a single probability."""
        if self.calibrator is None:
            return float(np.clip(prob, 0.0, 1.0))
        out = self.calibrator.calibrate(np.asarray([prob], dtype=float))
        return float(np.clip(np.asarray(out).reshape(-1)[0], 0.0, 1.0))

    def combined_uncertainty(self, result: PathResult) -> float:
        """Fuse calibrated point-uncertainty with the disagreement signal.

        Takes the max of the two (fail-safe: any strong uncertainty signal
        escalates), with disagreement scaled by
        :attr:`CascadeConfig.disagreement_weight`.
        """
        cal_prob = self._calibrate(result.prob)
        pu = point_uncertainty(cal_prob)
        w = float(np.clip(self.config.disagreement_weight, 0.0, 1.0))
        du = w * float(np.clip(result.disagreement, 0.0, 1.0))
        return max(pu, du)

    def _tier_and_reason(self, uncertainty: float, result: PathResult) -> tuple[str, str | None]:
        """Classify the uncertainty band and what (if anything) escalated."""
        cfg = self.config
        if uncertainty <= cfg.low_uncertainty:
            return "confident_cheap", None
        w = float(np.clip(cfg.disagreement_weight, 0.0, 1.0))
        by = (
            "disagreement"
            if w * float(np.clip(result.disagreement, 0.0, 1.0))
            > point_uncertainty(self._calibrate(result.prob))
            else "uncertainty"
        )
        tier = "high_uncertainty" if uncertainty >= cfg.high_uncertainty else "borderline"
        return tier, by

    def route_one(self, item: Any) -> RoutedOutcome:
        """Route a single item, updating instrumentation."""
        cfg = self.config
        inst = self.instrumentation

        t0 = self.clock()
        cheap_result = self.cheap(item)
        cheap_latency = self.clock() - t0

        uncertainty = self.combined_uncertainty(cheap_result)
        tier, escalated_by = self._tier_and_reason(uncertainty, cheap_result)

        # The all-heavy baseline would have charged one heavy call for this item.
        inst.baseline_heavy_cost += cfg.heavy_cost

        if tier == "confident_cheap":
            inst.n_cheap += 1
            inst.cheap_cost += cfg.cheap_cost
            inst.cheap_latency += cheap_latency
            return RoutedOutcome(
                path=RoutePath.CHEAP,
                result=cheap_result,
                uncertainty=uncertainty,
                cost=cfg.cheap_cost,
                latency=cheap_latency,
                tier=tier,
                escalated_by=None,
            )

        # Escalate: the cheap probe already ran (its cost/latency stand), then
        # the heavy path runs on top.
        t1 = self.clock()
        heavy_result = self.heavy(item)
        heavy_latency = self.clock() - t1

        inst.n_heavy += 1
        inst.cheap_cost += cfg.cheap_cost  # the probe we always pay
        inst.heavy_cost += cfg.heavy_cost
        inst.cheap_latency += cheap_latency
        inst.heavy_latency += heavy_latency
        return RoutedOutcome(
            path=RoutePath.HEAVY,
            result=heavy_result,
            uncertainty=uncertainty,
            cost=cfg.cheap_cost + cfg.heavy_cost,
            latency=cheap_latency + heavy_latency,
            tier=tier,
            escalated_by=escalated_by,
        )

    def route(self, items: Iterable[Any]) -> list[RoutedOutcome]:
        """Route every item, returning the per-item outcomes in order."""
        return [self.route_one(item) for item in items]


__all__ = [
    "CascadeConfig",
    "CascadeInstrumentation",
    "ConfidenceCascadeRouter",
    "Path",
    "PathResult",
    "RoutePath",
    "RoutedOutcome",
    "point_uncertainty",
]
