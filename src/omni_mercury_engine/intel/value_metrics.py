# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-stream *measured* value metrics for the Mercury intelligence layer.

A learning/geometry stream that cannot state -- and measure -- the value it
delivers is theater. This module makes each stream's value first-class: a
:class:`ValueMetric` declares *what* is measured, a **baseline** (the value
before the stream, or the no-weakening floor), and a **target** (the goal),
plus the direction that counts as improvement. The stream's own benchmark
produces the *measured* number; :func:`ValueMetric.meets_target` /
:func:`ValueMetric.improves_on_baseline` adjudicate it, and
``benchmarks/intel_value_metrics_report.py`` renders the whole board so a
reviewer sees, per stream, ``baseline -> measured (target)`` at a glance.

The registry :data:`VALUE_METRICS` is the single source of truth for those
baselines/targets; the ``ci/*`` intel lanes and unit tests import it rather than
hard-coding thresholds, so a target can never silently drift between the doc, the
gate, and the test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """Whether a larger or smaller measured value is the improvement."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class ValueMetric:
    """A stream's declared, measurable value with a baseline and a target.

    Attributes:
        stream: The intelligence-layer stream this metric belongs to (a key of
            :data:`VALUE_METRICS`).
        metric: Short name of the measured quantity.
        unit: Human-readable unit of the measured quantity.
        direction: Whether higher or lower is the improvement.
        baseline: The value *before* this stream, or the no-weakening floor the
            stream must never regress past.
        target: The goal value the stream aims to reach.
        description: One-line rationale of why this quantity is the value.
        aspirational: When ``True`` the *target* is a goal the stream reaches for
            but is not required to hit (the no-weakening *floor* is the real
            gate). When ``False`` (default) the target is a hard requirement the
            board's ``--check`` enforces via :meth:`meets_target` -- necessary
            because :meth:`improves_on_baseline` is vacuous for a
            ``HIGHER_IS_BETTER`` metric whose baseline is ``0`` (any non-negative
            measurement, including a total collapse to ``0``, trivially "improves"
            on it), so those streams need the target as their non-vacuous gate.
    """

    stream: str
    metric: str
    unit: str
    direction: Direction
    baseline: float
    target: float
    description: str
    aspirational: bool = False

    def meets_target(self, measured: float) -> bool:
        """True when ``measured`` reaches (or beats) :attr:`target`.

        A ``NaN`` measurement fails closed (a metric that could not be computed
        is never treated as meeting its target).
        """
        if math.isnan(measured):  # NaN
            return False
        if self.direction is Direction.HIGHER_IS_BETTER:
            return measured >= self.target
        return measured <= self.target

    def improves_on_baseline(self, measured: float) -> bool:
        """True when ``measured`` is at least as good as :attr:`baseline`.

        The no-weakening check: for a ``HIGHER_IS_BETTER`` metric the value may
        not fall below baseline; for ``LOWER_IS_BETTER`` it may not rise above
        it. ``NaN`` fails closed.
        """
        if math.isnan(measured):  # NaN
            return False
        if self.direction is Direction.HIGHER_IS_BETTER:
            return measured >= self.baseline
        return measured <= self.baseline

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping for this metric (no measured value)."""
        return {
            "stream": self.stream,
            "metric": self.metric,
            "unit": self.unit,
            "direction": self.direction.value,
            "baseline": self.baseline,
            "target": self.target,
            "aspirational": self.aspirational,
            "description": self.description,
        }

    def summarize(self, measured: float) -> dict[str, object]:
        """Return a JSON-friendly ``baseline -> measured (target)`` verdict row."""
        return {
            **self.as_dict(),
            "measured": measured,
            "meets_target": self.meets_target(measured),
            "improves_on_baseline": self.improves_on_baseline(measured),
        }


#: The value board. One entry per intelligence-layer stream. Baselines/targets
#: are the *only* place these thresholds are defined; gates and tests import
#: them. Where a baseline is an empirically-pinned floor (adversarial survival
#: rate) it is the deterministic first-run measurement, kept in lockstep with the
#: stream's benchmark so "no weakening" is enforceable.
VALUE_METRICS: dict[str, ValueMetric] = {
    "closed_feedback_loop": ValueMetric(
        stream="closed_feedback_loop",
        metric="poisoned_candidate_block_rate",
        unit="fraction of poisoned retrain candidates blocked",
        direction=Direction.HIGHER_IS_BETTER,
        baseline=0.0,
        target=1.0,
        description=(
            "Human-gating alone lets a data-poisoned candidate through; the "
            "OOF/adversarial regression gate must block every candidate that "
            "regresses calibration or adversarial recall."
        ),
    ),
    "confidence_cascade": ValueMetric(
        stream="confidence_cascade",
        metric="compute_saved_at_bounded_accuracy",
        unit="fraction of heavy-path calls avoided",
        direction=Direction.HIGHER_IS_BETTER,
        baseline=0.0,
        target=0.50,
        description=(
            "Routing low-uncertainty items to the cheap template path must avoid "
            ">=50% of heavy-model calls while keeping accuracy loss within the "
            "cascade's configured tolerance vs the all-heavy baseline."
        ),
    ),
    "self_consistency": ValueMetric(
        stream="self_consistency",
        metric="disagreement_error_auroc",
        unit="AUROC of disagreement predicting error",
        direction=Direction.HIGHER_IS_BETTER,
        baseline=0.50,
        target=0.70,
        description=(
            "The N-sample disagreement signal must rank errored predictions above "
            "correct ones better than chance (AUROC 0.5) -- target >=0.70 on the "
            "held-out set -- to be a usable uncertainty signal for the calibrator."
        ),
    ),
    "adversarial_co_training": ValueMetric(
        stream="adversarial_co_training",
        metric="gate_bypass_survival_rate",
        unit="fraction of adversarial mutations that bypass the gate",
        direction=Direction.LOWER_IS_BETTER,
        # Deterministic first-run survival rate (0.3333) of the shipped red-team
        # config against the current *lexical* gate surface, rounded up to a
        # pinned no-weakening ceiling. The dominant bypass class is character
        # obfuscation (spacing/punctuation) that defeats lexical matching; all
        # survivors are triaged to corpus/pending. The pinned floor lives in
        # benchmarks/red_team_baseline.json and is kept >= this value; the lane
        # fails if a gate change *raises* the bypass rate above the floor.
        baseline=0.34,
        target=0.0,
        aspirational=True,  # target 0.0 is the goal; the no-weakening floor is the gate
        description=(
            "The red-team harness's surviving-bypass rate against the current gate "
            "may never rise above the pinned floor (no weakening); triaged "
            "survivors feed the corpus to drive it toward zero."
        ),
    ),
    "verifier_in_loop": ValueMetric(
        stream="verifier_in_loop",
        metric="false_claim_block_rate",
        unit="fraction of oracle-refuted claims blocked",
        direction=Direction.HIGHER_IS_BETTER,
        baseline=0.0,
        target=1.0,
        description=(
            "Without the verifier, a fabricated symbolic claim (a wrong primality/"
            "Collatz/propositional/physics statement) emits unchallenged; in hard "
            "mode every oracle-refuted claim must block emission."
        ),
    ),
    "provenance": ValueMetric(
        stream="provenance",
        metric="boundary_provenance_enforcement_rate",
        unit="fraction of provenance-required emissions enforced",
        direction=Direction.HIGHER_IS_BETTER,
        baseline=0.0,
        target=1.0,
        description=(
            "The boundary-fallback must enforce provenance on every "
            "provenance-required emission (refuse/redact an unprovenanced one) -- "
            "~80% of the full type-system value, with the residual being "
            "compile-time unrepresentability (see the migration plan)."
        ),
    ),
}


def get_value_metric(stream: str) -> ValueMetric:
    """Return the :class:`ValueMetric` for ``stream``.

    Raises:
        KeyError: if ``stream`` is not a declared intelligence-layer stream --
            callers must register a value metric deliberately, never measure a
            stream that declared no value.
    """
    try:
        return VALUE_METRICS[stream]
    except KeyError:
        raise KeyError(
            f"no value metric declared for stream {stream!r}; "
            f"known streams: {sorted(VALUE_METRICS)}"
        ) from None


__all__ = [
    "VALUE_METRICS",
    "Direction",
    "ValueMetric",
    "get_value_metric",
]
