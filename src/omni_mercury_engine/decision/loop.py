"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

The closed decision/response loop: identify -> interpret -> decide -> deter -> verify.

:class:`DecisionResponseLoop` is the orchestration that turns calibrated
confidence into autonomy with a conscience. One :meth:`DecisionResponseLoop.step`
runs the full cycle:

1. **interpret** -- a :class:`ConfidenceSignal` (already normalised from a
   detector by :mod:`omni_mercury_engine.decision.confidence`),
2. **decide** -- the :class:`AbstentionPolicy` returns a typed decision, possibly
   an honest abstention,
3. **deter** -- the :class:`ResponsePlanner` selects a proportionate action and
   the :class:`ResponseActuator` applies it under the fail-closed safety
   contract,
4. **verify** -- the whole pass is appended to an :class:`AuditLedger` (an
   append-only, JSON-serialisable certificate) and offered to an optional
   feedback sink, the seam through which outcomes can flow back to calibration or
   learning (the omnidirectional, closed-loop property).

The loop is deliberately *detector-agnostic*: it consumes a
:class:`ConfidenceSignal`, so the entire policy + response + audit machinery is
exercisable without importing the heavy detection stack.
:meth:`DecisionResponseLoop.step_from_engine_result` is the thin adapter for a
real ``OmniMercuryEngine`` result dict.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from omni_mercury_engine.core.types import ThreatLevel
from omni_mercury_engine.decision.abstention import AbstentionPolicy
from omni_mercury_engine.decision.confidence import (
    ConfidenceSignal,
    confidence_from_engine_result,
)
from omni_mercury_engine.decision.response import (
    Authorization,
    EthicalGate,
    ResponseActuator,
    ResponsePlanner,
    threat_level_from_score,
)
from omni_mercury_engine.decision.types import Decision, LoopResult, Verdict

__all__ = [
    "AuditLedger",
    "DecisionResponseLoop",
    "FeedbackSink",
]

# A feedback sink receives every completed loop pass -- the seam to push outcomes
# back to calibration, RL, or a human queue. It must not raise; the loop's job is
# done once the ledger has the record.
FeedbackSink = Callable[[LoopResult], None]


class AuditLedger:
    """Append-only, JSON-serialisable record of every decision + response.

    The ledger is the "detail" pillar at decision granularity: a verifiable trail
    that says, for each event, what was decided, why, what response was taken (or
    deferred/blocked), and under which honesty state.
    """

    def __init__(self) -> None:
        self._entries: list[LoopResult] = []

    def record(self, result: LoopResult) -> None:
        """Append one loop pass to the ledger."""
        self._entries.append(result)

    @property
    def entries(self) -> tuple[LoopResult, ...]:
        """An immutable view of the recorded loop passes."""
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def to_list(self) -> list[dict[str, object]]:
        """Return the ledger as a list of JSON-friendly dicts."""
        return [entry.as_dict() for entry in self._entries]

    def summary(self) -> dict[str, object]:
        """Return verdict / response-status / honesty-state counts and rates."""
        by_verdict: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_state: dict[str, int] = {}
        for entry in self._entries:
            verdict = entry.decision.verdict.value
            status = entry.response.status.value
            state = entry.three_state.value
            by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1
            by_state[state] = by_state.get(state, 0) + 1
        total = len(self._entries)
        abstentions = by_verdict.get(Verdict.ABSTAIN.value, 0)
        return {
            "total": total,
            "by_verdict": by_verdict,
            "by_status": by_status,
            "by_state": by_state,
            "abstention_rate": (abstentions / total) if total else 0.0,
        }


class DecisionResponseLoop:
    """Orchestrate decide -> deter -> verify over calibrated confidence.

    Args:
        ethical_gate: The fail-closed veto consulted before any effectful
            response. Required and explicit -- there is no silent default. When
            no ``actuator`` is supplied, this gate is bound into a default
            :class:`ResponseActuator`.
        abstention_policy: The decision policy (defaults to
            :class:`AbstentionPolicy` with conservative bands).
        planner: The response planner (defaults to :class:`ResponsePlanner`).
        actuator: The response actuator (defaults to one wrapping
            ``ethical_gate``).
        ledger: The audit ledger (defaults to a fresh :class:`AuditLedger`).
        feedback: Optional sink invoked with each completed :class:`LoopResult`.
    """

    def __init__(
        self,
        *,
        ethical_gate: EthicalGate,
        abstention_policy: AbstentionPolicy | None = None,
        planner: ResponsePlanner | None = None,
        actuator: ResponseActuator | None = None,
        ledger: AuditLedger | None = None,
        feedback: FeedbackSink | None = None,
    ) -> None:
        self.abstention_policy = abstention_policy or AbstentionPolicy()
        self.planner = planner or ResponsePlanner()
        self.actuator = actuator or ResponseActuator(ethical_gate)
        # NB: an empty AuditLedger is falsy (``__len__ == 0``), so a passed-in
        # shared ledger must be checked with ``is None``, never ``or``.
        self.ledger = ledger if ledger is not None else AuditLedger()
        self.feedback = feedback

    def step(
        self,
        signal: ConfidenceSignal,
        *,
        domain: str | None = None,
        severity: ThreatLevel | None = None,
        authorization: Authorization | None = None,
        context: Mapping[str, object] | None = None,
    ) -> LoopResult:
        """Run one full decide -> deter -> verify pass over ``signal``.

        Args:
            signal: The normalised calibrated-confidence input.
            domain: Domain hint (context + provenance).
            severity: Proportionality anchor; when ``None`` it is derived from the
                decision's calibrated confidence.
            authorization: Optional human sign-off for escalatory / irreversible
                actions.
            context: Optional handler context.

        Returns:
            The recorded :class:`LoopResult`.
        """
        decision = self.abstention_policy.decide(signal)
        threat = severity if severity is not None else self._derive_severity(decision)
        action = self.planner.plan(decision, severity=threat, domain=domain)
        outcome = self.actuator.actuate(
            action,
            decision,
            domain=domain,
            authorization=authorization,
            context=context,
        )
        result = LoopResult(
            decision=decision,
            response=outcome,
            domain=domain,
            provenance={"severity": threat.name},
        )
        self.ledger.record(result)
        if self.feedback is not None:
            self.feedback(result)
        return result

    def step_from_engine_result(
        self,
        result: Mapping[str, Any],
        *,
        domain: str | None = None,
        authorization: Authorization | None = None,
        context: Mapping[str, object] | None = None,
    ) -> LoopResult:
        """Adapt an ``OmniMercuryEngine`` result dict and run one :meth:`step`.

        Reads the calibrated probability + optional conformal set via
        :func:`confidence_from_engine_result`, and the severity anchor from the
        result's ``severity`` field when present (else the calibrated point).
        """
        signal = confidence_from_engine_result(result)
        severity_score = result.get("severity")
        severity = threat_level_from_score(
            float(severity_score) if severity_score is not None else signal.anomaly_probability
        )
        return self.step(
            signal,
            domain=domain,
            severity=severity,
            authorization=authorization,
            context=context,
        )

    @staticmethod
    def _derive_severity(decision: Decision) -> ThreatLevel:
        """Severity anchor when none is supplied.

        A grounded-negative call is :attr:`ThreatLevel.NONE`; otherwise severity
        scales with the calibrated ``P(anomaly)`` that drove the decision.
        """
        if decision.verdict is Verdict.NEGATIVE:
            return ThreatLevel.NONE
        return threat_level_from_score(decision.confidence)
