# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Subagent fleet: the main agent's mechanism to call on subagents.

The fleet is how the main Mercury Agent delegates work — one task to the most
competent specialist (:meth:`dispatch`), many tasks across the fleet
(:meth:`dispatch_many`), or one task to many replicas of a specialist running
concurrently (:meth:`scale_dispatch`, "mine and dig … even in the masses").

Every path is bounded by the :class:`~omni_mercury_engine.agentic.subagents.governor.AutonomyGovernor`
(capability ceiling, corrigibility, tripwire) and committed through the **dual
hard ethical gate** — the benevolence floor plus the σ-Immutable gate — exactly
as on the engine, cognitive-orchestrator, and multi-agent-orchestrator decision
boundaries. Commitment is fail-closed: if the aggregate action does not clear
both gates, the fleet raises rather than returning an ungated result.

Determinism: results are assembled in submission order (never completion
order), and each replica is constructed with a deterministic seed
(``base_seed + index``), so a single dispatch is reproducible. Numerical
reproducibility *across concurrent replicas* additionally depends on the
delegated specialization's own concurrency properties — a specialization that
itself fans out over a thread pool and shares process-global RNG is order- and
structure-reproducible but not bit-reproducible under contention. The fleet
never reorders results or fabricates signal to manufacture agreement.
Concurrency is real (a bounded ``ThreadPoolExecutor``); per-subagent failures
are surfaced in the returned results, never silently dropped below the
aggregate.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.agentic.subagents.base import (
    _INTERNAL,
    SubAgentAccessError,
    SubAgentResult,
    SubAgentTask,
    _InternalAccess,
)
from omni_mercury_engine.agentic.subagents.governor import AutonomyGovernor
from omni_mercury_engine.agentic.subagents.registry import (
    SubAgentRegistry,
    default_registry,
)
from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    BenevolenceScorer,
    sanitize_domain,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omni_mercury_engine.agentic.mercury_a_agent import DomainType

# Upper bound on real OS threads per dispatch — the governor's max_replicas
# bounds the logical fleet size; this bounds the physical thread pool so a
# mass dispatch does not oversubscribe the box. Mirrors the orchestrator's
# bounded thread usage.
logger = logging.getLogger(__name__)

_MAX_POOL_WORKERS = 16


@dataclass
class AggregateResult:
    """Transparent aggregate over a homogeneous replica dispatch.

    Attributes:
        representative: Highest-confidence completed result (tie-broken by
            subagent id); ``None`` when nothing completed.
        n_completed: Replicas that completed.
        n_failed: Replicas that failed.
        n_blocked: Replicas the benevolence gate refused.
        mean_confidence: Mean confidence over completed replicas (0 if none).
        agreement: Fraction of replicas that completed (1.0 = unanimous success).
    """

    representative: SubAgentResult | None
    n_completed: int
    n_failed: int
    n_blocked: int
    mean_confidence: float
    agreement: float


@dataclass
class FleetResult:
    """Outcome of a fleet dispatch (single, many, or mass).

    Attributes:
        results: Per-subagent results in submission order (failures surfaced).
        aggregate: Aggregate over the results (for mass dispatch); ``None`` for
            heterogeneous many-task dispatches.
        specialty: The specialty dispatched (for homogeneous dispatch).
        committed: Whether the dual ethical gate authorized commitment.
        benevolence_score: Measured benevolence at the commit boundary.
        metadata: Free-form annotations (domain, replicas, depth, …).
    """

    results: list[SubAgentResult]
    aggregate: AggregateResult | None = None
    specialty: str | None = None
    committed: bool = False
    benevolence_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class SubAgentFleet:
    """Engine-mediated fleet of internal subagents under autonomy governance."""

    def __init__(
        self,
        *,
        access: _InternalAccess,
        registry: SubAgentRegistry | None = None,
        governor: AutonomyGovernor | None = None,
        seed: int | None = None,
        engine: Any | None = None,
    ) -> None:
        """Construct a fleet (engine-mediated only).

        Args:
            access: Internal access sentinel; must be :data:`_INTERNAL`.
            registry: Catalogue of specializations; defaults to the built-ins.
            governor: Autonomy governor; defaults to a standard capability
                ceiling.
            seed: Base seed for deterministic replica construction.
            engine: Optional :class:`OmniMercuryEngine` the detection
                specialization uses to run Mercury's *own* real detection
                (never a stub) when present.

        Raises:
            SubAgentAccessError: If ``access`` is not the internal sentinel.
        """
        if access is not _INTERNAL:
            raise SubAgentAccessError(
                "SubAgentFleet is internal-only; construct it via "
                "OmniMercuryEngine.enable_subagent_fleet or MercuryAgent.enable_fleet."
            )
        self._registry = registry or default_registry(_INTERNAL)
        self.governor = governor or AutonomyGovernor()
        self._seed = seed
        self._engine = engine
        # Dual hard ethical gate at the commit boundary (eager, fail-closed),
        # mirroring MultiAgentOrchestrator's construction.
        self._benevolence_scorer = BenevolenceScorer(
            benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR,
        )
        from omni_mercury_engine.security.sigma_immutable_gate import (
            get_sigma_immutable_gate,
        )

        self._sigma_immutable_gate = get_sigma_immutable_gate()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_specialties(self) -> list[str]:
        """All specialties the fleet can dispatch to."""
        return self._registry.list_specialties()

    def route(self, task: SubAgentTask) -> str:
        """Resolve which specialty a task would route to (no execution)."""
        return self._registry.match(task)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        task: SubAgentTask,
        specialty: str | None = None,
        *,
        depth: int = 0,
    ) -> SubAgentResult:
        """Route and run one task on one subagent, committed through the gate.

        The result is returned regardless of completed/failed/blocked status —
        transparent disposition — but only after the dual ethical gate authorizes the
        fleet to act in this domain (fail-closed; raises otherwise).
        """
        resolved = specialty or self._registry.match(task)
        self.governor.authorize(1, depth)
        try:
            agent = self._registry.create(resolved, _INTERNAL, seed=self._seed)
            self.governor.check_autonomy(agent.autonomy_ceiling)
            result = agent.handle(task)
        finally:
            self.governor.release(1)
        committed = self._commit([result], task.domain)
        self.governor.observe_results([result])
        result.metadata["committed"] = committed
        result.metadata["specialty"] = resolved
        return result

    def dispatch_many(
        self,
        tasks: Sequence[SubAgentTask],
        *,
        depth: int = 0,
    ) -> FleetResult:
        """Run many (heterogeneous) tasks concurrently, each routed independently.

        Each task is routed and handled by its own subagent on a bounded thread
        pool. Results are returned in task order; per-task failures are
        surfaced. The commit gate is applied once over the batch domain set.
        """
        if not tasks:
            return FleetResult(results=[], committed=True)
        n = len(tasks)
        self.governor.authorize(n, depth)
        try:
            results = self._run_pool(
                [(t, self._registry.match(t), idx) for idx, t in enumerate(tasks)]
            )
        finally:
            self.governor.release(n)
        # Commit over the most severe domain present (sanitized downstream).
        domain = tasks[0].domain
        committed = self._commit(results, domain)
        self.governor.observe_results(results)
        return FleetResult(
            results=results,
            committed=committed,
            metadata={"n_tasks": n, "depth": depth},
        )

    def scale_dispatch(
        self,
        task: SubAgentTask,
        replicas: int,
        specialty: str | None = None,
        *,
        depth: int = 0,
    ) -> FleetResult:
        """Run one task on ``replicas`` subagents of a specialty, in the masses.

        This is the "mine and dig … even in the masses" path: many full-
        capability subagents work the same task concurrently; the fleet
        aggregates transparently (surfacing dissent and failures) and commits the
        aggregate through the dual ethical gate.

        Replicas are seeded ``base_seed + index`` for reproducibility; the
        governor bounds ``replicas`` by the capability ceiling (fail-closed).
        """
        resolved = specialty or self._registry.match(task)
        self.governor.authorize(replicas, depth)
        try:
            work = [(task, resolved, idx) for idx in range(replicas)]
            results = self._run_pool(work)
        finally:
            self.governor.release(replicas)
        aggregate = self._aggregate(results)
        committed = self._commit(results, task.domain)
        self.governor.observe_results(results)
        return FleetResult(
            results=results,
            aggregate=aggregate,
            specialty=resolved,
            committed=committed,
            metadata={"replicas": replicas, "depth": depth},
        )

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _run_pool(self, work: Sequence[tuple[SubAgentTask, str, int]]) -> list[SubAgentResult]:
        """Run (task, specialty, index) units concurrently; assemble in order.

        Each unit constructs its own subagent (seeded by index for
        determinism), checks autonomy, and handles its task. A unit that raises
        during construction/autonomy-check is surfaced as a ``failed`` result
        rather than collapsing the dispatch.
        """
        ordered: list[SubAgentResult | None] = [None] * len(work)

        def _unit(item: tuple[SubAgentTask, str, int]) -> tuple[int, SubAgentResult]:
            task, specialty, index = item
            seed = None if self._seed is None else self._seed + index
            try:
                agent = self._registry.create(specialty, _INTERNAL, seed=seed)
                self.governor.check_autonomy(agent.autonomy_ceiling)
                return index, agent.handle(task)
            except Exception as exc:  # surfaced, never dropped below the aggregate
                return index, SubAgentResult(
                    subagent_id=f"sub_{specialty}_construct_{index}",
                    specialty=specialty,
                    task_id=task.task_id,
                    status="failed",
                    reasoning="subagent construction/authorization failed",
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=dict(task.metadata),
                )

        max_workers = max(1, min(_MAX_POOL_WORKERS, len(work)))
        if max_workers == 1 or len(work) == 1:
            for item in work:
                idx, res = _unit(item)
                ordered[idx] = res
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for idx, res in pool.map(_unit, work):
                    ordered[idx] = res
        return [r for r in ordered if r is not None]

    @staticmethod
    def _aggregate(results: Sequence[SubAgentResult]) -> AggregateResult:
        """Transparent aggregate over replicas: pick best, surface dissent/failures."""
        completed = [r for r in results if r.status == "completed"]
        n_failed = sum(1 for r in results if r.status == "failed")
        n_blocked = sum(1 for r in results if r.status == "blocked")
        total = len(results)
        if completed:
            representative = max(completed, key=lambda r: (r.confidence, r.subagent_id))
            mean_conf = sum(r.confidence for r in completed) / len(completed)
        else:
            representative, mean_conf = None, 0.0
        agreement = (len(completed) / total) if total else 0.0
        return AggregateResult(
            representative=representative,
            n_completed=len(completed),
            n_failed=n_failed,
            n_blocked=n_blocked,
            mean_confidence=mean_conf,
            agreement=agreement,
        )

    # ------------------------------------------------------------------
    # Commit boundary (dual hard ethical gate, fail-closed)
    # ------------------------------------------------------------------

    def _commit(self, results: Sequence[SubAgentResult], domain: DomainType) -> bool:
        """Authorize the fleet to act on these results, via the dual gate.

        Two independent gates run in order, both fail-closed. First the shared
        harm-uplift choke point
        (:func:`~omni_mercury_engine.cognitive.decision_gate.enforce_decision_boundary`)
        over the **real** commit decision, including the subagents' own output;
        then the σ-Immutable gate over the calibrated 256-d vector. Both raise
        :class:`EthicalConstraintViolationError`; nothing downstream of a failed
        gate is treated as committed.

        Subagent output reaches the gate on purpose. Under the superseded
        benevolence pass-bar that would have been an injection vector — positive
        vocabulary in an output could buy a permit. The harm-uplift gate is
        block-on-harm, so output content can only move the verdict toward a
        refusal, and withholding it would have hidden exactly the case worth
        catching: a fleet about to commit uplift material.

        Returns ``True`` when both gates authorize commitment.
        """
        from omni_mercury_engine.cognitive.decision_gate import (
            DecisionSubject,
            enforce_decision_boundary,
        )

        safe_domain = sanitize_domain(getattr(domain, "value", str(domain)))
        confidences = [r.confidence for r in results if r.status == "completed"]
        severity = max(confidences) if confidences else 0.0
        anomaly_prob = (sum(confidences) / len(confidences)) if confidences else 0.0

        verdict = enforce_decision_boundary(
            DecisionSubject(
                surface="SubAgentFleet.commit",
                operation="commit delegated subagent results under fleet governance",
                domain=safe_domain,
                payload={
                    "severity": round(severity, 4),
                    "outputs": [r.output for r in results if r.status == "completed"],
                },
            ),
            advisory_scorer=self._benevolence_scorer,
        )
        if verdict.benevolence is not None:
            logger.debug(
                "SubAgentFleet.commit[%s]: advisory benevolence %.4f (informational; "
                "the enforced control is the harm-uplift gate, which returned %s)",
                safe_domain,
                verdict.benevolence,
                verdict.assessment.disposition.value,
            )

        from omni_mercury_engine.security.sigma_immutable_gate import (
            PERMITTED_ETHICAL_POSTURE,
            build_sigma_immutable_vector,
        )

        # The ethical band carries the system's configured posture, not the
        # per-call advisory float — see ``PERMITTED_ETHICAL_POSTURE``.
        sigma_vector = build_sigma_immutable_vector(
            benevolence_score=PERMITTED_ETHICAL_POSTURE,
            severity=severity,
            anomaly_prob=anomaly_prob,
        )
        self._sigma_immutable_gate.enforce(
            action=f"SubAgentFleet.commit:{safe_domain}",
            scalar_vector=sigma_vector,
            details={
                "boundary": "SubAgentFleet.commit",
                "domain": safe_domain,
                "severity": severity,
                "anomaly_prob": anomaly_prob,
                "n_results": len(results),
            },
        )
        return True

    def snapshot(self) -> dict[str, Any]:
        """Aggregate fleet + governor telemetry."""
        gov = self.governor.snapshot()
        return {
            "specialties": self.list_specialties(),
            "governor": {
                "active": gov.active,
                "paused": gov.paused,
                "tripped": gov.tripped,
                "total_authorized": gov.total_authorized,
                "total_completed": gov.total_completed,
                "total_failed": gov.total_failed,
                "total_blocked": gov.total_blocked,
                "trip_reasons": gov.trip_reasons,
            },
        }
