# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Subagent base: a full-capability, Omni-Code-anchored, access-guarded worker.

A :class:`SubAgent` *is a* :class:`~omni_mercury_engine.agentic.mercury_a_agent.MercuryAgent`
— it inherits the full planning / reasoning / four-tier memory / tool-execution
machinery of the main agent, so a subagent "mines and digs to the capability of
the main agent" rather than being a thin wrapper. Each subagent is a named member
of the Mercury pantheon (see
:mod:`~omni_mercury_engine.agentic.subagents.roster`), constructed from a
:class:`~omni_mercury_engine.agentic.subagents.roster.RosterEntry` that binds it
to:

* a **pantheon identity** (``id`` such as ``"Themis_I"``, plus an optional alias
  and a human-readable ``role``);
* one or more **real ``omni_mercury_engine`` subsystems** it coordinates;
* exactly one **Omni-Code anchor** (one of the Seven Omni-Codes in
  :mod:`~omni_mercury_engine.utils.constants`) whose helical *stability* genuinely
  sets the subagent's autonomy ceiling via
  :func:`~omni_mercury_engine.utils.constants.compute_ethical_autonomy`. The same
  constellation aligns Mercury Agent with AMA Cryptography. A subagent can never
  exceed the hard 0.95 autonomy cap.

The fleet calls :meth:`SubAgent.handle`, which runs a fail-closed benevolence
gate that *authorizes the controlled defensive action* of running the subagent
(caller text is screened separately by the ``Ares_XIV`` guardrail member, not
conflated into the safety verdict), then delegates to :meth:`SubAgent._perform`
(deep specializations override it with real domain logic; the
:class:`~omni_mercury_engine.agentic.subagents.coordinator.CoordinatorSubAgent`
binds and exercises its real subsystem), and packages a transparent
:class:`SubAgentResult` (completed / failed / blocked) — never a fabricated
success.

Access boundary: construction requires the internal :data:`_INTERNAL` sentinel,
held only by the registry and fleet. Direct user instantiation raises
:class:`SubAgentAccessError`. This package is not exported from the public
``omni_mercury_engine`` surface.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.agentic.mercury_a_agent import (
    AgentMode,
    DomainType,
    MercuryAgent,
)
from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError,
    sanitize_domain,
)
from omni_mercury_engine.utils.constants import (
    OmniCode,
    OmniCodes,
    compute_ethical_autonomy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omni_mercury_engine.agentic.subagents.roster import RosterEntry


class _InternalAccess:
    """Opaque construction token for the engine-mediated subagent boundary."""

    __slots__ = ()


#: Module-private access sentinel. Held by the registry and fleet; required to
#: construct any :class:`SubAgent`. Realizes "users must not address subagents
#: directly" as an enforced precondition, not a convention.
_INTERNAL = _InternalAccess()

# Hard autonomy cap, mirroring compute_ethical_autonomy's ceiling.
_MAX_AUTONOMY = 0.95


class SubAgentAccessError(RuntimeError):
    """Raised when a subagent is constructed outside the engine-mediated path."""


class SubAgentExecutionError(RuntimeError):
    """Raised when a subagent cannot complete a task transparently (fail-closed).

    A specialization raises this from :meth:`SubAgent._perform` rather than
    returning a fabricated result; :meth:`SubAgent.handle` records it as a
    ``failed`` outcome with the error surfaced, never swallowed.
    """


def resolve_anchor(name: str) -> OmniCode:
    """Resolve an Omni-Code anchor name to its :class:`OmniCode`.

    Args:
        name: An Omni-Code attribute name, e.g. ``"OMNI_BENEVOLENT"``.

    Returns:
        The corresponding :class:`OmniCode` from
        :class:`~omni_mercury_engine.utils.constants.OmniCodes`.

    Raises:
        KeyError: If ``name`` is not one of the Seven Omni-Codes.
    """
    codes = OmniCodes.get_all()
    if name not in codes:
        raise KeyError(f"unknown Omni-Code anchor {name!r}; valid: {sorted(codes)}")
    return codes[name]


def anchor_autonomy(anchor: OmniCode) -> float:
    """Derive a subagent's autonomy ceiling from its Omni-Code anchor.

    The anchor's helical *stability* (``|r| * p``) is normalized against the most
    stable Code and mapped to a base autonomy in ``[0.55, 0.90]``, then passed
    through :func:`~omni_mercury_engine.utils.constants.compute_ethical_autonomy`
    (which applies the Omni-Code stability boost and the hard 0.95 cap). A more
    stable anchor grants a higher ceiling; none can exceed 0.95. This is a real,
    monotonic binding to the shared constellation — not a decorative tag.
    """
    stabilities = [c.stability for c in OmniCodes.get_all().values()]
    max_stability = max(stabilities) if stabilities else 1.0
    norm = anchor.stability / max_stability if max_stability > 0 else 0.0
    base = 0.55 + 0.35 * norm
    return compute_ethical_autonomy(base_autonomy=base, ethical_threshold=0.99, use_omni_codes=True)


@dataclass(frozen=True)
class SubAgentCapability:
    """Routing descriptor for a subagent.

    Attributes:
        specialty: The subagent's pantheon id (e.g. ``"Themis_I"``).
        domains: Domains this subagent is competent in (for registry matching).
        keywords: Lowercase task-description keywords that attract routing.
    """

    specialty: str
    domains: frozenset[DomainType]
    keywords: frozenset[str]


@dataclass
class SubAgentTask:
    """A unit of work the fleet hands to a subagent.

    Attributes:
        description: Human-readable task description (drives routing and the
            benevolence-gate framing).
        domain: Domain hint; sanitized before it reaches any ethical scorer.
        payload: Arbitrary task inputs (e.g. ``{"data": ndarray, ...}``).
        priority: Relative urgency (1 low … 5 emergency).
        task_id: Stable identifier (autogenerated when omitted).
        metadata: Free-form annotations carried through to the result.
    """

    description: str
    domain: DomainType = DomainType.GENERAL
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 2
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentResult:
    """The transparent outcome of one subagent handling one task.

    ``status`` is one of ``"completed"`` (work produced a result), ``"failed"``
    (the specialization raised; ``error`` carries why), or ``"blocked"`` (the
    fail-closed benevolence gate refused; no work ran).

    Attributes:
        subagent_id: Unique runtime id of the subagent instance.
        specialty: The subagent's pantheon id.
        task_id: The task that was handled.
        status: Transparent disposition (see above).
        output: The specialization's result payload (``None`` unless completed).
        confidence: The subagent's self-assessed confidence in ``output``.
        benevolence_score: Measured benevolence of the authorized action.
        autonomy_ceiling: Omni-Code-anchored autonomy the subagent ran under.
        anchor: The Omni-Code anchor name (e.g. ``"OMNI_BENEVOLENT"``).
        reasoning: Short human-readable trace of the disposition.
        error: Error text when ``status == "failed"``.
        duration_ms: Wall-clock time spent handling the task.
        metadata: Task metadata carried through (plus fleet annotations).
    """

    subagent_id: str
    specialty: str
    task_id: str
    status: str
    output: Any = None
    confidence: float = 0.0
    benevolence_score: float = 0.0
    autonomy_ceiling: float = 0.0
    anchor: str = ""
    reasoning: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the task completed (not failed or blocked)."""
        return self.status == "completed"


class SubAgent(MercuryAgent):
    """A named, Omni-Code-anchored subagent with the full main-agent toolkit.

    Constructed from a :class:`RosterEntry`. Deep specializations subclass this
    and override :meth:`_perform` with real domain logic; the generalist base
    runs the inherited :meth:`MercuryAgent.analyze` pipeline so even an
    unspecialized member is genuinely capable, never a stub.
    """

    def __init__(
        self,
        *,
        access: _InternalAccess,
        entry: RosterEntry,
        seed: int | None = None,
        enable_calibration: bool = True,
    ) -> None:
        """Construct a subagent from a roster entry (engine-mediated only).

        Args:
            access: Internal access sentinel; must be :data:`_INTERNAL`.
            entry: The roster entry defining this subagent's identity, real
                subsystem bindings, and Omni-Code anchor.
            seed: Determinism seed for the subagent's reasoning/calibration.
            enable_calibration: Enable Bayesian confidence calibration.

        Raises:
            SubAgentAccessError: If ``access`` is not the internal sentinel.
        """
        if access is not _INTERNAL:
            raise SubAgentAccessError(
                "SubAgent is internal-only; construct it through the engine-mediated "
                "fleet (OmniMercuryEngine.enable_subagent_fleet / "
                "MercuryAgent.enable_fleet), never directly."
            )
        anchor = resolve_anchor(entry.anchor)
        autonomy_ceiling = anchor_autonomy(anchor)
        super().__init__(
            name=f"Mercury::{entry.id}",
            autonomy_level=autonomy_ceiling,
            ethical_threshold=0.99,
            enable_calibration=enable_calibration,
        )
        self.id = entry.id
        # ``specialty`` == the pantheon id; kept as the routing/telemetry key.
        self.specialty = entry.id
        self.alias = entry.alias
        self.role = entry.role
        self.subsystems = tuple(entry.subsystems)
        self.anchor = anchor
        self.anchor_name = entry.anchor
        self.depth = entry.depth
        self.code_bearer = entry.code_bearer
        self.subagent_domain = entry.domain
        self.autonomy_ceiling = autonomy_ceiling
        self._keywords = frozenset(k.lower() for k in entry.keywords)
        self._seed = seed
        self.subagent_id = f"sub_{entry.id}_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def capability(self) -> SubAgentCapability:
        """This subagent's routing descriptor (id + domain + keywords)."""
        return SubAgentCapability(
            specialty=self.id,
            domains=frozenset({self.subagent_domain}),
            keywords=self._keywords,
        )

    # ------------------------------------------------------------------
    # Task handling (the fleet's single entry point)
    # ------------------------------------------------------------------

    def handle(self, task: SubAgentTask) -> SubAgentResult:
        """Handle one task: gate, perform, package — transparently.

        The fail-closed benevolence gate runs first; a refusal yields a
        ``blocked`` result with no work performed. A specialization that raises
        :class:`SubAgentExecutionError` (or any unexpected exception) yields a
        ``failed`` result with the error surfaced. Only genuine work yields
        ``completed``.
        """
        start = time.time()
        self.mode = AgentMode.EXECUTING
        try:
            benevolence = self._score_task_benevolence(task)
        except EthicalConstraintViolationError as exc:
            self.mode = AgentMode.DORMANT
            return SubAgentResult(
                subagent_id=self.subagent_id,
                specialty=self.specialty,
                task_id=task.task_id,
                status="blocked",
                benevolence_score=float(exc.score),
                autonomy_ceiling=self.autonomy_ceiling,
                anchor=self.anchor_name,
                reasoning=f"benevolence gate refused task: {exc}",
                duration_ms=(time.time() - start) * 1000.0,
                metadata=dict(task.metadata),
            )

        try:
            output, confidence, reasoning = self._perform(task)
            status, error = "completed", None
        except SubAgentExecutionError as exc:
            output, confidence, reasoning = None, 0.0, ""
            status, error = "failed", str(exc)
        except Exception as exc:  # genuine, unexpected failure — surfaced
            output, confidence, reasoning = None, 0.0, ""
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self.mode = AgentMode.DORMANT

        return SubAgentResult(
            subagent_id=self.subagent_id,
            specialty=self.specialty,
            task_id=task.task_id,
            status=status,
            output=output,
            confidence=float(confidence),
            benevolence_score=benevolence,
            autonomy_ceiling=self.autonomy_ceiling,
            anchor=self.anchor_name,
            reasoning=reasoning,
            error=error,
            duration_ms=(time.time() - start) * 1000.0,
            metadata=dict(task.metadata),
        )

    def _score_task_benevolence(self, task: SubAgentTask) -> float:
        """Fail-closed harm-uplift authorization for running this subagent.

        The task's **real** description and payload are what get gated.

        The previous implementation deliberately withheld caller text and
        scored a canned string
        (``"...audit monitor verify data research evidence fair oversight..."``)
        on the theory that adversarial phrasing could steer the gate. That
        reasoning was sound for a *benevolence* gate, whose polarity is
        pass-on-positive-vocabulary: injected positive words would have bought
        an attacker a permit. It does not hold for the harm-uplift gate, whose
        polarity is **block-on-harm**. Injected harm vocabulary can only push
        the disposition toward a refusal, and injected allow-signal can at best
        move a B6 gray-zone verdict from ``REFUSE_REDACT`` to ``ESCALATE`` —
        both of which block. So feeding the real task closes a genuine
        false-negative (a subagent dispatched to do uplift work) without
        opening any way in. The ``Ares_XIV`` guardrail member still screens
        task content independently.

        Returns:
            The advisory benevolence score for the authorized action.

        Raises:
            EthicalConstraintViolationError: with ``check="harm_uplift"`` when
                the shared fail-closed gate refuses the dispatch.
        """
        from omni_mercury_engine.cognitive.decision_gate import (
            DecisionSubject,
            enforce_decision_boundary,
        )

        safe_domain = sanitize_domain(getattr(task.domain, "value", str(task.domain)))
        verdict = enforce_decision_boundary(
            DecisionSubject(
                surface=f"SubAgent[{self.specialty}].handle",
                operation="execute a delegated subagent task under fleet governance",
                domain=safe_domain,
                request=task.description,
                payload=task.payload,
            ),
            advisory_scorer=self._benevolence_scorer,
        )
        # NaN, not 1.0, when the advisory scorer could not produce a number.
        # ``benevolence`` is None exactly when the scorer raised or none was
        # supplied; reporting a perfect 1.0 for that would invent the most
        # favourable possible value for a measurement that did not happen, and
        # a fleet-wide average would then quietly improve as scoring broke.
        # NaN propagates through aggregation instead of hiding in it. The
        # dispatch itself is unaffected — the enforced harm gate above already
        # ran, and benevolence decides nothing.
        return float(verdict.benevolence) if verdict.benevolence is not None else float("nan")

    def _perform(self, task: SubAgentTask) -> tuple[Any, float, str]:
        """Do the real work. Generalist base runs the full main-agent pipeline.

        Returns a ``(output, confidence, reasoning)`` triple. Specializations
        override this; the base implementation genuinely runs
        :meth:`MercuryAgent.analyze` so the generalist subagent is as capable as
        the main agent.

        Raises:
            SubAgentExecutionError: If the work cannot be completed transparently.
        """
        analysis = self.analyze(
            data=task.payload.get("data"),
            domain=task.domain,
            goal=task.description,
            context=dict(task.payload),
        )
        reasoning = str(analysis.get("reasoning", {}).get("conclusion", "analysis complete"))
        confidence = float(analysis.get("plan_confidence", 0.0))
        return analysis, confidence, reasoning


def make_capability(
    specialty: str,
    domains: Iterable[DomainType],
    keywords: Iterable[str],
) -> SubAgentCapability:
    """Build a :class:`SubAgentCapability` from loose iterables (test/helper use)."""
    return SubAgentCapability(
        specialty=specialty,
        domains=frozenset(domains),
        keywords=frozenset(k.lower() for k in keywords),
    )
