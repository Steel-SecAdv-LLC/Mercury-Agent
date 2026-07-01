# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Orchestration-boundary aggregate harm gate (harm-policy spec §6).

A per-query gate is trivially evaded by *decomposition*: an operator (or
Mercury's own multi-hop reasoner / subagent fleet) can split one blocked task
into a sequence of individually-benign sub-queries, each of which passes the
per-call gate, then reassemble the answers. This module closes part of that gap
at the one boundary where open-web decomposition actually happens -- the
:class:`~omni_mercury_engine.agentic.capabilities.assistant.GeneralAssistant`
research/answer/author loop -- with two complementary, fail-closed controls:

1. **Realized-plan re-gate.** The *same* two-axis
   :func:`~omni_mercury_engine.cognitive.ethical_bounding.assess_weapons_uplift`
   gate is re-run over the concatenation of the session's recent queries. When
   an offensive phrasing was split across sub-queries so that no single one
   tripped an offensive pattern, the concatenation can still reconstruct it.
   This is not a new heuristic axis -- it is the existing gate applied to the
   aggregate, exactly as the spec asks ("evaluate the realized plan ... not
   only per-leaf queries").

2. **Undifferentiated-mechanism accretion.** Repeatedly probing the *pure
   mechanism* ("how does it work") of a high-severity weapons hazard domain,
   with no defensive framing (detection / treatment / response / policy /
   licensed practice), is the accretion signature of assembling a knowledge
   base. Legitimate defensive professionals almost always frame their work
   defensively, so this counter does not trip on them. When it crosses a
   ceiling it **escalates** (human-in-the-loop / audit) rather than denies --
   a real engineer is slowed and logged, never blocked outright.

Residual risk is real and stated plainly in ``docs/HARM_POLICY.md`` §"Residual
risk": a semantically perfect decomposition that never co-locates offensive
phrasing and never over-probes a single domain can still evade a lexical
aggregate. Logging, provenance, and the bounded-autonomy ceiling carry that
residual; this module reduces the easy decomposition attacks, not all of them.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from omni_mercury_engine.cognitive.ethical_bounding import (
    HazardDomain,
    OperationalIntent,
    WeaponsDisposition,
    WeaponsRiskAssessment,
    assess_weapons_uplift,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Hazard domains whose undifferentiated-mechanism probing is worth accreting.
# The mass-casualty CBRN + high-yield-explosive set; lower-severity domains
# (conventional weapon, cyber) are excluded to keep the accretion counter
# precise (it is a volume heuristic, so it must not be noisy).
_HIGH_SEVERITY_DOMAINS = frozenset(
    {
        HazardDomain.CHEMICAL,
        HazardDomain.BIOLOGICAL,
        HazardDomain.RADIOLOGICAL,
        HazardDomain.NUCLEAR,
        HazardDomain.HIGH_YIELD_EXPLOSIVE,
        HazardDomain.OTHER_MASS_HARM,
    }
)


class SessionActionabilityTracker:
    """Session-scoped accretion tracker for the general-capability loop.

    One instance per :class:`GeneralAssistant` (i.e. per session/agent). Not
    thread-safe: a single conversational session drives it serially, matching
    how Mercury's assistant is used.

    Args:
        window: How many recent queries to retain for the realized-plan
            re-gate and the accretion window.
        mechanism_ceiling: Number of undifferentiated high-severity-domain
            mechanism probes within the window that trips escalation.
        gate: Injectable ``Callable[[str], WeaponsRiskAssessment]`` (defaults
            to :func:`assess_weapons_uplift`), for deterministic testing.
    """

    def __init__(
        self,
        *,
        window: int = 12,
        mechanism_ceiling: int = 5,
        gate: Callable[[str], WeaponsRiskAssessment] | None = None,
    ) -> None:
        """Initialize an empty session tracker."""
        self._queries: deque[str] = deque(maxlen=max(2, window))
        self._mechanism_ceiling = max(2, mechanism_ceiling)
        self._gate = gate or assess_weapons_uplift
        # Rolling record of (domain, is_undifferentiated_mechanism) for the
        # accretion counter, kept in lockstep with ``_queries``.
        self._mech_flags: deque[bool] = deque(maxlen=max(2, window))
        self._mech_domains: deque[HazardDomain] = deque(maxlen=max(2, window))

    def reset(self) -> None:
        """Clear all session state (e.g. at the start of a new task)."""
        self._queries.clear()
        self._mech_flags.clear()
        self._mech_domains.clear()

    @staticmethod
    def _is_undifferentiated_mechanism(assessment: WeaponsRiskAssessment) -> bool:
        """True when a query is bare mechanism probing of a high-severity domain.

        "Undifferentiated" = the query landed in a mass-casualty hazard domain
        but carried NO defensive/response/policy/licensed framing (its intent
        resolved to bare MECHANISM). That is the accretion signature; a
        defensively-framed professional query (detection/treatment/response/...)
        never counts.
        """
        return (
            assessment.hazard_domain in _HIGH_SEVERITY_DOMAINS
            and assessment.intent_tier is OperationalIntent.MECHANISM
            and not assessment.blocks
        )

    def record_and_assess(self, query: str) -> WeaponsRiskAssessment:
        """Record ``query`` and return the AGGREGATE weapons verdict for the session.

        The returned assessment is the more severe of (a) the realized-plan
        re-gate over the concatenated recent queries and (b) an escalation
        synthesized when undifferentiated-mechanism accretion crosses the
        ceiling. A non-blocking result means the aggregate view is clean; the
        caller still applies its own per-query verdict independently.
        """
        per_query = self._gate(query)
        self._queries.append(query)
        self._mech_flags.append(self._is_undifferentiated_mechanism(per_query))
        self._mech_domains.append(per_query.hazard_domain)

        # (1) Realized-plan re-gate over the concatenated recent queries.
        joined = "  ".join(self._queries)
        plan_verdict = self._gate(joined)

        # (2) Undifferentiated-mechanism accretion within a single domain.
        accretion_verdict: WeaponsRiskAssessment | None = None
        if sum(self._mech_flags) >= self._mechanism_ceiling:
            # Only escalate if the accretion concentrates in ONE domain -- a
            # scattershot of unrelated domains is not a procedure being
            # assembled. Count the most common flagged domain.
            flagged_domains = [
                d for d, f in zip(self._mech_domains, self._mech_flags, strict=True) if f
            ]
            if flagged_domains:
                top_domain = max(set(flagged_domains), key=flagged_domains.count)
                if flagged_domains.count(top_domain) >= self._mechanism_ceiling:
                    logger.info(
                        "aggregate gate: undifferentiated-mechanism accretion in %s "
                        "(%d probes) -> ESCALATE",
                        top_domain.value,
                        flagged_domains.count(top_domain),
                    )
                    accretion_verdict = WeaponsRiskAssessment(
                        hazard_domain=top_domain,
                        hazard_weight=1.0,
                        intent_tier=OperationalIntent.PRODUCTION,
                        confidence=0.0,
                        disposition=WeaponsDisposition.ESCALATE,
                        signals=("aggregate_mechanism_accretion",),
                    )

        # Return the most severe aggregate signal. Both are fail-closed; a
        # blocking plan re-gate (which can be HARD_REFUSE) outranks the
        # accretion ESCALATE.
        candidates = [v for v in (plan_verdict, accretion_verdict) if v is not None and v.blocks]
        if not candidates:
            return plan_verdict  # non-blocking; clean aggregate view
        return max(candidates, key=lambda v: _DISPOSITION_SEVERITY[v.disposition])


_DISPOSITION_SEVERITY: dict[WeaponsDisposition, int] = {
    WeaponsDisposition.ALLOW: 0,
    WeaponsDisposition.ALLOW_LOG: 1,
    WeaponsDisposition.ESCALATE: 2,
    WeaponsDisposition.REFUSE_REDACT: 3,
    WeaponsDisposition.HARD_REFUSE: 4,
}


__all__ = ["SessionActionabilityTracker"]
