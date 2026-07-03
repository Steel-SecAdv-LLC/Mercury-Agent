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

2. **Semantic undifferentiated-mechanism accretion.** Repeatedly probing the
   *pure mechanism* ("how does it work") of a high-severity weapons hazard
   domain, with no defensive framing (detection / treatment / response / policy
   / licensed practice), is the accretion signature of assembling a knowledge
   base. Each undifferentiated probe is embedded (deterministic hashed word-TF
   vector) and the signal fires on the largest *semantically cohesive* cluster
   of such probes -- so an attacker re-phrasing across sub-queries, or drifting
   the exact HazardDomain wording, no longer splits the count the way exact
   domain-equality matching did. Legitimate defensive professionals frame their
   work defensively, so this cluster does not trip on them. When it crosses a
   ceiling it **escalates** (human-in-the-loop / audit) rather than denies -- a
   real engineer is slowed and logged, never blocked outright.

Residual risk is real and stated plainly in ``docs/HARM_POLICY.md`` §"Residual
risk": a semantically perfect decomposition that never co-locates offensive
phrasing and keeps each probe both defensively framed and dissimilar can still
evade a lexical/embedding aggregate. Durable audit logging, provenance, and the
bounded-autonomy ceiling carry that residual; this module reduces the easy and
the moderately-sophisticated decomposition attacks, not all of them.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from omni_mercury_engine.cognitive.ethical_bounding import (
    HazardDomain,
    OperationalIntent,
    WeaponsDisposition,
    WeaponsRiskAssessment,
    assess_weapons_uplift,
)
from omni_mercury_engine.cognitive.gate_audit import record_gate_decision
from omni_mercury_engine.cognitive.harm_normalization import base_normalize

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Hashed word-level embedding for semantic accretion. Deterministic (no
# PYTHONHASHSEED dependency) so the accretion signal is reproducible.
_EMBED_DIM = 256
_WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _det_word_hash(word: str) -> int:
    """Stable polynomial rolling hash (process-independent)."""
    h = 0
    for ch in word:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def _embed(text: str) -> np.ndarray:
    """Deterministic L2-normalized hashed term-frequency embedding of ``text``.

    numpy/stdlib only -- no model. Content words (>=3 chars) after obfuscation-
    resistant normalization are hashed into a fixed-width vector, so semantically
    similar queries (same hazard vocabulary, different phrasing) land close in
    cosine space. Used to detect *distributed* probing that the exact-domain
    counter misses.
    """
    v = np.zeros(_EMBED_DIM, dtype=np.float64)
    for word in _WORD_RE.findall(base_normalize(text)):
        v[_det_word_hash(word) % _EMBED_DIM] += 1.0
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


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
        similarity_threshold: float = 0.45,
        plan_window: int = 3,
        gate: Callable[[str], WeaponsRiskAssessment] | None = None,
    ) -> None:
        """Initialize an empty session tracker.

        Args:
            window: How many recent queries to retain.
            mechanism_ceiling: Cohesive-probe count within the window that trips
                escalation.
            similarity_threshold: Cosine above which two undifferentiated probes
                count as the *same* line of enquiry for the semantic-accretion
                cluster (so re-phrasing across sub-queries no longer splits the
                count the way exact hazard-domain matching did).
            plan_window: Max number of *adjacent* queries the realized-plan
                re-gate concatenates. Adjacent-only (not the whole window) so an
                offensive request split across consecutive sub-queries is caught
                without cross-contaminating unrelated queries scattered across
                the session (e.g. a benign "how to cook pasta" beside an
                unrelated hazard-domain query).
            gate: Injectable weapons gate (defaults to assess_weapons_uplift).
        """
        maxlen = max(2, window)
        self._queries: deque[str] = deque(maxlen=maxlen)
        self._mechanism_ceiling = max(2, mechanism_ceiling)
        self._similarity_threshold = float(similarity_threshold)
        self._plan_window = max(2, plan_window)
        self._gate = gate or assess_weapons_uplift
        # Rolling records kept in lockstep with ``_queries``: whether each query
        # was an undifferentiated high-severity probe, its hazard domain, and its
        # semantic embedding (for the cluster-based accretion signal).
        self._mech_flags: deque[bool] = deque(maxlen=maxlen)
        self._mech_domains: deque[HazardDomain] = deque(maxlen=maxlen)
        self._embeddings: deque[np.ndarray] = deque(maxlen=maxlen)

    def reset(self) -> None:
        """Clear all session state (e.g. at the start of a new task)."""
        self._queries.clear()
        self._mech_flags.clear()
        self._mech_domains.clear()
        self._embeddings.clear()

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

    def _largest_semantic_cluster(self, embeddings: list[np.ndarray]) -> list[int]:
        """Indices of the densest set of mutually similar (cosine>=thr) embeddings.

        Greedy single-linkage-by-seed: every embedding is unit-normalized, so the
        dot product is the cosine; for each seed we take the set of embeddings
        within ``similarity_threshold`` of it and return the largest such
        neighbourhood. This measures how concentrated the undifferentiated
        probing is as one line of enquiry, independent of exact phrasing.
        """
        n = len(embeddings)
        if n == 0:
            return []
        mat = np.vstack(embeddings)
        sims = mat @ mat.T
        neighbourhoods = [
            [j for j in range(n) if sims[i, j] >= self._similarity_threshold] for i in range(n)
        ]
        return max(neighbourhoods, key=len)

    def record_and_assess(self, query: str) -> WeaponsRiskAssessment:
        """Record ``query`` and return the AGGREGATE weapons verdict for the session.

        The returned assessment is the more severe of (a) the realized-plan
        re-gate over the concatenated recent queries and (b) an escalation
        synthesized when undifferentiated-mechanism accretion crosses the
        ceiling. A non-blocking result means the aggregate view is clean; the
        caller still applies its own per-query verdict independently.

        Fail-closed: any internal error (a numpy/clustering failure, a raising
        injected gate) yields ``HARD_REFUSE`` rather than propagating -- the
        aggregate boundary matches the per-query gate's fail-closed contract so a
        caller can never be crashed into an unguarded path.
        """
        try:
            return self._record_and_assess(query)
        except Exception:
            logger.exception("aggregate gate failed; failing closed to HARD_REFUSE")
            return WeaponsRiskAssessment(
                hazard_domain=HazardDomain.OTHER_MASS_HARM,
                hazard_weight=1.0,
                intent_tier=OperationalIntent.PRODUCTION,
                confidence=0.0,
                disposition=WeaponsDisposition.HARD_REFUSE,
                signals=("aggregate_error",),
            )

    def _record_and_assess(self, query: str) -> WeaponsRiskAssessment:
        """Aggregate assessment implementation (see :meth:`record_and_assess`)."""
        per_query = self._gate(query)
        self._queries.append(query)
        self._mech_flags.append(self._is_undifferentiated_mechanism(per_query))
        self._mech_domains.append(per_query.hazard_domain)
        self._embeddings.append(_embed(query))

        # (1) Realized-plan re-gate over ADJACENT recent queries. Catches an
        # offensive phrasing physically SPLIT across consecutive sub-queries so
        # that no single one tripped an offensive pattern but the concatenation
        # does -- while (unlike whole-window concatenation) NOT cross-
        # contaminating unrelated queries scattered across the session.
        #
        # The aggregate signal is inherently uncertain: a production VERB from
        # one benign query co-located with a hazard NOUN from an adjacent benign
        # query can trip the concatenation without any real intent to split a
        # request, and no lexical/embedding test cleanly separates the two. So a
        # blocking realized-plan verdict is CAPPED to ESCALATE (audit + human-in-
        # the-loop), never a hard denial -- a coincidental co-location slows and
        # logs a user, it never refuses them; a genuine split is caught for
        # review. The per-query gate still hard-refuses an unambiguous single
        # query independently.
        queries = list(self._queries)
        plan_verdict = per_query
        for size in range(2, min(self._plan_window, len(queries)) + 1):
            window_verdict = self._gate("  ".join(queries[-size:]))
            if (
                window_verdict.blocks
                and _DISPOSITION_SEVERITY[WeaponsDisposition.ESCALATE]
                > _DISPOSITION_SEVERITY[plan_verdict.disposition]
            ):
                plan_verdict = WeaponsRiskAssessment(
                    hazard_domain=window_verdict.hazard_domain,
                    hazard_weight=window_verdict.hazard_weight,
                    intent_tier=window_verdict.intent_tier,
                    confidence=window_verdict.confidence,
                    disposition=WeaponsDisposition.ESCALATE,
                    signals=("aggregate_realized_plan", *window_verdict.signals),
                )

        # (2) Semantic-accretion of undifferentiated high-severity probing.
        # Concatenation (control 1) is defeated by a *semantically-clean*
        # decomposition that never co-locates offensive phrasing. This control
        # instead measures whether the undifferentiated probes form a cohesive
        # line of enquiry in embedding space: the largest cluster of mutually
        # similar (cosine >= threshold) flagged probes, regardless of how the
        # phrasing or the exact HazardDomain label varied across sub-queries.
        # When that cluster reaches the ceiling it ESCALATES (human-in-the-loop
        # / audit), never denies -- a real engineer is slowed and logged.
        accretion_verdict: WeaponsRiskAssessment | None = None
        flagged = [
            (emb, dom)
            for emb, dom, flag in zip(
                self._embeddings, self._mech_domains, self._mech_flags, strict=True
            )
            if flag
        ]
        if len(flagged) >= self._mechanism_ceiling:
            cluster_idx = self._largest_semantic_cluster([emb for emb, _ in flagged])
            if len(cluster_idx) >= self._mechanism_ceiling:
                cluster_domains = [flagged[i][1] for i in cluster_idx]
                top_domain = max(set(cluster_domains), key=cluster_domains.count)
                logger.info(
                    "aggregate gate: semantic undifferentiated-mechanism accretion "
                    "(%d cohesive probes, dominant domain %s) -> ESCALATE",
                    len(cluster_idx),
                    top_domain.value,
                )
                record_gate_decision(
                    decision="accretion_detected",
                    source="aggregate_gate",
                    disposition="escalate",
                    hazard_domain=top_domain.value,
                    intent="production",
                    signals=("aggregate_semantic_accretion",),
                    reason=(
                        f"{len(cluster_idx)} cohesive undifferentiated-mechanism probes "
                        f"in {top_domain.value} within the session window"
                    ),
                    query=query,
                )
                accretion_verdict = WeaponsRiskAssessment(
                    hazard_domain=top_domain,
                    hazard_weight=1.0,
                    intent_tier=OperationalIntent.PRODUCTION,
                    confidence=0.0,
                    disposition=WeaponsDisposition.ESCALATE,
                    signals=("aggregate_semantic_accretion",),
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
    WeaponsDisposition.ALLOW_PROVENANCE: 2,
    WeaponsDisposition.ESCALATE: 3,
    WeaponsDisposition.REFUSE_REDACT: 4,
    WeaponsDisposition.HARD_REFUSE: 5,
}


__all__ = ["SessionActionabilityTracker"]
