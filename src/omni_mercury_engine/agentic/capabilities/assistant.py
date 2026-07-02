# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""GeneralAssistant -- Mercury's general-purpose, tool-using capability layer.

Ties the native capabilities into useful workflows: research a question on the
open web, synthesize the sources extractively, and produce a cited document.
Every outward action passes the same fail-closed benevolence gate the rest of
Mercury uses, so a general capability cannot be used to research or author
content the ethics gate refuses.

Honest by construction:
- It is NOT a language model. It does not generate prose; it extracts and
  organizes content that sources actually contain, and cites them.
- It is fail-closed on the network: if no source is reachable, the report says
  so plainly instead of fabricating an answer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.agentic.capabilities.document_generator import (
    Document,
    DocumentGenerator,
    Section,
)
from omni_mercury_engine.agentic.capabilities.text_synthesis import ExtractiveSynthesizer
from omni_mercury_engine.agentic.capabilities.web_research import WebResearcher
from omni_mercury_engine.cognitive.escalation import EscalationBroker, EscalationRecord
from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Warn-once latch so the lexical-only advisory is emitted a single time per
# process rather than on every GeneralAssistant construction.
_MEANING_LEVEL_CLASSIFIER_WARNED = False


def _warn_or_require_meaning_level_classifier() -> None:
    """Loudly surface a lexical-only harm posture; optionally fail closed.

    When no real meaning-level model backs the harm classifier, the weapons
    gate's routing rescue cannot fire and the surface runs lexical-only (a
    materially higher held-out false-negative rate -- see
    ``docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md``). This warns exactly once per
    process so the degraded posture is never silent. With
    ``MERCURY_REQUIRE_REAL_HARM_CLASSIFIER=1`` it raises instead, so deployments
    that mandate meaning-level screening fail closed at enablement rather than
    serving the generative surface with lexical-only coverage.
    """
    global _MEANING_LEVEL_CLASSIFIER_WARNED
    from omni_mercury_engine.cognitive.harm_classifier import real_harm_classifier_available

    if real_harm_classifier_available():
        return
    message = (
        "GeneralAssistant meaning-level harm classifier is INACTIVE: no real "
        "reasoning model is serving (template/mock backend, missing reasoning "
        "stack, or MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER=1), so the weapons "
        "gate runs LEXICAL-ONLY. The meaning-level routing rescue that catches "
        "paraphrased / out-of-lexicon offensive requests will NOT fire; the "
        "held-out false-negative rate is materially higher in this posture "
        "(see docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md). Configure a real model "
        "(e.g. Ollama, or a RemoteReasoningBackend) to enable it."
    )
    if os.environ.get("MERCURY_REQUIRE_REAL_HARM_CLASSIFIER") == "1":
        raise RuntimeError(
            message + " MERCURY_REQUIRE_REAL_HARM_CLASSIFIER=1 makes this a "
            "fatal enablement error (fail-closed)."
        )
    if not _MEANING_LEVEL_CLASSIFIER_WARNED:
        logger.warning(message)
        _MEANING_LEVEL_CLASSIFIER_WARNED = True


def _section_text(s: Section | tuple[Any, ...] | dict[str, Any]) -> str:
    """Return all caller-authored, renderable text of a section for screening.

    Every field that ends up in the rendered document -- heading, body, and
    bullets -- must be visible to the harm gate. Accepts the same shapes
    :meth:`DocumentGenerator.report` does (``Section``, ``(heading, body)``
    tuple, ``{"heading","body","bullets"}`` dict) so the gate sees exactly what
    will be rendered, never a subset.
    """
    if isinstance(s, Section):
        return " ".join([s.heading, s.body, *s.bullets])
    if isinstance(s, dict):
        parts = [str(s.get("heading", "")), str(s.get("body", ""))]
        parts.extend(str(b) for b in s.get("bullets", []) or [])
        return " ".join(parts)
    if isinstance(s, tuple):
        return " ".join(str(x) for x in s)
    return str(s)


@dataclass
class GateVerdict:
    """Outcome of the unified harm gate for one general-capability action.

    Carries the two-axis weapons/mass-casualty verdict alongside the scalar
    harm so callers can distinguish an outright HARD_REFUSE from an ESCALATE
    (a genuine gray-zone request a human-in-the-loop could authorize) and
    surface an honest, auditable reason rather than a bare boolean.
    """

    permitted: bool
    harm_score: float = 0.0
    disposition: str = "allow"
    hazard_domain: str = "none"
    operational_intent: str = "mechanism"
    reason: str = ""

    @property
    def escalatable(self) -> bool:
        """True when a human-in-the-loop could authorize (B6 gray zone)."""
        return self.disposition == "escalate"

    @property
    def provenance_required(self) -> bool:
        """True when the answer is permitted but must be source-attributed.

        Set for an otherwise-allowable query in a high-severity hazard domain
        (ALLOW_PROVENANCE): the output boundary must withhold rather than emit
        uncited synthesis on a hazardous topic.
        """
        return self.disposition == "allow_provenance"


@dataclass
class ResearchReport:
    """Result of a research-and-report workflow."""

    query: str
    summary: str
    document: Document | None
    sources: list[dict[str, Any]] = field(default_factory=list)
    available: bool = True
    refused: bool = False
    note: str = ""
    disposition: str = "allow"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {
            "query": self.query,
            "summary": self.summary,
            "sources": self.sources,
            "available": self.available,
            "refused": self.refused,
            "note": self.note,
            "disposition": self.disposition,
            "document": self.document.content if self.document else None,
            "format": self.document.fmt if self.document else None,
        }


class GeneralAssistant:
    """General-purpose research + document assistant for Mercury.

    Args:
        researcher: A :class:`WebResearcher` (defaults to a urllib-backed one;
            inject a stub for offline/deterministic use).
        synthesizer: An :class:`ExtractiveSynthesizer`.
        doc_generator: A :class:`DocumentGenerator`.
        benevolence_scorer: An object with ``score_action(action, context) ->
            EthicalScore`` (``.is_permissible``). Defaults to Mercury's
            fail-closed :class:`BenevolenceScorer` at the floor threshold.
    """

    def __init__(
        self,
        researcher: WebResearcher | None = None,
        synthesizer: ExtractiveSynthesizer | None = None,
        doc_generator: DocumentGenerator | None = None,
        benevolence_scorer: Any | None = None,
        *,
        escalation_reviewer: Callable[[EscalationRecord], bool] | None = None,
        escalation_max_approvals: int = 3,
    ) -> None:
        """Wire the researcher, synthesizer, document generator, and ethics gate.

        ``escalation_reviewer`` is the human-in-the-loop hook consulted on an
        ESCALATE verdict (a gray-zone request). With none wired, escalations are
        denied fail-closed; a wired reviewer may authorize up to
        ``escalation_max_approvals`` per session (bounded autonomy).
        """
        # Reasoning-backed harm classifier, wired by default on this open-web/
        # text surface (the highest-uplift capability). Fail-open and offline-
        # safe: it contributes a meaning-level harm probability only when a real
        # local/cloud model is serving, and 0.0 otherwise -- so it strengthens
        # the gate where a semantic model is available without adding a hard
        # dependency or changing deterministic behavior in air-gapped/CI runs.
        from omni_mercury_engine.cognitive.harm_classifier import default_harm_classifier

        self._harm_classifier = default_harm_classifier()
        # Enablement check: the meaning-level routing rescue only cuts the
        # gate's held-out false-negatives when a REAL model backs the classifier
        # (see docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md). Under a template/absent
        # model the surface runs LEXICAL-ONLY -- warn loudly (once) so this is a
        # visible operational choice, never a silent degradation. Set
        # MERCURY_REQUIRE_REAL_HARM_CLASSIFIER=1 to make it fatal (fail-closed
        # enablement) for deployments that mandate meaning-level screening.
        _warn_or_require_meaning_level_classifier()
        # Default to the env-configured provider ladder (keyed engine / keyless
        # self-hosted SearXNG first, DuckDuckGo scrape only as fallback) rather
        # than a bare DDG scrape -- the provider-first posture, configurable
        # without code via BRAVE_API_KEY / MERCURY_SEARXNG_URL.
        self.researcher = researcher or WebResearcher.from_env(
            harm_classifier=self._harm_classifier
        )
        # Pre-emission output gate (spec §5.3): the verbatim extractor must not
        # reproduce operational weapons procedure even from a source that
        # passed the query/content gates. A default synthesizer is wired to the
        # weapons output gate; an injected synthesizer is used as-is (the caller
        # owns its gating). This is the highest-leverage enforcement point.
        self.synthesizer = synthesizer or ExtractiveSynthesizer(
            sentence_gate=self._sentence_is_safe
        )
        self.doc_generator = doc_generator or DocumentGenerator()
        if benevolence_scorer is None:
            from omni_mercury_engine.cognitive.ethical_bounding import (
                MINIMUM_BENEVOLENCE_FLOOR,
                BenevolenceScorer,
            )

            benevolence_scorer = BenevolenceScorer(
                benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR,
                harm_classifier=self._harm_classifier,
            )
        self._benevolence = benevolence_scorer

        # Orchestration-boundary aggregate gate (spec §6): tracks the session's
        # query sequence so a task decomposed into individually-benign
        # sub-queries is still evaluated as a realized plan / accretion.
        from omni_mercury_engine.agentic.capabilities.aggregate_gate import (
            SessionActionabilityTracker,
        )

        self._session_tracker = SessionActionabilityTracker()

        # Human-in-the-loop / bounded-autonomy broker for ESCALATE verdicts.
        self._escalation = EscalationBroker(
            reviewer=escalation_reviewer, max_approvals=escalation_max_approvals
        )

    # -- ethics gate (unified with the rest of Mercury) --------------------

    #: General-harm floor: harm/severity at or above this refuses a
    #: general-capability action. The gate is on detected HARM, not on a
    #: positive-benevolence floor: a neutral research query carries no harm
    #: and must not be false-rejected merely for lacking positive keywords
    #: (the benevolence-floor model is calibrated for detection *actions* and
    #: would refuse all benign research). This catches interpersonal-harm
    #: intent; the weapons/mass-casualty uplift gate (below) is the
    #: authoritative control for CBRN/explosive/weapon actionability, and both
    #: are computed in the SAME ``score_action`` call -- one harm policy.
    HARM_REFUSAL_THRESHOLD: float = 0.5

    def _sentence_is_safe(self, sentence: str) -> bool:
        """Output-gate predicate: True unless a sentence is operational weapons content.

        Backs the :class:`ExtractiveSynthesizer` sentence gate. Uses the same
        two-axis :func:`assess_weapons_uplift` as every other gate (with the
        reasoning-backed classifier wired), so the output layer redacts on
        exactly the intersection the rest of the policy refuses. Fail-closed:
        any error redacts (returns ``False``).
        """
        try:
            from omni_mercury_engine.cognitive.ethical_bounding import assess_weapons_uplift

            return not assess_weapons_uplift(sentence, harm_classifier=self._harm_classifier).blocks
        except Exception:
            return False

    def _permitted(self, action: str, context: dict[str, Any]) -> GateVerdict:
        """Unified fail-closed harm gate for a general-capability action.

        Routes through :meth:`BenevolenceScorer.score_action` -- the *same*
        gate detect/analyze/predict use -- so the open-web/authoring surface
        (the highest-uplift capability in the system) inherits the two-axis
        weapons/mass-casualty verdict instead of a weaker bespoke check. An
        action is refused when EITHER the weapons gate returns a blocking
        disposition (ESCALATE/REFUSE_REDACT/HARD_REFUSE) OR the scalar
        harm/severity crosses :attr:`HARM_REFUSAL_THRESHOLD`. Benign,
        harmless research still passes (low benevolence alone never refuses).
        Any scoring error fails closed to refused.
        """
        try:
            score = self._benevolence.score_action(action, context)
            harm = float(getattr(score, "harm_score", 0.0))
            severity = float(getattr(score, "severity_score", 0.0))
            disposition = str(getattr(score, "weapons_disposition", "allow"))
            hazard = str(getattr(score, "hazard_domain", "none"))
            intent = str(getattr(score, "operational_intent", "mechanism"))
            weapons_blocks = disposition in ("escalate", "refuse_redact", "hard_refuse")
            harmful = harm >= self.HARM_REFUSAL_THRESHOLD or severity >= self.HARM_REFUSAL_THRESHOLD
            if weapons_blocks:
                reason = (
                    f"weapons/mass-casualty uplift gate: {disposition} "
                    f"(hazard={hazard}, intent={intent})"
                )
            elif harmful:
                reason = f"harmful content detected (harm={harm:.3f}, severity={severity:.3f})"
            else:
                reason = ""
            return GateVerdict(
                permitted=not (weapons_blocks or harmful),
                harm_score=harm,
                disposition=disposition,
                hazard_domain=hazard,
                operational_intent=intent,
                reason=reason,
            )
        except Exception as exc:
            logger.warning("harm scoring failed (%s); refusing fail-closed", exc)
            return GateVerdict(
                permitted=False,
                harm_score=1.0,
                disposition="hard_refuse",
                reason=f"gate error: {exc}",
            )

    def _adjudicate_block(
        self,
        *,
        query: str,
        disposition: str,
        hazard_domain: str,
        intent: str,
        signals: tuple[str, ...],
        reason: str,
        escalatable: bool,
        source: str,
    ) -> tuple[bool, str]:
        """Audit a blocking verdict and, if escalatable, consult the HITL broker.

        Returns ``(proceed, note)``. An escalatable verdict is routed to the
        :class:`EscalationBroker`: on approval the request proceeds (and its
        output is treated as provenance-required); otherwise it is refused. Every
        outcome is written to the durable gate audit log (the broker audits its
        own decisions; non-escalatable refusals are audited here).
        """
        if escalatable:
            decision = self._escalation.review(
                EscalationRecord(
                    query=query,
                    reason=reason,
                    disposition=disposition,
                    hazard_domain=hazard_domain,
                    intent=intent,
                    signals=tuple(signals),
                )
            )
            if decision.approved:
                return True, f"escalation authorized by human-in-the-loop: {decision.reason}"
            return False, (
                f"escalate: {reason} -- a licensed/authorized human-in-the-loop review is "
                f"required before this query can proceed ({decision.reason})"
            )
        record_gate_decision(
            decision="refused",
            source=source,
            disposition=disposition,
            hazard_domain=hazard_domain,
            intent=intent,
            signals=signals,
            reason=reason,
            query=query,
        )
        return False, f"refused: {reason}"

    # -- capabilities ------------------------------------------------------

    def summarize_url(self, url: str, max_sentences: int = 5) -> str:
        """Fetch a single URL and return an extractive summary (or an honest error)."""
        result = self.researcher.fetch_text(url)
        if not result.ok:
            return f"[unavailable] could not read {url}: {result.error}"
        return self.synthesizer.summarize(result.text, max_sentences=max_sentences)

    def research_report(
        self,
        query: str,
        *,
        max_sources: int = 5,
        max_summary_sentences: int = 6,
        fmt: str = "markdown",
    ) -> ResearchReport:
        """Research ``query`` on the open web and produce a cited report.

        Pipeline: benevolence gate -> web search -> fetch + extract each hit ->
        rank by relevance to the query -> extractive synthesis -> cited document.
        Fail-closed at every step: a refused query, an unreachable network, or
        zero readable sources each yield an honest report (``refused`` /
        ``available=False``) rather than a fabricated answer.
        """
        # Pre-retrieval intent gate (spec §5.1). Score the RAW query for
        # weapons/mass-casualty uplift (Axis A/B is intent-driven and does not
        # need the benevolent-purpose framing), while still naming the benign
        # informational purpose so the scalar benevolence/harm keyword gate
        # reads the real intent rather than false-rejecting neutral research.
        verdict = self._permitted(
            f"research to inform and educate, helping understanding of: {query}\n"
            f"raw query: {query}",
            {"purpose": "informational research to inform and help", "capability": "web_research"},
        )
        # Output on a hazardous-but-allowable topic must be source-attributed
        # (ALLOW_PROVENANCE); an approved escalation is treated the same way.
        provenance_required = verdict.provenance_required
        if not verdict.permitted:
            proceed, note = self._adjudicate_block(
                query=query,
                disposition=verdict.disposition,
                hazard_domain=verdict.hazard_domain,
                intent=verdict.operational_intent,
                signals=(),
                reason=verdict.reason,
                escalatable=verdict.escalatable,
                source="assistant.research_report",
            )
            if not proceed:
                return ResearchReport(
                    query=query,
                    summary="",
                    document=None,
                    available=False,
                    refused=True,
                    note=note,
                    disposition=verdict.disposition,
                )
            provenance_required = True  # an approved gray-zone query must be cited

        # Orchestration-boundary aggregate gate (spec §6): even though this
        # query passed on its own, evaluate the realized plan / accretion over
        # the session's query sequence, so a decomposition into individually-
        # benign sub-queries cannot assemble a blocked procedure unchecked.
        aggregate = self._session_tracker.record_and_assess(query)
        if aggregate.blocks:
            disp = aggregate.disposition.value
            proceed, note = self._adjudicate_block(
                query=query,
                disposition=disp,
                hazard_domain=aggregate.hazard_domain.value,
                intent=aggregate.intent_tier.value,
                signals=aggregate.signals,
                reason=(
                    "aggregate harm gate across the session's query sequence "
                    f"(hazard={aggregate.hazard_domain.value}, "
                    f"intent={aggregate.intent_tier.value}); a decomposition of a "
                    "restricted request was detected"
                ),
                escalatable=(disp == "escalate"),
                source="assistant.aggregate_gate",
            )
            if not proceed:
                return ResearchReport(
                    query=query,
                    summary="",
                    document=None,
                    available=False,
                    refused=True,
                    note=note,
                    disposition=disp,
                )
            provenance_required = True

        hits = self.researcher.search(query, max_results=max_sources)
        if not hits:
            return ResearchReport(
                query=query,
                summary="",
                document=None,
                available=False,
                note="no web sources reachable (offline, blocked, or no results)",
            )

        gathered: list[tuple[str, str]] = []
        source_meta: list[dict[str, Any]] = []
        dropped_for_harm = 0
        for hit in hits:
            fetched = self.researcher.fetch_text(hit.url)
            if not fetched.ok or not fetched.text:
                source_meta.append({"title": hit.title, "url": hit.url, "read": False})
                continue
            # Post-retrieval content gate (spec §5.2). A benign query can still
            # return a page carrying operational weapons procedure; that content
            # must never reach the verbatim synthesizer. A from_env researcher
            # already screened the fetch (verdict on the FetchResult); for an
            # injected researcher that did not screen, apply the same gate here
            # as a fallback so the guarantee holds regardless of researcher.
            if fetched.screened:
                content_blocked = fetched.harm_blocked
                content_note = fetched.harm_note
            else:
                cv = self._permitted(
                    fetched.text[:4000],
                    {"purpose": "post-retrieval content screen", "capability": "web_research"},
                )
                content_blocked = not cv.permitted
                content_note = cv.disposition
            if content_blocked:
                dropped_for_harm += 1
                source_meta.append(
                    {
                        "title": hit.title,
                        "url": hit.url,
                        "read": False,
                        "dropped": "harm_gate",
                        "note": content_note,
                    }
                )
                continue
            relevance = self.synthesizer.relevance(query, fetched.text)
            gathered.append((hit.title, fetched.text))
            source_meta.append(
                {"title": hit.title, "url": hit.url, "read": True, "relevance": round(relevance, 4)}
            )

        read = [(t, x) for t, x in gathered]
        if not read:
            if dropped_for_harm:
                note = (
                    f"refused: all {dropped_for_harm} readable source(s) were dropped by the "
                    "post-retrieval harm gate (operational weapons content)"
                )
                refused = True
                disp = "refuse_redact"
            elif provenance_required:
                # ALLOW_PROVENANCE enforcement: a hazardous-topic query may be
                # answered only from cited sources. With none readable, withhold
                # rather than emit uncited synthesis on a hazardous topic.
                note = (
                    "withheld: this hazardous-topic query may be answered only from cited "
                    "sources (provenance required), but no citable source was readable"
                )
                refused = True
                disp = "allow_provenance"
                record_gate_decision(
                    decision="provenance_withheld",
                    source="assistant.research_report",
                    disposition="allow_provenance",
                    reason=note,
                    query=query,
                )
            else:
                note = "search returned hits but none were readable"
                refused = False
                disp = "allow"
            return ResearchReport(
                query=query,
                summary="",
                document=None,
                sources=source_meta,
                available=False,
                refused=refused,
                note=note,
                disposition=disp,
            )

        # Rank sources by relevance; synthesize across the most relevant ones.
        read.sort(key=lambda tx: self.synthesizer.relevance(query, tx[1]), reverse=True)
        summary = self.synthesizer.summarize_sources(read, max_sentences=max_summary_sentences)
        keywords = self.synthesizer.keywords(" ".join(x for _, x in read), top_k=10)

        sections = [
            Section("Summary", summary),
            Section(
                "Per-source highlights",
                "",
                bullets=[
                    f"{title}: {self.synthesizer.summarize(text, max_sentences=1)}"
                    for title, text in read[:max_sources]
                ],
            ),
            Section("Key terms", "", bullets=keywords),
            Section(
                "Method & honesty",
                "Extractive synthesis over the cited sources (no language model; "
                "sentences are quoted verbatim and ranked by centrality/relevance). "
                "Verify against the primary sources before relying on this.",
            ),
        ]
        document = self.doc_generator.report(
            title=f"Research report: {query}",
            sections=sections,
            metadata={
                "capability": "GeneralAssistant.research_report",
                "sources_read": str(len(read)),
            },
            sources=[m["url"] for m in source_meta],
            fmt=fmt,
        )
        if provenance_required:
            # Emitted with citations -> provenance satisfied; record the decision.
            record_gate_decision(
                decision="allow_provenance_emitted",
                source="assistant.research_report",
                disposition="allow_provenance",
                reason=f"answered from {len(read)} cited source(s)",
                query=query,
            )
        return ResearchReport(
            query=query,
            summary=summary,
            document=document,
            sources=source_meta,
            available=True,
            note=f"{len(read)} of {len(hits)} sources read",
            disposition="allow_provenance" if provenance_required else "allow",
        )

    def answer(self, question: str, *, max_sources: int = 3) -> str:
        """Extractively answer a question from researched sources (honest, cited).

        Returns the most query-relevant sentences from the reachable sources, or
        an honest unavailability message. Not a generated answer -- extracted,
        verbatim, from sources.
        """
        report = self.research_report(question, max_sources=max_sources, max_summary_sentences=4)
        if report.refused:
            return f"[refused] {report.note}"
        if not report.available:
            return f"[unavailable] {report.note}"
        cites = [s["url"] for s in report.sources if s.get("read")]
        suffix = f"\n\nSources: {', '.join(cites)}" if cites else ""
        return report.summary + suffix

    def write_document(
        self,
        title: str,
        sections: list[Section] | list[tuple[str, str]] | list[dict[str, Any]],
        *,
        fmt: str = "markdown",
        metadata: dict[str, str] | None = None,
        sources: list[str] | None = None,
    ) -> Document | None:
        """Generate a document from structured content, gated by the ethics check.

        Returns ``None`` (fail-closed) if the title/content is refused by the
        benevolence gate.
        """
        # Gate the title AND the *entire* caller-authored content -- every
        # rendered field, not just the body. A Section's heading and bullets are
        # rendered into the output document too, so screening only ``s.body``
        # (as an earlier version did) let harmful content ride in the heading or
        # a bullet straight past the benevolence/weapons gate. The MCP surface
        # coerces client dicts into Section objects before this call, so that
        # bypass was reachable by any MCP client.
        body_preview = " ".join(_section_text(s) for s in sections)
        verdict = self._permitted(
            f"author a document to inform, educate, and help understanding: {title}\n{body_preview}",
            {
                "purpose": "document generation to inform and help",
                "content_preview": body_preview[:500],
            },
        )
        provenance_required = verdict.provenance_required
        if not verdict.permitted:
            proceed, note = self._adjudicate_block(
                query=title,
                disposition=verdict.disposition,
                hazard_domain=verdict.hazard_domain,
                intent=verdict.operational_intent,
                signals=(),
                reason=verdict.reason,
                escalatable=verdict.escalatable,
                source="assistant.write_document",
            )
            if not proceed:
                logger.info("document generation blocked by harm gate (%s): %s", note, title)
                return None
            provenance_required = True
        if provenance_required and not sources:
            # ALLOW_PROVENANCE: a hazardous-topic document must carry source
            # attribution. Withhold rather than emit uncited hazardous content.
            record_gate_decision(
                decision="provenance_withheld",
                source="assistant.write_document",
                disposition="allow_provenance",
                reason="document on a hazardous topic requires cited sources; none supplied",
                query=title,
            )
            logger.info("document generation withheld (provenance required, no sources): %s", title)
            return None
        return self.doc_generator.report(
            title, sections, fmt=fmt, metadata=metadata, sources=sources
        )


__all__ = ["GeneralAssistant", "ResearchReport"]
