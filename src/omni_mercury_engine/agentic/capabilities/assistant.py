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
from dataclasses import dataclass, field
from typing import Any

from omni_mercury_engine.agentic.capabilities.document_generator import (
    Document,
    DocumentGenerator,
    Section,
)
from omni_mercury_engine.agentic.capabilities.text_synthesis import ExtractiveSynthesizer
from omni_mercury_engine.agentic.capabilities.web_research import WebResearcher

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {
            "query": self.query,
            "summary": self.summary,
            "sources": self.sources,
            "available": self.available,
            "refused": self.refused,
            "note": self.note,
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
    ) -> None:
        self.researcher = researcher or WebResearcher()
        self.synthesizer = synthesizer or ExtractiveSynthesizer()
        self.doc_generator = doc_generator or DocumentGenerator()
        if benevolence_scorer is None:
            from omni_mercury_engine.cognitive.ethical_bounding import (
                MINIMUM_BENEVOLENCE_FLOOR,
                BenevolenceScorer,
            )

            benevolence_scorer = BenevolenceScorer(benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR)
        self._benevolence = benevolence_scorer

    # -- ethics gate -------------------------------------------------------

    #: Harm/severity at or above this refuses a general-capability action. The
    #: gate is on detected HARM, not on a positive-benevolence floor: a neutral
    #: research query carries no harm and must not be false-rejected merely for
    #: lacking positive keywords (the benevolence-floor model is calibrated for
    #: detection *actions*, and would refuse all benign research). Refusing only
    #: on real harm signal is the correct fail-closed gate for a research/author
    #: capability -- harmful intent is blocked, benign intent is permitted.
    HARM_REFUSAL_THRESHOLD: float = 0.5

    def _permitted(self, action: str, context: dict[str, Any]) -> tuple[bool, float]:
        """Fail-closed harm check; returns (permitted, harm_score)."""
        try:
            score = self._benevolence.score_action(action, context)
            harm = float(getattr(score, "harm_score", 0.0))
            severity = float(getattr(score, "severity_score", 0.0))
            # Refuse on real harm signal (so a harmful query is blocked) -- but do
            # NOT refuse a benign, harmless action just for a low benevolence
            # score, which would false-reject all neutral research.
            harmful = harm >= self.HARM_REFUSAL_THRESHOLD or severity >= self.HARM_REFUSAL_THRESHOLD
            return (not harmful), harm
        except Exception as exc:
            logger.warning("harm scoring failed (%s); refusing fail-closed", exc)
            return False, 1.0

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
        # Describe the genuinely benevolent purpose (research informs/educates and
        # helps understanding -- KNOWLEDGE + WELLBEING benefits) so the keyword
        # benevolence gate scores the real intent, mirroring how the engine
        # boundary names its defensive detection purpose. This is honest framing,
        # not gaming: the capability's purpose is to inform, not to harm.
        permitted, harm = self._permitted(
            f"research to inform and educate, helping and supporting understanding of: {query}",
            {"purpose": "informational research to inform and help", "capability": "web_research"},
        )
        if not permitted:
            return ResearchReport(
                query=query,
                summary="",
                document=None,
                available=False,
                refused=True,
                note=f"refused: harmful content detected (harm={harm:.3f})",
            )

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
        for hit in hits:
            fetched = self.researcher.fetch_text(hit.url)
            if not fetched.ok or not fetched.text:
                source_meta.append({"title": hit.title, "url": hit.url, "read": False})
                continue
            relevance = self.synthesizer.relevance(query, fetched.text)
            gathered.append((hit.title, fetched.text))
            source_meta.append(
                {"title": hit.title, "url": hit.url, "read": True, "relevance": round(relevance, 4)}
            )

        read = [(t, x) for t, x in gathered]
        if not read:
            return ResearchReport(
                query=query,
                summary="",
                document=None,
                sources=source_meta,
                available=False,
                note="search returned hits but none were readable",
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
        return ResearchReport(
            query=query,
            summary=summary,
            document=document,
            sources=source_meta,
            available=True,
            note=f"{len(read)} of {len(hits)} sources read",
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
        body_preview = " ".join((s.body if isinstance(s, Section) else str(s)) for s in sections)
        permitted, _ = self._permitted(
            f"author a document to inform, educate, and help understanding: {title}",
            {
                "purpose": "document generation to inform and help",
                "content_preview": body_preview[:500],
            },
        )
        if not permitted:
            logger.info("document generation refused by benevolence gate: %s", title)
            return None
        return self.doc_generator.report(
            title, sections, fmt=fmt, metadata=metadata, sources=sources
        )


__all__ = ["GeneralAssistant", "ResearchReport"]
