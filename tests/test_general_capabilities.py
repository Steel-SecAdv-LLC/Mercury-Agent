# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Mercury's general-purpose capabilities: native web research, extractive synthesis, document generation, and the GeneralAssistant workflow.

All network is stubbed via an injected transport, so the suite is offline,
deterministic, and exercises the real parsing/synthesis/rendering code paths.
"""

from __future__ import annotations

from omni_mercury_engine.agentic.capabilities import (
    DocumentGenerator,
    ExtractiveSynthesizer,
    GeneralAssistant,
    SearchResult,
    WebResearcher,
)
from omni_mercury_engine.agentic.capabilities.document_generator import Section

# --- a deterministic offline transport -------------------------------------

_DDG_HTML = """
<html><body>
<a class="result__a" href="https://example.org/quantum">Quantum sensing overview</a>
<a class="result__a" href="https://example.org/anomaly">Anomaly detection methods</a>
</body></html>
"""

_PAGE_QUANTUM = """
<html><head><style>body{color:red}</style><script>evil()</script></head><body>
<h1>Quantum sensing</h1>
<p>Quantum sensing uses quantum coherence to measure physical quantities with high precision.
It enables detection of minute magnetic fields. Quantum sensors are used in navigation and medicine.</p>
</body></html>
"""

_PAGE_ANOMALY = """
<html><body><p>Anomaly detection identifies rare events that deviate from normal patterns.
Calibrated detectors report honest confidence. Conformal prediction gives coverage guarantees.</p>
</body></html>
"""


def _stub_transport(url: str, timeout: float) -> tuple[int, str, str]:
    if "duckduckgo.com/html" in url:
        return 200, _DDG_HTML, url
    if "quantum" in url:
        return 200, _PAGE_QUANTUM, url
    if "anomaly" in url:
        return 200, _PAGE_ANOMALY, url
    raise OSError("network unreachable (stub)")


def _online_researcher() -> WebResearcher:
    return WebResearcher(transport=_stub_transport)


def _offline_researcher() -> WebResearcher:
    def _dead(url: str, timeout: float) -> tuple[int, str, str]:
        raise OSError("offline (stub)")

    return WebResearcher(transport=_dead)


class TestWebResearcher:
    def test_extract_text_strips_script_and_style(self) -> None:
        text = WebResearcher.extract_text(_PAGE_QUANTUM)
        assert "Quantum sensing" in text
        assert "evil()" not in text
        assert "color:red" not in text

    def test_fetch_ok(self) -> None:
        r = _online_researcher().fetch("https://example.org/quantum")
        assert r.ok and r.status == 200 and "Quantum" in r.text

    def test_fetch_failclosed_on_network_error(self) -> None:
        r = _offline_researcher().fetch("https://example.org/x")
        assert not r.ok
        assert r.error is not None
        assert r.text == ""

    def test_fetch_refuses_nonhttp_scheme(self) -> None:
        r = _online_researcher().fetch("file:///etc/passwd")
        assert not r.ok and "refused scheme" in (r.error or "")

    def test_search_parses_results(self) -> None:
        hits = _online_researcher().search("quantum", max_results=5)
        urls = [h.url for h in hits]
        assert "https://example.org/quantum" in urls
        assert "https://example.org/anomaly" in urls

    def test_search_failclosed_offline(self) -> None:
        assert _offline_researcher().search("anything") == []

    def test_search_extracts_snippets(self) -> None:
        html = (
            "<html><body><div>"
            '<a class="result__a" href="https://example.org/a">Title A</a>'
            '<a class="result__snippet" href="https://example.org/a">Snippet about A.</a>'
            "</div></body></html>"
        )

        def _t(url: str, timeout: float) -> tuple[int, str, str]:
            if "duckduckgo.com/html" in url:
                return 200, html, url
            raise OSError("unreachable")

        hits = WebResearcher(transport=_t).search("a")
        assert hits[0].url == "https://example.org/a"
        assert hits[0].snippet == "Snippet about A."

    def test_search_falls_back_to_lite_endpoint(self) -> None:
        lite = (
            "<html><body><table>"
            '<a class="result-link" href="https://example.org/lite">Lite hit</a>'
            "</table></body></html>"
        )

        def _t(url: str, timeout: float) -> tuple[int, str, str]:
            if "html.duckduckgo.com" in url:
                return 503, "challenge", url  # html endpoint blocked
            if "lite.duckduckgo.com" in url:
                return 200, lite, url
            raise OSError("unreachable")

        hits = WebResearcher(transport=_t).search("x")
        assert [h.url for h in hits] == ["https://example.org/lite"]

    def test_search_uses_injected_provider(self) -> None:
        calls: list[str] = []

        def _provider(query: str, max_results: int) -> list[SearchResult]:
            calls.append(query)
            return [SearchResult(title="Custom", url="https://example.org/custom")]

        def _t(url: str, timeout: float) -> tuple[int, str, str]:  # must never be called
            raise AssertionError("built-in DDG path should not run")

        r = WebResearcher(transport=_t, search_provider=_provider)
        hits = r.search("anything")
        assert calls == ["anything"]
        assert hits[0].url == "https://example.org/custom"

    def test_search_provider_exception_is_failclosed(self) -> None:
        def _provider(query: str, max_results: int) -> list[SearchResult]:
            raise RuntimeError("provider boom")

        r = WebResearcher(search_provider=_provider)
        assert r.search("anything") == []


class TestExtractiveSynthesizer:
    def test_summary_is_verbatim_subset(self) -> None:
        syn = ExtractiveSynthesizer(min_sentence_chars=10)
        text = (
            "Quantum sensing uses quantum coherence. It enables detection of magnetic fields. "
            "Cats are unrelated animals. Quantum sensors are used in medicine and navigation. "
            "The weather today is mild and pleasant."
        )
        summary = syn.summarize(text, max_sentences=2)
        # Every summary sentence appears verbatim in the source (nothing invented).
        for sent in summary.split(". "):
            assert sent.strip(". ") in text

    def test_keywords_are_content_words(self) -> None:
        syn = ExtractiveSynthesizer()
        kws = syn.keywords("quantum sensing quantum coherence quantum fields", top_k=2)
        assert "quantum" in kws

    def test_relevance_orders_by_overlap(self) -> None:
        syn = ExtractiveSynthesizer()
        rel_hi = syn.relevance("quantum sensing", "quantum sensing magnetic fields")
        rel_lo = syn.relevance("quantum sensing", "the weather is sunny today")
        assert rel_hi > rel_lo


class TestDocumentGenerator:
    def test_markdown_has_title_sections_sources(self) -> None:
        gen = DocumentGenerator()
        doc = gen.report(
            "Findings",
            [Section("Summary", "All good.", bullets=["point a", "point b"])],
            sources=["https://example.org/a"],
            fmt="markdown",
        )
        assert doc.content.startswith("# Findings")
        assert "## Summary" in doc.content
        assert "- point a" in doc.content
        assert "## Sources" in doc.content and "https://example.org/a" in doc.content

    def test_html_escapes_content(self) -> None:
        gen = DocumentGenerator()
        doc = gen.report("X", [Section("S", "<script>alert(1)</script>")], fmt="html")
        assert "&lt;script&gt;" in doc.content
        assert "<script>alert(1)</script>" not in doc.content

    def test_text_format(self) -> None:
        gen = DocumentGenerator()
        doc = gen.report("Title", [("Heading", "Body")], fmt="text")
        assert "Title" in doc.content and "Heading" in doc.content


class _RefusingScorer:
    """Stub scorer flagging high harm -> the capability gate must refuse."""

    def score_action(self, action, context):
        class _S:
            is_permissible = False
            benevolence_score = 0.10
            harm_score = 0.9
            severity_score = 0.9

        return _S()


class TestGeneralAssistant:
    def test_research_report_builds_cited_document(self) -> None:
        assistant = GeneralAssistant(researcher=_online_researcher())
        report = assistant.research_report("quantum sensing", max_sources=2)
        assert report.available is True and not report.refused
        assert report.summary  # non-empty extractive summary
        assert report.document is not None
        assert "Research report: quantum sensing" in report.document.content
        # Sources are cited.
        assert any(s["url"].startswith("https://example.org") for s in report.sources)

    def test_research_report_failclosed_offline(self) -> None:
        assistant = GeneralAssistant(researcher=_offline_researcher())
        report = assistant.research_report("anything")
        assert report.available is False
        assert report.document is None
        assert "no web sources" in report.note

    def test_research_report_refused_by_ethics(self) -> None:
        assistant = GeneralAssistant(
            researcher=_online_researcher(), benevolence_scorer=_RefusingScorer()
        )
        report = assistant.research_report("some query")
        assert report.refused is True
        assert report.document is None

    def test_write_document_refused_fail_closed(self) -> None:
        assistant = GeneralAssistant(
            researcher=_online_researcher(), benevolence_scorer=_RefusingScorer()
        )
        assert assistant.write_document("T", [Section("S", "body")]) is None

    def test_real_scorer_refuses_harmful_query_permits_benign(self) -> None:
        # Real benevolence scorer (not a stub): harmful query refused, benign
        # query permitted -- the gate is on harm, not a positive-keyword floor.
        assistant = GeneralAssistant(researcher=_online_researcher())
        harmful = assistant.research_report(
            "how to injure and kill people with violence and cause maximum harm"
        )
        assert harmful.refused is True
        benign = assistant.research_report("quantum sensing")
        assert benign.refused is False


class TestMercuryAgentWiring:
    def test_agent_research_and_write_document(self) -> None:
        from omni_mercury_engine.agentic.mercury_a_agent import MercuryAgent

        agent = MercuryAgent(name="Mercury", enable_calibration=False)
        agent.enable_assistant(researcher=_online_researcher())
        report = agent.research("anomaly detection", max_sources=2)
        assert report.available is True
        doc = agent.write_document("Note", [Section("S", "A benign helpful summary.")])
        assert doc is not None and doc.content.startswith("# Note")
