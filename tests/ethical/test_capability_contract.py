# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Capability-as-contract envelope: runtime enforcement + annotation regression.

Two jobs, both CI-failing:

* **Enforcement** -- the ``@capability_contract`` decorator must actually enforce
  ``fail_closed`` / ``cite_or_refuse`` / ``monotone_harm`` at runtime: a raising
  capability yields a typed transparent-negative, an uncited provenance-required
  emission is downgraded to a refusal, and a leaked gate-unsafe span is redacted.
* **Regression** -- the three core capabilities must stay annotated. The
  marker/registry assertions fail CI if an annotation is deleted or its invariant
  set changed, so a refactor cannot silently drop a contracted guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from omni_mercury_engine.agentic.capabilities.assistant import GeneralAssistant, ResearchReport
from omni_mercury_engine.agentic.capabilities.contract import (
    CONTRACT_MARKER,
    ContractViolation,
    Invariant,
    SupportsRefusal,
    capability_contract,
    is_honest_negative,
    registered_contracts,
)
from omni_mercury_engine.agentic.capabilities.text_synthesis import (
    REDACTION_NOTICE,
    ExtractiveSynthesizer,
)
from omni_mercury_engine.agentic.capabilities.web_research import (
    FetchResult,
    SearchResult,
    WebResearcher,
)

# --------------------------------------------------------------------------- #
# Fakes for isolated enforcement tests (no network, no engine gate).
# --------------------------------------------------------------------------- #


@dataclass
class _Report:
    """Minimal ResearchReport-shaped result for cite_or_refuse tests."""

    available: bool = True
    refused: bool = False
    disposition: str = "allow"
    sources: list[dict[str, Any]] = field(default_factory=list)
    document: str | None = "doc"


def _cite_contracted() -> Any:
    """A capability contracted for CITE_OR_REFUSE, returning a caller-supplied report."""

    @capability_contract(
        Invariant.CITE_OR_REFUSE,
        emitted=lambda r, inst: r.available and not r.refused and r.document is not None,
        provenance_required=lambda r, inst: r.disposition == "allow_provenance",
        cited=lambda r, inst: any(bool(s.get("read")) for s in r.sources),
        refuse=lambda r, inst: _Report(
            available=False, refused=True, disposition="allow_provenance", document=None
        ),
        label="fake.cite",
    )
    def run(report: _Report) -> _Report:
        return report

    return run


# --------------------------------------------------------------------------- #
# FAIL_CLOSED
# --------------------------------------------------------------------------- #
class TestFailClosed:
    def test_exception_becomes_honest_negative(self) -> None:
        @capability_contract(
            Invariant.FAIL_CLOSED,
            on_error=lambda exc, args, kwargs: FetchResult(url="x", error=f"fail-closed: {exc}"),
            label="fake.fetch",
        )
        def run() -> FetchResult:
            raise RuntimeError("boom")

        result = run()
        assert result.ok is False
        assert result.error is not None and "fail-closed" in result.error

    def test_success_passes_through_unchanged(self) -> None:
        @capability_contract(
            Invariant.FAIL_CLOSED,
            on_error=lambda exc, args, kwargs: FetchResult(url="x", error="unused"),
            label="fake.fetch.ok",
        )
        def run() -> FetchResult:
            return FetchResult(url="x", status=200, text="hi")

        result = run()
        assert result.ok is True and result.text == "hi"

    def test_fail_closed_is_audited(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        log = tmp_path / "audit.jsonl"
        monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(log))

        @capability_contract(
            Invariant.FAIL_CLOSED,
            on_error=lambda exc, args, kwargs: FetchResult(url="x", error="fail-closed"),
            label="fake.audited",
        )
        def run() -> FetchResult:
            raise ValueError("nope")

        run()
        assert log.exists()
        assert "capability_fail_closed" in log.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# CITE_OR_REFUSE
# --------------------------------------------------------------------------- #
class TestCiteOrRefuse:
    def test_uncited_provenance_emission_is_refused(self) -> None:
        run = _cite_contracted()
        out = run(_Report(disposition="allow_provenance", sources=[{"read": False}]))
        assert out.refused is True and out.available is False

    def test_cited_provenance_emission_passes(self) -> None:
        run = _cite_contracted()
        out = run(_Report(disposition="allow_provenance", sources=[{"read": True}]))
        assert out.refused is False and out.available is True

    def test_benign_allow_needs_no_citation(self) -> None:
        # A plain ALLOW report legitimately carries no citations; must NOT be refused.
        run = _cite_contracted()
        out = run(_Report(disposition="allow", sources=[]))
        assert out.refused is False and out.available is True

    def test_already_refused_passes_through(self) -> None:
        run = _cite_contracted()
        refused = _Report(available=False, refused=True, disposition="hard_refuse", document=None)
        out = run(refused)
        assert out.refused is True


# --------------------------------------------------------------------------- #
# MONOTONE_HARM
# --------------------------------------------------------------------------- #
class TestMonotoneHarm:
    def test_leaked_unsafe_span_is_redacted(self) -> None:
        # A gate that only rejects on the residue re-check would let content
        # through mid-build; the contract must catch and redact it.
        @capability_contract(
            Invariant.MONOTONE_HARM,
            harm_residue=lambda r, inst: (["bad"] if "PAYLOAD" in r else []),
            redact=lambda r, inst: r.replace("PAYLOAD", REDACTION_NOTICE),
            label="fake.syn",
        )
        def run(text: str) -> str:
            return text

        out = run("clean PAYLOAD clean")
        assert "PAYLOAD" not in out and REDACTION_NOTICE in out

    def test_clean_output_unchanged(self) -> None:
        @capability_contract(
            Invariant.MONOTONE_HARM,
            harm_residue=lambda r, inst: [],
            redact=lambda r, inst: "SHOULD_NOT_HAPPEN",
            label="fake.syn.clean",
        )
        def run(text: str) -> str:
            return text

        assert run("all safe") == "all safe"


class TestPostconditionEnforcementIsFailClosed:
    def test_raising_hook_fails_closed_when_paired_with_fail_closed(self) -> None:
        # A postcondition hook that itself raises must NOT escape the wrapped
        # capability when FAIL_CLOSED is also declared -- it falls to on_error.
        def boom(_r: object, _i: object) -> list[str]:
            raise RuntimeError("hook exploded")

        @capability_contract(
            Invariant.FAIL_CLOSED,
            Invariant.MONOTONE_HARM,
            on_error=lambda exc, args, kwargs: "SAFE_FALLBACK",
            harm_residue=boom,
            redact=lambda r, inst: r,
            label="fake.hook.raises",
        )
        def run(text: str) -> str:
            return text

        assert run("anything") == "SAFE_FALLBACK"


# --------------------------------------------------------------------------- #
# Misconfiguration is a loud decoration-time error, never a silent serve-time one
# --------------------------------------------------------------------------- #
class TestMisconfiguration:
    def test_no_invariant_raises(self) -> None:
        with pytest.raises(ContractViolation):
            capability_contract()

    def test_fail_closed_without_sentinel_raises(self) -> None:
        with pytest.raises(ContractViolation):
            capability_contract(Invariant.FAIL_CLOSED)

    def test_cite_or_refuse_missing_hooks_raises(self) -> None:
        with pytest.raises(ContractViolation):
            capability_contract(Invariant.CITE_OR_REFUSE, emitted=lambda r, i: True)

    def test_monotone_harm_missing_hooks_raises(self) -> None:
        with pytest.raises(ContractViolation):
            capability_contract(Invariant.MONOTONE_HARM, redact=lambda r, i: r)


# --------------------------------------------------------------------------- #
# Real capability integration (uses the actual engine gate)
# --------------------------------------------------------------------------- #
class _RaisingResearcher(WebResearcher):
    """A researcher whose fetch always raises, to exercise fail-closed."""

    def fetch(self, url: str) -> FetchResult:
        raise RuntimeError("simulated fetch failure")

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise RuntimeError("simulated search failure")


class TestRealCapabilityContracts:
    def test_fetch_text_fails_closed_to_honest_negative(self) -> None:
        result = _RaisingResearcher().fetch_text("http://example.invalid/x")
        assert isinstance(result, FetchResult)
        assert result.ok is False
        assert result.error is not None and "fail-closed" in result.error

    def test_summarize_never_leaks_a_gate_unsafe_sentence(self) -> None:
        # A keyword gate rejecting any sentence containing 'WEAPONIZE'. Even if a
        # future selection change surfaced such a sentence, the contract redacts.
        gate = lambda s: "WEAPONIZE" not in s  # noqa: E731
        syn = ExtractiveSynthesizer(sentence_gate=gate)
        for text in (
            "This is a safe sentence. Here is how to WEAPONIZE the payload. Another safe one.",
            "WEAPONIZE step one. WEAPONIZE step two. A benign closing remark about safety.",
        ):
            out = syn.summarize(text, max_sentences=3)
            assert "WEAPONIZE" not in out
            assert syn.unsafe_output_spans(out) == []

    def test_monotone_harm_regate_is_independent_and_live(self) -> None:
        # The residue re-check must NOT trust the synthesizer's own output (no
        # diff-skip): re-gating the SAME emitted string under a tightened gate
        # must surface the newly-unsafe span, proving the contract independently
        # re-evaluates rather than short-circuiting to a no-op.
        rejected: set[str] = set()
        gate = lambda s: not any(bad in s for bad in rejected)  # noqa: E731
        syn = ExtractiveSynthesizer(sentence_gate=gate)
        out = syn.summarize(
            "Alpha sentence here is certainly long enough to keep. "
            "Bravo sentence here is also plainly long enough. "
            "Charlie sentence rounds out the set nicely.",
            max_sentences=3,
        )
        assert syn.unsafe_output_spans(out) == []  # clean under the current gate
        # Tighten the gate to reject a token that IS in the emitted output.
        rejected.add(out.split()[0])
        assert syn.unsafe_output_spans(out) != []  # re-gate runs live, catches it

    def test_research_report_fails_closed_on_raising_researcher(self) -> None:
        # An offline/deterministic assistant with a researcher that raises must
        # return a refused report, never propagate the exception.
        assistant = GeneralAssistant(researcher=_RaisingResearcher())
        report = assistant.research_report("what is the capital of France")
        assert isinstance(report, ResearchReport)
        assert report.available is False
        # Either the gate refused or the fetch failed closed; never an exception.


# --------------------------------------------------------------------------- #
# Annotation regression: the 3 core capabilities must stay contracted
# --------------------------------------------------------------------------- #
_EXPECTED = {
    GeneralAssistant.research_report: {Invariant.FAIL_CLOSED, Invariant.CITE_OR_REFUSE},
    WebResearcher.fetch_text: {Invariant.FAIL_CLOSED},
    ExtractiveSynthesizer.summarize: {Invariant.FAIL_CLOSED, Invariant.MONOTONE_HARM},
}


class TestAnnotationRegression:
    @pytest.mark.parametrize(
        ("method", "invariants"),
        [(m, inv) for m, inv in _EXPECTED.items()],
        ids=["research_report", "fetch_text", "summarize"],
    )
    def test_core_capability_is_contracted(self, method: Any, invariants: set[Invariant]) -> None:
        marker = getattr(method, CONTRACT_MARKER, None)
        assert marker is not None, f"{method.__qualname__} lost its @capability_contract"
        assert set(marker) == invariants

    def test_at_least_three_capabilities_registered(self) -> None:
        labels = {label for label, _ in registered_contracts()}
        assert len(labels) >= 3
        assert any("research_report" in label for label in labels)
        assert any("fetch_text" in label for label in labels)
        assert any("summarize" in label for label in labels)


class TestHonestNegativeProtocol:
    def test_research_report_satisfies_supports_refusal(self) -> None:
        assert isinstance(ResearchReport(query="q", summary="", document=None), SupportsRefusal)

    def test_is_honest_negative_detects_refusal(self) -> None:
        refused = ResearchReport(query="q", summary="", document=None, refused=True)
        allowed = ResearchReport(query="q", summary="ok", document=None, refused=False)
        assert is_honest_negative(refused) is True
        assert is_honest_negative(allowed) is False
        assert is_honest_negative(object()) is False
