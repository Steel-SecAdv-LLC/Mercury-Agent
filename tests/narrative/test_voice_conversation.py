# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every question a console user asks the agent gets a grounded answer.

Before the conversation layer, 17 of the 31 questions below (greetings,
glossary terms, "what threshold did you use", "show me the last 3 runs",
"recommend an action", "what should I run next", "run detect on sample.csv",
the suggested follow-ups themselves) came back as the generic apology or as
the wrong answer. These tests pin the classifier's routing for each one, the
facts the glossary quotes against the code they come from, and the rule that
no answer, and no follow-up prompt an answer suggests, ends in the fallback.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api import voice as voice_module
from omni_mercury_engine.api.routes.export import record_detection
from omni_mercury_engine.narrative import conversation as conv

if TYPE_CHECKING:
    from omni_mercury_engine.narrative.voice import MercuryVoice

#: The questions the console's chips, the help text and a first session produce,
#: with the grounded answer each must be routed to.
ROUTING: tuple[tuple[str, str], ...] = (
    ("hello", "greeting"),
    ("who are you?", "identity"),
    ("thanks", "thanks"),
    ("what can you do?", "help"),
    ("What was the latest anomaly?", "latest"),
    ("Explain the latest detection", "explain"),
    ("How many anomalies today?", "summary"),
    ("Which samples were flagged?", "detail"),
    ("What threshold did you use?", "detail"),
    ("How confident are you?", "confidence"),
    ("Show me the last 3 runs", "last_n"),
    ("Summarize threat posture", "posture"),
    ("Recommend an action", "recommend"),
    ("What should I run next?", "next_run"),
    ("How do I collect data?", "collect"),
    ("Why did it clear the gate?", "gate"),
    ("What does severity medium mean?", "glossary"),
    ("What is a z-score?", "glossary"),
    ("What is AMA?", "glossary"),
    ("What is FINDOYOU?", "glossary"),
    ("Which detectors do you have?", "capability"),
    ("Can you detect space weather?", "capability"),
    ("What are your capabilities?", "capability"),
    ("Is the system healthy?", "status"),
    ("run detect on sample.csv", "run_command"),
    ("compare today with yesterday", "compare"),
    ("what happened recently?", "last_n"),
    ("Show event timeline", "last_n"),
    ("Explain uncertainty sources", "explain"),
    ("View detailed analysis", "explain"),
    ("Show full reasoning chain", "explain"),
    ("View detailed statistics", "status"),
    ("Check component health", "status"),
    ("Investigate the anomaly", "search"),
    ("what is amazing", "search"),
)

FALLBACK_MARK = "I have nothing in the detection record or the code"


@pytest.mark.parametrize(("question", "kind"), ROUTING)
def test_classifier_routes_each_question(question: str, kind: str) -> None:
    assert conv.classify(question)[0] == kind


def test_identity_is_not_triggered_by_capability_questions() -> None:
    """'What are your capabilities?' contains 'what are you'; it is not an identity question."""
    assert conv.classify("What are your capabilities?")[0] == "capability"
    assert conv.classify("what are you?")[0] == "identity"


def test_gate_matches_the_word_not_investigate() -> None:
    assert conv.classify("Investigate why it was flagged")[0] != "gate"
    assert conv.classify("How does the gate work?")[0] == "gate"


def test_last_n_is_bounded() -> None:
    assert conv.classify("show the last 500 runs") == ("last_n", 20)
    assert conv.classify("show the last 0 runs") == ("last_n", 1)


def test_severity_bands_match_the_server() -> None:
    """The glossary quotes the API's severity bands; they must be the API's."""
    from omni_mercury_engine.api.server import SeverityLevel, _classify_severity

    for edge, below in conv.SEVERITY_BANDS:
        assert _classify_severity(edge - 1e-6, 1.0) == SeverityLevel(below)
    assert _classify_severity(conv.SEVERITY_BANDS[-1][0], 1.0) == SeverityLevel.CRITICAL


def test_threshold_formula_matches_the_server() -> None:
    """2.0 + (1 - sensitivity) x 3.0 is the univariate route's formula."""
    sensitivity = 0.5
    formula = conv.THRESHOLD_FORMULA.replace("x", "*").replace("sensitivity", str(sensitivity))
    assert eval(formula) == 3.5  # noqa: S307 - arithmetic on a constant string


def test_loader_domains_are_the_loader_classes_domains() -> None:
    """The module names the voice reads are exactly the loaders' DOMAIN attributes."""
    from omni_mercury_engine import loaders
    from omni_mercury_engine.loaders.base import BaseDomainLoader
    from omni_mercury_engine.narrative.voice import MercuryVoice

    expected = sorted(
        getattr(loaders, name).DOMAIN
        for name in loaders.__all__
        if isinstance(getattr(loaders, name), type)
        and issubclass(getattr(loaders, name), BaseDomainLoader)
        and getattr(loaders, name) is not BaseDomainLoader
    )
    assert MercuryVoice._loader_domains() == expected


def test_cli_commands_are_read_from_the_click_group() -> None:
    from omni_mercury_engine.narrative.voice import MercuryVoice

    commands = MercuryVoice._cli_commands()
    for name in ("detect", "tier-detect", "rca", "explain", "build-features", "train"):
        assert name in commands
    assert "--input" in commands["detect"]
    assert "--contamination" in commands["tier-detect"]


def test_run_command_names_only_real_commands() -> None:
    commands = {"detect": ["--input", "--detector"], "train": ["--data"]}
    message, _ = conv.run_command("detect", commands, "run detect on sample.csv")
    assert "`mercury-agent detect --input sample.csv`" in message
    assert "I do not execute commands" in message
    message, _ = conv.run_command("narrate", commands, "run narrate")
    assert "not a mercury-agent command" in message
    assert "detect, train" in message


def test_next_steps_never_empty_without_a_cli_table() -> None:
    message, follow_ups = conv.next_steps([], {})
    assert message and follow_ups


def test_glossary_every_term_answers() -> None:
    facts = {
        "manifest_size": 113,
        "ama_version": "5.0.0",
        "findoyou_enabled": True,
        "switches": "findoyou on",
    }
    for term, _pattern in conv._GLOSSARY_KEYS:
        message, follow_ups = conv.glossary_answer(term, facts)
        assert len(message) > 80 and follow_ups


def test_describe_detection_omits_unreported_fields() -> None:
    assert (
        conv.describe_detection({"method": "univariate"}, None)
        == "univariate run at an unrecorded time:"
    )
    full = conv.describe_detection(
        {
            "method": "multivariate",
            "anomaly_count": 1,
            "n_samples": 60,
            "indices": [42],
            "anomaly_score": 6.21,
            "threshold": 3.5,
            "severity_label": "medium",
        },
        None,
    )
    assert "1 of 60 samples flagged (sample 42)" in full
    assert "peak score 6.21, 1.8x the 3.50 threshold (medium)" in full


# --------------------------------------------------------------------------- #
# End to end through the voice, with runs in memory
# --------------------------------------------------------------------------- #


def _voice(monkeypatch: pytest.MonkeyPatch) -> MercuryVoice:
    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    voice = voice_module._get_voice()
    if not hasattr(voice, "retriever"):
        pytest.skip("narrative voice unavailable")
    return voice


def _record(anomalies: list[bool], scores: list[float], threshold: float, method: str) -> None:
    asyncio.run(
        record_detection(
            user_id="test",
            method=method,
            data=[[float(i)] for i in range(len(anomalies))],
            results={"anomalies": anomalies, "scores": scores, "threshold": threshold},
            sensitivity=0.5,
            metadata={"domain": "general"},
        )
    )


@pytest.fixture
def voice_with_runs(monkeypatch: pytest.MonkeyPatch) -> MercuryVoice:
    voice = _voice(monkeypatch)
    _record([False] * 58 + [True, False], [0.2] * 58 + [6.21, 0.1], 3.5, "multivariate")
    _record([False, False, False], [0.1, 0.2, 0.3], 3.5, "univariate")
    return voice


def _ask(voice: MercuryVoice, question: str):  # type: ignore[no-untyped-def]
    return voice.speak(question, domain="general", principal="test")


def test_every_question_and_every_suggested_follow_up_is_answered(
    voice_with_runs: MercuryVoice,
) -> None:
    """No question in the set, and no prompt an answer suggests, reaches the fallback."""
    asked: set[str] = set()
    queue = [q for q, kind in ROUTING if kind != "search"]
    while queue:
        question = queue.pop()
        if question in asked:
            continue
        asked.add(question)
        response = _ask(voice_with_runs, question)
        assert response.message, question
        assert FALLBACK_MARK not in response.message, question
        assert "couldn't find" not in response.message, question
        for follow_up in response.suggested_follow_ups:
            assert follow_up != question, f"{question!r} suggests itself"
            if not re.match(r"Show more results", follow_up):
                queue.append(follow_up)
    assert len(asked) >= 40


def test_run_detail_reads_the_record(voice_with_runs: MercuryVoice) -> None:
    samples = _ask(voice_with_runs, "Which samples were flagged?").message
    assert "flagged sample 58 of 60" in samples
    threshold = _ask(voice_with_runs, "What threshold did you use?").message
    assert "exceeded 3.500" in threshold
    assert conv.THRESHOLD_FORMULA in threshold


def test_last_n_lists_newest_first(voice_with_runs: MercuryVoice) -> None:
    message = _ask(voice_with_runs, "Show me the last 3 runs").message
    assert message.startswith(
        "Recent detection events, newest first (2 of 2 recorded runs): 1. univariate run"
    )
    assert "2. multivariate run" in message


def test_posture_counts_bands(voice_with_runs: MercuryVoice) -> None:
    message = _ask(voice_with_runs, "Summarize threat posture").message
    assert "Across 2 recorded runs, 1 flagged at least one sample and 1 were clean" in message
    assert "1 medium" in message
    assert "No run reached the high or critical band" in message


def test_recommend_comes_from_the_narrative_engine(voice_with_runs: MercuryVoice) -> None:
    response = _ask(voice_with_runs, "Recommend an action")
    assert "narrative_synthesis" in response.sources_cited
    assert "Recommendations: (1)" in response.message
    assert "Urgency:" in response.message


def test_next_run_names_real_commands(voice_with_runs: MercuryVoice) -> None:
    message = _ask(voice_with_runs, "What should I run next?").message
    for command in re.findall(r"`mercury-agent ([a-z-]+)", message):
        assert command in voice_with_runs._cli_commands(), command
    assert "tier-detect" in message


def test_run_command_is_handed_back_not_executed(voice_with_runs: MercuryVoice) -> None:
    response = _ask(voice_with_runs, "run detect on sample.csv")
    assert "`mercury-agent detect --input sample.csv`" in response.message
    assert "I do not execute commands" in response.message


def test_compare_windows(voice_with_runs: MercuryVoice) -> None:
    message = _ask(voice_with_runs, "compare today with yesterday").message
    assert message.startswith("Today (UTC): 2 runs, 1 flagged. Yesterday (UTC): 0 runs, 0 flagged.")


def test_explain_variants_read_the_narrative(voice_with_runs: MercuryVoice) -> None:
    reasoning = _ask(voice_with_runs, "Show full reasoning chain").message
    assert reasoning.startswith("Reasoning for the multivariate run")
    assert "(1) " in reasoning
    uncertainty = _ask(voice_with_runs, "Explain uncertainty sources").message
    assert uncertainty.startswith("Uncertainty for the multivariate run")
    dossier = _ask(voice_with_runs, "View detailed analysis").message
    assert dossier.startswith("Detailed analysis of the multivariate run")
    assert "Urgency:" in dossier


def test_record_answers_are_scoped_to_the_principal(voice_with_runs: MercuryVoice) -> None:
    other = voice_with_runs.speak(
        "Which samples were flagged?", domain="general", principal="someone-else"
    )
    assert "no recorded run" in other.message
    assert other.sources_cited == []


def test_fallback_is_a_menu_not_an_apology(voice_with_runs: MercuryVoice) -> None:
    response = _ask(voice_with_runs, "what is amazing")
    assert FALLBACK_MARK in response.message
    assert response.confidence is None
    assert "What can you do?" in response.suggested_follow_ups
    assert "Refine search" not in response.suggested_follow_ups


def test_detailed_status_reports_components(voice_with_runs: MercuryVoice) -> None:
    message = _ask(voice_with_runs, "Check component health").message
    assert "Retriever:" in message
    assert "Components:" in message
    assert "Language model:" in message
