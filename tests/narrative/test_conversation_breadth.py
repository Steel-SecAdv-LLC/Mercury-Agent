# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agent holds an ordinary conversation, not only a question-and-answer session.

Measured before this layer: of 70 realistic inputs a console user types
("ok", "anything new?", "should I be worried?", "what do you mean?", "say
that again", "that's not what I asked", "how do I use this?", "detect
anomalies in sensor.csv", "bye", and the same in six other languages), 52
fell through to the free-text search and ended in the fallback menu. These
tests pin the routing of each conversational turn, that a reaction never
overwrites the follow-up context ("ok" then "why?" still explains the run
that was discussed), that "say that again" and "in short" rebuild the
previous answer, that a correction never repeats the answer it corrects,
and that no turn in the battery ends in the fallback.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.narrative import conversation as conv

ROUTING: tuple[tuple[str, str], ...] = (
    ("ok", "ack"),
    ("yes", "ack"),
    ("got it", "ack"),
    ("cool!", "ack"),
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("never mind", "farewell"),
    ("anything new?", "latest"),
    ("any problems?", "latest"),
    ("is everything ok?", "latest"),
    ("what's wrong?", "latest"),
    ("what did you find?", "latest"),
    ("show me what you found", "latest"),
    ("did you detect anything in my data?", "latest"),
    ("should I be worried?", "posture"),
    ("is it serious?", "posture"),
    ("what now?", "next_run"),
    ("I don't understand", "confusion"),
    ("what do you mean?", "confusion"),
    ("huh?", "confusion"),
    ("repeat that", "repeat"),
    ("say that again", "repeat"),
    ("in short", "brief"),
    ("tl;dr", "brief"),
    ("that's wrong", "correction"),
    ("that's not what I asked", "correction"),
    ("you already said that", "correction"),
    ("what is mercury agent?", "identity"),
    ("who made you?", "identity"),
    ("how do I run a detection?", "howto"),
    ("how do I use this?", "howto"),
    ("give me an example", "howto"),
    ("detect anomalies in sensor.csv", "run_command"),
    ("score this file", "run_command"),
    ("what is a hazard?", "glossary"),
    ("what is an anomaly?", "glossary"),
    ("what is a domain?", "glossary"),
    ("how many detectors are there?", "capability"),
    ("list domains", "collect"),
    ("what's the weather data source?", "collect"),
    ("do you do earthquakes?", "domain"),
    ("do you cover space weather?", "domain"),
    # Record questions that the new turns must not capture
    ("How many anomalies today?", "summary"),
    ("how many anomalies did you detect in the file?", "summary"),
    ("Summarize threat posture", "posture"),
    ("Summarize the last run", "summary"),
    ("What was the latest anomaly?", "latest"),
    ("Which samples were flagged?", "detail"),
    ("Show me the last 3 runs", "last_n"),
    ("Can you detect space weather?", "capability"),
    ("What is the anomaly score?", "glossary"),
    ("Tell me about the sepsis domain", "domain"),
)


@pytest.mark.parametrize(("question", "kind"), ROUTING)
def test_conversational_turns_route(question: str, kind: str) -> None:
    assert conv.classify(question)[0] == kind


def test_a_run_request_in_plain_words_names_the_file() -> None:
    kind, name = conv.classify("detect anomalies in sensor.csv")
    assert (kind, name) == ("run_command", "detect")
    text, _ = conv.run_command(
        "detect", {"detect": ["--input", "-d"]}, "detect anomalies in sensor.csv"
    )
    assert "mercury-agent detect --input sensor.csv" in text


def test_domain_coverage_questions_accept_a_plural() -> None:
    assert conv.classify("do you do earthquakes?") == ("domain", ("earthquake", "overview"))
    assert conv.classify("do you handle floods?") == ("domain", ("flood", "overview"))


def test_builders_offer_prompts_that_depend_on_what_was_said() -> None:
    after_record = conv.acknowledgement("latest")[1]
    after_code = conv.acknowledgement("capability")[1]
    assert after_record != after_code
    assert "Explain the latest detection" in after_record
    assert "What was the latest anomaly?" in after_code
    assert conv.farewell()[1] == []
    assert "not what you asked" in conv.correction("latest")[0]
    assert "What was the latest anomaly?" not in conv.correction("latest")[1]


def test_how_to_names_only_commands_the_cli_has() -> None:
    text, prompts = conv.how_to({"detect": ["--input"], "tier-detect": ["--input"]})
    assert "mercury-agent detect --input" in text and "tier-detect" in text
    without_tier, _ = conv.how_to({"detect": ["--input"]})
    assert "tier-detect" not in without_tier
    assert all(conv.classify(p)[0] != "search" for p in prompts)


@pytest.mark.parametrize("term", ["hazard", "anomaly", "domain", "loader"])
def test_new_glossary_terms_answer_from_this_installation(term: str) -> None:
    text, prompts = conv.glossary_answer(term, {"loader_count": 21, "manifest_size": 113})
    assert len(text) > 120
    assert all(conv.classify(p)[0] != "search" for p in prompts)


# --------------------------------------------------------------------------- #
# End to end through the voice
# --------------------------------------------------------------------------- #

pytest.importorskip("fastapi", reason="the voice end-to-end tests need the [api] extra")


@pytest.fixture
def voice(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from omni_mercury_engine.api import voice as voice_module
    from omni_mercury_engine.narrative.voice import MercuryVoice

    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    instance = voice_module._get_voice()
    if not isinstance(instance, MercuryVoice):
        pytest.skip("narrative voice unavailable")
    return instance


def _ask(voice, text: str, session: str):  # type: ignore[no-untyped-def]
    return voice.speak(text, domain="general", principal="breadth", session_id=session)


def test_say_that_again_rebuilds_the_previous_answer(voice) -> None:  # type: ignore[no-untyped-def]
    first = _ask(voice, "Which detectors do you have?", "s1")
    again = _ask(voice, "say that again", "s1")
    assert again.answer_kind == "capability"
    assert again.message == first.message


def test_in_short_shortens_the_previous_answer(voice) -> None:  # type: ignore[no-untyped-def]
    first = _ask(voice, "How do I collect data?", "s2")
    short = _ask(voice, "in short", "s2")
    assert short.answer_kind == "collect"
    assert short.message.startswith("In short: ")
    assert len(short.message) < len(first.message)


def test_a_reaction_does_not_replace_the_follow_up_context(voice) -> None:  # type: ignore[no-untyped-def]
    _ask(voice, "What is a z-score?", "s3")
    ack = _ask(voice, "ok", "s3")
    assert ack.answer_kind == "ack"
    more = _ask(voice, "tell me more", "s3")
    assert more.answer_kind == "glossary"
    assert more.message != ack.message


def test_what_do_you_mean_gives_the_fuller_version(voice) -> None:  # type: ignore[no-untyped-def]
    _ask(voice, "What is severity?", "s4")
    plainer = _ask(voice, "what do you mean?", "s4")
    assert plainer.answer_kind == "glossary"
    assert "threshold" in plainer.message.lower()


def test_a_correction_never_repeats_the_answer_it_corrects(voice) -> None:  # type: ignore[no-untyped-def]
    first = _ask(voice, "What can you do?", "s5")
    corrected = _ask(voice, "that's not what I asked", "s5")
    assert corrected.answer_kind == "correction"
    assert corrected.message != first.message
    assert (
        corrected.suggested_follow_ups and "What can you do?" not in corrected.suggested_follow_ups
    )


def test_repeat_with_nothing_said_yet_says_so(voice) -> None:  # type: ignore[no-untyped-def]
    reply = _ask(voice, "repeat that", "s6-fresh")
    assert reply.answer_kind == "repeat"
    assert "no previous answer" in reply.message


def test_no_turn_in_the_battery_ends_in_the_fallback(voice) -> None:  # type: ignore[no-untyped-def]
    for question, _kind in ROUTING:
        reply = _ask(voice, question, "s7")
        assert reply.answer_kind != "fallback", question
        assert reply.message, question
