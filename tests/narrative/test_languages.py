# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agent understands seven languages and frames its replies in the asker's.

These tests pin the language layer (``narrative/languages.py``): a message's
language is detected from its own words, a native question is rendered in the
English the classifier routes (and reaches the same builder as its English
twin), the social and menu replies come back in the asker's language, a
grounded answer carries a note in that language saying how the question was
understood, every translated follow-up prompt round-trips to its English
intent when clicked, and the voice carries all of it end to end with an
explicit ``language`` or a detected one.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.narrative import (
    conversation as conv,
    languages as lang,
)

NON_ENGLISH = tuple(code for code in lang.SUPPORTED if code != "en")

#: One sentence per language, with the language it must be detected as and
#: the English intent it must reach.
SENTENCES: tuple[tuple[str, str, str], ...] = (
    ("¿Qué detectaste hoy?", "es", "latest"),
    ("¿Cuántas anomalías hoy?", "es", "summary"),
    ("Hola", "es", "greeting"),
    ("Qu'est-ce que tu peux faire ?", "fr", "help"),
    ("Explique la dernière détection", "fr", "explain"),
    ("Bonjour", "fr", "greeting"),
    ("Was kannst du?", "de", "help"),
    ("Welche Detektoren hast du?", "de", "capability"),
    ("Danke", "de", "thanks"),
    ("Cosa puoi fare?", "it", "help"),
    ("Qual è l'ultima anomalia?", "it", "latest"),
    ("Grazie mille", "it", "thanks"),
    ("O que você pode fazer?", "pt", "help"),
    ("Quais detectores você tem?", "pt", "capability"),
    ("Tchau", "pt", "farewell"),
    ("Wat kun je?", "nl", "help"),
    ("Hoeveel anomalieën vandaag?", "nl", "summary"),
    ("Bedankt", "nl", "thanks"),
    ("What was the latest anomaly?", "en", "latest"),
    ("bocpd?", "en", "search"),
)


@pytest.mark.parametrize(("text", "code", "kind"), SENTENCES)
def test_language_is_detected_and_the_question_reaches_its_english_intent(
    text: str, code: str, kind: str
) -> None:
    assert lang.detect_language(text) == code
    english = lang.to_english(text, code)
    assert english is not None
    assert conv.classify(english)[0] == kind, english


@pytest.mark.parametrize("code", NON_ENGLISH)
def test_every_translated_follow_up_round_trips_to_its_english_intent(code: str) -> None:
    table = lang.follow_up_translations(code)
    assert len(table) >= 15
    for english_prompt, native in table.items():
        assert lang.detect_language(native) == code, native
        rendered = lang.to_english(native, code)
        assert rendered is not None, native
        assert conv.classify(rendered)[0] == conv.classify(english_prompt)[0], (native, rendered)


@pytest.mark.parametrize("code", NON_ENGLISH)
def test_every_frame_exists_in_every_language_and_differs_from_english(code: str) -> None:
    for key in ("greeting", "thanks", "farewell", "ack", "confusion", "repeat", "correction"):
        native = lang.frame(code, key, domain="general")
        assert native and native != lang.frame("en", key, domain="general"), (code, key)
    assert "x" in lang.frame(code, "fallback", query="x")
    assert "y" in lang.frame(code, "understood", english="y")
    assert lang.frame("en", "understood", english="y") == ""


def test_an_unknown_language_falls_back_to_english_frames() -> None:
    assert lang.frame("xx", "thanks") == lang.frame("en", "thanks")
    assert lang.to_english("hola", "xx") is None
    assert lang.localise_follow_ups("xx", ["What can you do?"]) == ["What can you do?"]


def test_a_message_the_lexicon_does_not_cover_is_not_claimed_as_understood() -> None:
    assert lang.to_english("el tiempo en Madrid", "es") is None
    assert lang.to_english("What can you do?", "en") == "What can you do?"


def test_normalise_strips_accents_and_keeps_the_spanish_marks() -> None:
    assert lang.normalise("¿Qué detectaste? Straße") == "¿que detectaste? strasse"


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


def test_a_spanish_greeting_is_answered_in_spanish(voice) -> None:  # type: ignore[no-untyped-def]
    reply = voice.speak("Hola", domain="general", principal="t", session_id="es1")
    assert reply.language == "es"
    assert reply.answer_kind == "greeting"
    assert reply.message.startswith("Hola.")
    assert "¿Qué puedes hacer?" in reply.suggested_follow_ups


def test_a_german_record_question_is_answered_in_english_with_a_german_note(voice) -> None:  # type: ignore[no-untyped-def]
    reply = voice.speak("Welche Detektoren hast du?", domain="general", principal="t")
    assert reply.language == "de"
    assert reply.understood_as == "which detectors do you have hast du?"
    assert reply.answer_kind == "capability"
    assert "verstanden" in reply.message and "Englisch" in reply.message
    assert reply.suggested_follow_ups
    assert all(lang.detect_language(p) in {"de", "en"} for p in reply.suggested_follow_ups)


def test_an_explicit_language_overrides_detection(voice) -> None:  # type: ignore[no-untyped-def]
    reply = voice.speak("thanks", domain="general", principal="t", language="fr")
    assert reply.language == "fr"
    assert reply.message == lang.frame("fr", "thanks")


def test_an_unrecognised_italian_question_gets_the_italian_menu(voice) -> None:  # type: ignore[no-untyped-def]
    reply = voice.speak("Mi piace molto il gelato al limone", domain="general", principal="t")
    assert reply.language == "it"
    assert reply.answer_kind == "fallback"
    assert reply.message.startswith("Nel registro delle rilevazioni")
    assert "gelato" in reply.message
    assert reply.understood_as is None


def test_a_clicked_translated_prompt_is_understood(voice) -> None:  # type: ignore[no-untyped-def]
    first = voice.speak("Hola", domain="general", principal="t", session_id="es2")
    for prompt in first.suggested_follow_ups:
        reply = voice.speak(prompt, domain="general", principal="t", session_id="es2")
        assert reply.language == "es", prompt
        assert reply.answer_kind not in {"fallback", None}, (prompt, reply.answer_kind)


def test_english_replies_carry_no_note_and_english_prompts(voice) -> None:  # type: ignore[no-untyped-def]
    reply = voice.speak("What can you do?", domain="general", principal="t")
    assert reply.language == "en"
    assert reply.understood_as == "What can you do?"
    assert "(" not in reply.message[-3:]
    assert reply.suggested_follow_ups == [
        "What was the latest anomaly?",
        "Which detectors do you have?",
        "What are your capabilities?",
    ]
