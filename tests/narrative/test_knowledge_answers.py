# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agent answers "tell me about <domain>" and "what is <detector>" from the knowledge profiles.

Before this layer every domain was described by the same sentence with a
different name in it, and "what is a detector" had one answer for 113
detectors. These tests pin that a question about a named domain or detector
is routed to its own profile, that every one of them gets an answer no other
gets, that the aspects (people, references, limits, data, detectors) lead
with different content, that record questions that merely mention a domain
are not hijacked, and that the voice carries the answer end to end with its
sources named.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.core.detector_ontology import ONTOLOGY
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from omni_mercury_engine.core.knowledge_profiles import DETECTOR_PROFILES, DOMAIN_PROFILES
from omni_mercury_engine.core.loader_declarations import loader_declarations
from omni_mercury_engine.detectors.hazard_diagnostics import KNOWN_HAZARDS
from omni_mercury_engine.narrative import conversation as conv

_ENTRIES = {e.name: e for e in DETECTOR_MANIFEST}

ROUTING: tuple[tuple[str, tuple[str, tuple[str, str]]], ...] = (
    ("Tell me about the sepsis domain", ("domain", ("sepsis", "overview"))),
    ("tell me about network security", ("domain", ("network_security", "overview"))),
    ("What is bocpd?", ("detector_profile", ("bocpd", "overview"))),
    ("Explain the hail detector", ("detector_profile", ("hail", "overview"))),
    ("What is the sepsis detector?", ("detector_profile", ("sepsis", "overview"))),
    ("Who publishes the earthquake data?", ("domain", ("earthquake", "people"))),
    ("Who uses spot_evt?", ("detector_profile", ("spot_evt", "people"))),
    ("What are the references for spot evt?", ("detector_profile", ("spot_evt", "references"))),
    ("What are the limits of the drought domain?", ("domain", ("drought", "limits"))),
    ("Which features does the wildfire loader produce?", ("domain", ("wildfire", "data"))),
    ("Which detectors serve space weather?", ("domain", ("space_weather", "detectors"))),
    ("What is the lineage of patchcore?", ("detector_profile", ("patchcore", "references"))),
)

NOT_PROFILE: tuple[tuple[str, str], ...] = (
    ("What was the latest anomaly in the energy domain?", "latest"),
    ("How many anomalies today in sepsis?", "summary"),
    ("What is a z-score?", "glossary"),
    ("What is AMA?", "glossary"),
    ("Which detectors do you have?", "capability"),
    ("What are your capabilities?", "capability"),
    ("How do I collect data?", "collect"),
    ("What is a detector?", "glossary"),
)


@pytest.mark.parametrize(("question", "expected"), ROUTING)
def test_profile_questions_route_to_their_subject(
    question: str, expected: tuple[str, tuple[str, str]]
) -> None:
    assert conv.classify(question) == expected


@pytest.mark.parametrize(("question", "kind"), NOT_PROFILE)
def test_record_and_glossary_questions_are_not_hijacked(question: str, kind: str) -> None:
    assert conv.classify(question)[0] == kind


def test_every_domain_name_is_recognised() -> None:
    for domain in DOMAIN_PROFILES:
        kind, (subject, _aspect) = conv.classify(
            f"Tell me about the {domain.replace('_', ' ')} domain"
        )
        assert (kind, subject) == ("domain", domain), domain


def test_every_detector_name_is_recognised() -> None:
    for name in DETECTOR_PROFILES:
        kind, (subject, _aspect) = conv.classify(f"What is the {name.replace('_', ' ')} detector?")
        assert (kind, subject) == ("detector_profile", name), name


def _domain_text(domain: str, aspect: str = "overview") -> str:
    declaration = loader_declarations().get(domain)
    from dataclasses import asdict

    text, follow_ups = conv.domain_answer(
        domain, aspect, asdict(declaration) if declaration else None, domain in KNOWN_HAZARDS
    )
    assert follow_ups and all(f for f in follow_ups)
    return text


def _detector_text(name: str, aspect: str = "overview") -> str:
    text, follow_ups = conv.detector_profile_answer(name, aspect, _ENTRIES[name], ONTOLOGY[name])
    assert follow_ups and all(f for f in follow_ups)
    return text


def test_every_domain_gets_an_answer_no_other_domain_gets() -> None:
    answers = {d: _domain_text(d) for d in DOMAIN_PROFILES}
    assert len(set(answers.values())) == len(answers)
    for domain, text in answers.items():
        assert len(text) > 200, domain
        profile = DOMAIN_PROFILES[domain]
        assert profile.institutions[0].name in text, domain
        assert profile.discipline in text, domain


def test_every_detector_gets_an_answer_no_other_detector_gets() -> None:
    answers = {n: _detector_text(n) for n in DETECTOR_PROFILES}
    assert len(set(answers.values())) == len(answers)
    for name, text in answers.items():
        assert len(text) > 200, name
        assert _ENTRIES[name].class_name in text, name
        assert DETECTOR_PROFILES[name].discipline.lower() in text.lower(), name


@pytest.mark.parametrize("aspect", ["people", "references", "limits", "data", "detectors"])
def test_domain_aspects_lead_with_different_content(aspect: str) -> None:
    overview = _domain_text("sepsis")
    focused = _domain_text("sepsis", aspect)
    assert focused != overview
    if aspect == "people":
        assert "intensivists" in focused
    if aspect == "references":
        assert "Singer" in focused
    if aspect == "limits":
        assert "MIMIC-III" in focused
    if aspect == "data":
        assert "HR" in focused and "ground truth" in focused.lower()
    if aspect == "detectors":
        assert "sepsis (" in focused


@pytest.mark.parametrize("aspect", ["people", "references", "limits"])
def test_detector_aspects_lead_with_different_content(aspect: str) -> None:
    overview = _detector_text("bocpd")
    focused = _detector_text("bocpd", aspect)
    assert focused != overview
    if aspect == "references":
        assert "Adams" in focused and "0710.3742" in focused
    if aspect == "limits":
        assert "point" in focused.lower()
    if aspect == "people":
        assert "Statisticians" in focused


def test_keyed_loader_and_hazard_domain_are_stated() -> None:
    assert "NASA_FIRMS_MAP_KEY" in _domain_text("wildfire")
    schumann = _domain_text("schumann")
    assert "hazard-viz -d schumann" in schumann


def test_adapter_detector_names_what_it_serves() -> None:
    assert "SepsisDetector" in _detector_text("sepsis")


# --------------------------------------------------------------------------- #
# Through the voice
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


def test_voice_describes_a_domain_from_its_profile(voice) -> None:  # type: ignore[no-untyped-def]
    response = voice.speak("Tell me about the sepsis domain", domain="general", principal="test")
    assert "PhysioNet" in response.message
    assert "Critical-care" in response.message
    assert "knowledge_profiles" in response.sources_cited
    assert "loader_declarations" in response.sources_cited
    assert response.suggested_follow_ups


def test_voice_describes_a_detector_from_its_profile(voice) -> None:  # type: ignore[no-untyped-def]
    response = voice.speak("What is bocpd?", domain="general", principal="test")
    assert "Adams" in response.message and "run length" in response.message
    assert set(response.sources_cited) >= {
        "knowledge_profiles",
        "detector_manifest",
        "detector_ontology",
    }


def test_voice_follow_ups_from_a_profile_answer_are_answered(voice) -> None:  # type: ignore[no-untyped-def]
    first = voice.speak("Tell me about the earthquake domain", domain="general", principal="test")
    for prompt in first.suggested_follow_ups:
        kind = conv.classify(prompt)[0]
        assert kind in {"domain", "detector_profile"}, prompt
        reply = voice.speak(prompt, domain="general", principal="test")
        assert reply.message and "nothing in the detection record" not in reply.message, prompt
