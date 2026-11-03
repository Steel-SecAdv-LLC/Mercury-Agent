# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The speak route takes the asker's language and reports how the question was understood.

``POST /api/v1/voice/speak`` accepts ``language`` (an ISO 639-1 code the
agent supports, or ``auto``), detects the language when it is omitted,
answers social turns in that language, and carries ``language``,
``understood_as`` and ``answer_kind`` in the response metadata so the
console can show them. An unsupported code is a 422, not a silent English
answer.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omni_mercury_engine.narrative.languages import SUPPORTED


@pytest.fixture(scope="module")
def client() -> TestClient:
    from omni_mercury_engine.api.server import app

    return TestClient(app)


def test_an_explicit_language_frames_the_reply(client: TestClient) -> None:
    body = client.post("/api/v1/voice/speak", json={"message": "thanks", "language": "es"}).json()
    assert body["metadata"]["language"] == "es"
    assert body["metadata"]["answer_kind"] == "thanks"
    assert body["message"].startswith("De nada.")


def test_the_language_is_detected_when_omitted(client: TestClient) -> None:
    body = client.post("/api/v1/voice/speak", json={"message": "Was kannst du?"}).json()
    assert body["metadata"]["language"] == "de"
    assert body["metadata"]["understood_as"] == "what can you do?"
    assert body["intent"] == "help"
    assert "Englisch" in body["message"]
    assert any(p.startswith("Was war") for p in body["suggested_follow_ups"])


def test_english_stays_english(client: TestClient) -> None:
    body = client.post("/api/v1/voice/speak", json={"message": "What can you do?"}).json()
    assert body["metadata"]["language"] == "en"
    assert body["metadata"]["understood_as"] == "What can you do?"


def test_auto_and_case_are_accepted(client: TestClient) -> None:
    for value in ("auto", "ES", " fr "):
        response = client.post("/api/v1/voice/speak", json={"message": "hello", "language": value})
        assert response.status_code == 200, value


def test_an_unsupported_language_is_refused(client: TestClient) -> None:
    response = client.post("/api/v1/voice/speak", json={"message": "hello", "language": "xx"})
    assert response.status_code == 422
    assert "unsupported language" in response.text


def test_the_openapi_document_names_the_supported_languages(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    field = schema["components"]["schemas"]["SpeakRequest"]["properties"]["language"]
    for code in SUPPORTED:
        assert code in field["description"]
