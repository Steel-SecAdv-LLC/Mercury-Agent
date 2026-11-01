# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agent remembers what the engine detects.

``KnowledgeRetriever`` has carried a ``log_detection`` hook and a
detection-log search path since it was written, and ``record_detection`` has
recorded every API detection into the export ``DataStore`` -- but nothing
ever connected the two, so "have there been any anomalies?" was answered
from an empty log on every deployment while the voice route's documentation
advertised a ``detection`` intent. ``record_detection`` now mirrors each
record into the live voice's retriever; these tests pin that the mirror
carries the fields the retriever searches on, and that the voice then finds
the event instead of apologising.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from omni_mercury_engine.api import voice as voice_module
from omni_mercury_engine.api.routes.export import record_detection


@pytest.fixture
def fresh_voice(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A voice built for this test only, so earlier tests' memory cannot leak in."""
    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    voice = voice_module._get_voice()
    if not hasattr(voice, "retriever"):
        pytest.skip("narrative voice unavailable; fallback voice has no retriever")
    return voice


def _record(anomalies: list[bool], scores: list[float], domain: str = "network_security") -> None:
    asyncio.run(
        record_detection(
            user_id="test",
            method="multivariate",
            data=[[float(i)] for i in range(len(anomalies))],
            results={"anomalies": anomalies, "scores": scores, "summary": {"domain": domain}},
            sensitivity=0.5,
            metadata={"domain": domain},
        )
    )


def test_recorded_detection_lands_in_the_retriever_with_searchable_fields(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    before = len(fresh_voice.retriever._detection_log)
    _record([False, True, False], [0.2, 8.1, 0.3])
    log = fresh_voice.retriever._detection_log
    assert len(log) == before + 1
    entry = log[-1]
    # Stored through the voice's own domain whitelist: "network_security" is
    # not an EnvironmentDomain, so it is filed where a query will look for it.
    assert entry["domain"] == "general"
    det = entry["detection"]
    assert det["anomaly_detected"] is True
    assert det["anomaly_count"] == 1
    assert det["anomaly_score"] == pytest.approx(8.1)
    assert det["method"] == "multivariate"


def test_voice_now_finds_the_anomaly_instead_of_apologising(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    _record([False, True, False, False], [0.1, 7.4, 0.2, 0.1])
    response = fresh_voice.speak("What anomaly detection events happened recently?")
    message = getattr(response, "message", None) or response.get("message")
    count = getattr(response, "search_results_count", None)
    if count is None and isinstance(response, dict):
        count = response.get("search_results_count")
    assert "couldn't find" not in message.lower()
    assert count is None or count >= 1
    assert "event" in message.lower() or "anomal" in message.lower()


def test_normal_run_is_remembered_but_not_as_an_anomaly(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    _record([False, False], [0.1, 0.2], domain="general")
    det = fresh_voice.retriever._detection_log[-1]["detection"]
    assert det["anomaly_detected"] is False
    assert det["anomaly_count"] == 0


def _record_with_threshold(score: float, threshold: float) -> None:
    asyncio.run(
        record_detection(
            user_id="test",
            method="univariate",
            data=[[1.0], [score], [1.0]],
            results={
                "anomalies": [False, True, False],
                "scores": [0.1, score, 0.1],
                "threshold": threshold,
            },
            sensitivity=0.5,
            metadata=None,
        )
    )


def test_severity_is_bounded_and_banded_not_the_raw_score(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    """A z-score of 9 used to be stored as severity 9.0 and spoken as "900%"."""
    _record_with_threshold(9.0, 3.5)
    det = fresh_voice.retriever._detection_log[-1]["detection"]
    assert det["anomaly_score"] == pytest.approx(9.0)
    assert 0.0 <= det["severity"] <= 1.0
    assert det["severity_label"] == "high"  # 9 / 3.5 = 2.57x: the API's high band [2.5, 4)
    assert "confidence" not in det  # these detectors emit none; no constant stands in


def test_severity_bands_match_the_api(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    from omni_mercury_engine.api.routes.export import _severity_fraction
    from omni_mercury_engine.api.server import _classify_severity

    for ratio in (0.5, 1.49, 1.5, 2.49, 2.5, 3.99, 4.0, 10.0):
        frac, label = _severity_fraction(ratio * 2.0, 2.0)
        assert label == _classify_severity(ratio * 2.0, 2.0).value
        assert frac == pytest.approx(min(1.0, ratio / 4.0))
    assert _severity_fraction(5.0, None) == (None, None)


def test_spoken_reply_names_the_newest_detection_and_no_percent_over_100(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    _record_with_threshold(4.0, 4.0)  # older: ratio 1.0, low
    _record_with_threshold(20.0, 2.0)  # newest: ratio 10, critical
    reply = fresh_voice.speak("What anomaly detection events happened recently?")
    message = getattr(reply, "message", None) or reply.get("message")
    assert "anomaly_score=20.00" in message, message
    assert "critical" in message
    percents = [int(p) for p in re.findall(r"(\d+)%", message)]
    assert all(p <= 100 for p in percents), message


def test_plural_question_finds_the_recorded_anomaly(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    """ "Have there been any anomalies recently?" is the console's wording.

    Reproduced against ``mercury-agent serve``: after a detection with one
    flagged sample, this exact question answered "I couldn't find specific
    information" with ``search_results_count`` 0, because the detection-log
    scorer looked for the literal token ``anomaly`` (``anomalies`` does not
    contain it) and the intent table had no entry for the word at all.
    """
    _record([False, True, False], [0.2, 8.1, 0.1], domain="general")
    for question in (
        "Have there been any anomalies recently?",
        "anything flagged today?",
        "Were there any alerts lately?",
    ):
        response = fresh_voice.speak(question, domain="general")
        message = response.message if hasattr(response, "message") else response["message"]
        count = getattr(response, "search_results_count", None)
        assert "couldn't find" not in message.lower(), (question, message)
        assert count is None or count >= 1, (question, count)


def test_console_domain_does_not_hide_a_domainless_detection(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    """A detection recorded without a domain is visible from any domain.

    ``/detect/univariate`` and ``/detect/multivariate`` carry no domain, so
    the mirror stores them under ``general``; the console asks under its
    CONTEXT domain (``network_security`` by default). The log search filtered
    on strict equality, so the console could never see a detection it had
    just run. ``general`` is the collapse of "unknown", not a peer domain, and
    is treated as a wildcard on both sides of the comparison.
    """
    _record([False, True], [0.2, 8.1], domain="general")
    response = fresh_voice.speak(
        "What anomaly detection events happened recently?", domain="network_security"
    )
    message = response.message if hasattr(response, "message") else response["message"]
    assert "couldn't find" not in message.lower(), message
    # And a domain-specific record is still hidden from an unrelated domain,
    # while a ``general`` question sees everything.
    _record([False, True], [0.2, 7.0], domain="medical")
    assert fresh_voice.retriever.search("anomaly events", domain="financial").total_found == 1
    assert fresh_voice.retriever.search("anomaly events", domain="general").total_found == 2
