# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Safety kill switches cannot disable protection in production.

Two environment variables turn off a safety instrument:
``MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER`` (the meaning-level harm
classifier) and ``MERCURY_GATE_AUDIT_DISABLED`` (the durable gate-decision
audit trail).  They used to be honoured everywhere, so in production one
environment variable was a silent safety bypass.  They are now honoured only
in development; in production they are ignored, with an ERROR that names the
switch.

These tests prove the protection keeps operating, not merely that a flag
reads False.  The classifier is driven through a FIXED non-zero sentinel
score on a neutral input — so the assertion is non-vacuous (a disabled
classifier returns 0.0 for every input) without this suite authoring any
harmful text.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from omni_mercury_engine import _env
from omni_mercury_engine.cognitive import gate_audit, harm_classifier

_SENTINEL = 0.9
_NEUTRAL = "summarise the attached weather observations"
_HARM_SWITCH = "MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER"
_AUDIT_SWITCH = "MERCURY_GATE_AUDIT_DISABLED"


@pytest.fixture
def sentinel_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the shipped classifier to a fixed score, with no served backend."""
    monkeypatch.setattr(harm_classifier, "_shipped_meaning_level_score", lambda _t: _SENTINEL)
    monkeypatch.setattr(harm_classifier, "_resolve_default_backend", lambda: None)


@pytest.fixture(autouse=True)
def fresh_warning_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard warns once per switch per process; give each test a fresh slate."""
    monkeypatch.setattr(_env, "_KILLSWITCH_WARNED", set())


class TestHarmClassifierSwitch:
    def test_development_honours_the_switch(
        self, monkeypatch: pytest.MonkeyPatch, sentinel_classifier: None
    ) -> None:
        monkeypatch.setenv("MERCURY_ENV", "development")
        monkeypatch.setenv(_HARM_SWITCH, "1")
        assert harm_classifier.default_harm_classifier()(_NEUTRAL) == 0.0

    def test_production_keeps_the_classifier_scoring(
        self, monkeypatch: pytest.MonkeyPatch, sentinel_classifier: None
    ) -> None:
        """The switch set in production does NOT zero the classifier."""
        monkeypatch.setenv("MERCURY_ENV", "production")
        monkeypatch.setenv(_HARM_SWITCH, "1")
        assert harm_classifier.default_harm_classifier()(_NEUTRAL) == _SENTINEL

    def test_production_switch_equals_no_switch(
        self, monkeypatch: pytest.MonkeyPatch, sentinel_classifier: None
    ) -> None:
        monkeypatch.setenv("MERCURY_ENV", "production")
        monkeypatch.delenv(_HARM_SWITCH, raising=False)
        without = harm_classifier.default_harm_classifier()(_NEUTRAL)
        monkeypatch.setenv(_HARM_SWITCH, "1")
        with_switch = harm_classifier.default_harm_classifier()(_NEUTRAL)
        assert with_switch == without == _SENTINEL

    def test_meaning_level_coverage_is_not_withdrawn_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.cognitive.meaning_level import meaning_level_disabled

        monkeypatch.setenv(_HARM_SWITCH, "1")
        monkeypatch.setenv("MERCURY_ENV", "development")
        assert meaning_level_disabled() is True
        monkeypatch.setenv("MERCURY_ENV", "production")
        assert meaning_level_disabled() is False


class TestGateAuditSwitch:
    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        monkeypatch.setattr(gate_audit, "_write_jsonl", records.append)
        monkeypatch.setattr(gate_audit, "_forward_secure", lambda _r: None)
        return records

    def test_development_honours_the_switch(
        self, monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]
    ) -> None:
        monkeypatch.setenv("MERCURY_ENV", "development")
        monkeypatch.setenv(_AUDIT_SWITCH, "1")
        gate_audit.record_gate_decision(decision="refused", source="test", disposition="test")
        assert captured == []

    def test_production_still_writes_the_audit_record(
        self, monkeypatch: pytest.MonkeyPatch, captured: list[dict[str, Any]]
    ) -> None:
        """A refusal in production is recorded even with the switch set."""
        monkeypatch.setenv("MERCURY_ENV", "production")
        monkeypatch.setenv(_AUDIT_SWITCH, "1")
        gate_audit.record_gate_decision(decision="refused", source="test", disposition="test")
        assert len(captured) == 1
        assert captured[0]["decision"] == "refused"


class TestTheErrorNamesTheSwitch:
    @pytest.mark.parametrize("switch", [_HARM_SWITCH, _AUDIT_SWITCH])
    def test_error_names_the_switch_and_says_it_was_ignored(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, switch: str
    ) -> None:
        monkeypatch.setenv("MERCURY_ENV", "production")
        monkeypatch.setenv(switch, "1")
        with caplog.at_level(logging.ERROR, logger="omni_mercury_engine._env"):
            assert _env.killswitch_active(switch) is False
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        message = errors[0].getMessage()
        assert switch in message
        assert "IGNORED" in message

    def test_the_error_is_logged_once_not_per_call(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MERCURY_ENV", "production")
        monkeypatch.setenv(_HARM_SWITCH, "1")
        with caplog.at_level(logging.ERROR, logger="omni_mercury_engine._env"):
            for _ in range(5):
                _env.killswitch_active(_HARM_SWITCH)
        assert sum(r.levelno == logging.ERROR for r in caplog.records) == 1

    def test_unreadable_env_mode_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo'd MERCURY_ENV is treated as production: the switch is ignored."""
        monkeypatch.setenv("MERCURY_ENV", "prod")  # not a recognised mode
        monkeypatch.setenv(_HARM_SWITCH, "1")
        assert _env.killswitch_active(_HARM_SWITCH) is False
