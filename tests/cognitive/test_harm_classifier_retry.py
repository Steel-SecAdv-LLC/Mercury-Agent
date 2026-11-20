# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A failed meaning-level classifier import must not weaken the gate forever.

``_resolve_harm_classifier`` cached ``None`` permanently the first time the
shipped classifier failed to import, and said so only at INFO -- one transient
failure (a cold filesystem, a half-installed wheel) silently downgraded every
later decision in the process to lexical-only coverage.  These tests pin the
restored behaviour: the failure is logged at WARNING with the exception, and
the import is retried on a bounded backoff so a transient failure heals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from omni_mercury_engine.cognitive import ethical_bounding
from omni_mercury_engine.cognitive.ethical_bounding import (
    USE_SHIPPED_CLASSIFIER,
    _resolve_harm_classifier,
)


@pytest.fixture(autouse=True)
def _pristine_classifier_cache() -> Iterator[None]:
    """Isolate and restore the module-level classifier cache around each test."""
    saved = list(ethical_bounding._SHIPPED_CLASSIFIER_CACHE)
    ethical_bounding._SHIPPED_CLASSIFIER_CACHE.clear()
    saved_calls = getattr(ethical_bounding, "_shipped_classifier_failed_calls", 0)
    try:
        yield
    finally:
        ethical_bounding._SHIPPED_CLASSIFIER_CACHE.clear()
        ethical_bounding._SHIPPED_CLASSIFIER_CACHE.extend(saved)
        if hasattr(ethical_bounding, "_shipped_classifier_failed_calls"):
            ethical_bounding._shipped_classifier_failed_calls = saved_calls


def _break_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from omni_mercury_engine.cognitive import meaning_level

    def _boom() -> object:
        raise OSError("simulated transient weight-load failure")

    monkeypatch.setattr(meaning_level, "meaning_level_harm_classifier", _boom)


def _heal_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from omni_mercury_engine.cognitive import meaning_level

    def _classifier() -> object:
        return lambda text: 0.0

    monkeypatch.setattr(meaning_level, "meaning_level_harm_classifier", _classifier)


def test_failed_import_logs_at_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _break_import(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=ethical_bounding.logger.name):
        assert _resolve_harm_classifier(USE_SHIPPED_CLASSIFIER) is None
    assert any(
        rec.levelno >= logging.WARNING and "classifier" in rec.message.lower()
        for rec in caplog.records
    ), "a degraded-to-lexical-only gate must be visible at WARNING, not INFO"


def test_transient_failure_heals_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _break_import(monkeypatch)
    assert _resolve_harm_classifier(USE_SHIPPED_CLASSIFIER) is None

    # The failure clears; within a bounded number of later calls the shipped
    # classifier must be active again (the old code cached None forever).
    _heal_import(monkeypatch)
    resolved = [_resolve_harm_classifier(USE_SHIPPED_CLASSIFIER) for _ in range(64)]
    assert any(
        c is not None for c in resolved
    ), "one failed import permanently downgraded the gate to lexical-only"
    # Once healed, it stays healed.
    assert _resolve_harm_classifier(USE_SHIPPED_CLASSIFIER) is not None


def test_persistent_failure_stays_lexical_only_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _break_import(monkeypatch)
    for _ in range(64):
        assert _resolve_harm_classifier(USE_SHIPPED_CLASSIFIER) is None


def test_explicit_classifier_arguments_pass_through_unchanged() -> None:
    sentinel = object()
    assert _resolve_harm_classifier(sentinel) is sentinel
    assert _resolve_harm_classifier(None) is None
