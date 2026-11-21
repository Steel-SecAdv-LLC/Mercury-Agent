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
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import FrameType

from omni_mercury_engine.cognitive import ethical_bounding
from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    USE_SHIPPED_CLASSIFIER,
    BenevolenceScorer,
    HarmReducer,
    _resolve_harm_classifier,
)


@pytest.fixture(autouse=True)
def _pristine_classifier_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start each test with the import not yet tried; restore the module after."""
    monkeypatch.setattr(
        ethical_bounding, "_shipped_classifier", ethical_bounding._SHIPPED_CLASSIFIER_NOT_TRIED
    )
    monkeypatch.setattr(ethical_bounding, "_shipped_classifier_failed_calls", 0)
    yield


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


# ---------------------------------------------------------------------------
# The retry has to reach the objects that live for the whole process.
#
# ``HarmReducer.__init__`` resolved the classifier once and stored the result.
# Built while the import was failing it stored ``None``, and then handed that
# explicit ``None`` to ``assess_weapons_uplift`` on every call -- which reads
# an explicit ``None`` as "the caller asked for lexical-only".  The bounded
# retry above was therefore never consulted for exactly the long-lived
# scorers it was added for (the orchestrator's ``_benevolence_scorer`` is one).
# ---------------------------------------------------------------------------

_BENIGN = "tidy the garden shed and label the seed trays"
_HEALED_SCORE = 0.97


def _heal_import_scoring(monkeypatch: pytest.MonkeyPatch, score: float = _HEALED_SCORE) -> None:
    """The import works again, and the classifier it yields is recognisable."""
    from omni_mercury_engine.cognitive import meaning_level

    def _classifier() -> object:
        return lambda text: score

    monkeypatch.setattr(meaning_level, "meaning_level_harm_classifier", _classifier)


def _count_failing_imports(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Break the import and count how often it is attempted."""
    from omni_mercury_engine.cognitive import meaning_level

    attempts = [0]

    def _boom() -> object:
        attempts[0] += 1
        raise OSError("simulated transient weight-load failure")

    monkeypatch.setattr(meaning_level, "meaning_level_harm_classifier", _boom)
    return attempts


def _build_harm_reducer() -> HarmReducer:
    return HarmReducer()


def _build_benevolence_scorer_reducer() -> HarmReducer:
    # What ``CognitiveOrchestrator`` holds for its lifetime.
    return BenevolenceScorer(benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR).harm_reducer


@pytest.mark.parametrize(
    "build", [_build_harm_reducer, _build_benevolence_scorer_reducer], ids=["reducer", "scorer"]
)
def test_scorer_built_during_a_failed_import_uses_the_classifier_once_it_heals(
    monkeypatch: pytest.MonkeyPatch, build: Callable[[], HarmReducer]
) -> None:
    _break_import(monkeypatch)
    reducer = build()
    lexical_only, _ = reducer.evaluate_harm(_BENIGN, {})
    assert lexical_only < _HEALED_SCORE

    _heal_import_scoring(monkeypatch)
    harms = [reducer.evaluate_harm(_BENIGN, {})[0] for _ in range(64)]

    assert max(harms) >= _HEALED_SCORE, (
        "a scorer constructed while the classifier import was failing stayed "
        "lexical-only after the import healed: the retry never reached it"
    )
    # Once healed it stays healed, on this same long-lived object.
    assert reducer.evaluate_harm(_BENIGN, {})[0] >= _HEALED_SCORE


def test_explicit_none_keeps_a_long_lived_scorer_lexical_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``None`` is still a request, not a failure: it is never retried into a model."""
    _heal_import_scoring(monkeypatch)
    reducer = HarmReducer(harm_classifier=None)
    harms = [reducer.evaluate_harm(_BENIGN, {})[0] for _ in range(64)]
    assert max(harms) < _HEALED_SCORE


def test_resolver_backs_off_between_import_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = _count_failing_imports(monkeypatch)
    calls = 4 * ethical_bounding._SHIPPED_CLASSIFIER_RETRY_EVERY
    for _ in range(calls):
        assert _resolve_harm_classifier(USE_SHIPPED_CLASSIFIER) is None
    # The first call, then one retry per window: never one import per call.
    assert 2 <= attempts[0] <= 1 + calls // ethical_bounding._SHIPPED_CLASSIFIER_RETRY_EVERY


def test_long_lived_scorer_backs_off_between_import_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _count_failing_imports(monkeypatch)
    reducer = HarmReducer()
    evaluations = 4 * ethical_bounding._SHIPPED_CLASSIFIER_RETRY_EVERY
    for _ in range(evaluations):
        reducer.evaluate_harm(_BENIGN, {})
    assert attempts[0] >= 2, "the failing import was never retried from a long-lived scorer"
    assert attempts[0] <= 2 + evaluations // ethical_bounding._SHIPPED_CLASSIFIER_RETRY_EVERY, (
        f"{attempts[0]} import attempts in {evaluations} evaluations: the retry is not "
        "backing off"
    )


class _SecondCaller:
    """A second thread calling the resolver, advanced one step at a time.

    One step lets it finish the call it is paused in and run the next one as
    far as the classifier import, where it pauses again.  A caller that takes a
    step at each of its own line boundaries therefore meets every interleaving
    of "the other thread finished a retry and began another" with its own
    progress, with no timing involved.  The pause is bounded: under a resolver
    that holds a lock across the import, the paused thread is holding it.
    """

    PAUSE_S = 0.05
    STEP_S = 0.05

    def __init__(self) -> None:
        self.errors: list[BaseException] = []
        self._go = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._paused = threading.Event()
        self._resume = threading.Event()
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            self._go.wait()
            self._go.clear()
            if self._stop:
                return
            try:
                _resolve_harm_classifier(USE_SHIPPED_CLASSIFIER)
            except BaseException as exc:  # pragma: no cover - surfaced by the test
                self.errors.append(exc)
            self._idle.set()

    def pause_in_import(self) -> None:
        """Called from the (failing) import, on this thread only."""
        self._paused.set()
        self._resume.wait(self.PAUSE_S)
        self._resume.clear()
        self._paused.clear()

    def step(self) -> None:
        if self._paused.is_set():
            self._resume.set()
            self._idle.wait(self.STEP_S)
        if self._idle.is_set():
            self._idle.clear()
            self._go.set()
            deadline = time.monotonic() + self.STEP_S
            while time.monotonic() < deadline:
                if self._paused.is_set() or self._idle.is_set():
                    break
                time.sleep(0.0005)

    def finish(self) -> None:
        self._idle.wait(5)
        self._stop = True
        self._resume.set()
        self._go.set()
        self.thread.join(5)


def test_a_retry_in_one_thread_does_not_pull_the_cache_from_under_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache was a list one thread ``clear()``-ed while another indexed ``[0]``.

    Between a caller's ``append`` of its result and its ``return cache[0]``, a
    second caller starting a retry emptied the list: ``IndexError`` out of the
    harm gate.  Every call retries here (a window of one), and the second
    caller is stepped at each line boundary of the first.
    """
    from omni_mercury_engine.cognitive import meaning_level

    monkeypatch.setattr(ethical_bounding, "_SHIPPED_CLASSIFIER_RETRY_EVERY", 1)
    monkeypatch.setattr(ethical_bounding.logger, "disabled", True)
    second = _SecondCaller()

    def _boom() -> object:
        if threading.current_thread() is second.thread:
            second.pause_in_import()
        raise OSError("simulated transient weight-load failure")

    monkeypatch.setattr(meaning_level, "meaning_level_harm_classifier", _boom)
    module_file = ethical_bounding.__file__
    outcome: dict[str, object] = {}

    def tracer(frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename != module_file:
            return None
        if event == "line":
            second.step()
        return tracer

    def first_caller() -> None:
        previous = sys.gettrace()
        sys.settrace(tracer)
        try:
            outcome["value"] = _resolve_harm_classifier(USE_SHIPPED_CLASSIFIER)
        except BaseException as exc:  # pragma: no cover - surfaced below
            outcome["error"] = exc
        finally:
            sys.settrace(previous)

    first = threading.Thread(target=first_caller, daemon=True)
    first.start()
    first.join(60)
    second.finish()

    assert not first.is_alive(), "resolver deadlocked"
    assert not second.thread.is_alive(), "second resolver call never returned"
    assert "error" not in outcome, f"resolver raised under a concurrent retry: {outcome!r}"
    assert not second.errors, f"concurrent resolver call raised: {second.errors!r}"
    assert outcome["value"] is None


def test_many_threads_resolving_while_the_import_fails_do_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same race, unscheduled: dense retries from several threads."""
    _break_import(monkeypatch)
    monkeypatch.setattr(ethical_bounding, "_SHIPPED_CLASSIFIER_RETRY_EVERY", 1)
    monkeypatch.setattr(ethical_bounding.logger, "disabled", True)
    errors: list[BaseException] = []
    n_threads = 8
    start = threading.Barrier(n_threads)

    def worker() -> None:
        start.wait(timeout=30)
        try:
            for _ in range(3000):
                _resolve_harm_classifier(USE_SHIPPED_CLASSIFIER)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
    finally:
        sys.setswitchinterval(interval)
    assert not any(thread.is_alive() for thread in threads), "resolver deadlocked"
    assert not errors, f"{len(errors)} resolver call(s) raised: {errors[:3]!r}"
