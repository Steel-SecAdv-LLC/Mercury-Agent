# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A stream session must not mutate the engine's shared detectors.

``EngineStreamSession`` fitted ``engine.detectors[name]`` in place at warm-up
and every refit. That corrupted a trained engine, let two concurrent sessions
overwrite each other's fits, and -- because those in-place fits were not
recorded as inference auto-fits -- made ``engine._fit_for_training`` later skip
refitting a detector it believed was already trained. The session must operate
on its own detector instances; the engine's trained state must be identical
before and after a session.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture()
def trained_engine() -> Any:
    from omni_mercury_engine.engine import OmniMercuryEngine

    engine = OmniMercuryEngine(require_explicit_fit=False)
    rng = np.random.default_rng(101)
    engine.detectors["statistical"].fit(rng.standard_normal((200, 1)))
    return engine


def _probe_scores(engine: Any) -> np.ndarray:
    """Deterministic detector output -- a fingerprint of the fitted state."""
    probe = np.linspace(-3.0, 3.0, 40).reshape(-1, 1)
    return np.asarray(engine.detectors["statistical"].detect(probe)["scores"], dtype=np.float64)


def test_engine_detector_state_unchanged_after_session(trained_engine: Any) -> None:
    """Running a stream session must leave the engine's detector untouched.

    Fails today: the session fits ``engine.detectors['statistical']`` in place
    on its rolling window, so the probe fingerprint changes.
    """
    before = _probe_scores(trained_engine)

    session = trained_engine.stream_session(
        detector_types=["statistical"], min_samples=16, window_size=64, refit_interval=8
    )
    rng = np.random.default_rng(5)
    for _ in range(40):
        session.observe(float(rng.standard_normal() * 0.1))
    session.observe(25.0)  # a spike, to exercise scoring

    after = _probe_scores(trained_engine)
    assert np.array_equal(
        before, after
    ), "stream session mutated the engine's shared detector fitted state"


def test_two_sessions_use_independent_detectors(trained_engine: Any) -> None:
    """Concurrent sessions must hold distinct detector instances.

    Fails today: both sessions fit and score the very same
    ``engine.detectors['statistical']`` object, so one session's refit
    overwrites the other's.
    """
    session_a = trained_engine.stream_session(
        detector_types=["statistical"], min_samples=8, window_size=32
    )
    session_b = trained_engine.stream_session(
        detector_types=["statistical"], min_samples=8, window_size=32
    )

    shared = trained_engine.detectors["statistical"]
    det_a = session_a._detectors["statistical"]
    det_b = session_b._detectors["statistical"]
    assert det_a is not shared, "session A reuses the engine's shared detector"
    assert det_b is not shared, "session B reuses the engine's shared detector"
    assert det_a is not det_b, "the two sessions share one detector instance"

    # Interleaved use: A sees a spike, B stays calm; neither corrupts the other.
    rng = np.random.default_rng(9)
    verdict_a = verdict_b = None
    for i in range(40):
        verdict_a = session_a.observe(25.0 if i == 39 else float(rng.standard_normal() * 0.1))
        verdict_b = session_b.observe(float(rng.standard_normal() * 0.1))
    assert verdict_a is not None and verdict_b is not None
    assert verdict_a["per_detector"]["statistical"]["score"] > 0.5
    assert verdict_b["per_detector"]["statistical"]["score"] < 0.5


def test_engine_detectors_are_never_swapped_while_a_session_scores(trained_engine: Any) -> None:
    """At every point inside ``observe`` the engine holds its own detector.

    The session used to write its copy into ``engine.detectors[name]`` for the
    duration of ``engine.detect`` and restore the original afterwards, so any
    caller of the engine during that window was served the session's detector.
    The copy now reaches ``detect`` as a per-call override and the engine's
    mapping is never written.
    """
    original = trained_engine.detectors["statistical"]
    session = trained_engine.stream_session(
        detector_types=["statistical"], min_samples=8, window_size=32, refit_interval=6
    )
    copy_ = session._detectors["statistical"]
    seen: list[Any] = []
    real_detect, real_fit = copy_.detect, copy_.fit

    def watching_detect(data: Any) -> Any:
        seen.append(trained_engine.detectors["statistical"])
        return real_detect(data)

    def watching_fit(data: Any) -> Any:
        seen.append(trained_engine.detectors["statistical"])
        return real_fit(data)

    copy_.detect = watching_detect
    copy_.fit = watching_fit
    rng = np.random.default_rng(13)
    for _ in range(40):
        session.observe(float(rng.standard_normal() * 0.1))
        assert trained_engine.detectors["statistical"] is original
    assert len(seen) > 30, "the session copy was not exercised"
    assert all(
        detector is original for detector in seen
    ), "engine.detectors held a session copy while the session was scoring"
    assert not hasattr(session, "_engine_sees_session_detectors")


def test_detector_overrides_leave_the_engine_bookkeeping_alone(trained_engine: Any) -> None:
    """An unfitted override is fitted for the caller, not recorded on the engine."""
    import copy

    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    probe = np.random.default_rng(3).standard_normal((40, 1))
    plain = trained_engine.detect(probe, detector_types=["statistical"])
    override = MercuryAnomalyDetector()
    assert not override.is_fitted()
    with_override = trained_engine.detect(
        probe, detector_types=["statistical"], detector_overrides={"statistical": override}
    )
    assert override.is_fitted()
    assert "statistical" not in trained_engine._inference_auto_fit_detectors
    assert "inference_auto_fit_detectors" not in with_override
    again = trained_engine.detect(probe, detector_types=["statistical"])
    np.testing.assert_array_equal(
        np.asarray(plain["detectors"]["statistical"]["scores"]),
        np.asarray(again["detectors"]["statistical"]["scores"]),
    )
    # The override's own fitted state is what scored the overridden call.
    fitted_copy = copy.deepcopy(trained_engine.detectors["statistical"])
    same = trained_engine.detect(
        probe, detector_types=["statistical"], detector_overrides={"statistical": fitted_copy}
    )
    np.testing.assert_array_equal(
        np.asarray(plain["detectors"]["statistical"]["scores"]),
        np.asarray(same["detectors"]["statistical"]["scores"]),
    )


def test_concurrent_sessions_and_callers_do_not_corrupt_the_engine(trained_engine: Any) -> None:
    """Two sessions on two threads plus a plain ``detect`` caller on a third.

    With the swap, a concurrent ``detect`` could be served a session's
    detector, and two overlapping sessions could each capture the other's
    copy as the "original" and restore it for good.
    """
    import threading

    originals = dict(trained_engine.detectors)
    fingerprint = _probe_scores(trained_engine)
    probe = np.linspace(-3.0, 3.0, 40).reshape(-1, 1)
    expected = np.asarray(
        trained_engine.detect(probe, detector_types=["statistical"])["detectors"]["statistical"][
            "scores"
        ],
        dtype=np.float64,
    )
    errors: list[BaseException] = []
    mismatches: list[int] = []
    stop = threading.Event()
    ticks = 300

    def run_session(seed: int, scale: float) -> None:
        try:
            session = trained_engine.stream_session(
                detector_types=["statistical"], min_samples=8, window_size=32, refit_interval=5
            )
            rng = np.random.default_rng(seed)
            for _ in range(ticks):
                session.observe(float(rng.standard_normal() * scale + scale))
        except BaseException as exc:
            errors.append(exc)

    def run_plain_detect() -> None:
        try:
            calls = 0
            while not stop.is_set() or calls < 20:
                got = trained_engine.detect(probe, detector_types=["statistical"])
                scores = np.asarray(got["detectors"]["statistical"]["scores"], dtype=np.float64)
                if not np.array_equal(scores, expected):
                    mismatches.append(calls)
                calls += 1
        except BaseException as exc:
            errors.append(exc)

    sessions = [
        threading.Thread(target=run_session, args=(1, 0.1)),
        threading.Thread(target=run_session, args=(2, 50.0)),
    ]
    caller = threading.Thread(target=run_plain_detect)
    caller.start()
    for thread in sessions:
        thread.start()
    for thread in sessions:
        thread.join()
    stop.set()
    caller.join()

    assert not errors, errors
    assert not mismatches, f"{len(mismatches)} plain detect calls were served another detector"
    assert set(trained_engine.detectors) == set(originals)
    for name, detector in originals.items():
        assert trained_engine.detectors[name] is detector, f"{name} was replaced"
    np.testing.assert_array_equal(_probe_scores(trained_engine), fingerprint)


def test_a_session_over_the_engines_default_detectors_scores() -> None:
    """The default session takes every engine detector, and each must be copyable.

    ``directive`` holds a ``threading.local``; deep-copying it raised
    ``TypeError``, so ``engine.stream_session()`` with no ``detector_types``
    could not be constructed at all.
    """
    from omni_mercury_engine.engine import OmniMercuryEngine

    engine = OmniMercuryEngine(require_explicit_fit=False)
    originals = dict(engine.detectors)
    session = engine.stream_session(min_samples=16, window_size=48, refit_interval=16)
    rng = np.random.default_rng(11)
    verdict: dict[str, Any] = {}
    for _ in range(40):
        verdict = session.observe(rng.standard_normal(3))

    assert verdict["status"] == "ok"
    assert set(verdict["per_detector"]) == set(originals)
    assert all(engine.detectors[name] is original for name, original in originals.items())
    assert all(session._detectors[name] is not original for name, original in originals.items())


def test_an_uncopyable_detector_is_rebuilt_from_its_configuration(trained_engine: Any) -> None:
    """A lock-holding detector gets a new instance, never the engine's own."""
    import threading

    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    class Locked(MercuryAnomalyDetector):
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            super().__init__(config)
            self.guard = threading.Lock()

    locked = Locked({"threshold": 0.7})
    trained_engine.detectors["statistical"] = locked
    session = trained_engine.stream_session(
        detector_types=["statistical"], min_samples=16, window_size=48
    )

    private = session._detectors["statistical"]
    assert type(private) is Locked and private is not locked
    assert private.guard is not locked.guard
    assert private.config == locked.config and private.config is not locked.config


def test_a_detector_that_can_be_neither_copied_nor_rebuilt_is_refused(
    trained_engine: Any,
) -> None:
    import threading

    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    class NeedsAnArgument(MercuryAnomalyDetector):
        def __init__(self, *, required: int) -> None:
            super().__init__()
            self.guard = threading.Lock()
            self.required = required

    trained_engine.detectors["statistical"] = NeedsAnArgument(required=3)
    with pytest.raises(TypeError, match="cannot be given to a stream session"):
        trained_engine.stream_session(detector_types=["statistical"])
