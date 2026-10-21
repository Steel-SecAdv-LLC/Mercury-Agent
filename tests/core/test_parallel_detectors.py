# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The restored ParallelDetectorExecutor and its use by the engine and registry.

Pins the defects of the version removed in 7ab59916 and of the registry loop
at 25d624e9 — a timeout that discarded finished results and did not bound
latency, a hung worker starving later calls, completion-order results — and
pins that concurrent execution of the default detectors is
bit-identical to sequential execution.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from omni_mercury_engine.core import parallel_detectors
from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.core.parallel_detectors import (
    DetectorTask,
    ExecutorSaturatedError,
    OutcomeStatus,
    ParallelDetectorExecutor,
    first_failure,
    reraise,
)


def _sleeper(seconds: float, value: Any) -> Any:
    def fn() -> Any:
        time.sleep(seconds)
        return value

    return fn


class TestExecutor:
    def test_outcomes_follow_input_order_not_completion_order(self) -> None:
        tasks = [DetectorTask(f"t{i}", _sleeper(0.05 * (4 - i), i)) for i in range(4)]
        with ParallelDetectorExecutor(max_workers=4) as ex:
            outcomes = ex.run(tasks)
        assert [o.name for o in outcomes] == ["t0", "t1", "t2", "t3"]
        assert [o.value for o in outcomes] == [0, 1, 2, 3]
        assert all(o.status is OutcomeStatus.OK for o in outcomes)

    def test_runs_concurrently(self) -> None:
        tasks = [DetectorTask(f"t{i}", _sleeper(0.2, i)) for i in range(4)]
        with ParallelDetectorExecutor(max_workers=4) as ex:
            start = time.perf_counter()
            ex.run(tasks)
            elapsed = time.perf_counter() - start
        assert elapsed < 0.6  # sequential would take 0.8 s

    def test_timeout_keeps_finished_results_and_returns_at_deadline(self) -> None:
        tasks = [
            DetectorTask("fast", _sleeper(0.0, "ok")),
            DetectorTask("slow", _sleeper(2.0, "late")),
            DetectorTask("fast2", _sleeper(0.05, "ok2")),
        ]
        with ParallelDetectorExecutor(max_workers=3) as ex:
            start = time.perf_counter()
            outcomes = ex.run(tasks, timeout_s=0.4)
            elapsed = time.perf_counter() - start
            ex.shutdown(wait=False)
        by_name = {o.name: o for o in outcomes}
        assert by_name["fast"].value == "ok" and by_name["fast2"].value == "ok2"
        assert by_name["slow"].status is OutcomeStatus.TIMEOUT
        assert by_name["slow"].value is None
        assert 0.35 < by_name["slow"].duration_s < 1.0
        assert elapsed < 1.0  # the old as_completed loop waited the full 2 s

    def test_never_started_task_is_cancelled_and_reported(self) -> None:
        release = threading.Event()
        tasks = [
            DetectorTask("blocker_a", lambda: release.wait(5)),
            DetectorTask("blocker_b", lambda: release.wait(5)),
            DetectorTask("queued", _sleeper(0.0, "never")),
        ]
        ex = ParallelDetectorExecutor(max_workers=2)
        try:
            outcomes = ex.run(tasks, timeout_s=0.2)
        finally:
            release.set()
            ex.shutdown(wait=True)
        queued = outcomes[2]
        assert queued.status is OutcomeStatus.TIMEOUT
        assert queued.duration_s == 0.0  # it never started

    def test_timed_out_queued_task_never_runs_once_workers_free_up(self) -> None:
        # Without cancellation the queued task would start the moment a blocker
        # returned: work the caller already abandoned, still holding a worker.
        release = threading.Event()
        ran = threading.Event()
        tasks = [
            DetectorTask("blocker_a", lambda: release.wait(5)),
            DetectorTask("blocker_b", lambda: release.wait(5)),
            DetectorTask("queued", ran.set),
        ]
        ex = ParallelDetectorExecutor(max_workers=2)
        try:
            ex.run(tasks, timeout_s=0.2)
            release.set()
            assert not ran.wait(0.3)
        finally:
            release.set()
            ex.shutdown(wait=True)

    def test_declined_and_error_are_classified(self) -> None:
        def declines() -> None:
            raise InputModalityError("not my modality")

        def fails() -> None:
            raise RuntimeError("fault")

        with ParallelDetectorExecutor(max_workers=2) as ex:
            outcomes = ex.run(
                [
                    DetectorTask("d", declines),
                    DetectorTask("e", fails),
                    DetectorTask("o", lambda: 1),
                ]
            )
        assert [o.status for o in outcomes] == [
            OutcomeStatus.DECLINED,
            OutcomeStatus.ERROR,
            OutcomeStatus.OK,
        ]
        assert isinstance(outcomes[0].error, InputModalityError)
        assert first_failure(outcomes) is outcomes[0]
        with pytest.raises(RuntimeError, match="fault"):
            reraise(outcomes[1])
        reraise(outcomes[2])  # OK outcome: no-op

    def test_reraise_timeout(self) -> None:
        from omni_mercury_engine.core.parallel_detectors import TaskOutcome

        with pytest.raises(TimeoutError, match="slow"):
            reraise(TaskOutcome("slow", OutcomeStatus.TIMEOUT))

    def test_non_parallel_tasks_run_in_calling_thread_in_order(self) -> None:
        seen: list[tuple[str, str]] = []
        main = threading.current_thread().name

        def record(tag: str) -> Any:
            def fn() -> str:
                seen.append((tag, threading.current_thread().name))
                return tag

            return fn

        tasks = [
            DetectorTask("p1", _sleeper(0.05, "p1")),
            DetectorTask("s1", record("s1"), parallel_safe=False),
            DetectorTask("p2", _sleeper(0.05, "p2")),
            DetectorTask("s2", record("s2"), parallel_safe=False),
        ]
        with ParallelDetectorExecutor(max_workers=2) as ex:
            outcomes = ex.run(tasks)
        assert [o.value for o in outcomes] == ["p1", "s1", "p2", "s2"]
        assert seen == [("s1", main), ("s2", main)]

    def test_nested_call_from_worker_runs_sequentially_without_deadlock(self) -> None:
        ex = ParallelDetectorExecutor(max_workers=2)

        def inner_batch() -> list[Any]:
            inner = ex.run([DetectorTask("a", lambda: 1), DetectorTask("b", lambda: 2)])
            return [o.value for o in inner]

        try:
            outcomes = ex.run(
                [DetectorTask("x", inner_batch), DetectorTask("y", inner_batch)], timeout_s=5.0
            )
        finally:
            ex.shutdown()
        assert [o.value for o in outcomes] == [[1, 2], [1, 2]]

    def test_single_worker_runs_everything_in_calling_thread(self) -> None:
        main = threading.current_thread().name
        with ParallelDetectorExecutor(max_workers=1) as ex:
            outcomes = ex.run(
                [
                    DetectorTask("a", lambda: threading.current_thread().name),
                    DetectorTask("b", lambda: threading.current_thread().name),
                ]
            )
        assert [o.value for o in outcomes] == [main, main]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"max_workers": 0}, "max_workers"), ({"timeout_s": 0.0}, "timeout_s")],
    )
    def test_rejects_bad_configuration(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ParallelDetectorExecutor(**kwargs)

    def test_rejects_duplicate_names(self) -> None:
        with ParallelDetectorExecutor() as ex, pytest.raises(ValueError, match="unique"):
            ex.run([DetectorTask("a", lambda: 1), DetectorTask("a", lambda: 2)])

    def test_hung_task_does_not_starve_the_next_call(self) -> None:
        # At 25d624e9 the registry's next call queued behind the stuck worker
        # and timed out as well; the overrun now retires the pool.
        release = threading.Event()
        ex = ParallelDetectorExecutor(max_workers=2)
        try:
            first = ex.run(  # both workers stuck: the old pool has no free thread
                [
                    DetectorTask("hung_a", lambda: release.wait(5)),
                    DetectorTask("hung_b", lambda: release.wait(5)),
                ],
                timeout_s=0.2,
            )
            assert [o.status for o in first] == [OutcomeStatus.TIMEOUT] * 2
            assert ex.overrun_running == 2
            start = time.monotonic()
            second = ex.run(
                [DetectorTask("a", _sleeper(0.05, "a")), DetectorTask("b", _sleeper(0.05, "b"))],
                timeout_s=1.0,
            )
            assert time.monotonic() - start < 0.5
            assert [o.value for o in second] == ["a", "b"]
        finally:
            release.set()
            ex.shutdown(wait=True)

    def test_overrun_cap_refuses_parallel_work_until_threads_free(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(parallel_detectors, "OVERRUN_LIMIT_PER_WORKER", 1)
        release = threading.Event()
        ex = ParallelDetectorExecutor(max_workers=2)  # cap: 2 stranded threads
        try:
            ex.run(
                [
                    DetectorTask("h1", lambda: release.wait(5)),
                    DetectorTask("h2", lambda: release.wait(5)),
                ],
                timeout_s=0.1,
            )
            assert ex.overrun_running == 2
            refused = ex.run(
                [
                    DetectorTask("p1", lambda: 1),
                    DetectorTask("p2", lambda: 2),
                    DetectorTask("seq", lambda: 3, parallel_safe=False),
                ]
            )
            assert [o.status for o in refused] == [
                OutcomeStatus.ERROR,
                OutcomeStatus.ERROR,
                OutcomeStatus.OK,
            ]
            assert isinstance(refused[0].error, ExecutorSaturatedError)
            assert refused[2].value == 3  # sequential work is unaffected

            release.set()
            deadline = time.monotonic() + 5.0
            while ex.overrun_running and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ex.overrun_running == 0
            recovered = ex.run([DetectorTask("p1", lambda: 1), DetectorTask("p2", lambda: 2)])
            assert [o.value for o in recovered] == [1, 2]
        finally:
            release.set()
            ex.shutdown(wait=True)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
def _snapshot(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _snapshot(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_snapshot(v) for v in obj]
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return (obj.shape, obj.dtype.str, obj.tobytes())
    return obj


def _engine(num_workers: int) -> Any:
    from omni_mercury_engine import OmniMercuryEngine
    from omni_mercury_engine.core.config import EngineConfig

    return OmniMercuryEngine(config=EngineConfig(num_workers=num_workers))


class TestEngine:
    def test_engine_uses_config_num_workers(self) -> None:
        assert _engine(3).detector_executor.max_workers == 3
        assert _engine(1).detector_executor.max_workers == 1

    def test_default_detectors_declare_parallel_safety(self) -> None:
        engine = _engine(4)
        assert {n: d.parallel_safe for n, d in engine.detectors.items()} == {
            "statistical": True,
            "temporal": True,
            "spatial": True,
            "dimensional": True,
            "directive": True,
        }

    def test_concurrent_detection_is_bit_identical_to_sequential(self) -> None:
        X = np.random.default_rng(11).standard_normal((300, 6)).astype(np.float32)
        sequential, parallel = _engine(1), _engine(4)
        for engine in (sequential, parallel):
            for detector in engine.detectors.values():
                detector.fit(X)
        # ``detect()`` is stateful across calls (the directive detector keeps a
        # streaming history), so compare call k against call k: both engines
        # must evolve identically.
        for _ in range(5):
            for engine in (sequential, parallel):
                engine.feature_cache.clear()
            assert _snapshot(parallel.detect(X)) == _snapshot(sequential.detect(X))
            assert _snapshot(parallel._extract_detector_features(X)) == _snapshot(
                sequential._extract_detector_features(X)
            )

    def test_detect_reraises_first_failure_in_order_after_all_ran(self) -> None:
        engine = _engine(4)
        X = np.zeros((10, 3))
        calls: list[str] = []

        class _Stub:
            parallel_safe = True

            def __init__(self, name: str, exc: Exception | None) -> None:
                self.name, self.exc = name, exc

            def is_fitted(self) -> bool:
                return True

            def detect(self, data: Any) -> dict[str, Any]:
                calls.append(self.name)
                if self.exc is not None:
                    raise self.exc
                return {"is_anomaly": False, "scores": np.zeros(len(data))}

        engine.detectors = {
            "a": _Stub("a", None),
            "b": _Stub("b", KeyError("first")),
            "c": _Stub("c", RuntimeError("second")),
        }
        with pytest.raises(KeyError, match="first"):
            engine.detect(X)
        assert sorted(calls) == ["a", "b", "c"]

    def test_mock_detectors_are_not_treated_as_parallel_safe(self) -> None:
        from omni_mercury_engine.engine import _is_parallel_safe

        assert _is_parallel_safe(MagicMock()) is False
        assert _is_parallel_safe(type("D", (), {"parallel_safe": 1})()) is False
        assert _is_parallel_safe(type("D", (), {"parallel_safe": True})()) is True

    def test_thread_bound_streams_stay_on_the_callers_thread_for_detect(self) -> None:
        """directive's per-thread memory must advance on the caller's thread.

        Its buffer lives in thread-local storage. Run on a pool, consecutive
        ``detect()`` calls could land on different workers and see different
        histories, so ``detect()`` keeps it in the caller's thread; the feature
        path resets transient state first and may run it anywhere.
        """
        from omni_mercury_engine.engine import _is_parallel_safe

        directive = _engine(4).detectors["directive"]
        assert directive.thread_bound_state is True
        assert _is_parallel_safe(directive) is True
        assert _is_parallel_safe(directive, stateful_call=True) is False
        X = np.random.default_rng(3).standard_normal((40, 4))
        engine = _engine(4)
        for detector in engine.detectors.values():
            detector.fit(X)
        engine.detect(X)
        engine.detect(X)
        own_buffer = engine.detectors["directive"]._get_thread_state()
        assert own_buffer is engine.detectors["directive"]._get_thread_state()
        assert len(own_buffer.memory_buffer) > 0  # the caller's stream advanced


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
class _SlowDetector:
    parallel_safe = True

    def __init__(self, delay: float, width: int) -> None:
        self.delay, self.width = delay, width

    def is_fitted(self) -> bool:
        return True

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        time.sleep(self.delay)
        return np.full((len(data), self.width), self.delay)

    def predict(self, data: Any) -> dict[str, Any]:
        return {"scores": np.zeros(len(data))}


def test_registry_keeps_finished_results_when_one_detector_times_out() -> None:
    from omni_mercury_engine.core.detector_registry import DetectorCategory, DetectorRegistry

    registry = DetectorRegistry(max_workers=3, timeout_seconds=0.4)
    registry.register("fast", _SlowDetector(0.0, 2), category=DetectorCategory.BASE)
    registry.register("slow", _SlowDetector(2.0, 3), category=DetectorCategory.BASE)
    registry.register("medium", _SlowDetector(0.05, 4), category=DetectorCategory.BASE)
    start = time.perf_counter()
    results = registry.extract_all_features(np.zeros((5, 2)), parallel=True)
    elapsed = time.perf_counter() - start
    assert list(results) == ["fast", "slow", "medium"]  # request order
    assert results["fast"].success and results["medium"].success
    assert results["slow"].success is False
    assert "timed out" in (results["slow"].error or "")
    assert elapsed < 1.5


# --------------------------------------------------------------------------
# Deadline coverage, pool-retirement race, timeouts as breaker failures
# --------------------------------------------------------------------------
class TestDeadlineCoverage:
    def test_a_lone_parallel_task_is_still_bounded_by_the_budget(self) -> None:
        # With fewer than two parallel-safe tasks everything used to run in
        # the calling thread, so the registry's timeout_seconds did not bind.
        release = threading.Event()
        ex = ParallelDetectorExecutor(max_workers=2)
        try:
            start = time.monotonic()
            outcomes = ex.run(
                [
                    DetectorTask("hung", lambda: release.wait(5)),
                    DetectorTask("seq", lambda: "s", parallel_safe=False),
                ],
                timeout_s=0.2,
            )
            assert time.monotonic() - start < 1.5
            assert outcomes[0].status is OutcomeStatus.TIMEOUT
            assert outcomes[1].value == "s"
        finally:
            release.set()
            ex.shutdown(wait=True)

    def test_a_lone_parallel_task_without_budget_runs_inline(self) -> None:
        main = threading.current_thread().name
        with ParallelDetectorExecutor(max_workers=2) as ex:
            outcomes = ex.run([DetectorTask("a", lambda: threading.current_thread().name)])
        assert outcomes[0].value == main

    def test_sequential_tasks_outlasting_the_budget_are_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            ParallelDetectorExecutor(max_workers=2) as ex,
            caplog.at_level("WARNING", logger=parallel_detectors.__name__),
        ):
            outcomes = ex.run(
                [
                    DetectorTask("p", lambda: "p"),
                    DetectorTask("seq", _sleeper(0.2, "s"), parallel_safe=False),
                ],
                timeout_s=0.05,
            )
        assert [o.value for o in outcomes] == ["p", "s"]  # sequential work is not cut
        assert any("outside the deadline" in r.getMessage() for r in caplog.records)

    def test_a_concurrent_retirement_cannot_break_submission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The pool was fetched and then submitted to without the lock, so a
        # retirement in between shut it down and ``submit`` raised
        # RuntimeError out of run(). The racer retires the pool while the
        # first task is being submitted; it must wait until all are queued.
        from concurrent.futures import ThreadPoolExecutor

        ex = ParallelDetectorExecutor(max_workers=2)
        racers: list[threading.Thread] = []

        class _RacingPool(ThreadPoolExecutor):
            raced = False

            def submit(self, *args: Any, **kwargs: Any) -> Any:
                if not self.raced:
                    self.raced = True
                    racer = threading.Thread(target=ex._retire_pool, args=(self,))
                    racer.start()
                    racer.join(0.3)  # blocked on the pool lock under the fix
                    racers.append(racer)
                return super().submit(*args, **kwargs)

        monkeypatch.setattr(parallel_detectors, "ThreadPoolExecutor", _RacingPool)
        try:
            outcomes = ex.run([DetectorTask(f"t{i}", functools.partial(int, i)) for i in range(3)])
            assert [o.value for o in outcomes] == [0, 1, 2]
            for racer in racers:
                racer.join(5)
                assert not racer.is_alive()
            # The retired pool is gone; the next call gets a fresh one.
            again = ex.run([DetectorTask("a", lambda: 1), DetectorTask("b", lambda: 2)])
            assert [o.value for o in again] == [1, 2]
        finally:
            ex.shutdown(wait=True)


class TestEngineDetectorTimeout:
    def test_config_budget_reaches_the_executor(self) -> None:
        from omni_mercury_engine import OmniMercuryEngine
        from omni_mercury_engine.core.config import EngineConfig

        engine = OmniMercuryEngine(config=EngineConfig(detector_timeout_s=0.5))
        assert engine.detector_executor.timeout_s == 0.5
        assert _engine(2).detector_executor.timeout_s == 5.0  # default fixed part
        unbounded = OmniMercuryEngine(config=EngineConfig(detector_timeout_s=None))
        assert unbounded.detector_executor.timeout_s is None
        assert unbounded._detector_budget(np.zeros((10_000, 3))) is None

    def test_the_budget_grows_with_the_batch(self) -> None:
        from omni_mercury_engine import OmniMercuryEngine
        from omni_mercury_engine.core.config import EngineConfig

        engine = OmniMercuryEngine(
            config=EngineConfig(detector_timeout_s=5.0, detector_timeout_per_row_s=1e-3)
        )
        assert engine._detector_budget(np.zeros((0, 3))) == 5.0
        assert engine._detector_budget(np.zeros((10_000, 3))) == pytest.approx(15.0)
        multimodal = {"a": np.zeros((100, 2)), "b": np.zeros((2_000, 4))}
        assert engine._detector_budget(multimodal) == pytest.approx(7.0)  # largest array
        fixed = OmniMercuryEngine(
            config=EngineConfig(detector_timeout_s=5.0, detector_timeout_per_row_s=None)
        )
        assert fixed._detector_budget(np.zeros((10_000, 3))) == 5.0

    def test_the_defaults_are_the_committed_measurement(self) -> None:
        """EngineConfig's budget must be the one benchmarks/detector_latency.py derived."""
        import json
        from pathlib import Path

        from omni_mercury_engine.core.config import EngineConfig

        path = Path(__file__).resolve().parents[2] / "benchmarks/results/detector_latency.json"
        measured = json.loads(path.read_text(encoding="utf-8"))
        config = EngineConfig()
        assert config.detector_timeout_s == measured["recommended_timeout_floor_s"]
        assert config.detector_timeout_per_row_s == measured["recommended_timeout_per_row_s"]
        # Every served detector measured close to linear; a quadratic one would
        # outgrow a per-row budget, so the gate fails if one reappears.
        assert all(fit["b"] < 1.2 for fit in measured["power_law"].values()), measured["power_law"]

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_config_rejects_a_non_positive_budget(self, bad: float) -> None:
        from omni_mercury_engine.core.config import EngineConfig

        with pytest.raises(ValueError, match="detector_timeout_s"):
            EngineConfig(detector_timeout_s=bad)
        with pytest.raises(ValueError, match="detector_timeout_per_row_s"):
            EngineConfig(detector_timeout_per_row_s=-1e-3)

    def test_an_overrunning_detector_is_skipped(self) -> None:
        from omni_mercury_engine import OmniMercuryEngine
        from omni_mercury_engine.core.config import EngineConfig

        release = threading.Event()

        class _Det:
            parallel_safe = True

            def __init__(self, hang: bool) -> None:
                self.hang = hang

            def is_fitted(self) -> bool:
                return True

            def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
                if self.hang:
                    release.wait(5)
                return np.zeros((len(data), 2), dtype=np.float32)

            def detect(self, data: Any) -> dict[str, Any]:
                return {"scores": np.zeros(len(data))}

        engine = OmniMercuryEngine(config=EngineConfig(num_workers=2, detector_timeout_s=0.3))
        detectors: dict[str, Any] = {"hung": _Det(hang=True), "fine": _Det(hang=False)}
        engine.detectors = detectors
        try:
            start = time.monotonic()
            features, scores, _ = engine._extract_detector_features(
                np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
            )
            assert time.monotonic() - start < 2.0
            assert set(features) == {"fine"} and set(scores) == {"fine"}
        finally:
            release.set()
            engine.detector_executor.shutdown(wait=True)


class _HangingDetector:
    parallel_safe = True

    def __init__(self, release: threading.Event) -> None:
        self.release = release
        self.calls = 0

    def is_fitted(self) -> bool:
        return True

    def extract_features(self, data: Any) -> np.ndarray[Any, Any]:
        self.calls += 1
        self.release.wait(10)
        return np.zeros((len(data), 1))

    def predict(self, data: Any) -> dict[str, Any]:
        return {"scores": np.zeros(len(data))}


class TestRegistryTimeoutAccounting:
    def test_a_hung_detector_trips_its_breaker_instead_of_saturating_the_pool(self) -> None:
        # A timeout recorded neither a breaker failure nor error_count, and
        # the abandoned call later counted as a success, so the hung detector
        # was relaunched every call, stranding a thread each time until the
        # executor saturated and the healthy detectors failed too.
        import uuid

        from omni_mercury_engine.core.detector_registry import DetectorCategory, DetectorRegistry
        from omni_mercury_engine.resilience import api_circuit_breakers
        from omni_mercury_engine.resilience.api_circuit_breakers import get_detector_breaker
        from omni_mercury_engine.resilience.circuit_breaker import CircuitState

        tag = uuid.uuid4().hex[:8]
        hung, healthy = f"hung-{tag}", [f"ok{i}-{tag}" for i in range(2)]
        release = threading.Event()
        hanging = _HangingDetector(release)
        registry = DetectorRegistry(max_workers=2, timeout_seconds=0.25)  # cap: 8 stranded
        registry.register(hung, hanging, category=DetectorCategory.BASE)
        for name in healthy:
            registry.register(name, _SlowDetector(0.0, 2), category=DetectorCategory.BASE)
        breaker = get_detector_breaker(hung)
        threshold = breaker.config.failure_threshold
        try:
            for _ in range(threshold + 7):  # past the 8-thread cap without the fix
                results = registry.extract_all_features(np.zeros((3, 2)), parallel=True)
                for name in healthy:
                    assert results[name].success, results[name].error
                assert results[hung].success is False
            assert breaker.state == CircuitState.OPEN
            info = registry.get(hung)
            assert info is not None and info.error_count == threshold
            assert hanging.calls == threshold  # not relaunched once the breaker opened

            # The abandoned calls now finish: none may count as a success.
            release.set()
            executor = registry._get_executor()
            deadline = time.monotonic() + 10
            while executor.overrun_running and time.monotonic() < deadline:
                time.sleep(0.01)
            assert executor.overrun_running == 0
            assert breaker.get_stats()["total_successes"] == 0
            assert breaker.state == CircuitState.OPEN
            assert info.total_invocations == 0 and info.error_count == threshold
        finally:
            release.set()
            for name in (hung, *healthy):
                api_circuit_breakers._detector_breakers.pop(name, None)
            if registry._executor is not None:
                registry._executor.shutdown(wait=True)

    def test_a_call_abandoned_before_it_started_never_runs(self) -> None:
        from omni_mercury_engine.core.detector_registry import (
            DetectorCategory,
            DetectorRegistry,
            _CallTicket,
        )

        release = threading.Event()
        release.set()
        detector = _HangingDetector(release)
        registry = DetectorRegistry()
        registry.register("never-started", detector, category=DetectorCategory.BASE)
        ticket = _CallTicket()
        assert ticket.abandon() is False  # nothing ran, so nothing to report
        result = registry._extract_features("never-started", np.zeros((2, 2)), ticket)
        assert result.success is False and detector.calls == 0

    def test_ticket_reports_each_call_once(self) -> None:
        from omni_mercury_engine.core.detector_registry import _CallTicket

        finished = _CallTicket()
        assert finished.start() and finished.finish()
        assert finished.abandon() is False  # the worker already reported it

        overran = _CallTicket()
        assert overran.start()
        assert overran.abandon() is True  # the caller reports the timeout
        assert overran.finish() is False  # and the late completion is not reported
