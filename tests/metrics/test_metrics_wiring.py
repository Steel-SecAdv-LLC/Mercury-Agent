# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Prometheus series the alert rules and the engine HPA query are emitted.

Each test reads the live ``prometheus_client`` default registry before and
after a real call path -- engine detection, fusion inference, fusion training,
the streaming worker -- and asserts the series moved by the exact amount. The
decline tests pin the property the success-rate and inference-error alerts
depend on: a component refusing off-modality input is counted as a decline and
nowhere else.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

pytest.importorskip("prometheus_client")

from prometheus_client import REGISTRY

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.core.metrics import (
    set_detection_queue_depth,
    time_detection,
    time_feature_extraction,
)
from omni_mercury_engine.infrastructure.streaming import (
    InMemoryStreamBroker,
    InMemoryStreamConsumer,
    KafkaStreamConsumer,
    RedisStreamConsumer,
    StreamingAnomalyPipeline,
)


def _value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class _Snapshot:
    """Before/after reader for one component's detection-path series."""

    def __init__(self, component: str) -> None:
        self.component = component
        self.before = self._read()

    def _read(self) -> dict[str, float]:
        c = self.component
        return {
            "requests": _value("omni_detection_requests_total", detector_type=c),
            "success": _value("omni_detection_success_total", detector_type=c),
            "duration": _value("omni_detection_duration_seconds_count", detector_type=c),
            "errors": _value("omni_model_inference_errors_total", model_name=c),
            "declined": _value("omni_component_declined_total", component=c),
            "extract_ok": _value("omni_detector_extraction_success_total", detector_name=c),
            "extract_err": _value("omni_detector_extraction_errors_total", detector_name=c),
        }

    def delta(self) -> dict[str, float]:
        after = self._read()
        return {k: after[k] - self.before[k] for k in after}


_ZERO = {
    "requests": 0.0,
    "success": 0.0,
    "duration": 0.0,
    "errors": 0.0,
    "declined": 0.0,
    "extract_ok": 0.0,
    "extract_err": 0.0,
}


def _expect(**changes: float) -> dict[str, float]:
    return {**_ZERO, **changes}


# --------------------------------------------------------------------------
# Context managers
# --------------------------------------------------------------------------
class TestTimingContextManagers:
    def test_success_counts_request_success_and_duration(self) -> None:
        snap = _Snapshot("cm_ok")
        with time_detection("cm_ok"):
            pass
        assert snap.delta() == _expect(requests=1, success=1, duration=1)

    def test_failure_counts_request_without_success(self) -> None:
        snap = _Snapshot("cm_fail")
        with pytest.raises(RuntimeError), time_detection("cm_fail"):
            raise RuntimeError("boom")
        assert snap.delta() == _expect(requests=1, duration=1)

    def test_decline_touches_no_detection_series(self) -> None:
        snap = _Snapshot("cm_decline")
        with pytest.raises(InputModalityError), time_detection("cm_decline"):
            raise InputModalityError("wrong modality")
        assert snap.delta() == _ZERO

    def test_extraction_success_error_and_decline(self) -> None:
        snap = _Snapshot("cm_extract")
        with time_feature_extraction("cm_extract"):
            pass
        with pytest.raises(ValueError), time_feature_extraction("cm_extract"):
            raise ValueError("bad input")
        with pytest.raises(InputModalityError), time_feature_extraction("cm_extract"):
            raise InputModalityError("wrong modality")
        assert snap.delta() == _expect(extract_ok=1, extract_err=1)

    def test_input_modality_error_is_a_value_error_only(self) -> None:
        from omni_mercury_engine.core.exceptions import OmniAnomalyException

        assert issubclass(InputModalityError, ValueError)
        assert not issubclass(InputModalityError, OmniAnomalyException)

    def test_queue_depth_unknown_is_minus_one(self) -> None:
        set_detection_queue_depth(7)
        assert _value("omni_detection_queue_depth") == 7
        set_detection_queue_depth(None)
        assert _value("omni_detection_queue_depth") == -1


# --------------------------------------------------------------------------
# Engine detection path
# --------------------------------------------------------------------------
class _StubDetector:
    def __init__(self, exc: Exception | None) -> None:
        self.exc = exc

    def is_fitted(self) -> bool:
        return True

    def detect(self, data: Any) -> dict[str, Any]:
        if self.exc is not None:
            raise self.exc
        return {"is_anomaly": False, "scores": np.zeros(len(data))}


@pytest.fixture(scope="module")
def engine() -> Any:
    from omni_mercury_engine import OmniMercuryEngine

    return OmniMercuryEngine()


class TestEngineDetect:
    def test_real_detector_call_is_counted(self, engine: Any) -> None:
        X = np.random.default_rng(0).standard_normal((40, 4))
        snap = _Snapshot("statistical")
        engine.detect(X, detector_types=["statistical"])
        assert snap.delta() == _expect(requests=1, success=1, duration=1)

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (RuntimeError("detector fault"), _expect(requests=1, duration=1, errors=1)),
            (InputModalityError("not my modality"), _expect(declined=1)),
        ],
        ids=["fault", "decline"],
    )
    def test_failure_classification(
        self, engine: Any, exc: Exception, expected: dict[str, float]
    ) -> None:
        name = f"stub_{type(exc).__name__}"
        engine.detectors[name] = _StubDetector(exc)
        try:
            snap = _Snapshot(name)
            with pytest.raises(type(exc)):
                engine.detect(np.zeros((5, 3)), detector_types=[name])
            assert snap.delta() == expected
        finally:
            del engine.detectors[name]


# --------------------------------------------------------------------------
# Fusion path: training quality gauges, inference timing, model declines
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted_fusion() -> tuple[Any, dict[str, Any]]:
    from omni_mercury_engine import OmniMercuryEngine

    rng = np.random.default_rng(1)
    X = rng.standard_normal((160, 6)).astype(np.float32)
    y = np.zeros(160, dtype=int)
    y[::8] = 1
    X[y == 1] += 3.0
    eng = OmniMercuryEngine()
    metrics = eng.fit_fusion(X, y, epochs=3)
    return eng, metrics


class TestFusionMetrics:
    def test_validation_quality_published(self, fitted_fusion: tuple[Any, dict[str, Any]]) -> None:
        from omni_mercury_engine.engine import __version__

        _, metrics = fitted_fusion
        assert metrics["val_roc_auc"] is not None
        labels = {"model_version": __version__, "dataset": "fit_validation"}
        assert REGISTRY.get_sample_value("omni_model_roc_auc", labels) == pytest.approx(
            metrics["val_roc_auc"]
        )
        assert REGISTRY.get_sample_value("omni_model_f1_score", labels) == pytest.approx(
            metrics["val_f1"]
        )
        assert 0.0 <= metrics["val_f1"] <= 1.0
        assert 0.0 <= metrics["val_roc_auc"] <= 1.0

    def test_f1_uses_the_served_threshold_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """F1 is scored at the engine's own threshold rule on calibrated probabilities."""
        from omni_mercury_engine import OmniMercuryEngine

        rng = np.random.default_rng(3)
        X = rng.standard_normal((120, 5)).astype(np.float32)
        y = np.zeros(120, dtype=int)
        y[::6] = 1
        X[y == 1] += 3.0
        eng = OmniMercuryEngine()
        seen: list[np.ndarray[Any, Any]] = []

        def _above_everything(current: float, all_scores: Any = None) -> float:
            seen.append(np.asarray(all_scores))
            return 2.0  # no probability exceeds it -> no predicted positives

        monkeypatch.setattr(eng, "_get_anomaly_threshold", _above_everything)
        metrics = eng.fit_fusion(X, y, epochs=2)
        assert metrics["val_f1"] == 0.0
        assert len(seen) == 1 and seen[0].ndim == 1 and seen[0].size > 0
        assert np.all((seen[0] >= 0.0) & (seen[0] <= 1.0))

    def test_inference_timed_and_off_modality_models_declined(
        self, fitted_fusion: tuple[Any, dict[str, Any]]
    ) -> None:
        eng, _ = fitted_fusion
        fusion_type = type(eng.fusion_model).__name__
        fusion_before = _value(
            "omni_fusion_inference_duration_seconds_count", fusion_type=fusion_type
        )
        # medical_abms needs a patient-record mapping: on a plain array it must
        # decline, never register as an inference error or a failed detection.
        abms = _Snapshot("medical_abms")
        eng.detect_with_fusion(np.random.default_rng(2).standard_normal((20, 6)).astype(np.float32))
        assert (
            _value("omni_fusion_inference_duration_seconds_count", fusion_type=fusion_type)
            == fusion_before + 1
        )
        delta = abms.delta()
        assert delta["declined"] >= 1
        assert delta["errors"] == 0
        assert delta["requests"] == 0

    def test_fusion_failure_counted_and_reraised(
        self, fitted_fusion: tuple[Any, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        eng, _ = fitted_fusion

        def _boom(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("fusion head fault")

        monkeypatch.setattr(eng.fusion_inference, "predict", _boom)
        before = _value("omni_model_inference_errors_total", model_name="fusion")
        with pytest.raises(RuntimeError, match="fusion head fault"):
            eng._timed_fusion_predict({})
        assert _value("omni_model_inference_errors_total", model_name="fusion") == before + 1


# --------------------------------------------------------------------------
# Streaming worker: detection series and consumer-group backlog
# --------------------------------------------------------------------------
def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class TestStreamingMetrics:
    def test_in_memory_lag_tracks_unread_entries(self) -> None:
        async def scenario() -> list[int | None]:
            broker = InMemoryStreamBroker()
            for i in range(5):
                await broker.append("in", {"v": i})
            consumer = InMemoryStreamConsumer(broker=broker)
            await consumer.subscribe(["in"])
            lags = [await consumer.lag()]
            seen = 0
            async for _ in consumer.consume(timeout_ms=200):
                seen += 1
                if seen == 2:
                    break
            lags.append(await consumer.lag())
            return lags

        assert _run(scenario()) == [5, 3]

    def test_refresh_backlog_publishes_and_rate_limits(self) -> None:
        async def scenario() -> tuple[Any, ...]:
            broker = InMemoryStreamBroker()
            pipeline = StreamingAnomalyPipeline("in", "out", backend="memory", broker=broker)
            await pipeline._consumer.subscribe(["in"])
            for i in range(4):
                await broker.append("in", {"v": i})
            first = await pipeline.refresh_backlog()
            await broker.append("in", {"v": 99})
            limited = await pipeline.refresh_backlog()
            forced = await pipeline.refresh_backlog(force=True)
            return first, limited, forced, pipeline.get_stats()

        first, limited, forced, stats = _run(scenario())
        assert (first, limited, forced) == (4, None, 5)
        assert stats["queue_depth"] == stats["consumer_lag"] == 5
        assert _value("omni_detection_queue_depth") == 5

    def test_failing_lag_query_publishes_unknown(self) -> None:
        async def scenario() -> tuple[Any, Any]:
            pipeline = StreamingAnomalyPipeline(
                "in", "out", backend="memory", broker=InMemoryStreamBroker()
            )

            async def _broken() -> int:
                raise ConnectionError("broker unreachable")

            pipeline._consumer.lag = _broken  # type: ignore[method-assign]
            return await pipeline.refresh_backlog(force=True), pipeline.get_stats()

        depth, stats = _run(scenario())
        assert depth is None
        assert stats["queue_depth"] is None
        assert _value("omni_detection_queue_depth") == -1

    def test_detector_calls_feed_detection_series(self) -> None:
        pipeline = StreamingAnomalyPipeline(
            "in", "out", backend="memory", broker=InMemoryStreamBroker()
        )
        snap = _Snapshot("stream")
        pipeline._detect({"x": 1.0})
        pipeline.detector = lambda _v: (_ for _ in ()).throw(TypeError("bad payload"))
        with pytest.raises(TypeError):
            pipeline._detect({"x": 1.0})
        assert snap.delta() == _expect(requests=2, success=1, duration=2, errors=1)


def test_empty_consume_cycles_yield_to_the_event_loop() -> None:
    """An open circuit breaker must not starve the loop (stop() has to run).

    Kafka/Redis ``consume()`` returns without awaiting while its breaker is
    open. The worker loop used to spin on that forever, never suspending, so
    ``stop()`` and the SIGTERM handler could not run. Runs in a thread so a
    regression fails on the join timeout instead of hanging the session.
    """
    import threading

    async def scenario() -> str:
        pipeline = StreamingAnomalyPipeline(
            "in", "out", backend="memory", broker=InMemoryStreamBroker()
        )

        async def _open_breaker(timeout_ms: int = 1000) -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

        pipeline._consumer.consume = _open_breaker  # type: ignore[method-assign]
        await pipeline.start()
        await asyncio.sleep(0.3)
        await pipeline.stop()
        return "stopped"

    result: dict[str, str] = {}
    worker = threading.Thread(target=lambda: result.update(r=asyncio.run(scenario())), daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert result.get("r") == "stopped", "worker loop starved the event loop"


class _FakeRedis:
    def __init__(self, groups: dict[str, list[dict[str, Any]]]) -> None:
        self.groups = groups

    async def xinfo_groups(self, topic: str) -> list[dict[str, Any]]:
        return self.groups[topic]


class TestBrokerLag:
    def test_redis_lag_is_undelivered_plus_pending(self) -> None:
        consumer = RedisStreamConsumer(group_id="g")
        consumer._subscribed_topics = ["a", "b"]
        consumer._redis = _FakeRedis(  # type: ignore[assignment, unused-ignore]
            {
                "a": [
                    {"name": "other", "lag": 100, "pending": 9},
                    {"name": "g", "lag": 4, "pending": 2},
                ],
                "b": [{"name": "g", "lag": 0, "pending": 1}],
            }
        )
        assert _run(consumer.lag()) == 7

    @pytest.mark.parametrize(
        "groups",
        [
            {"a": [{"name": "g", "lag": None, "pending": 2}]},  # pre-7.0 / undeterminable
            {"a": [{"name": "someone-else", "lag": 3, "pending": 0}]},  # group missing
        ],
        ids=["lag-unknown", "group-missing"],
    )
    def test_redis_lag_unknown(self, groups: dict[str, list[dict[str, Any]]]) -> None:
        consumer = RedisStreamConsumer(group_id="g")
        consumer._subscribed_topics = ["a"]
        consumer._redis = _FakeRedis(groups)  # type: ignore[assignment, unused-ignore]
        assert _run(consumer.lag()) is None

    def test_kafka_lag_covers_every_partition_from_committed_offsets(self) -> None:
        aiokafka = pytest.importorskip("aiokafka")
        TopicPartition = aiokafka.TopicPartition

        class _FakeKafka:
            def subscription(self) -> set[str]:
                return {"t"}

            def partitions_for_topic(self, topic: str) -> set[int]:
                return {0, 1, 2}

            async def end_offsets(self, tps: list[Any]) -> dict[Any, int]:
                return dict.fromkeys(tps, 100)

            async def beginning_offsets(self, tps: list[Any]) -> dict[Any, int]:
                return dict.fromkeys(tps, 20)

            async def committed(self, tp: Any) -> int | None:
                # p0 committed at 90; p1 never committed (counts from the
                # beginning offset); p2 committed below retention (clamped).
                return {0: 90, 1: None, 2: 5}[tp.partition]

        consumer = KafkaStreamConsumer(group_id="g")
        consumer._consumer = _FakeKafka()  # type: ignore[assignment, unused-ignore]
        assert isinstance(TopicPartition("t", 0), tuple)
        # (100-90) + (100-20) + (100-20)
        assert _run(consumer.lag()) == 170


def test_stream_worker_metrics_endpoint_exports_engine_series() -> None:
    """`mercury stream` serves the omni_* registry next to its pipeline counters.

    Starts the real worker (memory backend) and scrapes its /metrics: the
    engine HPA's omni_detection_queue_depth and the alert rules' omni_detection_*
    series must be on the endpoint Prometheus scrapes, not only in-process.
    """
    import os
    import signal
    import socket
    import subprocess
    import sys
    import time
    import urllib.request

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "MERCURY_METRICS_HOST": "127.0.0.1"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omni_mercury_engine.cli",
            "stream",
            "--backend",
            "memory",
            "--stats-interval",
            "0",
            "--metrics-port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        body = ""
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as resp:
                    body = resp.read().decode()
                if "omni_detection_queue_depth 0.0" in body:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        assert "mercury_stream_messages_processed" in body
        assert "omni_detection_queue_depth 0.0" in body
        assert "# TYPE omni_detection_requests_total counter" in body
        assert "# TYPE omni_model_inference_errors_total counter" in body
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
