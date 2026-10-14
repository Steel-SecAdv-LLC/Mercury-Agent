# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Platform-native delivery (on the wire) and TLP-gated alert egress."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web
from aiohttp.test_utils import TestServer

from omni_mercury_engine.api import alert_egress
from omni_mercury_engine.api.routes.export import DetectionRecord
from omni_mercury_engine.integrations.cross_platform_hub import (
    AnomalyEvent,
    CrossPlatformHub,
    DataTransformer,
    HTTPPlatformAdapter,
    PlatformConfig,
    PlatformType,
    PrometheusAdapter,
    merge_prometheus_exposition,
)


class _Recorder:
    """Loopback HTTP server recording every request."""

    def __init__(self, bulk_errors: int = 0) -> None:
        self.requests: list[dict[str, Any]] = []
        self.bulk_errors = bulk_errors

    async def handle(self, request: web.Request) -> web.Response:
        body = await request.read()
        self.requests.append(
            {
                "method": request.method,
                "path": request.path,
                "headers": dict(request.headers),
                "body": body.decode(),
            }
        )
        if request.path == "/_bulk":
            n = body.decode().count('{"index"')
            items = [
                (
                    {"index": {"status": 400, "error": {"type": "x"}}}
                    if i < self.bulk_errors
                    else {"index": {"status": 201}}
                )
                for i in range(n)
            ]
            return web.json_response({"errors": self.bulk_errors > 0, "items": items})
        return web.json_response({"text": "Success", "code": 0})


async def _server(recorder: _Recorder) -> TestServer:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", recorder.handle)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    return server


def _event(score: float = 0.8, source: str = "mercury-agent", **labels: str) -> AnomalyEvent:
    return AnomalyEvent(
        event_id=f"e{score}",
        timestamp=datetime.now(UTC),
        source=source,
        severity="high",
        score=score,
        is_anomaly=True,
        labels=dict(labels),
    )


def _config(platform: PlatformType, endpoint: str, **kw: Any) -> PlatformConfig:
    return PlatformConfig(platform_type=platform, name=platform.value, endpoint=endpoint, **kw)


async def test_splunk_hec_path_auth_and_batch() -> None:
    rec = _Recorder()
    server = await _server(rec)
    try:
        adapter = HTTPPlatformAdapter(
            _config(
                PlatformType.SPLUNK,
                str(server.make_url("")).rstrip("/"),
                auth_type="api_key",
                credentials={"api_key": "hec-token"},
                metadata={"index": "security"},
            )
        )
        assert await adapter.connect()
        assert await adapter.send_event(_event())
        assert await adapter.send_batch([_event(0.7), _event(0.9)]) == 2
        await adapter.disconnect()
    finally:
        await server.close()
    paths = [r["path"] for r in rec.requests]
    assert paths == [
        "/services/collector/health",
        "/services/collector/event",
        "/services/collector/event",
    ]
    assert rec.requests[1]["headers"]["Authorization"] == "Splunk hec-token"
    single = json.loads(rec.requests[1]["body"])
    assert single["index"] == "security"
    assert single["sourcetype"] == "mercury:anomaly"
    batch = [json.loads(line) for line in rec.requests[2]["body"].splitlines()]
    assert [b["event"]["score"] for b in batch] == [0.7, 0.9]


async def test_datadog_series_endpoint_and_key_header() -> None:
    rec = _Recorder()
    server = await _server(rec)
    try:
        adapter = HTTPPlatformAdapter(
            _config(
                PlatformType.DATADOG,
                str(server.make_url("")).rstrip("/"),
                auth_type="api_key",
                credentials={"api_key": "dd-key"},
            )
        )
        assert await adapter.send_batch([_event(0.6), _event(0.95)]) == 2
        await adapter.disconnect()
    finally:
        await server.close()
    post = rec.requests[-1]
    assert post["path"] == "/api/v1/series"
    assert post["headers"]["DD-API-KEY"] == "dd-key"
    assert "Authorization" not in post["headers"]
    assert len(json.loads(post["body"])["series"]) == 4  # 2 metrics x 2 events


async def test_elastic_doc_bulk_and_partial_failure_count() -> None:
    rec = _Recorder(bulk_errors=1)
    server = await _server(rec)
    try:
        adapter = HTTPPlatformAdapter(
            _config(
                PlatformType.ELASTIC,
                str(server.make_url("")).rstrip("/"),
                auth_type="api_key",
                credentials={"api_key": "es-key"},
                metadata={"index": "alerts"},
            )
        )
        assert await adapter.send_event(_event())
        assert await adapter.send_batch([_event(0.5), _event(0.6), _event(0.7)]) == 2
        await adapter.disconnect()
    finally:
        await server.close()
    doc, bulk = rec.requests[-2], rec.requests[-1]
    assert doc["path"] == "/alerts/_doc"
    assert doc["headers"]["Authorization"] == "ApiKey es-key"
    assert isinstance(json.loads(doc["body"])["event"]["severity"], int)
    assert bulk["path"] == "/_bulk"
    assert bulk["headers"]["Content-Type"].startswith("application/x-ndjson")
    assert bulk["body"].count('{"index": {"_index": "alerts"}}') == 3


async def test_generic_receiver_keeps_bearer_contract() -> None:
    rec = _Recorder()
    server = await _server(rec)
    try:
        adapter = HTTPPlatformAdapter(
            _config(
                PlatformType.CUSTOM,
                str(server.make_url("")).rstrip("/"),
                auth_type="api_key",
                credentials={"api_key": "k"},
            )
        )
        assert await adapter.send_event(_event())
        await adapter.disconnect()
    finally:
        await server.close()
    assert rec.requests[-1]["path"] == "/api/events"
    assert rec.requests[-1]["headers"]["Authorization"] == "Bearer k"


async def test_prometheus_push_is_one_valid_exposition() -> None:
    from prometheus_client.parser import text_string_to_metric_families

    rec = _Recorder()
    server = await _server(rec)
    try:
        adapter = PrometheusAdapter(
            _config(PlatformType.PROMETHEUS, str(server.make_url("")).rstrip("/"))
        )
        events = [_event(0.5, zone='a"b\\c'), _event(0.6, zone='a"b\\c'), _event(0.7, zone="z")]
        assert await adapter.send_batch(events) == 3
    finally:
        await server.close()
    body = rec.requests[-1]["body"]
    assert body.count("# TYPE mercury_anomaly_score gauge") == 1
    families = {f.name: f for f in text_string_to_metric_families(body)}
    score = families["mercury_anomaly_score"]
    # Same label set -> last value wins; a different zone is its own series.
    values = sorted((s.labels["zone"], s.value) for s in score.samples)
    assert values == [('a"b\\c', 0.6), ("z", 0.7)]


def test_merge_keeps_last_sample_and_single_metadata() -> None:
    merged = merge_prometheus_exposition(
        [DataTransformer.to_prometheus(_event(0.1)), DataTransformer.to_prometheus(_event(0.2))]
    )
    assert merged.count("# HELP mercury_anomaly_score") == 1
    assert "mercury_anomaly_score{" in merged
    assert " 0.2" in merged and " 0.1\n" not in merged


async def test_batch_publish_follows_routes() -> None:
    sent: dict[str, list[str]] = {}

    class _Capture(HTTPPlatformAdapter):
        async def send_batch(self, events: list[AnomalyEvent]) -> int:
            sent.setdefault(self.config.name, []).extend(e.source for e in events)
            return len(events)

    hub = CrossPlatformHub()
    for name in ("soc", "ops"):
        hub.register_platform(
            name, _config(PlatformType.CUSTOM, "http://127.0.0.1:1"), adapter_class=_Capture
        )
        hub._adapters[name].config.name = name
    hub.add_route("security*", ["soc"])
    hub.add_route("infra*", ["ops"])
    result = await hub.publish_batch(
        [_event(source="security:ids"), _event(source="infra:disk"), _event(source="other")]
    )
    assert sorted(sent["soc"]) == ["other", "security:ids"]
    assert sorted(sent["ops"]) == ["infra:disk", "other"]
    assert result == {"soc": 2, "ops": 2}


def test_correlation_honours_min_correlation() -> None:
    hub = CrossPlatformHub()
    t0 = datetime.now(UTC)
    a = _event(source="s", site="x")
    b = _event(source="s", site="x")
    c = _event(source="s", site="y")
    a.timestamp, b.timestamp, c.timestamp = t0, t0 + timedelta(seconds=5), t0 + timedelta(seconds=9)
    hub._event_buffer = [a, b, c]
    assert hub.get_correlated_events(60.0, min_correlation=0.9) == [[a, b]]
    assert hub.get_correlated_events(60.0, min_correlation=0.5) == [[a, b, c]]


# ---------------------------------------------------------------------------
# Alert egress
# ---------------------------------------------------------------------------


def _record(score: float, anomalies: int = 1, **metadata: Any) -> DetectionRecord:
    return DetectionRecord(
        detection_id=f"d{score}",
        timestamp=datetime.now(),
        user_id="someone@example.org",
        method="statistical",
        data_hash="h",
        anomaly_count=anomalies,
        max_score=score,
        is_batch=False,
        sensitivity=0.5,
        metadata=metadata,
    )


def test_sink_config_validation() -> None:
    assert alert_egress.load_sink_configs(None) == []
    with pytest.raises(ValueError, match="JSON"):
        alert_egress.load_sink_configs("{")
    with pytest.raises(ValueError, match="platform"):
        alert_egress.load_sink_configs('[{"name": "a", "endpoint": "x", "platform": "nope"}]')
    with pytest.raises(ValueError, match="max_tlp"):
        alert_egress.load_sink_configs('[{"name": "a", "endpoint": "x", "max_tlp": "BLUE"}]')
    with pytest.raises(ValueError, match="duplicate"):
        alert_egress.load_sink_configs(
            '[{"name": "a", "endpoint": "x"}, {"name": "a", "endpoint": "y"}]'
        )
    (sink,) = alert_egress.load_sink_configs(
        '[{"name": "a", "endpoint": "http://h/", "max_tlp": "tlp:amber"}]'
    )
    assert sink.endpoint == "http://h"
    assert sink.max_tlp.value == "AMBER"


async def test_alert_egress_is_tlp_gated_and_carries_no_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder()
    server = await _server(rec)
    base = str(server.make_url("")).rstrip("/")
    monkeypatch.setenv("SOC_TOKEN", "tok")
    sinks = alert_egress.load_sink_configs(
        json.dumps(
            [
                {
                    "name": "soc",
                    "platform": "splunk",
                    "endpoint": base,
                    "auth_type": "api_key",
                    "credentials_env": {"api_key": "SOC_TOKEN"},
                    "max_tlp": "RED",
                },
                {"name": "public", "platform": "custom", "endpoint": base, "max_tlp": "GREEN"},
            ]
        )
    )
    egress = alert_egress.AlertEgress(sinks)
    try:
        red = await egress.dispatch(_record(0.95))
        green = await egress.dispatch(_record(0.4))
        nothing = await egress.dispatch(_record(0.9, anomalies=0))
        await egress.shutdown()
    finally:
        await server.close()
    assert red == {"soc": True}  # RED withheld from the GREEN-capped sink
    assert green == {"soc": True, "public": True}
    assert nothing == {}
    assert egress.stats["withheld_tlp"] == 1
    assert egress.stats["delivered"] == 3
    posted = [r for r in rec.requests if r["method"] == "POST"]
    assert all("someone@example.org" not in r["body"] for r in posted)
    soc_posts = [r for r in posted if r["path"] == "/services/collector/event"]
    assert soc_posts[0]["headers"]["Authorization"] == "Splunk tok"
    assert json.loads(soc_posts[0]["body"])["fields"]["tlp"] == "TLP:RED"


async def test_offline_refusal_is_counted_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCURY_OFFLINE", "1")
    sinks = alert_egress.load_sink_configs(
        '[{"name": "ext", "platform": "custom", "endpoint": "https://siem.example.com"}]'
    )
    egress = alert_egress.AlertEgress(sinks)
    assert await egress.dispatch(_record(0.6)) == {"ext": False}
    assert egress.stats["suppressed_offline"] == 1


async def test_record_detection_schedules_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    from omni_mercury_engine.api.routes import export as export_module

    seen: list[str] = []

    async def _fake_dispatch(self: Any, record: DetectionRecord) -> dict[str, bool]:
        seen.append(record.method)
        return {}

    monkeypatch.setenv(
        alert_egress.ALERT_SINKS_ENV,
        '[{"name": "x", "platform": "custom", "endpoint": "http://127.0.0.1:9"}]',
    )
    monkeypatch.setattr(alert_egress.AlertEgress, "dispatch", _fake_dispatch)
    monkeypatch.setattr(export_module, "_data_store", export_module.DataStore())
    alert_egress.reset_alert_egress()
    try:
        await export_module.record_detection(
            "u", "flagged", [[1.0]], {"anomalies": [True], "scores": [0.9]}
        )
        await export_module.record_detection(
            "u", "clean", [[1.0]], {"anomalies": [False], "scores": [0.1]}
        )
        await alert_egress.shutdown_alert_egress()
    finally:
        alert_egress.reset_alert_egress()
    assert seen == ["flagged"]


def test_alert_carries_the_request_trace_id() -> None:
    from omni_mercury_engine.api.trace_context import continue_or_start, trace_context_ctx

    egress = alert_egress.AlertEgress(
        alert_egress.load_sink_configs(
            '[{"name": "x", "platform": "custom", "endpoint": "http://127.0.0.1:9"}]'
        )
    )
    trace = continue_or_start("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
    token = trace_context_ctx.set(trace)
    try:
        event, _ = egress.build_event(_record(0.8))
    finally:
        trace_context_ctx.reset(token)
    assert event.labels["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert "trace_id" not in egress.build_event(_record(0.8))[0].labels
