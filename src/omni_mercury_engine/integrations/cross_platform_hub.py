# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-Platform Anomaly Detection Hub.

Multi-Cross Platform Anomaly Detection Hub providing:
- Unified adapter pattern for external platforms
- Protocol-agnostic data ingestion (REST, gRPC, WebSocket, MQTT)
- Standard output formats (JSON, Prometheus, OpenTelemetry, CSV)
- Platform connectors for: Netdata, Elastic, Splunk, Azure, Datadog
- Multi-platform orchestration and routing
- Real-time data transformation pipelines
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine._version import __version__

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


def _enforce_offline_egress(endpoint: str) -> None:
    """Air-gap gate shared by every adapter transport in this module.

    The platform adapters keep their own aiohttp transports (they speak
    platform-native protocols ``SafeHTTPClient`` does not), so each
    egress method calls this before touching its session.  Under
    ``MERCURY_OFFLINE`` any non-loopback platform endpoint raises
    ``OfflineModeError`` pre-socket; a loopback endpoint (an on-box
    Prometheus pushgateway or OTLP collector sidecar) stays reachable.
    The gate runs *outside* the adapters' catch-all error handling so
    the refusal propagates loudly out of the adapter instead of
    degrading into a return value that reads as a transport blip. One
    layer up, ``CrossPlatformHub.publish_event`` / ``publish_batch``
    keep their per-platform isolation contract but catch
    ``OfflineModeError`` distinctly, recording the refusal as an
    explicit MERCURY_OFFLINE suppression -- never as a generic publish
    failure.
    """
    # Lazy import -- the codebase-wide pattern -- so this module never
    # pulls the security package's crypto surface at import time.
    from omni_mercury_engine.security.safe_http import enforce_offline_egress

    enforce_offline_egress(endpoint)


class PlatformType(StrEnum):
    """Supported external platform types."""

    NETDATA = "netdata"
    ELASTIC = "elastic"
    SPLUNK = "splunk"
    AZURE_ANOMALY = "azure_anomaly"
    DATADOG = "datadog"
    PROMETHEUS = "prometheus"
    INFLUXDB = "influxdb"
    GRAFANA = "grafana"
    OPENSEARCH = "opensearch"
    OPENTELEMETRY = "opentelemetry"
    CUSTOM = "custom"


class DataFormat(StrEnum):
    """Supported data formats."""

    JSON = "json"
    PROMETHEUS = "prometheus"
    OPENTELEMETRY = "opentelemetry"
    CSV = "csv"
    MSGPACK = "msgpack"
    AVRO = "avro"
    PARQUET = "parquet"


class ProtocolType(StrEnum):
    """Supported communication protocols."""

    REST = "rest"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    KAFKA = "kafka"
    REDIS_STREAM = "redis_stream"


@dataclass
class PlatformConfig:
    """Configuration for external platform connection."""

    platform_type: PlatformType
    name: str
    endpoint: str
    protocol: ProtocolType = ProtocolType.REST
    auth_type: str = "none"  # none, api_key, oauth2, basic
    credentials: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    ssl_verify: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyEvent:
    """Standardized anomaly event for cross-platform communication."""

    event_id: str
    timestamp: datetime
    source: str
    severity: str  # low, medium, high, critical
    score: float  # 0.0 - 1.0
    is_anomaly: bool
    dimensions: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    raw_data: Any = None
    detector_type: str = "unknown"
    confidence: float = 0.0
    explanation: str = ""
    related_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "severity": self.severity,
            "score": self.score,
            "is_anomaly": self.is_anomaly,
            "dimensions": self.dimensions,
            "metrics": self.metrics,
            "labels": self.labels,
            "detector_type": self.detector_type,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "related_events": self.related_events,
        }

    @classmethod
    def from_detection_result(
        cls,
        result: dict[str, Any],
        source: str = "mercury-agent",
        index: int = 0,
    ) -> AnomalyEvent:
        """Create AnomalyEvent from detector output."""
        scores = result.get("scores", [0.0])
        is_anomaly_arr = result.get("is_anomaly", [False])

        score = float(scores[index] if hasattr(scores, "__getitem__") else scores)
        is_anomaly = bool(
            is_anomaly_arr[index] if hasattr(is_anomaly_arr, "__getitem__") else is_anomaly_arr
        )

        # Determine severity from score
        if score >= 0.9:
            severity = "critical"
        elif score >= 0.7:
            severity = "high"
        elif score >= 0.5:
            severity = "medium"
        else:
            severity = "low"

        event_id = hashlib.sha3_256(f"{source}:{time.time_ns()}:{index}".encode()).hexdigest()[:16]

        return cls(
            event_id=event_id,
            timestamp=datetime.now(UTC),
            source=source,
            severity=severity,
            score=score,
            is_anomaly=is_anomaly,
            detector_type=result.get("detector_type", "unknown"),
            confidence=result.get("confidence", score),
            metrics=result.get("metrics", {}),
            labels=result.get("labels", {}),
        )


_PROM_METRIC_INVALID = re.compile(r"[^a-zA-Z0-9_:]")
_PROM_LABEL_INVALID = re.compile(r"[^a-zA-Z0-9_]")


def _prom_metric_name(name: str) -> str:
    """Sanitise a Prometheus metric name (``[a-zA-Z_:][a-zA-Z0-9_:]*``)."""
    cleaned = _PROM_METRIC_INVALID.sub("_", name)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"_{cleaned}"


def _prom_label_name(name: str) -> str:
    """Sanitise a Prometheus label name (``[a-zA-Z_][a-zA-Z0-9_]*``, no ``__``)."""
    cleaned = _PROM_LABEL_INVALID.sub("_", name) or "label"
    while cleaned.startswith("__"):  # the "__" prefix is reserved
        cleaned = cleaned[1:]
    return f"_{cleaned}" if cleaned[0].isdigit() else cleaned


def _prom_label_value(value: Any) -> str:
    """Escape a label value per the text exposition format."""
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def merge_prometheus_exposition(blocks: list[str]) -> str:
    """Merge per-event exposition blocks into one valid push payload.

    Concatenating blocks repeats ``# HELP`` / ``# TYPE`` lines and identical
    series, which the pushgateway rejects. The merge keeps the first
    HELP/TYPE per family, keeps the LAST sample per series (name + label
    set), and emits every family's samples contiguously.
    """
    meta: dict[str, list[str]] = {}
    samples: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for block in blocks:
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.split(None, 3)
                if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                    family = parts[2]
                    kinds = [m.split(None, 2)[1] for m in meta.get(family, [])]
                    if parts[1] not in kinds:
                        meta.setdefault(family, []).append(line)
                    if family not in samples:
                        samples[family] = {}
                        order.append(family)
                continue
            series, _, value = line.rpartition(" ")
            family = series.split("{", 1)[0]
            if family not in samples:
                samples[family] = {}
                order.append(family)
            samples[family][series] = value
    out: list[str] = []
    for family in order:
        out.extend(sorted(meta.get(family, []), key=lambda m: m.split(None, 2)[1] != "HELP"))
        out.extend(f"{series} {value}" for series, value in samples[family].items())
    return "\n".join(out) + "\n"


class DataTransformer:
    """Transform data between different formats."""

    @staticmethod
    def to_prometheus(event: AnomalyEvent) -> str:
        """Convert anomaly event to Prometheus metrics format.

        Label names are sanitised and values escaped; custom metric names
        are sanitised. Merge several events with
        :func:`merge_prometheus_exposition` before pushing.
        """
        labels = ",".join(
            f'{_prom_label_name(k)}="{_prom_label_value(v)}"'
            for k, v in {
                "source": event.source,
                "severity": event.severity,
                "detector": event.detector_type,
                **event.labels,
            }.items()
        )

        lines = [
            "# HELP mercury_anomaly_score Anomaly score from Mercury Agent",
            "# TYPE mercury_anomaly_score gauge",
            f"mercury_anomaly_score{{{labels}}} {event.score}",
            "# HELP mercury_anomaly_detected Whether anomaly was detected",
            "# TYPE mercury_anomaly_detected gauge",
            f"mercury_anomaly_detected{{{labels}}} {1 if event.is_anomaly else 0}",
            "# HELP mercury_anomaly_confidence Detection confidence",
            "# TYPE mercury_anomaly_confidence gauge",
            f"mercury_anomaly_confidence{{{labels}}} {event.confidence}",
        ]

        # Add custom metrics
        for metric_name, metric_value in event.metrics.items():
            safe_name = _prom_metric_name(f"mercury_{metric_name}")
            lines.append(f"{safe_name}{{{labels}}} {metric_value}")

        return "\n".join(lines)

    @staticmethod
    def to_opentelemetry(event: AnomalyEvent) -> dict[str, Any]:
        """Convert anomaly event to OpenTelemetry format."""
        return {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": event.source}},
                            {"key": "service.version", "value": {"stringValue": __version__}},
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "scope": {"name": "mercury-agent"},
                            "metrics": [
                                {
                                    "name": "mercury.anomaly.score",
                                    "description": "Anomaly detection score",
                                    "gauge": {
                                        "dataPoints": [
                                            {
                                                "asDouble": event.score,
                                                "timeUnixNano": int(
                                                    event.timestamp.timestamp() * 1e9
                                                ),
                                                "attributes": [
                                                    {"key": k, "value": {"stringValue": str(v)}}
                                                    for k, v in event.labels.items()
                                                ],
                                            }
                                        ]
                                    },
                                },
                                {
                                    "name": "mercury.anomaly.detected",
                                    "description": "Anomaly detection flag",
                                    "gauge": {
                                        "dataPoints": [
                                            {
                                                "asInt": 1 if event.is_anomaly else 0,
                                                "timeUnixNano": int(
                                                    event.timestamp.timestamp() * 1e9
                                                ),
                                            }
                                        ]
                                    },
                                },
                            ],
                        }
                    ],
                }
            ]
        }

    @staticmethod
    def to_elastic(event: AnomalyEvent) -> dict[str, Any]:
        """Convert anomaly event to Elasticsearch format."""
        return {
            "@timestamp": event.timestamp.isoformat(),
            "event": {
                "id": event.event_id,
                "kind": "alert" if event.is_anomaly else "metric",
                "category": ["anomaly_detection"],
                "type": ["indicator"],
                # ECS: event.severity is a number (long); the name goes in
                # mercury.severity so the mapping never conflicts.
                "severity": round(float(event.score) * 100),
                "risk_score": float(event.score * 100),
            },
            "source": {"component": event.source},
            "mercury": {
                "score": event.score,
                "severity": event.severity,
                "is_anomaly": event.is_anomaly,
                "detector_type": event.detector_type,
                "confidence": event.confidence,
                "explanation": event.explanation,
            },
            "labels": event.labels,
            "metrics": event.metrics,
            "dimensions": event.dimensions,
        }

    @staticmethod
    def to_splunk(
        event: AnomalyEvent, index: str | None = None, sourcetype: str = "mercury:anomaly"
    ) -> dict[str, Any]:
        """Convert anomaly event to Splunk HEC format.

        ``index`` is included only when given: HEC tokens are commonly
        restricted to specific indexes, and a hard-coded ``main`` is
        rejected by such tokens.
        """
        payload: dict[str, Any] = {
            "time": event.timestamp.timestamp(),
            "event": {
                "event_id": event.event_id,
                "severity": event.severity,
                "score": event.score,
                "is_anomaly": event.is_anomaly,
                "detector_type": event.detector_type,
                "confidence": event.confidence,
                "explanation": event.explanation,
                **event.dimensions,
                **event.metrics,
            },
            "source": event.source,
            "sourcetype": sourcetype,
            "fields": {k: str(v) for k, v in event.labels.items()},
        }
        if index:
            payload["index"] = index
        return payload

    @staticmethod
    def to_datadog(event: AnomalyEvent) -> dict[str, Any]:
        """Convert anomaly event to Datadog format."""
        tags = [f"{k}:{v}" for k, v in event.labels.items()]
        tags.extend(
            [
                f"source:{event.source}",
                f"severity:{event.severity}",
                f"detector:{event.detector_type}",
            ]
        )

        return {
            "series": [
                {
                    "metric": "mercury.anomaly.score",
                    "type": "gauge",
                    "points": [[int(event.timestamp.timestamp()), event.score]],
                    "tags": tags,
                },
                {
                    "metric": "mercury.anomaly.detected",
                    "type": "gauge",
                    "points": [[int(event.timestamp.timestamp()), 1 if event.is_anomaly else 0]],
                    "tags": tags,
                },
            ]
        }

    @staticmethod
    def to_csv_row(event: AnomalyEvent) -> dict[str, Any]:
        """Convert anomaly event to CSV row."""
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "severity": event.severity,
            "score": event.score,
            "is_anomaly": event.is_anomaly,
            "detector_type": event.detector_type,
            "confidence": event.confidence,
            "explanation": event.explanation,
            **{f"label_{k}": v for k, v in event.labels.items()},
            **{f"metric_{k}": v for k, v in event.metrics.items()},
        }


class PlatformAdapter(ABC):
    """Abstract base class for platform adapters."""

    def __init__(self, config: PlatformConfig):
        """Initialize the instance."""
        self.config = config
        self._connected = False
        self._last_error: str | None = None

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to platform."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from platform."""
        pass

    @abstractmethod
    async def send_event(self, event: AnomalyEvent) -> bool:
        """Send anomaly event to platform."""
        pass

    @abstractmethod
    async def send_batch(self, events: list[AnomalyEvent]) -> int:
        """Send batch of events.

        Returns count of successful sends.
        """
        pass

    @abstractmethod
    def fetch_data(
        self,
        query: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Fetch data from platform."""
        ...

    @property
    def is_connected(self) -> bool:
        """Is connected."""
        return self._connected

    @property
    def last_error(self) -> str | None:
        """Last error."""
        return self._last_error


#: Platform-native ingest paths: (single event, batch, health probe).
_SPLUNK_PATHS: tuple[str, str, str] = (
    "/services/collector/event",
    "/services/collector/event",
    "/services/collector/health",
)
_DATADOG_PATHS: tuple[str, str, str] = ("/api/v1/series", "/api/v1/series", "/api/v1/validate")
_GENERIC_PATHS: tuple[str, str, str] = ("/api/events", "/api/events/batch", "/health")


class HTTPPlatformAdapter(PlatformAdapter):
    """HTTP platform adapter speaking each platform's native ingest API.

    * Splunk: HEC ``/services/collector/event`` with ``Authorization:
      Splunk <token>``; a batch is newline-separated HEC events in one POST.
    * Datadog: ``/api/v1/series`` with the ``DD-API-KEY`` header; a batch is
      one ``series`` list.
    * Elastic / OpenSearch: ``/<index>/_doc`` (batch: ``/_bulk`` NDJSON,
      per-item errors counted) with ``Authorization: ApiKey`` or Basic.
    * Anything else: the generic ``/api/events`` (+ ``/batch``) receiver
      contract with Bearer auth.

    ``config.metadata`` may set ``index`` (Splunk / Elastic) and
    ``sourcetype`` (Splunk).
    """

    def __init__(self, config: PlatformConfig):
        """Initialize the instance."""
        super().__init__(config)
        self._session: Any = None

    @property
    def _is_elastic(self) -> bool:
        return self.config.platform_type in (PlatformType.ELASTIC, PlatformType.OPENSEARCH)

    def _paths(self) -> tuple[str, str, str]:
        """(single event, batch, health) paths for the configured platform."""
        if self.config.platform_type == PlatformType.SPLUNK:
            return _SPLUNK_PATHS
        if self.config.platform_type == PlatformType.DATADOG:
            return _DATADOG_PATHS
        if self._is_elastic:
            index = str(self.config.metadata.get("index", "mercury-anomalies"))
            return (f"/{index}/_doc", "/_bulk", "/")
        return _GENERIC_PATHS

    def _auth_headers(self) -> dict[str, str]:
        """Platform-native authentication headers."""
        creds = self.config.credentials
        if self.config.auth_type == "basic":
            import base64

            token = f"{creds.get('username', '')}:{creds.get('password', '')}"
            return {"Authorization": f"Basic {base64.b64encode(token.encode()).decode()}"}
        if self.config.auth_type != "api_key":
            return {}
        api_key = creds.get("api_key", "")
        if self.config.platform_type == PlatformType.SPLUNK:
            return {"Authorization": f"Splunk {api_key}"}
        if self.config.platform_type == PlatformType.DATADOG:
            headers = {"DD-API-KEY": api_key}
            if creds.get("app_key"):
                headers["DD-APPLICATION-KEY"] = creds["app_key"]
            return headers
        if self._is_elastic:
            return {"Authorization": f"ApiKey {api_key}"}
        return {"Authorization": f"Bearer {api_key}"}

    def _batch_body(self, events: list[AnomalyEvent]) -> tuple[bytes, str]:
        """Encode a batch in the platform's native batch format."""
        if self.config.platform_type == PlatformType.SPLUNK:
            body = "\n".join(json.dumps(self._transform_event(e), default=str) for e in events)
            return body.encode(), "application/json"
        if self.config.platform_type == PlatformType.DATADOG:
            series = [s for e in events for s in DataTransformer.to_datadog(e)["series"]]
            return json.dumps({"series": series}, default=str).encode(), "application/json"
        if self._is_elastic:
            index = str(self.config.metadata.get("index", "mercury-anomalies"))
            lines: list[str] = []
            for e in events:
                lines.append(json.dumps({"index": {"_index": index}}))
                lines.append(json.dumps(self._transform_event(e), default=str))
            return ("\n".join(lines) + "\n").encode(), "application/x-ndjson"
        payload = {"events": [self._transform_event(e) for e in events]}
        return json.dumps(payload, default=str).encode(), "application/json"

    async def connect(self) -> bool:
        """Establish HTTP session.

        Raises:
            OfflineModeError: If ``MERCURY_OFFLINE`` is set and the
                configured endpoint is not a loopback host.
        """
        _enforce_offline_egress(self.config.endpoint)
        try:
            import aiohttp

            connector = aiohttp.TCPConnector(
                ssl=self.config.ssl_verify,
                limit=100,
            )

            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

            headers = {**self.config.headers, **self._auth_headers()}

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
            )

            # Test connection against the platform's own health/validate
            # endpoint; only the generic receiver may lack one (404).
            healthy = (200, 404) if self._paths() == _GENERIC_PATHS else (200,)
            health_path = self._paths()[2]
            async with self._session.get(f"{self.config.endpoint}{health_path}") as resp:
                self._connected = resp.status in healthy

        except ImportError:
            # This adapter's only transport IS aiohttp — there is no fallback.
            # Reporting connected here would make send_event() drop every event
            # (None session → error per event) while status shows healthy.
            logger.error("aiohttp not installed; HTTP integration adapter cannot deliver events")
            self._connected = False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Connection failed: {e}")
            self._connected = False
            # Don't leak the ClientSession opened above if the health probe raised.
            if self._session is not None:
                await self._session.close()
                self._session = None

        return self._connected

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def send_event(self, event: AnomalyEvent) -> bool:
        """Send single event via HTTP POST."""
        # Gate every send, not just connect(): the flag is dynamic, so a
        # session opened while online must not keep sending air-gapped.
        _enforce_offline_egress(self.config.endpoint)
        if not self._session:
            await self.connect()

        try:
            if self.config.platform_type == PlatformType.DATADOG:
                body, content_type = self._batch_body([event])
            else:
                body = json.dumps(self._transform_event(event), default=str).encode()
                content_type = "application/json"

            async with self._session.post(
                f"{self.config.endpoint}{self._paths()[0]}",
                data=body,
                headers={"Content-Type": content_type},
            ) as resp:
                return resp.status in (200, 201, 202)

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to send event: {e}")
            return False

    async def send_batch(self, events: list[AnomalyEvent]) -> int:
        """Send batch of events."""
        _enforce_offline_egress(self.config.endpoint)
        if not self._session:
            await self.connect()

        if not events:
            return 0
        try:
            body, content_type = self._batch_body(events)

            async with self._session.post(
                f"{self.config.endpoint}{self._paths()[1]}",
                data=body,
                headers={"Content-Type": content_type},
            ) as resp:
                if resp.status not in (200, 201, 202):
                    return 0
                if self._is_elastic:
                    # _bulk answers 200 even when items fail; count successes.
                    result = await resp.json()
                    items = result.get("items", [])
                    failed = sum(1 for item in items if next(iter(item.values())).get("error"))
                    return len(events) - failed
                return len(events)

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to send batch: {e}")
            return 0

    async def fetch_data(
        self,
        query: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Fetch data via HTTP GET."""
        _enforce_offline_egress(self.config.endpoint)
        if not self._session:
            await self.connect()

        try:
            async with self._session.get(
                f"{self.config.endpoint}/api/data",
                params=query,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data.get("results", []):
                        yield item

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to fetch data: {e}")

    def _transform_event(self, event: AnomalyEvent) -> dict[str, Any]:
        """Transform event based on platform type."""
        if self._is_elastic:
            return DataTransformer.to_elastic(event)
        elif self.config.platform_type == PlatformType.SPLUNK:
            index = self.config.metadata.get("index")
            sourcetype = str(self.config.metadata.get("sourcetype", "mercury:anomaly"))
            return DataTransformer.to_splunk(
                event, index=str(index) if index else None, sourcetype=sourcetype
            )
        elif self.config.platform_type == PlatformType.DATADOG:
            return DataTransformer.to_datadog(event)
        else:
            return event.to_dict()


class PrometheusAdapter(PlatformAdapter):
    """Prometheus push gateway adapter."""

    def __init__(self, config: PlatformConfig):
        """Initialize the instance."""
        super().__init__(config)
        self._metrics_buffer: list[str] = []

    async def connect(self) -> bool:
        """Prometheus uses push model, no persistent connection needed.

        Raises:
            OfflineModeError: If ``MERCURY_OFFLINE`` is set and the configured
                pushgateway is not a loopback host -- an adapter that can never
                deliver must not report itself connected/healthy.
        """
        _enforce_offline_egress(self.config.endpoint)
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Flush remaining metrics.

        Cleanup must never raise: an offline refusal from the final flush is
        logged (the buffered metrics are retained for inspection) and the
        adapter still transitions to disconnected.
        """
        if self._metrics_buffer:
            from omni_mercury_engine.datasets.exceptions import OfflineModeError

            try:
                await self._flush_metrics()
            except OfflineModeError as e:
                logger.warning(
                    "Final metrics flush suppressed on disconnect: %s "
                    "(%d buffered metric(s) retained)",
                    e,
                    len(self._metrics_buffer),
                )
        self._connected = False

    async def send_event(self, event: AnomalyEvent) -> bool:
        """Send event as Prometheus metrics."""
        metrics = DataTransformer.to_prometheus(event)
        self._metrics_buffer.append(metrics)

        if len(self._metrics_buffer) >= 100:
            return await self._flush_metrics()

        return True

    async def send_batch(self, events: list[AnomalyEvent]) -> int:
        """Send batch as Prometheus metrics."""
        for event in events:
            metrics = DataTransformer.to_prometheus(event)
            self._metrics_buffer.append(metrics)

        if await self._flush_metrics():
            return len(events)
        return 0

    async def _flush_metrics(self) -> bool:
        """Push metrics to Prometheus pushgateway."""
        if not self._metrics_buffer:
            return True

        # A loopback pushgateway sidecar keeps working air-gapped; any
        # external gateway is refused loudly before a socket opens.
        _enforce_offline_egress(self.config.endpoint)

        try:
            import aiohttp

            payload = merge_prometheus_exposition(self._metrics_buffer)

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.config.endpoint}/metrics/job/mercury_agent",
                    data=payload.encode(),
                    headers={"Content-Type": "text/plain; version=0.0.4"},
                ) as resp,
            ):
                success = resp.status in (200, 202)
                if success:
                    self._metrics_buffer.clear()
                return success

        except ImportError:
            # Nothing was pushed: keep the metrics and report the failure
            # (clearing them and returning True masked data loss as success).
            logger.error("aiohttp not installed; Prometheus metrics were not pushed")
            return False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to push metrics: {e}")
            return False

    async def fetch_data(
        self,
        query: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Query Prometheus for data."""
        _enforce_offline_egress(self.config.endpoint)
        try:
            import aiohttp

            promql = query.get("query", "mercury_anomaly_score")

            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"{self.config.endpoint}/api/v1/query",
                    params={"query": promql},
                ) as resp,
            ):
                if resp.status == 200:
                    data = await resp.json()
                    for result in data.get("data", {}).get("result", []):
                        yield {
                            "metric": result.get("metric", {}),
                            "value": result.get("value", [0, 0])[1],
                            "timestamp": result.get("value", [0, 0])[0],
                        }

        except ImportError:
            logger.warning("aiohttp not installed")
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to query Prometheus: {e}")


class OpenTelemetryAdapter(PlatformAdapter):
    """OpenTelemetry collector adapter."""

    def __init__(self, config: PlatformConfig):
        """Initialize the instance."""
        super().__init__(config)
        self._exporter: Any = None

    async def connect(self) -> bool:
        """Initialize OpenTelemetry exporter.

        Raises:
            OfflineModeError: If ``MERCURY_OFFLINE`` is set and the configured
                collector is not a loopback host -- an adapter whose every
                delivery would be refused must not report itself connected.
        """
        _enforce_offline_egress(self.config.endpoint)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            self._exporter = OTLPMetricExporter(
                endpoint=self.config.endpoint,
                insecure=not self.config.ssl_verify,
            )

            self._connected = True

        except ImportError:
            # Primary OTLP gRPC exporter is unavailable. send_event() has an
            # HTTP fallback, but that path needs aiohttp — only claim connected
            # if that transport actually exists, otherwise this adapter cannot
            # deliver and must not report healthy.
            import importlib.util

            if importlib.util.find_spec("aiohttp") is not None:
                logger.warning("OpenTelemetry SDK not installed; using aiohttp HTTP fallback")
                self._connected = True
            else:
                logger.error(
                    "OpenTelemetry SDK and aiohttp both unavailable; "
                    "OTLP adapter cannot deliver events"
                )
                self._connected = False
        except Exception as e:
            self._last_error = str(e)
            self._connected = False

        return self._connected

    async def disconnect(self) -> None:
        """Shutdown OpenTelemetry exporter."""
        if self._exporter and hasattr(self._exporter, "shutdown"):
            self._exporter.shutdown()
        self._connected = False

    async def send_event(self, event: AnomalyEvent) -> bool:
        """Send event via OpenTelemetry."""
        # A loopback OTLP collector sidecar keeps working air-gapped;
        # any external collector is refused loudly before a socket opens.
        _enforce_offline_egress(self.config.endpoint)
        try:
            import aiohttp

            data = DataTransformer.to_opentelemetry(event)

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.config.endpoint}/v1/metrics",
                    json=data,
                    headers={"Content-Type": "application/json"},
                ) as resp,
            ):
                return resp.status in (200, 202)

        except ImportError:
            # No aiohttp → nothing was sent. Returning True here would make
            # send_batch() count this event as delivered (data loss masked as
            # success). Report the failure transparently.
            logger.error("aiohttp not installed; OTLP event was not delivered")
            return False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to send OTLP event: {e}")
            return False

    async def send_batch(self, events: list[AnomalyEvent]) -> int:
        """Send batch via OpenTelemetry."""
        success_count = 0
        for event in events:
            if await self.send_event(event):
                success_count += 1
        return success_count

    async def fetch_data(
        self,
        query: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Opentelemetry is push-only, no fetch support."""
        logger.warning("OpenTelemetry adapter does not support data fetching")
        return
        yield  # Make this a generator


class CrossPlatformHub:
    """Central hub for cross-platform anomaly detection integration.

    Features:
    - Multi-platform event routing
    - Format transformation
    - Load balancing across platforms
    - Fault tolerance with retry logic
    - Event correlation across platforms
    """

    def __init__(
        self,
        default_format: DataFormat = DataFormat.JSON,
        enable_correlation: bool = True,
        buffer_size: int = 1000,
    ):
        """Initialize the cross-platform hub.

        Args:
            default_format: Default output format
            enable_correlation: Enable event correlation
            buffer_size: Internal event buffer size
        """
        self.default_format = default_format
        self.enable_correlation = enable_correlation
        self.buffer_size = buffer_size

        self._adapters: dict[str, PlatformAdapter] = {}
        self._routes: dict[str, list[str]] = {}  # source -> [platform_names]
        self._event_buffer: list[AnomalyEvent] = []
        self._correlation_window: dict[str, list[AnomalyEvent]] = {}
        self._lock = asyncio.Lock()
        self._running = False

    def register_platform(
        self,
        name: str,
        config: PlatformConfig,
        adapter_class: type[PlatformAdapter] | None = None,
    ) -> None:
        """Register a platform adapter.

        Args:
            name: Unique platform name
            config: Platform configuration
            adapter_class: Optional custom adapter class
        """
        if adapter_class:
            adapter = adapter_class(config)
        elif config.platform_type == PlatformType.PROMETHEUS:
            adapter = PrometheusAdapter(config)
        elif config.platform_type == PlatformType.OPENTELEMETRY:
            adapter = OpenTelemetryAdapter(config)
        elif config.protocol == ProtocolType.REST:
            adapter = HTTPPlatformAdapter(config)
        else:
            adapter = HTTPPlatformAdapter(config)

        self._adapters[name] = adapter
        logger.info(f"Registered platform: {name} ({config.platform_type})")

    def add_route(
        self,
        source_pattern: str,
        platform_names: list[str],
    ) -> None:
        """Add routing rule for events.

        Args:
            source_pattern: Source pattern to match (supports wildcards)
            platform_names: List of platform names to route to
        """
        self._routes[source_pattern] = platform_names

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all registered platforms."""
        results = {}
        for name, adapter in self._adapters.items():
            try:
                results[name] = await adapter.connect()
            except Exception as e:
                logger.error(f"Failed to connect {name}: {e}")
                results[name] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all platforms."""
        for adapter in self._adapters.values():
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect: {e}")

    def _match_routes(self, source: str) -> list[str]:
        """Find platforms matching source pattern."""
        matched = []
        for pattern, platforms in self._routes.items():
            if pattern == "*" or source.startswith(pattern.rstrip("*")):
                matched.extend(platforms)
        return list(set(matched))

    async def publish_event(
        self,
        event: AnomalyEvent,
        platforms: list[str] | None = None,
    ) -> dict[str, bool]:
        """Publish anomaly event to platforms.

        Args:
            event: Anomaly event to publish
            platforms: Specific platforms (None = use routing rules)

        Returns:
            Dictionary of platform -> success status
        """
        target_platforms = platforms or self._match_routes(event.source)

        if not target_platforms:
            # Publish to all if no routes defined
            target_platforms = list(self._adapters.keys())

        results: dict[str, bool] = {}
        names = [name for name in target_platforms if name in self._adapters]

        from omni_mercury_engine.datasets.exceptions import OfflineModeError

        # Platforms are independent: send concurrently, isolate failures.
        outcomes = await asyncio.gather(
            *(self._adapters[name].send_event(event) for name in names),
            return_exceptions=True,
        )
        for platform_name, outcome in zip(names, outcomes, strict=True):
            if isinstance(outcome, OfflineModeError):
                # Per-platform isolation is the hub's contract, but the
                # air-gap refusal is recorded as such -- never as a generic
                # publish failure.
                logger.warning("Publish to %s suppressed: %s", platform_name, outcome)
                results[platform_name] = False
            elif isinstance(outcome, BaseException):
                logger.error(f"Failed to publish to {platform_name}: {outcome}")
                results[platform_name] = False
            else:
                results[platform_name] = bool(outcome)

        # Store for correlation
        if self.enable_correlation:
            async with self._lock:
                self._event_buffer.append(event)
                if len(self._event_buffer) > self.buffer_size:
                    self._event_buffer.pop(0)

        return results

    async def publish_batch(
        self,
        events: list[AnomalyEvent],
        platforms: list[str] | None = None,
    ) -> dict[str, int]:
        """Publish batch of events.

        Args:
            events: List of anomaly events
            platforms: Target platforms. ``None`` applies the routing rules
                per event (as :meth:`publish_event` does), falling back to
                every platform for an event no route matches.

        Returns:
            Dictionary of platform -> successful count
        """
        per_platform: dict[str, list[AnomalyEvent]] = {}
        for event in events:
            targets = platforms or self._match_routes(event.source) or list(self._adapters)
            for name in targets:
                if name in self._adapters:
                    per_platform.setdefault(name, []).append(event)

        from omni_mercury_engine.datasets.exceptions import OfflineModeError

        names = list(per_platform)
        outcomes = await asyncio.gather(
            *(self._adapters[name].send_batch(per_platform[name]) for name in names),
            return_exceptions=True,
        )
        results: dict[str, int] = {}
        for platform_name, outcome in zip(names, outcomes, strict=True):
            if isinstance(outcome, OfflineModeError):
                logger.warning("Batch publish to %s suppressed: %s", platform_name, outcome)
                results[platform_name] = 0
            elif isinstance(outcome, BaseException):
                logger.error(f"Failed batch publish to {platform_name}: {outcome}")
                results[platform_name] = 0
            else:
                results[platform_name] = int(outcome)
        return results

    async def publish_detection_result(
        self,
        result: dict[str, Any],
        source: str = "mercury-agent",
        platforms: list[str] | None = None,
    ) -> dict[str, bool | int]:
        """Publish detector result, converting to standard events.

        Args:
            result: Detection result from Mercury detectors
            source: Source identifier
            platforms: Target platforms

        Returns:
            Publication results per platform
        """
        # Extract arrays from result
        scores = np.asarray(result.get("scores", [0.0]))
        is_anomaly = np.asarray(result.get("is_anomaly", [False]))

        # Create events for anomalies or all points
        events = []
        for i in range(len(scores)):
            if is_anomaly[i]:  # Only create events for anomalies
                event = AnomalyEvent.from_detection_result(result, source, i)
                events.append(event)

        if not events:
            return {}

        if len(events) == 1:
            single_event_result = await self.publish_event(events[0], platforms)
            return {k: v for k, v in single_event_result.items()}
        else:
            return await self.publish_batch(events, platforms)

    async def fetch_from_platform(
        self,
        platform_name: str,
        query: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fetch data from a specific platform.

        Args:
            platform_name: Platform to fetch from
            query: Query parameters

        Returns:
            List of data points
        """
        adapter = self._adapters.get(platform_name)
        if not adapter:
            raise ValueError(f"Unknown platform: {platform_name}")

        results = []
        async for item in adapter.fetch_data(query):
            results.append(item)

        return results

    def get_correlated_events(
        self,
        time_window_seconds: float = 60.0,
        min_correlation: float = 0.8,
    ) -> list[list[AnomalyEvent]]:
        """Find correlated anomaly events within time window.

        Two events correlate when they fall within ``time_window_seconds``
        of each other AND the Jaccard similarity of their attribute sets
        (source, detector type, severity, and every label / dimension
        key=value) reaches ``min_correlation``. Each event joins the first
        open group whose most recent member it correlates with.

        Args:
            time_window_seconds: Time window for correlation
            min_correlation: Minimum attribute similarity in ``[0, 1]``

        Returns:
            List of correlated event groups (two or more events each)
        """
        if not self._event_buffer:
            return []

        def _attributes(event: AnomalyEvent) -> set[str]:
            items = {
                f"source={event.source}",
                f"detector={event.detector_type}",
                f"severity={event.severity}",
            }
            items.update(f"label:{k}={v}" for k, v in event.labels.items())
            items.update(f"dim:{k}={v}" for k, v in event.dimensions.items())
            return items

        def _similarity(a: AnomalyEvent, b: AnomalyEvent) -> float:
            sa, sb = _attributes(a), _attributes(b)
            return len(sa & sb) / len(sa | sb)

        groups: list[list[AnomalyEvent]] = []
        for event in sorted(self._event_buffer, key=lambda e: e.timestamp):
            for group in groups:
                last = group[-1]
                within = (event.timestamp - last.timestamp).total_seconds() <= time_window_seconds
                if within and _similarity(event, last) >= min_correlation:
                    group.append(event)
                    break
            else:
                groups.append([event])

        return [group for group in groups if len(group) > 1]

    def get_platform_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered platforms."""
        status = {}
        for name, adapter in self._adapters.items():
            status[name] = {
                "connected": adapter.is_connected,
                "platform_type": adapter.config.platform_type,
                "endpoint": adapter.config.endpoint,
                "last_error": adapter.last_error,
            }
        return status

    def get_statistics(self) -> dict[str, Any]:
        """Get hub statistics."""
        return {
            "registered_platforms": len(self._adapters),
            "active_routes": len(self._routes),
            "buffered_events": len(self._event_buffer),
            "buffer_size": self.buffer_size,
            "platforms": list(self._adapters.keys()),
        }


def create_default_hub() -> CrossPlatformHub:
    """Create a hub with common platform configurations.

    Returns:
        Configured CrossPlatformHub instance
    """
    # No routes: publish_event / publish_batch send every event to every
    # registered platform until routes are added.
    return CrossPlatformHub()


# Exports
__all__ = [
    "AnomalyEvent",
    "CrossPlatformHub",
    "DataFormat",
    "DataTransformer",
    "HTTPPlatformAdapter",
    "OpenTelemetryAdapter",
    "PlatformAdapter",
    "PlatformConfig",
    "PlatformType",
    "PrometheusAdapter",
    "ProtocolType",
    "create_default_hub",
    "merge_prometheus_exposition",
]
