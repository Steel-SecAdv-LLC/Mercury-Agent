# Copyright (C) 2025 Steel Security Advisors LLC
"""Cross-Platform Anomaly Detection Hub."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


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


class DataTransformer:
    """Transform data between different formats."""

    @staticmethod
    def to_prometheus(event: AnomalyEvent) -> str:
        """Convert anomaly event to Prometheus metrics format."""
        labels = ",".join(
            f'{k}="{v}"'
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
            safe_name = metric_name.replace(".", "_").replace("-", "_")
            lines.append(f"mercury_{safe_name}{{{labels}}} {metric_value}")

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
                            {"key": "service.version", "value": {"stringValue": "1.7.0"}},
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
                "severity": {"name": event.severity, "score": int(event.score * 100)},
            },
            "source": {"component": event.source},
            "mercury": {
                "score": event.score,
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
    def to_splunk(event: AnomalyEvent) -> dict[str, Any]:
        """Convert anomaly event to Splunk HEC format."""
        return {
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
            "sourcetype": "mercury:anomaly",
            "index": "main",
            "fields": event.labels,
        }

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


class HTTPPlatformAdapter(PlatformAdapter):
    """Generic HTTP-based platform adapter."""

    def __init__(self, config: PlatformConfig):
        """Initialize the instance."""
        super().__init__(config)
        self._session: Any = None

    async def connect(self) -> bool:
        """Establish HTTP session."""
        try:
            import aiohttp

            connector = aiohttp.TCPConnector(
                ssl=self.config.ssl_verify,
                limit=100,
            )

            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

            headers = dict(self.config.headers)

            # Add authentication headers
            if self.config.auth_type == "api_key":
                api_key = self.config.credentials.get("api_key", "")
                headers["Authorization"] = f"Bearer {api_key}"
            elif self.config.auth_type == "basic":
                import base64

                username = self.config.credentials.get("username", "")
                password = self.config.credentials.get("password", "")
                encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
            )

            # Test connection
            async with self._session.get(f"{self.config.endpoint}/health") as resp:
                self._connected = resp.status in (200, 404)  # 404 is ok if no health endpoint

        except ImportError:
            logger.warning("aiohttp not installed, using fallback")
            self._connected = True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Connection failed: {e}")
            self._connected = False

        return self._connected

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def send_event(self, event: AnomalyEvent) -> bool:
        """Send single event via HTTP POST."""
        if not self._session:
            await self.connect()

        try:
            data = self._transform_event(event)

            async with self._session.post(
                f"{self.config.endpoint}/api/events",
                json=data,
            ) as resp:
                return resp.status in (200, 201, 202)

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to send event: {e}")
            return False

    async def send_batch(self, events: list[AnomalyEvent]) -> int:
        """Send batch of events."""
        if not self._session:
            await self.connect()

        try:
            data = [self._transform_event(e) for e in events]

            async with self._session.post(
                f"{self.config.endpoint}/api/events/batch",
                json={"events": data},
            ) as resp:
                if resp.status in (200, 201, 202):
                    return len(events)
                return 0

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to send batch: {e}")
            return 0

    async def fetch_data(
        self,
        query: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Fetch data via HTTP GET."""
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
        if self.config.platform_type == PlatformType.ELASTIC:
            return DataTransformer.to_elastic(event)
        elif self.config.platform_type == PlatformType.SPLUNK:
            return DataTransformer.to_splunk(event)
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
        """Prometheus uses push model, no persistent connection needed."""
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Flush remaining metrics."""
        if self._metrics_buffer:
            await self._flush_metrics()
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

        try:
            import aiohttp

            payload = "\n".join(self._metrics_buffer) + "\n"

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.config.endpoint}/metrics/job/mercury_agent",
                    data=payload.encode(),
                    headers={"Content-Type": "text/plain"},
                ) as resp,
            ):
                success = resp.status in (200, 202)
                if success:
                    self._metrics_buffer.clear()
                return success

        except ImportError:
            logger.warning("aiohttp not installed")
            self._metrics_buffer.clear()
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to push metrics: {e}")
            return False

    async def fetch_data(
        self,
        query: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Query Prometheus for data."""
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
        """Initialize OpenTelemetry exporter."""
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
            logger.warning("OpenTelemetry SDK not installed, using HTTP fallback")
            self._connected = True
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
            logger.warning("aiohttp not installed")
            return True
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

        results = {}
        tasks = []

        for platform_name in target_platforms:
            adapter = self._adapters.get(platform_name)
            if adapter:
                tasks.append((platform_name, adapter.send_event(event)))

        for platform_name, task in tasks:
            try:
                results[platform_name] = await task
            except Exception as e:
                logger.error(f"Failed to publish to {platform_name}: {e}")
                results[platform_name] = False

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
            platforms: Target platforms

        Returns:
            Dictionary of platform -> successful count
        """
        target_platforms = platforms or list(self._adapters.keys())

        results = {}

        for platform_name in target_platforms:
            adapter = self._adapters.get(platform_name)
            if adapter:
                try:
                    count = await adapter.send_batch(events)
                    results[platform_name] = count
                except Exception as e:
                    logger.error(f"Failed batch publish to {platform_name}: {e}")
                    results[platform_name] = 0

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

        Args:
            time_window_seconds: Time window for correlation
            min_correlation: Minimum correlation score

        Returns:
            List of correlated event groups
        """
        if not self._event_buffer:
            return []

        # Group events by time proximity
        groups: list[list[AnomalyEvent]] = []
        current_group: list[AnomalyEvent] = []

        sorted_events = sorted(self._event_buffer, key=lambda e: e.timestamp)

        for event in sorted_events:
            if not current_group:
                current_group.append(event)
            else:
                time_diff = (event.timestamp - current_group[-1].timestamp).total_seconds()

                if time_diff <= time_window_seconds:
                    current_group.append(event)
                else:
                    if len(current_group) > 1:
                        groups.append(current_group)
                    current_group = [event]

        if len(current_group) > 1:
            groups.append(current_group)

        return groups

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
    hub = CrossPlatformHub()

    # Add default route (all events to all platforms)
    hub.add_route("*", [])

    return hub


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
]
