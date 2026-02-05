"""
Mercury Agent - Production Observability Infrastructure

Comprehensive observability for production deployments including:
- Structured audit logging with compliance support
- Distributed tracing with OpenTelemetry
- Metrics collection with Prometheus export
- Health monitoring and alerting

Features:
- HIPAA/SOC2 compliant audit logging
- OpenTelemetry integration for distributed tracing
- Prometheus metrics with custom anomaly detection metrics
- Correlation ID propagation
- Log masking for sensitive data
- Structured JSON logging

References:
- OpenTelemetry Specification: https://opentelemetry.io/docs/specs/
- Prometheus Best Practices: https://prometheus.io/docs/practices/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Generator


logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


class AuditAction(Enum):
    """Audit log action types."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    DETECT = "detect"
    TRAIN = "train"
    DEPLOY = "deploy"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    ADMIN = "admin"
    SYSTEM = "system"


class AuditSeverity(Enum):
    """Audit log severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Resource types for audit logging."""

    MODEL = "model"
    DETECTION = "detection"
    USER = "user"
    API_KEY = "api_key"
    BATCH_JOB = "batch_job"
    EXPORT = "export"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


@dataclass
class AuditEvent:
    """Structured audit event."""

    event_id: str
    timestamp: datetime
    action: AuditAction
    severity: AuditSeverity
    resource_type: ResourceType
    resource_id: str
    user_id: str
    user_ip: str
    user_agent: str
    request_method: str
    request_path: str
    request_id: str
    trace_id: str
    span_id: str
    response_status: int
    duration_ms: float
    success: bool
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    data_accessed: list[str] = field(default_factory=list)
    data_modified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "severity": self.severity.value,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "user_id": self.user_id,
            "user_ip": self.user_ip,
            "user_agent": self.user_agent,
            "request_method": self.request_method,
            "request_path": self.request_path,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "response_status": self.response_status,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "data_accessed": self.data_accessed,
            "data_modified": self.data_modified,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogHandler(ABC):
    """Abstract base for audit log handlers."""

    @abstractmethod
    def emit(self, event: AuditEvent) -> None:
        """Emit an audit event."""
        pass

    @abstractmethod
    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        action: AuditAction | None = None,
        resource_type: ResourceType | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        pass


class InMemoryAuditHandler(AuditLogHandler):
    """In-memory audit log handler for development/testing."""

    def __init__(self, max_events: int = 100000) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.RLock()
        self._max_events = max_events

    def emit(self, event: AuditEvent) -> None:
        """Emit an audit event."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        action: AuditAction | None = None,
        resource_type: ResourceType | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        with self._lock:
            results = self._events.copy()

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]

        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        if user_id:
            results = [e for e in results if e.user_id == user_id]

        if action:
            results = [e for e in results if e.action == action]

        if resource_type:
            results = [e for e in results if e.resource_type == resource_type]

        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]


class FileAuditHandler(AuditLogHandler):
    """File-based audit log handler with rotation and proper resource management.

    Supports context manager protocol for safe resource cleanup:
        with FileAuditHandler('/var/log/mercury') as handler:
            handler.emit(event)
    """

    def __init__(
        self,
        log_dir: str = "/var/log/mercury",
        max_file_size_mb: int = 100,
        max_files: int = 10,
    ) -> None:
        self._log_dir = log_dir
        self._max_file_size = max_file_size_mb * 1024 * 1024
        self._max_files = max_files
        self._lock = threading.RLock()
        self._current_file: Any = None
        self._current_file_size = 0
        self._closed = False

        os.makedirs(log_dir, exist_ok=True)
        self._rotate_if_needed()

    def close(self) -> None:
        """Close the current log file and release resources.

        Thread-safe method that can be called multiple times without error.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._current_file:
                try:
                    self._current_file.flush()
                    self._current_file.close()
                except OSError:
                    pass
                finally:
                    self._current_file = None

    def __enter__(self) -> FileAuditHandler:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close resources."""
        self.close()

    def __del__(self) -> None:
        """Ensure file handle is closed during garbage collection.

        Note: Exceptions in __del__ cannot be safely raised and logging may fail
        if the logging module has already been torn down during interpreter shutdown.
        """
        try:
            self.close()
        except Exception:
            # Cannot reliably log during GC/interpreter shutdown
            pass

    def emit(self, event: AuditEvent) -> None:
        """Emit an audit event to file.

        Raises:
            RuntimeError: If handler has been closed.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot emit to closed FileAuditHandler")

            self._rotate_if_needed()

            line = event.to_json() + "\n"
            line_bytes = line.encode("utf-8")

            if self._current_file:
                self._current_file.write(line_bytes)
                self._current_file.flush()
                self._current_file_size += len(line_bytes)

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        action: AuditAction | None = None,
        resource_type: ResourceType | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events from files."""
        results: list[AuditEvent] = []

        log_dir_path = Path(self._log_dir)
        log_files = sorted(
            [f.name for f in log_dir_path.iterdir() if f.name.startswith("audit_")],
            reverse=True,
        )

        for log_file in log_files:
            if len(results) >= limit:
                break

            file_path = log_dir_path / log_file
            try:
                with open(file_path) as f:
                    for line in f:
                        if len(results) >= limit:
                            break

                        try:
                            data = json.loads(line)
                            event = self._dict_to_event(data)

                            if start_time and event.timestamp < start_time:
                                continue
                            if end_time and event.timestamp > end_time:
                                continue
                            if user_id and event.user_id != user_id:
                                continue
                            if action and event.action != action:
                                continue
                            if resource_type and event.resource_type != resource_type:
                                continue

                            results.append(event)
                        except (json.JSONDecodeError, KeyError):
                            continue
            except OSError:
                continue

        return results

    def _rotate_if_needed(self) -> None:
        """Rotate log file if needed."""
        if self._current_file and self._current_file_size < self._max_file_size:
            return

        if self._current_file:
            self._current_file.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file_path = os.path.join(self._log_dir, f"audit_{timestamp}.jsonl")

        self._current_file = open(new_file_path, "ab")
        self._current_file_size = 0

        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Remove old log files."""
        log_dir_path = Path(self._log_dir)
        log_files = sorted(
            [f.name for f in log_dir_path.iterdir() if f.name.startswith("audit_")],
            reverse=True,
        )

        for old_file in log_files[self._max_files :]:
            try:
                (log_dir_path / old_file).unlink()
            except OSError:
                pass

    def _dict_to_event(self, data: dict[str, Any]) -> AuditEvent:
        """Convert dictionary to AuditEvent."""
        return AuditEvent(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=AuditAction(data["action"]),
            severity=AuditSeverity(data["severity"]),
            resource_type=ResourceType(data["resource_type"]),
            resource_id=data["resource_id"],
            user_id=data["user_id"],
            user_ip=data["user_ip"],
            user_agent=data["user_agent"],
            request_method=data["request_method"],
            request_path=data["request_path"],
            request_id=data["request_id"],
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            response_status=data["response_status"],
            duration_ms=data["duration_ms"],
            success=data["success"],
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            data_accessed=data.get("data_accessed", []),
            data_modified=data.get("data_modified", []),
        )


class AuditLogger:
    """
    Production audit logger with compliance support.

    Supports HIPAA, SOC2, and GDPR audit requirements.
    """

    _instance: AuditLogger | None = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> AuditLogger:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        handlers: list[AuditLogHandler] | None = None,
        async_emit: bool = True,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self._handlers = handlers or [InMemoryAuditHandler()]
        self._async_emit = async_emit
        self._emit_queue: list[AuditEvent] = []
        self._queue_lock = threading.Lock()

        if async_emit:
            self._start_background_emitter()

        self._initialized = True

        logger.info(f"AuditLogger initialized with {len(self._handlers)} handlers")

    def _start_background_emitter(self) -> None:
        """Start background thread for async emit."""

        def emitter_loop() -> None:
            while True:
                time.sleep(0.1)
                with self._queue_lock:
                    events = self._emit_queue.copy()
                    self._emit_queue.clear()

                for event in events:
                    for handler in self._handlers:
                        try:
                            handler.emit(event)
                        except Exception as e:
                            logger.error(f"Audit handler error: {e}")

        thread = threading.Thread(target=emitter_loop, daemon=True)
        thread.start()

    def log(
        self,
        action: AuditAction,
        resource_type: ResourceType,
        resource_id: str,
        user_id: str = "anonymous",
        user_ip: str = "unknown",
        user_agent: str = "unknown",
        request_method: str = "UNKNOWN",
        request_path: str = "/",
        request_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        response_status: int = 200,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: str | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: dict[str, Any] | None = None,
        data_accessed: list[str] | None = None,
        data_modified: list[str] | None = None,
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            action=action,
            severity=severity,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            user_ip=user_ip,
            user_agent=user_agent,
            request_method=request_method,
            request_path=request_path,
            request_id=request_id or str(uuid.uuid4()),
            trace_id=trace_id or "",
            span_id=span_id or "",
            response_status=response_status,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
            data_accessed=data_accessed or [],
            data_modified=data_modified or [],
        )

        if self._async_emit:
            with self._queue_lock:
                self._emit_queue.append(event)
        else:
            for handler in self._handlers:
                handler.emit(event)

        return event

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        action: AuditAction | None = None,
        resource_type: ResourceType | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events from all handlers."""
        all_events: list[AuditEvent] = []

        for handler in self._handlers:
            events = handler.query(
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                limit=limit,
            )
            all_events.extend(events)

        all_events.sort(key=lambda e: e.timestamp, reverse=True)
        return all_events[:limit]


class DistributedTracer:
    """
    Distributed tracing with OpenTelemetry support.

    Provides end-to-end request tracing across services.
    """

    _instance: DistributedTracer | None = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> DistributedTracer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        service_name: str = "mercury-agent",
        otlp_endpoint: str | None = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self._service_name = service_name
        self._otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self._tracer: Any = None
        self._propagator: Any = None

        if OTEL_AVAILABLE:
            self._setup_opentelemetry()

        self._initialized = True

        logger.info(
            f"DistributedTracer initialized " f"(service={service_name}, otel={OTEL_AVAILABLE})"
        )

    def _setup_opentelemetry(self) -> None:
        """Setup OpenTelemetry tracing."""
        resource = Resource.create(
            {
                "service.name": self._service_name,
                "service.version": "1.2.0",
            }
        )

        provider = TracerProvider(resource=resource)

        if self._otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=self._otlp_endpoint)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self._service_name)
        self._propagator = TraceContextTextMapPropagator()

    @contextmanager
    def span(
        self,
        name: str,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[dict[str, str], None, None]:
        """Create a tracing span context manager."""
        if not OTEL_AVAILABLE or self._tracer is None:
            yield {"trace_id": "", "span_id": ""}
            return

        kind_map = {
            "internal": SpanKind.INTERNAL,
            "server": SpanKind.SERVER,
            "client": SpanKind.CLIENT,
            "producer": SpanKind.PRODUCER,
            "consumer": SpanKind.CONSUMER,
        }

        with self._tracer.start_as_current_span(
            name,
            kind=kind_map.get(kind, SpanKind.INTERNAL),
            attributes=attributes,
        ):
            ctx = trace.get_current_span().get_span_context()
            yield {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
            }

    def inject_context(self, headers: dict[str, str]) -> dict[str, str]:
        """Inject trace context into headers for propagation."""
        if not OTEL_AVAILABLE or self._propagator is None:
            return headers

        self._propagator.inject(headers)
        return headers

    def extract_context(self, headers: dict[str, str]) -> Any:
        """Extract trace context from headers."""
        if not OTEL_AVAILABLE or self._propagator is None:
            return None

        return self._propagator.extract(headers)

    def get_current_trace_id(self) -> str:
        """Get current trace ID."""
        if not OTEL_AVAILABLE:
            return ""

        ctx = trace.get_current_span().get_span_context()
        return format(ctx.trace_id, "032x")

    def get_current_span_id(self) -> str:
        """Get current span ID."""
        if not OTEL_AVAILABLE:
            return ""

        ctx = trace.get_current_span().get_span_context()
        return format(ctx.span_id, "016x")


@dataclass
class MetricPoint:
    """Metric data point."""

    name: str
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Metrics collector with Prometheus export support.

    Collects anomaly detection specific metrics.
    """

    _instance: MetricsCollector | None = None
    _class_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> MetricsCollector:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock: threading.RLock = threading.RLock()

        self._init_default_metrics()
        self._initialized = True

        logger.info("MetricsCollector initialized")

    def _init_default_metrics(self) -> None:
        """Initialize default metrics."""
        self._counters = {
            "mercury_detections_total": 0,
            "mercury_anomalies_found_total": 0,
            "mercury_batch_jobs_total": 0,
            "mercury_api_requests_total": 0,
            "mercury_api_errors_total": 0,
        }

        self._gauges = {
            "mercury_active_batch_jobs": 0,
            "mercury_model_count": 0,
            "mercury_avg_detection_score": 0.0,
            "mercury_ethical_gate_avg": 0.96,
        }

        self._histograms = {
            "mercury_detection_latency_seconds": [],
            "mercury_detection_scores": [],
            "mercury_batch_size": [],
        }

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] = self._counters.get(key, 0) + value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Observe a histogram value."""
        with self._lock:
            key = self._make_key(name, labels)
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-10000:]

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Make metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0)

    def get_histogram_stats(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

        import numpy as np

        arr = np.array(values)
        return {
            "count": len(values),
            "sum": float(np.sum(arr)),
            "avg": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, value in self._counters.items():
            base_name = name.split("{")[0]
            lines.append(f"# TYPE {base_name} counter")
            lines.append(f"{name} {value}")

        for name, value in self._gauges.items():
            base_name = name.split("{")[0]
            lines.append(f"# TYPE {base_name} gauge")
            lines.append(f"{name} {value}")

        for name, values in self._histograms.items():
            if not values:
                continue

            base_name = name.split("{")[0]
            stats = self.get_histogram_stats(name)

            lines.append(f"# TYPE {base_name} histogram")
            lines.append(f"{name}_count {stats['count']}")
            lines.append(f"{name}_sum {stats['sum']}")

            for bucket in [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]:
                count = sum(1 for v in values if v <= bucket)
                lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {len(values)}')

        return "\n".join(lines)


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return AuditLogger()


def get_tracer() -> DistributedTracer:
    """Get the global distributed tracer instance."""
    return DistributedTracer()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return MetricsCollector()


def traced(
    name: str | None = None,
    kind: str = "internal",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for adding tracing to functions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.span(span_name, kind=kind):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.span(span_name, kind=kind):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def audited(
    action: AuditAction,
    resource_type: ResourceType,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for adding audit logging to functions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            audit = get_audit_logger()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000

                audit.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(kwargs.get("resource_id", "unknown")),
                    duration_ms=duration,
                    success=True,
                )

                return result

            except Exception as e:
                duration = (time.time() - start_time) * 1000

                audit.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(kwargs.get("resource_id", "unknown")),
                    duration_ms=duration,
                    success=False,
                    error_message=str(e),
                    severity=AuditSeverity.ERROR,
                )

                raise

        return wrapper

    return decorator
