# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent - Data Export API Endpoints.

Production-grade data export for historical anomaly detections, audit logs,
and analytics data. Supports multiple formats and streaming for large datasets.

Features:
- Historical detection export with filtering
- Audit log export for compliance
- Multiple export formats (JSON, CSV, Parquet)
- Streaming support for large datasets
- Scheduled export jobs
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, Permission, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["Data Export"])


class ExportFormat(StrEnum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"  # JSON Lines for streaming
    PARQUET = "parquet"


class ExportType(StrEnum):
    """Types of data to export."""

    DETECTIONS = "detections"
    AUDIT_LOGS = "audit_logs"
    METRICS = "metrics"
    MODELS = "models"
    JOBS = "jobs"


class ExportStatus(StrEnum):
    """Export job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class DetectionRecord:
    """Historical detection record."""

    detection_id: str
    timestamp: datetime
    user_id: str
    method: str
    data_hash: str
    anomaly_count: int
    max_score: float
    is_batch: bool
    sensitivity: float
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLogRecord:
    """Audit log record for compliance."""

    log_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    ip_address: str
    user_agent: str
    request_method: str
    request_path: str
    response_status: int
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportJob:
    """Export job record."""

    job_id: str
    user_id: str
    export_type: ExportType
    format: ExportFormat
    status: ExportStatus
    filters: dict[str, Any]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    file_path: str | None = None
    file_size_bytes: int = 0
    record_count: int = 0
    error_message: str | None = None
    expires_at: datetime | None = None


class DataStore:
    """In-memory data store for demonstration.

    Production deployment should use a proper time-series database (InfluxDB, TimescaleDB) or data
    warehouse (BigQuery, Snowflake).
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self._detections: list[DetectionRecord] = []
        self._audit_logs: list[AuditLogRecord] = []
        self._export_jobs: dict[str, ExportJob] = {}
        self._lock = asyncio.Lock()
        self._max_records = int(os.getenv("MAX_EXPORT_RECORDS", "1000000"))

    async def add_detection(self, record: DetectionRecord) -> None:
        """Add a detection record."""
        async with self._lock:
            self._detections.append(record)
            if len(self._detections) > self._max_records:
                self._detections = self._detections[-self._max_records :]

    async def add_audit_log(self, record: AuditLogRecord) -> None:
        """Add an audit log record."""
        async with self._lock:
            self._audit_logs.append(record)
            if len(self._audit_logs) > self._max_records:
                self._audit_logs = self._audit_logs[-self._max_records :]

    async def query_detections(
        self,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        method: str | None = None,
        min_score: float | None = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> tuple[list[DetectionRecord], int]:
        """Query detection records with filtering."""
        records = self._detections.copy()

        if user_id:
            records = [r for r in records if r.user_id == user_id]

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]

        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        if method:
            records = [r for r in records if r.method == method]

        if min_score is not None:
            records = [r for r in records if r.max_score >= min_score]

        records.sort(key=lambda r: r.timestamp, reverse=True)
        total = len(records)

        return records[offset : offset + limit], total

    async def query_audit_logs(
        self,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> tuple[list[AuditLogRecord], int]:
        """Query audit log records with filtering."""
        records = self._audit_logs.copy()

        if user_id:
            records = [r for r in records if r.user_id == user_id]

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]

        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        if action:
            records = [r for r in records if r.action == action]

        if resource_type:
            records = [r for r in records if r.resource_type == resource_type]

        records.sort(key=lambda r: r.timestamp, reverse=True)
        total = len(records)

        return records[offset : offset + limit], total

    async def create_export_job(
        self,
        user_id: str,
        export_type: ExportType,
        format: ExportFormat,
        filters: dict[str, Any],
    ) -> ExportJob:
        """Create an export job."""
        job_id = str(uuid.uuid4())
        job = ExportJob(
            job_id=job_id,
            user_id=user_id,
            export_type=export_type,
            format=format,
            status=ExportStatus.PENDING,
            filters=filters,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
        )
        async with self._lock:
            self._export_jobs[job_id] = job
        return job

    async def get_export_job(self, job_id: str) -> ExportJob | None:
        """Get an export job."""
        return self._export_jobs.get(job_id)

    async def update_export_job(
        self,
        job_id: str,
        **updates: Any,
    ) -> ExportJob | None:
        """Update an export job."""
        job = self._export_jobs.get(job_id)
        if not job:
            return None

        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)

        return job


_data_store = DataStore()


def get_data_store() -> DataStore:
    """Get the data store instance."""
    return _data_store


async def record_detection(
    user_id: str,
    method: str,
    data: list[Any],
    results: dict[str, Any],
    is_batch: bool = False,
    sensitivity: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a detection for historical tracking."""
    store = get_data_store()

    data_hash = hashlib.sha3_256(json.dumps(data, default=str).encode()).hexdigest()[:16]

    anomaly_count = sum(1 for a in results.get("anomalies", []) if a)
    scores = results.get("scores", [])
    max_score = max(scores) if scores else 0.0

    record = DetectionRecord(
        detection_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        user_id=user_id,
        method=method,
        data_hash=data_hash,
        anomaly_count=anomaly_count,
        max_score=max_score,
        is_batch=is_batch,
        sensitivity=sensitivity,
        summary=results.get("summary", {}),
        metadata=metadata or {},
    )

    await store.add_detection(record)


async def record_audit_log(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    request: Any,
    response_status: int,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record an audit log entry."""
    store = get_data_store()

    ip_address = "unknown"
    user_agent = "unknown"
    request_method = "unknown"
    request_path = "unknown"

    if hasattr(request, "client") and request.client:
        ip_address = request.client.host
    if hasattr(request, "headers"):
        user_agent = request.headers.get("user-agent", "unknown")
    if hasattr(request, "method"):
        request_method = request.method
    if hasattr(request, "url"):
        request_path = str(request.url.path)

    record = AuditLogRecord(
        log_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        user_agent=user_agent,
        request_method=request_method,
        request_path=request_path,
        response_status=response_status,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )

    await store.add_audit_log(record)


class ExportRequest(BaseModel):
    """Request to export data."""

    export_type: ExportType = Field(
        ...,
        description="Type of data to export",
    )
    format: ExportFormat = Field(
        default=ExportFormat.JSON,
        description="Export format",
    )
    start_time: datetime | None = Field(
        default=None,
        description="Start of time range",
    )
    end_time: datetime | None = Field(
        default=None,
        description="End of time range",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Additional filters",
    )
    limit: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum records to export",
    )


class ExportJobResponse(BaseModel):
    """Export job status response."""

    job_id: str
    status: ExportStatus
    export_type: ExportType
    format: ExportFormat
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    record_count: int
    file_size_bytes: int
    error_message: str | None
    expires_at: datetime | None
    download_url: str | None


class ExportSummaryResponse(BaseModel):
    """Summary of available export data."""

    total_detections: int
    total_audit_logs: int
    oldest_detection: datetime | None
    newest_detection: datetime | None
    oldest_audit_log: datetime | None
    newest_audit_log: datetime | None
    available_methods: list[str]
    available_actions: list[str]


def _get_current_user(
    api_key_user: User | None = Depends(APIKeyAuth(auto_error=False)),
    jwt_user: User | None = Depends(JWTAuth(auto_error=False)),
) -> User:
    """Get current authenticated user."""
    user = api_key_user or jwt_user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )
    return user


def _detection_to_dict(record: DetectionRecord) -> dict[str, Any]:
    """Convert detection record to dictionary."""
    return {
        "detection_id": record.detection_id,
        "timestamp": record.timestamp.isoformat(),
        "user_id": record.user_id,
        "method": record.method,
        "data_hash": record.data_hash,
        "anomaly_count": record.anomaly_count,
        "max_score": record.max_score,
        "is_batch": record.is_batch,
        "sensitivity": record.sensitivity,
        "summary": record.summary,
        "metadata": record.metadata,
    }


def _audit_log_to_dict(record: AuditLogRecord) -> dict[str, Any]:
    """Convert audit log record to dictionary."""
    return {
        "log_id": record.log_id,
        "timestamp": record.timestamp.isoformat(),
        "user_id": record.user_id,
        "action": record.action,
        "resource_type": record.resource_type,
        "resource_id": record.resource_id,
        "ip_address": record.ip_address,
        "user_agent": record.user_agent,
        "request_method": record.request_method,
        "request_path": record.request_path,
        "response_status": record.response_status,
        "duration_ms": record.duration_ms,
        "metadata": record.metadata,
    }


async def _stream_json(records: list[Any], to_dict: Any) -> AsyncIterator[bytes]:
    """Stream records as JSON array."""
    yield b"[\n"
    for i, record in enumerate(records):
        data = to_dict(record)
        json_line = json.dumps(data, default=str)
        if i > 0:
            yield b",\n"
        yield json_line.encode()
    yield b"\n]"


async def _stream_jsonl(records: list[Any], to_dict: Any) -> AsyncIterator[bytes]:
    """Stream records as JSON Lines."""
    for record in records:
        data = to_dict(record)
        yield (json.dumps(data, default=str) + "\n").encode()


async def _stream_csv(records: list[Any], to_dict: Any) -> AsyncIterator[bytes]:
    """Stream records as CSV."""
    if not records:
        yield b""
        return

    first = to_dict(records[0])
    headers = list(first.keys())

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    yield output.getvalue().encode()

    for record in records:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        data = to_dict(record)
        for key in data:
            if isinstance(data[key], (dict, list)):
                data[key] = json.dumps(data[key])
        writer.writerow(data)
        yield output.getvalue().encode()


@router.get(
    "/summary",
    response_model=ExportSummaryResponse,
    summary="Get Export Summary",
    description="Get a summary of available data for export.",
)
async def get_export_summary(
    user: User = Depends(_get_current_user),
) -> ExportSummaryResponse:
    """Get summary of exportable data."""
    store = get_data_store()

    is_admin = user.has_permission(Permission.ADMIN)
    user_filter = None if is_admin else user.id

    detections, total_detections = await store.query_detections(
        user_id=user_filter,
        limit=100000,
    )

    audit_logs, total_audit_logs = await store.query_audit_logs(
        user_id=user_filter,
        limit=100000,
    )

    methods = list({d.method for d in detections})
    actions = list({a.action for a in audit_logs})

    return ExportSummaryResponse(
        total_detections=total_detections,
        total_audit_logs=total_audit_logs,
        oldest_detection=min((d.timestamp for d in detections), default=None),
        newest_detection=max((d.timestamp for d in detections), default=None),
        oldest_audit_log=min((a.timestamp for a in audit_logs), default=None),
        newest_audit_log=max((a.timestamp for a in audit_logs), default=None),
        available_methods=sorted(methods),
        available_actions=sorted(actions),
    )


@router.get(
    "/detections",
    summary="Export Detections",
    description="""
Export historical detection results.

Supports streaming for large datasets. Use format parameter to specify output format:
- json: Full JSON array
- jsonl: JSON Lines (one record per line, best for streaming)
- csv: Comma-separated values
""",
)
async def export_detections(
    format: ExportFormat = Query(default=ExportFormat.JSON, description="Export format"),
    start_time: datetime | None = Query(default=None, description="Start of time range"),
    end_time: datetime | None = Query(default=None, description="End of time range"),
    method: str | None = Query(default=None, description="Filter by detection method"),
    min_score: float | None = Query(default=None, ge=0.0, description="Minimum anomaly score"),
    limit: int = Query(default=10000, ge=1, le=100000, description="Maximum records"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    user: User = Depends(_get_current_user),
) -> StreamingResponse:
    """Export detection records."""
    store = get_data_store()

    is_admin = user.has_permission(Permission.ADMIN)
    user_filter = None if is_admin else user.id

    records, total = await store.query_detections(
        user_id=user_filter,
        start_time=start_time,
        end_time=end_time,
        method=method,
        min_score=min_score,
        limit=limit,
        offset=offset,
    )

    logger.info(
        f"Exporting {len(records)} detections for user {user.id} "
        f"(total available: {total}, format: {format.value})"
    )

    content_type_map = {
        ExportFormat.JSON: "application/json",
        ExportFormat.JSONL: "application/x-ndjson",
        ExportFormat.CSV: "text/csv",
    }

    ext_map = {
        ExportFormat.JSON: "json",
        ExportFormat.JSONL: "jsonl",
        ExportFormat.CSV: "csv",
    }

    if format == ExportFormat.JSON:
        stream = _stream_json(records, _detection_to_dict)
    elif format == ExportFormat.JSONL:
        stream = _stream_jsonl(records, _detection_to_dict)
    elif format == ExportFormat.CSV:
        stream = _stream_csv(records, _detection_to_dict)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {format.value}",
        )

    filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext_map[format]}"

    return StreamingResponse(
        stream,
        media_type=content_type_map[format],
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Total-Records": str(total),
            "X-Returned-Records": str(len(records)),
        },
    )


@router.get(
    "/audit-logs",
    summary="Export Audit Logs",
    description="""
Export audit log records for compliance and security analysis.

Requires admin permission to export logs from all users.
Regular users can only export their own audit logs.
""",
)
async def export_audit_logs(
    format: ExportFormat = Query(default=ExportFormat.JSON, description="Export format"),
    start_time: datetime | None = Query(default=None, description="Start of time range"),
    end_time: datetime | None = Query(default=None, description="End of time range"),
    action: str | None = Query(default=None, description="Filter by action"),
    resource_type: str | None = Query(default=None, description="Filter by resource type"),
    limit: int = Query(default=10000, ge=1, le=100000, description="Maximum records"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    user: User = Depends(_get_current_user),
) -> StreamingResponse:
    """Export audit log records."""
    store = get_data_store()

    is_admin = user.has_permission(Permission.ADMIN)
    user_filter = None if is_admin else user.id

    records, total = await store.query_audit_logs(
        user_id=user_filter,
        start_time=start_time,
        end_time=end_time,
        action=action,
        resource_type=resource_type,
        limit=limit,
        offset=offset,
    )

    logger.info(
        f"Exporting {len(records)} audit logs for user {user.id} "
        f"(total available: {total}, format: {format.value})"
    )

    content_type_map = {
        ExportFormat.JSON: "application/json",
        ExportFormat.JSONL: "application/x-ndjson",
        ExportFormat.CSV: "text/csv",
    }

    ext_map = {
        ExportFormat.JSON: "json",
        ExportFormat.JSONL: "jsonl",
        ExportFormat.CSV: "csv",
    }

    if format == ExportFormat.JSON:
        stream = _stream_json(records, _audit_log_to_dict)
    elif format == ExportFormat.JSONL:
        stream = _stream_jsonl(records, _audit_log_to_dict)
    elif format == ExportFormat.CSV:
        stream = _stream_csv(records, _audit_log_to_dict)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {format.value}",
        )

    filename = f"audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext_map[format]}"

    return StreamingResponse(
        stream,
        media_type=content_type_map[format],
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Total-Records": str(total),
            "X-Returned-Records": str(len(records)),
        },
    )


def _nearest_rank_percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of a pre-sorted list (pct in [0, 100])."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(pct / 100.0 * len(sorted_values)))
    return float(sorted_values[min(rank, len(sorted_values)) - 1])


def _truncate_to_bucket(ts: datetime, bucket: str) -> datetime:
    """Floor a timestamp to the start of its time bucket."""
    if bucket == "day":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    if bucket == "minute":
        return ts.replace(second=0, microsecond=0)
    # default: hour
    return ts.replace(minute=0, second=0, microsecond=0)


def _compute_metrics_summary(
    detections: list[DetectionRecord],
    start_time: datetime | None,
    end_time: datetime | None,
    bucket: str,
) -> dict[str, Any]:
    """Aggregate detection records into a rich metrics summary.

    Beyond the headline counts this exposes score percentiles, per-method
    statistics, and a time-bucketed series — the surfaces a researcher analysing
    STEM / medical / meteorological / space detection runs actually plots. Every
    field is derived from the real ``DetectionRecord`` data, nothing synthetic.
    """
    if not detections:
        return {
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "total_detections": 0,
            "total_anomalies_found": 0,
            "anomaly_rate": 0.0,
            "avg_max_score": 0.0,
            "score_percentiles": {},
            "method_breakdown": {},
            "batch_vs_realtime": {"batch": 0, "realtime": 0},
            "score_distribution": {},
            "time_series": [],
        }

    total = len(detections)
    total_anomalies = sum(d.anomaly_count for d in detections)
    scores = sorted(d.max_score for d in detections)
    avg_score = sum(scores) / total

    # Per-method statistics: count, avg score, total anomalies.
    method_stats: dict[str, dict[str, Any]] = {}
    for d in detections:
        m = method_stats.setdefault(d.method, {"count": 0, "score_sum": 0.0, "anomalies": 0})
        m["count"] += 1
        m["score_sum"] += d.max_score
        m["anomalies"] += d.anomaly_count
    method_breakdown = {
        name: {
            "count": s["count"],
            "avg_max_score": round(s["score_sum"] / s["count"], 4),
            "total_anomalies": s["anomalies"],
        }
        for name, s in sorted(method_stats.items())
    }

    # Time-bucketed series (detections + anomalies per bucket).
    series_map: dict[datetime, dict[str, int]] = {}
    for d in detections:
        key = _truncate_to_bucket(d.timestamp, bucket)
        b = series_map.setdefault(key, {"detections": 0, "anomalies": 0})
        b["detections"] += 1
        b["anomalies"] += d.anomaly_count
    time_series = [
        {
            "bucket_start": k.isoformat(),
            "detections": v["detections"],
            "anomalies": v["anomalies"],
            "anomaly_rate": round(v["anomalies"] / v["detections"], 4) if v["detections"] else 0.0,
        }
        for k, v in sorted(series_map.items())
    ]

    return {
        "period": {
            "start": min(d.timestamp for d in detections).isoformat(),
            "end": max(d.timestamp for d in detections).isoformat(),
        },
        "total_detections": total,
        "total_anomalies_found": total_anomalies,
        "anomaly_rate": round(total_anomalies / total, 4) if total > 0 else 0.0,
        "avg_max_score": round(avg_score, 4),
        "score_percentiles": {
            "min": round(scores[0], 4),
            "p50": round(_nearest_rank_percentile(scores, 50), 4),
            "p90": round(_nearest_rank_percentile(scores, 90), 4),
            "p95": round(_nearest_rank_percentile(scores, 95), 4),
            "p99": round(_nearest_rank_percentile(scores, 99), 4),
            "max": round(scores[-1], 4),
        },
        "method_breakdown": method_breakdown,
        "batch_vs_realtime": {
            "batch": sum(1 for d in detections if d.is_batch),
            "realtime": sum(1 for d in detections if not d.is_batch),
        },
        "score_distribution": {
            "0.0-0.25": sum(1 for d in detections if d.max_score < 0.25),
            "0.25-0.50": sum(1 for d in detections if 0.25 <= d.max_score < 0.50),
            "0.50-0.75": sum(1 for d in detections if 0.50 <= d.max_score < 0.75),
            "0.75-1.0": sum(1 for d in detections if d.max_score >= 0.75),
        },
        "time_series": time_series,
    }


def _flatten_metrics_to_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nested metrics summary into tidy ``{metric, value}`` rows.

    A long/tidy two-column shape loads directly into pandas / R for the tabular
    formats. Nested keys become dotted (``method_breakdown.zscore.count``); the
    time series is emitted one metric per bucket field.
    """
    rows: list[dict[str, Any]] = []

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "bucket_start" in item:
                    stamp = item["bucket_start"]
                    for field_name, field_value in item.items():
                        if field_name == "bucket_start":
                            continue
                        rows.append(
                            {"metric": f"{prefix}.{stamp}.{field_name}", "value": field_value}
                        )
                else:
                    _walk(prefix, item)
        else:
            rows.append({"metric": prefix, "value": value})

    _walk("", summary)
    return rows


@router.get(
    "/metrics",
    summary="Export Metrics",
    description="Export system detection metrics (json summary, or tidy csv/jsonl rows).",
)
async def export_metrics(
    format: ExportFormat = Query(
        default=ExportFormat.JSON,
        description="Export format: json (nested summary), or csv/jsonl (tidy metric,value rows).",
    ),
    bucket: str = Query(
        default="hour",
        pattern="^(minute|hour|day)$",
        description="Time-series bucket granularity for the metrics time series.",
    ),
    start_time: datetime | None = Query(default=None, description="Start of time range"),
    end_time: datetime | None = Query(default=None, description="End of time range"),
    user: User = Depends(_get_current_user),
) -> Any:
    """Export system detection metrics in json/csv/jsonl.

    Researchers across Mercury's domains (STEM, medical, meteorological, space, …)
    consume detection metrics in whatever format their analysis stack expects, so
    the summary is offered as a nested JSON object *and* as tidy two-column
    csv/jsonl (``metric,value``) that loads straight into pandas / R.
    """
    store = get_data_store()

    is_admin = user.has_permission(Permission.ADMIN)
    user_filter = None if is_admin else user.id

    detections, _ = await store.query_detections(
        user_id=user_filter,
        start_time=start_time,
        end_time=end_time,
        limit=100000,
    )

    summary = _compute_metrics_summary(detections, start_time, end_time, bucket)

    if format == ExportFormat.JSON:
        return summary

    rows = _flatten_metrics_to_rows(summary)
    ext = format.value
    filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
    if format == ExportFormat.JSONL:
        stream = _stream_jsonl(rows, lambda r: r)
        media_type = "application/x-ndjson"
    elif format == ExportFormat.CSV:
        stream = _stream_csv(rows, lambda r: r)
        media_type = "text/csv"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported metrics export format: {format.value}. "
                "Use one of: json, csv, jsonl."
            ),
        )

    return StreamingResponse(
        stream,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
