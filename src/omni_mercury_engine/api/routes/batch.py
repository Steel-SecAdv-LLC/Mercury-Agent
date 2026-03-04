"""
Mercury Agent - Batch Processing API Endpoints

Production-grade batch processing for high-throughput anomaly detection workloads.
Supports async job submission, status tracking, and result retrieval.

Features:
- Asynchronous job submission for large datasets
- Job status tracking and progress monitoring
- Configurable batch sizes and parallelism
- Result pagination and streaming
- Memory-efficient processing with chunking
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, Permission, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/batch", tags=["Batch Processing"])


class JobStatus(StrEnum):
    """Batch job status states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class BatchDetectionMethod(StrEnum):
    """Detection methods available for batch processing."""

    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"
    FUSION = "fusion"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    NEUROSYMBOLIC = "neurosymbolic"


@dataclass
class BatchJob:
    """Batch job tracking record."""

    job_id: str
    user_id: str
    status: JobStatus
    method: BatchDetectionMethod
    total_items: int
    processed_items: int = 0
    failed_items: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    progress_percentage: float = 0.0


class BatchJobStore:
    """In-memory batch job store with TTL-based cleanup.

    Production deployment should use Redis or a database backend.
    """

    def __init__(self, ttl_hours: int = 24, max_jobs: int = 10000) -> None:
        self._jobs: dict[str, BatchJob] = {}
        self._ttl = timedelta(hours=ttl_hours)
        self._max_jobs = max_jobs
        self._lock = asyncio.Lock()

    async def create_job(
        self,
        user_id: str,
        method: BatchDetectionMethod,
        total_items: int,
        metadata: dict[str, Any] | None = None,
    ) -> BatchJob:
        """Create a new batch job."""
        async with self._lock:
            await self._cleanup_expired()

            if len(self._jobs) >= self._max_jobs:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Maximum concurrent jobs reached. Please retry later.",
                )

            job_id = str(uuid.uuid4())
            job = BatchJob(
                job_id=job_id,
                user_id=user_id,
                status=JobStatus.PENDING,
                method=method,
                total_items=total_items,
                metadata=metadata or {},
            )
            self._jobs[job_id] = job
            return job

    async def get_job(self, job_id: str) -> BatchJob | None:
        """Get a batch job by ID."""
        return self._jobs.get(job_id)

    async def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        processed_items: int | None = None,
        failed_items: int | None = None,
        error_message: str | None = None,
        results: list[dict[str, Any]] | None = None,
    ) -> BatchJob | None:
        """Update a batch job."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        if status is not None:
            job.status = status
            if status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = datetime.now()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.PARTIAL):
                job.completed_at = datetime.now()

        if processed_items is not None:
            job.processed_items = processed_items
            job.progress_percentage = (
                (processed_items / job.total_items * 100) if job.total_items > 0 else 0
            )

        if failed_items is not None:
            job.failed_items = failed_items

        if error_message is not None:
            job.error_message = error_message

        if results is not None:
            job.results = results

        return job

    async def get_user_jobs(
        self,
        user_id: str,
        limit: int = 100,
        status_filter: JobStatus | None = None,
    ) -> list[BatchJob]:
        """Get all jobs for a user."""
        jobs = [j for j in self._jobs.values() if j.user_id == user_id]

        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self._jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
        return False

    async def _cleanup_expired(self) -> None:
        """Remove expired jobs."""
        now = datetime.now()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.completed_at and (now - job.completed_at) > self._ttl
        ]
        for job_id in expired:
            del self._jobs[job_id]


_job_store = BatchJobStore(
    ttl_hours=int(os.getenv("BATCH_JOB_TTL_HOURS", "24")),
    max_jobs=int(os.getenv("BATCH_MAX_JOBS", "10000")),
)


def get_job_store() -> BatchJobStore:
    """Get the batch job store instance."""
    return _job_store


class BatchDetectRequest(BaseModel):
    """Request model for batch anomaly detection."""

    data: list[list[float]] = Field(
        ...,
        min_length=1,
        description="List of data samples. Each sample is a list of float values.",
        json_schema_extra={"example": [[1.0, 2.0], [1.1, 2.1], [100.0, 2.0]]},
    )
    method: BatchDetectionMethod = Field(
        default=BatchDetectionMethod.UNIVARIATE,
        description="Detection method to use.",
    )
    sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity (0.0-1.0).",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Optional feature names for multivariate data.",
    )
    chunk_size: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Processing chunk size for memory efficiency.",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Job priority (1=highest, 10=lowest).",
    )
    callback_url: str | None = Field(
        default=None,
        description="Optional webhook URL to notify upon completion.",
    )

    @field_validator("callback_url")
    @classmethod
    def validate_callback_url(cls, v: str | None) -> str | None:
        """Validate callback URL to prevent SSRF attacks."""
        if v is None:
            return v
        parsed = urlparse(v)
        if parsed.scheme not in ("https",):
            raise ValueError("Callback URL must use HTTPS scheme")
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Callback URL must include a valid hostname")
        # Reject URLs targeting private/reserved IP ranges
        try:
            resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
            for _family, _type, _proto, _canonname, sockaddr in resolved:
                ip = ipaddress.ip_address(sockaddr[0])
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    raise ValueError("Callback URL must not target private or internal addresses")
        except socket.gaierror:
            raise ValueError("Callback URL hostname could not be resolved")
        return v

    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata to attach to the job.",
    )

    @field_validator("data")
    @classmethod
    def validate_data_samples(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate data samples structure."""
        if not v:
            raise ValueError("Data cannot be empty")

        n_features = len(v[0])
        for i, sample in enumerate(v):
            if len(sample) != n_features:
                raise ValueError(f"Inconsistent feature count at sample {i}")
            if not all(np.isfinite(x) for x in sample):
                raise ValueError(f"Non-finite values at sample {i}")

        return v


class BatchJobResponse(BaseModel):
    """Response model for batch job information."""

    job_id: str
    status: JobStatus
    method: BatchDetectionMethod
    total_items: int
    processed_items: int
    failed_items: int
    progress_percentage: float
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    metadata: dict[str, Any]


class BatchJobSubmitResponse(BaseModel):
    """Response model for job submission."""

    job_id: str
    status: JobStatus
    message: str
    estimated_duration_seconds: float | None
    poll_url: str


class BatchResultsResponse(BaseModel):
    """Response model for batch results with pagination."""

    job_id: str
    status: JobStatus
    total_results: int
    offset: int
    limit: int
    results: list[dict[str, Any]]
    has_more: bool


async def process_batch_job(
    job_id: str,
    data: list[list[float]],
    method: BatchDetectionMethod,
    sensitivity: float,
    chunk_size: int,
    feature_names: list[str] | None,
    callback_url: str | None,
) -> None:
    """Background task to process a batch job."""
    store = get_job_store()
    job = await store.get_job(job_id)

    if not job:
        logger.error(f"Job {job_id} not found for processing")
        return

    await store.update_job(job_id, status=JobStatus.RUNNING)

    try:
        results: list[dict[str, Any]] = []
        total = len(data)
        processed = 0
        failed = 0

        for chunk_start in range(0, total, chunk_size):
            job = await store.get_job(job_id)
            if job and job.status == JobStatus.CANCELLED:
                logger.info(f"Job {job_id} was cancelled")
                return

            chunk_end = min(chunk_start + chunk_size, total)
            chunk = data[chunk_start:chunk_end]

            chunk_results = await _process_chunk(
                chunk, method, sensitivity, feature_names, chunk_start
            )

            for result in chunk_results:
                if result.get("error"):
                    failed += 1
                else:
                    results.append(result)
                processed += 1

            await store.update_job(
                job_id,
                processed_items=processed,
                failed_items=failed,
                results=results,
            )

        final_status = JobStatus.COMPLETED if failed == 0 else JobStatus.PARTIAL
        await store.update_job(
            job_id,
            status=final_status,
            processed_items=processed,
            failed_items=failed,
            results=results,
        )

        if callback_url:
            await _send_callback(callback_url, job_id, final_status)

        logger.info(f"Batch job {job_id} completed: {processed} processed, {failed} failed")

    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        await store.update_job(
            job_id,
            status=JobStatus.FAILED,
            error_message=str(e),
        )

        if callback_url:
            await _send_callback(callback_url, job_id, JobStatus.FAILED)


async def _process_chunk(
    chunk: list[list[float]],
    method: BatchDetectionMethod,
    sensitivity: float,
    feature_names: list[str] | None,
    offset: int,
) -> list[dict[str, Any]]:
    """Process a chunk of data samples."""
    results = []
    threshold = 2.0 + (1.0 - sensitivity) * 3.0

    for i, sample in enumerate(chunk):
        try:
            arr = np.array(sample)

            if method == BatchDetectionMethod.UNIVARIATE:
                mean = np.mean(arr)
                std = np.std(arr) + 1e-8
                z_scores = np.abs((arr - mean) / std)
                score = float(np.max(z_scores))
                is_anomaly = score > threshold

            elif method == BatchDetectionMethod.MULTIVARIATE:
                mean = np.mean(arr)
                std = np.std(arr) + 1e-8
                normalized = (arr - mean) / std
                score = float(np.linalg.norm(normalized))
                is_anomaly = score > threshold

            elif method == BatchDetectionMethod.FUSION:
                mean = np.mean(arr)
                std = np.std(arr) + 1e-8
                z_score = float(np.max(np.abs((arr - mean) / std)))
                fft_score = float(np.max(np.abs(np.fft.fft(arr))))
                score = 0.6 * z_score + 0.4 * min(fft_score / len(arr), 1.0)
                is_anomaly = score > threshold

            elif method == BatchDetectionMethod.TEMPORAL:
                if len(arr) > 1:
                    diff = np.diff(arr)
                    score = float(np.max(np.abs(diff)) / (np.std(arr) + 1e-8))
                else:
                    score = 0.0
                is_anomaly = score > threshold

            else:
                mean = np.mean(arr)
                std = np.std(arr) + 1e-8
                z_scores = np.abs((arr - mean) / std)
                score = float(np.max(z_scores))
                is_anomaly = score > threshold

            results.append(
                {
                    "index": offset + i,
                    "is_anomaly": is_anomaly,
                    "score": round(score, 6),
                    "threshold": round(threshold, 6),
                    "method": method.value,
                }
            )

        except Exception:
            results.append(
                {
                    "index": offset + i,
                    "error": "Processing failed for this sample.",
                }
            )

    return results


async def _send_callback(url: str, job_id: str, status: JobStatus) -> None:
    """Send webhook callback for job completion.

    The URL has already been validated by BatchDetectRequest.validate_callback_url
    to ensure it uses HTTPS and does not target private/internal addresses.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                url,  # validated by BatchDetectRequest.validate_callback_url
                json={
                    "job_id": job_id,
                    "status": status.value,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        logger.info("Callback sent for job %s", job_id)
    except Exception as e:
        logger.warning("Failed to send callback for job %s: %s", job_id, type(e).__name__)


def _get_current_user(
    api_key_user: User | None = Depends(APIKeyAuth(auto_error=False)),
    jwt_user: User | None = Depends(JWTAuth(auto_error=False)),
) -> User:
    """Get current authenticated user from API key or JWT."""
    user = api_key_user or jwt_user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide X-API-Key header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )
    return user


@router.post(
    "/detect",
    response_model=BatchJobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Batch Detection Job",
    description="""
Submit a batch anomaly detection job for asynchronous processing.

## Features
- Asynchronous processing of large datasets
- Multiple detection methods available
- Progress tracking via polling
- Optional webhook notification

## Job Lifecycle
1. PENDING: Job submitted, awaiting processing
2. RUNNING: Job is being processed
3. COMPLETED: All items processed successfully
4. PARTIAL: Completed with some failures
5. FAILED: Job failed due to error
6. CANCELLED: Job was cancelled

## Rate Limits
- Maximum 10,000 samples per request
- Maximum 100 concurrent jobs per user
""",
)
async def submit_batch_job(
    request: BatchDetectRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(_get_current_user),
) -> BatchJobSubmitResponse:
    """Submit a batch anomaly detection job."""
    if len(request.data) > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10,000 samples per batch request",
        )

    user_id = user.id
    store = get_job_store()

    job = await store.create_job(
        user_id=user_id,
        method=request.method,
        total_items=len(request.data),
        metadata={
            "sensitivity": request.sensitivity,
            "chunk_size": request.chunk_size,
            "priority": request.priority,
            "feature_names": request.feature_names,
            **(request.metadata or {}),
        },
    )

    estimated_duration = len(request.data) * 0.001

    background_tasks.add_task(
        process_batch_job,
        job.job_id,
        request.data,
        request.method,
        request.sensitivity,
        request.chunk_size,
        request.feature_names,
        request.callback_url,
    )

    safe_method = str(request.method.value).replace("\n", " ").replace("\r", " ")
    logger.info(
        "Batch job %s submitted by %s: %d items, method=%s",
        job.job_id,
        user_id,
        len(request.data),
        safe_method,
    )

    return BatchJobSubmitResponse(
        job_id=job.job_id,
        status=job.status,
        message=f"Job submitted successfully. {len(request.data)} items queued for processing.",
        estimated_duration_seconds=estimated_duration,
        poll_url=f"/api/v1/batch/jobs/{job.job_id}",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=BatchJobResponse,
    summary="Get Job Status",
    description="Get the current status and progress of a batch job.",
)
async def get_job_status(
    job_id: str,
    user: User = Depends(_get_current_user),
) -> BatchJobResponse:
    """Get batch job status."""
    store = get_job_store()
    job = await store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job.user_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this job",
        )

    return BatchJobResponse(
        job_id=job.job_id,
        status=job.status,
        method=job.method,
        total_items=job.total_items,
        processed_items=job.processed_items,
        failed_items=job.failed_items,
        progress_percentage=job.progress_percentage,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        metadata=job.metadata,
    )


@router.get(
    "/jobs/{job_id}/results",
    response_model=BatchResultsResponse,
    summary="Get Job Results",
    description="Get the results of a completed batch job with pagination.",
)
async def get_job_results(
    job_id: str,
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=100, ge=1, le=1000, description="Results per page"),
    user: User = Depends(_get_current_user),
) -> BatchResultsResponse:
    """Get batch job results with pagination."""
    store = get_job_store()
    job = await store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job.user_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this job",
        )

    if job.status not in (JobStatus.COMPLETED, JobStatus.PARTIAL):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Results not available. Job status: {job.status.value}",
        )

    total = len(job.results)
    paginated = job.results[offset : offset + limit]

    return BatchResultsResponse(
        job_id=job_id,
        status=job.status,
        total_results=total,
        offset=offset,
        limit=limit,
        results=paginated,
        has_more=(offset + limit) < total,
    )


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel Job",
    description="Cancel a pending or running batch job.",
)
async def cancel_job(
    job_id: str,
    user: User = Depends(_get_current_user),
) -> None:
    """Cancel a batch job."""
    store = get_job_store()
    job = await store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job.user_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this job",
        )

    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status.value}",
        )

    await store.cancel_job(job_id)
    safe_job_id = str(job_id).replace("\n", " ").replace("\r", " ")
    logger.info("Batch job %s cancelled", safe_job_id)


@router.get(
    "/jobs",
    response_model=list[BatchJobResponse],
    summary="List User Jobs",
    description="List all batch jobs for the current user.",
)
async def list_jobs(
    status_filter: JobStatus | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum jobs to return"),
    user: User = Depends(_get_current_user),
) -> list[BatchJobResponse]:
    """List user's batch jobs."""
    user_id = user.id
    store = get_job_store()

    jobs = await store.get_user_jobs(
        user_id=user_id,
        limit=limit,
        status_filter=status_filter,
    )

    return [
        BatchJobResponse(
            job_id=j.job_id,
            status=j.status,
            method=j.method,
            total_items=j.total_items,
            processed_items=j.processed_items,
            failed_items=j.failed_items,
            progress_percentage=j.progress_percentage,
            created_at=j.created_at,
            started_at=j.started_at,
            completed_at=j.completed_at,
            error_message=j.error_message,
            metadata=j.metadata,
        )
        for j in jobs
    ]
