"""
Mercury Agent - Model Management API Endpoints

Production-grade model registry and lifecycle management for anomaly detection models.
Supports model versioning, deployment, A/B testing, and performance monitoring.

Features:
- Model registration with metadata and versioning
- Model deployment and rollback
- A/B testing configuration
- Performance metrics tracking
- Model comparison and evaluation
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, Permission, User


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["Model Management"])


class ModelType(StrEnum):
    """Supported model types."""

    FUSION = "fusion"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    NEUROSYMBOLIC = "neurosymbolic"
    LSTM_AE = "lstm_autoencoder"
    ISOLATION_FOREST = "isolation_forest"
    CUSTOM = "custom"


class ModelStatus(StrEnum):
    """Model deployment status."""

    DRAFT = "draft"
    STAGED = "staged"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelFramework(StrEnum):
    """Model framework/runtime."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    CUSTOM = "custom"


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_rps: float = 0.0
    inference_count: int = 0
    error_count: int = 0
    last_evaluated: datetime | None = None


@dataclass
class ModelVersion:
    """Model version record."""

    version_id: str
    model_id: str
    version_number: str
    file_path: str | None
    file_hash: str | None
    file_size_bytes: int = 0
    framework: ModelFramework = ModelFramework.PYTORCH
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    is_default: bool = False


@dataclass
class Model:
    """Model registry entry."""

    model_id: str
    name: str
    model_type: ModelType
    status: ModelStatus
    owner_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    tags: list[str] = field(default_factory=list)
    versions: dict[str, ModelVersion] = field(default_factory=dict)
    current_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    deployment_config: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Model registry with versioning and lifecycle management.

    Thread-safe model management with file-based storage.
    Production deployment should use object storage (S3, GCS) with database metadata.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        self._models: dict[str, Model] = {}
        self._lock = threading.RLock()
        default_storage = os.path.join(tempfile.gettempdir(), "mercury_models")
        resolved_path = storage_path or os.getenv("MODEL_STORAGE_PATH") or default_storage
        self._storage_path = Path(resolved_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._version_counter: dict[str, int] = {}

    def register_model(
        self,
        name: str,
        model_type: ModelType,
        owner_id: str,
        description: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Model:
        """Register a new model."""
        with self._lock:
            model_id = hashlib.sha256(f"{name}:{owner_id}:{time.time()}".encode()).hexdigest()[:16]

            if any(m.name == name and m.owner_id == owner_id for m in self._models.values()):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Model '{name}' already exists for this user",
                )

            model = Model(
                model_id=model_id,
                name=name,
                model_type=model_type,
                status=ModelStatus.DRAFT,
                owner_id=owner_id,
                description=description,
                tags=tags or [],
                metadata=metadata or {},
            )

            self._models[model_id] = model
            self._version_counter[model_id] = 0

            model_dir = self._storage_path / model_id
            model_dir.mkdir(exist_ok=True)

            logger.info(f"Model registered: {model_id} ({name}) by {owner_id}")
            return model

    def get_model(self, model_id: str) -> Model | None:
        """Get a model by ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        owner_id: str | None = None,
        model_type: ModelType | None = None,
        status: ModelStatus | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[Model]:
        """List models with filtering."""
        models = list(self._models.values())

        if owner_id:
            models = [m for m in models if m.owner_id == owner_id]

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if status:
            models = [m for m in models if m.status == status]

        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]

        models.sort(key=lambda m: m.updated_at, reverse=True)
        return models[:limit]

    def add_version(
        self,
        model_id: str,
        created_by: str,
        file_content: bytes | None = None,
        framework: ModelFramework = ModelFramework.PYTORCH,
        description: str = "",
        config: dict[str, Any] | None = None,
        set_as_default: bool = False,
    ) -> ModelVersion:
        """Add a new version to a model."""
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {model_id} not found",
                )

            self._version_counter[model_id] += 1
            version_num = f"v{self._version_counter[model_id]}"
            version_id = f"{model_id}:{version_num}"

            file_path = None
            file_hash = None
            file_size = 0

            if file_content:
                version_dir = self._storage_path / model_id / version_num
                version_dir.mkdir(parents=True, exist_ok=True)

                ext = ".pt" if framework == ModelFramework.PYTORCH else ".bin"
                file_path = str(version_dir / f"model{ext}")

                with open(file_path, "wb") as f:
                    f.write(file_content)

                file_hash = hashlib.sha256(file_content).hexdigest()
                file_size = len(file_content)

            version = ModelVersion(
                version_id=version_id,
                model_id=model_id,
                version_number=version_num,
                file_path=file_path,
                file_hash=file_hash,
                file_size_bytes=file_size,
                framework=framework,
                created_by=created_by,
                description=description,
                config=config or {},
                is_default=set_as_default or not model.versions,
            )

            model.versions[version_num] = version
            model.updated_at = datetime.now()

            if set_as_default or not model.current_version:
                for v in model.versions.values():
                    v.is_default = False
                version.is_default = True
                model.current_version = version_num

            logger.info(f"Model version added: {version_id}")
            return version

    def set_default_version(self, model_id: str, version_number: str) -> ModelVersion:
        """Set the default version for a model."""
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {model_id} not found",
                )

            if version_number not in model.versions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Version {version_number} not found",
                )

            for v in model.versions.values():
                v.is_default = False

            version = model.versions[version_number]
            version.is_default = True
            model.current_version = version_number
            model.updated_at = datetime.now()

            return version

    def update_status(self, model_id: str, new_status: ModelStatus) -> Model:
        """Update model deployment status."""
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {model_id} not found",
                )

            model.status = new_status
            model.updated_at = datetime.now()

            logger.info(f"Model {model_id} status updated to {new_status.value}")
            return model

    def update_metrics(
        self,
        model_id: str,
        version_number: str,
        metrics: dict[str, float],
    ) -> ModelVersion:
        """Update version metrics."""
        with self._lock:
            model = self._models.get(model_id)
            if not model or version_number not in model.versions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Model or version not found",
                )

            version = model.versions[version_number]

            if "accuracy" in metrics:
                version.metrics.accuracy = metrics["accuracy"]
            if "precision" in metrics:
                version.metrics.precision = metrics["precision"]
            if "recall" in metrics:
                version.metrics.recall = metrics["recall"]
            if "f1_score" in metrics:
                version.metrics.f1_score = metrics["f1_score"]
            if "auc_roc" in metrics:
                version.metrics.auc_roc = metrics["auc_roc"]
            if "latency_p50_ms" in metrics:
                version.metrics.latency_p50_ms = metrics["latency_p50_ms"]
            if "latency_p99_ms" in metrics:
                version.metrics.latency_p99_ms = metrics["latency_p99_ms"]
            if "throughput_rps" in metrics:
                version.metrics.throughput_rps = metrics["throughput_rps"]

            version.metrics.last_evaluated = datetime.now()

            return version

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its versions."""
        with self._lock:
            if model_id not in self._models:
                return False

            model_dir = self._storage_path / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)

            del self._models[model_id]
            if model_id in self._version_counter:
                del self._version_counter[model_id]

            logger.info(f"Model {model_id} deleted")
            return True


_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the model registry instance."""
    return _model_registry


class ModelCreateRequest(BaseModel):
    """Request to create a new model."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Model name",
    )
    model_type: ModelType = Field(
        default=ModelType.FUSION,
        description="Type of anomaly detection model",
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Model description",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Tags for categorization",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata",
    )


class ModelResponse(BaseModel):
    """Model information response."""

    model_id: str
    name: str
    model_type: ModelType
    status: ModelStatus
    owner_id: str
    created_at: datetime
    updated_at: datetime
    description: str
    tags: list[str]
    current_version: str | None
    version_count: int
    metadata: dict[str, Any]


class ModelVersionRequest(BaseModel):
    """Request to add a model version."""

    framework: ModelFramework = Field(
        default=ModelFramework.PYTORCH,
        description="Model framework",
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Version description",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Model configuration",
    )
    set_as_default: bool = Field(
        default=False,
        description="Set as default version",
    )


class ModelVersionResponse(BaseModel):
    """Model version information response."""

    version_id: str
    model_id: str
    version_number: str
    framework: ModelFramework
    created_at: datetime
    created_by: str
    description: str
    config: dict[str, Any]
    file_hash: str | None
    file_size_bytes: int
    is_default: bool
    metrics: dict[str, Any]


class MetricsUpdateRequest(BaseModel):
    """Request to update model metrics."""

    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1_score: float | None = Field(default=None, ge=0.0, le=1.0)
    auc_roc: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_p50_ms: float | None = Field(default=None, ge=0.0)
    latency_p99_ms: float | None = Field(default=None, ge=0.0)
    throughput_rps: float | None = Field(default=None, ge=0.0)


class DeploymentRequest(BaseModel):
    """Request to deploy a model."""

    version_number: str = Field(
        ...,
        description="Version to deploy",
    )
    replicas: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of model replicas",
    )
    resources: dict[str, Any] | None = Field(
        default=None,
        description="Resource requirements (cpu, memory, gpu)",
    )
    canary_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Percentage of traffic for canary deployment",
    )


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


@router.post(
    "",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Model",
    description="Register a new model in the registry.",
)
async def register_model(
    request: ModelCreateRequest,
    user: User = Depends(_get_current_user),
) -> ModelResponse:
    """Register a new model."""
    registry = get_model_registry()

    model = registry.register_model(
        name=request.name,
        model_type=request.model_type,
        owner_id=user.id,
        description=request.description,
        tags=request.tags,
        metadata=request.metadata,
    )

    return ModelResponse(
        model_id=model.model_id,
        name=model.name,
        model_type=model.model_type,
        status=model.status,
        owner_id=model.owner_id,
        created_at=model.created_at,
        updated_at=model.updated_at,
        description=model.description,
        tags=model.tags,
        current_version=model.current_version,
        version_count=len(model.versions),
        metadata=model.metadata,
    )


@router.get(
    "",
    response_model=list[ModelResponse],
    summary="List Models",
    description="List all models with optional filtering.",
)
async def list_models(
    model_type: ModelType | None = Query(default=None, description="Filter by type"),
    status_filter: ModelStatus | None = Query(
        default=None, alias="status", description="Filter by status"
    ),
    tags: list[str] | None = Query(default=None, description="Filter by tags"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results"),
    user: User = Depends(_get_current_user),
) -> list[ModelResponse]:
    """List models."""
    registry = get_model_registry()

    models = registry.list_models(
        owner_id=user.id if not user.has_permission(Permission.ADMIN) else None,
        model_type=model_type,
        status=status_filter,
        tags=tags,
        limit=limit,
    )

    return [
        ModelResponse(
            model_id=m.model_id,
            name=m.name,
            model_type=m.model_type,
            status=m.status,
            owner_id=m.owner_id,
            created_at=m.created_at,
            updated_at=m.updated_at,
            description=m.description,
            tags=m.tags,
            current_version=m.current_version,
            version_count=len(m.versions),
            metadata=m.metadata,
        )
        for m in models
    ]


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Get Model",
    description="Get model details by ID.",
)
async def get_model(
    model_id: str,
    user: User = Depends(_get_current_user),
) -> ModelResponse:
    """Get model details."""
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.owner_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return ModelResponse(
        model_id=model.model_id,
        name=model.name,
        model_type=model.model_type,
        status=model.status,
        owner_id=model.owner_id,
        created_at=model.created_at,
        updated_at=model.updated_at,
        description=model.description,
        tags=model.tags,
        current_version=model.current_version,
        version_count=len(model.versions),
        metadata=model.metadata,
    )


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Model",
    description="Delete a model and all its versions.",
)
async def delete_model(
    model_id: str,
    user: User = Depends(_get_current_user),
) -> None:
    """Delete a model."""
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.owner_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    registry.delete_model(model_id)


@router.post(
    "/{model_id}/versions",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add Version",
    description="Add a new version to a model with optional file upload.",
)
async def add_version(
    model_id: str,
    framework: ModelFramework = ModelFramework.PYTORCH,
    description: str = "",
    set_as_default: bool = False,
    file: UploadFile | None = File(default=None),
    user: User = Depends(_get_current_user),
) -> ModelVersionResponse:
    """Add a model version."""
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.owner_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    file_content = None
    if file:
        file_content = await file.read()

        max_size = int(os.getenv("MAX_MODEL_SIZE_MB", "500")) * 1024 * 1024
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Model file exceeds maximum size of {max_size // (1024*1024)} MB",
            )

    version = registry.add_version(
        model_id=model_id,
        created_by=user.id,
        file_content=file_content,
        framework=framework,
        description=description,
        set_as_default=set_as_default,
    )

    return ModelVersionResponse(
        version_id=version.version_id,
        model_id=version.model_id,
        version_number=version.version_number,
        framework=version.framework,
        created_at=version.created_at,
        created_by=version.created_by,
        description=version.description,
        config=version.config,
        file_hash=version.file_hash,
        file_size_bytes=version.file_size_bytes,
        is_default=version.is_default,
        metrics={
            "accuracy": version.metrics.accuracy,
            "precision": version.metrics.precision,
            "recall": version.metrics.recall,
            "f1_score": version.metrics.f1_score,
            "auc_roc": version.metrics.auc_roc,
        },
    )


@router.get(
    "/{model_id}/versions",
    response_model=list[ModelVersionResponse],
    summary="List Versions",
    description="List all versions of a model.",
)
async def list_versions(
    model_id: str,
    user: User = Depends(_get_current_user),
) -> list[ModelVersionResponse]:
    """List model versions."""
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.owner_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return [
        ModelVersionResponse(
            version_id=v.version_id,
            model_id=v.model_id,
            version_number=v.version_number,
            framework=v.framework,
            created_at=v.created_at,
            created_by=v.created_by,
            description=v.description,
            config=v.config,
            file_hash=v.file_hash,
            file_size_bytes=v.file_size_bytes,
            is_default=v.is_default,
            metrics={
                "accuracy": v.metrics.accuracy,
                "precision": v.metrics.precision,
                "recall": v.metrics.recall,
                "f1_score": v.metrics.f1_score,
                "auc_roc": v.metrics.auc_roc,
            },
        )
        for v in sorted(model.versions.values(), key=lambda x: x.created_at, reverse=True)
    ]


@router.put(
    "/{model_id}/versions/{version_number}/default",
    response_model=ModelVersionResponse,
    summary="Set Default Version",
    description="Set a version as the default for the model.",
)
async def set_default_version(
    model_id: str,
    version_number: str,
    user: User = Depends(_get_current_user),
) -> ModelVersionResponse:
    """Set default model version."""
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.owner_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    version = registry.set_default_version(model_id, version_number)

    return ModelVersionResponse(
        version_id=version.version_id,
        model_id=version.model_id,
        version_number=version.version_number,
        framework=version.framework,
        created_at=version.created_at,
        created_by=version.created_by,
        description=version.description,
        config=version.config,
        file_hash=version.file_hash,
        file_size_bytes=version.file_size_bytes,
        is_default=version.is_default,
        metrics={
            "accuracy": version.metrics.accuracy,
            "precision": version.metrics.precision,
            "recall": version.metrics.recall,
            "f1_score": version.metrics.f1_score,
            "auc_roc": version.metrics.auc_roc,
        },
    )


@router.put(
    "/{model_id}/versions/{version_number}/metrics",
    response_model=ModelVersionResponse,
    summary="Update Metrics",
    description="Update performance metrics for a model version.",
)
async def update_metrics(
    model_id: str,
    version_number: str,
    request: MetricsUpdateRequest,
    user: User = Depends(_get_current_user),
) -> ModelVersionResponse:
    """Update model metrics."""
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.owner_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    metrics = {k: v for k, v in request.model_dump().items() if v is not None}
    version = registry.update_metrics(model_id, version_number, metrics)

    return ModelVersionResponse(
        version_id=version.version_id,
        model_id=version.model_id,
        version_number=version.version_number,
        framework=version.framework,
        created_at=version.created_at,
        created_by=version.created_by,
        description=version.description,
        config=version.config,
        file_hash=version.file_hash,
        file_size_bytes=version.file_size_bytes,
        is_default=version.is_default,
        metrics={
            "accuracy": version.metrics.accuracy,
            "precision": version.metrics.precision,
            "recall": version.metrics.recall,
            "f1_score": version.metrics.f1_score,
            "auc_roc": version.metrics.auc_roc,
            "latency_p50_ms": version.metrics.latency_p50_ms,
            "latency_p99_ms": version.metrics.latency_p99_ms,
            "throughput_rps": version.metrics.throughput_rps,
            "last_evaluated": (
                version.metrics.last_evaluated.isoformat()
                if version.metrics.last_evaluated
                else None
            ),
        },
    )


@router.put(
    "/{model_id}/status",
    response_model=ModelResponse,
    summary="Update Status",
    description="Update model deployment status.",
)
async def update_model_status(
    model_id: str,
    new_status: ModelStatus = Query(..., alias="status"),
    user: User = Depends(_get_current_user),
) -> ModelResponse:
    """Update model status."""
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.owner_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    model = registry.update_status(model_id, new_status)

    return ModelResponse(
        model_id=model.model_id,
        name=model.name,
        model_type=model.model_type,
        status=model.status,
        owner_id=model.owner_id,
        created_at=model.created_at,
        updated_at=model.updated_at,
        description=model.description,
        tags=model.tags,
        current_version=model.current_version,
        version_count=len(model.versions),
        metadata=model.metadata,
    )


@router.post(
    "/{model_id}/deploy",
    response_model=dict[str, Any],
    summary="Deploy Model",
    description="Deploy a model version to production.",
)
async def deploy_model(
    model_id: str,
    request: DeploymentRequest,
    user: User = Depends(_get_current_user),
) -> dict[str, Any]:
    """Deploy a model."""
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    if model.owner_id != user.id and not user.has_permission(Permission.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    if request.version_number not in model.versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {request.version_number} not found",
        )

    model.deployment_config = {
        "deployed_version": request.version_number,
        "replicas": request.replicas,
        "resources": request.resources,
        "canary_percentage": request.canary_percentage,
        "deployed_at": datetime.now().isoformat(),
        "deployed_by": user.id,
    }

    registry.update_status(model_id, ModelStatus.DEPLOYED)

    logger.info(f"Model {model_id} deployed: version={request.version_number}")

    return {
        "status": "deployed",
        "model_id": model_id,
        "version": request.version_number,
        "deployment_config": model.deployment_config,
        "message": f"Model {model.name} deployed successfully",
    }
