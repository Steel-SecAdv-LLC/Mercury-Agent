"""
Mercury Agent API Routes Package

Provides modular API route organization:
- detection: Anomaly detection endpoints
- batch: Batch processing endpoints
- models: Model management endpoints
- export: Data export endpoints
- admin: Administrative endpoints
"""

from omni_mercury_engine.api.routes.batch import router as batch_router
from omni_mercury_engine.api.routes.detection import router as detection_router
from omni_mercury_engine.api.routes.export import router as export_router
from omni_mercury_engine.api.routes.models import router as models_router

__all__ = [
    "batch_router",
    "detection_router",
    "export_router",
    "models_router",
]
