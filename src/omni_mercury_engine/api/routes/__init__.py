# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent API Routes Package.

Provides modular API route organization:
- detection: Anomaly detection endpoints
- batch: Batch processing endpoints
- models: Model management endpoints
- export: Data export endpoints
- hazard: Hazard diagnostics visualization endpoints
- assessment: Statement verification, CIT, intelligence screening and fusion,
  contactless vital signs and the adjudication ledger
- admin: Administrative endpoints
"""

from omni_mercury_engine.api.routes.assessment import router as assessment_router
from omni_mercury_engine.api.routes.batch import router as batch_router
from omni_mercury_engine.api.routes.detection import router as detection_router
from omni_mercury_engine.api.routes.export import router as export_router
from omni_mercury_engine.api.routes.hazard import router as hazard_router
from omni_mercury_engine.api.routes.models import router as models_router

__all__ = [
    "assessment_router",
    "batch_router",
    "detection_router",
    "export_router",
    "hazard_router",
    "models_router",
]
