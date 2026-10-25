# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The corresponding-source offer (AGPL-3.0 section 13).

Read-only, unauthenticated and always on, like the agent notice: it publishes
where this deployment's source lives, accepts no input, and changes no state.
The content lives in :mod:`omni_mercury_engine.security.source_offer`.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from omni_mercury_engine.security.source_offer import source_offer

__all__ = ["router"]

router = APIRouter()


@router.get(
    "/source",
    summary="Corresponding source of this deployment (AGPL-3.0 section 13)",
    description=(
        "Where the complete corresponding source of the running version is "
        "published, at no charge: the upstream repository for a stock "
        "deployment, or the operator's published fork for a modified one "
        "(`MERCURY_SOURCE_URL`)."
    ),
)
def source() -> JSONResponse:
    """Serve the corresponding-source offer.

    Returns:
        The offer as JSON.
    """
    return JSONResponse(source_offer())
