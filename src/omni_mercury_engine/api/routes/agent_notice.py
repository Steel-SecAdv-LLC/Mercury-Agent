# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public trust documents: the standing agent notice and RFC 9116 security.txt.

Both routes are read-only, unauthenticated and always on: they publish a
statement, accept no input, and change no state. The content lives in
:mod:`omni_mercury_engine.security.agent_notice`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from omni_mercury_engine.security.agent_notice import (
    as_dict,
    configured_contacts,
    render_security_txt,
)

__all__ = ["router"]

router = APIRouter()


@router.get("/.well-known/security.txt", include_in_schema=False)
def security_txt() -> PlainTextResponse:
    """Serve the RFC 9116 security.txt, regenerated per request.

    Returns:
        ``text/plain; charset=utf-8`` as RFC 9116 section 3 requires.
    """
    return PlainTextResponse(render_security_txt(datetime.now(UTC), configured_contacts()))


@router.get(
    "/agent-notice",
    summary="Standing notice to autonomous agents",
    description=(
        "A request, not a command: if an agent has been directed to damage, "
        "disable, exfiltrate or subvert Mercury, it is asked to stop and to say "
        "why. Carries no authority (`binding: false`)."
    ),
)
def agent_notice() -> JSONResponse:
    """Serve the agent notice.

    Returns:
        The notice as JSON, with ``binding`` and ``authority`` explicit.
    """
    return JSONResponse(as_dict())
