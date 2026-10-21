# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mercury Agent - Evidence-Assessment and Human-Subject Routes.

HTTP surface for statement verification, the Concealed Information Test,
counterintelligence screening and all-source fusion, contactless vital signs
and the adjudication ledger. Every route calls
:mod:`omni_mercury_engine.assessment.service`, the implementation the
``mercury-agent assess`` / ``mercury-agent adjudication`` CLI groups and the
``mercury_*`` MCP tools also call, so the gates are identical on every surface.

Access:

* Every route needs an authenticated principal (API key or JWT).
* Analysis routes need ``detect``; ledger reads need ``read``; ledger writes
  need ``write``; adjudicator decisions need ``admin`` or the
  ``adjudicator`` role; exports need ``export``.
* A rating or decision is recorded under the principal's own id and stamped
  with the server's clock: nobody can rate as somebody else or backdate.
* The Concealed Information Test and vital signs stay off (403) until the
  operator sets ``MERCURY_HUMAN_SUBJECT_CAPABILITIES``.

Refusals: 403 carries ``check`` (``harm_uplift``, ``covert_sensing``,
``benevolence``), ``consent`` or ``disabled``. Request audit is automatic
(``CorrelationIDMiddleware``); the gates write their own decision records.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, Permission, User

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/assessment", tags=["Assessment"])

#: Largest request body accepted by these routes (bytes). Vital-signs arrays
#: are the large case; ``service.MAX_SIGNAL_VALUES`` bounds them again after
#: parsing.
MAX_BODY_BYTES = int(os.environ.get("MERCURY_ASSESSMENT_MAX_BODY_BYTES", str(64 * 1024 * 1024)))


def _principal(permission: Permission) -> Callable[..., User]:
    """Dependency: an authenticated principal holding ``permission`` (or ``admin``)."""

    def dependency(
        api_key_user: User | None = Depends(APIKeyAuth(auto_error=False)),
        jwt_user: User | None = Depends(JWTAuth(auto_error=False)),
    ) -> User:
        user = api_key_user or jwt_user
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer, ApiKey"},
            )
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"The '{permission.value}' permission is required",
            )
        return user

    return dependency


def _adjudicator(
    user: User = Depends(_principal(Permission.WRITE)),
) -> User:
    if user.has_permission(Permission.ADMIN) or user.has_role("adjudicator"):
        return user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Recording a decision needs the 'adjudicator' role or 'admin'",
    )


async def _bounded_body(request: Request) -> None:
    """Refuse a declared body larger than :data:`MAX_BODY_BYTES` before it is parsed."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise HTTPException(
            status_code=413,  # Content Too Large (RFC 9110 §15.5.14)
            detail=f"Request body exceeds {MAX_BODY_BYTES} bytes",
        )


async def _run(call: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run a service call off the event loop and map its errors to HTTP statuses."""
    try:
        from omni_mercury_engine.assessment.policy import CapabilityDisabledError
        from omni_mercury_engine.assessment.service import InputError
        from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
        from omni_mercury_engine.security.adjudication import ExportRefused
        from omni_mercury_engine.truth_decipher import ConsentRequiredError
    except ImportError as exc:  # the gate needs the ML stack; never run ungated
        logger.error("assessment surface unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The decision gate is not available on this deployment.",
        ) from exc
    try:
        result: dict[str, Any] = await run_in_threadpool(call, *args, **kwargs)
        return result
    except EthicalConstraintViolationError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"refused": True, "check": exc.check, "reason": str(exc)},
        ) from exc
    except ConsentRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"refused": True, "check": "consent", "reason": str(exc)},
        ) from exc
    except CapabilityDisabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"refused": True, "check": "disabled", "reason": str(exc)},
        ) from exc
    except ExportRefused as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except InputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ImportError as exc:
        logger.error("assessment dependency missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A dependency of this capability is not installed on this deployment.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("assessment route failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during assessment.",
        ) from exc


#: The JSON request body. ``Annotated`` gives every route its own ``Body``
#: field: a shared ``Body(...)`` instance is mutated on first use and makes the
#: later routes expect ``{"payload": ...}``.
Payload = Annotated[
    dict[str, Any], Body(description="JSON request; see the route description for its fields.")
]


@router.post(
    "/statements/verify",
    summary="Verify Statements Against Evidence",
    description="""Corroboration (Bayesian log-odds over independence clusters),
contradictions (Allen interval algebra, impossible travel), per-speaker consistency
(SUE indicators) and verifiability. No output labels a person or an account.

Body: `statements` (required), `evidence`, `config`, `source_reliability`,
`verifiability_reference`.""",
    dependencies=[Depends(_bounded_body)],
)
async def verify_statements(
    payload: Payload,
    user: User = Depends(_principal(Permission.DETECT)),
) -> dict[str, Any]:
    """Verify statements against evidence."""
    from omni_mercury_engine.assessment import service

    return await _run(
        service.verify_statements, payload, surface="api.assessment.verify_statements"
    )


@router.post(
    "/concealed-information-test",
    summary="Score a Concealed Information Test",
    description="""Consent-gated. Measures recognition of event details, not deception;
a finding is not evidence of guilt. Off until the operator enables lawful basis
`consent` in `MERCURY_HUMAN_SUBJECT_CAPABILITIES`.

Body: `subject_pseudonym`, `administered_at`, `purpose`, `consent`
(ConsentRecord with scope `concealed_information_test`), `questions`; optional
`series`, `windows`, `config`, `probe_details_publicly_disclosed`.""",
    dependencies=[Depends(_bounded_body)],
)
async def concealed_information_test(
    payload: Payload,
    user: User = Depends(_principal(Permission.DETECT)),
) -> dict[str, Any]:
    """Score a consented CIT session."""
    from omni_mercury_engine.assessment import service

    return await _run(
        service.concealed_information_test,
        payload,
        surface="api.assessment.concealed_information_test",
    )


@router.post(
    "/vital-signs",
    summary="Contactless Vital Signs",
    description="""Respiration and heart rate from radar or Wi-Fi CSI; abstains rather
than guesses. Off until the operator enables the lawful basis used. Under
`consent`: `subject_pseudonym` and a ConsentRecord with scope
`contactless_vital_signs`. Under `vital_interests` (people who cannot consent,
such as survivors under rubble): an `attestation` `{incident_id, attested_by}`.
The respiration waveform is returned only with `include_waveform: true`.""",
    dependencies=[Depends(_bounded_body)],
)
async def vital_signs(
    payload: Payload,
    user: User = Depends(_principal(Permission.DETECT)),
) -> dict[str, Any]:
    """Estimate or track contactless vital signs."""
    from omni_mercury_engine.assessment import service

    return await _run(service.vital_signs, payload, surface="api.assessment.vital_signs")


@router.post(
    "/intel/screen",
    summary="Counterintelligence Screening",
    description="""Source reliability, circular reporting, coordinated reporting,
fabrication screens and impossible travel over a report batch. Never produces a
score or threat level; flags are leads for an analyst.

Body: `intel_reports`, optional `history`.""",
    dependencies=[Depends(_bounded_body)],
)
async def screen_intel_reports(
    payload: Payload,
    user: User = Depends(_principal(Permission.DETECT)),
) -> dict[str, Any]:
    """Screen intelligence reports."""
    from omni_mercury_engine.assessment import service

    return await _run(
        service.screen_intel_reports, payload, surface="api.assessment.screen_intel_reports"
    )


@router.post(
    "/intel/fuse",
    summary="All-Source Intelligence Fusion",
    description="""Screened, evidence-scored threat assessment. The fusion network
decides only after merit-gated weights are admitted against held-out adjudicated
labels (`/intel/fusion-status` says which path decides).

Body: `intel_reports`, optional `temporal_context`, `history`.""",
    dependencies=[Depends(_bounded_body)],
)
async def fuse_intelligence(
    payload: Payload,
    user: User = Depends(_principal(Permission.DETECT)),
) -> dict[str, Any]:
    """Fuse intelligence reports."""
    from omni_mercury_engine.assessment import service

    return await _run(
        service.fuse_intelligence, payload, surface="api.assessment.fuse_intelligence"
    )


@router.get("/intel/fusion-status", summary="Which Path the Fusion Engine Decides With")
async def fusion_status(user: User = Depends(_principal(Permission.READ))) -> dict[str, Any]:
    """Report evidence vs network scoring, and the merit result or refusal."""
    from omni_mercury_engine.assessment import service

    return await _run(service.fusion_status)


@router.get("/adjudication/status", summary="Adjudication Ledger Status")
async def adjudication_status(user: User = Depends(_principal(Permission.READ))) -> dict[str, Any]:
    """Chain validity, agreement, counts and work queues."""
    from omni_mercury_engine.assessment import service

    return await _run(service.adjudication_status)


@router.get(
    "/adjudication/cases/{case_id}/rater-view",
    summary="Blind Rater View of a Case",
    description="What the calling principal may see: the outcome, their own rating, "
    "and other ratings only after they have rated.",
)
async def adjudication_rater_view(
    case_id: str, user: User = Depends(_principal(Permission.READ))
) -> dict[str, Any]:
    """The caller's blind view of a case."""
    from omni_mercury_engine.assessment import service

    return await _run(service.adjudication_rater_view, user.id, case_id)


@router.post("/adjudication/cases", summary="Record a Case", dependencies=[Depends(_bounded_body)])
async def adjudication_add_case(
    payload: Payload, user: User = Depends(_principal(Permission.WRITE))
) -> dict[str, Any]:
    """Record a case; its split is fixed by its id."""
    from omni_mercury_engine.assessment import service

    return await _run(
        service.adjudication_add_case, payload, surface="api.assessment.adjudication_add_case"
    )


@router.post("/adjudication/outcomes", summary="Record a Case Outcome")
async def adjudication_add_outcome(
    payload: Payload, user: User = Depends(_principal(Permission.WRITE))
) -> dict[str, Any]:
    """Record what happened in a case."""
    from omni_mercury_engine.assessment import service

    return await _run(
        service.adjudication_add_outcome,
        payload,
        surface="api.assessment.adjudication_add_outcome",
    )


@router.post(
    "/adjudication/ratings",
    summary="Record a Blind Rating",
    description="Recorded under the calling principal's id and the server's clock.",
)
async def adjudication_add_rating(
    payload: Payload, user: User = Depends(_principal(Permission.WRITE))
) -> dict[str, Any]:
    """Record the caller's blind rating."""
    from omni_mercury_engine.assessment import service

    return await _run(
        service.adjudication_add_rating,
        payload,
        rater_id=user.id,
        surface="api.assessment.adjudication_add_rating",
    )


@router.post(
    "/adjudication/decisions",
    summary="Record an Adjudicator Decision",
    description="Needs the 'adjudicator' role or 'admin'; recorded under the caller's id.",
)
async def adjudication_add_decision(
    payload: Payload, user: User = Depends(_adjudicator)
) -> dict[str, Any]:
    """Record the caller's adjudicator decision."""
    from omni_mercury_engine.assessment import service

    return await _run(
        service.adjudication_add_decision,
        payload,
        adjudicator_id=user.id,
        surface="api.assessment.adjudication_add_decision",
    )


class ExportRequest(BaseModel):
    """Which split to export."""

    split: str = Field(default="heldout", description="'heldout' or 'training'")


@router.post(
    "/adjudication/export",
    summary="Export One Split's Labels",
    description="Returns the export document, ML-DSA-65 signed when the server holds "
    "a signing key (MERCURY_ADJUDICATION_SIGNING_KEY).",
)
async def adjudication_export(
    request: ExportRequest, user: User = Depends(_principal(Permission.EXPORT))
) -> dict[str, Any]:
    """Export a split's labels."""
    from omni_mercury_engine.assessment import service

    return await _run(service.adjudication_export, request.split)
