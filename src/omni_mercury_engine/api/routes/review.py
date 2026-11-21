# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mercury Agent - Artifact Review API Routes.

Upload a document, an image or a recording and get a Mercury decision about
it: is some part of it unlike the rest?  The upload is decoded within the
served ``review_max_*`` bounds, read by the harm-uplift gate, reduced to a
segments x features matrix measured against the artifact itself
(:mod:`omni_mercury_engine.review`), and scored on the flagship path -- the
fusion network, the conformal certificate, the engine's ethical gates and the
decision layer.  The response is a flagship-shaped detection result with its
``decision`` record and a ``review`` block saying which segments were flagged.

Two ways to send the file, one handler behind both:

* ``POST /api/v1/review/upload`` -- ``multipart/form-data`` (``curl -F``);
* ``POST /api/v1/review/inline`` -- JSON with the file base64-encoded, for
  clients that only speak JSON (the operator console).

Every bound the flagship route enforces applies here, through the same
objects: the per-surface body cap (``review_max_body_bytes``), the rate
limiter, the flagship admission bound (a review holds one slot from decode to
decision), and the engine serialization lock.  One bound is the surface's
own: at most ``review_max_concurrent`` reviews are decoded and measured at
once, because that is where a review's memory goes.
"""

from __future__ import annotations

import base64
import binascii
import logging
import threading
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, User
from omni_mercury_engine.api.routes.detection import (
    _ethical_refusal_detail,
    _raise_if_too_large_to_assess,
    flagship_admission_slot,
    flagship_engine_lock,
)
from omni_mercury_engine.api.routes.export import record_detection
from omni_mercury_engine.serving_limits import serving_limits

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/review", tags=["Review"])

Kind = Literal["document", "image", "audio"]

_LIMITS = serving_limits()

#: What a review accepts, by kind (the decoders recognise an upload by its
#: leading bytes; the name only settles what the bytes leave open).
FORMATS: dict[str, tuple[str, ...]] = {
    "document": ("text", "markdown", "csv", "json", "html", "docx", "pdf"),
    "image": ("png", "jpeg", "gif", "bmp", "webp", "tiff"),
    "audio": ("wav",),
}

# Reviews decoded and measured at once.  Measuring a large upload is the costly
# part of a review (up to about 0.6 GB and a few seconds, measured), and the
# flagship admission bound alone would let eight run together; this is the
# bound on that work.  Beyond it a caller is refused with 429, not queued.
_review_slots = threading.BoundedSemaphore(_LIMITS.review_max_concurrent)
_review_refusals = {"rejected_total": 0}
_review_refusals_lock = threading.Lock()

# One engine per artifact kind, built on first use and kept, each serving its
# own review checkpoint.  Held and run under the flagship engine lock.
_review_engines: dict[str, Any] = {}
# Bounded audit trail of every review decision this server has made (built
# with the first engine); GET /api/v1/review serves its summary.
_review_ledger: Any = None

_DESCRIPTION = """
Review one uploaded **document**, **image** or **recording**: is some part of it
unlike the rest of the same artifact?

The upload is decoded within the served bounds, read by the harm-uplift gate
(a document's text and every upload's name), cut into segments, and each
segment's features are standardized against the artifact's own bulk. That
matrix is scored on the flagship path: the fusion network, a **batch-level**
conformal certificate (it covers "does this artifact contain an anomalous
segment"), the engine's ethical gates, and the decision layer. Before the
certificate is claimed the artifact is tested for membership of the family the
certificate was calibrated on; one outside it is deferred, not judged.

The response is a flagship-shaped detection result. `decision` is the
`DecisionRecord`; `review` names the flagged segments (character span and
excerpt, tile box, or time span), the features that set each apart, the
upload's SHA-256, what the gate read, and the checkpoint's own statement of
what its certificate has been measured to cover.

**Scope.** A flagged segment is unlike the rest of *this* artifact. That is not
a finding that it was altered, and an artifact that is unusual as a whole is
not what this detects.

**Accepted.** Documents: text, Markdown, CSV, JSON, HTML, DOCX, PDF (with a
text layer). Images: PNG, JPEG, GIF, BMP, WebP, TIFF. Audio: uncompressed WAV;
compressed audio is refused with **415** and the conversion to run.

**Refusals.** **413** over a bound (body, characters, pixels, seconds, or more
text than the gate reads), **415** a format that is not decoded, **422** a file
that cannot be read or is too short to be its own reference, **403** an
ethical-gate refusal (fail-closed, with a blocked `DecisionRecord`), **429**
the rate limit, the reviews-in-flight bound or the flagship admission bound.
"""


def _get_optional_user(
    api_key_user: User | None = Depends(APIKeyAuth(auto_error=False)),
    jwt_user: User | None = Depends(JWTAuth(auto_error=False)),
) -> User | None:
    """Get current user if authenticated (review is open, like detection)."""
    return api_key_user or jwt_user


def _served_bounds() -> Any:
    """The decode bounds this server enforces, from the single source of truth."""
    from omni_mercury_engine.review import DecodeBounds

    return DecodeBounds(
        max_text_chars=_LIMITS.review_max_text_chars,
        max_pixels=_LIMITS.review_max_pixels,
        max_audio_seconds=_LIMITS.review_max_audio_seconds,
    )


def _served_review_engine(kind: str) -> Any:
    """The served review engine for ``kind``, built once. Hold the engine lock."""
    global _review_ledger
    engine = _review_engines.get(kind)
    if engine is None:
        from omni_mercury_engine import review
        from omni_mercury_engine.decision import DecisionLedger

        if _review_ledger is None:
            _review_ledger = DecisionLedger(maxlen=4096)
        engine = review.build_engine(kind, ledger=_review_ledger)
        _review_engines[kind] = engine
    return engine


def _run_review(data: bytes, filename: str, kind: str | None) -> dict[str, Any]:
    """Decode, gate, measure and score one upload. Runs in a worker thread.

    Two bounds are held for all of it: a review slot (the uploads being
    decoded and measured at once) and a slot of the flagship admission bound
    (everything heavy on this server).  The engine lock is held only for the
    scoring itself, so a long decode does not stall the flagship route.

    Raises:
        HTTPException: 429 when either bound is full (nothing is held).
    """
    from omni_mercury_engine import review

    if not _review_slots.acquire(blocking=False):
        with _review_refusals_lock:
            _review_refusals["rejected_total"] += 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Artifact review is at capacity; retry shortly. "
                f"(reviews in flight at once: {_LIMITS.review_max_concurrent})"
            ),
            headers={"Retry-After": "2"},
        )
    try:
        with flagship_admission_slot():
            prepared = review.prepare(data, filename, kind=kind, bounds=_served_bounds())
            with flagship_engine_lock():
                result: dict[str, Any] = review.review_prepared(
                    prepared, _served_review_engine(prepared.kind)
                )
                return result
    finally:
        _review_slots.release()


async def _review(
    data: bytes, filename: str, kind: str | None, user: User | None
) -> dict[str, Any]:
    """The one handler behind both routes."""
    try:
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )
        from omni_mercury_engine.review import ReviewError
    except ImportError as e:
        logger.error("Artifact review unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact review is not available in this environment.",
        ) from e

    try:
        result: dict[str, Any] = await run_in_threadpool(_run_review, data, filename, kind)
    except ReviewError as e:
        # The decoder's own, bounded statement of why (and which bound).
        raise HTTPException(status_code=e.status, detail=str(e)[:400]) from e
    except EthicalConstraintViolationError as e:
        # Fail closed: the harm-uplift gate, or an engine gate, refused.
        logger.warning("Artifact review blocked by ethical gate '%s'", e.check)
        _raise_if_too_large_to_assess(e)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_ethical_refusal_detail(e, f"Review blocked by the '{e.check}' ethical gate."),
        ) from e
    except HTTPException:
        raise  # the 429 admission refusal, already shaped
    except ImportError as e:
        logger.error("Artifact review unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact review is not available in this environment.",
        ) from e
    except Exception as e:
        logger.error("Artifact review failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during artifact review.",
        ) from e

    described = result.get("review") or {}
    probability = result.get("anomaly_prob")
    probability = (
        float(probability)
        if isinstance(probability, (int, float)) and not isinstance(probability, bool)
        else None
    )
    await record_detection(
        user_id=user.id if user else "anonymous",
        method=f"review_{described.get('kind', 'artifact')}",
        # The upload itself is not kept; its digest identifies it.
        data=[described.get("sha256")],
        results={
            "anomalies": [bool(result.get("is_anomaly", False))],
            "scores": [probability] if probability is not None else [],
            "summary": {
                "anomaly_prob": probability,
                "kind": described.get("kind"),
                "format": described.get("format"),
                "segments": described.get("segments"),
                "flagged_segments": described.get("flagged_count"),
            },
        },
    )
    return result


@router.get(
    "",
    summary="What artifact review accepts, and its decision trail",
    description="""
The artifact kinds and formats `POST /api/v1/review/upload` accepts, the bounds
enforced on them (rendered from `omni_mercury_engine.serving_limits`, so the
numbers here are the numbers enforced), and the aggregate summary of the
bounded, process-wide audit trail of review decisions since this server
started (counts by transparency state, disposition and response action).
""",
)
async def describe_review(
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Serve the accepted formats, the enforced bounds and the review decision trail."""
    limits = serving_limits().to_dict()
    # Lock-free, like the flagship decisions route: reading the module global
    # is atomic and the ledger summarizes under its own lock.
    ledger = _review_ledger
    decisions: dict[str, Any] = {"total": 0, "recorded": False}
    if ledger is not None:
        decisions = {**ledger.summary(), "recorded": True}
    return {
        "schema": "mercury.review/v1",
        "formats": {kind: list(formats) for kind, formats in FORMATS.items()},
        "limits": {name: limits[name] for name in limits if name.startswith("review_")},
        "admission": {
            "max_concurrent": _LIMITS.review_max_concurrent,
            "rejected_total": _review_refusals["rejected_total"],
        },
        "decisions": decisions,
    }


@router.post(
    "/upload",
    summary="Review an uploaded document, image or recording",
    description=_DESCRIPTION,
)
async def review_upload(
    file: UploadFile = File(..., description="The document, image or recording to review."),
    kind: Kind | None = Form(
        default=None,
        description="Require this kind; an upload recognised as another is refused (422).",
    ),
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Review one file sent as ``multipart/form-data``."""
    data = await file.read()
    return await _review(data, file.filename or "", kind, user)


class InlineReviewRequest(BaseModel):
    """One file to review, carried in JSON."""

    content_base64: str = Field(
        ...,
        min_length=1,
        description="The file's bytes, base64-encoded (standard alphabet).",
    )
    filename: str = Field(
        default="",
        max_length=255,
        description="The file's name; used only when its bytes do not settle its format.",
    )
    kind: Kind | None = Field(
        default=None,
        description="Require this kind; an upload recognised as another is refused (422).",
    )


@router.post(
    "/inline",
    summary="Review a document, image or recording sent as JSON",
    description=(
        "The same review as `POST /api/v1/review/upload`, for clients that only speak JSON: "
        "the file travels base64-encoded in `content_base64`.\n" + _DESCRIPTION
    ),
)
async def review_inline(
    request: InlineReviewRequest,
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Review one file sent base64-encoded in a JSON body."""
    try:
        data = base64.b64decode(request.content_base64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'content_base64' is not valid base64",
        ) from e
    return await _review(data, request.filename, request.kind, user)
