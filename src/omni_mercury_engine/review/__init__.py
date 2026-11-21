# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Review an uploaded document, image or recording: is some part of it unlike the rest?

An upload has no population to be compared with, so it is measured against
itself.  :func:`review_bytes` decodes it within stated bounds
(:mod:`~omni_mercury_engine.review.decode`), passes what the caller sent
through the harm-uplift gate, cuts it into segments and standardizes each
segment's features against the artifact's own bulk
(:mod:`~omni_mercury_engine.review.features`), and scores that matrix on the
same path every flagship detection takes: the fusion network, the conformal
certificate, the ethical gates and the decision layer of
:meth:`OmniMercuryEngine.detect_with_fusion`.  What comes back is an ordinary
detection result with its ``DecisionRecord``, plus a ``review`` block that
says which segments were flagged, where they are, and which features set them
apart.

There is one way in.  :func:`prepare` is the only function here that turns
bytes into something scoreable, and the step that measures a decoded upload
carries the ``GATED_BOUNDARY`` capability contract: the gate reads what the
caller sent before a single feature is computed, and deleting the annotation
is a test failure.  :func:`review_prepared` scores only what :func:`prepare`
returned.  The API, the CLI and :func:`review_bytes` are those two calls, so
no surface can review an upload the gate did not read.

Three things make the decision honest about what it is:

* the certificate is **batch-level** (regime ``batch_max``): it was calibrated
  on whole reference artifacts, each represented by its most anomalous
  segment, so it covers the question asked -- does this artifact contain an
  anomalous segment -- instead of a per-row guarantee misapplied to a maximum;
* before the certificate is claimed, the artifact is asked whether it is one
  of the family the certificate was calibrated on (the engine's batch-family
  test); when it is not, the decision defers;
* each kind's checkpoint states, in its own provenance, what it was calibrated
  on and what that has been measured to cover; the statement rides into
  every record as a caveat (``certificate scope: ...``).

The scope is internal consistency.  An artifact that is unusual as a whole,
or altered throughout, has no bulk to be measured against and is not what
this finds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.agentic.capabilities.contract import Invariant, capability_contract
from omni_mercury_engine.cognitive.decision_gate import DecisionSubject
from omni_mercury_engine.review.answers import three_answers
from omni_mercury_engine.review.decode import (
    AUDIO,
    DOCUMENT,
    IMAGE,
    KINDS,
    DecodeBounds,
    Decoded,
    ReviewError,
    decode,
    sniff,
)
from omni_mercury_engine.review.features import MIN_SEGMENTS, WIDTH, Segmented, extract

if TYPE_CHECKING:
    from omni_mercury_engine.engine import OmniMercuryEngine

__all__ = [
    "AUDIO",
    "BASIS",
    "DOCUMENT",
    "IMAGE",
    "KINDS",
    "MIN_SEGMENTS",
    "SCHEMA",
    "WIDTH",
    "DecodeBounds",
    "Prepared",
    "ReviewError",
    "build_engine",
    "checkpoint_path",
    "decode",
    "extract",
    "prepare",
    "review_bytes",
    "review_prepared",
    "review_segments",
    "sniff",
    "three_answers",
]

SCHEMA = "mercury.review/v1"

#: What a review measures, stated with every result.
BASIS = (
    "self-referenced: each segment is scored against the bulk of this same artifact "
    "(robust z-scores of its features). A flagged segment is unlike the rest of this "
    "artifact; that is not a finding that it was altered, and an artifact that is unusual "
    "as a whole is not what this detects."
)

#: Most flagged segments described in a result (the per-row flags are complete).
_MAX_DESCRIBED = 20
#: Longest upload name kept (it is caller text: read by the gate, echoed in the result).
_MAX_FILENAME_CHARS = 255
#: The surface and operation the harm-uplift gate is told it is deciding.  A
#: description of the code path, fixed, so it cannot be tuned per call.
_GATE_SURFACE = "omni_mercury_engine.review"
_GATE_OPERATION = (
    "measure each part of an uploaded {kind} against the rest of the same {kind} and emit a "
    "calibrated anomaly verdict"
)


def checkpoint_path(kind: str) -> Path:
    """The shipped review checkpoint for ``kind`` (``document``/``image``/``audio``)."""
    if kind not in KINDS:
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of {KINDS}")
    return Path(__file__).resolve().parent.parent / "models" / "checkpoints" / f"review_{kind}.pt"


def build_engine(kind: str, *, ledger: Any = None) -> OmniMercuryEngine:
    """An engine serving the review checkpoint for ``kind``, decision layer on.

    The cognitive critique is not enabled: its instruments reason about a
    time-ordered stream and a caller's history of comparable windows, and an
    artifact's segments are neither.

    Args:
        kind: ``document``, ``image`` or ``audio``.
        ledger: A :class:`~omni_mercury_engine.decision.DecisionLedger` every
            decision is recorded into (a server keeps one audit trail).

    Raises:
        ReviewError: 503 when the checkpoint is not installed.
    """
    from omni_mercury_engine.engine import OmniMercuryEngine

    path = checkpoint_path(kind)
    if not path.is_file():
        raise ReviewError(
            f"the {kind} review checkpoint is not installed ({path.name})", status=503
        )
    engine = OmniMercuryEngine(mode="fusion")
    engine.load_model(str(path))
    engine.enable_decision_layer(ledger=ledger)
    return engine


def _describe(segmented: Segmented, result: dict[str, Any]) -> dict[str, Any]:
    """Which segments were flagged, where they are, and what sets each apart."""
    probabilities = np.asarray(result.get("anomaly_probs", []), dtype=np.float64).ravel()
    flags = np.asarray(result.get("is_anomaly_rows", []), dtype=bool).ravel()
    order = [int(i) for i in np.argsort(-probabilities) if flags[i]][:_MAX_DESCRIBED]

    def describe(index: int) -> dict[str, Any]:
        row = segmented.matrix[index]
        leading = np.argsort(-np.abs(row))[:3]
        return {
            **segmented.segments[index],
            "probability": float(probabilities[index]),
            "top_features": [
                {"name": segmented.names[int(j)], "robust_z": round(float(row[int(j)]), 2)}
                for j in leading
            ],
        }

    top = int(result["top_row"]) if isinstance(result.get("top_row"), int) else None
    return {
        "flagged_count": int(flags.sum()),
        "flagged_segments": [describe(index) for index in order],
        "top_segment": describe(top) if top is not None and top < len(segmented.segments) else None,
    }


def review_segments(
    segmented: Segmented, engine: OmniMercuryEngine, **detect: Any
) -> dict[str, Any]:
    """Score an artifact's segments on the flagship path and describe what was flagged.

    Args:
        segmented: The artifact as segments x features.
        engine: An engine serving this kind's review checkpoint.
        **detect: Passed to :meth:`OmniMercuryEngine.detect_with_fusion`
            (``domain``, ``reference_scope`` ...).

    Returns:
        The detection result (with its ``decision`` record) plus the per-segment
        description under ``review``.
    """
    result: dict[str, Any] = engine.detect_with_fusion(segmented.matrix, **detect)
    result["review"] = _describe(segmented, result)
    return result


@dataclass(frozen=True)
class Prepared:
    """An upload that has been decoded, read by the harm-uplift gate, and measured.

    Built by :func:`prepare` only; :func:`review_prepared` scores nothing else.

    Attributes:
        segmented: The artifact as segments x features.
        format: The format decoded (``pdf``, ``png``, ``wav`` ...).
        filename: The upload's base name, bounded, or ``None``.
        size: The upload's size in bytes.
        sha256: Its SHA-256, so the reviewed bytes can be identified later.
        decode_detail: What the decoder recorded (encoding, channels ...).
        harm_gate: What the harm-uplift gate read before permitting the review.
    """

    segmented: Segmented
    format: str
    filename: str | None
    size: int
    sha256: str
    decode_detail: dict[str, Any]
    harm_gate: dict[str, Any]

    @property
    def kind(self) -> str:
        """``document``, ``image`` or ``audio``."""
        return self.segmented.kind


def _review_subject(args: tuple[Any, ...], kwargs: dict[str, Any]) -> DecisionSubject:
    """The decision the harm-uplift gate is asked about: the caller's own upload.

    The gate reads text.  For a document that is the document's text; for
    every kind it is also the upload's name.
    """
    decoded: Decoded = kwargs["decoded"] if "decoded" in kwargs else args[0]
    filename = kwargs["filename"] if "filename" in kwargs else args[1]
    return DecisionSubject(
        surface=_GATE_SURFACE,
        operation=_GATE_OPERATION.format(kind=decoded.kind),
        request=decoded.text or "",
        payload={"filename": filename or ""},
    )


@capability_contract(
    Invariant.GATED_BOUNDARY, boundary_subject=_review_subject, label="review.measure"
)
def _measure(decoded: Decoded, filename: str | None) -> Segmented:
    """Measure a decoded upload -- after the harm-uplift gate has read it.

    The contract runs :func:`~omni_mercury_engine.cognitive.decision_gate.enforce_decision_boundary`
    over :func:`_review_subject` before this body: a refusal (or a gate that
    cannot decide) raises and nothing is measured.
    """
    return extract(decoded)


def _gate_summary(decoded: Decoded, filename: str | None) -> dict[str, Any]:
    """What the gate read on the way in, stated with the result.

    Pixels and samples are numbers: words inside an image or spoken in a
    recording are not read, and the summary says so.
    """
    return {
        "check": "harm_uplift",
        "passed": True,
        "characters_read": len(decoded.text or "") + len(filename or ""),
        "read": (
            "the document's text and the upload's name"
            if decoded.kind == DOCUMENT
            else (
                f"the upload's name (the {decoded.kind}'s content is numeric; words in it "
                "are not read)"
            )
        ),
    }


def prepare(
    data: bytes,
    filename: str = "",
    *,
    kind: str | None = None,
    bounds: DecodeBounds | None = None,
) -> Prepared:
    """Decode an upload, pass it through the harm-uplift gate, and measure it.

    The order is the point: nothing is measured, and nothing can be scored,
    before the gate has read what the caller sent.

    Args:
        data: The uploaded bytes.
        filename: The upload's name (a hint for recognising its format).
        kind: Require this kind; an upload recognised as another is refused.
        bounds: Decode bounds (defaults to :class:`DecodeBounds`).

    Returns:
        The upload, ready for :func:`review_prepared`.

    Raises:
        ReviewError: 413 / 415 / 422 with the reason.
        EthicalConstraintViolationError: When the harm-uplift gate refuses.
    """
    name = Path(filename).name[:_MAX_FILENAME_CHARS] if filename else None
    decoded = decode(data, filename, kind=kind, bounds=bounds)
    segmented = _measure(decoded, name)  # gated: raises before measuring on a refusal
    return Prepared(
        segmented=segmented,
        format=decoded.format,
        filename=name,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        decode_detail=dict(decoded.detail or {}),
        harm_gate=_gate_summary(decoded, name),
    )


def review_prepared(prepared: Prepared, engine: OmniMercuryEngine) -> dict[str, Any]:
    """Score a prepared upload on the flagship path and describe what was flagged.

    Args:
        prepared: What :func:`prepare` returned.
        engine: An engine serving this kind's review checkpoint
            (:func:`build_engine`).

    Returns:
        A flagship-shaped detection result with ``decision`` and ``review``.

    Raises:
        EthicalConstraintViolationError: When an engine gate refuses.
    """
    segmented = prepared.segmented
    result = review_segments(segmented, engine)
    scope = (result.get("model_provenance") or {}).get("certificate_scope")
    result["review"] = {
        "schema": SCHEMA,
        "kind": segmented.kind,
        "format": prepared.format,
        "filename": prepared.filename,
        "bytes": prepared.size,
        "sha256": prepared.sha256,
        "segments": len(segmented.segments),
        "segmentation": {**prepared.decode_detail, **segmented.detail},
        "features": list(segmented.names),
        "basis": BASIS,
        "certificate_scope": scope,
        "harm_gate": dict(prepared.harm_gate),
        **result["review"],
    }
    # Why this segment, how sure, and the record: stated in plain language,
    # each beside the fields it was read from.
    result["review"]["answers"] = three_answers(result)
    return result


def review_bytes(
    data: bytes,
    filename: str = "",
    *,
    engine: OmniMercuryEngine | None = None,
    kind: str | None = None,
    bounds: DecodeBounds | None = None,
) -> dict[str, Any]:
    """Review one uploaded artifact end to end.

    Args:
        data: The uploaded bytes.
        filename: The upload's name (a hint for recognising its format).
        engine: An engine serving the artifact kind's review checkpoint; built
            here when omitted (a server keeps one per kind).
        kind: Require this kind; an upload recognised as another is refused.
        bounds: Decode bounds (defaults to :class:`DecodeBounds`).

    Returns:
        A flagship-shaped detection result with ``decision`` and ``review``.

    Raises:
        ReviewError: 413 / 415 / 422 / 503 with the reason.
        EthicalConstraintViolationError: When an ethical gate refuses.
    """
    prepared = prepare(data, filename, kind=kind, bounds=bounds)
    return review_prepared(prepared, engine if engine is not None else build_engine(prepared.kind))
