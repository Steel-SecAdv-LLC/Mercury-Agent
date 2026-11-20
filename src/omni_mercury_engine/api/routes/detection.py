# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mercury Agent - Detection API Routes.

Extracted detection endpoints for modular organization.
Provides univariate, multivariate, and advanced detection methods.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from omni_mercury_engine._env import flagship_cognitive_enabled
from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, User
from omni_mercury_engine.api.detection_metrics import records_detection
from omni_mercury_engine.api.fusion_weights import (
    FUSION_WEIGHTS,
    as_matrix,
    column_statistics,
    deseasonalize,
    fuse,
    fuse_calibrated,
    normalized_weights,
    null_dof,
    series_components,
    spectral_peak_z,
)
from omni_mercury_engine.api.routes.export import record_detection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/detect", tags=["Detection"])

# ---------------------------------------------------------------------------
# Flagship fusion engine -- one process-wide instance, serialized.
#
# ``/detect/flagship`` invokes the full OmniMercuryEngine fusion path (trained
# fusion network + GOSNN scalar integration + the sigma_Immutable ethical gate),
# the same engine ``mercury-agent detect -d fusion`` runs. That engine is NOT
# safe to drive from multiple worker threads at once: the first call auto-fits
# the base detectors (a fit race), and the path reads/writes the process-global
# GOSNN singleton (cross-request scalar bleed). FastAPI runs a sync detection in
# a threadpool, so we build the engine once, lazily, and serialize every
# detection through a single lock. Correctness over throughput -- this is the
# flagship decision path, not a high-QPS lane. The lazy build also means a slim
# (no-torch) install degrades to a clean 503 on first call instead of failing at
# import time.
# ---------------------------------------------------------------------------
_flagship_lock = threading.Lock()
_flagship_engine: Any = None
# Bounded audit trail of the served flagship's decisions (built with the
# engine); GET /detect/flagship/decisions serves its summary, churn included.
_flagship_ledger: Any = None


def _ethical_refusal_detail(refusal: Any, message: str) -> dict[str, Any]:
    """Structured 403 payload for an ethical refusal.

    The exception has carried ``check``, ``score``, ``threshold`` and
    ``details`` all along; the route used to flatten them to a one-line
    string naming only the check, so an operator could not see what refused
    at what margin, under which ruleset — and the console showed the refusal
    as an authentication problem. The payload is bounded and JSON-safe:
    ``details`` values are truncated, never echoed at full length.

    Args:
        refusal: The ``EthicalConstraintViolationError`` (or compatible).
        message: The human-readable first line, kept for existing consumers.

    Returns:
        ``{"message", "ethics": {check, score, threshold, ruleset_version,
        details}, "decision"}`` — the ``decision`` is a full fail-closed
        :class:`~omni_mercury_engine.decision.record.DecisionRecord` (as a
        dict) whose ``signals.ethics_report`` carries the same structured
        report, so a blocked decision is reconstructable from its record
        exactly like a served verdict.
    """
    from omni_mercury_engine.core.centralized_constants import ETHICAL

    def bounded(value: Any, depth: int = 0) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            as_float = float(value)
            return as_float if math.isfinite(as_float) else None
        if isinstance(value, str):
            return value[:240]
        if depth >= 2:
            return str(value)[:240]
        if isinstance(value, Mapping):
            return {str(k)[:60]: bounded(v, depth + 1) for k, v in list(value.items())[:12]}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [bounded(v, depth + 1) for v in list(value)[:8]]
        return str(value)[:240]

    detail: dict[str, Any] = {
        "message": message,
        "ethics": {
            "check": str(getattr(refusal, "check", "unknown")),
            "score": bounded(getattr(refusal, "score", None)),
            "threshold": bounded(getattr(refusal, "threshold", None)),
            "ruleset_version": int(ETHICAL.RULESET_VERSION),
            "details": bounded(getattr(refusal, "details", None) or {}),
        },
    }
    # Every blocked decision carries a full fail-closed DecisionRecord, not
    # only the refusal report: the decider's ethical Stage 1 (UNDECIDABLE /
    # HOLD) was unreachable from serving because the gates raise before a
    # record exists — this is where the blocked branch gets its record. The
    # layer is pure and deterministic, and must never turn a refusal into a
    # 500, hence the broad guard: a record-construction failure degrades to
    # the structured refusal alone.
    try:
        from omni_mercury_engine.decision import DecisionAbstentionResponder, Evidence

        record = DecisionAbstentionResponder().decide_from_evidence(
            Evidence.from_ethical_block(detail["ethics"])
        )
        detail["decision"] = record.to_dict()
    except Exception:  # pragma: no cover - defensive: refusal must still serve
        logger.exception("decision record for ethical refusal could not be built")
    return detail


def _raise_if_too_large_to_assess(refusal: Exception) -> None:
    """Answer 413 when the gate refused a request for carrying more text than it reads.

    The refusal is the harm gate's and keeps its check code. But it is about
    how much was sent, not about what it said, and a 403 naming the harm gate
    gave the caller nothing to act on.
    """
    from omni_mercury_engine.cognitive.decision_gate import refused_for_size

    too_much = refused_for_size(refusal)
    if too_much is not None:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request not assessed by the harm-uplift decision gate: {too_much}.",
        ) from refusal


def _enforce_route_gate(surface: str, operation: str, payload: Any) -> None:
    """Run the fail-closed harm-uplift gate over a detection route's request.

    ``cognitive/decision_gate.py`` is "the single fail-closed choke point every
    public decision surface passes through"; /fusion, /three-r, /tier and /rca
    are public decision surfaces, so they pass through it exactly like
    ``/detect/neurosymbolic`` does: the gate reads the caller's whole payload
    (every string in it, at any depth), a refusal maps to HTTP 403 (or 413 when
    the request carried more text than the gate reads), and an unavailable or
    broken gate is a 503/403, never a silent allow.

    Raises:
        HTTPException: 503 when the gate machinery cannot be imported, 413 when
            the request was too large to assess, 403 when the gate refuses
            (including fail-closed on any internal gate error).
    """
    # Import the gate machinery up front so a missing module is a clean 503
    # (a genuine environment fault), mirroring the neuro-symbolic endpoint.
    try:
        from omni_mercury_engine.cognitive.decision_gate import (
            DecisionSubject,
            enforce_decision_boundary,
        )
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )
    except ImportError as e:
        logger.error("Harm-uplift gate unavailable for %s: %s", surface, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The decision gate is not available in this environment.",
        ) from e

    try:
        enforce_decision_boundary(
            DecisionSubject(surface=surface, operation=operation, payload=payload)
        )
    except EthicalConstraintViolationError as e:
        # Fail closed: the harm-uplift gate refused this request.
        logger.warning("%s blocked by harm-uplift gate '%s'", surface, e.check)
        _raise_if_too_large_to_assess(e)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_ethical_refusal_detail(
                e, f"Request refused by the harm-uplift decision gate (check={e.check})."
            ),
        ) from e


def _served_flagship_engine() -> Any:
    """Return the served flagship engine, building it once. Hold ``_flagship_lock``.

    Shared by the detection route and the model-description route so both
    describe the same loaded checkpoint.
    """
    global _flagship_engine
    if _flagship_engine is None:
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
        engine.load_default_fusion_checkpoint()
        # Deployment posture: the served flagship closes the loop -- every
        # detection carries a ``decision`` record (grounded verdict or
        # explicit abstention plus a bounded, non-destructive response plan).
        # Additive only: the detection route returns the raw result dict, so
        # existing consumers see one extra key. The engine constructor already
        # builds a default responder (the layer is ON by default engine-wide);
        # the explicit call here is what attaches the served ledger below.
        #
        # The layer records into a bounded, process-wide ledger (the "verify"
        # step): GET /detect/flagship/decisions serves its summary, including
        # the measured decision churn (consecutive-disposition flip fraction).
        # Before this, DecisionLedger existed but no served path ever wrote
        # to one, so churn was a library-only number.
        from omni_mercury_engine.decision import DecisionLedger

        global _flagship_ledger
        _flagship_ledger = DecisionLedger(maxlen=4096)
        engine.enable_decision_layer(ledger=_flagship_ledger)
        # Cognitive self-critique on the served flagship, on by default:
        # every detection also carries ``cognitive_analysis`` (reliability,
        # confidence and uncertainties with their measured/placeholder flags,
        # historical analogs), and the decision record's signals keep a
        # bounded summary of it.
        # Measured cost on single-row serving with the shipped checkpoint
        # (median of 12 warm calls, 2026-11-20): 9.1 -> 139.4 ms per call.
        # Operators who need the latency back set
        # MERCURY_FLAGSHIP_COGNITIVE=off (the k6/locust load lanes exercise
        # /detect/univariate and are unaffected). The orchestrator's working
        # memory is bounded (OBSERVATION_MEMORY_LIMIT observation nodes,
        # oldest evicted), so a long-lived served engine does not grow per
        # anomalous request.
        if flagship_cognitive_enabled():
            # Curiosity novelty is ON for the served flagship: the signal is
            # measured to discriminate (read-only mean AUROC 0.9282 over the
            # 15 recorded evaluations, artifacts/novelty_discovery.json), it
            # runs only on detected anomalies at O(features) cost, and it
            # rides the record's cognitive summary as a *signal* — never a
            # verdict-mover (that adoption would need its own harness).
            engine.enable_cognitive_analysis(enable_curiosity=True)
        _flagship_engine = engine
    return _flagship_engine


def _describe_flagship_model() -> dict[str, Any]:
    """The served model's input contract, under the serialization lock."""
    with _flagship_lock:
        engine = _served_flagship_engine()
        provenance = getattr(engine, "_fusion_provenance", None)
        return {
            "trained": bool(getattr(engine, "_fusion_trained", False)),
            "fitted_feature_width": engine.fitted_feature_width,
            "provenance": dict(provenance) if isinstance(provenance, dict) else None,
        }


def _run_flagship_detection(
    matrix: np.ndarray[Any, Any],
    domain: str | None,
    explain: bool,
    gdpr_report: bool = False,
    subject_id: str | None = None,
) -> dict[str, Any]:
    """Build (once) and run the flagship fusion engine under the serialization lock.

    Runs in a worker thread via :func:`run_in_threadpool`. Holding
    ``_flagship_lock`` across both the one-time build and every detection makes
    the shared engine safe against the auto-fit race and the GOSNN-singleton
    scalar bleed described above. Any :class:`EthicalConstraintViolationError`
    propagates to the caller, which maps it to HTTP 403.
    """
    with _flagship_lock:
        result: dict[str, Any] = _served_flagship_engine().detect_with_fusion(
            matrix,
            domain=domain,
            explain=explain,
            gdpr_report=gdpr_report,
            subject_id=subject_id,
        )
        return result


class NeurosymbolicRequest(BaseModel):
    """Request for neuro-symbolic anomaly detection."""

    data: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of data entries for neuro-symbolic analysis",
    )
    sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity",
    )
    fusion_strategy: str = Field(
        default="confidence_weighted",
        description="Fusion strategy: weighted_average, attention, gated, confidence_weighted",
    )
    include_explanations: bool = Field(
        default=True,
        description="Include detailed explanations",
    )


class FusionRequest(BaseModel):
    """Request for multi-detector fusion analysis."""

    data: list[float] | list[list[float]] = Field(
        ...,
        min_length=3,
        description="Time series data (univariate or multivariate)",
    )
    detectors: list[str] = Field(
        default=["statistical", "temporal", "collective", "dimensional"],
        description=(
            "Detectors to include in fusion: statistical (largest per-feature |z|), temporal "
            "(largest step between consecutive rows), collective (a stretch of rows departing "
            "together: a spread spike, ramp, level shift or burst), dimensional (the row's "
            "standardised norm) and spectral (a dominant frequency, Fisher's test)."
        ),
    )
    sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity",
    )
    return_contributions: bool = Field(
        default=True,
        description="Return per-detector contributions",
    )


#: /three-r stability: the submitted series is cut into this many consecutive
#: segments (the estimator compares its first and last 10 scores, so it needs
#: more than 10), each at least this long for its own scores to mean anything.
_STABILITY_SEGMENTS: int = 20
_STABILITY_MIN_SEGMENT: int = 16


class ThreeRRequest(BaseModel):
    """Request for 3R mechanism analysis (Recursion-Resonance-Refactoring)."""

    data: list[float] = Field(
        ...,
        min_length=3,
        description="Time series data for 3R analysis",
    )
    recursion_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Recursion depth for hierarchical analysis",
    )
    harmonic_bands: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of harmonic frequency bands",
    )
    ethical_threshold: float = Field(
        default=0.96,
        ge=0.90,
        le=0.99,
        description=(
            "Score-scaling base for the 3R fusion equation, NOT an ethics control. "
            "The fused score is multiplied by this value raised to the golden "
            "ratio (eta**1.618), so it scales every score monotonically: 0.90 "
            "scales by 0.845, 0.99 by 0.984. Raising it therefore RAISES scores "
            "(more detections) -- the opposite of what 'stricter ethical "
            "threshold' suggests. It refuses nothing. Ethics enforcement is the "
            "fail-closed harm-uplift gate in cognitive.decision_gate, which no "
            "request field can tune."
        ),
    )


class NeurosymbolicResponse(BaseModel):
    """Response from neuro-symbolic detection."""

    overall_score: float
    overall_confidence: float
    neural_contribution: float
    symbolic_contribution: float
    patterns_detected: int
    rules_fired: int
    anomaly_scores: list[dict[str, Any]]
    decision: str
    explanation: str
    audit_trail: list[dict[str, Any]]


class FusionResponse(BaseModel):
    """Response from fusion detection."""

    is_anomaly: bool
    fused_score: float
    confidence: float
    detector_scores: dict[str, float]
    detector_contributions: dict[str, float]
    threshold: float
    explanation: str


class ThreeRResponse(BaseModel):
    """Response from 3R mechanism analysis."""

    fusion_score: float
    recursion_score: float
    resonance_score: float
    optimization_score: float
    ethical_scaling: float
    lyapunov_bound: float
    is_stable: bool = Field(
        description=(
            "Whether the fusion score contracted across the consecutive segments of the "
            "submitted series. Meaningful only when stability_evaluated is true."
        )
    )
    stability_evaluated: bool = Field(
        default=False,
        description=(
            "False when the series is too short to form the score trajectory a stability "
            "estimate needs; is_stable is then false because nothing was measured."
        ),
    )
    weights: dict[str, float]
    harmonic_analysis: dict[str, Any]


def _get_optional_user(
    api_key_user: User | None = Depends(APIKeyAuth(auto_error=False)),
    jwt_user: User | None = Depends(JWTAuth(auto_error=False)),
) -> User | None:
    """Get current user if authenticated."""
    return api_key_user or jwt_user


@router.post(
    "/neurosymbolic",
    response_model=NeurosymbolicResponse,
    summary="Neuro-Symbolic Detection",
    description="""
Perform hybrid neural-symbolic anomaly detection using the NeurosymbolicFusionEngine.

## Features
- Neural pattern detection via memory embeddings
- Symbolic logic reasoning with rule-based inference
- Attention-based fusion of neural and symbolic outputs
- Fail-closed harm-uplift decision gate (`cognitive.decision_gate`); the
  benevolence score is reported, not enforced
- Full audit trail for explainability

## Fusion Strategies
- **weighted_average**: Fixed neural/symbolic weight combination
- **attention**: Learned attention-based weighting
- **gated**: Confidence-gated information flow
- **confidence_weighted**: Dynamic weighting based on component confidence
""",
)
@records_detection("neurosymbolic")
async def detect_neurosymbolic(
    request: NeurosymbolicRequest,
    user: User | None = Depends(_get_optional_user),
) -> NeurosymbolicResponse:
    """Perform neuro-symbolic anomaly detection."""
    # Import the gate machinery up front so its ``EthicalConstraintViolationError``
    # name is always bound when the except clauses below are evaluated (mirrors
    # the flagship endpoint). These are core modules; a failure here is a genuine
    # environment fault, surfaced as 503 rather than swallowed as a 500.
    try:
        from omni_mercury_engine.cognitive.decision_gate import (
            DecisionSubject,
            enforce_decision_boundary,
        )
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )
    except ImportError as e:
        logger.error("Harm-uplift gate unavailable for neuro-symbolic endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The decision gate is not available in this environment.",
        ) from e

    try:
        from omni_mercury_engine.cognitive.neurosymbolic_fusion import (
            FusionStrategy,
            NeurosymbolicFusionEngine,
        )

        # Fail-closed harm-uplift gate. This endpoint's own OpenAPI description
        # advertises it, and ``NeurosymbolicFusionEngine`` does not run it
        # internally, so it must be enforced here or the documented control is
        # absent on a public decision surface. A refusal maps to HTTP 403 below.
        #
        # The records are the payload, which the gate reads whole: every
        # string in them, at any depth. Their text used to be folded into
        # ``request`` as well, from when a payload was read through a bounded
        # summary; with both, the same text was read twice and counted twice
        # against what one decision may carry, so a request half the size the
        # gate reads was refused for its size.
        enforce_decision_boundary(
            DecisionSubject(
                surface="api.detect_neurosymbolic",
                operation="neuro-symbolic anomaly detection over caller-supplied records",
                payload=request.data,
            )
        )

        strategy_map = {
            "weighted_average": FusionStrategy.WEIGHTED_AVERAGE,
            "attention": FusionStrategy.ATTENTION,
            "gated": FusionStrategy.GATED,
            "confidence_weighted": FusionStrategy.CONFIDENCE_WEIGHTED,
            "hierarchical": FusionStrategy.HIERARCHICAL,
        }

        strategy = strategy_map.get(
            request.fusion_strategy.lower(),
            FusionStrategy.CONFIDENCE_WEIGHTED,
        )

        engine = NeurosymbolicFusionEngine(
            fusion_strategy=strategy,
            confidence_threshold=request.sensitivity,
        )

        engine.ingest_data(request.data)
        result = engine.analyze()

        user_id = user.id if user else "anonymous"
        await record_detection(
            user_id=user_id,
            method="neurosymbolic",
            data=request.data,
            results={
                "anomalies": [s.is_anomaly for s in result.anomaly_scores],
                "scores": [s.anomaly_score for s in result.anomaly_scores],
                "summary": {
                    "overall_score": result.overall_score,
                    "patterns_detected": result.patterns_detected,
                },
            },
            sensitivity=request.sensitivity,
        )

        return NeurosymbolicResponse(
            overall_score=result.overall_score,
            overall_confidence=result.overall_confidence,
            neural_contribution=result.neural_contribution,
            symbolic_contribution=result.symbolic_contribution,
            patterns_detected=result.patterns_detected,
            rules_fired=result.rules_fired,
            anomaly_scores=(
                [
                    {
                        "score_id": s.score_id,
                        "anomaly_score": s.anomaly_score,
                        "neural_score": s.neural_score,
                        "symbolic_score": s.symbolic_score,
                        "confidence": s.confidence,
                        "category": s.category.value,
                        "is_anomaly": s.is_anomaly,
                        "explanation": s.explanation,
                    }
                    for s in result.anomaly_scores
                ]
                if request.include_explanations
                else []
            ),
            decision=result.decision.decision_type.value,
            explanation=result.explanation,
            audit_trail=result.audit_trail if request.include_explanations else [],
        )

    except EthicalConstraintViolationError as e:
        # Fail closed: the harm-uplift gate refused this request.
        logger.warning("Neuro-symbolic detection blocked by harm-uplift gate '%s'", e.check)
        _raise_if_too_large_to_assess(e)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_ethical_refusal_detail(
                e, f"Request refused by the harm-uplift decision gate (check={e.check})."
            ),
        ) from e
    except ImportError as e:
        logger.error("Neuro-symbolic module not available: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neuro-symbolic module is not available.",
        ) from e
    except Exception as e:
        logger.error("Neuro-symbolic detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during detection.",
        ) from e


def _spectral_peak_z(values: np.ndarray[Any, Any]) -> float:
    """How surprising the strongest periodic component is, in z-score units.

    The fusion score compares every detector's output with one threshold in
    standard-deviation units. The spectral score was ``max|FFT| / N`` of the
    raw data: its DC bin alone is the mean, so the same signal offset by 1000
    scored 1000 (and scaled by 0.001, 0.00015) while the statistical score
    stayed 3.07, and any data away from zero was an anomaly.

    This is Fisher's (1929) test for a dominant frequency, on the standardised
    series. Its periodogram ordinates (DC removed) are, under white noise,
    independent exponentials, so the strongest of the ``m`` ordinates has the
    tail ``1 - (1 - exp(-x))^m`` (x in units of the mean ordinate); that
    p-value is returned as the two-sided normal quantile, so 0 means "no more
    periodic than noise" and 2-5 spans the route's threshold range. Invariant
    to the data's offset and scale.

    Args:
        values: A one-dimensional series.

    Returns:
        The z-score, ``0.0`` for fewer than four samples or a constant series.
    """
    return spectral_peak_z(np.asarray(values, dtype=float))[0]


@router.post(
    "/fusion",
    response_model=FusionResponse,
    summary="Multi-Detector Fusion",
    description="""
Perform anomaly detection using multiple detector fusion.

Combines outputs from statistical, temporal, dimensional, and other
specialized detectors using weighted ensemble.
""",
)
@records_detection("fusion")
async def detect_fusion(
    request: FusionRequest,
    user: User | None = Depends(_get_optional_user),
) -> FusionResponse:
    """Perform multi-detector fusion analysis."""
    # Fail-closed harm-uplift gate, first, over the whole request (the
    # ``detectors`` list is caller text).  Same choke point as /neurosymbolic.
    _enforce_route_gate(
        surface="api.detect_fusion",
        operation="multi-detector fusion anomaly detection over a caller-supplied series",
        payload=request.model_dump(),
    )

    # M3: a detector list is validated against the engine's actual detector
    # set.  Unknown names used to be silently dropped, so a typo'd list scored
    # nothing and answered a *confident normal* (fused_score 0, is_anomaly
    # False, confidence 1.0).  An empty or unknown selection is the caller's
    # error: 422, never a fabricated verdict.
    unknown = sorted(set(request.detectors) - set(FUSION_WEIGHTS))
    if not request.detectors or unknown:
        known = ", ".join(sorted(FUSION_WEIGHTS))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"unknown detector name(s): {unknown!r}; " if unknown else "empty detector list; "
            )
            + f"valid detectors are: {known}",
        )

    # Rows of features (a 1-D series is one feature), standardised per
    # feature: the components the batch job's fusion scores per sample,
    # so the two routes agree on multi-feature input too.  A ragged 2-D
    # array is a malformed request (422), not an internal error (it used to
    # surface as a 500 from numpy's inhomogeneous-shape ValueError).
    try:
        matrix = as_matrix(request.data)
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'data' must be a rectangular 1-D or 2-D numeric array "
            "(every row the same length).",
        ) from e

    try:
        # A dominant periodic component is removed before the point components
        # score (a spike riding a strong season is one seasonal amplitude above
        # the crest, a z of 2 against the whole series' spread); the spectral
        # component reads the original series, where the season is its signal.
        residual, seasonal_periods = deseasonalize(matrix)
        mean, std = column_statistics(residual)

        threshold = 2.0 + (1.0 - request.sensitivity) * 3.0

        # Every component is the z-equivalent of its whole-series maximum under
        # the null, given the number of values that maximum ranged over
        # (``fusion_weights.series_components``). Before this calibration the
        # raw maxima met a fixed threshold, so a wide or long noise matrix was
        # "anomaly" by its size alone. The spectral peaks are taken per
        # feature, never over the flattened matrix (which would splice the
        # columns into one false series).
        spectral_peaks = (
            [_spectral_peak_z(matrix[:, j]) for j in range(matrix.shape[1])]
            if "spectral" in request.detectors
            else None
        )
        calibrated = series_components(
            residual, mean, std, spectral_z=spectral_peaks, dof=null_dof(matrix.shape[0])
        )
        detector_scores: dict[str, float] = {
            name: calibrated[name] for name in request.detectors if name in calibrated
        }

        # The weights are the table the batch route's fusion reads too, so the
        # same data reaches the same verdict whichever route scores it.
        detector_contributions = normalized_weights(request.detectors)
        # The weighted average, or one component decisive on its own (a
        # weighted average diluted a spectral z of 15 to 3.3 and read "normal").
        fused_score = fuse_calibrated(detector_scores, request.detectors)

        is_anomaly = fused_score > threshold

        explanation_parts = [
            f"{d}: {detector_scores.get(d, 0.0):.3f} (weight: {detector_contributions.get(d, 0.0):.2f})"
            for d in request.detectors
        ]
        explanation = f"Fused score {fused_score:.3f} from: " + ", ".join(explanation_parts)
        removed = [p for p in seasonal_periods if p]
        if removed:
            explanation += (
                "; a periodic component was removed before scoring (period "
                + ", ".join(str(p) for p in removed)
                + " rows)"
            )
        decisive = max((detector_scores.get(d, 0.0) for d in request.detectors), default=0.0)
        if fused_score > fuse(detector_scores, request.detectors) + 1e-12:
            explanation += f"; one component is decisive on its own ({decisive:.3f})"

        user_id = user.id if user else "anonymous"
        await record_detection(
            user_id=user_id,
            method="fusion",
            data=request.data,
            results={
                "anomalies": [is_anomaly],
                "scores": [fused_score],
                # The threshold the fused score was judged against: the agent
                # derives the severity band from the score-to-threshold ratio.
                "threshold": threshold,
                "summary": {"fused_score": fused_score, "detector_scores": detector_scores},
            },
            sensitivity=request.sensitivity,
        )

        return FusionResponse(
            is_anomaly=is_anomaly,
            fused_score=fused_score,
            confidence=(
                min(fused_score / threshold, 1.0) if is_anomaly else 1.0 - (fused_score / threshold)
            ),
            detector_scores=detector_scores,
            detector_contributions=detector_contributions if request.return_contributions else {},
            threshold=threshold,
            explanation=explanation,
        )

    except Exception as e:
        logger.error("Fusion detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during detection.",
        ) from e


@router.post(
    "/three-r",
    response_model=ThreeRResponse,
    summary="3R Mechanism Analysis",
    description="""
Perform anomaly detection using the 3R mechanism (Recursion-Resonance-Refactoring).

## Mathematical Framework
A = (w_R * R(x) + w_H * H(ω) + w_O * O(θ)) * η_Ethical^Φ

Where:
- R(x): Recursion score from hierarchical feature extraction
- H(ω): Harmonic resonance from frequency-domain analysis
- O(θ): Optimization score from adaptive enhancement
- η_Ethical: Ethical compliance threshold (0.93-0.96)
- Φ: Golden ratio (1.618)

## Lyapunov Stability
Guarantees convergence: V(S_t) ≤ ε × e^(-0.25t)
""",
)
@records_detection("three_r")
async def detect_three_r(
    request: ThreeRRequest,
    user: User | None = Depends(_get_optional_user),
) -> ThreeRResponse:
    """Perform 3R mechanism analysis."""
    # Fail-closed harm-uplift gate: /three-r is a public decision surface and
    # passes through the same choke point as every other one.
    _enforce_route_gate(
        surface="api.detect_three_r",
        operation="3R (recursion-resonance-refactoring) anomaly analysis over a series",
        payload=request.model_dump(),
    )

    try:
        from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation

        data = np.array(request.data)

        def compute_recursion_score(x: np.ndarray[Any, Any], depth: int) -> float:
            """Compute R(x) - hierarchical feature extraction."""
            if depth == 0 or len(x) < 2:
                return float(np.std(x) / (np.mean(np.abs(x)) + 1e-8))

            mid = len(x) // 2
            left_score = compute_recursion_score(x[:mid], depth - 1)
            right_score = compute_recursion_score(x[mid:], depth - 1)

            return float(
                0.5 * (left_score + right_score) + 0.5 * np.std(x) / (np.mean(np.abs(x)) + 1e-8)
            )

        phi = 1.618033988749895

        def three_r_scores(series: np.ndarray[Any, Any]) -> dict[str, Any]:
            """R, H and O of one series, plus its band analysis."""
            recursion_score = compute_recursion_score(series, request.recursion_depth)
            recursion_score = float(np.clip(recursion_score, 0.0, 1.0))

            fft_result = np.fft.fft(series)
            magnitudes = np.abs(fft_result)

            n_bands = min(request.harmonic_bands, len(magnitudes) // 2)
            band_size = len(magnitudes) // (2 * n_bands) if n_bands > 0 else 1

            band_energies = []
            for i in range(n_bands):
                start = i * band_size
                end = (i + 1) * band_size
                band_energy = float(np.sum(magnitudes[start:end] ** 2))
                band_energies.append(band_energy)

            total_energy = sum(band_energies) + 1e-8
            band_ratios = [e / total_energy for e in band_energies]

            expected_ratios = [1.0 / (phi**i) for i in range(n_bands)]
            expected_sum = sum(expected_ratios)
            expected_ratios = [r / expected_sum for r in expected_ratios]

            harmonic_deviation = sum(abs(a - e) for a, e in zip(band_ratios, expected_ratios))
            # Normalise by the maximum L1 distance between two probability vectors
            # (2.0). Both band_ratios and expected_ratios sum to 1, so the raw
            # deviation ranges over [0, 2] and ``1 - deviation`` is negative for any
            # spectrum that is not already phi-shaped -- which clipped the score to
            # a saturated 0 on real data. Measured on the four recorded NAB files
            # (128-point windows, stride 16): the served form scored at chance,
            # mean AUROC 0.511, and was a CONSTANT 0.0 across all 63 windows of
            # realTraffic/speed_7578.csv, so the stream carried no signal at all
            # while holding 27.6% of the OAE's weight. Dividing by 2 recovers it:
            # mean AUROC 0.755, and 0.849 on the file where it had been constant.
            resonance_score = float(np.clip(1.0 - harmonic_deviation / 2.0, 0.0, 1.0))

            signal_variance = np.var(series)
            noise_estimate = np.var(np.diff(series)) / 2
            snr = signal_variance / (noise_estimate + 1e-8)
            optimization_score = float(np.clip(1.0 / (1.0 + np.exp(-np.log10(snr + 1))), 0.0, 1.0))
            return {
                "recursion": recursion_score,
                "resonance": resonance_score,
                "optimization": optimization_score,
                "n_bands": n_bands,
                "band_energies": band_energies,
                "band_ratios": band_ratios,
                "harmonic_deviation": harmonic_deviation,
                "magnitudes": magnitudes,
            }

        scored = three_r_scores(data)
        recursion_score = scored["recursion"]
        resonance_score = scored["resonance"]
        optimization_score = scored["optimization"]
        n_bands = scored["n_bands"]
        band_energies = scored["band_energies"]
        band_ratios = scored["band_ratios"]
        harmonic_deviation = scored["harmonic_deviation"]
        magnitudes = scored["magnitudes"]

        aafe = OmniAvaEquation(
            ethical_compliance_threshold=request.ethical_threshold,
        )

        result = aafe.compute(
            recursion_score=recursion_score,
            resonance_score=resonance_score,
            optimization_score=optimization_score,
        )

        # Stability is a property of a TRAJECTORY of fusion scores, and one
        # request yields one score: asking the equation after a single
        # ``compute`` returned False for every input, whatever it contained.
        # The trajectory is taken from the request itself -- the same scoring
        # over its consecutive segments, in time order -- when the series is
        # long enough to give the estimator the history it needs; otherwise
        # the response says stability was not evaluated.
        stability_evaluated = len(data) >= _STABILITY_SEGMENTS * _STABILITY_MIN_SEGMENT
        is_stable = False
        if stability_evaluated:
            trajectory = OmniAvaEquation(ethical_compliance_threshold=request.ethical_threshold)
            for segment in np.array_split(data, _STABILITY_SEGMENTS):
                part = three_r_scores(segment)
                trajectory.compute(
                    recursion_score=part["recursion"],
                    resonance_score=part["resonance"],
                    optimization_score=part["optimization"],
                )
            is_stable = bool(trajectory.verify_lyapunov_stability()[0])

        user_id = user.id if user else "anonymous"
        await record_detection(
            user_id=user_id,
            method="three_r",
            data=request.data,
            results={
                "anomalies": [result.fusion_score > 0.5],
                "scores": [result.fusion_score],
                "summary": {
                    "recursion_score": recursion_score,
                    "resonance_score": resonance_score,
                    "optimization_score": optimization_score,
                },
            },
            sensitivity=1.0 - request.ethical_threshold,
        )

        return ThreeRResponse(
            fusion_score=result.fusion_score,
            recursion_score=result.recursion_score,
            resonance_score=result.resonance_score,
            optimization_score=result.optimization_score,
            ethical_scaling=request.ethical_threshold**phi,
            lyapunov_bound=result.lyapunov_bound,
            is_stable=is_stable,
            stability_evaluated=stability_evaluated,
            weights=result.fusion_weights,
            harmonic_analysis={
                "n_bands": n_bands,
                "band_energies": band_energies,
                "band_ratios": band_ratios,
                "harmonic_deviation": harmonic_deviation,
                "dominant_frequency_idx": int(np.argmax(magnitudes[: len(magnitudes) // 2])),
            },
        )

    except ImportError as e:
        logger.error("3R module not available: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="3R module is not available.",
        ) from e
    except Exception as e:
        logger.error("3R detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during detection.",
        ) from e


class TierDetectRequest(BaseModel):
    """Request for the streaming detector-tier ensemble."""

    data: list[float] = Field(
        ...,
        min_length=8,
        description=(
            "1-D anomaly series (the tier's native temporal contract). At least 8 "
            "points; the default tier includes spot_evt, which needs 10, so send 10 "
            "or more unless 'subset' leaves spot_evt out"
        ),
    )
    labels: list[int] | None = Field(
        default=None,
        description="Optional per-point 0/1 labels (enables supervised stacking/BMA)",
    )
    reference: list[float] | None = Field(
        default=None,
        min_length=8,
        description=(
            "Known-normal 1-D series to fit and calibrate on; 'data' is then only "
            "scored, so the threshold and conformal cut rest on evidence independent "
            "of it. Without it (and without labels) the tier calibrates on 'data' "
            "itself and flags about the configured share by construction. Mutually "
            "exclusive with 'labels'."
        ),
    )
    subset: list[str] | None = Field(
        default=None,
        description="Detector names to include (default: the full streaming tier)",
    )
    method: str | None = Field(
        default=None,
        description="Combiner: stacking | bma | average | consensus (default: auto)",
    )
    contamination: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description=(
            "Expected anomaly fraction. The tier thresholds at this quantile of its fused "
            "score, so the flag count is this share of the points by construction (measured "
            "slope 1.0 against the parameter); set conformal_alpha for flags whose false-positive "
            "rate is controlled by the data instead."
        ),
    )
    conformal_alpha: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Distribution-free false-positive rate; adds conformal flags",
    )
    include_counterfactual: bool = Field(
        default=False,
        description=(
            "Attach a verified minimal counterfactual for one point: the "
            "replacement value that flips its decision, re-scored through the "
            "same fitted ensemble (off by default)."
        ),
    )
    counterfactual_index: int | None = Field(
        default=None,
        description="Point to explain (default: highest-scoring flagged point).",
    )
    counterfactual_method: str = Field(
        default="prototype",
        pattern="^(wachter|dice|growing_spheres|prototype|genetic)$",
        description="Counterfactual search method.",
    )
    include_attribution: bool = Field(
        default=False,
        description="Also return the calibrated per-detector score matrix (which detectors fired)",
    )


@router.post(
    "/tier",
    summary="Streaming Detector-Tier Ensemble",
    description="""
Run the streaming / statistical / state-space detector-tier calibrated ensemble.

Returns per-point calibrated anomaly probabilities, flags at the calibrated
threshold, cross-detector uncertainty, and (when ``conformal_alpha`` is set)
flags with a distribution-free false-positive guarantee. Torch-free.
""",
)
@records_detection("tier")
async def detect_tier(
    request: TierDetectRequest,
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Run the detector-tier ensemble on a 1-D series."""
    # Fail-closed harm-uplift gate, before any work: ``subset``, ``method``
    # and ``counterfactual_method`` carry caller text.
    _enforce_route_gate(
        surface="api.detect_tier",
        operation="streaming detector-tier ensemble detection over a 1-D series",
        payload=request.model_dump(),
    )

    try:
        from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble

        series = np.asarray(request.data, dtype=float).ravel()
        labels = None if request.labels is None else np.asarray(request.labels, dtype=int)
        reference = (
            None if request.reference is None else np.asarray(request.reference, dtype=float)
        )
        subset = tuple(request.subset) if request.subset else None

        result = run_tier_ensemble(
            series,
            labels=labels,
            reference=reference,
            subset=subset,
            method=request.method,
            contamination=request.contamination,
            conformal_alpha=request.conformal_alpha,
            include_attribution=request.include_attribution,
            include_counterfactual=request.include_counterfactual,
            counterfactual_index=request.counterfactual_index,
            counterfactual_method=request.counterfactual_method,
        )

        user_id = user.id if user else "anonymous"
        await record_detection(
            user_id=user_id,
            method="detector_tier",
            data=request.data,
            results={
                # The keys record_detection reads: the history record and the
                # SIEM alert (which skips records with no anomalies) count
                # these, so they must be the tier's own flags and scores.
                "anomalies": result["flags"],
                "scores": result["scores"],
                "threshold": result["threshold"],
                "summary": {
                    "n_flagged": result["n_flagged"],
                    "n_points": result["n_points"],
                    "threshold": result["threshold"],
                    "combiner": result["method"],
                },
            },
        )
        return result

    except ValueError as e:
        # Bad request: unknown detector name, stacking-without-labels, bad alpha.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        logger.error("Tier detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during tier detection.",
        ) from e


class FlagshipDetectRequest(BaseModel):
    """Request for the flagship neuro-symbolic fusion engine."""

    data: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Rows of equal-length numeric feature vectors (a feature matrix).",
    )
    domain: str | None = Field(
        default=None,
        description="Optional domain for GOSNN threshold tuning (e.g. 'medical').",
    )
    explain: bool = Field(
        default=False,
        description=(
            "Attach an Integrated-Gradients attribution of this decision "
            "(expensive; off by default)."
        ),
    )
    gdpr_report: bool = Field(
        default=False,
        description=(
            "Attach a GDPR Art. 22-style explanation report with Wachter "
            "counterfactuals from the engine's explainability pipeline "
            "(expensive; off by default)."
        ),
    )
    subject_id: str | None = Field(
        default=None,
        description="Optional data-subject identifier recorded in the GDPR report.",
    )


@router.get(
    "/flagship/model",
    summary="Describe the served flagship model",
    description="""
The input contract of the model ``POST /detect/flagship`` serves: whether it is
trained, the number of feature columns it was fitted on (``null`` when the
model does not constrain the width), and its training provenance. A matrix of
any other width is refused with HTTP 400, so a client reads this first instead
of learning the width from a failed detection.
""",
)
async def describe_flagship_model(
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Report the served flagship model's width, training state and provenance."""
    try:
        description: dict[str, Any] = await run_in_threadpool(_describe_flagship_model)
    except ImportError as e:
        logger.error("Flagship fusion engine unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The flagship fusion engine is not available in this environment.",
        ) from e
    return description


@router.get(
    "/flagship/decisions",
    summary="Decision trail summary of the served flagship",
    description="""
Aggregates over the bounded, process-wide audit trail of `POST /detect/flagship`
decisions since this server started (ring buffer, most recent 4096): counts by
transparency state, disposition and response action, the abstention and
calibration fractions, and the measured decision **churn** — the fraction of
consecutive decisions whose disposition flipped. Empty counts mean no flagship
decision has been recorded yet.
""",
)
async def flagship_decision_summary(
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Serve the flagship decision ledger's aggregate summary."""
    with _flagship_lock:
        ledger = _flagship_ledger
    if ledger is None:
        return {"total": 0, "recorded": False}
    summary: dict[str, Any] = ledger.summary()
    summary["recorded"] = True
    return summary


@router.get(
    "/flagship/improvements",
    summary="Improvement Ledger: every adaptive change, with its rollback",
    description="""
Two trails. `runtime` is the bounded, process-wide Improvement Ledger: every
adaptive change applied in this server since it started — an online-recalibrated
operating threshold taking over or moving, a conformal certificate refit, the
label-scarcity schedule resolving the symbolic weight, probe decorrelation or
contrastive refinement rescaling weights — each with what changed, why, the
measured impact where one exists, and how to roll it back. `committed` is the
shipped record of measured adaptive-change *decisions* (adopted or refused,
each with its harness numbers), from `artifacts/improvement_ledger.json`.
""",
)
async def flagship_improvement_ledger(
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Serve the runtime Improvement Ledger and the committed decisions."""
    from omni_mercury_engine.governance.improvement_ledger import (
        default_improvement_ledger,
        published,
    )

    ledger = default_improvement_ledger()
    recent = [entry.to_dict() for entry in ledger.entries()[-64:]]
    return {
        "runtime": {"summary": ledger.summary(), "recent": recent},
        "committed": published(),
    }


@router.post(
    "/flagship",
    summary="Flagship Neuro-Symbolic Fusion Detection",
    description="""
Run Mercury's flagship anomaly detector: the full ``OmniMercuryEngine`` fusion
path -- the trained neural fusion network, GOSNN scalar integration, and the
``σ_Immutable`` second hard ethical gate -- loaded from the shipped default
fusion checkpoint. This is the same engine ``mercury-agent detect -d fusion``
runs, now reachable over HTTP.

Distinct from ``/detect/fusion`` (a lightweight statistical weighted ensemble):
this endpoint returns a *calibrated* anomaly probability, the decision, severity,
per-detector importances, and the ethical-gate metadata. A detection the ethical
gate refuses returns **HTTP 403** (fail-closed), never a silent allow.
""",
)
async def detect_flagship(
    request: FlagshipDetectRequest,
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Run the flagship neuro-symbolic fusion engine on a feature matrix."""
    # Import the gate exception up front so the fail-closed refusal maps to a
    # distinct 403 rather than being swallowed by the generic 500 handler.
    try:
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )
    except ImportError as e:
        logger.error("Flagship fusion engine unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The flagship fusion engine is not available in this environment.",
        ) from e

    try:
        matrix = np.asarray(request.data, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("'data' must be a 2-D feature matrix")

        # Serialize + offload the blocking, stateful fusion detection.
        # Annotated: run_in_threadpool is Any-typed under some Python/stub
        # combinations (CI's 3.12 lane), and the route declares dict[str, Any].
        result: dict[str, Any] = await run_in_threadpool(
            _run_flagship_detection,
            matrix,
            request.domain,
            request.explain,
            request.gdpr_report,
            request.subject_id,
        )

        user_id = user.id if user else "anonymous"
        # A result without a probability is recorded as having none ("scores"
        # stays empty and the summary carries None), never as a fabricated
        # 0.0 — the same placeholder the decision layer explicitly removed
        # from its records.
        recorded_prob = result.get("anomaly_prob")
        recorded_prob = (
            float(recorded_prob)
            if isinstance(recorded_prob, (int, float)) and not isinstance(recorded_prob, bool)
            else None
        )
        await record_detection(
            user_id=user_id,
            method="flagship_fusion",
            data=request.data,
            results={
                "anomalies": [bool(result.get("is_anomaly", False))],
                "scores": [recorded_prob] if recorded_prob is not None else [],
                "summary": {
                    "anomaly_prob": recorded_prob,
                    "severity": float(result.get("severity", 0.0)),
                    "class_prediction": result.get("class_prediction"),
                },
            },
        )
        return result

    except EthicalConstraintViolationError as e:
        # Fail closed: the flagship refused this input at a hard ethical gate.
        logger.warning("Flagship detection blocked by ethical gate '%s'", e.check)
        _raise_if_too_large_to_assess(e)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_ethical_refusal_detail(
                e, f"Detection blocked by the '{e.check}' ethical gate."
            ),
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except ImportError as e:
        logger.error("Flagship fusion engine unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The flagship fusion engine is not available in this environment.",
        ) from e
    except Exception as e:
        logger.error("Flagship detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during flagship detection.",
        ) from e


class RootCauseRequest(BaseModel):
    """Request for multivariate root-cause localization."""

    observations: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Rows x nodes; the last row is the anomaly to localise.",
    )
    adjacency: list[list[float]] | None = Field(
        default=None,
        description="Optional (n_nodes x n_nodes) non-negative causal adjacency.",
    )
    train: list[list[float]] | None = Field(
        default=None,
        description="Optional normal-behaviour rows for the per-node baselines.",
    )
    top_k: int | None = Field(default=None, ge=1, description="Return only the top-K root causes.")
    node_names: list[str] | None = Field(
        default=None, description="Optional labels (one per node)."
    )


@router.post(
    "/rca",
    summary="Root-Cause Localization",
    description="""
Attribute a multivariate anomaly to its most likely root-cause nodes.

Runs the tier's graph-based root-cause analysis (a reverse personalised random
walk over a causal / service adjacency): given ``(n_rows x n_nodes)``
observations whose last row is anomalous, it ranks which node (sensor, service,
channel) most likely originated the fault. Torch-free. The same analysis behind
``mercury-agent rca`` and the ``mercury_localize_root_cause`` MCP tool.
""",
)
@records_detection("rca")
async def detect_rca(
    request: RootCauseRequest,
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Localise the root cause of a multivariate anomaly."""
    # Fail-closed harm-uplift gate, before any work: ``node_names`` carries
    # caller text.
    _enforce_route_gate(
        surface="api.detect_rca",
        operation="graph-based root-cause localization over multivariate observations",
        payload=request.model_dump(),
    )

    try:
        from omni_mercury_engine.detectors.detection_tier import localize_root_cause

        result = localize_root_cause(
            request.observations,
            adjacency=request.adjacency,
            train=request.train,
            top_k=request.top_k,
            node_names=request.node_names,
        )

        user_id = user.id if user else "anonymous"
        await record_detection(
            user_id=user_id,
            method="root_cause",
            data=request.observations,
            results={
                "n_nodes": result["n_nodes"],
                "top_root_cause": result["top_root_cause"],
            },
        )
        return result

    except ValueError as e:
        # Bad request: non-2-D observations, mismatched node_names length.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        logger.error("Root-cause localization failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during root-cause localization.",
        ) from e
