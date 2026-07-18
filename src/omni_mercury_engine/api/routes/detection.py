# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent - Detection API Routes.

Extracted detection endpoints for modular organization.
Provides univariate, multivariate, and advanced detection methods.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, User
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
    global _flagship_engine
    with _flagship_lock:
        if _flagship_engine is None:
            from omni_mercury_engine.engine import OmniMercuryEngine

            engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
            engine.load_default_fusion_checkpoint()
            # Deployment posture: the served flagship closes the loop --
            # every detection carries a ``decision`` record (grounded verdict
            # or explicit abstention plus a bounded, non-destructive response
            # plan). Additive only: this endpoint returns the raw result
            # dict, so existing consumers see one extra key. The core
            # engine's own default stays opt-in
            # (``OmniMercuryEngine.enable_decision_layer``).
            engine.enable_decision_layer()
            _flagship_engine = engine
        result: dict[str, Any] = _flagship_engine.detect_with_fusion(
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
        default=["statistical", "temporal", "dimensional"],
        description="Detectors to include in fusion",
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
        description="Ethical compliance threshold",
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
    is_stable: bool
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
- Ethical gating with benevolence threshold enforcement
- Full audit trail for explainability

## Fusion Strategies
- **weighted_average**: Fixed neural/symbolic weight combination
- **attention**: Learned attention-based weighting
- **gated**: Confidence-gated information flow
- **confidence_weighted**: Dynamic weighting based on component confidence
""",
)
async def detect_neurosymbolic(
    request: NeurosymbolicRequest,
    user: User | None = Depends(_get_optional_user),
) -> NeurosymbolicResponse:
    """Perform neuro-symbolic anomaly detection."""
    try:
        from omni_mercury_engine.cognitive.neurosymbolic_fusion import (
            FusionStrategy,
            NeurosymbolicFusionEngine,
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
async def detect_fusion(
    request: FusionRequest,
    user: User | None = Depends(_get_optional_user),
) -> FusionResponse:
    """Perform multi-detector fusion analysis."""
    try:
        data = np.array(request.data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        threshold = 2.0 + (1.0 - request.sensitivity) * 3.0
        detector_scores: dict[str, float] = {}
        detector_contributions: dict[str, float] = {}

        if "statistical" in request.detectors:
            mean = np.mean(data)
            std = np.std(data) + 1e-8
            z_scores = np.abs((data - mean) / std)
            detector_scores["statistical"] = float(np.max(z_scores))
            detector_contributions["statistical"] = 0.35

        if "temporal" in request.detectors:
            if data.size > 1:
                diff = np.diff(data.flatten())
                temporal_score = float(np.max(np.abs(diff)) / (np.std(data) + 1e-8))
            else:
                temporal_score = 0.0
            detector_scores["temporal"] = temporal_score
            detector_contributions["temporal"] = 0.25

        if "dimensional" in request.detectors:
            if data.ndim == 2 and data.shape[1] > 1:
                normalized = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                dimensional_score = float(np.max(np.linalg.norm(normalized, axis=1)))
            else:
                dimensional_score = detector_scores.get("statistical", 0.0)
            detector_scores["dimensional"] = dimensional_score
            detector_contributions["dimensional"] = 0.20

        if "spectral" in request.detectors:
            fft = np.fft.fft(data.flatten())
            magnitudes = np.abs(fft)
            spectral_score = float(np.max(magnitudes) / len(magnitudes))
            detector_scores["spectral"] = spectral_score
            detector_contributions["spectral"] = 0.20

        total_weight = sum(detector_contributions.values())
        detector_contributions = {k: v / total_weight for k, v in detector_contributions.items()}

        fused_score = sum(
            detector_scores.get(d, 0.0) * detector_contributions.get(d, 0.0)
            for d in request.detectors
        )

        is_anomaly = fused_score > threshold

        explanation_parts = [
            f"{d}: {detector_scores.get(d, 0.0):.3f} (weight: {detector_contributions.get(d, 0.0):.2f})"
            for d in request.detectors
        ]
        explanation = f"Fused score {fused_score:.3f} from: " + ", ".join(explanation_parts)

        user_id = user.id if user else "anonymous"
        await record_detection(
            user_id=user_id,
            method="fusion",
            data=request.data,
            results={
                "anomalies": [is_anomaly],
                "scores": [fused_score],
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
async def detect_three_r(
    request: ThreeRRequest,
    user: User | None = Depends(_get_optional_user),
) -> ThreeRResponse:
    """Perform 3R mechanism analysis."""
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

        recursion_score = compute_recursion_score(data, request.recursion_depth)
        recursion_score = float(np.clip(recursion_score, 0.0, 1.0))

        fft_result = np.fft.fft(data)
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

        phi = 1.618033988749895
        expected_ratios = [1.0 / (phi**i) for i in range(n_bands)]
        expected_sum = sum(expected_ratios)
        expected_ratios = [r / expected_sum for r in expected_ratios]

        harmonic_deviation = sum(abs(a - e) for a, e in zip(band_ratios, expected_ratios))
        resonance_score = float(np.clip(1.0 - harmonic_deviation, 0.0, 1.0))

        signal_variance = np.var(data)
        noise_estimate = np.var(np.diff(data)) / 2
        snr = signal_variance / (noise_estimate + 1e-8)
        optimization_score = float(np.clip(1.0 / (1.0 + np.exp(-np.log10(snr + 1))), 0.0, 1.0))

        aafe = OmniAvaEquation(
            ethical_compliance_threshold=request.ethical_threshold,
        )

        result = aafe.compute(
            recursion_score=recursion_score,
            resonance_score=resonance_score,
            optimization_score=optimization_score,
        )

        is_stable, _ = aafe.verify_lyapunov_stability()

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
        )
    except Exception as e:
        logger.error("3R detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during detection.",
        )


class TierDetectRequest(BaseModel):
    """Request for the streaming detector-tier ensemble."""

    data: list[float] = Field(
        ...,
        min_length=8,
        description="1-D anomaly series (the tier's native temporal contract)",
    )
    labels: list[int] | None = Field(
        default=None,
        description="Optional per-point 0/1 labels (enables supervised stacking/BMA)",
    )
    subset: list[str] | None = Field(
        default=None,
        description="Detector names to include (default: the full streaming tier)",
    )
    method: str | None = Field(
        default=None,
        description="Combiner: stacking | bma | average | consensus (default: auto)",
    )
    contamination: float = Field(default=0.05, ge=0.0, le=0.5, description="Anomaly fraction")
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
async def detect_tier(
    request: TierDetectRequest,
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Run the detector-tier ensemble on a 1-D series."""
    try:
        from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble

        series = np.asarray(request.data, dtype=float).ravel()
        labels = None if request.labels is None else np.asarray(request.labels, dtype=int)
        subset = tuple(request.subset) if request.subset else None

        result = run_tier_ensemble(
            series,
            labels=labels,
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
                "n_flagged": result["n_flagged"],
                "n_points": result["n_points"],
                "threshold": result["threshold"],
                "combiner": result["method"],
                "max_score": max(result["scores"]) if result["scores"] else 0.0,
            },
        )
        return result

    except ValueError as e:
        # Bad request: unknown detector name, stacking-without-labels, bad alpha.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Tier detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during tier detection.",
        )


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
        await record_detection(
            user_id=user_id,
            method="flagship_fusion",
            data=request.data,
            results={
                "anomalies": [bool(result.get("is_anomaly", False))],
                "scores": [float(result.get("anomaly_prob", 0.0))],
                "summary": {
                    "anomaly_prob": float(result.get("anomaly_prob", 0.0)),
                    "severity": float(result.get("severity", 0.0)),
                    "class_prediction": result.get("class_prediction"),
                },
            },
        )
        return result

    except EthicalConstraintViolationError as e:
        # Fail closed: the flagship refused this input at a hard ethical gate.
        logger.warning("Flagship detection blocked by ethical gate '%s'", e.check)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Detection blocked by the '{e.check}' ethical gate.",
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
async def detect_rca(
    request: RootCauseRequest,
    user: User | None = Depends(_get_optional_user),
) -> dict[str, Any]:
    """Localise the root cause of a multivariate anomaly."""
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
