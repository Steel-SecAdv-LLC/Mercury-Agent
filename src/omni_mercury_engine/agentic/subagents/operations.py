# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Coordinator operations: real ``omni_mercury_engine`` entrypoint adapters.

This module deepens each coordinator pantheon member from a *subsystem binding*
(import + introspect) into a genuine *subsystem operator*. It is a registry —
:data:`OPERATIONS`, a ``dict[pantheon_id -> adapter]`` — that the
:class:`~omni_mercury_engine.agentic.subagents.coordinator.CoordinatorSubAgent`
dispatches to: each adapter invokes the member's **real**
``omni_mercury_engine`` entrypoint with inputs derived from
``task.payload`` and returns the transparent result of that call.

Contract of an adapter ``(agent, task) -> (output, confidence, reasoning) | None``:

* It invokes the *real* entrypoint (never a stub, never fabricated signal) and
  returns a ``(output, confidence, reasoning)`` triple. ``output`` is a plain,
  JSON-friendly ``dict`` carrying the genuine result; the coordinator stamps it
  with ``mode="operation"`` and the member's identity/anchor.
* It returns ``None`` when it is *input-gated* and the required inputs are absent
  from the payload — the coordinator then falls back to the transparent
  import+introspect *binding report* (``mode="binding"``). Members whose real
  entrypoint takes no input (e.g. ``Artemis_VI`` registry fetch, ``Rhea_XXXIII``
  resilience stats) never return ``None``: their no-input call *is* the real op,
  and their binding report is reached only via the explicit ``mode="introspect"``
  readiness probe.
* It raises :class:`SubAgentExecutionError` (fail-closed) when inputs are present
  but malformed, or the entrypoint refuses — it never substitutes synthetic
  output for a real failure.
* It returns ``None`` when its subsystem is genuinely unimportable, so the
  fallback binding report can surface that unavailability transparently (and fail
  closed if *no* subsystem of the member binds at all).

Friction members are handled transparently, not papered over:

* ``Artemis_VI`` genuinely attempts a bounded network fetch over real data
  sources and reports true per-source reachability — green or red, never faked.
* ``Eos_XVIII`` / ``Hecate_XXVIII`` use the in-process surfaces (native JWT,
  in-proc request routing), not a live FastAPI server.
* ``Atlas_XXX`` drives the in-memory async cluster via ``asyncio.run`` under a
  short timeout.
* ``Nyx_XXIX`` security key generation and ``Prometheus_XXVII`` AutoML are the
  heavier paths; the always-safe primitives (crypto / lightweight scoring) are
  the primary path, the heavier path is explicitly opt-in via the payload.

The terminology here is **Omni-Codes only** — no other code system is named.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.agentic.subagents.base import SubAgentExecutionError

if TYPE_CHECKING:
    from omni_mercury_engine.agentic.subagents.base import SubAgentTask
    from omni_mercury_engine.agentic.subagents.coordinator import CoordinatorSubAgent

#: An adapter's return: ``(output, confidence, reasoning)`` or ``None`` (fallback).
OperationResult = tuple[dict[str, Any], float, str]
Adapter = Callable[["CoordinatorSubAgent", "SubAgentTask"], "OperationResult | None"]

# Bounded wall-clock for the genuine-but-external operations (network fetch,
# async cluster) so a coordinator op can never hang the fleet.
_NETWORK_TIMEOUT_S = 20.0
_CLUSTER_TIMEOUT_S = 15.0


# ---------------------------------------------------------------------------
# Small transparent helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    """Clamp a confidence into ``[0, 1]`` (no fabrication, just bounding)."""
    return float(min(1.0, max(0.0, x)))


def _as_2d_float(value: Any, *, what: str) -> np.ndarray:
    """Coerce a payload value to a non-empty 2-D float array or fail closed."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.size == 0:
        raise SubAgentExecutionError(f"{what} expects a non-empty 2-D array; got shape {arr.shape}")
    return arr


def _as_bytes(value: Any, *, what: str) -> bytes:
    """Coerce a payload value to ``bytes`` or fail closed."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tobytes()
    raise SubAgentExecutionError(f"{what} expects bytes/str/ndarray; got {type(value).__name__}")


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    """Return ``payload[key]`` as a dict when present and mapping-typed, else None."""
    value = payload.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return None


# ---------------------------------------------------------------------------
# Hestia_II — core foundation / stability (utils)
# ---------------------------------------------------------------------------


def _op_hestia(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``utils.normalize_data`` (with data) or ``OmniCodes`` stability (no input)."""
    try:
        from omni_mercury_engine import utils
        from omni_mercury_engine.utils.constants import OmniCodes
    except ImportError:
        return None

    data = task.payload.get("data")
    if data is not None:
        method = str(task.payload.get("method", "standard"))
        arr = _as_2d_float(data, what="Hestia_II.normalize")
        normalized = np.asarray(utils.normalize_data(arr, method=method), dtype=np.float64)
        output = {
            "operation": "utils.normalize_data",
            "method": method,
            "shape": list(normalized.shape),
            "mean": float(np.mean(normalized)),
            "std": float(np.std(normalized)),
        }
        return output, 1.0, f"Hestia_II normalized {arr.shape} via '{method}'"

    total = float(OmniCodes.get_total_stability())
    valid = bool(OmniCodes.validate_stability())
    output = {
        "operation": "utils.constants.OmniCodes.stability",
        "total_stability": total,
        "stable": valid,
        "n_codes": len(OmniCodes.get_all()),
    }
    return (
        output,
        1.0 if valid else 0.0,
        f"Hestia_II foundation stability {total:.2f} (stable={valid})",
    )


# ---------------------------------------------------------------------------
# Hermes_III — communication / synchronization (narrative)
# ---------------------------------------------------------------------------


def _op_hermes(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``narrative.create_mercury_interface().process_detection``."""
    detection = _mapping(task.payload, "detection_result")
    if detection is None:
        return None
    try:
        from omni_mercury_engine.narrative import create_mercury_interface
    except ImportError:
        return None

    interface = create_mercury_interface(enable_proactive=False, enable_memory=True)
    response = interface.process_detection(detection)
    summary = str(getattr(response, "summary", ""))
    output = {
        "operation": "narrative.process_detection",
        "summary": summary,
        "domain": getattr(response, "domain", None),
        "confidence_statement": str(getattr(response, "confidence_statement", "")),
        "response_time_ms": float(getattr(response, "response_time_ms", 0.0)),
        "n_follow_ups": len(getattr(response, "follow_up_suggestions", []) or []),
    }
    confidence = _clamp01(float(detection.get("confidence", 0.9)))
    return output, confidence, f"Hermes_III synchronized a detection narrative ({summary[:60]})"


# ---------------------------------------------------------------------------
# Athena_IV — upload validation / intake (validation)
# ---------------------------------------------------------------------------


def _op_athena(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``validation.ValidationPipeline.validate(model, X, y)``."""
    payload = task.payload
    model = payload.get("model")
    if model is None or "X" not in payload or "y" not in payload:
        return None
    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        raise SubAgentExecutionError("Athena_IV.validate requires a model exposing fit()/predict()")
    try:
        from omni_mercury_engine.validation import ValidationPipeline
    except ImportError:
        return None

    X = _as_2d_float(payload["X"], what="Athena_IV.X")
    y = np.asarray(payload["y"]).ravel()
    if y.shape[0] != X.shape[0]:
        raise SubAgentExecutionError(
            f"Athena_IV.validate: X/y length mismatch ({X.shape[0]} vs {y.shape[0]})"
        )
    n_folds = int(payload.get("n_folds", min(5, max(2, X.shape[0] // 5))))
    result = ValidationPipeline(n_folds=n_folds).validate(
        model, X, y, dataset_name=str(task.task_id), model_name=type(model).__name__
    )
    f1 = float(getattr(result, "f1_score", 0.0) or 0.0)
    n_samp = getattr(result, "num_samples", None)
    n_feat = getattr(result, "num_features", None)
    output = {
        "operation": "validation.ValidationPipeline.validate",
        "f1_score": f1,
        "precision": float(getattr(result, "precision", 0.0) or 0.0),
        "recall": float(getattr(result, "recall", 0.0) or 0.0),
        "auc_roc": float(getattr(result, "auc_roc", 0.0) or 0.0),
        "num_samples": int(n_samp) if n_samp is not None else int(X.shape[0]),
        "num_features": int(n_feat) if n_feat is not None else int(X.shape[1]),
    }
    return output, _clamp01(f1), f"Athena_IV validated {type(model).__name__}: F1={f1:.3f}"


# ---------------------------------------------------------------------------
# Apollo_V — biometric matching (biometric)
# ---------------------------------------------------------------------------


def _op_apollo(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``biometric.BiometricAnomalyDetector.detect_anomaly``."""
    payload = task.payload
    modal_keys = {
        "iris": "iris_image",
        "fingerprint": "fingerprint_image",
        "voice": "voice_sample",
    }
    provided = {m: payload[k] for m, k in modal_keys.items() if payload.get(k) is not None}
    if not provided:
        return None
    try:
        from omni_mercury_engine.biometric import BiometricAnomalyDetector
    except ImportError:
        return None

    kwargs: dict[str, np.ndarray] = {}
    for modality, key in modal_keys.items():
        if modality in provided:
            dtype = np.uint8 if modality != "voice" else np.float32
            kwargs[key] = np.asarray(provided[modality], dtype=dtype)
    detector = BiometricAnomalyDetector(modalities=sorted(provided), liveness_required=False)
    result = detector.detect_anomaly(**kwargs)
    score = float(getattr(result, "anomaly_score", 0.0))
    output = {
        "operation": "biometric.detect_anomaly",
        "modalities": sorted(provided),
        "is_anomaly": bool(getattr(result, "is_anomaly", False)),
        "anomaly_score": score,
        "anomaly_type": getattr(result, "anomaly_type", None),
        "modality_scores": {
            k: float(v) for k, v in (getattr(result, "modality_scores", {}) or {}).items()
        },
    }
    return (
        output,
        _clamp01(score),
        (f"Apollo_V biometric scan over {sorted(provided)}: score={score:.3f}"),
    )


# ---------------------------------------------------------------------------
# Artemis_VI — OSINT / registry cross-check (data_sources) — genuine network
# ---------------------------------------------------------------------------

#: Curated, key-free public sources Artemis genuinely probes for reachability.
_ARTEMIS_DEFAULT_SOURCES: tuple[str, ...] = ("USGSEarthquakeSource", "NASADONKISource")


def _op_artemis(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: genuinely attempt ``data_sources`` fetch; report transparent reachability."""
    try:
        from omni_mercury_engine import data_sources
        from omni_mercury_engine.data_sources import DataSourceManager
    except ImportError:
        return None

    requested = task.payload.get("sources")
    names: Sequence[str]
    if isinstance(requested, Sequence) and not isinstance(requested, (str, bytes)):
        names = [str(n) for n in requested]
    else:
        names = _ARTEMIS_DEFAULT_SOURCES

    manager = DataSourceManager()
    registered: list[str] = []
    skipped: dict[str, str] = {}
    for name in names:
        cls = getattr(data_sources, name, None)
        if cls is None:
            skipped[name] = "unknown source class"
            continue
        try:
            manager.register_source(cls())
            registered.append(name)
        except Exception as exc:  # construction needs config we do not have — transparent skip
            skipped[name] = f"{type(exc).__name__}: {exc}"

    if not registered:
        raise SubAgentExecutionError(
            f"Artemis_VI: no data source could be constructed (skipped={skipped})"
        )

    timeout = float(task.payload.get("timeout", _NETWORK_TIMEOUT_S))

    async def _fetch() -> dict[str, Any]:
        return await asyncio.wait_for(manager.fetch_all(), timeout=timeout)

    reachability: dict[str, dict[str, Any]] = {}
    try:
        results = asyncio.run(_fetch())
    except TimeoutError:
        reachability = {n: {"reachable": False, "error": "timeout"} for n in registered}
    else:
        for key, fetch_result in results.items():
            reachability[key] = {
                "reachable": bool(getattr(fetch_result, "success", False)),
                "n_points": len(getattr(fetch_result, "data_points", []) or []),
                "error": getattr(fetch_result, "error", None),
            }

    n_reachable = sum(1 for r in reachability.values() if r["reachable"])
    output = {
        "operation": "data_sources.fetch_all",
        "registered": registered,
        "skipped": skipped,
        "reachability": reachability,
        "n_reachable": n_reachable,
        "n_attempted": len(reachability),
    }
    confidence = _clamp01(n_reachable / len(reachability)) if reachability else 0.0
    return (
        output,
        confidence,
        (f"Artemis_VI probed {len(reachability)} source(s): {n_reachable} reachable"),
    )


# ---------------------------------------------------------------------------
# Poseidon_IX — data flow / secure vault (crypto)
# ---------------------------------------------------------------------------


def _op_poseidon(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``crypto.hash_data`` + ``crypto.encrypt`` round-trip over payload data."""
    if "data" not in task.payload:
        return None
    try:
        from omni_mercury_engine import crypto
    except ImportError:
        return None

    plaintext = _as_bytes(task.payload["data"], what="Poseidon_IX.data")
    key = task.payload.get("key")
    key_bytes = _as_bytes(key, what="Poseidon_IX.key") if key is not None else crypto.generate_key()
    digest = crypto.hash_data(plaintext)
    ciphertext, nonce = crypto.encrypt(plaintext, key_bytes)
    recovered = crypto.decrypt(ciphertext, key_bytes, nonce)
    roundtrip_ok = recovered == plaintext
    if not roundtrip_ok:
        raise SubAgentExecutionError("Poseidon_IX: encrypt/decrypt round-trip failed")
    output = {
        "operation": "crypto.encrypt+hash_data",
        "digest_hex": digest.hex(),
        "digest_len": len(digest),
        "ciphertext_len": len(ciphertext),
        "nonce_len": len(nonce),
        "roundtrip_ok": roundtrip_ok,
    }
    return output, 1.0, f"Poseidon_IX sealed {len(plaintext)}B (hash+AEAD, round-trip verified)"


# ---------------------------------------------------------------------------
# Demeter_X — cognitive evolution (cognitive)
# ---------------------------------------------------------------------------


def _cognitive_analyze(
    detection: dict[str, Any], raw: Any, context: dict[str, Any] | None
) -> tuple[dict[str, Any], float]:
    """Shared real call into ``cognitive.CognitiveOrchestrator.analyze`` (fail-closed)."""
    from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
    from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

    raw_arr = None if raw is None else np.asarray(raw, dtype=np.float64)
    try:
        result = CognitiveOrchestrator().analyze(detection, raw_arr, context)
    except EthicalConstraintViolationError as exc:
        raise SubAgentExecutionError(f"cognitive analysis refused by ethical gate: {exc}") from exc
    confidence = _clamp01(float(getattr(result, "confidence", 0.0)))
    summary = {
        "anomaly_detected": bool(getattr(result, "anomaly_detected", False)),
        "anomaly_score": float(getattr(result, "anomaly_score", 0.0)),
        "benevolence_score": float(getattr(result, "benevolence_score", 0.0)),
        "ethical_permissible": bool(getattr(result, "ethical_permissible", False)),
        "confidence": confidence,
        "n_reasoning_steps": len(getattr(result, "reasoning_chain", []) or []),
        "recommended_actions": list(getattr(result, "recommended_actions", []) or [])[:5],
    }
    return summary, confidence


def _op_demeter(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``cognitive.CognitiveOrchestrator.analyze`` (cognitive evolution)."""
    detection = _mapping(task.payload, "detection_result")
    if detection is None:
        return None
    try:
        summary, confidence = _cognitive_analyze(detection, task.payload.get("raw_data"), None)
    except ImportError:
        return None
    summary["operation"] = "cognitive.CognitiveOrchestrator.analyze"
    return (
        summary,
        confidence,
        (
            f"Demeter_X cognitive analysis: score={summary['anomaly_score']:.3f}, "
            f"{summary['n_reasoning_steps']} reasoning step(s)"
        ),
    )


# ---------------------------------------------------------------------------
# Hephaestus_XI — infrastructure / auto-scaling (scaling, infrastructure)
# ---------------------------------------------------------------------------


def _op_hephaestus(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``scaling`` compute allocation (with workloads) or module provisioning."""
    payload = task.payload
    workloads = payload.get("workloads")
    if isinstance(workloads, Sequence) and not isinstance(workloads, (str, bytes)) and workloads:
        try:
            from omni_mercury_engine.scaling.bain_ai_scaling import (
                BainAIScaling,
                ComputeResource,
            )
        except ImportError:
            return None
        res = _mapping(payload, "resources") or {}
        resource = ComputeResource(
            cpu_cores=int(res.get("cpu_cores", 8)),
            gpu_count=int(res.get("gpu_count", 1)),
            memory_gb=float(res.get("memory_gb", 32.0)),
            power_watts=float(res.get("power_watts", 400.0)),
            cost_per_hour=float(res.get("cost_per_hour", 5.0)),
        )
        allocation = BainAIScaling().optimize_compute_allocation(
            [dict(w) for w in workloads], resource
        )
        output = {
            "operation": "scaling.BainAIScaling.optimize_compute_allocation",
            "n_workloads": len(allocation),
            "allocated_cpu": {k: int(v.cpu_cores) for k, v in allocation.items()},
            "allocated_gpu": {k: int(v.gpu_count) for k, v in allocation.items()},
        }
        return output, 1.0, f"Hephaestus_XI allocated compute across {len(allocation)} workload(s)"

    try:
        from omni_mercury_engine.infrastructure import InfrastructureCoordinator
    except ImportError:
        return None
    priorities = payload.get("priorities")
    prio = (
        [str(p) for p in priorities]
        if isinstance(priorities, Sequence) and not isinstance(priorities, (str, bytes))
        else None
    )
    modules = InfrastructureCoordinator().instantiate_filtered_modules(priorities=prio)
    output = {
        "operation": "infrastructure.instantiate_filtered_modules",
        "n_modules": len(modules),
        "modules": sorted(modules),
        "priorities": prio,
    }
    return output, 1.0, f"Hephaestus_XI provisioned {len(modules)} infrastructure module(s)"


# ---------------------------------------------------------------------------
# Eleos_XII — empathy / survivor support (narrative)
# ---------------------------------------------------------------------------


def _op_eleos(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``narrative.MercuryConversationInterface.process_detection`` (empathetic)."""
    detection = _mapping(task.payload, "detection_result")
    if detection is None:
        return None
    try:
        from omni_mercury_engine.narrative.interface import MercuryConversationInterface
    except ImportError:
        return None

    interface = MercuryConversationInterface(
        enable_proactive=False, enable_memory=True, default_domain="humanitarian"
    )
    response = interface.process_detection(detection)
    output = {
        "operation": "narrative.MercuryConversationInterface.process_detection",
        "summary": str(getattr(response, "summary", "")),
        "message": str(getattr(response, "message", ""))[:400],
        "style": str(getattr(response, "style", "")),
        "follow_up_suggestions": list(getattr(response, "follow_up_suggestions", []) or [])[:5],
    }
    confidence = _clamp01(float(detection.get("confidence", 0.9)))
    return output, confidence, "Eleos_XII produced a transparent, supportive response"


# ---------------------------------------------------------------------------
# Hades_XV — compression / cold storage (utils, crypto)
# ---------------------------------------------------------------------------


def _op_hades(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``utils.compress_information`` + ``crypto.hash_data`` for sealed cold storage."""
    if "data" not in task.payload:
        return None
    try:
        from omni_mercury_engine import crypto, utils
    except ImportError:
        return None

    arr = _as_2d_float(task.payload["data"], what="Hades_XV.data")
    level = int(task.payload.get("compression_level", 9))
    blob, meta = utils.compress_information(arr, compression_level=level)
    digest = crypto.hash_data(bytes(blob))
    ratio = float(meta.get("compression_ratio", 0.0))
    output = {
        "operation": "utils.compress_information+crypto.hash_data",
        "compression_ratio": ratio,
        "original_size": int(meta.get("original_size", arr.nbytes)),
        "compressed_size": int(meta.get("compressed_size", len(blob))),
        "integrity_digest_hex": digest.hex(),
    }
    return output, 1.0, f"Hades_XV archived {arr.shape} at {ratio:.2f}x (integrity-sealed)"


# ---------------------------------------------------------------------------
# Selene_XVI — cron / temporal coordination (streaming) — stateful
# ---------------------------------------------------------------------------


def _op_selene(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``streaming.StreamingDetector.ingest`` over a warmup sequence."""
    points = task.payload.get("points")
    if points is None:
        points = task.payload.get("data")
    if points is None:
        return None
    try:
        from omni_mercury_engine.streaming.streaming_detector import StreamingDetector
    except ImportError:
        return None

    seq = _as_2d_float(points, what="Selene_XVI.points")
    n = seq.shape[0]
    min_samples = int(task.payload.get("min_samples", max(2, min(30, n // 2))))
    detector = StreamingDetector(
        window_size=max(n, 100), min_samples=min_samples, refit_interval=max(1, min_samples)
    )
    last: dict[str, Any] | None = None
    n_anomalies = 0
    for row in seq:
        result = detector.ingest(row)
        if isinstance(result, dict):
            last = result
            flags = np.asarray(result.get("is_anomaly", []), dtype=bool).ravel()
            n_anomalies += int(flags.sum())
    output = {
        "operation": "streaming.StreamingDetector.ingest",
        "n_ingested": n,
        "is_ready": bool(detector.is_ready),
        "produced_detection": last is not None,
        "n_anomalies": n_anomalies,
        "detector_type": (last or {}).get("detector_type"),
    }
    if last is None and not detector.is_ready:
        raise SubAgentExecutionError(
            f"Selene_XVI: stream of {n} point(s) below warmup ({min_samples}); no detection produced"
        )
    return (
        output,
        1.0 if detector.is_ready else 0.5,
        (f"Selene_XVI streamed {n} point(s); ready={detector.is_ready}, anomalies={n_anomalies}"),
    )


# ---------------------------------------------------------------------------
# Helios_XVII — telemetry / monitoring (metrics)
# ---------------------------------------------------------------------------


def _op_helios(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``metrics.AnomalyMetrics.compute_all(labels, scores)``."""
    payload = task.payload
    labels = payload.get("labels")
    scores = payload.get("scores")
    if labels is None or scores is None:
        return None
    try:
        from omni_mercury_engine.metrics import AnomalyMetrics
    except ImportError:
        return None

    y_true = np.asarray(labels).ravel()
    y_score = np.asarray(scores, dtype=np.float64).ravel()
    if y_true.shape[0] != y_score.shape[0] or y_true.size == 0:
        raise SubAgentExecutionError(
            f"Helios_XVII: labels/scores must be equal non-empty length "
            f"({y_true.shape[0]} vs {y_score.shape[0]})"
        )
    metrics = AnomalyMetrics.compute_all(y_true, y_score)
    output = {
        "operation": "metrics.AnomalyMetrics.compute_all",
        **{k: float(v) for k, v in metrics.items()},
    }
    auroc = float(metrics.get("auroc", 0.0))
    return (
        output,
        _clamp01(auroc),
        f"Helios_XVII telemetry: AUROC={auroc:.3f} over {y_true.size} samples",
    )


# ---------------------------------------------------------------------------
# Eos_XVIII — onboarding / session (api.auth native JWT)
# ---------------------------------------------------------------------------


def _op_eos(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: native JWT create + in-process validate (no FastAPI server)."""
    payload = task.payload
    user_id = payload.get("user_id")
    username = payload.get("username")
    if user_id is None or username is None:
        return None
    try:
        from omni_mercury_engine.api.auth import JWTAuth
        from omni_mercury_engine.security import native_jwt
    except ImportError:
        return None

    secret = str(payload.get("secret_key", "")) or "mercury-onboarding-dev-secret-key-0123456789"
    roles = [str(r) for r in payload.get("roles", ["user"])]
    token = JWTAuth.create_token(
        user_id=str(user_id),
        username=str(username),
        secret_key=secret,
        roles=roles,
        expires_in_hours=int(payload.get("expires_in_hours", 1)),
    )
    try:
        claims = native_jwt.decode(token, secret, algorithms=["HS256"])
        validated = str(claims.get("sub")) == str(user_id)
    except Exception as exc:
        raise SubAgentExecutionError(
            f"Eos_XVIII: issued JWT failed its own validation: {exc}"
        ) from exc
    output = {
        "operation": "api.auth.JWTAuth.create_token+validate",
        "token_len": len(token),
        "validated": validated,
        "subject": str(claims.get("sub")),
        "roles": list(claims.get("roles", [])),
    }
    return (
        output,
        1.0 if validated else 0.0,
        f"Eos_XVIII issued+validated a session token for {username}",
    )


# ---------------------------------------------------------------------------
# Nemesis_XIX — fairness / bias audit (evaluation, ethical)
# ---------------------------------------------------------------------------


def _op_nemesis(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``evaluation.evaluate_anomaly_detection`` or ``ethical.TwelveFold`` verify."""
    payload = task.payload
    if payload.get("y_true") is not None and payload.get("y_score") is not None:
        try:
            from omni_mercury_engine.evaluation import evaluate_anomaly_detection
        except ImportError:
            return None
        y_true = np.asarray(payload["y_true"]).ravel()
        y_score = np.asarray(payload["y_score"], dtype=np.float64).ravel()
        if y_true.shape[0] != y_score.shape[0] or y_true.size == 0:
            raise SubAgentExecutionError(
                "Nemesis_XIX: y_true/y_score must be equal non-empty length"
            )
        metrics = evaluate_anomaly_detection(y_true, y_score)
        auc = float(getattr(metrics, "auc_roc", 0.0))
        output = {
            "operation": "evaluation.evaluate_anomaly_detection",
            "auc_roc": auc,
            "auc_pr": float(getattr(metrics, "auc_pr", 0.0)),
            "best_f1": float(getattr(metrics, "best_f1", getattr(metrics, "f1", 0.0))),
            "precision": float(getattr(metrics, "precision", 0.0)),
            "recall": float(getattr(metrics, "recall", 0.0)),
        }
        return output, _clamp01(auc), f"Nemesis_XIX fairness audit: AUROC={auc:.3f}"

    dimension_scores = _mapping(payload, "dimension_scores")
    if dimension_scores is None:
        return None
    try:
        from omni_mercury_engine.ethical import TwelveFoldVerificationSystem
    except ImportError:
        return None
    verifier = TwelveFoldVerificationSystem()
    result = verifier.verify({k: float(v) for k, v in dimension_scores.items()})
    overall = float(getattr(result, "overall_score", 0.0))
    output = {
        "operation": "ethical.TwelveFoldVerificationSystem.verify",
        "overall_score": overall,
        "verification_status": str(getattr(result, "verification_status", "")),
        "passed_dimensions": list(getattr(result, "passed_dimensions", []) or []),
        "failed_dimensions": list(getattr(result, "failed_dimensions", []) or []),
    }
    return (
        output,
        _clamp01(overall),
        f"Nemesis_XIX twelve-fold verify: {output['verification_status']}",
    )


# ---------------------------------------------------------------------------
# Tyche_XX — risk / probabilistic decisioning (decision)
# ---------------------------------------------------------------------------


def _op_tyche(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``decision.DecisionAbstentionResponder.decide``."""
    detection = _mapping(task.payload, "detection_result")
    if detection is None:
        return None
    try:
        from omni_mercury_engine.decision import DecisionAbstentionResponder
    except ImportError:
        return None

    domain = task.payload.get("domain") or getattr(task.domain, "value", str(task.domain))
    record = DecisionAbstentionResponder().decide(detection, domain=str(domain))
    confidence = _clamp01(float(getattr(record, "decision_confidence", 0.0)))
    output = {
        "operation": "decision.DecisionAbstentionResponder.decide",
        "state": str(getattr(record, "state", "")),
        "disposition": str(getattr(record, "disposition", "")),
        "decision_label": getattr(record, "decision_label", None),
        "abstained": bool(getattr(record, "abstained", False)),
        "decision_confidence": confidence,
    }
    return (
        output,
        confidence,
        (
            f"Tyche_XX decided {output['disposition']} (label={output['decision_label']}, "
            f"abstained={output['abstained']})"
        ),
    )


# ---------------------------------------------------------------------------
# Zelos_XXI — performance / throughput (ml, scaling)
# ---------------------------------------------------------------------------


def _op_zelos(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``ml.quick_anomaly_score`` (throughput scoring) over a batch."""
    data = task.payload.get("data")
    if data is None:
        data = task.payload.get("X")
    if data is None:
        return None
    try:
        from omni_mercury_engine.ml import quick_anomaly_score
    except ImportError:
        return None

    X = _as_2d_float(data, what="Zelos_XXI.data")
    method = str(task.payload.get("method", "isolation"))
    scores = np.asarray(quick_anomaly_score(X.astype(np.float32), method=method), dtype=np.float64)
    mean_score = float(np.mean(scores))
    output = {
        "operation": "ml.quick_anomaly_score",
        "method": method,
        "n_samples": int(X.shape[0]),
        "mean_score": mean_score,
        "max_score": float(np.max(scores)),
    }
    return (
        output,
        _clamp01(mean_score),
        (f"Zelos_XXI scored {X.shape[0]} sample(s) via '{method}' (mean={mean_score:.3f})"),
    )


# ---------------------------------------------------------------------------
# Kronos_XXII — time-series / indexing (detectors)
# ---------------------------------------------------------------------------


def _op_kronos(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``detectors.MercuryAnomalyDetector.fit(train).detect(X)``."""
    data = task.payload.get("data")
    if data is None:
        return None
    try:
        from omni_mercury_engine.detectors import MercuryAnomalyDetector
    except ImportError:
        return None

    X = _as_2d_float(data, what="Kronos_XXII.data")
    train = _as_2d_float(task.payload.get("train", X), what="Kronos_XXII.train")
    detector = MercuryAnomalyDetector()
    detector.fit(train)
    result = detector.detect(X)
    flags = np.asarray(result.get("is_anomaly", []), dtype=bool).ravel()
    scores = np.asarray(result.get("scores", []), dtype=np.float64).ravel()
    mean_score = float(np.mean(scores)) if scores.size else 0.0
    output = {
        "operation": "detectors.MercuryAnomalyDetector.detect",
        "n_samples": int(X.shape[0]),
        "n_anomalies": int(flags.sum()),
        "mean_score": mean_score,
        "detector_type": result.get("detector_type"),
        "threshold": float(result.get("threshold", 0.0)),
    }
    return (
        output,
        _clamp01(mean_score),
        (f"Kronos_XXII indexed {X.shape[0]} point(s): {int(flags.sum())} anomalies"),
    )


# ---------------------------------------------------------------------------
# Morpheus_XXIII — simulation / scenario (cognitive)
# ---------------------------------------------------------------------------


def _op_morpheus(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``cognitive.CognitiveOrchestrator.analyze`` under a scenario context."""
    detection = _mapping(task.payload, "detection_result")
    if detection is None:
        return None
    scenario = task.payload.get("scenario", "what-if")
    domain = task.payload.get("domain") or getattr(task.domain, "value", str(task.domain))
    context = {"scenario": str(scenario), "domain": str(domain)}
    try:
        summary, confidence = _cognitive_analyze(detection, task.payload.get("raw_data"), context)
    except ImportError:
        return None
    summary["operation"] = "cognitive.CognitiveOrchestrator.analyze"
    summary["scenario"] = str(scenario)
    return (
        summary,
        confidence,
        (f"Morpheus_XXIII simulated scenario '{scenario}': score={summary['anomaly_score']:.3f}"),
    )


# ---------------------------------------------------------------------------
# Iris_XXIV — notification routing (alerting)
# ---------------------------------------------------------------------------


def _op_iris(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``alerting.CAPAlertGenerator.generate_alert`` (CAP 1.2 XML)."""
    payload = task.payload
    headline = payload.get("headline")
    description = payload.get("description")
    if headline is None or description is None:
        return None
    try:
        from omni_mercury_engine.alerting import CAPAlertGenerator
    except ImportError:
        return None

    domain = str(payload.get("domain") or getattr(task.domain, "value", "security"))
    score = _clamp01(float(payload.get("score", payload.get("anomaly_score", 0.5))))
    area = str(payload.get("area", payload.get("area_description", "unspecified area")))
    xml = CAPAlertGenerator().generate_alert(
        domain=domain,
        headline=str(headline),
        description=str(description),
        anomaly_score=score,
        area_description=area,
    )
    output = {
        "operation": "alerting.CAPAlertGenerator.generate_alert",
        "domain": domain,
        "cap_xml_len": len(xml),
        "is_cap_xml": xml.lstrip().startswith("<") and "alert" in xml.lower(),
        "area": area,
    }
    if not output["is_cap_xml"]:
        raise SubAgentExecutionError("Iris_XXIV: generated alert is not well-formed CAP XML")
    return output, score, f"Iris_XXIV generated a CAP alert for '{domain}' ({len(xml)}B)"


# ---------------------------------------------------------------------------
# Pan_XXV — sensor fusion (core) — in-memory
# ---------------------------------------------------------------------------


def _op_pan(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``core.GlobalOmniScalarNetwork`` register + compute global score."""
    scalars = _mapping(task.payload, "scalars")
    if not scalars:
        return None
    try:
        from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork
    except ImportError:
        return None

    network = GlobalOmniScalarNetwork()
    component = str(task.payload.get("component", f"sensor::{task.task_id}"))
    network.register_scalars(
        component_name=component, scalars={k: float(v) for k, v in scalars.items()}
    )
    global_score = float(network.compute_global_intelligence_score())
    harmony = float(network.compute_triadic_harmony())
    output = {
        "operation": "core.GlobalOmniScalarNetwork.compute_global_intelligence_score",
        "n_registered": len(scalars),
        "global_intelligence_score": global_score,
        "triadic_harmony": harmony,
    }
    return (
        output,
        _clamp01(global_score),
        (
            f"Pan_XXV fused {len(scalars)} scalar(s): global={global_score:.3f}, harmony={harmony:.3f}"
        ),
    )


# ---------------------------------------------------------------------------
# Persephone_XXVI — lifecycle / archival (resilience, federation) — no input
# ---------------------------------------------------------------------------


def _op_persephone(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``resilience.get_all_breaker_stats`` — live lifecycle/retention posture."""
    try:
        from omni_mercury_engine.resilience import get_all_breaker_stats
    except ImportError:
        return None

    stats = get_all_breaker_stats()
    open_breakers = [name for name, s in stats.items() if str(s.get("state", "")).upper() == "OPEN"]
    output = {
        "operation": "resilience.get_all_breaker_stats",
        "n_breakers": len(stats),
        "n_open": len(open_breakers),
        "open_breakers": open_breakers[:20],
    }
    confidence = (
        1.0 if not open_breakers else _clamp01(1.0 - len(open_breakers) / max(1, len(stats)))
    )
    return (
        output,
        confidence,
        (f"Persephone_XXVI lifecycle posture: {len(stats)} breaker(s), {len(open_breakers)} open"),
    )


# ---------------------------------------------------------------------------
# Prometheus_XXVII — model training / provisioning (ml, automl)
# ---------------------------------------------------------------------------


def _op_prometheus(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``ml.quick_anomaly_score`` (light) or budget-gated ``automl.fit`` (heavy)."""
    payload = task.payload
    heavy = bool(payload.get("automl")) and payload.get("X_train") is not None
    if heavy:
        try:
            from omni_mercury_engine.automl import MercuryAutoML
        except ImportError:
            return None
        X_train = _as_2d_float(payload["X_train"], what="Prometheus_XXVII.X_train")
        y_train = payload.get("y_train")
        y = None if y_train is None else np.asarray(y_train).ravel()
        n_trials = int(payload.get("n_trials", 3))
        result = MercuryAutoML(
            n_trials=n_trials, time_budget=int(payload.get("time_budget", 30))
        ).fit(X_train, y)
        best = float(getattr(result, "best_metric", 0.0))
        output = {
            "operation": "automl.MercuryAutoML.fit",
            "best_metric": best,
            "n_trials": n_trials,
            "best_config_keys": sorted((getattr(result, "best_config", {}) or {}).keys())[:10],
        }
        return (
            output,
            _clamp01(best),
            f"Prometheus_XXVII AutoML provisioned a model (best={best:.3f})",
        )

    data = payload.get("X")
    if data is None:
        data = payload.get("data")
    if data is None:
        return None
    try:
        from omni_mercury_engine.ml import quick_anomaly_score
    except ImportError:
        return None
    X = _as_2d_float(data, what="Prometheus_XXVII.X")
    scores = np.asarray(quick_anomaly_score(X.astype(np.float32)), dtype=np.float64)
    mean_score = float(np.mean(scores))
    output = {
        "operation": "ml.quick_anomaly_score",
        "n_samples": int(X.shape[0]),
        "mean_score": mean_score,
    }
    return (
        output,
        _clamp01(mean_score),
        (f"Prometheus_XXVII provisioned a lightweight scorer over {X.shape[0]} sample(s)"),
    )


# ---------------------------------------------------------------------------
# Hecate_XXVIII — gateway / protocol (integrations.routing) — in-process
# ---------------------------------------------------------------------------


def _op_hecate(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``integrations.routing.RequestRouter`` register + ``match`` a request."""
    request = _mapping(task.payload, "request")
    if request is None or "path" not in request:
        return None
    try:
        from omni_mercury_engine.integrations.routing import RequestRouter
    except ImportError:
        return None

    async def _handler(req: Any, **params: Any) -> dict[str, Any]:
        return {"routed": True, "params": params}

    router = RequestRouter()
    routes = task.payload.get("routes")
    declared: list[dict[str, Any]] = []
    if isinstance(routes, Sequence) and not isinstance(routes, (str, bytes)):
        for r in routes:
            if isinstance(r, Mapping) and "pattern" in r:
                declared.append(dict(r))
    if not declared:
        declared = [
            {"pattern": str(request["path"]), "methods": [str(request.get("method", "GET"))]}
        ]
    for r in declared:
        router.add_route(
            str(r["pattern"]),
            _handler,
            methods=[str(m) for m in r.get("methods", ["GET"])],
        )
    match = router.match(str(request["path"]), method=str(request.get("method", "GET")))
    matched = match is not None and getattr(match, "handler", None) is not None
    output = {
        "operation": "integrations.routing.RequestRouter.match",
        "matched": bool(matched),
        "params": dict(getattr(match, "params", {}) or {}) if matched else {},
        "n_routes": len(declared),
    }
    return (
        output,
        1.0 if matched else 0.0,
        (f"Hecate_XXVIII routed {request['path']} -> matched={matched}"),
    )


# ---------------------------------------------------------------------------
# Nyx_XXIX — secrets / enclave (crypto primary, security keygen secondary)
# ---------------------------------------------------------------------------


def _op_nyx(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``crypto`` seal (primary) or ``security.MercuryCrypto`` keygen (secondary)."""
    payload = task.payload
    if payload.get("keygen"):
        try:
            from omni_mercury_engine.security import MercuryCrypto
        except ImportError:
            return None
        keypair = MercuryCrypto().generate_signing_keypair()
        pk = bytes(getattr(keypair, "public_key", b""))
        sk = bytes(getattr(keypair, "secret_key", b""))
        if not pk or not sk:
            raise SubAgentExecutionError("Nyx_XXIX: keygen produced an empty keypair")
        output = {
            "operation": "security.MercuryCrypto.generate_signing_keypair",
            "algorithm": str(getattr(keypair, "algorithm", "")),
            "public_key_len": len(pk),
            "secret_key_len": len(sk),
        }
        return output, 1.0, f"Nyx_XXIX generated a {output['algorithm']} signing keypair"

    if "data" not in payload:
        return None
    try:
        from omni_mercury_engine import crypto
    except ImportError:
        return None
    plaintext = _as_bytes(payload["data"], what="Nyx_XXIX.data")
    key = crypto.generate_key()
    ciphertext, nonce = crypto.encrypt(plaintext, key)
    digest = crypto.hash_data(plaintext)
    if crypto.decrypt(ciphertext, key, nonce) != plaintext:
        raise SubAgentExecutionError("Nyx_XXIX: enclave seal round-trip failed")
    output = {
        "operation": "crypto.encrypt+hash_data",
        "ciphertext_len": len(ciphertext),
        "nonce_len": len(nonce),
        "digest_hex": digest.hex(),
    }
    return output, 1.0, f"Nyx_XXIX sealed {len(plaintext)}B in the enclave (round-trip verified)"


# ---------------------------------------------------------------------------
# Atlas_XXX — distributed orchestration (distributed) — async-in-proc
# ---------------------------------------------------------------------------


def _op_atlas(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``distributed.DistributedMercuryCluster.detect_anomalies`` via asyncio.run."""
    data = task.payload.get("data")
    if data is None:
        return None
    try:
        from omni_mercury_engine.distributed import DistributedMercuryCluster
    except ImportError:
        return None

    X = _as_2d_float(data, what="Atlas_XXX.data")
    raw_nodes = task.payload.get("nodes")
    nodes = (
        [str(n) for n in raw_nodes]
        if isinstance(raw_nodes, Sequence) and not isinstance(raw_nodes, (str, bytes))
        else ["node_0", "node_1"]
    )
    timeout = float(task.payload.get("timeout", _CLUSTER_TIMEOUT_S))

    async def _run() -> dict[str, Any]:
        cluster = DistributedMercuryCluster(nodes=nodes)
        await cluster.start()
        try:
            result: dict[str, Any] = await cluster.detect_anomalies(X)
            return result
        finally:
            await cluster.stop()

    try:
        result = asyncio.run(asyncio.wait_for(_run(), timeout=timeout))
    except TimeoutError as exc:
        raise SubAgentExecutionError(
            f"Atlas_XXX: cluster detection timed out after {timeout}s"
        ) from exc
    scores = np.asarray(result.get("anomaly_scores", []), dtype=np.float64).ravel()
    output = {
        "operation": "distributed.DistributedMercuryCluster.detect_anomalies",
        "nodes": nodes,
        "n_results": int(result.get("n_results", scores.size)),
        "aggregation_method": result.get("aggregation_method"),
        "mean_score": float(np.mean(scores)) if scores.size else 0.0,
    }
    return (
        output,
        1.0,
        (
            f"Atlas_XXX ran distributed detection on {len(nodes)} node(s) over {X.shape[0]} sample(s)"
        ),
    )


# ---------------------------------------------------------------------------
# Harmonia_XXXI — normalization / canonicalization (utils)
# ---------------------------------------------------------------------------


def _op_harmonia(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``utils.normalize_data`` (canonicalize into a standard scale)."""
    data = task.payload.get("data")
    if data is None:
        return None
    try:
        from omni_mercury_engine import utils
    except ImportError:
        return None

    arr = _as_2d_float(data, what="Harmonia_XXXI.data")
    method = str(task.payload.get("method", "standard"))
    normalized = np.asarray(utils.normalize_data(arr, method=method), dtype=np.float64)
    output = {
        "operation": "utils.normalize_data",
        "method": method,
        "shape": list(normalized.shape),
        "min": float(np.min(normalized)),
        "max": float(np.max(normalized)),
        "mean": float(np.mean(normalized)),
    }
    return output, 1.0, f"Harmonia_XXXI canonicalized {arr.shape} via '{method}'"


# ---------------------------------------------------------------------------
# Hyperion_XXXII — HPC / GPU scheduling (scaling, ml)
# ---------------------------------------------------------------------------


def _op_hyperion(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``scaling`` power/impact estimation (HPC budget) or batch scoring."""
    payload = task.payload
    if payload.get("model_size") is not None:
        try:
            from omni_mercury_engine.scaling import BainAIScaling
        except ImportError:
            return None
        scaler = BainAIScaling()
        power = float(
            scaler.estimate_power_consumption(
                model_size=int(payload["model_size"]),
                batch_size=int(payload.get("batch_size", 32)),
                sequence_length=int(payload.get("sequence_length", 512)),
            )
        )
        impact = scaler.estimate_agentic_ai_impact(
            current_workforce_size=int(payload.get("workforce_size", 100)),
            process_automation_target=float(payload.get("automation_target", 0.3)),
        )
        output = {
            "operation": "scaling.BainAIScaling.estimate_power_consumption",
            "power_watts": power,
            # The power/impact numbers are an uncalibrated order-of-magnitude
            # heuristic (illustrative coefficients, no hardware curve), surfaced
            # explicitly so a consumer never treats them as a real HPC budget.
            "calibrated": False,
            "impact_keys": sorted(impact)[:8] if isinstance(impact, dict) else [],
        }
        return (
            output,
            1.0,
            f"Hyperion_XXXII heuristic power estimate (uncalibrated): {power:.1f}W",
        )

    data = payload.get("data")
    if data is None:
        data = payload.get("X")
    if data is None:
        return None
    try:
        from omni_mercury_engine.ml import quick_anomaly_score
    except ImportError:
        return None
    X = _as_2d_float(data, what="Hyperion_XXXII.data")
    scores = np.asarray(quick_anomaly_score(X.astype(np.float32)), dtype=np.float64)
    mean_score = float(np.mean(scores))
    output = {
        "operation": "ml.quick_anomaly_score",
        "n_samples": int(X.shape[0]),
        "mean_score": mean_score,
    }
    return (
        output,
        _clamp01(mean_score),
        (f"Hyperion_XXXII scored {X.shape[0]} sample(s) on the compute fabric"),
    )


# ---------------------------------------------------------------------------
# Rhea_XXXIII — dependency / resilience control (resilience) — no input
# ---------------------------------------------------------------------------


def _op_rhea(agent: CoordinatorSubAgent, task: SubAgentTask) -> OperationResult | None:
    """Real op: ``resilience.get_all_breaker_stats`` + ``SelfHealingEngine.get_system_health``."""
    try:
        from omni_mercury_engine.resilience import get_all_breaker_stats
        from omni_mercury_engine.resilience.self_healing import SelfHealingEngine
    except ImportError:
        return None

    stats = get_all_breaker_stats()
    health = SelfHealingEngine().get_system_health()
    overall = str(health.get("overall_health", "unknown"))
    open_breakers = [n for n, s in stats.items() if str(s.get("state", "")).upper() == "OPEN"]
    output = {
        "operation": "resilience.get_all_breaker_stats+SelfHealingEngine.get_system_health",
        "n_breakers": len(stats),
        "n_open": len(open_breakers),
        "overall_health": overall,
        "adaptive_defense": dict(health.get("adaptive_defense", {}) or {}),
    }
    confidence = 1.0 if overall == "healthy" and not open_breakers else 0.5
    return (
        output,
        confidence,
        (f"Rhea_XXXIII resilience: {overall}, {len(stats)} breaker(s), {len(open_breakers)} open"),
    )


# ---------------------------------------------------------------------------
# The registry: pantheon id -> real-entrypoint adapter.
# ---------------------------------------------------------------------------

OPERATIONS: dict[str, Adapter] = {
    "Hestia_II": _op_hestia,
    "Hermes_III": _op_hermes,
    "Athena_IV": _op_athena,
    "Apollo_V": _op_apollo,
    "Artemis_VI": _op_artemis,
    "Poseidon_IX": _op_poseidon,
    "Demeter_X": _op_demeter,
    "Hephaestus_XI": _op_hephaestus,
    "Eleos_XII": _op_eleos,
    "Hades_XV": _op_hades,
    "Selene_XVI": _op_selene,
    "Helios_XVII": _op_helios,
    "Eos_XVIII": _op_eos,
    "Nemesis_XIX": _op_nemesis,
    "Tyche_XX": _op_tyche,
    "Zelos_XXI": _op_zelos,
    "Kronos_XXII": _op_kronos,
    "Morpheus_XXIII": _op_morpheus,
    "Iris_XXIV": _op_iris,
    "Pan_XXV": _op_pan,
    "Persephone_XXVI": _op_persephone,
    "Prometheus_XXVII": _op_prometheus,
    "Hecate_XXVIII": _op_hecate,
    "Nyx_XXIX": _op_nyx,
    "Atlas_XXX": _op_atlas,
    "Harmonia_XXXI": _op_harmonia,
    "Hyperion_XXXII": _op_hyperion,
    "Rhea_XXXIII": _op_rhea,
}


def operation_for(pantheon_id: str) -> Adapter | None:
    """Return the real-entrypoint adapter for a coordinator, or ``None`` if none."""
    return OPERATIONS.get(pantheon_id)
