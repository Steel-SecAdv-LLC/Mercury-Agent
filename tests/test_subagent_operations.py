# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the coordinator *operations* layer (phase 2).

Phase 1 made every coordinator a real subsystem *binding* (import + introspect).
Phase 2 deepens each into a genuine subsystem *operator*: this module proves, for
all 28 coordinators, that

* the **real-op** path runs the member's actual ``omni_mercury_engine``
  entrypoint with valid payload-derived inputs (``mode == "operation"``, with the
  concrete entrypoint named in ``output["operation"]``) — *not* the fallback; and
* the **no-input fallback** still returns the transparent live binding report
  (``mode == "binding"``) via the explicit ``mode="introspect"`` readiness probe.

A coordinator whose only passing path is the fallback is a gap — every member
below has a real-op case with genuine inputs and a domain-specific assertion on
the real result. Fail-closed behavior (malformed inputs → ``failed``, never a
fabricated success) is pinned at the bottom. All work flows through the
engine-mediated fleet and its dual commit gate, exactly as in production.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from omni_mercury_engine.agentic.mercury_a_agent import DomainType
from omni_mercury_engine.agentic.subagents.base import _INTERNAL, SubAgentTask
from omni_mercury_engine.agentic.subagents.fleet import SubAgentFleet
from omni_mercury_engine.agentic.subagents.operations import OPERATIONS
from omni_mercury_engine.agentic.subagents.roster import ROSTER


def _fleet() -> SubAgentFleet:
    return SubAgentFleet(access=_INTERNAL, seed=0)


def _dispatch(agent_id: str, payload: dict[str, Any]) -> Any:
    return _fleet().dispatch(
        SubAgentTask(
            description=f"{agent_id} operation", domain=DomainType.GENERAL, payload=payload
        ),
        agent_id,
    )


# A representative detection_result accepted by the cognitive/narrative/decision ops.
_DET: dict[str, Any] = {
    "is_anomaly": True,
    "anomaly_detected": True,
    "anomaly_prob": 0.72,
    "severity": 0.5,
    "confidence": 0.8,
    "threshold": 0.5,
    "calibrated": True,
}


class _ThresholdEstimator:
    """A genuine (if minimal) duck-typed estimator for the validation pipeline."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _ThresholdEstimator:
        self._center = np.asarray(X, dtype=np.float64).mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(np.asarray(X, dtype=np.float64) - self._center, axis=1)
        return np.asarray(dist > np.median(dist), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(np.asarray(X, dtype=np.float64) - self._center, axis=1)
        p = (dist - dist.min()) / (np.ptp(dist) + 1e-9)
        return np.asarray(np.c_[1.0 - p, p], dtype=np.float64)


def _athena_payload() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (48, 4))
    y = (np.linalg.norm(X, axis=1) > 2.2).astype(int)
    return {"model": _ThresholdEstimator(), "X": X.tolist(), "y": y.tolist()}


_RNG = np.random.default_rng(0)

# Each case: (id, payload, expected operation substring, predicate over output dict).
_CASES: list[tuple[str, dict[str, Any], str, Callable[[dict[str, Any]], bool]]] = [
    (
        "Hestia_II",
        {"data": [[1.0, 2, 3], [4, 5, 6], [7, 8, 9]], "method": "minmax"},
        "utils.normalize_data",
        lambda o: o["shape"] == [3, 3],
    ),
    (
        "Hermes_III",
        {"detection_result": _DET},
        "narrative.process_detection",
        lambda o: isinstance(o["summary"], str),
    ),
    (
        "Athena_IV",
        _athena_payload(),
        "validation.ValidationPipeline.validate",
        lambda o: 0.0 <= o["f1_score"] <= 1.0,
    ),
    (
        "Apollo_V",
        {"iris_image": (_RNG.random((100, 100)) * 255).astype("uint8").tolist()},
        "biometric.detect_anomaly",
        lambda o: 0.0 <= o["anomaly_score"] <= 1.0,
    ),
    (
        "Artemis_VI",
        {"sources": ["USGSEarthquakeSource"], "timeout": 20},
        "data_sources.fetch_all",
        lambda o: o["n_attempted"] >= 1 and "reachability" in o,
    ),
    (
        "Poseidon_IX",
        {"data": b"sensitive vault payload"},
        "crypto.encrypt+hash_data",
        lambda o: o["roundtrip_ok"] is True,
    ),
    (
        "Demeter_X",
        {"detection_result": _DET},
        "cognitive.CognitiveOrchestrator.analyze",
        lambda o: "benevolence_score" in o,
    ),
    (
        "Hephaestus_XI",
        {
            "workloads": [{"id": "w1", "priority": 1.0}, {"id": "w2", "priority": 0.5}],
            "resources": {"cpu_cores": 8, "gpu_count": 2},
        },
        "scaling.BainAIScaling.optimize_compute_allocation",
        lambda o: o["n_workloads"] == 2,
    ),
    (
        "Eleos_XII",
        {"detection_result": _DET},
        "narrative.MercuryConversationInterface.process_detection",
        lambda o: "summary" in o,
    ),
    (
        "Hades_XV",
        {"data": _RNG.normal(0, 1, (8, 6)).tolist()},
        "utils.compress_information+crypto.hash_data",
        lambda o: o["compression_ratio"] > 0.0,
    ),
    (
        "Selene_XVI",
        {"points": _RNG.normal(0, 1, (80, 1)).tolist()},
        "streaming.StreamingDetector.ingest",
        lambda o: o["is_ready"] is True,
    ),
    (
        "Helios_XVII",
        {"labels": [0, 0, 0, 1, 1, 1], "scores": [0.1, 0.2, 0.15, 0.8, 0.9, 0.85]},
        "metrics.AnomalyMetrics.compute_all",
        lambda o: "auroc" in o,
    ),
    (
        "Eos_XVIII",
        {"user_id": "u1", "username": "alice", "roles": ["user"]},
        "api.auth.JWTAuth.create_token+validate",
        lambda o: o["validated"] is True,
    ),
    (
        "Nemesis_XIX",
        {"y_true": [0, 0, 1, 1], "y_score": [0.1, 0.3, 0.7, 0.95]},
        "evaluation.evaluate_anomaly_detection",
        lambda o: 0.0 <= o["auc_roc"] <= 1.0,
    ),
    (
        "Tyche_XX",
        {"detection_result": _DET},
        "decision.DecisionAbstentionResponder.decide",
        lambda o: o["disposition"] != "",
    ),
    (
        "Zelos_XXI",
        {"data": _RNG.normal(0, 1, (30, 4)).tolist()},
        "ml.quick_anomaly_score",
        lambda o: o["n_samples"] == 30,
    ),
    (
        "Kronos_XXII",
        {"data": _RNG.normal(0, 1, (15, 4)).tolist(), "train": _RNG.normal(0, 1, (60, 4)).tolist()},
        "detectors.MercuryAnomalyDetector.detect",
        lambda o: o["n_samples"] == 15,
    ),
    (
        "Morpheus_XXIII",
        {"detection_result": _DET, "scenario": "ransomware_outbreak"},
        "cognitive.CognitiveOrchestrator.analyze",
        lambda o: o["scenario"] == "ransomware_outbreak",
    ),
    (
        "Iris_XXIV",
        {
            "headline": "Seismic event",
            "description": "Anomalous activity detected",
            "score": 0.8,
            "area": "Region-1",
            "domain": "earthquake",
        },
        "alerting.CAPAlertGenerator.generate_alert",
        lambda o: o["is_cap_xml"] is True,
    ),
    (
        "Pan_XXV",
        {"scalars": {"bias": 0.9, "fairness": 0.85, "integrity": 0.8}},
        "core.GlobalOmniScalarNetwork.compute_global_intelligence_score",
        lambda o: o["n_registered"] == 3,
    ),
    ("Persephone_XXVI", {}, "resilience.get_all_breaker_stats", lambda o: "n_breakers" in o),
    (
        "Prometheus_XXVII",
        {"X": _RNG.normal(0, 1, (25, 5)).tolist()},
        "ml.quick_anomaly_score",
        lambda o: o["n_samples"] == 25,
    ),
    (
        "Hecate_XXVIII",
        {
            "routes": [{"pattern": "/api/x/{id}", "methods": ["GET"]}],
            "request": {"path": "/api/x/42", "method": "GET"},
        },
        "integrations.routing.RequestRouter.match",
        lambda o: o["matched"] and o["params"] == {"id": "42"},
    ),
    (
        "Nyx_XXIX",
        {"data": b"enclave secret"},
        "crypto.encrypt+hash_data",
        lambda o: o["ciphertext_len"] > 0,
    ),
    (
        "Atlas_XXX",
        {"data": _RNG.normal(0, 1, (20, 4)).tolist(), "timeout": 15},
        "distributed.DistributedMercuryCluster.detect_anomalies",
        lambda o: len(o["nodes"]) >= 1,
    ),
    (
        "Harmonia_XXXI",
        {"data": _RNG.normal(0, 1, (12, 4)).tolist(), "method": "robust"},
        "utils.normalize_data",
        lambda o: o["method"] == "robust",
    ),
    (
        "Hyperion_XXXII",
        {"model_size": 1_000_000_000, "batch_size": 16, "sequence_length": 256},
        "scaling.BainAIScaling.estimate_power_consumption",
        lambda o: o["power_watts"] > 0.0,
    ),
    (
        "Rhea_XXXIII",
        {},
        "resilience.get_all_breaker_stats+SelfHealingEngine.get_system_health",
        lambda o: "overall_health" in o,
    ),
]

_CASE_IDS = [c[0] for c in _CASES]


def test_all_coordinators_have_a_real_op_case() -> None:
    """Every coordinator in the roster must (a) have an adapter and (b) be covered.

    This is the gap guard: a coordinator with no real-op adapter, or one not
    exercised by a real-op case below, blocks the PR.
    """
    coordinators = {e.id for e in ROSTER if e.depth == "coordinator"}
    assert coordinators == set(OPERATIONS), (
        f"adapter/roster mismatch: "
        f"missing_adapter={coordinators - set(OPERATIONS)}, "
        f"orphan_adapter={set(OPERATIONS) - coordinators}"
    )
    assert coordinators == set(_CASE_IDS), f"untested coordinators: {coordinators - set(_CASE_IDS)}"
    assert len(coordinators) == 28


@pytest.mark.parametrize(("agent_id", "payload", "op", "predicate"), _CASES, ids=_CASE_IDS)
def test_coordinator_real_op_runs(
    agent_id: str,
    payload: dict[str, Any],
    op: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> None:
    """The real entrypoint runs (mode='operation'), not the binding fallback."""
    result = _dispatch(agent_id, payload)
    assert result.status == "completed", f"{agent_id} failed: {result.error}"
    out = result.output
    assert out["mode"] == "operation", f"{agent_id} fell back to {out.get('mode')}"
    assert out["operation"] == op, f"{agent_id} op was {out['operation']!r}"
    assert predicate(out), f"{agent_id} real-op assertion failed: {out}"
    assert result.anchor == _entry_anchor(agent_id)
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize("agent_id", _CASE_IDS)
def test_coordinator_introspect_fallback_binding(agent_id: str) -> None:
    """The transparent no-input fallback still returns the live binding report."""
    result = _dispatch(agent_id, {"mode": "introspect"})
    assert result.status == "completed", f"{agent_id} fallback failed: {result.error}"
    out = result.output
    assert out["mode"] == "binding"
    assert out["bound_subsystems"], f"{agent_id} bound no subsystems"
    assert "subsystems" in out


def _entry_anchor(agent_id: str) -> str:
    return next(e.anchor for e in ROSTER if e.id == agent_id)


# ---------------------------------------------------------------------------
# No-input coordinators: their no-input call *is* the real op (not a fallback).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("agent_id", "op"),
    [
        ("Hestia_II", "utils.constants.OmniCodes.stability"),
        ("Hephaestus_XI", "infrastructure.instantiate_filtered_modules"),
        ("Persephone_XXVI", "resilience.get_all_breaker_stats"),
        ("Rhea_XXXIII", "resilience.get_all_breaker_stats+SelfHealingEngine.get_system_health"),
    ],
)
def test_no_input_coordinators_operate_on_empty_payload(agent_id: str, op: str) -> None:
    """A no-input member runs its real entrypoint even with an empty payload."""
    result = _dispatch(agent_id, {})
    assert result.status == "completed", f"{agent_id} failed: {result.error}"
    assert result.output["mode"] == "operation"
    assert result.output["operation"] == op


# ---------------------------------------------------------------------------
# Secondary real paths (distinct genuine capabilities of a member).
# ---------------------------------------------------------------------------


def test_nemesis_twelvefold_verification_path() -> None:
    dims = {
        "wisdom": 0.8,
        "justice": 0.9,
        "truth": 0.85,
        "protection": 0.7,
        "healing": 0.75,
        "judgment": 0.8,
        "authority": 0.7,
        "knowledge": 0.8,
        "balance": 0.85,
        "strategy": 0.7,
        "order": 0.7,
        "hope": 0.9,
    }
    result = _dispatch("Nemesis_XIX", {"dimension_scores": dims})
    assert result.status == "completed"
    assert result.output["operation"] == "ethical.TwelveFoldVerificationSystem.verify"
    assert "verification_status" in result.output


def test_nyx_security_keygen_path() -> None:
    result = _dispatch("Nyx_XXIX", {"keygen": True})
    assert result.status == "completed", f"keygen failed: {result.error}"
    assert result.output["operation"] == "security.MercuryCrypto.generate_signing_keypair"
    assert result.output["public_key_len"] > 0
    assert result.output["secret_key_len"] > 0


def test_hestia_normalize_and_stability_are_both_real() -> None:
    norm = _dispatch("Hestia_II", {"data": [[1.0, 2], [3, 4]], "method": "standard"})
    assert norm.output["operation"] == "utils.normalize_data"
    stab = _dispatch("Hestia_II", {})
    assert stab.output["operation"] == "utils.constants.OmniCodes.stability"
    assert stab.output["stable"] is True


# ---------------------------------------------------------------------------
# Fail-closed: malformed inputs are surfaced transparently, never fabricated.
# ---------------------------------------------------------------------------


def test_athena_fails_closed_on_non_estimator_model() -> None:
    result = _dispatch("Athena_IV", {"model": object(), "X": [[1.0, 2]], "y": [0]})
    assert result.status == "failed"
    assert result.error is not None


def test_helios_fails_closed_on_length_mismatch() -> None:
    result = _dispatch("Helios_XVII", {"labels": [0, 1, 1], "scores": [0.2, 0.8]})
    assert result.status == "failed"
    assert result.error is not None


def test_poseidon_fails_closed_on_unencodable_data() -> None:
    result = _dispatch("Poseidon_IX", {"data": 12345})
    assert result.status == "failed"
    assert result.error is not None


def test_kronos_fails_closed_on_empty_batch() -> None:
    result = _dispatch("Kronos_XXII", {"data": []})
    assert result.status == "failed"
    assert result.error is not None


def test_real_op_commits_through_dual_gate() -> None:
    """A real coordinator op is committed through the fleet's dual ethical gate."""
    result = _dispatch("Helios_XVII", {"labels": [0, 1, 0, 1], "scores": [0.1, 0.9, 0.2, 0.8]})
    assert result.status == "completed"
    assert result.metadata["committed"] is True
