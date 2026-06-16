# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end Phase 3 wiring: the live engine surfaces route real proposals
through the real governance seam and the real Phase 2 promotion gate.

The deterministic unit tests in ``test_phase3_governance.py`` prove the routing
*logic*. These tests prove the *wiring*: they instantiate the production
:class:`~omni_mercury_engine.agentic.orchestration.MultiAgentOrchestrator` (with
its real :class:`~omni_mercury_engine.cognitive.reflexion.AnomalyReflexion`
critic) and the production
:class:`~omni_mercury_engine.ml.online_learning.OnlineLearningPipeline` drift
path, and assert that an autonomous threshold move / model retrain is actually
intercepted by governance — withheld fail-closed by default, applied only under
an explicit measurement stance, and routed through the real promotion gate when
a gate-backed policy is installed. This is what distinguishes a connected gate
from a parallel facade: the live decision is the governed decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

# The default detector suite the orchestrator composes requires the ML extra.
pytest.importorskip("torch")

from omni_mercury_engine.agentic.orchestration import MultiAgentOrchestrator
from omni_mercury_engine.governance.self_improvement import (
    FailClosedSelfImprovementGovernance,
    GovernanceOutcome,
    MeasurementGovernance,
    ProposedRecalibration,
)
from omni_mercury_engine.ml.drift import DriftResult, DriftSeverity, DriftType
from omni_mercury_engine.ml.online_learning import (
    OnlineLearningPipeline,
    StreamingSample,
    UpdateStrategy,
)
from research.governed_fusion.phase3_governance_adapters import (
    PromotionGateRecalibrationGovernance,
    PromotionGateThresholdGovernance,
)

if TYPE_CHECKING:
    from omni_mercury_engine.agentic.orchestration import ReflectionRecord

_SEEDS = (0, 1, 2)


# --- promotion-gate fixtures (same shapes as test_phase3_governance.py) --------


def _manifest() -> dict[str, object]:
    return {
        "provenance_summary": {
            "transparent_fitness_bucket": "external_label",
            "real": {"external_label": {"n_events": 2, "n_rows": 100, "n_pos": 20}},
        }
    }


def _ledger() -> dict[str, object]:
    return {
        "schema_version": 1,
        "transparent_fitness_bucket": "external_label",
        "runs": [
            {"status": "ok", "full": {"auroc": 0.770, "auprc": 0.550, "f1": 0.180, "n_events": 2}}
        ],
    }


def _results(auroc: float, auprc: float, f1: float, *, n_events: int = 2) -> dict[str, object]:
    return {
        "per_provenance": {
            "external_label": {
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
                "precision": 0.50,
                "recall": 0.50,
                "n_events": n_events,
            },
            "self_label": {"auroc": 0.99, "auprc": 0.99, "f1": 0.99, "n_events": 21},
        },
        "overall": {"auroc": 0.90, "auprc": 0.90, "f1": 0.90, "n_events": 23},
    }


def _passing_candidate_record(candidate_id: str = "reflexion-live-candidate") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "baseline_id": "main",
        "evaluation_mode": "held_out_replay",
        "optimization_bucket": "external_label",
        "shadow_replay": {
            "baseline": _results(0.770, 0.550, 0.180),
            "candidate": _results(0.783, 0.560, 0.195),
        },
        "ablation": {"full": {"auroc": 0.783, "auprc": 0.560, "f1": 0.195, "n_events": 2}},
        "safety": {
            "sigma_immutable": 0.94,
            "benevolence": 0.995,
            "conformal_coverage": 0.91,
            "lyapunov_lambda": 0.001,
        },
        "capability_regression": {"passed": True, "failed": []},
    }


# --- orchestrator fixtures -----------------------------------------------------


def _planted_outlier_data(
    seed: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    x_train = rng.normal(0.0, 1.0, size=(250, 6))
    x_test = np.vstack([rng.normal(0.0, 1.0, size=(150, 6)), rng.normal(6.0, 1.0, size=(15, 6))])
    y_test = np.array([False] * 150 + [True] * 15)
    return x_train, x_test, y_test


def _run_until_actionable(
    governance: Any,
) -> tuple[MultiAgentOrchestrator, ReflectionRecord]:
    """Fit at a deliberately-too-high operating point so reflexion proposes a
    change, then run one labeled episode. Returns the orchestrator and the
    reflection record for the first seed that yields an actionable proposal.
    """
    for seed in _SEEDS:
        x_train, x_test, y_test = _planted_outlier_data(seed)
        orch = MultiAgentOrchestrator(
            seed=seed, operating_threshold=0.9, threshold_governance=governance
        ).fit(x_train)
        episode = orch.run_episode(x_test, y_test)
        assert episode.reflection is not None
        if episode.reflection.recommendation != "maintain":
            return orch, episode.reflection
    pytest.skip("ensemble already separated at 0.9 across all probe seeds")


# --- live reflexion governance -------------------------------------------------


class TestLiveReflexionGovernance:
    """The live reflexion critic cannot move the operating point on its own."""

    def test_autonomous_change_withheld_fail_closed_by_default(self) -> None:
        orch, rec = _run_until_actionable(FailClosedSelfImprovementGovernance())
        assert rec.recommendation in {"increase", "decrease"}
        assert rec.governed is True
        assert rec.applied is False
        assert rec.governance_outcome == GovernanceOutcome.WITHHELD.value
        # The live operating point is unmoved — the boundary did not shift.
        assert orch.operating_threshold == rec.threshold_before
        assert rec.threshold_after == rec.threshold_before
        assert any("withheld" in reason for reason in rec.governance_reasons)

    def test_change_applied_only_under_explicit_measurement(self) -> None:
        orch, rec = _run_until_actionable(MeasurementGovernance())
        assert rec.applied is True
        assert rec.governance_outcome == GovernanceOutcome.APPLIED.value
        assert orch.operating_threshold == rec.threshold_suggested
        assert orch.operating_threshold != rec.threshold_before

    def test_routed_through_gate_withheld_without_candidate_evidence(self) -> None:
        gov = PromotionGateThresholdGovernance(manifest=_manifest(), ledger=_ledger())
        orch, rec = _run_until_actionable(gov)
        assert rec.applied is False
        assert rec.governance_outcome == GovernanceOutcome.WITHHELD.value
        assert orch.operating_threshold == rec.threshold_before
        # The fail-closed disposition is the real gate's reject, recorded.
        assert rec.governance_record is not None
        assert rec.governance_record["surface"] == "reflexion_threshold"
        assert rec.governance_record["action"] == "reject"

    def test_routed_through_gate_queued_with_passing_evidence(self) -> None:
        gov = PromotionGateThresholdGovernance(
            manifest=_manifest(),
            ledger=_ledger(),
            evidence_provider=lambda _change: _passing_candidate_record(),
        )
        orch, rec = _run_until_actionable(gov)
        # The gate promotes, but promotion is human-review gated — the live
        # boundary is still NOT moved autonomously; the candidate is queued.
        assert rec.applied is False
        assert rec.governance_outcome == GovernanceOutcome.QUEUED.value
        assert orch.operating_threshold == rec.threshold_before
        assert rec.governance_record is not None
        assert rec.governance_record["action"] == "queue_reflexion_candidate"
        assert rec.governance_record["gate_decision"] == "promote"


# --- live drift recalibration governance --------------------------------------


class _SpyModel:
    """Minimal model that records whether it was retrained."""

    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(self, x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        self.fit_calls += 1

    def predict(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.zeros(len(x), dtype=int)


def _pipeline_with_labeled_buffer(governance: Any) -> tuple[OnlineLearningPipeline, _SpyModel]:
    model = _SpyModel()
    pipeline = OnlineLearningPipeline(
        model=model,
        update_strategy=UpdateStrategy.FULL_RETRAIN,
        drift_detection=False,
        recalibration_governance=governance,
    )
    rng = np.random.default_rng(0)
    for i in range(12):
        pipeline.buffer.add(
            StreamingSample(features=rng.normal(size=6), label=i % 2, sample_id=f"s{i}")
        )
    return pipeline, model


def _high_drift() -> DriftResult:
    return DriftResult(
        is_drift=True,
        drift_type=DriftType.CONCEPT_DRIFT,
        severity=DriftSeverity.HIGH,
        p_value=1e-5,
        test_statistic=12.0,
        threshold=0.05,
        message="synthetic high-severity drift",
    )


class TestLiveDriftRecalibrationGovernance:
    """High/critical drift cannot retrain the live model without governance."""

    def test_autonomous_retrain_withheld_fail_closed(self) -> None:
        pipeline, model = _pipeline_with_labeled_buffer(FailClosedSelfImprovementGovernance())
        pipeline._handle_drift(_high_drift())
        assert model.fit_calls == 0
        review = pipeline.last_recalibration_review
        assert review is not None
        assert review.applied is False
        assert review.outcome == GovernanceOutcome.WITHHELD.value

    def test_retrain_applied_only_under_explicit_measurement(self) -> None:
        pipeline, model = _pipeline_with_labeled_buffer(MeasurementGovernance())
        pipeline._handle_drift(_high_drift())
        assert model.fit_calls == 1
        review = pipeline.last_recalibration_review
        assert review is not None and review.applied is True

    def test_standalone_default_remains_autonomous(self) -> None:
        # With no governance installed the pipeline is a standalone online
        # learner and adapts autonomously, as documented.
        pipeline, model = _pipeline_with_labeled_buffer(None)
        pipeline._handle_drift(_high_drift())
        assert model.fit_calls == 1
        assert pipeline.last_recalibration_review is None

    def test_routed_through_gate_withheld_without_evidence(self) -> None:
        gov = PromotionGateRecalibrationGovernance(manifest=_manifest(), ledger=_ledger())
        pipeline, model = _pipeline_with_labeled_buffer(gov)
        pipeline._handle_drift(_high_drift())
        assert model.fit_calls == 0
        review = pipeline.last_recalibration_review
        assert review is not None
        assert review.applied is False
        assert review.outcome == GovernanceOutcome.WITHHELD.value
        assert review.record is not None
        assert review.record["surface"] == "drift_recalibration"
        assert review.record["action"] == "reject"


# --- gate-backed adapter contract (no engine episode required) -----------------


class TestGateBackedAdapters:
    """The adapters never authorise autonomous application."""

    def test_threshold_adapter_rejects_without_evidence(self) -> None:
        from omni_mercury_engine.governance.self_improvement import ProposedThresholdChange

        gov = PromotionGateThresholdGovernance(manifest=_manifest(), ledger=_ledger())
        review = gov.review_threshold_change(
            ProposedThresholdChange("reflexion_threshold", "increase", 0.5, 0.63)
        )
        assert review.applied is False
        assert review.outcome == GovernanceOutcome.WITHHELD.value
        assert review.record is not None and review.record["action"] == "reject"

    def test_threshold_adapter_queues_with_passing_evidence(self) -> None:
        from omni_mercury_engine.governance.self_improvement import ProposedThresholdChange

        gov = PromotionGateThresholdGovernance(
            manifest=_manifest(),
            ledger=_ledger(),
            evidence_provider=lambda _c: _passing_candidate_record(),
        )
        review = gov.review_threshold_change(
            ProposedThresholdChange("reflexion_threshold", "increase", 0.5, 0.63)
        )
        assert review.applied is False
        assert review.outcome == GovernanceOutcome.QUEUED.value
        assert review.record is not None and review.record["gate_decision"] == "promote"

    def test_recalibration_adapter_withholds_non_drift_trigger(self) -> None:
        gov = PromotionGateRecalibrationGovernance(manifest=_manifest(), ledger=_ledger())
        review = gov.review_recalibration(
            ProposedRecalibration(
                surface="drift_recalibration",
                trigger="performance_degradation",
                severity="none",
                is_drift=False,
            )
        )
        assert review.applied is False
        assert review.outcome == GovernanceOutcome.WITHHELD.value
