# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Causal contributions ride the record under a stable, bounded key.

``detect_with_fusion`` computes ``detector_importance`` on every call and,
opt-in, an Integrated-Gradients ``explanation`` and a GDPR Shapley report --
but none of it reached the decision record, so a reloaded audit trail could
not say *what drove* the probability it archived. The record's signals now
carry ``contributions``: a compact summary (at most 16 entries per map,
floats rounded to 6 dp), never the raw full explanation, which stays at the
result's top level only.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from omni_mercury_engine.decision import DecisionAbstentionResponder, DecisionRecord
from omni_mercury_engine.decision.evidence import Evidence


def _result(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "anomaly_prob": 0.93,
        "is_anomaly": True,
        "threshold_used": 0.5,
        "severity": 0.4,
        "detector_importance": {"statistical": 0.6123456789, "temporal": 0.25, "spectral": 0.15},
    }
    base.update(over)
    return base


def _explanation(n_features: int = 20) -> dict[str, Any]:
    """An ``Explanation.to_dict()``-shaped IG payload with many features."""
    return {
        "explanation_id": "exp-1",
        "explanation_type": "integrated_gradients",
        "prediction": 0.93,
        "base_value": 0.5,
        "local_accuracy": 0.99,
        "feature_importances": [
            {
                "feature": f"feature_{i}",
                "importance": (0.5 - 0.01 * i) * (-1 if i % 3 == 0 else 1),
                "direction": "positive",
                "confidence": 1.0,
            }
            for i in range(n_features)
        ],
        "human_readable": "long narrative that must never ride the record",
        "counterfactuals": [],
        "faithfulness_scores": {"comprehensiveness": 0.8},
    }


def _gdpr_report(n_features: int = 20) -> dict[str, Any]:
    """An ``ExplanationReport.to_dict()``-shaped Shapley payload."""
    return {
        "report_id": "rep-1",
        "decision": {"id": "d-1", "value": "anomaly"},
        "subject": {"id": "anon", "consent_given": False},
        "explanation": {
            "logic": "narrative",
            "feature_contributions": {
                f"feature_{i}": (0.4 - 0.01 * i) * (-1 if i % 2 else 1) for i in range(n_features)
            },
            "top_factors": [],
            "counterfactual_actions": [],
        },
        "impact": {"significance": "s", "consequences": "c"},
        "rights": "r",
        "generated_at": 0.0,
        "explanation_level": "DETAILED",
    }


class TestContributionsShape:
    """The bounded summary shapes, per source."""

    def test_without_explain_only_importance_is_present(self) -> None:
        rec = DecisionAbstentionResponder().decide(_result())
        contributions = rec.signals["contributions"]
        assert contributions["integrated_gradients"] is None
        assert contributions["shapley"] is None
        importance = contributions["detector_importance"]
        assert importance == {
            "statistical": pytest.approx(0.612346),
            "temporal": pytest.approx(0.25),
            "spectral": pytest.approx(0.15),
        }
        # Rounded to 6 dp exactly, not merely approximately.
        assert importance["statistical"] == 0.612346

    def test_synthetic_explanation_and_gdpr_yield_bounded_maps(self) -> None:
        rec = DecisionAbstentionResponder().decide(
            _result(explanation=_explanation(), gdpr_report=_gdpr_report())
        )
        contributions = rec.signals["contributions"]
        ig = contributions["integrated_gradients"]
        shap = contributions["shapley"]
        assert isinstance(ig, dict) and len(ig) == 16
        assert isinstance(shap, dict) and len(shap) == 16
        # Top-k by |value|: the strongest feature is kept, the weakest cut.
        assert ig["feature_0"] == pytest.approx(-0.5)
        assert "feature_19" not in ig
        assert shap["feature_0"] == pytest.approx(0.4)
        assert "feature_19" not in shap
        # Never the raw full explanation: no narrative or nested payloads.
        assert all(isinstance(v, float) for v in ig.values())
        assert all(isinstance(v, float) for v in shap.values())
        assert "human_readable" not in json.dumps(dict(rec.signals))

    def test_empty_importance_reads_absent(self) -> None:
        rec = DecisionAbstentionResponder().decide(_result(detector_importance={}))
        assert rec.signals["contributions"]["detector_importance"] is None

    def test_malformed_payloads_read_absent_not_crash(self) -> None:
        rec = DecisionAbstentionResponder().decide(
            _result(
                detector_importance={"statistical": "not-a-number"},
                explanation="not-a-mapping",
                gdpr_report={"explanation": []},
            )
        )
        contributions = rec.signals["contributions"]
        assert contributions == {
            "detector_importance": None,
            "integrated_gradients": None,
            "shapley": None,
        }


class TestContributionsRoundTrip:
    """Serialisation carries the summary; old records reload without it."""

    def test_record_round_trips_contributions(self) -> None:
        rec = DecisionAbstentionResponder().decide(
            _result(explanation=_explanation(), gdpr_report=_gdpr_report())
        )
        wire = json.loads(json.dumps(rec.to_dict()))
        reloaded = DecisionRecord.from_dict(wire)
        assert reloaded.signals["contributions"] == rec.signals["contributions"]

    def test_from_dict_tolerates_records_saved_without_contributions(self) -> None:
        rec = DecisionAbstentionResponder().decide(_result())
        wire = rec.to_dict()
        wire["signals"].pop("contributions", None)
        reloaded = DecisionRecord.from_dict(wire)
        assert "contributions" not in reloaded.signals
        assert reloaded.to_dict() == DecisionRecord.from_dict(reloaded.to_dict()).to_dict()

    def test_evidence_default_contributions_is_none(self) -> None:
        evidence = Evidence(anomaly_prob=0.5, is_anomaly=False, threshold=0.5, severity=0.0)
        assert evidence.contributions is None
        assert evidence.to_dict()["contributions"] is None
