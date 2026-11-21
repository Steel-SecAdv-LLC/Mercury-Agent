# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The self-critique challenges detector results in both directions.

Before this, the critique was advisory only: the decider never read
``Evidence.cognitive``, so the orchestrator could neither demote a verdict it
found unreliable nor put a challenge to an abstention on the record.  These
tests pin the two directions and their guard rails:

* a *measured* unreliable critique demotes a grounded verdict to DEFER
  (needs a human);
* a critique built only from placeholder uncertainties never moves a verdict;
* a reliable, *calibrated* critique at or above the policy floor disputes a
  resolvable abstention on the record, without overturning it;
* a fail-closed hold (ethical block, atypical point) is never challenged.
"""

from __future__ import annotations

from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.evidence import Evidence
from omni_mercury_engine.decision.policy import DecisionPolicy
from omni_mercury_engine.decision.states import Disposition
from omni_mercury_engine.verifiers.three_state import ThreeState


def _grounded_evidence(cognitive: dict[str, object] | None) -> Evidence:
    """A singleton-certificate positive that grounds as ACT absent overlays."""
    return Evidence(
        anomaly_prob=0.95,
        is_anomaly=True,
        threshold=0.5,
        severity=0.8,
        conformal_set_size=1,
        conformal_labels=(1,),
        coverage=0.9,
        cognitive=cognitive,
    )


class TestDisputeDirection:
    """A measured unreliable critique demotes; a placeholder one cannot."""

    def test_measured_unreliable_critique_demotes_grounded_verdict(self) -> None:
        record = DecisionAbstentionResponder().decide_from_evidence(
            _grounded_evidence({"is_reliable": False, "epistemic_measured": True})
        )
        assert record.state is ThreeState.UNAVAILABLE
        assert record.disposition is Disposition.DEFER
        assert record.abstained is True
        assert any("self-critique" in reason for reason in record.reasons)
        # Model-vs-critic contradiction routes to a human, not a re-input.
        assert record.response.requires_human is True

    def test_placeholder_critique_never_moves_a_verdict(self) -> None:
        record = DecisionAbstentionResponder().decide_from_evidence(
            _grounded_evidence(
                {
                    "is_reliable": False,
                    "epistemic_measured": False,
                    "aleatoric_measured": False,
                    "confidence_calibrated": False,
                }
            )
        )
        assert record.state is ThreeState.GROUNDED
        assert record.abstained is False

    def test_unmeasured_flags_absent_count_as_placeholder(self) -> None:
        record = DecisionAbstentionResponder().decide_from_evidence(
            _grounded_evidence({"is_reliable": False})
        )
        assert record.state is ThreeState.GROUNDED

    def test_policy_knob_disables_the_overlay(self) -> None:
        responder = DecisionAbstentionResponder(
            policy=DecisionPolicy(defer_on_unreliable_critique=False)
        )
        record = responder.decide_from_evidence(
            _grounded_evidence({"is_reliable": False, "epistemic_measured": True})
        )
        assert record.state is ThreeState.GROUNDED


class TestSupportDirection:
    """A calibrated, reliable critique disputes an abstention on the record."""

    def test_calibrated_reliable_critique_disputes_a_deferral(self) -> None:
        evidence = Evidence(
            anomaly_prob=0.52,
            is_anomaly=True,
            threshold=0.5,
            severity=0.3,
            cognitive={
                "is_reliable": True,
                "confidence": 0.9,
                "confidence_calibrated": True,
            },
        )
        record = DecisionAbstentionResponder().decide_from_evidence(evidence)
        # The abstention stands: the challenge is on the record, not a flip.
        assert record.abstained is True
        assert any("disputes this abstention" in reason for reason in record.reasons)

    def test_uncalibrated_confidence_cannot_dispute(self) -> None:
        evidence = Evidence(
            anomaly_prob=0.52,
            is_anomaly=True,
            threshold=0.5,
            severity=0.3,
            cognitive={
                "is_reliable": True,
                "confidence": 0.9,
                "confidence_calibrated": False,
            },
        )
        record = DecisionAbstentionResponder().decide_from_evidence(evidence)
        assert not any("disputes this abstention" in reason for reason in record.reasons)

    def test_confidence_below_floor_cannot_dispute(self) -> None:
        evidence = Evidence(
            anomaly_prob=0.52,
            is_anomaly=True,
            threshold=0.5,
            severity=0.3,
            cognitive={
                "is_reliable": True,
                "confidence": 0.5,
                "confidence_calibrated": True,
            },
        )
        record = DecisionAbstentionResponder().decide_from_evidence(evidence)
        assert not any("disputes this abstention" in reason for reason in record.reasons)

    def test_fail_closed_hold_is_never_challenged(self) -> None:
        evidence = Evidence.from_ethical_block(
            {"check": "harm_uplift", "score": 0.1, "threshold": 0.5}
        )
        # Even a perfect critique may not dispute an ethical hold.
        evidence = Evidence(
            **{
                **{
                    f: getattr(evidence, f)
                    for f in (
                        "anomaly_prob",
                        "is_anomaly",
                        "threshold",
                        "severity",
                        "ethical_gate_passed",
                        "ethical_score",
                        "ethical_threshold",
                        "ethics_report",
                        "domain",
                    )
                },
                "cognitive": {
                    "is_reliable": True,
                    "confidence": 0.99,
                    "confidence_calibrated": True,
                },
            }
        )
        record = DecisionAbstentionResponder().decide_from_evidence(evidence)
        assert record.state is ThreeState.UNDECIDABLE
        assert record.disposition is Disposition.HOLD
        assert not any("disputes" in reason for reason in record.reasons)

    def test_policy_floor_is_recorded_in_signals(self) -> None:
        record = DecisionAbstentionResponder().decide_from_evidence(_grounded_evidence(None))
        policy = record.signals["policy"]
        assert policy["defer_on_unreliable_critique"] is True
        assert policy["critique_support_confidence_floor"] == 0.75
