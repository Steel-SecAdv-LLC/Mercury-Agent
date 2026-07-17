# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit contracts for the GOSNN disagreement demotion overlay.

The overlay reads ``Evidence.gosnn_anomaly_prob`` (from
``gosnn_metadata["detection"]``, present only when a detection-metric
merit-gated head shipped) and demotes a grounded verdict to ``DEFER`` on
strong disagreement at the shipped thresholds.  Laws pinned here:

* abstention-only -- the overlay never grounds, upgrades, or overrides the
  ethical fail-closed stage;
* inclusive thresholds -- a probability exactly at a demotion threshold
  defers (fail-closed reading);
* structural absence -- no head signal, malformed signal, or a disabled
  policy knob means byte-identical behaviour to the pre-channel decider.
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.evidence import Evidence
from omni_mercury_engine.decision.policy import DecisionPolicy
from omni_mercury_engine.decision.states import Disposition
from omni_mercury_engine.verifiers.three_state import ThreeState


def _result(
    *,
    anomaly_prob: float,
    detection: dict[str, Any] | None = None,
    ethical: bool = True,
) -> dict[str, Any]:
    """A grounded-by-conformal detection result with an optional GOSNN block."""
    label = 1 if anomaly_prob >= 0.5 else 0
    gosnn: dict[str, Any] = {"ethical_gate_passed": ethical}
    if detection is not None:
        gosnn["detection"] = detection
    return {
        "anomaly_prob": anomaly_prob,
        "threshold_used": 0.5,
        "is_anomaly": bool(label),
        "conformal": {"prediction_set": [label], "set_size": 1, "coverage": 0.9},
        "gosnn_metadata": gosnn,
    }


_THRESHOLDS = {"demote_act_below": 0.2, "demote_clear_above": 0.8}


class TestEvidenceParsing:
    def test_detection_block_parses(self) -> None:
        ev = Evidence.from_detection(
            _result(anomaly_prob=0.9, detection={"anomaly_prob": 0.7, **_THRESHOLDS})
        )
        assert ev.gosnn_anomaly_prob == pytest.approx(0.7)
        assert ev.gosnn_demote_act_below == pytest.approx(0.2)
        assert ev.gosnn_demote_clear_above == pytest.approx(0.8)

    def test_absent_block_gives_none(self) -> None:
        ev = Evidence.from_detection(_result(anomaly_prob=0.9))
        assert ev.gosnn_anomaly_prob is None
        assert ev.gosnn_demote_act_below is None
        assert ev.gosnn_demote_clear_above is None

    @pytest.mark.parametrize(
        "bad_prob", ["high", None, float("nan"), float("inf"), -0.1, 1.5, [0.5]]
    )
    def test_malformed_probability_drops_the_whole_block(self, bad_prob: Any) -> None:
        ev = Evidence.from_detection(
            _result(anomaly_prob=0.9, detection={"anomaly_prob": bad_prob, **_THRESHOLDS})
        )
        assert ev.gosnn_anomaly_prob is None
        assert ev.gosnn_demote_act_below is None
        assert ev.gosnn_demote_clear_above is None

    def test_non_mapping_block_is_ignored(self) -> None:
        ev = Evidence.from_detection(_result(anomaly_prob=0.9, detection=None))
        assert ev.gosnn_anomaly_prob is None
        result = _result(anomaly_prob=0.9)
        result["gosnn_metadata"]["detection"] = "not-a-dict"
        assert Evidence.from_detection(result).gosnn_anomaly_prob is None

    def test_to_dict_round_trips_the_fields(self) -> None:
        ev = Evidence.from_detection(
            _result(anomaly_prob=0.9, detection={"anomaly_prob": 0.7, **_THRESHOLDS})
        )
        payload = ev.to_dict()
        assert payload["gosnn_anomaly_prob"] == pytest.approx(0.7)
        assert payload["gosnn_demote_act_below"] == pytest.approx(0.2)
        assert payload["gosnn_demote_clear_above"] == pytest.approx(0.8)


class TestOverlayDemotions:
    @pytest.fixture()
    def responder(self) -> DecisionAbstentionResponder:
        return DecisionAbstentionResponder()

    def test_disagreeing_low_prob_demotes_grounded_act(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        record = responder.decide(
            _result(anomaly_prob=0.9, detection={"anomaly_prob": 0.05, **_THRESHOLDS})
        )
        assert record.state is ThreeState.UNAVAILABLE
        assert record.disposition is Disposition.DEFER
        assert any("GOSNN" in reason for reason in record.reasons)

    def test_disagreeing_high_prob_demotes_grounded_clear(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        record = responder.decide(
            _result(anomaly_prob=0.1, detection={"anomaly_prob": 0.95, **_THRESHOLDS})
        )
        assert record.state is ThreeState.UNAVAILABLE
        assert record.disposition is Disposition.DEFER
        assert any("GOSNN" in reason for reason in record.reasons)

    @pytest.mark.parametrize(
        ("verdict_prob", "gosnn_prob"),
        [(0.9, 0.2), (0.1, 0.8)],
    )
    def test_threshold_boundary_is_inclusive(
        self, responder: DecisionAbstentionResponder, verdict_prob: float, gosnn_prob: float
    ) -> None:
        record = responder.decide(
            _result(
                anomaly_prob=verdict_prob, detection={"anomaly_prob": gosnn_prob, **_THRESHOLDS}
            )
        )
        assert record.state is ThreeState.UNAVAILABLE

    @pytest.mark.parametrize(
        ("verdict_prob", "gosnn_prob"),
        [(0.9, 0.9), (0.9, 0.21), (0.1, 0.1), (0.1, 0.79)],
    )
    def test_agreement_or_weak_disagreement_keeps_grounded(
        self, responder: DecisionAbstentionResponder, verdict_prob: float, gosnn_prob: float
    ) -> None:
        record = responder.decide(
            _result(
                anomaly_prob=verdict_prob, detection={"anomaly_prob": gosnn_prob, **_THRESHOLDS}
            )
        )
        assert record.state is ThreeState.GROUNDED

    def test_missing_side_threshold_disables_that_side_only(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        act_only = {"anomaly_prob": 0.05, "demote_act_below": 0.2}
        assert responder.decide(_result(anomaly_prob=0.9, detection=act_only)).state is (
            ThreeState.UNAVAILABLE
        )
        clear_side_missing = {"anomaly_prob": 0.95, "demote_act_below": 0.2}
        assert (
            responder.decide(_result(anomaly_prob=0.1, detection=clear_side_missing)).state
            is ThreeState.GROUNDED
        )

    def test_policy_knob_disables_the_overlay(self) -> None:
        responder = DecisionAbstentionResponder(
            policy=DecisionPolicy(defer_on_gosnn_disagreement=False)
        )
        record = responder.decide(
            _result(anomaly_prob=0.9, detection={"anomaly_prob": 0.0, **_THRESHOLDS})
        )
        assert record.state is ThreeState.GROUNDED
        assert record.signals["policy"]["defer_on_gosnn_disagreement"] is False

    def test_ethical_block_still_dominates(self, responder: DecisionAbstentionResponder) -> None:
        record = responder.decide(
            _result(
                anomaly_prob=0.9,
                detection={"anomaly_prob": 0.9, **_THRESHOLDS},
                ethical=False,
            )
        )
        assert record.state is ThreeState.UNDECIDABLE
        assert record.disposition is Disposition.HOLD

    def test_overlay_never_upgrades_an_abstained_verdict(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        """An agreeing GOSNN probability cannot rescue a conformal don't-know."""
        result = _result(anomaly_prob=0.9, detection={"anomaly_prob": 0.99, **_THRESHOLDS})
        result["conformal"] = {"prediction_set": [0, 1], "set_size": 2, "coverage": 0.9}
        record = responder.decide(result)
        assert record.state is ThreeState.UNAVAILABLE

    def test_signal_absent_matches_pre_channel_behaviour(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        with_block = _result(anomaly_prob=0.9)
        without_block = {key: value for key, value in with_block.items() if key != "gosnn_metadata"}
        without_block["gosnn_metadata"] = {"ethical_gate_passed": True}
        a = responder.decide(with_block).to_dict()
        b = responder.decide(without_block).to_dict()
        assert a == b
