# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The decision loop adds the 'verify' step (ledger + feedback) around decide."""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.decision import (
    DecisionAbstentionResponder,
    DecisionLedger,
    DecisionLoop,
    DecisionPolicy,
    DecisionRecord,
    Disposition,
)


def _conformal(labels: list[int], coverage: float = 0.9) -> dict[str, Any]:
    return {
        "prediction_set": labels,
        "set_size": len(labels),
        "abstain": len(labels) == 2,
        "coverage": coverage,
    }


def _result(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"anomaly_prob": 0.5, "is_anomaly": False, "threshold_used": 0.5}
    base.update(over)
    return base


class TestStep:
    def test_step_decides_records_and_returns(self) -> None:
        loop = DecisionLoop()
        rec = loop.step(_result(anomaly_prob=0.95, is_anomaly=True, conformal=_conformal([1])))
        assert isinstance(rec, DecisionRecord)
        assert rec.disposition is Disposition.ACT
        # The 'verify' step: the record is in the ledger.
        assert len(loop.ledger) == 1
        assert loop.ledger.entries[0] is rec

    def test_default_ledger_is_fresh_and_unbounded(self) -> None:
        loop = DecisionLoop()
        assert isinstance(loop.ledger, DecisionLedger)
        assert loop.ledger.maxlen is None

    def test_feedback_sink_receives_each_record(self) -> None:
        seen: list[DecisionRecord] = []
        loop = DecisionLoop(feedback=seen.append)
        rec = loop.step(_result(anomaly_prob=0.9, conformal=_conformal([1])))
        assert seen == [rec]

    def test_custom_ledger_and_responder_are_used(self) -> None:
        ledger = DecisionLedger(maxlen=10)
        responder = DecisionAbstentionResponder(
            policy=DecisionPolicy(require_calibrated_for_act=True)
        )
        loop = DecisionLoop(responder, ledger=ledger)
        assert loop.ledger is ledger
        assert loop.responder is responder
        # require_calibrated_for_act: an uncalibrated positive defers.
        rec = loop.step(_result(anomaly_prob=0.95, is_anomaly=True))
        assert rec.disposition is Disposition.DEFER


class TestRunAndSummary:
    def test_run_processes_a_batch(self) -> None:
        loop = DecisionLoop()
        records = loop.run(
            [
                _result(anomaly_prob=0.95, is_anomaly=True, conformal=_conformal([1])),
                _result(anomaly_prob=0.02, conformal=_conformal([0])),
                _result(anomaly_prob=0.55, conformal=_conformal([0, 1])),
            ],
            domain="medical",
        )
        assert len(records) == 3
        assert len(loop.ledger) == 3

    def test_summary_delegates_to_ledger(self) -> None:
        loop = DecisionLoop()
        loop.run(
            [
                _result(anomaly_prob=0.95, is_anomaly=True, conformal=_conformal([1])),
                _result(anomaly_prob=0.55, conformal=_conformal([0, 1])),
            ]
        )
        summary = loop.summary()
        assert summary == loop.ledger.summary()
        assert summary["total"] == 2
        assert summary["abstention_rate"] == pytest.approx(0.5)
