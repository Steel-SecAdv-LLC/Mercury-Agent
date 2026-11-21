# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ledger summary reports decision churn across consecutive records.

A trail that flips disposition on every event is operationally unstable even
when each record is individually sound, and the summary never said so. It now
carries ``churn``: consecutive-record disposition flips over ``n - 1`` pairs
(``None`` with fewer than two records), maintained incrementally so the O(1)
summary stays O(1), ring-buffer eviction included.
"""

from __future__ import annotations

from typing import Any

from omni_mercury_engine.decision import (
    DecisionAbstentionResponder,
    DecisionLedger,
    DecisionRecord,
)


def _conformal(labels: list[int], coverage: float = 0.9) -> dict[str, Any]:
    return {
        "prediction_set": labels,
        "set_size": len(labels),
        "abstain": len(labels) == 2,
        "coverage": coverage,
    }


def _act() -> DecisionRecord:
    return DecisionAbstentionResponder().decide(
        {
            "anomaly_prob": 0.95,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "severity": 0.3,
            "conformal": _conformal([1]),
        }
    )


def _clear() -> DecisionRecord:
    return DecisionAbstentionResponder().decide(
        {
            "anomaly_prob": 0.02,
            "is_anomaly": False,
            "threshold_used": 0.5,
            "conformal": _conformal([0]),
        }
    )


class TestLedgerChurn:
    def test_alternating_dispositions_churn_one(self) -> None:
        ledger = DecisionLedger()
        for record in (_act(), _clear(), _act(), _clear()):
            ledger.record(record)
        assert ledger.summary()["churn"] == 1.0

    def test_constant_dispositions_churn_zero(self) -> None:
        ledger = DecisionLedger()
        for record in (_act(), _act(), _act()):
            ledger.record(record)
        assert ledger.summary()["churn"] == 0.0

    def test_fewer_than_two_records_is_none(self) -> None:
        ledger = DecisionLedger()
        assert ledger.summary()["churn"] is None
        ledger.record(_act())
        assert ledger.summary()["churn"] is None

    def test_mixed_sequence_counts_each_flip_once(self) -> None:
        ledger = DecisionLedger()
        for record in (_act(), _act(), _clear(), _clear(), _act()):
            ledger.record(record)
        assert ledger.summary()["churn"] == 2 / 4

    def test_eviction_keeps_churn_about_the_retained_records(self) -> None:
        ledger = DecisionLedger(maxlen=3)
        for record in (_act(), _act(), _clear(), _clear()):
            ledger.record(record)
        # Retained: ACT, CLEAR, CLEAR -> one flip over two pairs.
        assert ledger.summary()["churn"] == 0.5
        ledger.record(_clear())
        # Retained: CLEAR, CLEAR, CLEAR -> no flips.
        assert ledger.summary()["churn"] == 0.0

    def test_maxlen_one_never_has_a_pair(self) -> None:
        ledger = DecisionLedger(maxlen=1)
        for record in (_act(), _clear(), _act()):
            ledger.record(record)
        assert ledger.summary()["churn"] is None

    def test_clear_resets_churn(self) -> None:
        ledger = DecisionLedger()
        for record in (_act(), _clear()):
            ledger.record(record)
        assert ledger.summary()["churn"] == 1.0
        ledger.clear()
        assert ledger.summary()["churn"] is None

    def test_churn_survives_the_json_round_trip(self) -> None:
        ledger = DecisionLedger()
        for record in (_act(), _clear(), _act()):
            ledger.record(record)
        reloaded = DecisionLedger.from_json(ledger.to_json())
        assert reloaded.summary()["churn"] == 1.0
