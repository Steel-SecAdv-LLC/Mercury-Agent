# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The audit ledger is an append-only, bounded, summarisable trail."""

from __future__ import annotations

import json
from typing import Any

import pytest

from omni_mercury_engine.decision import DecisionAbstentionResponder, DecisionLedger


def _conformal(labels: list[int], coverage: float = 0.9) -> dict[str, Any]:
    return {
        "prediction_set": labels,
        "set_size": len(labels),
        "abstain": len(labels) == 2,
        "coverage": coverage,
    }


def _rec(**over: Any):
    base: dict[str, Any] = {"anomaly_prob": 0.5, "is_anomaly": False, "threshold_used": 0.5}
    base.update(over)
    return DecisionAbstentionResponder().decide(base, domain="security")


@pytest.fixture
def mixed_records() -> list[Any]:
    return [
        _rec(
            anomaly_prob=0.95, is_anomaly=True, severity=0.9, conformal=_conformal([1])
        ),  # grounded/act
        _rec(anomaly_prob=0.02, conformal=_conformal([0])),  # grounded/clear
        _rec(anomaly_prob=0.55, conformal=_conformal([0, 1])),  # unavailable/defer
        _rec(anomaly_prob=0.40, conformal=_conformal([])),  # undecidable/hold
        _rec(anomaly_prob=0.52, is_anomaly=True),  # unavailable/defer, uncalibrated
    ]


class TestRecording:
    def test_record_appends_and_returns(self) -> None:
        ledger = DecisionLedger()
        rec = _rec(anomaly_prob=0.9, conformal=_conformal([1]))
        assert ledger.record(rec) is rec
        assert len(ledger) == 1
        assert ledger.entries == (rec,)

    def test_entries_is_immutable_snapshot(self) -> None:
        ledger = DecisionLedger()
        ledger.record(_rec(anomaly_prob=0.9))
        snap = ledger.entries
        ledger.record(_rec(anomaly_prob=0.1))
        assert len(snap) == 1 and len(ledger) == 2  # snapshot did not mutate

    def test_extend_and_iter(self, mixed_records: list[Any]) -> None:
        ledger = DecisionLedger()
        ledger.extend(mixed_records)
        assert len(ledger) == len(mixed_records)
        assert list(ledger) == mixed_records

    def test_clear(self, mixed_records: list[Any]) -> None:
        ledger = DecisionLedger()
        ledger.extend(mixed_records)
        ledger.clear()
        assert len(ledger) == 0
        assert ledger.summary()["total"] == 0


class TestBounded:
    def test_maxlen_evicts_oldest(self) -> None:
        ledger = DecisionLedger(maxlen=2)
        a, b, c = _rec(anomaly_prob=0.1), _rec(anomaly_prob=0.2), _rec(anomaly_prob=0.3)
        ledger.extend([a, b, c])
        assert len(ledger) == 2
        assert ledger.entries == (b, c)  # 'a' was evicted

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_maxlen_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="maxlen"):
            DecisionLedger(maxlen=bad)


class TestSerialisationAndSummary:
    def test_to_list_is_json_safe(self, mixed_records: list[Any]) -> None:
        ledger = DecisionLedger()
        ledger.extend(mixed_records)
        as_list = ledger.to_list()
        assert len(as_list) == len(mixed_records)
        json.dumps(as_list)  # must not raise
        assert as_list[0]["state"] == "grounded"

    def test_summary_counts_and_rates(self, mixed_records: list[Any]) -> None:
        ledger = DecisionLedger()
        ledger.extend(mixed_records)
        s = ledger.summary()
        assert s["total"] == 5
        assert s["by_state"] == {"grounded": 2, "unavailable": 2, "undecidable": 1}
        assert s["by_disposition"]["act"] == 1
        assert s["by_disposition"]["clear"] == 1
        assert s["by_disposition"]["defer"] == 2
        assert s["by_disposition"]["hold"] == 1
        # 3 of 5 abstained (defer x2 + hold x1); 4 of 5 carried a conformal cert.
        assert s["abstention_rate"] == pytest.approx(3 / 5)
        assert s["calibrated_rate"] == pytest.approx(4 / 5)
        assert set(s["by_response_action"]).issubset(
            {
                "monitor",
                "alert",
                "recommend_mitigation",
                "escalate_to_human",
                "request_input",
                "hold",
            }
        )

    def test_empty_summary_has_zero_rates(self) -> None:
        s = DecisionLedger().summary()
        assert s["total"] == 0
        assert s["abstention_rate"] == 0.0
        assert s["calibrated_rate"] == 0.0
        assert s["by_state"] == {}
