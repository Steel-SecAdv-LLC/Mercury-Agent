# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The audit ledger is an append-only, bounded, summarisable trail."""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, Any

import pytest

from omni_mercury_engine.decision import DecisionAbstentionResponder, DecisionLedger

if TYPE_CHECKING:
    from pathlib import Path


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


def _fresh_summary(entries: list[Any]) -> dict[str, Any]:
    """An independent, from-scratch recount -- the oracle for the incremental one."""
    by_state: dict[str, int] = {}
    by_disp: dict[str, int] = {}
    by_action: dict[str, int] = {}
    abstained = calibrated = 0
    for r in entries:
        by_state[r.state.value] = by_state.get(r.state.value, 0) + 1
        by_disp[r.disposition.value] = by_disp.get(r.disposition.value, 0) + 1
        by_action[r.response.action.value] = by_action.get(r.response.action.value, 0) + 1
        abstained += r.abstained
        calibrated += r.calibrated
    total = len(entries)
    return {
        "total": total,
        "by_state": by_state,
        "by_disposition": by_disp,
        "by_response_action": by_action,
        "abstention_rate": (abstained / total) if total else 0.0,
        "calibrated_rate": (calibrated / total) if total else 0.0,
    }


class TestIncrementalSummary:
    """The O(1) incremental summary equals a from-scratch recount, always."""

    def test_incremental_matches_recount_under_eviction(self, mixed_records: list[Any]) -> None:
        ledger = DecisionLedger(maxlen=3)
        for rec in mixed_records * 4:
            ledger.record(rec)
            assert ledger.summary() == _fresh_summary(list(ledger.entries))

    def test_eviction_drops_a_zeroed_category(self) -> None:
        # maxlen=1: a grounded record then a deferring one fully evicts 'grounded',
        # so it must disappear from the summary (not linger as a zero count).
        ledger = DecisionLedger(maxlen=1)
        ledger.record(_rec(anomaly_prob=0.95, is_anomaly=True, conformal=_conformal([1])))
        assert ledger.summary()["by_state"] == {"grounded": 1}
        ledger.record(_rec(anomaly_prob=0.55, conformal=_conformal([0, 1])))
        s = ledger.summary()
        assert s["by_state"] == {"unavailable": 1}  # 'grounded' is gone, not 0
        assert "grounded" not in s["by_state"]
        assert s["calibrated_rate"] == 1.0

    def test_clear_resets_aggregates(self, mixed_records: list[Any]) -> None:
        ledger = DecisionLedger()
        ledger.extend(mixed_records)
        ledger.clear()
        assert ledger.summary() == _fresh_summary([])


class TestPersistence:
    """The trail round-trips through JSON text and disk (a reloadable audit log)."""

    def test_to_json_from_json_round_trip(self, mixed_records: list[Any]) -> None:
        ledger = DecisionLedger()
        ledger.extend(mixed_records)
        restored = DecisionLedger.from_json(ledger.to_json())
        assert restored.to_list() == ledger.to_list()
        assert restored.summary() == ledger.summary()

    def test_save_load_round_trip(self, mixed_records: list[Any], tmp_path: Path) -> None:
        ledger = DecisionLedger()
        ledger.extend(mixed_records)
        path = tmp_path / "trail.json"
        ledger.save(path)
        assert path.exists()
        restored = DecisionLedger.load(path)
        assert restored.to_list() == ledger.to_list()

    def test_from_records_preserves_order_and_counts(self, mixed_records: list[Any]) -> None:
        ledger = DecisionLedger.from_records(mixed_records)
        assert list(ledger) == mixed_records
        assert ledger.summary() == _fresh_summary(mixed_records)

    def test_from_list_rebuilds_via_record_round_trip(self, mixed_records: list[Any]) -> None:
        src = DecisionLedger.from_records(mixed_records)
        rebuilt = DecisionLedger.from_list(src.to_list())
        assert rebuilt.to_list() == src.to_list()

    def test_load_respects_maxlen(self, mixed_records: list[Any], tmp_path: Path) -> None:
        DecisionLedger.from_records(mixed_records).save(tmp_path / "t.json")
        bounded = DecisionLedger.load(tmp_path / "t.json", maxlen=2)
        assert len(bounded) == 2  # only the most recent two survive the ring buffer


class TestThreadSafety:
    """Concurrent recording loses no count -- the aggregates stay consistent."""

    def test_concurrent_record_is_lossless(self) -> None:
        ledger = DecisionLedger()
        rec = _rec(anomaly_prob=0.95, is_anomaly=True, conformal=_conformal([1]))
        per_thread, n_threads = 1000, 8

        def worker() -> None:
            for _ in range(per_thread):
                ledger.record(rec)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = per_thread * n_threads
        assert len(ledger) == expected
        assert ledger.summary()["total"] == expected
        assert ledger.summary()["by_state"]["grounded"] == expected
