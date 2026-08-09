# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pillar: control — every decision leaves an immutable, append-only record.

Control means an operator can answer "what did it decide, on what, and why?"
after the fact, and can trust the answer was not edited afterwards. Three
properties:

* **The record is immutable.** ``DecisionRecord`` is a frozen dataclass and its
  provenance mapping is a read-only view, so a record already in the ledger
  cannot be rewritten.
* **The ledger is append-only.** It exposes ``record``/``extend``/``clear`` and
  no update or delete-by-index; its ``entries`` view is a snapshot copy, so
  mutating what a reader got back does not reach the trail.
* **Every decision emits one.** With the decision layer enabled, a detection
  carries a record, and when a ledger is attached the record lands in it.

The freeze is *shallow* by design (``DecisionRecord.__post_init__`` documents
this): top-level provenance keys cannot be added, dropped or rebound, and the
values stay plain JSON-safe scalars so ``to_dict`` remains serialisable. This
module pins the guarantee that exists rather than asserting a deep freeze that
does not.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.ledger import DecisionLedger
from omni_mercury_engine.decision.record import DecisionRecord
from omni_mercury_engine.decision.states import Disposition

if TYPE_CHECKING:
    from pathlib import Path


def _record(**overrides: Any) -> DecisionRecord:
    detection: dict[str, Any] = {
        "anomaly_prob": 0.82,
        "is_anomaly": True,
        "threshold_used": 0.5,
        "severity": 0.6,
        "conformal": {"set_size": 1, "prediction_set": [1], "coverage": 0.9},
        "gosnn_metadata": {"ethical_gate_passed": True},
    }
    detection.update(overrides)
    return DecisionAbstentionResponder().decide(detection, domain="cyber")


class TestRecordIsImmutable:
    def test_the_dataclass_is_frozen(self) -> None:
        assert dataclasses.fields(DecisionRecord)  # sanity: it is a dataclass
        record = _record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.decision_label = 0  # type: ignore[misc]

    def test_every_scalar_field_rejects_assignment(self) -> None:
        record = _record()
        for field in dataclasses.fields(DecisionRecord):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(record, field.name, None)

    def test_provenance_keys_cannot_be_added_dropped_or_rebound(self) -> None:
        record = _record()
        with pytest.raises(TypeError):
            record.signals["injected"] = 1  # type: ignore[index]
        with pytest.raises(TypeError):
            del record.signals["policy"]  # type: ignore[attr-defined]

    def test_mutating_the_source_mapping_does_not_reach_the_record(self) -> None:
        """The record copies its provenance; the caller's dict is not shared."""
        signals = {"a": 1}
        record = DecisionRecord.from_dict({**_record().to_dict(), "signals": signals})
        signals["a"] = 999
        signals["b"] = 2
        assert record.signals["a"] == 1
        assert "b" not in record.signals

    def test_the_wire_form_is_a_copy_not_a_view(self) -> None:
        record = _record()
        payload = record.to_dict()
        payload["decision_label"] = 12345
        payload["signals"]["tampered"] = True
        assert record.decision_label != 12345
        assert "tampered" not in record.signals

    def test_the_record_is_json_serialisable(self) -> None:
        """An audit artifact that cannot be written down is not an audit trail."""
        json.dumps(_record().to_dict())


class TestLedgerIsAppendOnly:
    def test_no_update_or_delete_api_is_exposed(self) -> None:
        """``clear`` (reset the whole trail) is the only removal; no edit-in-place."""
        forbidden = {"update", "replace", "set", "pop", "remove", "delete", "__setitem__"}
        exposed = {name for name in dir(DecisionLedger) if not name.startswith("_")}
        assert not (exposed & forbidden), exposed & forbidden

    def test_entries_is_a_snapshot_that_cannot_be_written_through(self) -> None:
        ledger = DecisionLedger()
        ledger.record(_record())
        entries = ledger.entries
        assert isinstance(entries, tuple)
        with pytest.raises(TypeError):
            entries[0] = _record()  # type: ignore[index]

    def test_a_recorded_decision_cannot_be_edited_afterwards(self) -> None:
        ledger = DecisionLedger()
        ledger.record(_record())
        stored = ledger.entries[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            # Rebind to a *valid* Disposition rather than None: the property
            # under test is that a recorded decision cannot be edited at all,
            # and a well-typed assignment proves that without needing a
            # ``type: ignore`` to launder an argument the field never accepts.
            stored.disposition = Disposition.CLEAR  # type: ignore[misc]
        with pytest.raises(TypeError):
            stored.signals["forged"] = True  # type: ignore[index]

    def test_order_is_preserved_and_records_accumulate(self) -> None:
        ledger = DecisionLedger()
        probs = [0.10, 0.55, 0.82, 0.95]
        for prob in probs:
            ledger.record(_record(anomaly_prob=prob))
        assert len(ledger) == len(probs)
        assert [entry.anomaly_prob for entry in ledger.entries] == probs

    def test_the_ring_buffer_evicts_oldest_first_and_never_rewrites(self) -> None:
        ledger = DecisionLedger(maxlen=3)
        for prob in (0.1, 0.2, 0.3, 0.4, 0.5):
            ledger.record(_record(anomaly_prob=prob))
        assert [round(e.anomaly_prob, 1) for e in ledger.entries] == [0.3, 0.4, 0.5]
        assert ledger.summary()["total"] == 3

    def test_a_non_positive_maxlen_is_refused(self) -> None:
        with pytest.raises(ValueError, match="maxlen"):
            DecisionLedger(maxlen=0)

    def test_the_summary_matches_a_full_recount(self) -> None:
        """O(1) aggregates must agree with the trail they claim to summarise."""
        ledger = DecisionLedger()
        for prob in (0.05, 0.51, 0.82, 0.95, 0.30):
            ledger.record(_record(anomaly_prob=prob))
        summary = ledger.summary()
        recount: dict[str, int] = {}
        for entry in ledger.entries:
            recount[entry.state.value] = recount.get(entry.state.value, 0) + 1
        assert summary["by_state"] == recount
        assert summary["total"] == len(ledger)

    def test_persistence_round_trips_the_trail(self, tmp_path: Path) -> None:
        ledger = DecisionLedger()
        for prob in (0.1, 0.9):
            ledger.record(_record(anomaly_prob=prob))
        path = tmp_path / "ledger.json"
        ledger.save(path)
        assert DecisionLedger.load(path).to_list() == ledger.to_list()


@pytest.mark.slow
class TestEveryDecisionEmitsARecord:
    @staticmethod
    def _engine() -> Any:
        from omni_mercury_engine.engine import OmniMercuryEngine

        return OmniMercuryEngine(mode="fusion", require_explicit_fit=False, device="cpu")

    def test_detection_carries_a_decision_record_when_the_layer_is_enabled(self) -> None:
        engine = self._engine()
        engine.enable_decision_layer()
        data = np.random.default_rng(0).standard_normal((16, 6))
        result = engine.detect_with_fusion(data, domain="cyber")
        assert "decision" in result
        # The emitted payload is the record's own wire form, so the audit trail
        # and the API response cannot drift apart.
        assert DecisionRecord.from_dict(result["decision"]).to_dict() == result["decision"]

    def test_the_attached_ledger_receives_the_record(self) -> None:
        engine = self._engine()
        ledger = DecisionLedger()
        engine.enable_decision_layer(ledger=ledger)
        data = np.random.default_rng(1).standard_normal((16, 6))
        engine.detect_with_fusion(data, domain="cyber")
        assert len(ledger) == 1
        assert ledger.entries[0].domain == "cyber"

    def test_the_layer_is_on_by_default(self) -> None:
        """The gap is closed: a plain engine already emits a record.

        This used to assert the opposite — that ``enable_decision_layer()`` was
        opt-in and a deployment which never called it produced detections with
        no ``decision`` key. That made the abstention gate and the audit record
        opt-in for precisely the callers least likely to know they existed:
        every first-party entry point enabled it, and every library embedder
        silently did not. The layer is additive and non-destructive, so there
        was no safety reason for it to be conditional.
        """
        engine = self._engine()
        data = np.random.default_rng(2).standard_normal((16, 6))
        result = engine.detect_with_fusion(data, domain="cyber")
        assert "decision" in result
        assert result["decision"]["domain"] == "cyber"

    def test_an_uncalibrated_engine_abstains_rather_than_inventing_confidence(
        self,
    ) -> None:
        """Default-on must not mean default-confident.

        Without a conformal certificate the decider has no coverage guarantee,
        so the honest output is an explicit abstention. If this ever returns a
        grounded label on an uncalibrated engine, defaulting the layer on would
        be manufacturing verdicts rather than recording them.
        """
        engine = self._engine()
        data = np.random.default_rng(3).standard_normal((16, 6))
        decision = engine.detect_with_fusion(data, domain="cyber")["decision"]
        assert decision["state"] in {"unavailable", "undecidable"}
        assert decision["disposition"] in {"defer", "hold"}

    def test_the_layer_can_still_be_opted_out_of(self) -> None:
        """``decision_layer=False`` returns the bare detection dict."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(
            mode="fusion", require_explicit_fit=False, device="cpu", decision_layer=False
        )
        assert engine.decision_layer is None
        data = np.random.default_rng(4).standard_normal((16, 6))
        assert "decision" not in engine.detect_with_fusion(data, domain="cyber")


class TestEveryFirstPartyEntryPointEnablesTheLayer:
    """The opt-in above is only acceptable if every shipped entry point opts in."""

    ENTRY_POINTS: tuple[str, ...] = (
        "src/omni_mercury_engine/cli.py",
        "src/omni_mercury_engine/mcp_server.py",
        "src/omni_mercury_engine/api/routes/detection.py",
    )

    @pytest.mark.parametrize("relative_path", ENTRY_POINTS)
    def test_entry_point_enables_the_decision_layer(self, relative_path: str) -> None:
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[2]
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "enable_decision_layer(" in source, relative_path
