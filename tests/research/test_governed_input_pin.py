# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The governed-fusion input pin: hashes that are checked, not merely recorded.

``build_manifest.py`` has always written a SHA-256 of every event's ``(X, y)``,
but nothing read one back. That is why the suite's per-event drift has stood
unexplained: a 2026-08-04 refit moved ``nsl_kdd 0.679 -> 0.728`` and
``batadal 0.862 -> 0.889`` (headline 0.770 -> 0.809), and the repository had no
way to say whether the *inputs* or the *environment* had moved. Both hypotheses
fit, so neither could be ruled out, so the number could not be improved --
any gain smaller than the drift is unfalsifiable.

These tests pin the two properties that make the drift attributable:

* the digest is **canonical** (dtype/layout-insensitive, shape-sensitive), so a
  match means the data really is the same and a mismatch really is a change; and
* the promotion gate **refuses** to compare across a changed event set, rather
  than reading a difference in data as a difference in capability.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from research.governed_fusion.input_pin import (
    DEFAULT_MANIFEST,
    EventPin,
    PinReport,
    PinStatus,
    external_label_count,
    external_label_keys,
    load_pins,
    sha256_xy,
    verify_pinned_results,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestDigestIsCanonical:
    """A pin is only meaningful if identical data always hashes identically."""

    def test_dtype_and_memory_layout_do_not_change_the_digest(self) -> None:
        X = np.arange(12, dtype=np.float64).reshape(4, 3)
        y = np.array([0, 1, 0, 1])
        assert sha256_xy(X, y) == sha256_xy(np.asfortranarray(X.astype(np.float32)), y)
        assert sha256_xy(X, y) == sha256_xy(X.copy(order="F"), y.astype(np.int32))

    def test_label_dtype_and_shape_are_normalized(self) -> None:
        X = np.zeros((3, 2))
        assert sha256_xy(X, np.array([1, 0, 1])) == sha256_xy(X, np.array([[1], [0], [1]]))

    def test_a_changed_value_changes_the_digest(self) -> None:
        X = np.zeros((4, 3))
        y = np.array([0, 1, 0, 1])
        mutated = X.copy()
        mutated[2, 1] = 1e-9
        assert sha256_xy(X, y) != sha256_xy(mutated, y)

    def test_a_changed_label_changes_the_digest(self) -> None:
        X = np.zeros((4, 3))
        assert sha256_xy(X, np.array([0, 1, 0, 1])) != sha256_xy(X, np.array([0, 1, 1, 1]))

    def test_reshape_cannot_collide(self) -> None:
        """Shape is folded in, so the same bytes in a different shape differ."""
        flat = np.arange(12, dtype=np.float64)
        y = np.zeros(4, dtype=int)
        assert sha256_xy(flat.reshape(4, 3), y) != sha256_xy(flat.reshape(3, 4), np.zeros(3, int))

    def test_row_order_is_significant(self) -> None:
        """A reordered fetch is a different input, and must not read as a match.

        The detector's own data-type detection branches on adjacent-row
        coherence, so row order materially changes the score -- a pin that
        ignored order would call two genuinely different measurements equal.
        """
        X = np.arange(12, dtype=np.float64).reshape(4, 3)
        y = np.array([0, 1, 0, 1])
        order = [1, 0, 3, 2]
        assert sha256_xy(X, y) != sha256_xy(X[order], y[order])

    def test_writer_and_checker_share_one_implementation(self) -> None:
        """Two copies of the digest could disagree about "the same data"."""
        from research.governed_fusion import build_manifest

        assert build_manifest._sha256_xy is sha256_xy


class TestManifestPins:
    def test_committed_manifest_pins_every_event(self) -> None:
        pins = load_pins()
        assert pins, "manifest carries no events"
        for key, entry in pins.items():
            assert entry.get("sha256_xy"), f"{key} has no pinned digest"
            assert len(str(entry["sha256_xy"])) == 64, key

    def test_external_label_events_are_identified(self) -> None:
        """These two are the only events whose metric is claimed as skill."""
        keys = external_label_keys()
        assert keys, "no external-label events pinned"
        with DEFAULT_MANIFEST.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert external_label_count(manifest) == len(keys)

    def test_external_label_substrate_is_still_only_two_events(self) -> None:
        """A guard on the headline's own fragility, not a target to satisfy.

        The externally-defensible AUROC is the mean of this many events. At two,
        its environment-induced swing (measured 0.770 -> 0.809 on identical code)
        exceeds any plausible single-change improvement, which is why expanding
        this set is the prerequisite for optimizing the number rather than a
        nice-to-have. When the substrate grows, update this and say so.
        """
        assert len(external_label_keys()) == 2


class TestGateRefusesToCompareAcrossChangedInputs:
    """The load-bearing property: a data change must not read as capability."""

    @staticmethod
    def _manifest(n_external: int = 2) -> dict[str, Any]:
        """A manifest shaped like the real one: entries AND the rollup."""
        return {
            "provenance_summary": {
                "transparent_fitness_bucket": "external_label",
                "real": {"external_label": {"n_events": n_external, "n_rows": 100, "n_pos": 20}},
            },
            "real": [
                {
                    "domain": "network_security",
                    "event_id": f"ev{i}",
                    "external_label": True,
                    "sha256_xy": "0" * 64,
                }
                for i in range(n_external)
            ],
            "reconstructed": [],
        }

    def test_matching_event_count_raises_no_objection(self) -> None:
        assert verify_pinned_results({"external_label_events": 2}, manifest=self._manifest()) == []

    def test_mismatched_event_count_is_refused(self) -> None:
        reasons = verify_pinned_results({"external_label_events": 5}, manifest=self._manifest(2))
        assert reasons and "not comparable" in reasons[0]

    def test_undeclared_event_count_is_not_invented(self) -> None:
        """Absence of a declaration is the gate's required-field problem."""
        assert verify_pinned_results({}, manifest=self._manifest()) == []

    def test_non_integer_declaration_is_refused(self) -> None:
        reasons = verify_pinned_results({"external_label_events": "two"}, manifest=self._manifest())
        assert reasons and "not an integer" in reasons[0]

    def test_promotion_gate_rejects_a_candidate_measured_on_a_different_set(self) -> None:
        """End-to-end: a strong-looking candidate must still be refused.

        The record is otherwise promotable -- it improves AUROC and F1 on the
        fitness bucket and clears every safety threshold. Only its event set
        differs, which before this check the gate could not see.
        """
        from research.governed_fusion.promotion_gate import GateDecision, evaluate_candidate
        from tests.research.test_governed_promotion_gate import (
            _candidate_record,
            _ledger_with_ok_baseline,
        )

        clean = _candidate_record()
        assert (
            evaluate_candidate(
                clean, manifest=self._manifest(2), ledger=_ledger_with_ok_baseline()
            ).decision
            == GateDecision.PROMOTE.value
        ), "fixture must be promotable, or this test proves nothing"

        record = _candidate_record()
        record["external_label_events"] = 7  # measured over a different set
        result = evaluate_candidate(
            record,
            manifest=self._manifest(2),
            ledger=_ledger_with_ok_baseline(),
        )
        assert result.decision != GateDecision.PROMOTE.value
        assert any("not comparable" in r for r in result.reasons), result.reasons

    def test_a_manifest_that_disagrees_with_itself_is_refused(self) -> None:
        """The rollup the gate reads and the entries the pin reads must agree."""
        manifest = self._manifest(2)
        manifest["provenance_summary"]["real"]["external_label"]["n_events"] = 3
        reasons = verify_pinned_results({"external_label_events": 2}, manifest=manifest)
        assert reasons and "disagrees with itself" in reasons[0]


class TestReportSemantics:
    def test_drift_fails_but_an_unreachable_upstream_does_not(self) -> None:
        """An upstream being down is availability, not evidence of change."""
        reachable = PinReport("real", [EventPin("d", "e", PinStatus.MATCH)])
        unreachable = PinReport("real", [EventPin("d", "e", PinStatus.UNREACHABLE)])
        drifted = PinReport("real", [EventPin("d", "e", PinStatus.DRIFT, "a" * 64, "b" * 64)])
        unpinned = PinReport("real", [EventPin("d", "e", PinStatus.UNPINNED)])

        assert reachable.ok
        assert unreachable.ok, "a down upstream must not be reported as a drift"
        assert not drifted.ok
        assert not unpinned.ok, "an unpinned event is an unverifiable measurement"

    def test_report_is_json_safe(self) -> None:
        report = PinReport("real", [EventPin("d", "e", PinStatus.DRIFT, "a" * 64, "b" * 64)])
        payload = json.dumps(report.to_dict())
        assert "drift" in payload
        assert report.to_dict()["ok"] is False


def test_cli_check_exits_nonzero_only_on_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--check` is what a CI lane would run; it must bite on drift and only then."""
    from research.governed_fusion import input_pin

    def _fake(kind: str, **_kw: object) -> PinReport:
        return PinReport(kind, [EventPin("d", "e", PinStatus.DRIFT, "a" * 64, "b" * 64)])

    monkeypatch.setattr(input_pin, "verify_suite", _fake)
    assert input_pin.main(["--check"]) == 1
    assert input_pin.main([]) == 0  # reports without failing when not asked to gate

    def _clean(kind: str, **_kw: object) -> PinReport:
        return PinReport(kind, [EventPin("d", "e", PinStatus.MATCH)])

    monkeypatch.setattr(input_pin, "verify_suite", _clean)
    assert input_pin.main(["--check"]) == 0


def test_verify_suite_reports_unreachable_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Building an event hits the network; that must degrade, not explode."""
    from research.governed_fusion import input_pin

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"real": [], "reconstructed": []}), encoding="utf-8")

    import research.governed_fusion.suite as suite_mod

    def _boom(**_kw: object) -> list[object]:
        raise RuntimeError("upstream down")

    monkeypatch.setattr(suite_mod, "build_suite", _boom)
    report = input_pin.verify_suite("real", manifest_path=manifest)
    assert report.by_status(PinStatus.UNREACHABLE)
    assert report.ok, "an unreachable suite is not a drift"
