# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The per-hazard record is said as it was measured, and the measuring does what it says.

``research/hazard_calibration/measure.py`` judges each real event of a hazard
by a calibration and a certificate fitted on the hazard's other events. Its
evidence is fetched data's and cannot be re-derived here, so these tests hold
three things instead: the document and the shipped copy say what the evidence
says; the evidence is consistent with itself; and the measuring, run on
scores whose answer is known, gives that answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research.hazard_calibration import measure

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE: dict[str, Any] = json.loads(measure.EVIDENCE.read_text(encoding="utf-8"))
DOC = (_REPO_ROOT / "docs" / "HAZARD_CALIBRATION.md").read_text(encoding="utf-8")
MEASURED = {
    hazard: told for hazard, told in EVIDENCE["hazards"].items() if "not_measured" not in told
}


def test_the_package_ships_the_evidence_as_recorded() -> None:
    assert measure.SHIPPED.read_bytes() == measure.EVIDENCE.read_bytes()


def test_the_document_prints_the_evidences_table() -> None:
    assert measure.table(EVIDENCE) in DOC
    assert f"Measured {EVIDENCE['measured_on']}" in DOC


def test_what_the_document_says_plainly_is_made_from_the_evidence() -> None:
    plain = " ".join(DOC.split())
    lines = measure.summary(EVIDENCE)

    assert len(lines) == 6  # three questions, and the three hazards with independent labels
    for line in lines:
        assert " ".join(line.split()) in plain, line


def test_a_certificate_is_said_to_hold_only_where_both_halves_agree() -> None:
    verdicts = {hazard: measure.verdict(told) for hazard, told in MEASURED.items()}
    target = EVIDENCE["coverage_target"]

    for hazard, told in MEASURED.items():
        over = [split["coverage"] >= target for split in told["by_split"]]
        assert len(over) == 2
        assert verdicts[hazard] == ("held" if all(over) else "near" if any(over) else "fell short")
    assert sorted(h for h, v in verdicts.items() if v == "fell short") == ["network_security"]
    assert sum(v == "held" for v in verdicts.values()) == 10
    assert not [h for h, v in verdicts.items() if v == "near"]


def test_halves_that_disagree_are_near_and_neither_held_nor_short() -> None:
    """One judging alone would have said "held" of a hazard the other says fell short."""

    def told(first: float, second: float) -> dict[str, Any]:
        return {
            "certificate": {"target": 0.9},
            "by_split": [{"coverage": first}, {"coverage": second}],
        }

    assert measure.verdict(told(0.913, 0.860)) == "near"  # energy, as recorded 2026-11-18
    assert measure.verdict(told(0.906, 0.940)) == "held"
    assert measure.verdict(told(0.729, 0.720)) == "fell short"


_COUNT: tuple[str, ...] = (
    *("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
    *("eleven", "twelve", "thirteen"),
)


def test_the_changelog_gives_the_same_account() -> None:
    """The release's changelog entry quotes the evidence as recorded, not an earlier record."""
    changelog = " ".join((_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split())
    verdicts = [measure.verdict(told) for told in MEASURED.values()]
    held, short = verdicts.count("held"), verdicts.count("fell short")
    more = [h for h, told in MEASURED.items() if told["probability"]["skill"] > 0.05]
    abstained = sorted(told["decision"]["abstained"] for told in MEASURED.values())

    assert (
        f"held for {_COUNT[held]} of the {_COUNT[len(MEASURED)]} hazards that could be measured "
        f"and fell short for {_COUNT[short]}" in changelog
    )
    attacks, sepsis = MEASURED["network_security"], MEASURED["sepsis"]
    assert f"network security {attacks['certificate']['coverage']:.1%}" in changelog
    assert f"says more than the base rate for {_COUNT[len(more)]}" in changelog
    called_normal = attacks["decision"]["anomalies_called_normal"]
    assert f"{called_normal:.1%} of true attacks called normal" in changelog
    assert f"{sepsis['decision']['anomaly_call_precision']:.1%} of anomaly calls right" in changelog
    assert f"from {abstained[0]:.1%} of rows" in changelog
    assert f"to {abstained[-1]:.1%}" in changelog
    assert f"measured {EVIDENCE['measured_on']}" in changelog


def test_the_readme_gives_the_same_account() -> None:
    """The README's sentence is made of the evidence's counts, spelled out."""
    readme = " ".join((_REPO_ROOT / "README.md").read_text(encoding="utf-8").split())
    verdicts = [measure.verdict(told) for told in MEASURED.values()]
    more = [h for h, told in MEASURED.items() if told["probability"]["skill"] > 0.05]
    independent = [
        h for h, told in MEASURED.items() if told["label_source"] in measure.INDEPENDENT_LABELS
    ]

    assert (len(MEASURED), verdicts.count("held"), verdicts.count("fell short")) == (11, 10, 1)
    assert (
        f"held for {_COUNT[verdicts.count('held')]} of the {_COUNT[len(MEASURED)]} hazards that "
        f"could be measured and fell short for {_COUNT[verdicts.count('fell short')]}" in readme
    )
    assert len(more) == 6
    assert f"says more than the base rate for {_COUNT[len(more)]}" in readme
    assert independent == ["fema_activity", "network_security", "sepsis"]
    assert f"the {_COUNT[len(independent)]} hazards whose labels are independent" in readme
    attacks = MEASURED["network_security"]
    assert f"network security {attacks['certificate']['coverage']:.1%}" in readme
    assert f"measured {EVIDENCE['measured_on']}" in readme


def test_the_hazards_it_could_not_measure_are_named() -> None:
    plain = " ".join(DOC.split())

    assert EVIDENCE["not_fetched"] == ["financial", "volcanic", "wildfire"]
    assert "Financial, volcanic and wildfire could not be fetched" in plain
    one_event = sorted(h for h, told in EVIDENCE["hazards"].items() if "not_measured" in told)
    assert one_event == ["flood", "landslide"]
    assert "Landslide and flood have one event" in plain


def test_each_hazards_labels_are_what_its_loader_declares() -> None:
    for hazard, told in EVIDENCE["hazards"].items():
        assert told["label_source"] == measure.label_source(hazard), hazard
    from omni_mercury_engine.datasets.metadata import GENUINE_LABEL_SOURCES

    assert set(measure.INDEPENDENT_LABELS) == set(GENUINE_LABEL_SOURCES)
    independent = sorted(
        h
        for h, told in EVIDENCE["hazards"].items()
        if told["label_source"] in measure.INDEPENDENT_LABELS
    )
    assert independent == ["fema_activity", "network_security", "sepsis"]
    assert EVIDENCE["hazards"]["fema_activity"]["label_source"] == "expert_annotated"


def test_the_evidence_adds_up() -> None:
    for hazard, told in MEASURED.items():
        decision = told["decision"]
        assert decision["decided"] + decision["abstained"] == pytest.approx(1.0, abs=2e-4), hazard
        assert decision["anomalies_called"] + decision["anomalies_abstained"] + decision[
            "anomalies_called_normal"
        ] == pytest.approx(1.0, abs=3e-4), hazard
        scored = [
            min(event["rows"], EVIDENCE["row_limit"])
            for event in told["events"]
            if event["event"] not in told["folds_skipped"]
        ]
        assert told["rows"] == sum(scored), hazard
        halves = [split["coverage"] for split in told["by_split"]]
        assert min(halves) - 1e-4 <= told["certificate"]["coverage"] <= max(halves) + 1e-4
        assert told["anomalies"] == sum(
            event["anomalies_scored"]
            for event in told["events"]
            if event["event"] not in told["folds_skipped"]
        ), hazard
        assert told["certificate"]["target"] == EVIDENCE["coverage_target"] == measure.COVERAGE


# --------------------------------------------------------------------------- #
# The measuring, on scores whose answer is known
# --------------------------------------------------------------------------- #


def _event(scores: list[float], labels: list[int]) -> dict[str, Any]:
    return {"scores": np.array(scores, dtype=float), "labels": np.array(labels, dtype=int)}


def _separable(rows: int = 400, every: int = 10) -> dict[str, Any]:
    """Anomalies score high and normal rows low, with nothing in between."""
    labels = [int(n % every == 0) for n in range(rows)]
    return _event([0.9 if label else 0.1 for label in labels], labels)


def _unrelated(rows: int = 400, every: int = 10) -> dict[str, Any]:
    """The score cycles through its values whatever the label is."""
    labels = [int(n % every == 0) for n in range(rows)]
    return _event([(n * 7 % 11) / 10.0 for n in range(rows)], labels)


def test_one_event_is_not_measured() -> None:
    told = measure.leave_one_event_out({"only": _separable()})

    assert told == {"not_measured": "one event: there is no other to calibrate on"}


def test_a_score_that_separates_is_decided_and_decided_right() -> None:
    told = measure.leave_one_event_out({"a": _separable(), "b": _separable(), "c": _separable()})

    assert (told["rows"], told["anomalies"]) == (1200, 120)
    assert told["by_split"] == [{"coverage": 1.0, "abstained": 0.0}] * 2
    assert told["probability"]["brier"] == 0.0 and told["probability"]["skill"] == 1.0
    assert told["certificate"]["coverage"] == 1.0
    assert told["decision"]["abstained"] == 0.0 and told["decision"]["decided_accuracy"] == 1.0
    assert told["decision"]["anomaly_call_precision"] == 1.0
    assert told["decision"]["anomalies_called"] == 1.0
    assert told["score"]["brier"] == pytest.approx(0.01)  # 0.9 for a 1, 0.1 for a 0


def test_a_score_that_says_nothing_is_abstained_on_and_claims_no_skill() -> None:
    told = measure.leave_one_event_out({"a": _unrelated(), "b": _unrelated()})

    assert told["probability"]["skill"] <= 0.01
    assert told["decision"]["abstained"] >= 0.8
    assert told["decision"]["anomaly_calls"] == 0
    assert told["decision"]["anomaly_call_precision"] is None  # none was made: not 0%, not 100%
    assert told["certificate"]["coverage"] >= measure.COVERAGE - 0.05


def test_what_is_fitted_never_sees_the_event_it_is_judged_on() -> None:
    """An event whose scores mean the opposite of the others' is judged wrong, not fitted to."""
    backwards = _separable()
    backwards["scores"] = 1.0 - backwards["scores"]

    told = measure.leave_one_event_out({"a": _separable(), "b": _separable(), "z": backwards})

    # a and b are each judged by a map that the backwards event has half spoiled or
    # not at all; z is judged by a and b alone and gets every row wrong.
    assert told["decision"]["decided_accuracy"] < 0.9
    assert told["certificate"]["coverage"] < 0.9


def test_an_event_whose_others_hold_one_class_is_skipped_and_said_so() -> None:
    no_anomalies = _event([0.1] * 50, [0] * 50)
    no_anomalies["labels"][0] = 1  # one, so the archive would have been kept
    told = measure.leave_one_event_out({"a": _separable(), "b": no_anomalies})

    assert told["folds_skipped"] == ["a"]  # judged by b alone, whose second half is one class
    assert told["rows"] == 50
