# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every assessment entry point passes the decision gate; none can run ungated.

The gate is replaced by one that always refuses. Each service function that
accepts caller content must then refuse before doing any work: a function that
skipped the gate would return a result here and fail the test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")  # the decision gate needs the ML stack

from omni_mercury_engine.assessment import service
from omni_mercury_engine.cognitive import decision_gate
from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

ENV = {"MERCURY_HUMAN_SUBJECT_CAPABILITIES": "consent,vital_interests"}
PURPOSE = "University lab experiment with consenting volunteers, IRB protocol 2026-114"
CIT = {
    "subject_pseudonym": "subj-01",
    "administered_at": 10.0,
    "purpose": PURPOSE,
    "consent": {
        "subject_pseudonym": "subj-01",
        "obtained_at": 0.0,
        "scope": ["concealed_information_test"],
        "obtained_by": "examiner-7",
    },
    "questions": [
        {
            "question_id": "q1",
            "probe": "p",
            "irrelevants": ["a", "b"],
            "responses": {"skin_conductance": [[2.0, 1.0, 1.0]]},
        }
    ],
}
VITALS = {
    "purpose": PURPOSE,
    "lawful_basis": "consent",
    "fs": 20.0,
    "recorded_at": 5.0,
    "subject_pseudonym": "subj-02",
    "consent": {
        "subject_pseudonym": "subj-02",
        "obtained_at": 0.0,
        "scope": ["contactless_vital_signs"],
        "obtained_by": "nurse-3",
    },
    "radar_displacement_m": [0.0] * 1200,
}
REPORTS = {"intel_reports": {"open_source": {"threat_score": 0.5, "text": "convoy"}}}

ENTRY_POINTS: list[tuple[str, Callable[[], Any]]] = [
    (
        "verify_statements",
        lambda: service.verify_statements(
            {"statements": [{"statement_id": "s", "speaker": "w", "made_at": 1}]}, surface="t"
        ),
    ),
    (
        "concealed_information_test",
        lambda: service.concealed_information_test(CIT, surface="t", environ=ENV),
    ),
    ("vital_signs", lambda: service.vital_signs(VITALS, surface="t", environ=ENV)),
    ("screen_intel_reports", lambda: service.screen_intel_reports(REPORTS, surface="t")),
    ("fuse_intelligence", lambda: service.fuse_intelligence(REPORTS, surface="t")),
    (
        "adjudication_add_case",
        lambda: service.adjudication_add_case(
            {
                "case_id": "c",
                "intel_reports": {"open_source": {"threat_score": 0.5}},
                "report_source_ids": ["Reuters"],
                "latest_report_at": "2026-01-01T00:00:00+00:00",
            },
            surface="t",
        ),
    ),
    (
        "adjudication_add_outcome",
        lambda: service.adjudication_add_outcome(
            {
                "case_id": "c",
                "observed_at": "2026-01-02T00:00:00+00:00",
                "description": "closed",
                "outcome_source_ids": ["UN"],
            },
            surface="t",
        ),
    ),
    (
        "adjudication_add_rating",
        lambda: service.adjudication_add_rating(
            {"case_id": "c", "level": 2, "rationale": "r"}, rater_id="alice", surface="t"
        ),
    ),
    (
        "adjudication_add_decision",
        lambda: service.adjudication_add_decision(
            {"case_id": "c", "level": 2, "rationale": "r"}, adjudicator_id="dana", surface="t"
        ),
    ),
]


@pytest.fixture(autouse=True)
def _refusing_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(tmp_path / "gate.jsonl"))
    monkeypatch.setenv("MERCURY_ADJUDICATION_PATH", str(tmp_path / "ledger.jsonl"))
    seen: list[str] = []

    def refuse(subject: Any, **_kwargs: Any) -> Any:
        seen.append(subject.surface)
        raise EthicalConstraintViolationError(
            action=subject.surface, score=0.0, threshold=1.0, check="harm_uplift"
        )

    monkeypatch.setattr(decision_gate, "enforce_decision_boundary", refuse)
    return seen


@pytest.mark.parametrize(("name", "call"), ENTRY_POINTS, ids=[n for n, _ in ENTRY_POINTS])
def test_every_entry_point_is_refused_by_the_gate(
    name: str, call: Callable[[], Any], _refusing_gate: list[str], tmp_path: Path
) -> None:
    with pytest.raises(EthicalConstraintViolationError):
        call()
    assert _refusing_gate == ["t"]
    assert not (tmp_path / "ledger.jsonl").exists()  # nothing was written


def test_every_public_content_entry_point_is_covered() -> None:
    """A new service function that takes caller content must be added above."""
    reads = {
        "adjudication_status",
        "adjudication_verify",
        "adjudication_rater_view",
        "adjudication_export",
        "adjudication_repair",
        "fusion_engine",
        "fusion_status",
        "jsonable",
    }
    functions = {
        name
        for name in service.__all__
        if callable(getattr(service, name)) and name[0].islower() and name not in reads
    }
    assert functions == {name for name, _ in ENTRY_POINTS}
