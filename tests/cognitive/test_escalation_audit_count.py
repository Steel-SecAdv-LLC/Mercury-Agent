# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The audited ``approvals_used`` is the count of the decision being audited.

``EscalationBroker.review`` adjudicated under the approvals lock and then read
``self._approvals`` again, outside it, while building the audit row.  Another
review completing in between put *its* count on this decision's row: an
approval that was the first of the session could be audited as the second.

Also pinned here, because it is the reviewer contract and it is fail-closed:
only the Python literal ``True`` approves.  ``numpy.bool_(True)`` does not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.cognitive import escalation
from omni_mercury_engine.cognitive.escalation import EscalationBroker, EscalationRecord


@pytest.fixture
def audit_rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what the broker hands to the durable gate audit."""
    rows: list[dict[str, Any]] = []

    def _capture(**fields: Any) -> None:
        rows.append(fields)

    monkeypatch.setattr(escalation, "record_gate_decision", _capture)
    return rows


class _RecordReadWhileAnotherReviewLands:
    """An escalation record whose audit fields are read during a second review.

    ``review`` reads the record's fields to build the audit row after it has
    adjudicated.  Reading ``query`` here runs one more complete review on the
    same broker first: what a second thread does when it is scheduled between
    this decision and its audit write.
    """

    reason = "licensed engineer's production-adjacent query"
    disposition = "escalate"
    hazard_domain = "none"
    intent = "mechanism"
    signals: tuple[str, ...] = ()

    def __init__(self, broker: EscalationBroker) -> None:
        self._broker = broker
        self._landed = False

    @property
    def query(self) -> str:
        if not self._landed:
            self._landed = True
            self._broker.review(EscalationRecord(query="second", reason="second request"))
        return "first"


def _row(rows: list[dict[str, Any]], query: str) -> dict[str, Any]:
    (row,) = [r for r in rows if r["query"] == query]
    return row


def test_an_approval_is_audited_with_its_own_count(audit_rows: list[dict[str, Any]]) -> None:
    broker = EscalationBroker(reviewer=lambda rec: True, max_approvals=5)

    decision = broker.review(_RecordReadWhileAnotherReviewLands(broker))  # type: ignore[arg-type]

    assert decision.approved is True
    assert broker.approvals_used == 2
    assert _row(audit_rows, "second")["extra"]["approvals_used"] == 2
    assert _row(audit_rows, "first")["extra"]["approvals_used"] == 1, (
        "the first approval of the session was audited with the count another "
        "review reached afterwards"
    )


def test_a_denial_is_audited_with_the_count_at_its_decision(
    audit_rows: list[dict[str, Any]],
) -> None:
    verdicts = iter([False, True])
    broker = EscalationBroker(reviewer=lambda rec: next(verdicts), max_approvals=5)

    decision = broker.review(_RecordReadWhileAnotherReviewLands(broker))  # type: ignore[arg-type]

    assert decision.approved is False
    assert broker.approvals_used == 1
    assert _row(audit_rows, "first")["decision"] == "escalation_denied"
    assert _row(audit_rows, "first")["extra"]["approvals_used"] == 0


def test_sequential_reviews_are_audited_in_step(audit_rows: list[dict[str, Any]]) -> None:
    broker = EscalationBroker(reviewer=lambda rec: True, max_approvals=2)
    record = EscalationRecord(query="q", reason="r")

    decisions = [broker.review(record) for _ in range(3)]

    assert [d.approved for d in decisions] == [True, True, False]
    assert [row["extra"]["approvals_used"] for row in audit_rows] == [1, 2, 2]


def test_only_the_python_literal_true_approves(audit_rows: list[dict[str, Any]]) -> None:
    """``np.bool_(True)`` is truthy and equal to ``True`` -- and is not ``True``."""
    broker = EscalationBroker(
        reviewer=lambda rec: np.bool_(True),  # type: ignore[arg-type,return-value]
        max_approvals=5,
    )

    decision = broker.review(EscalationRecord(query="q", reason="r"))

    assert decision.approved is False
    assert broker.approvals_used == 0
    assert audit_rows[0]["decision"] == "escalation_denied"
