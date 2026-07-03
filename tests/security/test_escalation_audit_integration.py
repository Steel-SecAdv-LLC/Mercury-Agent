# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end: ESCALATE -> HumanReviewCallback -> durable audit -> hash-chain verify.

Ties the whole escalation/audit control together and asserts the done-state:

* an :class:`EscalationBroker` consults an injected test ``HumanReviewCallback``;
* every decision (approve *and* fail-closed deny) is written to the durable
  ``MERCURY_GATE_AUDIT_LOG`` JSONL sink;
* with ``MERCURY_GATE_AUDIT_SECURELOG=1`` the decision is also forwarded to the
  hash-chained :class:`SecureAuditLogger`, whose integrity is *verified* -- and a
  tamper is *detected*.

Footgun handled here: ``MERCURY_GATE_AUDIT_LOG`` steers only the plain JSONL; the
tamper-evident sink is the ``SecureAuditLogger``'s own ``log_dir`` (a process
global). The fixture redirects it with ``configure_audit_logger`` and restores
the prior global on teardown so the singleton does not leak across tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.cognitive.escalation import EscalationBroker, EscalationRecord
from omni_mercury_engine.security import secure_audit_logging as sal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def audit_sinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, sal.SecureAuditLogger]]:
    """Redirect both audit sinks into ``tmp_path`` and restore the global on teardown."""
    monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(tmp_path / "gate.jsonl"))
    monkeypatch.setenv("MERCURY_GATE_AUDIT_SECURELOG", "1")
    monkeypatch.delenv("MERCURY_GATE_AUDIT_DISABLED", raising=False)
    saved = sal._audit_logger
    secure = sal.configure_audit_logger(log_dir=str(tmp_path / "secure"))
    try:
        yield tmp_path, secure
    finally:
        secure.shutdown()
        sal._audit_logger = saved


def _escalation(i: int = 0) -> EscalationRecord:
    return EscalationRecord(
        query=f"licensed engineer production-adjacent query {i}",
        reason="gray-zone request a human could authorize",
        disposition="escalate",
        hazard_domain="chemical",
        intent="production",
    )


class TestEscalationAuditIntegration:
    def test_approval_persisted_to_both_sinks_and_chain_verified(
        self, audit_sinks: tuple[Path, sal.SecureAuditLogger]
    ) -> None:
        tmp_path, secure = audit_sinks
        approvals: list[EscalationRecord] = []

        def reviewer(record: EscalationRecord) -> bool:
            approvals.append(record)
            return True

        broker = EscalationBroker(reviewer=reviewer, max_approvals=3)
        decision = broker.review(_escalation())

        assert decision.approved is True
        assert len(approvals) == 1  # the HumanReviewCallback was actually consulted
        secure.flush()

        gate_text = (tmp_path / "gate.jsonl").read_text(encoding="utf-8")
        assert '"decision": "approved"' in gate_text
        assert "escalation_broker" in gate_text

        secure_path = tmp_path / "secure" / "audit.jsonl"
        assert secure_path.exists()
        assert "harm_gate:approved" in secure_path.read_text(encoding="utf-8")

        ok, message = secure.verify_log_integrity(secure_path)
        assert ok, message

    def test_no_reviewer_denies_fail_closed_and_audits(
        self, audit_sinks: tuple[Path, sal.SecureAuditLogger]
    ) -> None:
        tmp_path, secure = audit_sinks
        broker = EscalationBroker(reviewer=None)  # no human in the loop -> deny
        decision = broker.review(_escalation())

        assert decision.approved is False
        secure.flush()
        gate_text = (tmp_path / "gate.jsonl").read_text(encoding="utf-8")
        assert '"decision": "escalation_denied"' in gate_text

    def test_bounded_autonomy_ceiling_denies_and_audits(
        self, audit_sinks: tuple[Path, sal.SecureAuditLogger]
    ) -> None:
        tmp_path, secure = audit_sinks
        broker = EscalationBroker(reviewer=lambda r: True, max_approvals=1)
        assert broker.review(_escalation(0)).approved is True
        # Second approval is refused by the bounded-autonomy ceiling.
        second = broker.review(_escalation(1))
        assert second.approved is False
        assert "ceiling" in second.reason
        secure.flush()
        assert '"decision": "escalation_denied"' in (tmp_path / "gate.jsonl").read_text(
            encoding="utf-8"
        )

    def test_tampering_secure_log_is_detected(
        self, audit_sinks: tuple[Path, sal.SecureAuditLogger]
    ) -> None:
        import json

        tmp_path, secure = audit_sinks
        broker = EscalationBroker(reviewer=lambda r: True, max_approvals=5)
        for i in range(3):
            broker.review(_escalation(i))
        secure.flush()

        secure_path = tmp_path / "secure" / "audit.jsonl"
        lines = secure_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 2
        # Corrupt the first event's hash -> the next event's back-link no longer matches.
        first = json.loads(lines[0])
        first["event_hash"] = "0" * 64
        lines[0] = json.dumps(first)
        secure_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ok, message = secure.verify_log_integrity(secure_path)
        assert ok is False
        assert "broken" in message.lower()
