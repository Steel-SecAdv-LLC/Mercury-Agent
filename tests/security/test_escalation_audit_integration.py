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

    def test_secure_audit_dir_env_routes_the_hash_chained_sink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # MERCURY_SECURE_AUDIT_DIR is the only env knob for the tamper-evident
        # sink's directory; a decision must land there, not in the code default.
        from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

        monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(tmp_path / "gate.jsonl"))
        monkeypatch.setenv("MERCURY_GATE_AUDIT_SECURELOG", "1")
        monkeypatch.setenv("MERCURY_SECURE_AUDIT_DIR", str(tmp_path / "secure_env"))
        monkeypatch.delenv("MERCURY_GATE_AUDIT_DISABLED", raising=False)
        saved = sal._audit_logger
        sal._audit_logger = None
        try:
            record_gate_decision(
                decision="refused", source="test_env_dir", disposition="hard_refuse", reason="x"
            )
            secure = sal.get_audit_logger()
            secure.flush()
            secure_path = tmp_path / "secure_env" / "audit.jsonl"
            assert secure_path.exists()
            assert "harm_gate:refused" in secure_path.read_text(encoding="utf-8")
            assert str(secure.log_dir) == str(tmp_path / "secure_env")
        finally:
            current = sal._audit_logger
            if current is not None and current is not saved:
                current.shutdown()
            sal._audit_logger = saved

    def test_reconfigure_retires_prior_secure_logger_no_leak(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Repointing MERCURY_SECURE_AUDIT_DIR must *retire* the previous
        # SecureAuditLogger -- flush its buffered events and stop its flush thread
        # -- not orphan it. Orphaning would leak the daemon thread + file handle
        # and silently drop any events still sitting in the old logger's buffer.
        from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

        monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(tmp_path / "gate.jsonl"))
        monkeypatch.setenv("MERCURY_GATE_AUDIT_SECURELOG", "1")
        monkeypatch.delenv("MERCURY_GATE_AUDIT_DISABLED", raising=False)
        saved = sal._audit_logger
        sal._audit_logger = None
        try:
            # First decision -> logger bound to dir A; the event sits in A's buffer
            # (buffer flush threshold is 100, so a single event is not auto-written).
            dir_a = tmp_path / "secure_a"
            monkeypatch.setenv("MERCURY_SECURE_AUDIT_DIR", str(dir_a))
            record_gate_decision(
                decision="refused", source="leak_test_a", disposition="hard_refuse", reason="a"
            )
            logger_a = sal._audit_logger
            assert logger_a is not None
            assert str(logger_a.log_dir) == str(dir_a)
            thread_a = logger_a._flush_thread
            assert thread_a is not None and thread_a.is_alive()

            # Repoint to dir B -> A must be shut down and replaced by a new logger.
            dir_b = tmp_path / "secure_b"
            monkeypatch.setenv("MERCURY_SECURE_AUDIT_DIR", str(dir_b))
            record_gate_decision(
                decision="refused", source="leak_test_b", disposition="hard_refuse", reason="b"
            )
            logger_b = sal._audit_logger
            assert logger_b is not None
            assert logger_b is not logger_a  # global was replaced, not reused
            assert str(logger_b.log_dir) == str(dir_b)

            # A was retired: stop event set and the flush thread joins promptly
            # (the responsive wait-based loop, not a 5s sleep).
            assert logger_a._stop_event.is_set()
            thread_a.join(timeout=5.0)
            assert not thread_a.is_alive()

            # A's buffered event was flushed by shutdown, not lost -- it is on disk
            # in A's own file even though this test never called flush() on A.
            a_file = dir_a / "audit.jsonl"
            assert a_file.exists()
            assert "harm_gate:refused" in a_file.read_text(encoding="utf-8")
        finally:
            current = sal._audit_logger
            if current is not None and current is not saved:
                current.shutdown()
            sal._audit_logger = saved

    def test_shutdown_is_prompt_and_flushes_buffer(self, tmp_path: Path) -> None:
        # Direct teeth for the flush-loop change (Event.wait, not time.sleep) and
        # for shutdown flushing the buffer. With a long flush_interval the loop
        # would block the whole interval under the old `time.sleep`, so its
        # shutdown() join(5s) would time out and leave the thread ALIVE -- the
        # assertions below go red. The new wait-based loop wakes on the stop event
        # and returns in milliseconds. Uses a logger built directly so the test
        # can set flush_interval (the gate-audit path always uses the default),
        # which also removes the background-flush timing confound.
        import time as _time

        logger = sal.SecureAuditLogger(log_dir=str(tmp_path / "prompt"), flush_interval=30.0)
        thread = logger._flush_thread
        assert thread is not None and thread.is_alive()
        # One event sits in the buffer (flush threshold is 100), not yet on disk.
        logger.log_security_incident(action="harm_gate:refused", details={"k": "v"}, resource="t")
        log_file = tmp_path / "prompt" / "audit.jsonl"

        start = _time.perf_counter()
        logger.shutdown()
        elapsed = _time.perf_counter() - start

        assert (
            elapsed < 3.0
        ), f"shutdown blocked {elapsed:.1f}s (~flush_interval => sleep, not wait)"
        assert not thread.is_alive()  # the flush thread was actually reaped
        # The buffered event was flushed by shutdown, not dropped.
        assert log_file.exists()
        assert "harm_gate:refused" in log_file.read_text(encoding="utf-8")

    def test_same_secure_dir_reuses_logger_and_keeps_one_hash_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two decisions under the SAME MERCURY_SECURE_AUDIT_DIR must hit the
        # hot-path early return and REUSE the one logger, never rebuild it.
        # Rebuilding per decision would mint a fresh HMAC key + reset the genesis
        # hash, silently breaking cross-event chain linkage -- an integrity
        # regression the other tests miss (they use the no-secure_dir sink).
        from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

        monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(tmp_path / "gate.jsonl"))
        monkeypatch.setenv("MERCURY_GATE_AUDIT_SECURELOG", "1")
        monkeypatch.setenv("MERCURY_SECURE_AUDIT_DIR", str(tmp_path / "secure_same"))
        monkeypatch.delenv("MERCURY_GATE_AUDIT_DISABLED", raising=False)
        saved = sal._audit_logger
        sal._audit_logger = None
        try:
            record_gate_decision(
                decision="refused", source="same_dir_1", disposition="hard_refuse", reason="1"
            )
            logger1 = sal._audit_logger
            record_gate_decision(
                decision="escalated", source="same_dir_2", disposition="escalate", reason="2"
            )
            logger2 = sal._audit_logger

            assert logger1 is not None
            assert logger2 is logger1  # same dir -> identical logger, no reset
            logger1.flush()

            secure_path = tmp_path / "secure_same" / "audit.jsonl"
            assert len(secure_path.read_text(encoding="utf-8").splitlines()) >= 2
            ok, message = logger1.verify_log_integrity(secure_path)
            assert ok, message  # both events form ONE verifiable hash chain
        finally:
            current = sal._audit_logger
            if current is not None and current is not saved:
                current.shutdown()
            sal._audit_logger = saved

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

    def test_content_tampering_is_detected(
        self, audit_sinks: tuple[Path, sal.SecureAuditLogger]
    ) -> None:
        # Editing a hashed field (details) while leaving the hash columns intact
        # must still be caught -- the per-event hash recompute, not just linkage.
        import json

        tmp_path, secure = audit_sinks
        broker = EscalationBroker(reviewer=lambda r: True, max_approvals=5)
        for i in range(3):
            broker.review(_escalation(i))
        secure.flush()

        secure_path = tmp_path / "secure" / "audit.jsonl"
        assert secure.verify_log_integrity(secure_path)[0] is True  # clean first

        lines = secure_path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[1])
        record["details"] = {"tampered": True}  # edit content; DO NOT touch the hashes
        lines[1] = json.dumps(record)
        secure_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ok, message = secure.verify_log_integrity(secure_path)
        assert ok is False
        assert "broken" in message.lower()
