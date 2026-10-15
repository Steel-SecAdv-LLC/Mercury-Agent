# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The audit hash chain survives restarts and never interleaves two writers.

Before the fix every process start opened a fresh genesis while appending to
the same ``audit.jsonl``, so a log that had seen one restart failed
``verify_log_integrity`` -- indistinguishable from tampering. A second process
on the same directory interleaved a second chain into the same file.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from itertools import pairwise
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.security.secure_audit_logging import (
    AuditEventCategory,
    SecureAuditLogger,
)

if TYPE_CHECKING:
    from pathlib import Path

KEY = b"\x11" * 32


def _logger(path: Path, **kwargs: object) -> SecureAuditLogger:
    return SecureAuditLogger(log_dir=path, hmac_key=KEY, flush_interval=3600, **kwargs)  # type: ignore[arg-type]


def _write(path: Path, *actions: str) -> None:
    log = _logger(path)
    for action in actions:
        log.log(AuditEventCategory.SYSTEM, action)
    log.shutdown()


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_chain_verifies_across_restarts(tmp_path: Path) -> None:
    _write(tmp_path, "a", "b")
    _write(tmp_path, "c")
    _write(tmp_path, "d", "e")
    events = _events(tmp_path / "audit.jsonl")
    assert [e["sequence_number"] for e in events] == [0, 1, 2, 3, 4]
    for prev, nxt in pairwise(events):
        assert nxt["previous_hash"] == prev["event_hash"]
    ok, message = _logger(tmp_path).verify_log_integrity()
    assert ok, message


def test_tampering_is_still_detected_after_resume(tmp_path: Path) -> None:
    _write(tmp_path, "a", "b")
    _write(tmp_path, "c")
    log_path = tmp_path / "audit.jsonl"
    lines = log_path.read_text().splitlines()
    record = json.loads(lines[1])
    record["action"] = "forged"
    lines[1] = json.dumps(record)
    log_path.write_text("\n".join(lines) + "\n")
    ok, message = _logger(tmp_path).verify_log_integrity()
    assert not ok
    assert "[1]" in message


def test_resume_after_rotation_links_to_the_rotated_segment(tmp_path: Path) -> None:
    _write(tmp_path, "a", "b")
    rotated = tmp_path / "audit_20260101_000000.jsonl"
    (tmp_path / "audit.jsonl").rename(rotated)
    _write(tmp_path, "c")
    first_new = _events(tmp_path / "audit.jsonl")[0]
    last_old = _events(rotated)[-1]
    assert first_new["previous_hash"] == last_old["event_hash"]
    assert first_new["sequence_number"] == last_old["sequence_number"] + 1  # type: ignore[operator]


def test_unreadable_tail_starts_a_visible_new_chain(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path, "a")
    with open(tmp_path / "audit.jsonl", "a") as f:
        f.write("{truncated\n")
    with caplog.at_level("CRITICAL"):
        _logger(tmp_path).shutdown()
    assert any("does not end in a readable event" in r.message for r in caplog.records)


@pytest.mark.skipif(sys.platform == "win32", reason="advisory flock is POSIX-only")
def test_second_process_writes_its_own_chain(tmp_path: Path) -> None:
    """A concurrent writer is moved to audit-<pid>.jsonl; both files verify."""
    holder = _logger(tmp_path)
    holder.log(AuditEventCategory.SYSTEM, "primary")
    script = textwrap.dedent(f"""
        import os
        from omni_mercury_engine.security.secure_audit_logging import (
            AuditEventCategory, SecureAuditLogger,
        )
        log = SecureAuditLogger(log_dir={str(tmp_path)!r}, hmac_key={KEY!r}, flush_interval=3600)
        log.log(AuditEventCategory.SYSTEM, "secondary")
        log.shutdown()
        ok, _ = log.verify_log_integrity()
        print(log._current_log_path.name, os.getpid(), ok)
        """)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120
    ).stdout.split()
    name, pid, ok = out[-3], out[-2], out[-1]
    assert name == f"audit-{pid}.jsonl"
    assert ok == "True"
    holder.shutdown()
    assert [e["action"] for e in _events(tmp_path / "audit.jsonl")] == ["primary"]
    verified, message = holder.verify_log_integrity()
    assert verified, message


def test_shutdown_releases_the_writer_lock(tmp_path: Path) -> None:
    first = _logger(tmp_path)
    first.shutdown()
    second = _logger(tmp_path)
    try:
        assert second._current_log_path.name == "audit.jsonl"
    finally:
        second.shutdown()
