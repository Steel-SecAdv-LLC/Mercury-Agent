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


# ---------------------------------------------------------------------------
# Torn tail, unique rotation, per-writer retention, orphans, verification
# ---------------------------------------------------------------------------


def test_torn_tail_is_cut_and_the_chain_resumes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A crash mid-write leaves a fragment; the next start cuts only that."""
    _write(tmp_path, "a", "b")
    log_path = tmp_path / "audit.jsonl"
    with open(log_path, "a") as f:
        f.write('{"event_id": "AE-torn", "actio')
    with caplog.at_level("CRITICAL"):
        _write(tmp_path, "c")
    events = _events(log_path)  # parses: the fragment did not swallow "c"
    assert [e["action"] for e in events] == ["a", "b", "c"]
    assert [e["sequence_number"] for e in events] == [0, 1, 2]
    assert events[2]["previous_hash"] == events[1]["event_hash"]
    assert any("unterminated" in r.message for r in caplog.records)
    kept = list(tmp_path.glob("audit.jsonl.torn-*"))
    assert len(kept) == 1 and kept[0].read_text() == '{"event_id": "AE-torn", "actio'
    ok, message = _logger(tmp_path).verify_log_integrity()
    assert ok, message


def test_file_of_only_a_fragment_starts_clean(tmp_path: Path) -> None:
    (tmp_path / "audit.jsonl").write_text('{"half')
    _write(tmp_path, "a")
    assert [e["action"] for e in _events(tmp_path / "audit.jsonl")] == ["a"]


def test_rotations_in_the_same_instant_never_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    log = _logger(tmp_path, max_rotated_files=50)
    frozen = 1_800_000_000.25
    monkeypatch.setattr(time, "time", lambda: frozen)
    for i in range(4):
        log.log(AuditEventCategory.SYSTEM, f"e{i}")
        log.flush()
        with log._file_lock:
            log._rotate_log()
    monkeypatch.undo()
    log.shutdown()
    segments = sorted(p.name for p in tmp_path.glob("audit_*.jsonl"))
    assert len(segments) == 4, segments
    actions = sorted(str(e["action"]) for p in tmp_path.glob("audit_*.jsonl") for e in _events(p))
    assert actions == ["e0", "e1", "e2", "e3"]
    assert sum(1 for name in segments if "-" in name) == 3  # -1, -2, -3 suffixes


def test_rename_refuses_to_clobber(tmp_path: Path) -> None:
    from omni_mercury_engine.security.secure_audit_logging import _rename_no_clobber

    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_text("new")
    dst.write_text("old")
    assert _rename_no_clobber(src, dst) is False
    assert dst.read_text() == "old" and src.read_text() == "new"
    dst.unlink()
    assert _rename_no_clobber(src, dst) is True
    assert dst.read_text() == "new" and not src.exists()


def _segment(path: Path, name: str, mtime: float, action: str = "x") -> Path:
    import os

    target = path / name
    target.write_text(json.dumps({"action": action}) + "\n")
    os.utime(target, (mtime, mtime))
    return target


def test_retention_counts_only_this_writers_segments(tmp_path: Path) -> None:
    """A worker's segments never count against (or get pruned by) the primary."""
    import os

    live_pid = os.getpid()  # "a live worker": this process is certainly running
    worker = [
        _segment(tmp_path, f"audit_20260101_00000{i}.000000_{live_pid}.jsonl", 1000 + i)
        for i in range(3)
    ]
    primary = [
        _segment(tmp_path, f"audit_20260102_00000{i}.000000.jsonl", 2000 + i) for i in range(3)
    ]
    log = _logger(tmp_path, max_rotated_files=2)
    with log._file_lock:
        log._cleanup_old_logs()
    log.shutdown()
    assert [p.exists() for p in primary] == [False, True, True]
    assert all(p.exists() for p in worker)


def test_resume_ignores_other_writers_segments(tmp_path: Path) -> None:
    import os

    _write(tmp_path, "a")
    own = tmp_path / "audit_20260101_000000.000000.jsonl"
    (tmp_path / "audit.jsonl").rename(own)
    # A newer segment of another writer must not become this chain's parent.
    foreign = _segment(tmp_path, f"audit_20990101_000000.000000_{os.getpid()}.jsonl", 9e9)
    foreign.write_text(json.dumps({"event_hash": "f" * 64, "sequence_number": 99}) + "\n")
    _write(tmp_path, "b")
    first_new = _events(tmp_path / "audit.jsonl")[0]
    assert first_new["previous_hash"] == _events(own)[-1]["event_hash"]
    assert first_new["sequence_number"] == 1


def test_orphaned_writer_files_are_capped(tmp_path: Path) -> None:
    """Dead writers' active logs and segments share one retention pool."""
    import os

    dead_pid = 2**22 + 12345  # above the default pid_max: never a live process
    orphans = [_segment(tmp_path, f"audit-{dead_pid + i}.jsonl", 1000 + i) for i in range(3)]
    orphans += [
        _segment(tmp_path, f"audit_20260101_00000{i}.000000_{dead_pid}.jsonl", 2000 + i)
        for i in range(2)
    ]
    live = _segment(tmp_path, f"audit-{os.getpid()}.jsonl", 10)  # oldest, but live
    log = _logger(tmp_path, max_rotated_files=2)  # primary cleans orphans at start
    log.shutdown()
    remaining = {p.name for p in orphans if p.exists()}
    assert remaining == {orphans[-1].name, orphans[-2].name}  # newest two by mtime
    assert live.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="advisory flock is POSIX-only")
def test_worker_lock_marks_a_live_writer(tmp_path: Path) -> None:
    from omni_mercury_engine.security.secure_audit_logging import _writer_is_live

    script = textwrap.dedent(f"""
        import sys, time
        from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger
        log = SecureAuditLogger(log_dir={str(tmp_path)!r}, hmac_key={KEY!r}, flush_interval=3600)
        print(log._writer_pid, flush=True)
        sys.stdin.readline()
        log.shutdown()
        """)
    holder = _logger(tmp_path)  # this process is the primary
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout is not None and proc.stdin is not None
        pid = int(proc.stdout.readline().split()[-1])
        assert _writer_is_live(tmp_path, pid) is True
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.wait(timeout=120)
        assert _writer_is_live(tmp_path, pid) is False
    finally:
        if proc.poll() is None:
            proc.kill()
        holder.shutdown()


def test_interior_deletion_reports_the_sequence_gap(tmp_path: Path) -> None:
    _write(tmp_path, "a", "b", "c", "d")
    log_path = tmp_path / "audit.jsonl"
    lines = log_path.read_text().splitlines()
    del lines[1:3]
    log_path.write_text("\n".join(lines) + "\n")
    ok, message = _logger(tmp_path).verify_log_integrity()
    assert not ok
    assert "sequence 0 -> 3 (2 record(s) missing)" in message


def test_sequence_check_flags_renumbering(tmp_path: Path) -> None:
    from omni_mercury_engine.security.secure_audit_logging import (
        SecureHashChain,
        load_audit_events,
    )

    _write(tmp_path, "a", "b")
    events = load_audit_events(tmp_path / "audit.jsonl")
    events[1].sequence_number = 5
    ok, invalid = SecureHashChain(KEY).verify_chain(events)
    assert not ok and invalid == [1]


def test_directory_verification_spans_segments(tmp_path: Path) -> None:
    from omni_mercury_engine.security.secure_audit_logging import verify_audit_directory

    log = _logger(tmp_path, rotate_size_mb=1e-6)  # rotate before every write
    for action in "abcd":
        log.log(AuditEventCategory.SYSTEM, action)
        log.flush()
    log.shutdown()
    _write(tmp_path, "e")  # a restart in between keeps the chain linked
    ok, chains = verify_audit_directory(tmp_path, KEY)
    assert ok, chains
    assert [c["writer"] for c in chains] == ["primary"]
    assert len(chains[0]["files"]) == 4  # three segments + the active log

    # Deleting a whole interior segment breaks the cross-file link.
    segments = sorted(tmp_path.glob("audit_*.jsonl"))
    segments[1].unlink()
    ok, chains = verify_audit_directory(tmp_path, KEY)
    assert not ok
    assert any("does not continue" in f["message"] for f in chains[0]["files"])


def test_directory_verification_checks_sidecars(tmp_path: Path) -> None:
    from omni_mercury_engine.security.secure_audit_logging import verify_audit_directory

    log = _logger(tmp_path, rotate_size_mb=1e-6)
    for action in "ab":
        log.log(AuditEventCategory.SYSTEM, action)
        log.flush()
    log.shutdown()
    segment = next(tmp_path.glob("audit_*.jsonl"))
    segment.with_suffix(".sha256").write_text("0" * 64)
    ok, chains = verify_audit_directory(tmp_path, KEY)
    assert not ok
    assert any(".sha256" in f["message"] for f in chains[0]["files"])


def test_directory_verification_of_an_empty_dir_fails(tmp_path: Path) -> None:
    from omni_mercury_engine.security.secure_audit_logging import verify_audit_directory

    ok, _chains = verify_audit_directory(tmp_path, KEY)
    assert not ok
