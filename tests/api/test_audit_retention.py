# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for operator-configurable audit retention.

Covers the environment-driven rotation config on ``build_auth_auditor``
(rotate size + max rotated files), the ``MERCURY_AUDIT_RETENTION_DAYS`` parsing,
and the time-based segment prune (:func:`prune_rotated_audit_segments`): it
deletes whole rotated segments and their ``.sha256`` sidecars older than the
cutoff, never touches the active ``audit.jsonl`` or individual lines, and so
leaves the hash chain of every retained log independently verifiable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omni_mercury_engine.api import auth_audit
from omni_mercury_engine.api.auth_audit import (
    audit_retention_days,
    build_auth_auditor,
    prune_rotated_audit_segments,
)


class TestRotationConfigFromEnv:
    """``build_auth_auditor`` reads the rotation knobs from the environment."""

    def test_defaults_when_unset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With only the dir set, the SecureAuditLogger defaults apply."""
        monkeypatch.setenv(auth_audit.AUDIT_DIR_ENV, str(tmp_path))
        monkeypatch.delenv(auth_audit.AUDIT_ROTATE_SIZE_MB_ENV, raising=False)
        monkeypatch.delenv(auth_audit.AUDIT_MAX_FILES_ENV, raising=False)
        auditor = build_auth_auditor()
        assert auditor._secure is not None
        assert auditor._secure.rotate_size_mb == 100.0
        assert auditor._secure.max_rotated_files == 10

    def test_env_overrides_are_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rotate size and file cap come from the environment when set."""
        monkeypatch.setenv(auth_audit.AUDIT_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(auth_audit.AUDIT_ROTATE_SIZE_MB_ENV, "5")
        monkeypatch.setenv(auth_audit.AUDIT_MAX_FILES_ENV, "3")
        auditor = build_auth_auditor()
        assert auditor._secure is not None
        assert auditor._secure.rotate_size_mb == 5.0
        assert auditor._secure.max_rotated_files == 3

    def test_malformed_env_falls_back_to_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Junk values keep the known-good defaults rather than degrading."""
        monkeypatch.setenv(auth_audit.AUDIT_DIR_ENV, str(tmp_path))
        monkeypatch.setenv(auth_audit.AUDIT_ROTATE_SIZE_MB_ENV, "not-a-number")
        monkeypatch.setenv(auth_audit.AUDIT_MAX_FILES_ENV, "-4")
        auditor = build_auth_auditor()
        assert auditor._secure is not None
        assert auditor._secure.rotate_size_mb == 100.0
        assert auditor._secure.max_rotated_files == 10


class TestRetentionDaysParsing:
    """``MERCURY_AUDIT_RETENTION_DAYS`` — unset/malformed disables pruning."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("", 0.0), ("30", 30.0), ("0.5", 0.5), ("bad", 0.0), ("-5", 0.0), ("0", 0.0)],
    )
    def test_parsing(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
        """Only a positive number enables time-based retention."""
        if raw:
            monkeypatch.setenv(auth_audit.AUDIT_RETENTION_DAYS_ENV, raw)
        else:
            monkeypatch.delenv(auth_audit.AUDIT_RETENTION_DAYS_ENV, raising=False)
        assert audit_retention_days() == expected


def _make_segment(directory: Path, name: str, *, age_days: float) -> Path:
    """Create a rotated segment + its ``.sha256`` and backdate its mtime."""
    segment = directory / name
    segment.write_text('{"event": "x"}\n')
    sidecar = segment.with_suffix(".sha256")
    sidecar.write_text("deadbeef")
    stamp = (Path(directory).stat().st_mtime) - age_days * 86400
    os.utime(segment, (stamp, stamp))
    os.utime(sidecar, (stamp, stamp))
    return segment


class TestSegmentPrune:
    """Time-based deletion of rotated segments only."""

    def test_deletes_old_keeps_recent_and_active(self, tmp_path: Path) -> None:
        """Old segments and sidecars go; recent segments and the active stay."""
        import time

        active = tmp_path / "audit.jsonl"
        active.write_text('{"event": "live"}\n')
        old = _make_segment(tmp_path, "audit_20200101_000000.jsonl", age_days=400)
        recent = _make_segment(tmp_path, "audit_20991231_000000.jsonl", age_days=0)

        cutoff = time.time() - 90 * 86400  # keep 90 days
        removed = prune_rotated_audit_segments(tmp_path, cutoff)

        assert removed == 1
        assert not old.exists()
        assert not old.with_suffix(".sha256").exists()
        assert recent.exists()
        assert recent.with_suffix(".sha256").exists()
        assert active.exists()  # the active file is never a prune target

    def test_active_file_never_pruned_even_if_old(self, tmp_path: Path) -> None:
        """A stale active ``audit.jsonl`` is out of the glob and survives."""
        import time

        active = tmp_path / "audit.jsonl"
        active.write_text('{"event": "live"}\n')
        stamp = time.time() - 1000 * 86400
        os.utime(active, (stamp, stamp))

        removed = prune_rotated_audit_segments(tmp_path, time.time())
        assert removed == 0
        assert active.exists()

    def test_missing_directory_is_zero(self, tmp_path: Path) -> None:
        """A nonexistent audit dir prunes nothing (never raises)."""
        assert prune_rotated_audit_segments(tmp_path / "nope", 0.0) == 0

    def test_retained_segment_integrity_survives_prune(self, tmp_path: Path) -> None:
        """A real hash-chained log still verifies after an unrelated prune.

        Writes events through the SecureAuditLogger (a genuine hash chain),
        deletes a *different* old segment, and re-verifies the written log —
        proving the prune touches only whole old segments, never the content
        or linkage of a retained chain.
        """
        import time

        from omni_mercury_engine.security.secure_audit_logging import (
            AuditEventCategory,
            SecureAuditLogger,
        )

        logger = SecureAuditLogger(log_dir=str(tmp_path))
        try:
            for i in range(5):
                logger.log(
                    AuditEventCategory.AUTHENTICATION, action=f"login-{i}", outcome="success"
                )
            logger.flush()
            ok_before, _msg = logger.verify_log_integrity()
            assert ok_before

            # An unrelated, older rotated segment exists and is pruned.
            _make_segment(tmp_path, "audit_20000101_000000.jsonl", age_days=9000)
            removed = prune_rotated_audit_segments(tmp_path, time.time() - 30 * 86400)
            assert removed == 1

            # The active hash chain is untouched and still verifies.
            ok_after, _msg2 = logger.verify_log_integrity()
            assert ok_after
        finally:
            logger.shutdown()
