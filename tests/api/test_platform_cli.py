# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the ``mercury-agent platform`` operator commands.

Drives every subcommand through Click's CliRunner against real SQLite state
in a tmp path: account show/list/set-tier/disable/enable, quota override
set/show/clear (asserting the change is visible through
``QuotaEnforcer.config_for`` — the exact resolution enforcement uses), usage
reports, the no-secrets output contract, the refusal without
``MERCURY_KEYSTORE_PATH``, and audit-chain verification passing on a real
chain and failing on a tampered line.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from omni_mercury_engine.api import auth
from omni_mercury_engine.api.identity_store import Account, Session, SqliteIdentityStore
from omni_mercury_engine.api.quota import build_quota_enforcer
from omni_mercury_engine.api.usage_ledger import SqliteUsageLedger, UsageEvent
from omni_mercury_engine.cli import main

if TYPE_CHECKING:
    from pathlib import Path

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

#: A stand-in stored hash — clearly not a credential; the show/list contract
#: is that even this placeholder never surfaces in operator output.
_PLACEHOLDER_HASH = "not-a-real-hash"
_PLACEHOLDER_HASH_2 = "also-not-a-hash"


@pytest.fixture
def keystore(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point MERCURY_KEYSTORE_PATH at a tmp SQLite file with one account."""
    db_path = tmp_path / "platform.db"
    monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(db_path))
    store = SqliteIdentityStore(db_path)
    store.create_account(
        Account(
            id="acct-cli-1",
            email="op@b.com",
            password_hash=_PLACEHOLDER_HASH,
            is_verified=True,
            is_active=True,
            created_at=_T0,
        )
    )
    store.close()
    return db_path


def _invoke(*args: str) -> tuple[int, str]:
    """Run the CLI and return (exit_code, stdout)."""
    result = CliRunner().invoke(main, list(args))
    return result.exit_code, result.output


def test_refuses_without_keystore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every store-backed command refuses clearly when the env var is unset."""
    monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
    result = CliRunner().invoke(main, ["platform", "account", "list"])
    assert result.exit_code == 1
    assert "MERCURY_KEYSTORE_PATH" in result.output


def test_account_show_and_list_never_print_secrets(keystore: Path) -> None:
    """Operator views carry no password hash / TOTP material."""
    code, out = _invoke("platform", "account", "show", "op@b.com")
    assert code == 0
    payload = json.loads(out)
    assert payload["email"] == "op@b.com"
    assert "password_hash" not in out
    assert "not-a-real-hash" not in out
    assert "totp_secret" not in out

    code, out = _invoke("platform", "account", "list")
    assert code == 0
    assert json.loads(out)[0]["id"] == "acct-cli-1"
    assert "not-a-real-hash" not in out


def test_show_unknown_account_fails(keystore: Path) -> None:
    """An unknown identifier exits 1 with a clear message."""
    code, out = _invoke("platform", "account", "show", "ghost@b.com")
    assert code == 1


def test_set_tier_round_trip_changes_config_for(
    keystore: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """set-tier persists and visibly changes QuotaEnforcer.config_for."""
    monkeypatch.setenv("MERCURY_QUOTA_TIER_SUPPORTER", "5000,3600000")
    before = build_quota_enforcer().config_for("acct-cli-1", "free").max_requests

    code, out = _invoke("platform", "account", "set-tier", "op@b.com", "supporter")
    assert code == 0
    payload = json.loads(out)
    assert payload["tier"] == "supporter"
    assert payload["effective_quota"]["max_requests"] == 5000

    store = SqliteIdentityStore(keystore)
    account = store.get_account_by_id("acct-cli-1")
    store.close()
    assert account is not None and account.tier == "supporter"
    after = build_quota_enforcer().config_for(account.id, account.tier).max_requests
    assert (before, after) == (1000, 5000)


def test_disable_drops_sessions_and_enable_restores(keystore: Path) -> None:
    """disable flips is_active and kills live sessions; enable flips back."""
    store = SqliteIdentityStore(keystore)
    store.create_session(
        Session(
            token_hash="h" * 64,
            account_id="acct-cli-1",
            created_at=_T0,
            expires_at=_T0 + timedelta(days=1),
        )
    )
    store.close()

    code, out = _invoke("platform", "account", "disable", "op@b.com")
    assert code == 0
    assert json.loads(out)["is_active"] is False
    store = SqliteIdentityStore(keystore)
    assert store.get_session("h" * 64) is None  # live session revoked
    store.close()

    code, out = _invoke("platform", "account", "enable", "op@b.com")
    assert code == 0
    assert json.loads(out)["is_active"] is True


def test_quota_override_set_show_clear_round_trip(
    keystore: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override wins in config_for until cleared, then tier applies again."""
    monkeypatch.setenv("MERCURY_QUOTA_TIER_SUPPORTER", "5000,3600000")
    _invoke("platform", "account", "set-tier", "op@b.com", "supporter")

    code, out = _invoke(
        "platform",
        "quota",
        "override",
        "set",
        "op@b.com",
        "--max-requests",
        "7",
        "--max-compute-ms",
        "99.5",
        "--window-seconds",
        "60",
    )
    assert code == 0
    assert json.loads(out)["effective_quota"] == {
        "window_seconds": 60,
        "max_requests": 7,
        "max_compute_ms": 99.5,
    }
    config = build_quota_enforcer().config_for("acct-cli-1", "supporter")
    assert (config.max_requests, config.max_compute_ms) == (7, 99.5)

    code, out = _invoke("platform", "quota", "override", "show", "op@b.com")
    assert code == 0
    assert json.loads(out)["override"]["max_requests"] == 7

    code, out = _invoke("platform", "quota", "override", "clear", "op@b.com")
    assert code == 0
    assert json.loads(out)["effective_quota"]["max_requests"] == 5000  # tier again
    assert build_quota_enforcer().config_for("acct-cli-1", "supporter").max_requests == 5000
    code, out = _invoke("platform", "quota", "override", "show", "op@b.com")
    assert json.loads(out)["override"] is None


def test_usage_report_top_and_single_account(keystore: Path) -> None:
    """The report ranks accounts by requests and can focus one account."""
    store = SqliteIdentityStore(keystore)
    store.create_account(
        Account(
            id="acct-cli-2",
            email="heavy@b.com",
            password_hash=_PLACEHOLDER_HASH_2,
            is_verified=True,
            is_active=True,
            created_at=_T0,
        )
    )
    store.close()
    ledger = SqliteUsageLedger(keystore)
    now = datetime.now(UTC)
    for _ in range(3):
        ledger.record(UsageEvent("acct-cli-2", now, "/api/v1/detect", 10.0))
    ledger.record(UsageEvent("acct-cli-1", now, "/api/v1/detect", 5.0))
    ledger.close()

    code, out = _invoke("platform", "usage", "report", "--top", "1")
    assert code == 0
    payload = json.loads(out)
    assert len(payload["accounts"]) == 1
    assert payload["accounts"][0]["email"] == "heavy@b.com"
    assert payload["accounts"][0]["requests"] == 3

    code, out = _invoke("platform", "usage", "report", "--account", "op@b.com")
    assert code == 0
    payload = json.loads(out)
    assert payload["accounts"][0]["requests"] == 1
    assert payload["accounts"][0]["compute_ms"] == pytest.approx(5.0)


def test_audit_verify_passes_then_fails_on_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """verify is OK on a real chain and exits 1 after a line is doctored."""
    from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger

    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("MERCURY_AUDIT_LOG_DIR", str(audit_dir))
    monkeypatch.setenv("AMA_MASTER_SEED", "ab" * 64)
    monkeypatch.setattr(auth, "_auth_key_manager", None)  # re-derive from our seed

    writer = SecureAuditLogger(log_dir=audit_dir)
    try:
        writer.log_authentication(action="login", actor="acct-cli-1", outcome="success")
        writer.log_authentication(action="login", actor="acct-cli-1", outcome="failure")
        writer.flush()
    finally:
        writer.shutdown()

    code, out = _invoke("platform", "audit", "verify")
    assert code == 0
    assert json.loads(out)["ok"] is True

    log_path = audit_dir / "audit.jsonl"
    lines = log_path.read_text().splitlines()
    lines[0] = lines[0].replace('"success"', '"failure"')
    log_path.write_text("\n".join(lines) + "\n")

    code, out = _invoke("platform", "audit", "verify")
    assert code == 1
    assert json.loads(out)["ok"] is False


def test_audit_verify_refuses_without_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """audit verify refuses clearly when MERCURY_AUDIT_LOG_DIR is unset."""
    monkeypatch.delenv("MERCURY_AUDIT_LOG_DIR", raising=False)
    result = CliRunner().invoke(main, ["platform", "audit", "verify"])
    assert result.exit_code == 1
    assert "MERCURY_AUDIT_LOG_DIR" in result.output
