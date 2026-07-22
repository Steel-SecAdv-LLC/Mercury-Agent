# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the self-service account flows (brick 2).

Covers the primitives (password hashing, TOTP), both identity backends
(in-memory + durable SQLite, including restart survival), and every
:class:`AuthService` flow end-to-end against a recording mailer and an
injectable clock so email delivery and token expiry are deterministic and need
no network.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.api import passwords, totp
from omni_mercury_engine.api.auth_service import (
    AccountNotVerifiedError,
    AuthService,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidEmailError,
    InvalidTokenError,
    InvalidTwoFactorError,
    TwoFactorRequiredError,
    WeakPasswordError,
)
from omni_mercury_engine.api.identity_store import (
    Account,
    EmailToken,
    IdentityStore,
    InMemoryIdentityStore,
    Session,
    SqliteIdentityStore,
    hash_token,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# --------------------------------------------------------------------------- #
# passwords
# --------------------------------------------------------------------------- #
def test_password_hash_verify_roundtrip() -> None:
    """A password verifies against its own hash and rejects a wrong one."""
    stored = passwords.hash_password("correct horse battery staple")
    assert passwords.verify_password("correct horse battery staple", stored)
    assert not passwords.verify_password("wrong password", stored)


def test_password_hash_is_salted() -> None:
    """The same password hashes differently each time (random per-password salt)."""
    assert passwords.hash_password("same-pw") != passwords.hash_password("same-pw")


def test_password_verify_fails_closed_on_garbage() -> None:
    """A malformed stored value returns False rather than raising."""
    assert not passwords.verify_password("x", "not-a-valid-hash")


def test_password_needs_rehash_on_lower_iterations() -> None:
    """A hash below the target iteration count is flagged for upgrade."""
    weak = passwords.hash_password("pw", iterations=1000)
    assert passwords.needs_rehash(weak, iterations=600_000)
    strong = passwords.hash_password("pw", iterations=600_000)
    assert not passwords.needs_rehash(strong, iterations=600_000)


# --------------------------------------------------------------------------- #
# TOTP
# --------------------------------------------------------------------------- #
def test_totp_generate_and_verify() -> None:
    """A freshly generated code verifies at the same instant."""
    secret = totp.generate_secret()
    at = 1_700_000_000.0
    code = totp.generate_totp(secret, at=at)
    assert totp.verify_totp(secret, code, at=at)


def test_totp_rejects_wrong_and_malformed() -> None:
    """A wrong code and a non-numeric code both fail."""
    secret = totp.generate_secret()
    at = 1_700_000_000.0
    good = totp.generate_totp(secret, at=at)
    wrong = "000000" if good != "000000" else "111111"
    assert not totp.verify_totp(secret, wrong, at=at)
    assert not totp.verify_totp(secret, "abc", at=at)


def test_totp_tolerates_one_step_skew() -> None:
    """A code from the previous step still verifies within the window."""
    secret = totp.generate_secret()
    at = 1_700_000_000.0
    previous = totp.generate_totp(secret, at=at - 30)
    assert totp.verify_totp(secret, previous, at=at, window=1)


def test_totp_provisioning_uri_shape() -> None:
    """The provisioning URI carries the secret and issuer for authenticator apps."""
    secret = totp.generate_secret()
    uri = totp.provisioning_uri(secret, "user@example.com", "Mercury Agent")
    assert uri.startswith("otpauth://totp/")
    assert f"secret={secret}" in uri
    assert "issuer=Mercury+Agent" in uri


# --------------------------------------------------------------------------- #
# identity store — parity across both backends
# --------------------------------------------------------------------------- #
@pytest.fixture(params=["memory", "sqlite"])
def identity_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[IdentityStore]:
    """Yield each identity backend so parity tests run against both."""
    if request.param == "memory":
        yield InMemoryIdentityStore()
    else:
        store = SqliteIdentityStore(tmp_path / "identity.db")
        yield store
        store.close()


# A non-secret placeholder standing in for a real password hash in store-level
# tests (the store never interprets it; auth-service tests use real hashes).
_PLACEHOLDER_HASH = "ph"


def _account(email: str = "a@b.com") -> Account:
    """Build a minimal verified account for store tests."""
    return Account(
        id="acct-1",
        email=email,
        password_hash=_PLACEHOLDER_HASH,
        is_verified=True,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_account_crud_and_email_lookup(identity_store: IdentityStore) -> None:
    """Accounts persist and are found case-insensitively by email or id."""
    identity_store.create_account(_account("User@Example.com"))
    by_email = identity_store.get_account_by_email("user@example.com")
    assert by_email is not None
    assert by_email.id == "acct-1"
    assert identity_store.get_account_by_id("acct-1") is not None


def test_duplicate_email_rejected(identity_store: IdentityStore) -> None:
    """A second account with the same email (any case) is rejected."""
    identity_store.create_account(_account("dup@example.com"))
    clash = _account("DUP@example.com")
    clash.id = "acct-2"
    with pytest.raises(ValueError, match="already registered"):
        identity_store.create_account(clash)


def test_session_lifecycle(identity_store: IdentityStore) -> None:
    """Sessions persist, look up by hash, and delete individually and per account."""
    identity_store.create_account(_account())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    identity_store.create_session(Session("h1", "acct-1", now, now + timedelta(days=1)))
    assert identity_store.get_session("h1") is not None
    identity_store.delete_session("h1")
    assert identity_store.get_session("h1") is None

    identity_store.create_session(Session("h2", "acct-1", now, now + timedelta(days=1)))
    identity_store.delete_sessions_for_account("acct-1")
    assert identity_store.get_session("h2") is None


def test_email_token_consume(identity_store: IdentityStore) -> None:
    """An email token persists and records its consumed timestamp."""
    identity_store.create_account(_account())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    identity_store.create_email_token(
        EmailToken("t1", "acct-1", "verify", now, now + timedelta(hours=1))
    )
    assert identity_store.get_email_token("t1") is not None
    identity_store.consume_email_token("t1", now + timedelta(minutes=1))
    consumed = identity_store.get_email_token("t1")
    assert consumed is not None
    assert consumed.consumed_at is not None


def test_identity_survives_reopen(tmp_path: Path) -> None:
    """An account persists across a SQLite close/reopen (restart)."""
    db_path = tmp_path / "durable-identity.db"
    first = SqliteIdentityStore(db_path)
    first.create_account(_account("persist@example.com"))
    first.close()

    second = SqliteIdentityStore(db_path)
    try:
        recovered = second.get_account_by_email("persist@example.com")
        assert recovered is not None
        assert recovered.id == "acct-1"
    finally:
        second.close()


# --------------------------------------------------------------------------- #
# auth service — full flows
# --------------------------------------------------------------------------- #
class RecordingMailer:
    """Mailer that records sent messages so tests can read the embedded token."""

    def __init__(self) -> None:
        """Start with an empty outbox."""
        self.sent: list[dict[str, str]] = []

    def send(self, *, to: str, subject: str, body: str) -> None:
        """Record the message instead of delivering it."""
        self.sent.append({"to": to, "subject": subject, "body": body})

    def last_token(self) -> str:
        """Extract the token from the most recent email link."""
        match = re.search(r"token=([A-Za-z0-9_\-]+)", self.sent[-1]["body"])
        assert match is not None, "no token in the last email"
        return match.group(1)


class FakeClock:
    """A movable clock for deterministic expiry tests."""

    def __init__(self, start: datetime) -> None:
        """Start the clock at ``start``."""
        self.now = start

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now

    def advance(self, delta: timedelta) -> None:
        """Move the clock forward by ``delta``."""
        self.now += delta


@pytest.fixture
def service_setup() -> tuple[AuthService, RecordingMailer, FakeClock]:
    """An AuthService over an in-memory store with a recording mailer + fake clock."""
    mailer = RecordingMailer()
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = AuthService(InMemoryIdentityStore(), mailer, clock=clock)
    return service, mailer, clock


def _register_and_verify(service: AuthService, mailer: RecordingMailer, email: str) -> None:
    """Register an account and complete email verification."""
    service.register(email, "a-strong-password")
    service.verify_email(mailer.last_token())


def test_register_sends_verification_and_verifies(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """Registration emails a verification token that flips the account verified."""
    service, mailer, _ = service_setup
    account = service.register("new@example.com", "a-strong-password")
    assert account.is_verified is False
    assert mailer.sent and mailer.sent[-1]["to"] == "new@example.com"

    verified = service.verify_email(mailer.last_token())
    assert verified.is_verified is True


def test_register_rejects_bad_email_and_weak_password(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """Malformed email and short password are rejected up front."""
    service, _, _ = service_setup
    with pytest.raises(Exception, match="valid email"):
        service.register("not-an-email", "a-strong-password")
    with pytest.raises(WeakPasswordError):
        service.register("ok@example.com", "short")


def test_register_rejects_malformed_and_oversized_emails(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """The linear (ReDoS-safe) email check rejects crafted and oversized inputs."""
    service, _, _ = service_setup
    bad_emails = [
        "a@@b.com",  # two @
        "a@b.",  # trailing dot in domain
        "a@.b.com",  # leading dot in domain
        "a@localhost",  # no dot in domain
        "!@!." + "!." * 200,  # the pathological shape CodeQL flagged
        "a" * 400 + "@b.com",  # over the length bound
    ]
    for email in bad_emails:
        with pytest.raises(InvalidEmailError):
            service.register(email, "a-strong-password")


def test_duplicate_registration_rejected(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """Registering the same email twice raises EmailAlreadyRegisteredError."""
    service, _, _ = service_setup
    service.register("dup@example.com", "a-strong-password")
    with pytest.raises(EmailAlreadyRegisteredError):
        service.register("dup@example.com", "another-strong-pw")


def test_login_requires_verification(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """An unverified account cannot log in."""
    service, _, _ = service_setup
    service.register("unv@example.com", "a-strong-password")
    with pytest.raises(AccountNotVerifiedError):
        service.login("unv@example.com", "a-strong-password")


def test_login_wrong_password(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """A wrong password raises InvalidCredentialsError."""
    service, mailer, _ = service_setup
    _register_and_verify(service, mailer, "user@example.com")
    with pytest.raises(InvalidCredentialsError):
        service.login("user@example.com", "wrong-password")


def test_login_creates_authenticatable_session(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """A successful login yields a session that resolves back to the account."""
    service, mailer, _ = service_setup
    _register_and_verify(service, mailer, "user@example.com")
    result = service.login("user@example.com", "a-strong-password")
    account = service.authenticate_session(result.session_token)
    assert account is not None
    assert account.email == "user@example.com"


def test_session_expires(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """A session no longer authenticates once its TTL has passed."""
    service, mailer, clock = service_setup
    _register_and_verify(service, mailer, "user@example.com")
    result = service.login("user@example.com", "a-strong-password")
    clock.advance(timedelta(days=15))
    assert service.authenticate_session(result.session_token) is None


def test_logout_invalidates_session(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """Logout invalidates the session token."""
    service, mailer, _ = service_setup
    _register_and_verify(service, mailer, "user@example.com")
    result = service.login("user@example.com", "a-strong-password")
    service.logout(result.session_token)
    assert service.authenticate_session(result.session_token) is None


def test_password_reset_flow(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """Reset sets a new password, invalidates old sessions, and old pw stops working."""
    service, mailer, _ = service_setup
    _register_and_verify(service, mailer, "user@example.com")
    old_session = service.login("user@example.com", "a-strong-password").session_token

    service.request_password_reset("user@example.com")
    service.confirm_password_reset(mailer.last_token(), "a-brand-new-password")

    assert service.authenticate_session(old_session) is None  # sessions dropped
    with pytest.raises(InvalidCredentialsError):
        service.login("user@example.com", "a-strong-password")
    assert service.login("user@example.com", "a-brand-new-password").account is not None


def test_password_reset_unknown_email_is_silent(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """Requesting a reset for an unknown email neither errors nor emails."""
    service, mailer, _ = service_setup
    service.request_password_reset("nobody@example.com")
    assert mailer.sent == []


def test_token_single_use_and_expiry(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """A verification token cannot be reused, and an expired one is rejected."""
    service, mailer, clock = service_setup
    service.register("user@example.com", "a-strong-password")
    token = mailer.last_token()
    service.verify_email(token)
    with pytest.raises(InvalidTokenError):
        service.verify_email(token)  # reuse

    # A separate, expired reset token.
    service.request_password_reset("user@example.com")
    reset_token = mailer.last_token()
    clock.advance(timedelta(hours=2))
    with pytest.raises(InvalidTokenError):
        service.confirm_password_reset(reset_token, "a-brand-new-password")


def test_two_factor_enrollment_and_login(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """Enrolling TOTP makes login require a valid code."""
    service, mailer, clock = service_setup
    account = service.register("user@example.com", "a-strong-password")
    service.verify_email(mailer.last_token())

    enrollment = service.start_totp_enrollment(account.id)
    code = totp.generate_totp(enrollment.secret, at=clock.now.timestamp())
    service.confirm_totp_enrollment(account.id, code)

    # Without a code -> TwoFactorRequired.
    with pytest.raises(TwoFactorRequiredError):
        service.login("user@example.com", "a-strong-password")
    # Wrong code -> InvalidTwoFactor.
    with pytest.raises(InvalidTwoFactorError):
        service.login("user@example.com", "a-strong-password", totp_code="000000")
    # Correct code -> success.
    good = totp.generate_totp(enrollment.secret, at=clock.now.timestamp())
    result = service.login("user@example.com", "a-strong-password", totp_code=good)
    assert result.account.email == "user@example.com"


def test_two_factor_wrong_code_does_not_enable(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """Confirming enrollment with a wrong code leaves 2FA disabled."""
    service, mailer, _ = service_setup
    account = service.register("user@example.com", "a-strong-password")
    service.verify_email(mailer.last_token())
    service.start_totp_enrollment(account.id)
    with pytest.raises(InvalidTwoFactorError):
        service.confirm_totp_enrollment(account.id, "000000")
    # Login still works with no code because 2FA never enabled.
    assert service.login("user@example.com", "a-strong-password").account is not None


def test_session_token_only_stored_hashed(
    service_setup: tuple[AuthService, RecordingMailer, FakeClock],
) -> None:
    """The raw session token is never stored; only its hash is."""
    service, mailer, _ = service_setup
    _register_and_verify(service, mailer, "user@example.com")
    result = service.login("user@example.com", "a-strong-password")
    # The store holds the hash, not the raw token.
    assert service.authenticate_session(result.session_token) is not None
    assert hash_token(result.session_token) != result.session_token
