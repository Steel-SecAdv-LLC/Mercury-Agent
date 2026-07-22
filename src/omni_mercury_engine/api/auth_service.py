# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-service account flows: register, verify, login, reset, 2FA.

The :class:`AuthService` ties the durable :mod:`identity_store`, the
:mod:`passwords` hashing, the :mod:`totp` second factor, and the :mod:`mailer`
seam into the flows a browser needs:

* **register** → creates an unverified account and emails a verification token.
* **verify_email** → consumes the token and marks the account verified.
* **login** → checks password (and TOTP when enabled), issues a session token.
* **authenticate_session** → resolves a session cookie back to an account.
* **request_password_reset / confirm_password_reset** → emailed, single-use,
  expiring reset (no account-enumeration: an unknown email succeeds silently).
* **TOTP enrollment** → enrol / confirm / disable an authenticator second factor.

Raw session and email tokens are returned to the *caller* (the route layer)
exactly once so it can set a cookie or send an email; only their hashes are ever
persisted. Every failure is a typed :class:`AuthError` subclass so the route
layer can map each to the right HTTP status without string-matching.
"""

from __future__ import annotations

import logging
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from omni_mercury_engine.api import passwords, totp
from omni_mercury_engine.api.identity_store import (
    Account,
    DuplicateEmailError,
    EmailToken,
    Session,
    build_identity_store,
    hash_token,
)
from omni_mercury_engine.api.mailer import build_mailer

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.api.identity_store import IdentityStore
    from omni_mercury_engine.api.mailer import Mailer

logger = logging.getLogger(__name__)

__all__ = [
    "AccountDisabledError",
    "AccountNotVerifiedError",
    "AuthError",
    "AuthService",
    "EmailAlreadyRegisteredError",
    "EnrollmentResult",
    "InvalidCredentialsError",
    "InvalidEmailError",
    "InvalidTokenError",
    "InvalidTwoFactorError",
    "LoginResult",
    "TwoFactorRequiredError",
    "WeakPasswordError",
    "build_auth_service",
]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PASSWORD_LENGTH = 8


class AuthError(Exception):
    """Base class for every auth-flow failure."""


class InvalidEmailError(AuthError):
    """The supplied email is not a valid address."""


class WeakPasswordError(AuthError):
    """The supplied password does not meet the minimum policy."""


class EmailAlreadyRegisteredError(AuthError):
    """Registration attempted with an email that already exists."""


class InvalidCredentialsError(AuthError):
    """Email/password did not match an account."""


class AccountNotVerifiedError(AuthError):
    """Login attempted before the account's email was verified."""


class AccountDisabledError(AuthError):
    """Login attempted on a deactivated account."""


class TwoFactorRequiredError(AuthError):
    """A valid TOTP code is required but was not supplied."""


class InvalidTwoFactorError(AuthError):
    """The supplied TOTP code was wrong."""


class InvalidTokenError(AuthError):
    """An email token was unknown, expired, already used, or wrong-purpose."""


class LoginResult:
    """The outcome of a successful login: the account and its raw session token."""

    def __init__(self, account: Account, session_token: str, expires_at: datetime) -> None:
        """Store the authenticated account and its freshly minted session token."""
        self.account = account
        self.session_token = session_token
        self.expires_at = expires_at


class EnrollmentResult:
    """The outcome of starting TOTP enrollment: the secret and its QR URI."""

    def __init__(self, secret: str, provisioning_uri: str) -> None:
        """Store the enrollment secret and the authenticator provisioning URI."""
        self.secret = secret
        self.provisioning_uri = provisioning_uri


class AuthService:
    """Coordinates the account lifecycle over a store, a mailer, and 2FA."""

    def __init__(
        self,
        store: IdentityStore,
        mailer: Mailer,
        *,
        issuer: str = "Mercury Agent",
        base_url: str = "https://mercuryagent.global",
        verify_ttl: timedelta = timedelta(hours=24),
        reset_ttl: timedelta = timedelta(hours=1),
        session_ttl: timedelta = timedelta(days=14),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Wire the service to its collaborators and lifetime policy.

        Args:
            store: Durable identity backend.
            mailer: Transactional email sender.
            issuer: Service name shown in authenticator apps and emails.
            base_url: Public base URL used to build email links.
            verify_ttl: Lifetime of an email-verification token.
            reset_ttl: Lifetime of a password-reset token.
            session_ttl: Lifetime of a browser session.
            clock: Injectable time source (defaults to ``datetime.now(UTC)``);
                overridden in tests for deterministic expiry.
        """
        self._store = store
        self._mailer = mailer
        self._issuer = issuer
        self._base_url = base_url.rstrip("/")
        self._verify_ttl = verify_ttl
        self._reset_ttl = reset_ttl
        self._session_ttl = session_ttl
        self._clock = clock or (lambda: datetime.now(UTC))

    # -- registration + verification ------------------------------------- #
    def register(self, email: str, password: str) -> Account:
        """Create an unverified account and email a verification token.

        Args:
            email: The user's email address.
            password: The chosen plaintext password.

        Returns:
            The created (unverified) account.

        Raises:
            InvalidEmailError: If ``email`` is malformed.
            WeakPasswordError: If ``password`` is shorter than the minimum.
            EmailAlreadyRegisteredError: If the email is already registered.
        """
        if not _EMAIL_RE.match(email.strip()):
            raise InvalidEmailError("enter a valid email address")
        if len(password) < _MIN_PASSWORD_LENGTH:
            raise WeakPasswordError(f"password must be at least {_MIN_PASSWORD_LENGTH} characters")
        account = Account(
            id=secrets.token_hex(16),
            email=email.strip(),
            password_hash=passwords.hash_password(password),
            is_verified=False,
            is_active=True,
            created_at=self._clock(),
        )
        try:
            self._store.create_account(account)
        except DuplicateEmailError as exc:
            raise EmailAlreadyRegisteredError("email already registered") from exc

        raw_token = self._issue_email_token(account.id, "verify", self._verify_ttl)
        link = f"{self._base_url}/verify-email?token={raw_token}"
        self._mailer.send(
            to=account.email,
            subject=f"Verify your {self._issuer} account",
            body=(
                f"Welcome to {self._issuer}.\n\n"
                f"Confirm your email to activate your account:\n{link}\n\n"
                f"This link expires in {int(self._verify_ttl.total_seconds() // 3600)} hours. "
                "If you did not sign up, ignore this message."
            ),
        )
        return account

    def verify_email(self, raw_token: str) -> Account:
        """Consume a verification token and mark the account verified.

        Args:
            raw_token: The token from the verification email link.

        Returns:
            The now-verified account.

        Raises:
            InvalidTokenError: If the token is unknown, expired, used, or not a
                verification token.
        """
        token = self._consume_valid_token(raw_token, purpose="verify")
        account = self._require_account(token.account_id)
        account.is_verified = True
        self._store.update_account(account)
        return account

    # -- login + sessions ------------------------------------------------- #
    def login(self, email: str, password: str, totp_code: str | None = None) -> LoginResult:
        """Authenticate credentials (and TOTP when enabled) and open a session.

        Args:
            email: The account email.
            password: The plaintext password.
            totp_code: The current authenticator code, required when the account
                has 2FA enabled.

        Returns:
            A :class:`LoginResult` carrying the account and raw session token.

        Raises:
            InvalidCredentialsError: On unknown email or wrong password.
            AccountDisabledError: If the account is deactivated.
            AccountNotVerifiedError: If the email is not yet verified.
            TwoFactorRequiredError: If 2FA is enabled but no code was supplied.
            InvalidTwoFactorError: If the supplied 2FA code is wrong.
        """
        account = self._store.get_account_by_email(email)
        if account is None or not passwords.verify_password(password, account.password_hash):
            raise InvalidCredentialsError("invalid email or password")
        if not account.is_active:
            raise AccountDisabledError("account is disabled")
        if not account.is_verified:
            raise AccountNotVerifiedError("email not verified")
        if account.totp_enabled and account.totp_secret is not None:
            if not totp_code:
                raise TwoFactorRequiredError("two-factor code required")
            if not totp.verify_totp(account.totp_secret, totp_code, at=self._clock().timestamp()):
                raise InvalidTwoFactorError("invalid two-factor code")

        # Transparent work-factor upgrade on successful password check.
        if passwords.needs_rehash(account.password_hash):
            account.password_hash = passwords.hash_password(password)
            self._store.update_account(account)

        now = self._clock()
        expires_at = now + self._session_ttl
        raw_session = secrets.token_urlsafe(32)
        self._store.create_session(
            Session(
                token_hash=hash_token(raw_session),
                account_id=account.id,
                created_at=now,
                expires_at=expires_at,
            )
        )
        return LoginResult(account=account, session_token=raw_session, expires_at=expires_at)

    def authenticate_session(self, raw_session: str) -> Account | None:
        """Resolve a session token to its active account, or ``None``.

        Args:
            raw_session: The raw session token from the browser cookie.

        Returns:
            The account if the session exists, is unexpired, and the account is
            active; otherwise ``None`` (an expired session is also deleted).
        """
        session = self._store.get_session(hash_token(raw_session))
        if session is None:
            return None
        if session.expires_at <= self._clock():
            self._store.delete_session(session.token_hash)
            return None
        account = self._store.get_account_by_id(session.account_id)
        if account is None or not account.is_active:
            return None
        return account

    def logout(self, raw_session: str) -> None:
        """Invalidate a single session (idempotent)."""
        self._store.delete_session(hash_token(raw_session))

    # -- password reset --------------------------------------------------- #
    def request_password_reset(self, email: str) -> None:
        """Email a reset link if the account exists; succeed silently otherwise.

        The silent success on an unknown email prevents account enumeration.

        Args:
            email: The email requesting a reset.
        """
        account = self._store.get_account_by_email(email)
        if account is None:
            logger.info("password reset requested for unknown email; no-op")
            return
        raw_token = self._issue_email_token(account.id, "reset", self._reset_ttl)
        link = f"{self._base_url}/reset-password?token={raw_token}"
        self._mailer.send(
            to=account.email,
            subject=f"Reset your {self._issuer} password",
            body=(
                f"A password reset was requested for your {self._issuer} account.\n\n"
                f"Reset it here:\n{link}\n\n"
                f"This link expires in {int(self._reset_ttl.total_seconds() // 60)} minutes. "
                "If you did not request this, ignore this message; your password is unchanged."
            ),
        )

    def confirm_password_reset(self, raw_token: str, new_password: str) -> Account:
        """Consume a reset token, set the new password, and drop all sessions.

        Args:
            raw_token: The token from the reset email.
            new_password: The new plaintext password.

        Returns:
            The updated account.

        Raises:
            InvalidTokenError: If the token is unknown, expired, used, or not a
                reset token.
            WeakPasswordError: If the new password is too short.
        """
        if len(new_password) < _MIN_PASSWORD_LENGTH:
            raise WeakPasswordError(f"password must be at least {_MIN_PASSWORD_LENGTH} characters")
        token = self._consume_valid_token(raw_token, purpose="reset")
        account = self._require_account(token.account_id)
        account.password_hash = passwords.hash_password(new_password)
        self._store.update_account(account)
        # A reset invalidates every existing session (a compromised session
        # must not survive the credential change).
        self._store.delete_sessions_for_account(account.id)
        return account

    # -- two-factor (TOTP) ------------------------------------------------ #
    def start_totp_enrollment(self, account_id: str) -> EnrollmentResult:
        """Generate and store a TOTP secret (disabled until confirmed).

        Args:
            account_id: The account enrolling a second factor.

        Returns:
            The secret and the ``otpauth://`` provisioning URI for the QR code.

        Raises:
            InvalidTokenError: If the account does not exist.
        """
        account = self._require_account(account_id)
        secret = totp.generate_secret()
        account.totp_secret = secret
        account.totp_enabled = False
        self._store.update_account(account)
        uri = totp.provisioning_uri(secret, account.email, self._issuer)
        return EnrollmentResult(secret=secret, provisioning_uri=uri)

    def confirm_totp_enrollment(self, account_id: str, code: str) -> None:
        """Enable 2FA once the user proves possession with a valid code.

        Args:
            account_id: The enrolling account.
            code: The current authenticator code.

        Raises:
            InvalidTokenError: If the account does not exist or has no pending
                enrollment.
            InvalidTwoFactorError: If ``code`` does not match the secret.
        """
        account = self._require_account(account_id)
        if account.totp_secret is None:
            raise InvalidTokenError("no TOTP enrollment in progress")
        if not totp.verify_totp(account.totp_secret, code, at=self._clock().timestamp()):
            raise InvalidTwoFactorError("invalid two-factor code")
        account.totp_enabled = True
        self._store.update_account(account)

    def disable_totp(self, account_id: str) -> None:
        """Turn off 2FA and clear the stored secret."""
        account = self._require_account(account_id)
        account.totp_secret = None
        account.totp_enabled = False
        self._store.update_account(account)

    # -- internals -------------------------------------------------------- #
    def _issue_email_token(self, account_id: str, purpose: str, ttl: timedelta) -> str:
        """Create and persist an email token; return its raw value."""
        raw_token = secrets.token_urlsafe(32)
        now = self._clock()
        self._store.create_email_token(
            EmailToken(
                token_hash=hash_token(raw_token),
                account_id=account_id,
                purpose=purpose,
                created_at=now,
                expires_at=now + ttl,
            )
        )
        return raw_token

    def _consume_valid_token(self, raw_token: str, *, purpose: str) -> EmailToken:
        """Validate an email token for ``purpose`` and mark it consumed."""
        token = self._store.get_email_token(hash_token(raw_token))
        if token is None or token.purpose != purpose:
            raise InvalidTokenError("invalid or unknown token")
        if token.consumed_at is not None:
            raise InvalidTokenError("token already used")
        if token.expires_at <= self._clock():
            raise InvalidTokenError("token expired")
        self._store.consume_email_token(token.token_hash, self._clock())
        return token

    def _require_account(self, account_id: str) -> Account:
        """Fetch an account by id or raise :class:`InvalidTokenError`."""
        account = self._store.get_account_by_id(account_id)
        if account is None:
            raise InvalidTokenError("account no longer exists")
        return account


def build_auth_service() -> AuthService:
    """Construct an :class:`AuthService` wired from the environment.

    Uses :func:`build_identity_store` (durable when ``MERCURY_KEYSTORE_PATH`` is
    set) and :func:`build_mailer` (real SMTP when ``MERCURY_SMTP_HOST`` is set),
    plus ``MERCURY_PUBLIC_BASE_URL`` for email links when provided.

    Returns:
        A ready-to-use auth service.
    """
    import os

    base_url = os.getenv("MERCURY_PUBLIC_BASE_URL", "https://mercuryagent.global")
    return AuthService(build_identity_store(), build_mailer(), base_url=base_url)
