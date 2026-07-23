# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-service account flows: register, verify, login, reset, 2FA, lifecycle.

The :class:`AuthService` ties the durable :mod:`identity_store`, the
:mod:`passwords` hashing, the :mod:`totp` second factor, the
:mod:`secret_sealer` at-rest encryption, the :mod:`mailer` seam, and the
:mod:`auth_audit` trail into the flows a browser needs:

* **register** → creates an unverified account and emails a verification token.
* **verify_email / resend_verification** → consume or re-issue that token.
* **login** → checks password (and TOTP or a recovery code when 2FA is on),
  issues a session token + CSRF token. Unknown emails burn the same KDF cost
  as real ones so timing does not enumerate accounts.
* **resolve_session** → cookie → (account, session), enforcing both the
  absolute session lifetime and the idle timeout.
* **request_password_reset / confirm_password_reset** → emailed, single-use,
  expiring reset (no account-enumeration: an unknown email succeeds silently).
* **TOTP enrollment** → enrol / confirm / disable; the stored secret is sealed
  at rest (AES-256-GCM via SecureDataHandler, AAD-bound to the account),
  accepted codes are replay-checked against the last used time step, and
  confirmation issues single-use recovery codes.
* **Lifecycle** → change password (session-rotating), change email with
  re-verification of the new address, account deletion, and data export.

Raw session/CSRF/email tokens and recovery codes are returned to the *caller*
(the route layer) exactly once; only their hashes are ever persisted. Every
failure is a typed :class:`AuthError` subclass so the route layer can map each
to the right HTTP status without string-matching. Email delivery is
**best-effort by contract**: state changes commit first, and a mailer failure
is logged and audited but never unwinds them (see :meth:`AuthService._send`).
"""

from __future__ import annotations

import logging
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from omni_mercury_engine.api import email_templates, passwords, platform_metrics, totp
from omni_mercury_engine.api.auth_audit import AuthAuditor, build_auth_auditor
from omni_mercury_engine.api.identity_store import (
    Account,
    DuplicateEmailError,
    EmailToken,
    Session,
    build_identity_store,
    hash_token,
    identity_store_is_durable,
)
from omni_mercury_engine.api.mailer import build_mailer
from omni_mercury_engine.api.secret_sealer import (
    SealedSecretError,
    SecretSealer,
    build_secret_sealer,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Executor

    from omni_mercury_engine.api.email_templates import EmailContent
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

# Linear (backtracking-free) email shape check. The two ``[^@\s]+`` runs are
# separated by a literal ``@`` that neither can match, so there is exactly one
# possible split and no polynomial backtracking (the previous
# ``...@[^@\s]+\.[^@\s]+`` overlapped the dot with the surrounding runs, a ReDoS
# on adversarial input). The required dot in the domain is checked separately.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+$")
_MAX_EMAIL_LENGTH = 320  # RFC 5321 max; also bounds the shape check up front.
_MIN_PASSWORD_LENGTH = 8

#: How many single-use 2FA recovery codes an enrollment issues.
_RECOVERY_CODE_COUNT = 10
#: Recovery-code entropy in bytes (64 bits — single-use + rate-limited online).
_RECOVERY_CODE_BYTES = 8

# Hash burned on login attempts against unknown emails, so the unknown-email
# path costs the same KDF work as a wrong-password attempt and response
# timing does not enumerate accounts. Built lazily once per process (the KDF
# is deliberately expensive; import must stay cheap).
_dummy_hash: str | None = None


def _get_dummy_hash() -> str:
    """Return (building once) the account-enumeration decoy password hash."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = passwords.hash_password(secrets.token_hex(16))
    return _dummy_hash


def _is_valid_email(email: str) -> bool:
    """Validate an email's shape in linear time (length-bounded, no backtracking).

    Args:
        email: The candidate address (already stripped by the caller).

    Returns:
        ``True`` if it has exactly one ``@``, non-empty local and domain parts,
        and a dot in the domain; ``False`` otherwise.
    """
    if len(email) > _MAX_EMAIL_LENGTH or not _EMAIL_RE.match(email):
        return False
    domain = email.rsplit("@", 1)[1]
    return "." in domain and not domain.startswith(".") and not domain.endswith(".")


def _normalize_recovery_code(code: str) -> str:
    """Canonicalise a user-typed recovery code (case/dash/space-insensitive)."""
    return code.strip().lower().replace("-", "").replace(" ", "")


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
    """The supplied TOTP code or recovery code was wrong."""


class InvalidTokenError(AuthError):
    """An email token was unknown, expired, already used, or wrong-purpose."""


class LoginResult:
    """The outcome of a successful login: account, session, and CSRF token."""

    def __init__(
        self,
        account: Account,
        session_token: str,
        csrf_token: str,
        expires_at: datetime,
        max_age_seconds: int,
        persistent: bool,
    ) -> None:
        """Store the authenticated account and its freshly issued tokens.

        Args:
            account: The authenticated account.
            session_token: Raw session token (cookie value; hash is stored).
            csrf_token: Raw CSRF token bound to this session (double-submit).
            expires_at: Absolute session expiry.
            max_age_seconds: Session lifetime — the correct cookie ``max_age``.
            persistent: Whether the cookie should survive the browser session
                (the remember-me choice).
        """
        self.account = account
        self.session_token = session_token
        self.csrf_token = csrf_token
        self.expires_at = expires_at
        self.max_age_seconds = max_age_seconds
        self.persistent = persistent


class EnrollmentResult:
    """The outcome of starting TOTP enrollment: the secret and its QR URI."""

    def __init__(self, secret: str, provisioning_uri: str) -> None:
        """Store the enrollment secret and the authenticator provisioning URI."""
        self.secret = secret
        self.provisioning_uri = provisioning_uri


class AuthService:
    """Coordinates the account lifecycle over a store, mailer, sealer, and 2FA."""

    def __init__(
        self,
        store: IdentityStore,
        mailer: Mailer,
        *,
        issuer: str = "Mercury Agent",
        base_url: str = "https://mercuryagent.global",
        contact: str = "steel.sa.llc@gmail.com",
        verify_ttl: timedelta = timedelta(hours=24),
        reset_ttl: timedelta = timedelta(hours=1),
        session_ttl: timedelta = timedelta(days=14),
        session_ttl_short: timedelta = timedelta(days=1),
        session_idle_timeout: timedelta = timedelta(days=1),
        clock: Callable[[], datetime] | None = None,
        sealer: SecretSealer | None = None,
        auditor: AuthAuditor | None = None,
        mail_executor: Executor | None = None,
    ) -> None:
        """Wire the service to its collaborators and lifetime policy.

        Args:
            store: Durable identity backend.
            mailer: Transactional email sender.
            issuer: Service name shown in authenticator apps and emails.
            base_url: Public base URL used to build email links.
            contact: Operator contact address (List-Unsubscribe target).
            verify_ttl: Lifetime of an email-verification token.
            reset_ttl: Lifetime of a password-reset token.
            session_ttl: Absolute lifetime of a remembered browser session.
            session_ttl_short: Absolute lifetime without remember-me.
            session_idle_timeout: Maximum gap between authenticated uses.
            clock: Injectable time source (defaults to ``datetime.now(UTC)``);
                overridden in tests for deterministic expiry.
            sealer: At-rest sealer for TOTP secrets; defaults to an ephemeral
                stable key (right for the in-memory store tests use).
            auditor: Auth-event audit sink; defaults to the logging fallback.
            mail_executor: When provided, email sends are submitted to it
                (fire-and-forget) so a slow SMTP server can neither block the
                request thread nor leak timing; ``None`` sends synchronously
                but still failure-tolerantly.
        """
        self._store = store
        self._mailer = mailer
        self._issuer = issuer
        self._base_url = base_url.rstrip("/")
        self._contact = contact
        self._verify_ttl = verify_ttl
        self._reset_ttl = reset_ttl
        self._session_ttl = session_ttl
        self._session_ttl_short = session_ttl_short
        self._session_idle_timeout = session_idle_timeout
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sealer = sealer or SecretSealer(secrets.token_bytes(32), key_is_stable=True)
        self._auditor = auditor or AuthAuditor(None)
        self._mail_executor = mail_executor

    # -- email delivery (best-effort by contract) -------------------------- #
    def _send(self, to: str, content: EmailContent) -> None:
        """Deliver one email without ever failing the calling flow.

        State changes (account rows, tokens) are committed by the caller
        *before* this runs; a mail failure is logged and audited so the
        operator sees delivery problems, and the user has resend/reset flows
        to recover. With an executor the send happens off-thread entirely.
        """

        def _deliver() -> None:
            try:
                self._mailer.send(
                    to=to,
                    subject=content.subject,
                    body=content.body,
                    html_body=content.html_body,
                    headers=content.headers,
                )
            except Exception:
                logger.exception("email delivery failed (subject=%r)", content.subject)
                platform_metrics.record_email("failed")
                self._auditor.record(
                    "email_delivery", outcome="failure", details={"subject": content.subject}
                )
            else:
                platform_metrics.record_email("sent")

        if self._mail_executor is not None:
            self._mail_executor.submit(_deliver)
        else:
            _deliver()

    # -- registration + verification --------------------------------------- #
    def register(self, email: str, password: str, client_ip: str | None = None) -> Account:
        """Create an unverified account and email a verification token.

        The account row is committed before any email I/O — a mail outage
        must not orphan the registration (the user can resend).

        Args:
            email: The user's email address.
            password: The chosen plaintext password.
            client_ip: Trusted-proxy-resolved caller address, for the audit trail.

        Returns:
            The created (unverified) account.

        Raises:
            InvalidEmailError: If ``email`` is malformed.
            WeakPasswordError: If ``password`` is shorter than the minimum.
            EmailAlreadyRegisteredError: If the email is already registered.
        """
        if not _is_valid_email(email.strip()):
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

        self._auditor.record(
            "register", outcome="success", account_id=account.id, client_ip=client_ip
        )
        self._email_verification_token(account)
        return account

    def _email_verification_token(self, account: Account) -> None:
        """Issue a fresh verification token and email it (best-effort)."""
        raw_token = self._issue_email_token(account.id, "verify", self._verify_ttl)
        link = f"{self._base_url}/verify-email?token={raw_token}"
        ttl_hours = int(self._verify_ttl.total_seconds() // 3600)
        self._send(
            account.email,
            email_templates.verification_email(self._issuer, link, ttl_hours, self._contact),
        )

    def resend_verification(self, email: str, client_ip: str | None = None) -> None:
        """Re-send the verification email if the account exists and needs it.

        Enumeration-safe: unknown addresses and already-verified accounts
        return silently, exactly like the reset-request flow.

        Args:
            email: The address asking for a fresh link.
            client_ip: Caller address for the audit trail.
        """
        account = self._store.get_account_by_email(email)
        if account is None or account.is_verified or not account.is_active:
            logger.info("verification resend requested; no action applicable")
            return
        # Outstanding verify tokens are superseded, not additive — one live
        # link at a time bounds the value of a mailbox compromise.
        self._store.delete_email_tokens_for_account(account.id, purpose="verify")
        self._auditor.record(
            "resend_verification", outcome="success", account_id=account.id, client_ip=client_ip
        )
        self._email_verification_token(account)

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
        self._auditor.record("verify_email", outcome="success", account_id=account.id)
        return account

    # -- login + sessions --------------------------------------------------- #
    def login(
        self,
        email: str,
        password: str,
        totp_code: str | None = None,
        recovery_code: str | None = None,
        *,
        remember_me: bool = True,
        client_ip: str | None = None,
    ) -> LoginResult:
        """Authenticate credentials (and a second factor when enabled).

        Args:
            email: The account email.
            password: The plaintext password.
            totp_code: Current authenticator code (when 2FA is enabled).
            recovery_code: A single-use backup code, accepted in place of
                ``totp_code`` when the authenticator is unavailable.
            remember_me: Chooses the long (persistent-cookie) or short
                session lifetime.
            client_ip: Trusted-proxy-resolved caller address for auditing.

        Returns:
            A :class:`LoginResult` carrying the account and raw tokens.

        Raises:
            InvalidCredentialsError: On unknown email or wrong password.
            AccountDisabledError: If the account is deactivated.
            AccountNotVerifiedError: If the email is not yet verified.
            TwoFactorRequiredError: If 2FA is enabled but no code was supplied.
            InvalidTwoFactorError: If the second factor is wrong or replayed.
        """
        account = self._store.get_account_by_email(email)
        if account is None:
            # Burn the same KDF cost as a real verification so the response
            # time of unknown-vs-wrong-password is indistinguishable.
            passwords.verify_password(password, _get_dummy_hash())
            self._auditor.record(
                "login", outcome="failure", client_ip=client_ip, details={"reason": "unknown_email"}
            )
            raise InvalidCredentialsError("invalid email or password")
        if not passwords.verify_password(password, account.password_hash):
            self._auditor.record(
                "login",
                outcome="failure",
                account_id=account.id,
                client_ip=client_ip,
                details={"reason": "bad_password"},
            )
            raise InvalidCredentialsError("invalid email or password")
        if not account.is_active:
            self._auditor.record(
                "login",
                outcome="failure",
                account_id=account.id,
                client_ip=client_ip,
                details={"reason": "disabled"},
            )
            raise AccountDisabledError("account is disabled")
        if not account.is_verified:
            self._auditor.record(
                "login",
                outcome="failure",
                account_id=account.id,
                client_ip=client_ip,
                details={"reason": "unverified"},
            )
            raise AccountNotVerifiedError("email not verified")
        if account.totp_enabled and account.totp_secret is not None:
            self._check_second_factor(account, totp_code, recovery_code, client_ip)

        # Transparent work-factor upgrade on successful password check.
        if passwords.needs_rehash(account.password_hash):
            account.password_hash = passwords.hash_password(password)
            self._store.update_account(account)

        result = self._open_session(account, remember_me=remember_me)
        self._auditor.record("login", outcome="success", account_id=account.id, client_ip=client_ip)
        return result

    def _check_second_factor(
        self,
        account: Account,
        totp_code: str | None,
        recovery_code: str | None,
        client_ip: str | None,
    ) -> None:
        """Enforce TOTP (with replay rejection) or consume a recovery code."""
        if recovery_code:
            normalized = _normalize_recovery_code(recovery_code)
            consumed = self._store.consume_recovery_code(
                account.id, hash_token(normalized), self._clock()
            )
            if not consumed:
                self._auditor.record(
                    "login",
                    outcome="failure",
                    account_id=account.id,
                    client_ip=client_ip,
                    details={"reason": "bad_recovery_code"},
                )
                raise InvalidTwoFactorError("invalid recovery code")
            remaining = self._store.count_unused_recovery_codes(account.id)
            self._auditor.record(
                "recovery_code_used",
                outcome="success",
                account_id=account.id,
                client_ip=client_ip,
                details={"remaining": remaining},
            )
            self._send(
                account.email,
                email_templates.recovery_codes_notice(self._issuer, remaining, self._contact),
            )
            return

        if not totp_code:
            raise TwoFactorRequiredError("two-factor code required")

        secret = self._open_totp_secret(account)
        step = totp.verify_totp_with_step(secret, totp_code, at=self._clock().timestamp())
        if step is None:
            self._auditor.record(
                "login",
                outcome="failure",
                account_id=account.id,
                client_ip=client_ip,
                details={"reason": "bad_totp"},
            )
            raise InvalidTwoFactorError("invalid two-factor code")
        if account.totp_last_step is not None and step <= account.totp_last_step:
            self._auditor.record(
                "login",
                outcome="failure",
                account_id=account.id,
                client_ip=client_ip,
                details={"reason": "totp_replay"},
            )
            raise InvalidTwoFactorError("two-factor code already used")
        account.totp_last_step = step
        self._store.update_account(account)

    def _open_totp_secret(self, account: Account) -> str:
        """Unseal the account's TOTP secret, failing closed on tamper."""
        assert account.totp_secret is not None
        try:
            return self._sealer.unseal(account.totp_secret, aad=account.id)
        except SealedSecretError:
            logger.error(
                "sealed TOTP secret failed to open for an account; "
                "denying the second factor (tamper or key mismatch)"
            )
            raise InvalidTwoFactorError("invalid two-factor code") from None

    def _store_totp_secret(self, account: Account, secret: str | None) -> None:
        """Seal (when the key is stable) and store a TOTP secret."""
        if secret is None:
            account.totp_secret = None
        elif self._sealer.key_is_stable:
            account.totp_secret = self._sealer.seal(secret, aad=account.id)
        else:
            # A durable store with no stable key: sealing would brick 2FA on
            # restart. build_secret_sealer already warned loudly.
            account.totp_secret = secret
        self._store.update_account(account)

    def _open_session(self, account: Account, *, remember_me: bool) -> LoginResult:
        """Create a fresh session + CSRF pair for ``account``."""
        now = self._clock()
        ttl = self._session_ttl if remember_me else self._session_ttl_short
        expires_at = now + ttl
        raw_session = secrets.token_urlsafe(32)
        raw_csrf = secrets.token_urlsafe(32)
        self._store.create_session(
            Session(
                token_hash=hash_token(raw_session),
                account_id=account.id,
                created_at=now,
                expires_at=expires_at,
                csrf_hash=hash_token(raw_csrf),
                last_seen_at=now,
            )
        )
        return LoginResult(
            account=account,
            session_token=raw_session,
            csrf_token=raw_csrf,
            expires_at=expires_at,
            max_age_seconds=int(ttl.total_seconds()),
            persistent=remember_me,
        )

    def resolve_session(self, raw_session: str) -> tuple[Account, Session] | None:
        """Resolve a session token to its account + session record.

        Enforces the absolute lifetime *and* the idle timeout, and stamps
        ``last_seen_at`` (throttled to once a minute so a busy dashboard does
        not turn every request into a write).

        Args:
            raw_session: The raw session token from the browser cookie.

        Returns:
            ``(account, session)`` when valid; ``None`` otherwise (expired or
            idle sessions are deleted as a side effect).
        """
        session = self._store.get_session(hash_token(raw_session))
        if session is None:
            return None
        now = self._clock()
        if session.expires_at <= now:
            self._store.delete_session(session.token_hash)
            return None
        last_seen = session.last_seen_at or session.created_at
        if now - last_seen > self._session_idle_timeout:
            self._store.delete_session(session.token_hash)
            return None
        account = self._store.get_account_by_id(session.account_id)
        if account is None or not account.is_active:
            return None
        if now - last_seen > timedelta(seconds=60):
            self._store.touch_session(session.token_hash, now)
        return account, session

    def authenticate_session(self, raw_session: str) -> Account | None:
        """Resolve a session token to its active account, or ``None``.

        Thin wrapper over :meth:`resolve_session` for callers that do not
        need the session record itself.
        """
        resolved = self.resolve_session(raw_session)
        return resolved[0] if resolved else None

    def get_account(self, account_id: str) -> Account | None:
        """Return the account with ``account_id``, or ``None``.

        A read-only accessor over the identity store for collaborators that
        hold an account id but not a session — e.g. the quota middleware
        resolving an API key's owning account to charge usage at that
        account's tier. Never raises (unlike the internal
        :meth:`_require_account`), so a best-effort caller can branch on
        ``None`` without a try/except.
        """
        return self._store.get_account_by_id(account_id)

    def record_event(
        self,
        action: str,
        *,
        outcome: str,
        account_id: str | None = None,
        client_ip: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Record a security-relevant account event on the shared audit trail.

        Exposes the service's single :class:`AuthAuditor` to route-layer
        collaborators that manage account-scoped credentials outside the core
        flows — notably API-key issuance and revocation — so those events land
        on the *same* tamper-evident trail as login/password/2FA events rather
        than a second, independent sink. Never raises (the auditor swallows
        sink failures by contract).

        Args:
            action: Stable event name (e.g. ``"api_key_created"``).
            outcome: ``"success"`` or ``"failure"``.
            account_id: The acted-on account.
            client_ip: Trusted-proxy-resolved caller address, when known.
            details: Small, non-PII extras (e.g. the affected ``key_id``).
        """
        self._auditor.record(
            action,
            outcome=outcome,
            account_id=account_id,
            client_ip=client_ip,
            details=details,
        )

    def verify_csrf(self, session: Session, csrf_token: str | None) -> bool:
        """Check a submitted CSRF token against the session's stored hash."""
        if not csrf_token or not session.csrf_hash:
            return False
        import hmac as _hmac

        return _hmac.compare_digest(session.csrf_hash, hash_token(csrf_token))

    def logout(self, raw_session: str) -> None:
        """Invalidate a single session (idempotent)."""
        self._store.delete_session(hash_token(raw_session))

    def _rotate_sessions(self, account: Account, *, remember_me: bool = True) -> LoginResult:
        """Drop every session for ``account`` and issue a fresh one.

        The rotation-on-privilege-change primitive: after a password or 2FA
        change, any token an attacker may hold dies, while the legitimate
        caller continues seamlessly on the returned fresh pair.
        """
        self._store.delete_sessions_for_account(account.id)
        return self._open_session(account, remember_me=remember_me)

    # -- password reset ------------------------------------------------------ #
    def request_password_reset(self, email: str, client_ip: str | None = None) -> None:
        """Email a reset link if the account exists; succeed silently otherwise.

        The silent success on an unknown email prevents account enumeration.

        Args:
            email: The email requesting a reset.
            client_ip: Caller address for the audit trail.
        """
        account = self._store.get_account_by_email(email)
        if account is None:
            logger.info("password reset requested for unknown email; no-op")
            return
        self._auditor.record(
            "password_reset_request", outcome="success", account_id=account.id, client_ip=client_ip
        )
        raw_token = self._issue_email_token(account.id, "reset", self._reset_ttl)
        link = f"{self._base_url}/reset-password?token={raw_token}"
        ttl_minutes = int(self._reset_ttl.total_seconds() // 60)
        self._send(
            account.email,
            email_templates.reset_email(self._issuer, link, ttl_minutes, self._contact),
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
        # A reset invalidates every existing session AND any other live reset
        # token (a compromised session or second emailed link must not survive
        # the credential change).
        self._store.delete_sessions_for_account(account.id)
        self._store.delete_email_tokens_for_account(account.id, purpose="reset")
        self._auditor.record("password_reset", outcome="success", account_id=account.id)
        return account

    def change_password(
        self,
        account_id: str,
        current_password: str,
        new_password: str,
        client_ip: str | None = None,
    ) -> LoginResult:
        """Change the password of a logged-in account (re-authenticated).

        Args:
            account_id: The authenticated account.
            current_password: The existing password (fresh proof of identity —
                a hijacked cookie alone must not be able to take the account).
            new_password: The replacement password.
            client_ip: Caller address for the audit trail.

        Returns:
            A fresh :class:`LoginResult` — every prior session is invalidated
            and the caller continues on the new one.

        Raises:
            InvalidCredentialsError: If ``current_password`` is wrong.
            WeakPasswordError: If the new password is too short.
        """
        account = self._require_account(account_id)
        if not passwords.verify_password(current_password, account.password_hash):
            self._auditor.record(
                "password_change",
                outcome="failure",
                account_id=account.id,
                client_ip=client_ip,
                details={"reason": "bad_current_password"},
            )
            raise InvalidCredentialsError("current password is incorrect")
        if len(new_password) < _MIN_PASSWORD_LENGTH:
            raise WeakPasswordError(f"password must be at least {_MIN_PASSWORD_LENGTH} characters")
        account.password_hash = passwords.hash_password(new_password)
        self._store.update_account(account)
        self._store.delete_email_tokens_for_account(account.id, purpose="reset")
        result = self._rotate_sessions(account)
        self._auditor.record(
            "password_change", outcome="success", account_id=account.id, client_ip=client_ip
        )
        return result

    # -- email change ---------------------------------------------------------- #
    def request_email_change(
        self,
        account_id: str,
        new_email: str,
        current_password: str,
        client_ip: str | None = None,
    ) -> None:
        """Start an email change: re-authenticate, then verify the NEW address.

        The account's address does not change until the new mailbox proves
        receipt of the confirmation link.

        Args:
            account_id: The authenticated account.
            new_email: The requested new address.
            current_password: Fresh proof of identity.
            client_ip: Caller address for the audit trail.

        Raises:
            InvalidCredentialsError: If the password is wrong.
            InvalidEmailError: If the new address is malformed.
            EmailAlreadyRegisteredError: If the new address is taken.
        """
        account = self._require_account(account_id)
        if not passwords.verify_password(current_password, account.password_hash):
            raise InvalidCredentialsError("current password is incorrect")
        candidate = new_email.strip()
        if not _is_valid_email(candidate):
            raise InvalidEmailError("enter a valid email address")
        existing = self._store.get_account_by_email(candidate)
        if existing is not None and existing.id != account.id:
            raise EmailAlreadyRegisteredError("email already registered")
        # One live change request at a time.
        self._store.delete_email_tokens_for_account(account.id, purpose="email_change")
        raw_token = self._issue_email_token(
            account.id, "email_change", self._verify_ttl, payload=candidate
        )
        link = f"{self._base_url}/confirm-email-change?token={raw_token}"
        ttl_hours = int(self._verify_ttl.total_seconds() // 3600)
        self._auditor.record(
            "email_change_request", outcome="success", account_id=account.id, client_ip=client_ip
        )
        self._send(
            candidate,
            email_templates.email_change_email(self._issuer, link, ttl_hours, self._contact),
        )

    def confirm_email_change(self, raw_token: str) -> Account:
        """Complete an email change from the link sent to the new address.

        All sessions are dropped (an address change is a privilege change);
        the user signs back in with the new email.

        Args:
            raw_token: The token from the confirmation email.

        Returns:
            The updated account.

        Raises:
            InvalidTokenError: If the token is invalid or its payload is gone.
            EmailAlreadyRegisteredError: If the address was claimed since the
                request (uniqueness is re-checked at commit time).
        """
        token = self._consume_valid_token(raw_token, purpose="email_change")
        account = self._require_account(token.account_id)
        if not token.payload or not _is_valid_email(token.payload):
            raise InvalidTokenError("invalid or unknown token")
        account.email = token.payload
        account.is_verified = True  # the new mailbox just proved receipt
        try:
            self._store.update_account(account)
        except DuplicateEmailError as exc:
            raise EmailAlreadyRegisteredError("email already registered") from exc
        self._store.delete_sessions_for_account(account.id)
        self._auditor.record("email_change", outcome="success", account_id=account.id)
        return account

    # -- deletion + export ------------------------------------------------------ #
    def delete_account(
        self, account_id: str, current_password: str, client_ip: str | None = None
    ) -> None:
        """Hard-delete an account after re-authentication.

        Sessions, email tokens, and recovery codes are removed with the
        account row. Usage-ledger rows are pseudonymous accounting facts and
        age out via the retention sweep instead (documented in the security
        notes).

        Args:
            account_id: The authenticated account.
            current_password: Fresh proof of identity.
            client_ip: Caller address for the audit trail.

        Raises:
            InvalidCredentialsError: If the password is wrong.
        """
        account = self._require_account(account_id)
        if not passwords.verify_password(current_password, account.password_hash):
            self._auditor.record(
                "account_delete",
                outcome="failure",
                account_id=account.id,
                client_ip=client_ip,
                details={"reason": "bad_current_password"},
            )
            raise InvalidCredentialsError("current password is incorrect")
        self._store.delete_account(account.id)
        self._auditor.record(
            "account_delete", outcome="success", account_id=account.id, client_ip=client_ip
        )

    def export_account_data(self, account_id: str) -> dict[str, object]:
        """Return the account's stored personal data (portability export).

        Secrets never leave: the password hash, sealed TOTP secret, and
        recovery-code hashes are structurally excluded.

        Args:
            account_id: The authenticated account.

        Returns:
            A JSON-safe mapping of the account's data.
        """
        account = self._require_account(account_id)
        return {
            "id": account.id,
            "email": account.email,
            "is_verified": account.is_verified,
            "is_active": account.is_active,
            "totp_enabled": account.totp_enabled,
            "tier": account.tier,
            "created_at": account.created_at.isoformat(),
            "unused_recovery_codes": self._store.count_unused_recovery_codes(account.id),
        }

    # -- two-factor (TOTP) ------------------------------------------------------- #
    def start_totp_enrollment(self, account_id: str) -> EnrollmentResult:
        """Generate and store a (sealed) TOTP secret, disabled until confirmed.

        Args:
            account_id: The account enrolling a second factor.

        Returns:
            The secret and the ``otpauth://`` provisioning URI for the QR code.

        Raises:
            InvalidTokenError: If the account does not exist.
        """
        account = self._require_account(account_id)
        secret = totp.generate_secret()
        account.totp_enabled = False
        account.totp_last_step = None
        self._store_totp_secret(account, secret)
        uri = totp.provisioning_uri(secret, account.email, self._issuer)
        return EnrollmentResult(secret=secret, provisioning_uri=uri)

    def confirm_totp_enrollment(
        self, account_id: str, code: str, client_ip: str | None = None
    ) -> list[str]:
        """Enable 2FA once the user proves possession with a valid code.

        Args:
            account_id: The enrolling account.
            code: The current authenticator code.
            client_ip: Caller address for the audit trail.

        Returns:
            The freshly issued recovery codes — shown exactly once; only
            their hashes are stored.

        Raises:
            InvalidTokenError: If the account does not exist or has no pending
                enrollment.
            InvalidTwoFactorError: If ``code`` does not match the secret.
        """
        account = self._require_account(account_id)
        if account.totp_secret is None:
            raise InvalidTokenError("no TOTP enrollment in progress")
        secret = self._open_totp_secret(account)
        step = totp.verify_totp_with_step(secret, code, at=self._clock().timestamp())
        if step is None:
            raise InvalidTwoFactorError("invalid two-factor code")
        account.totp_enabled = True
        account.totp_last_step = step
        self._store.update_account(account)
        codes = self._issue_recovery_codes(account.id)
        self._auditor.record(
            "totp_enabled", outcome="success", account_id=account.id, client_ip=client_ip
        )
        return codes

    def disable_totp(self, account_id: str, client_ip: str | None = None) -> None:
        """Turn off 2FA, clear the stored secret, and void recovery codes."""
        account = self._require_account(account_id)
        account.totp_enabled = False
        account.totp_last_step = None
        self._store_totp_secret(account, None)
        self._store.replace_recovery_codes(account.id, [], self._clock())
        self._auditor.record(
            "totp_disabled", outcome="success", account_id=account.id, client_ip=client_ip
        )

    def regenerate_recovery_codes(
        self, account_id: str, current_password: str, client_ip: str | None = None
    ) -> list[str]:
        """Void all recovery codes and issue a fresh set (re-authenticated).

        Args:
            account_id: The authenticated account (must have 2FA enabled).
            current_password: Fresh proof of identity.
            client_ip: Caller address for the audit trail.

        Returns:
            The new recovery codes (shown exactly once).

        Raises:
            InvalidCredentialsError: If the password is wrong.
            InvalidTokenError: If 2FA is not enabled.
        """
        account = self._require_account(account_id)
        if not passwords.verify_password(current_password, account.password_hash):
            raise InvalidCredentialsError("current password is incorrect")
        if not account.totp_enabled:
            raise InvalidTokenError("two-factor authentication is not enabled")
        codes = self._issue_recovery_codes(account.id)
        self._auditor.record(
            "recovery_codes_regenerated",
            outcome="success",
            account_id=account.id,
            client_ip=client_ip,
        )
        return codes

    def _issue_recovery_codes(self, account_id: str) -> list[str]:
        """Mint, store (hashes only), and return a fresh recovery-code set."""
        codes: list[str] = []
        for _ in range(_RECOVERY_CODE_COUNT):
            raw = secrets.token_hex(_RECOVERY_CODE_BYTES)
            codes.append("-".join(raw[i : i + 4] for i in range(0, len(raw), 4)))
        hashes = [hash_token(_normalize_recovery_code(code)) for code in codes]
        self._store.replace_recovery_codes(account_id, hashes, self._clock())
        return codes

    # -- internals ----------------------------------------------------------------- #
    def _issue_email_token(
        self,
        account_id: str,
        purpose: str,
        ttl: timedelta,
        payload: str | None = None,
    ) -> str:
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
                payload=payload,
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


def _positive_env_seconds(name: str, default_seconds: float) -> timedelta:
    """Read a positive seconds value from the environment (default on junk)."""
    import os

    raw = os.getenv(name, "").strip()
    if not raw:
        return timedelta(seconds=default_seconds)
    try:
        value = float(raw)
    except ValueError:
        return timedelta(seconds=default_seconds)
    if value <= 0:
        return timedelta(seconds=default_seconds)
    return timedelta(seconds=value)


def build_auth_service() -> AuthService:
    """Construct an :class:`AuthService` wired from the environment.

    Uses :func:`build_identity_store` (durable when ``MERCURY_KEYSTORE_PATH``
    is set), :func:`build_mailer` (real SMTP when ``MERCURY_SMTP_HOST`` is
    set), :func:`build_secret_sealer` (stable at-rest key from
    ``MERCURY_DATA_ENC_KEY`` / ``AMA_MASTER_SEED``), and
    :func:`build_auth_auditor` (tamper-evident when ``MERCURY_AUDIT_LOG_DIR``
    is set). Session lifetimes come from ``MERCURY_SESSION_TTL_SECONDS`` /
    ``MERCURY_SESSION_TTL_SHORT_SECONDS`` / ``MERCURY_SESSION_IDLE_SECONDS``;
    ``MERCURY_PUBLIC_BASE_URL`` builds email links. When an SMTP host is
    configured, sends run on a small background executor so a slow mail
    server cannot stall (or time-fingerprint) request handling.

    Returns:
        A ready-to-use auth service.
    """
    import os

    base_url = os.getenv("MERCURY_PUBLIC_BASE_URL", "https://mercuryagent.global")
    contact = os.getenv("MERCURY_CONTACT_EMAIL", "steel.sa.llc@gmail.com")

    mail_executor = None
    if os.getenv("MERCURY_SMTP_HOST", "").strip():
        from concurrent.futures import ThreadPoolExecutor

        mail_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mercury-mail")

    return AuthService(
        build_identity_store(),
        build_mailer(),
        base_url=base_url,
        contact=contact,
        session_ttl=_positive_env_seconds("MERCURY_SESSION_TTL_SECONDS", 14 * 86400),
        session_ttl_short=_positive_env_seconds("MERCURY_SESSION_TTL_SHORT_SECONDS", 86400),
        session_idle_timeout=_positive_env_seconds("MERCURY_SESSION_IDLE_SECONDS", 86400),
        sealer=build_secret_sealer(store_is_durable=identity_store_is_durable()),
        auditor=build_auth_auditor(),
        mail_executor=mail_executor,
    )
