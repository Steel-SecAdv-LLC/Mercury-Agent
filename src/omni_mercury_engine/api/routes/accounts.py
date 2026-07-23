# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""HTTP routes for the self-service account flows.

Thin FastAPI wrappers over :class:`~omni_mercury_engine.api.auth_service.AuthService`:
register, verify email, resend verification, login (+ TOTP or recovery code),
logout, ``/me``, password reset/change, email change, account deletion, data
export, and 2FA management. Handlers are plain ``def`` so FastAPI runs the
blocking KDF/SQLite work in a threadpool rather than on the event loop.

Abuse controls live at this layer:

* **Per-action rate limits** (shared, restart-surviving counters — see
  :mod:`~omni_mercury_engine.api.rate_limit_store`): login, register,
  password-reset request, and resend-verification are throttled per client IP
  and per target account, with 429 + ``Retry-After`` on breach. The client IP
  is trusted-proxy-resolved (:mod:`~omni_mercury_engine.api.client_ip`), never
  a raw client-writable header.
* **CSRF defense-in-depth**: the session cookie is SameSite=Lax httpOnly, and
  every state-changing authenticated POST additionally requires the
  ``X-CSRF-Token`` header to match the token issued at login (double-submit,
  hash-stored server side). Disable only for non-browser clients via
  ``MERCURY_CSRF_PROTECTION=false``.

The router is **opt-in and inert by default**: mounting it changes nothing for
a solo self-hoster who never calls it, and the backing store/mailer default to
in-memory/console unless ``MERCURY_KEYSTORE_PATH`` / ``MERCURY_SMTP_HOST`` are
set.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from omni_mercury_engine.api.auth import Permission, get_api_key_store
from omni_mercury_engine.api.auth_service import (
    AccountDisabledError,
    AccountNotVerifiedError,
    AuthError,
    AuthService,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidEmailError,
    InvalidTokenError,
    InvalidTwoFactorError,
    LoginResult,
    TwoFactorRequiredError,
    WeakPasswordError,
    build_auth_service,
)
from omni_mercury_engine.api.client_ip import resolve_client_ip
from omni_mercury_engine.api.identity_store import hash_token
from omni_mercury_engine.api.rate_limit_store import (
    ActionRateLimiter,
    build_action_rate_limiter,
)

if TYPE_CHECKING:
    from omni_mercury_engine.api.auth import APIKey
    from omni_mercury_engine.api.identity_store import Account, Session

router = APIRouter(prefix="/api/v1/auth", tags=["Accounts"])

#: Name of the browser session cookie.
SESSION_COOKIE = "mercury_session"
#: Name of the readable CSRF cookie (double-submit pair of the session).
CSRF_COOKIE = "mercury_csrf"
#: Header carrying the CSRF token on state-changing requests.
CSRF_HEADER = "X-CSRF-Token"

#: Permissions a self-service account may grant to its own API keys. The
#: privileged verbs (``WRITE`` / ``DELETE`` / ``ADMIN``) are deliberately
#: excluded: a public free-service key must never be able to mutate or
#: administer the platform, only read and run detections/exports.
_SELF_SERVICE_KEY_PERMISSIONS: frozenset[Permission] = frozenset(
    {Permission.READ, Permission.DETECT, Permission.EXPORT}
)
#: Default permission set when a create request names none.
_DEFAULT_KEY_PERMISSIONS: frozenset[Permission] = frozenset({Permission.READ, Permission.DETECT})
#: Fallback per-account active-key cap (override with
#: ``MERCURY_MAX_API_KEYS_PER_ACCOUNT``).
_DEFAULT_MAX_API_KEYS_PER_ACCOUNT = 10

_service: AuthService | None = None
_service_lock = threading.Lock()

_action_limiter: ActionRateLimiter | None = None
_action_limiter_lock = threading.Lock()


def get_auth_service() -> AuthService:
    """Return the process-wide auth service, building it lazily from the env.

    Construction is lock-guarded (double-checked): two request threads racing
    the first build must never each construct a service — with the in-memory
    store the loser's accounts would silently vanish when the winner's
    instance is returned to later callers.

    Exposed as a dependency so tests can override it with an in-memory
    instance via ``app.dependency_overrides``.
    """
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = build_auth_service()
    return _service


def get_action_limiter() -> ActionRateLimiter:
    """Return the process-wide per-action rate limiter (lock-guarded build)."""
    global _action_limiter
    if _action_limiter is None:
        with _action_limiter_lock:
            if _action_limiter is None:
                _action_limiter = build_action_rate_limiter()
    return _action_limiter


def _client_ip(request: Request) -> str:
    """Trusted-proxy-resolved client address for keys and audit."""
    return resolve_client_ip(
        request.client.host if request.client else None,
        request.headers.get("X-Forwarded-For"),
    )


def _email_bucket(email: str) -> str:
    """Rate-limit key for a target account: hashed so no PII hits storage."""
    return hash_token(email.strip().lower())[:32]


def _enforce_action_limits(request: Request, *checks: tuple[str, str]) -> None:
    """Apply per-action throttles; raise 429 with ``Retry-After`` on breach.

    Every listed check counts this attempt (a denied action still consumed
    one), and the first breached rule wins — the caller learns only that it
    must wait, never which discriminator tripped.
    """
    limiter = get_action_limiter()
    for action, key in checks:
        allowed, retry_after = limiter.check(action, key)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"message": "too many attempts; please retry later"},
                headers={"Retry-After": str(retry_after)},
            )


def _cookie_secure() -> bool:
    """Whether to mark cookies ``Secure`` (default yes; off for tests)."""
    return os.getenv("MERCURY_SESSION_COOKIE_SECURE", "true").strip().lower() != "false"


def _csrf_enabled() -> bool:
    """Whether CSRF double-submit enforcement is on (default yes)."""
    return os.getenv("MERCURY_CSRF_PROTECTION", "true").strip().lower() != "false"


def _set_session_cookies(response: Response, result: LoginResult) -> None:
    """Set the session (httpOnly) + CSRF (readable) cookie pair.

    A non-persistent login (remember-me off) omits ``max_age`` so the cookie
    dies with the browser session; the server-side absolute/idle timeouts
    bound it regardless.
    """
    max_age = result.max_age_seconds if result.persistent else None
    response.set_cookie(
        key=SESSION_COOKIE,
        value=result.session_token,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        max_age=max_age,
    )
    response.set_cookie(
        key=CSRF_COOKIE,
        value=result.csrf_token,
        httponly=False,  # the frontend must read it to echo the header back
        secure=_cookie_secure(),
        samesite="lax",
        max_age=max_age,
    )


def _clear_session_cookies(response: Response) -> None:
    """Delete the session + CSRF cookie pair."""
    response.delete_cookie(SESSION_COOKIE)
    response.delete_cookie(CSRF_COOKIE)


#: Maps each auth failure to the HTTP status the route should return.
_STATUS_BY_ERROR: dict[type[AuthError], int] = {
    InvalidEmailError: status.HTTP_400_BAD_REQUEST,
    WeakPasswordError: status.HTTP_400_BAD_REQUEST,
    InvalidTokenError: status.HTTP_400_BAD_REQUEST,
    EmailAlreadyRegisteredError: status.HTTP_409_CONFLICT,
    InvalidCredentialsError: status.HTTP_401_UNAUTHORIZED,
    TwoFactorRequiredError: status.HTTP_401_UNAUTHORIZED,
    InvalidTwoFactorError: status.HTTP_401_UNAUTHORIZED,
    AccountNotVerifiedError: status.HTTP_403_FORBIDDEN,
    AccountDisabledError: status.HTTP_403_FORBIDDEN,
}


def _http_from_auth_error(exc: AuthError) -> HTTPException:
    """Translate an :class:`AuthError` into the matching :class:`HTTPException`.

    ``TwoFactorRequiredError`` carries a stable ``code`` so the frontend can tell
    "prompt for the 2FA code" apart from a plain bad-credentials rejection.
    """
    http_status = _STATUS_BY_ERROR.get(type(exc), status.HTTP_400_BAD_REQUEST)
    detail: dict[str, str] = {"message": str(exc)}
    if isinstance(exc, TwoFactorRequiredError):
        detail["code"] = "two_factor_required"
    return HTTPException(status_code=http_status, detail=detail)


# --------------------------------------------------------------------------- #
# request / response models
# --------------------------------------------------------------------------- #
class RegisterRequest(BaseModel):
    """Registration payload."""

    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=1024)


class TokenRequest(BaseModel):
    """A payload carrying a single email token (verify or confirm flows)."""

    token: str = Field(..., max_length=512)


class LoginRequest(BaseModel):
    """Login payload; second factor and remember-me are optional."""

    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=1024)
    totp_code: str | None = Field(default=None, max_length=12)
    recovery_code: str | None = Field(default=None, max_length=64)
    remember_me: bool = Field(default=True)


class EmailRequest(BaseModel):
    """A payload carrying just an email address (reset / resend requests)."""

    email: str = Field(..., max_length=320)


class ResetConfirmRequest(BaseModel):
    """Password-reset confirmation payload."""

    token: str = Field(..., max_length=512)
    new_password: str = Field(..., max_length=1024)


class ChangePasswordRequest(BaseModel):
    """Authenticated password-change payload."""

    current_password: str = Field(..., max_length=1024)
    new_password: str = Field(..., max_length=1024)


class ChangeEmailRequest(BaseModel):
    """Authenticated email-change request payload."""

    new_email: str = Field(..., max_length=320)
    current_password: str = Field(..., max_length=1024)


class PasswordConfirmRequest(BaseModel):
    """A payload re-authenticating a sensitive action with the password."""

    current_password: str = Field(..., max_length=1024)


class TotpCodeRequest(BaseModel):
    """A payload carrying a TOTP code (enrollment confirmation)."""

    code: str = Field(..., max_length=12)


class AccountResponse(BaseModel):
    """Public view of an account (no secrets)."""

    id: str
    email: str
    is_verified: bool
    totp_enabled: bool
    tier: str


class LoginResponse(BaseModel):
    """Successful-login payload: the account plus the CSRF token.

    The CSRF token is also set as a readable cookie; returning it in the body
    lets single-page frontends store it without cookie parsing.
    """

    account: AccountResponse
    csrf_token: str


class MessageResponse(BaseModel):
    """A simple human-readable status message."""

    message: str


class EnrollmentResponse(BaseModel):
    """TOTP enrollment material for rendering a QR code."""

    secret: str
    provisioning_uri: str


class RecoveryCodesResponse(BaseModel):
    """Freshly issued single-use recovery codes (shown exactly once)."""

    recovery_codes: list[str]
    message: str


class CreateApiKeyRequest(BaseModel):
    """Payload to mint a new API key for the current account."""

    name: str = Field(..., min_length=1, max_length=100)
    #: Optional lifetime in days; ``None`` mints a non-expiring key.
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)
    #: Optional permission names; ``None`` uses the safe default set. Every
    #: name must fall inside the self-service whitelist or the request is
    #: rejected 400 (a user cannot grant its own key privileges it lacks).
    permissions: list[str] | None = Field(default=None, max_length=16)


class ApiKeyResponse(BaseModel):
    """Public view of a stored API key (never the raw secret or its hash)."""

    key_id: str
    name: str
    permissions: list[str]
    created_at: str
    expires_at: str | None
    last_used_at: str | None
    is_active: bool


class ApiKeyCreatedResponse(BaseModel):
    """Creation result: the raw key (shown once) plus its stored metadata."""

    #: The raw API key. Returned exactly once — only its hash is persisted, so
    #: it can never be retrieved again.
    api_key: str
    key: ApiKeyResponse
    message: str


def _account_view(account: Account) -> AccountResponse:
    """Project an :class:`Account` to its public response shape."""
    return AccountResponse(
        id=account.id,
        email=account.email,
        is_verified=account.is_verified,
        totp_enabled=account.totp_enabled,
        tier=account.tier,
    )


def _api_key_view(key: APIKey) -> ApiKeyResponse:
    """Project an :class:`APIKey` to its public response shape (no secret)."""
    return ApiKeyResponse(
        key_id=key.key_id,
        name=key.name,
        permissions=sorted(p.value for p in key.permissions),
        created_at=key.created_at.isoformat(),
        expires_at=key.expires_at.isoformat() if key.expires_at is not None else None,
        last_used_at=key.last_used_at.isoformat() if key.last_used_at is not None else None,
        is_active=key.is_active,
    )


def _resolve_key_permissions(names: list[str] | None) -> set[Permission]:
    """Validate requested permission names against the self-service whitelist.

    Args:
        names: The requested permission values, or ``None`` for the default set.

    Returns:
        The resolved permission set.

    Raises:
        HTTPException: 400 if a name is unknown or outside the whitelist — a
            self-service user must not be able to grant its own key a
            privilege (``WRITE`` / ``DELETE`` / ``ADMIN``) it does not hold.
    """
    if not names:
        return set(_DEFAULT_KEY_PERMISSIONS)
    resolved: set[Permission] = set()
    for name in names:
        try:
            permission = Permission(name.strip().lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": f"unknown permission: {name!r}"},
            ) from None
        if permission not in _SELF_SERVICE_KEY_PERMISSIONS:
            allowed = ", ".join(sorted(p.value for p in _SELF_SERVICE_KEY_PERMISSIONS))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": (
                        f"permission not allowed for self-service keys: "
                        f"{permission.value} (allowed: {allowed})"
                    )
                },
            )
        resolved.add(permission)
    return resolved


def _max_api_keys_per_account() -> int:
    """The per-account active-key cap (``MERCURY_MAX_API_KEYS_PER_ACCOUNT``)."""
    raw = os.getenv("MERCURY_MAX_API_KEYS_PER_ACCOUNT", "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_MAX_API_KEYS_PER_ACCOUNT
    except ValueError:
        return _DEFAULT_MAX_API_KEYS_PER_ACCOUNT
    return value if value > 0 else _DEFAULT_MAX_API_KEYS_PER_ACCOUNT


def current_session(
    request: Request,
    service: AuthService = Depends(get_auth_service),
) -> tuple[Account, Session]:
    """Resolve the session cookie to (account, session), or raise 401.

    Args:
        request: The incoming request (for the session cookie).
        service: The auth service.

    Returns:
        The authenticated account and its session record.

    Raises:
        HTTPException: 401 if there is no valid session.
    """
    raw = request.cookies.get(SESSION_COOKIE)
    resolved = service.resolve_session(raw) if raw else None
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
    return resolved


def current_account(
    session_pair: tuple[Account, Session] = Depends(current_session),
) -> Account:
    """Resolve the session cookie to an account, or raise 401."""
    return session_pair[0]


def csrf_protected(
    request: Request,
    session_pair: tuple[Account, Session] = Depends(current_session),
    service: AuthService = Depends(get_auth_service),
) -> tuple[Account, Session]:
    """Authenticate AND require a matching CSRF header (state-changing POSTs).

    Defense-in-depth on top of SameSite=Lax: the ``X-CSRF-Token`` header must
    hash-match the token bound to this session at login. Cross-site HTML can
    neither read the CSRF cookie nor set custom headers, so a forged request
    fails here even if the browser attached the session cookie.

    Raises:
        HTTPException: 403 when enforcement is on and the header is absent or
            wrong.
    """
    account, session = session_pair
    if _csrf_enabled() and not service.verify_csrf(session, request.headers.get(CSRF_HEADER)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": f"missing or invalid {CSRF_HEADER} header"},
        )
    return account, session


# --------------------------------------------------------------------------- #
# registration + verification
# --------------------------------------------------------------------------- #
@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    request: Request,
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Register an account and email a verification link."""
    ip = _client_ip(request)
    _enforce_action_limits(request, ("register_ip", ip))
    try:
        service.register(payload.email, payload.password, client_ip=ip)
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    return MessageResponse(message="Check your email to verify your account.")


@router.post("/verify-email", response_model=AccountResponse)
def verify_email(
    payload: TokenRequest,
    service: AuthService = Depends(get_auth_service),
) -> AccountResponse:
    """Confirm an email address with a verification token."""
    try:
        account = service.verify_email(payload.token)
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    return _account_view(account)


@router.post(
    "/resend-verification",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def resend_verification(
    payload: EmailRequest,
    request: Request,
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Re-send the verification email (enumeration-safe; always 202)."""
    ip = _client_ip(request)
    _enforce_action_limits(
        request,
        ("resend_ip", ip),
        ("resend_account", _email_bucket(payload.email)),
    )
    service.resend_verification(payload.email, client_ip=ip)
    return MessageResponse(
        message="If that account exists and is unverified, a new link has been sent."
    )


# --------------------------------------------------------------------------- #
# login / logout / me
# --------------------------------------------------------------------------- #
@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    service: AuthService = Depends(get_auth_service),
) -> LoginResponse:
    """Authenticate and set the session + CSRF cookie pair."""
    ip = _client_ip(request)
    _enforce_action_limits(
        request,
        ("login_ip", ip),
        ("login_account", _email_bucket(payload.email)),
    )
    try:
        result = service.login(
            payload.email,
            payload.password,
            payload.totp_code,
            payload.recovery_code,
            remember_me=payload.remember_me,
            client_ip=ip,
        )
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    _set_session_cookies(response, result)
    return LoginResponse(account=_account_view(result.account), csrf_token=result.csrf_token)


@router.post("/logout", response_model=MessageResponse)
def logout(
    request: Request,
    response: Response,
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Invalidate the current session and clear the cookies."""
    raw = request.cookies.get(SESSION_COOKIE)
    if raw:
        service.logout(raw)
    _clear_session_cookies(response)
    return MessageResponse(message="Logged out.")


@router.get("/me", response_model=AccountResponse)
def me(account: Account = Depends(current_account)) -> AccountResponse:
    """Return the currently authenticated account."""
    return _account_view(account)


@router.get("/export")
def export_data(
    account: Account = Depends(current_account),
    service: AuthService = Depends(get_auth_service),
) -> dict[str, object]:
    """Export the account's stored personal data (portability)."""
    return service.export_account_data(account.id)


# --------------------------------------------------------------------------- #
# password reset + change
# --------------------------------------------------------------------------- #
@router.post(
    "/password-reset/request",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_password_reset(
    payload: EmailRequest,
    request: Request,
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Email a reset link if the account exists (always returns 202)."""
    ip = _client_ip(request)
    _enforce_action_limits(
        request,
        ("reset_ip", ip),
        ("reset_account", _email_bucket(payload.email)),
    )
    service.request_password_reset(payload.email, client_ip=ip)
    return MessageResponse(message="If that account exists, a reset link has been sent.")


@router.post("/password-reset/confirm", response_model=MessageResponse)
def confirm_password_reset(
    payload: ResetConfirmRequest,
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Set a new password from a reset token, invalidating existing sessions."""
    try:
        service.confirm_password_reset(payload.token, payload.new_password)
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    return MessageResponse(message="Password updated. Please log in again.")


@router.post("/password/change", response_model=MessageResponse)
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    response: Response,
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Change the password (re-authenticated); rotates every session."""
    account, _session = session_pair
    try:
        result = service.change_password(
            account.id,
            payload.current_password,
            payload.new_password,
            client_ip=_client_ip(request),
        )
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    _set_session_cookies(response, result)
    return MessageResponse(message="Password changed. Other sessions were signed out.")


# --------------------------------------------------------------------------- #
# email change
# --------------------------------------------------------------------------- #
@router.post(
    "/email-change/request",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_email_change(
    payload: ChangeEmailRequest,
    request: Request,
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Start an email change; a confirmation link goes to the new address.

    Throttled per IP and per acting account: the route emails a
    caller-supplied address, so without a ceiling a logged-in account could be
    used to bomb arbitrary mailboxes with confirmation links.
    """
    account, _session = session_pair
    ip = _client_ip(request)
    _enforce_action_limits(
        request,
        ("email_change_ip", ip),
        ("email_change_account", account.id),
    )
    try:
        service.request_email_change(
            account.id,
            payload.new_email,
            payload.current_password,
            client_ip=ip,
        )
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    return MessageResponse(message="Check the new address for a confirmation link.")


@router.post("/email-change/confirm", response_model=AccountResponse)
def confirm_email_change(
    payload: TokenRequest,
    service: AuthService = Depends(get_auth_service),
) -> AccountResponse:
    """Complete an email change from the link sent to the new address."""
    try:
        account = service.confirm_email_change(payload.token)
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    return _account_view(account)


# --------------------------------------------------------------------------- #
# account deletion
# --------------------------------------------------------------------------- #
@router.post("/account/delete", response_model=MessageResponse)
def delete_account(
    payload: PasswordConfirmRequest,
    request: Request,
    response: Response,
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Permanently delete the account (re-authenticated)."""
    account, _session = session_pair
    try:
        service.delete_account(account.id, payload.current_password, client_ip=_client_ip(request))
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    _clear_session_cookies(response)
    return MessageResponse(message="Account deleted.")


# --------------------------------------------------------------------------- #
# two-factor authentication
# --------------------------------------------------------------------------- #
@router.post("/2fa/enroll", response_model=EnrollmentResponse)
def enroll_totp(
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> EnrollmentResponse:
    """Start TOTP enrollment; returns the secret and QR provisioning URI."""
    account, _session = session_pair
    enrollment = service.start_totp_enrollment(account.id)
    return EnrollmentResponse(
        secret=enrollment.secret, provisioning_uri=enrollment.provisioning_uri
    )


@router.post("/2fa/confirm", response_model=RecoveryCodesResponse)
def confirm_totp(
    payload: TotpCodeRequest,
    request: Request,
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> RecoveryCodesResponse:
    """Enable 2FA after verifying a code; returns one-time recovery codes."""
    account, _session = session_pair
    try:
        codes = service.confirm_totp_enrollment(
            account.id, payload.code, client_ip=_client_ip(request)
        )
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    return RecoveryCodesResponse(
        recovery_codes=codes,
        message=(
            "Two-factor authentication enabled. Store these recovery codes "
            "somewhere safe — they are shown only once."
        ),
    )


@router.post("/2fa/disable", response_model=MessageResponse)
def disable_totp(
    request: Request,
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Disable 2FA for the current account."""
    account, _session = session_pair
    service.disable_totp(account.id, client_ip=_client_ip(request))
    return MessageResponse(message="Two-factor authentication disabled.")


@router.post("/2fa/recovery-codes", response_model=RecoveryCodesResponse)
def regenerate_recovery_codes(
    payload: PasswordConfirmRequest,
    request: Request,
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> RecoveryCodesResponse:
    """Void all recovery codes and issue a fresh set (re-authenticated)."""
    account, _session = session_pair
    try:
        codes = service.regenerate_recovery_codes(
            account.id, payload.current_password, client_ip=_client_ip(request)
        )
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    return RecoveryCodesResponse(
        recovery_codes=codes,
        message="New recovery codes issued; the old ones no longer work.",
    )


# --------------------------------------------------------------------------- #
# API-key management (account-scoped, one-time reveal)
# --------------------------------------------------------------------------- #
@router.post(
    "/api-keys",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_api_key(
    payload: CreateApiKeyRequest,
    request: Request,
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> ApiKeyCreatedResponse:
    """Mint a new API key for the current account (raw key shown once).

    The key is owned by the account (``user_id == account.id``), so it inherits
    the account's quota tier and appears in :func:`list_api_keys`. Only a
    verified account may issue keys, permissions are whitelist-validated, and a
    per-account active-key cap bounds blast radius. The raw key is returned in
    this response body only — never stored, never retrievable again.
    """
    account, _session = session_pair
    if not account.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "verify your email before issuing API keys"},
        )
    permissions = _resolve_key_permissions(payload.permissions)
    store = get_api_key_store()
    active = [key for key in store.list_by_user(account.id) if key.is_active and not key.is_expired]
    if len(active) >= _max_api_keys_per_account():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": (
                    "active API-key limit reached; revoke an existing key before "
                    "creating another"
                )
            },
        )
    raw_key, key = store.create_key(
        name=payload.name,
        user_id=account.id,
        permissions=permissions,
        expires_in_days=payload.expires_in_days,
    )
    service.record_event(
        "api_key_created",
        outcome="success",
        account_id=account.id,
        client_ip=_client_ip(request),
        details={"key_id": key.key_id},
    )
    return ApiKeyCreatedResponse(
        api_key=raw_key,
        key=_api_key_view(key),
        message="Store this key now — it is shown only once and cannot be retrieved again.",
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
def list_api_keys(
    account: Account = Depends(current_account),
) -> list[ApiKeyResponse]:
    """List the current account's API keys (metadata only; newest first)."""
    store = get_api_key_store()
    return [_api_key_view(key) for key in store.list_by_user(account.id)]


@router.delete("/api-keys/{key_id}", response_model=MessageResponse)
def revoke_api_key(
    key_id: str,
    request: Request,
    session_pair: tuple[Account, Session] = Depends(csrf_protected),
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Revoke one of the current account's API keys.

    Ownership is enforced by returning 404 for a key that does not exist *or*
    belongs to another account — a caller learns nothing about keys it does
    not own. Revocation is idempotent (a second call still 200s if the key is
    still the caller's).
    """
    account, _session = session_pair
    store = get_api_key_store()
    key = store.get_by_id(key_id)
    if key is None or key.user_id != account.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "API key not found"},
        )
    store.revoke(key_id)
    service.record_event(
        "api_key_revoked",
        outcome="success",
        account_id=account.id,
        client_ip=_client_ip(request),
        details={"key_id": key_id},
    )
    return MessageResponse(message="API key revoked.")
