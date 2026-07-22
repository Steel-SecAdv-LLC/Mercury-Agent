# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""HTTP routes for the self-service account flows.

Thin FastAPI wrappers over :class:`~omni_mercury_engine.api.auth_service.AuthService`:
register, verify email, login (+ optional TOTP), logout, ``/me``, password
reset, and TOTP enrollment. Handlers are plain ``def`` so FastAPI runs the
blocking PBKDF2/SQLite work in a threadpool rather than on the event loop.

The router is **opt-in and inert by default**: mounting it changes nothing for a
solo self-hoster who never calls it, and the backing store/mailer default to
in-memory/console unless ``MERCURY_KEYSTORE_PATH`` / ``MERCURY_SMTP_HOST`` are
set. The browser session is an httpOnly cookie holding an opaque token whose
hash alone is stored server-side.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

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
    TwoFactorRequiredError,
    WeakPasswordError,
    build_auth_service,
)

if TYPE_CHECKING:
    from omni_mercury_engine.api.identity_store import Account

router = APIRouter(prefix="/api/v1/auth", tags=["Accounts"])

#: Name of the browser session cookie.
SESSION_COOKIE = "mercury_session"

_service: AuthService | None = None


def get_auth_service() -> AuthService:
    """Return the process-wide auth service, building it lazily from the env.

    Exposed as a dependency so tests can override it with an in-memory instance
    via ``app.dependency_overrides``.
    """
    global _service
    if _service is None:
        _service = build_auth_service()
    return _service


def _cookie_secure() -> bool:
    """Whether to mark the session cookie ``Secure`` (default yes; off for tests)."""
    return os.getenv("MERCURY_SESSION_COOKIE_SECURE", "true").strip().lower() != "false"


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
    """A payload carrying a single email token (verify or reset request)."""

    token: str = Field(..., max_length=512)


class LoginRequest(BaseModel):
    """Login payload; ``totp_code`` is only needed when 2FA is enabled."""

    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=1024)
    totp_code: str | None = Field(default=None, max_length=12)


class EmailRequest(BaseModel):
    """A payload carrying just an email address (password-reset request)."""

    email: str = Field(..., max_length=320)


class ResetConfirmRequest(BaseModel):
    """Password-reset confirmation payload."""

    token: str = Field(..., max_length=512)
    new_password: str = Field(..., max_length=1024)


class TotpCodeRequest(BaseModel):
    """A payload carrying a TOTP code (enrollment confirmation)."""

    code: str = Field(..., max_length=12)


class AccountResponse(BaseModel):
    """Public view of an account (no secrets)."""

    id: str
    email: str
    is_verified: bool
    totp_enabled: bool


class MessageResponse(BaseModel):
    """A simple human-readable status message."""

    message: str


class EnrollmentResponse(BaseModel):
    """TOTP enrollment material for rendering a QR code."""

    secret: str
    provisioning_uri: str


def _account_view(account: Account) -> AccountResponse:
    """Project an :class:`Account` to its public response shape."""
    return AccountResponse(
        id=account.id,
        email=account.email,
        is_verified=account.is_verified,
        totp_enabled=account.totp_enabled,
    )


def current_account(
    request: Request,
    service: AuthService = Depends(get_auth_service),
) -> Account:
    """Resolve the session cookie to an account, or raise 401.

    Args:
        request: The incoming request (for the session cookie).
        service: The auth service.

    Returns:
        The authenticated account.

    Raises:
        HTTPException: 401 if there is no valid session.
    """
    raw = request.cookies.get(SESSION_COOKIE)
    account = service.authenticate_session(raw) if raw else None
    if account is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
    return account


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Register an account and email a verification link."""
    try:
        service.register(payload.email, payload.password)
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


@router.post("/login", response_model=AccountResponse)
def login(
    payload: LoginRequest,
    response: Response,
    service: AuthService = Depends(get_auth_service),
) -> AccountResponse:
    """Authenticate and set the session cookie."""
    try:
        result = service.login(payload.email, payload.password, payload.totp_code)
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    response.set_cookie(
        key=SESSION_COOKIE,
        value=result.session_token,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        max_age=result.max_age_seconds,
    )
    return _account_view(result.account)


@router.post("/logout", response_model=MessageResponse)
def logout(
    request: Request,
    response: Response,
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Invalidate the current session and clear the cookie."""
    raw = request.cookies.get(SESSION_COOKIE)
    if raw:
        service.logout(raw)
    response.delete_cookie(SESSION_COOKIE)
    return MessageResponse(message="Logged out.")


@router.get("/me", response_model=AccountResponse)
def me(account: Account = Depends(current_account)) -> AccountResponse:
    """Return the currently authenticated account."""
    return _account_view(account)


@router.post(
    "/password-reset/request",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_password_reset(
    payload: EmailRequest,
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Email a reset link if the account exists (always returns 202)."""
    service.request_password_reset(payload.email)
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


@router.post("/2fa/enroll", response_model=EnrollmentResponse)
def enroll_totp(
    account: Account = Depends(current_account),
    service: AuthService = Depends(get_auth_service),
) -> EnrollmentResponse:
    """Start TOTP enrollment; returns the secret and QR provisioning URI."""
    enrollment = service.start_totp_enrollment(account.id)
    return EnrollmentResponse(
        secret=enrollment.secret, provisioning_uri=enrollment.provisioning_uri
    )


@router.post("/2fa/confirm", response_model=MessageResponse)
def confirm_totp(
    payload: TotpCodeRequest,
    account: Account = Depends(current_account),
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Enable 2FA after verifying a code from the authenticator app."""
    try:
        service.confirm_totp_enrollment(account.id, payload.code)
    except AuthError as exc:
        raise _http_from_auth_error(exc) from exc
    return MessageResponse(message="Two-factor authentication enabled.")


@router.post("/2fa/disable", response_model=MessageResponse)
def disable_totp(
    account: Account = Depends(current_account),
    service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Disable 2FA for the current account."""
    service.disable_totp(account.id)
    return MessageResponse(message="Two-factor authentication disabled.")
