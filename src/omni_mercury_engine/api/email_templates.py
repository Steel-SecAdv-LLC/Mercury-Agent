# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Transactional email templates for the account flows.

Every account email is produced here as a matched (subject, plaintext, HTML,
headers) tuple so the auth service composes messages from one audited place:

* Both a plaintext body and an HTML alternative are always produced — the
  pair is what mainstream providers' spam scoring expects, and text-only
  clients keep working.
* A ``List-Unsubscribe`` header (mailto) is attached to every message.
  Transactional mail does not strictly require it, but Gmail/Yahoo bulk-sender
  rules reward its presence and it gives recipients a working escape hatch.
* All user-influenced values (links carry URL-safe tokens; the issuer name is
  operator-configured) are HTML-escaped before interpolation, so a hostile
  value cannot break out of the markup.

Templates are deliberately plain: inline styles only (email clients strip
``<style>`` blocks unpredictably), no images, no external resources — nothing
to block, nothing to track.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

__all__ = [
    "EmailContent",
    "email_change_email",
    "recovery_codes_notice",
    "reset_email",
    "verification_email",
]

_UNSUBSCRIBE_MAILTO = "mailto:{contact}?subject=unsubscribe"


@dataclass
class EmailContent:
    """A fully rendered transactional email."""

    subject: str
    body: str
    html_body: str
    headers: dict[str, str] = field(default_factory=dict)


def _headers(contact: str) -> dict[str, str]:
    """Deliverability headers shared by every account email."""
    return {
        "List-Unsubscribe": f"<{_UNSUBSCRIBE_MAILTO.format(contact=contact)}>",
        "Auto-Submitted": "auto-generated",
    }


def _html_shell(issuer: str, heading: str, lines: list[str], link: str | None, cta: str) -> str:
    """Render the shared HTML frame around a message's content lines."""
    safe_issuer = html.escape(issuer)
    paragraphs = "".join(
        f'<p style="margin:0 0 12px;color:#333;font-size:15px;line-height:1.5;">'
        f"{html.escape(line)}</p>"
        for line in lines
    )
    button = ""
    if link is not None:
        safe_link = html.escape(link, quote=True)
        button = (
            f'<p style="margin:20px 0;"><a href="{safe_link}" '
            'style="background:#1a1a2e;color:#ffffff;text-decoration:none;'
            'padding:12px 24px;border-radius:6px;font-size:15px;display:inline-block;">'
            f"{html.escape(cta)}</a></p>"
            f'<p style="margin:0 0 12px;color:#777;font-size:13px;word-break:break-all;">'
            f"Or paste this link into your browser:<br>{safe_link}</p>"
        )
    return (
        '<div style="max-width:560px;margin:0 auto;padding:24px;'
        "font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;"
        'background:#ffffff;">'
        f'<h2 style="margin:0 0 16px;color:#1a1a2e;font-size:20px;">{html.escape(heading)}</h2>'
        f"{paragraphs}{button}"
        f'<hr style="border:none;border-top:1px solid #eee;margin:24px 0 12px;">'
        f'<p style="margin:0;color:#999;font-size:12px;">{safe_issuer}</p>'
        "</div>"
    )


def verification_email(issuer: str, link: str, ttl_hours: int, contact: str) -> EmailContent:
    """Render the account-verification email."""
    lines = [
        f"Welcome to {issuer}.",
        "Confirm your email to activate your account.",
        f"This link expires in {ttl_hours} hours. " "If you did not sign up, ignore this message.",
    ]
    body = (
        f"Welcome to {issuer}.\n\n"
        f"Confirm your email to activate your account:\n{link}\n\n"
        f"This link expires in {ttl_hours} hours. "
        "If you did not sign up, ignore this message."
    )
    return EmailContent(
        subject=f"Verify your {issuer} account",
        body=body,
        html_body=_html_shell(issuer, "Confirm your email", lines, link, "Verify email"),
        headers=_headers(contact),
    )


def reset_email(issuer: str, link: str, ttl_minutes: int, contact: str) -> EmailContent:
    """Render the password-reset email."""
    lines = [
        f"A password reset was requested for your {issuer} account.",
        f"This link expires in {ttl_minutes} minutes. If you did not request "
        "this, ignore this message; your password is unchanged.",
    ]
    body = (
        f"A password reset was requested for your {issuer} account.\n\n"
        f"Reset it here:\n{link}\n\n"
        f"This link expires in {ttl_minutes} minutes. "
        "If you did not request this, ignore this message; your password is unchanged."
    )
    return EmailContent(
        subject=f"Reset your {issuer} password",
        body=body,
        html_body=_html_shell(issuer, "Reset your password", lines, link, "Reset password"),
        headers=_headers(contact),
    )


def email_change_email(issuer: str, link: str, ttl_hours: int, contact: str) -> EmailContent:
    """Render the change-of-address confirmation email (sent to the NEW address)."""
    lines = [
        f"A request was made to move a {issuer} account to this email address.",
        "Confirm to complete the change. Until then the account keeps its " "current address.",
        f"This link expires in {ttl_hours} hours. If this wasn't you, ignore " "this message.",
    ]
    body = (
        f"A request was made to move a {issuer} account to this email address.\n\n"
        f"Confirm the change here:\n{link}\n\n"
        f"This link expires in {ttl_hours} hours. "
        "If this wasn't you, ignore this message."
    )
    return EmailContent(
        subject=f"Confirm your new {issuer} email address",
        body=body,
        html_body=_html_shell(issuer, "Confirm your new address", lines, link, "Confirm change"),
        headers=_headers(contact),
    )


def recovery_codes_notice(issuer: str, remaining: int, contact: str) -> EmailContent:
    """Render the security notice sent when a 2FA recovery code is used."""
    lines = [
        f"A two-factor recovery code was just used to sign in to your {issuer} account.",
        f"{remaining} unused recovery code(s) remain.",
        "If this was you, no action is needed. If not, reset your password "
        "immediately and regenerate your recovery codes.",
    ]
    body = "\n\n".join(lines)
    return EmailContent(
        subject=f"{issuer}: a recovery code was used",
        body=body,
        html_body=_html_shell(issuer, "Recovery code used", lines, None, ""),
        headers=_headers(contact),
    )
