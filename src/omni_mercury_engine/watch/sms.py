# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text messages: the watch's findings, to a phone that asked for them.

Email reaches someone at a desk. A flood warning at two in the morning needs
the phone on the bedside table. This sends a finding as a text message
through a provider the operator has an account with; Mercury holds no
gateway of its own and sends nothing until the operator supplies credentials.

* **To whom.** A number its owner has verified: a code is texted to it and
  typed back. No number is texted a finding on anyone's say-so.
* **Through whom.** Twilio's Programmable Messaging API, with the operator's
  ``MERCURY_SMS_TWILIO_ACCOUNT_SID`` and ``MERCURY_SMS_TWILIO_AUTH_TOKEN``,
  from ``MERCURY_SMS_FROM`` (a number the account owns) or
  ``MERCURY_SMS_TWILIO_MESSAGING_SERVICE_SID``.
* **Stopping.** The provider honours a reply of STOP itself. Its refusal
  to send to that number afterwards (error 21610) is read as the owner's
  answer: the number stops being texted here too.

The request shape was checked against the live API as far as it can be
without an account: the endpoint, Basic authentication and form encoding
are accepted and answered with the documented error object
(``{"code": 20003, "message": ..., "status": 401}``). A message accepted and
delivered needs the operator's credentials; ``mercury-agent alerts phone``
is the proof on a real deployment.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "E164",
    "SmsOptedOutError",
    "SmsRefusedError",
    "SmsSender",
    "SmsUnreachableError",
    "TwilioSms",
    "build_sms_sender",
    "text_message",
]

#: An international phone number as E.164 writes it: ``+``, then 7 to 15 digits.
E164 = re.compile(r"\+[1-9]\d{6,14}")

ACCOUNT_SID_ENV = "MERCURY_SMS_TWILIO_ACCOUNT_SID"
AUTH_TOKEN_ENV = "MERCURY_SMS_TWILIO_AUTH_TOKEN"  # noqa: S105 - the variable's name, not a secret
FROM_ENV = "MERCURY_SMS_FROM"
SERVICE_SID_ENV = "MERCURY_SMS_TWILIO_MESSAGING_SERVICE_SID"

#: Twilio: "Attempt to send to unsubscribed recipient" (the owner replied STOP).
_OPTED_OUT = 21610
#: A text is at most this long: two segments. A message in the 7-bit GSM
#: alphabet that is longer than one segment (160) is sent as segments of 153,
#: so two are 306 characters; the old limit of 320 was three.
MAX_TEXT = 306
#: Two segments of a message that is not in that alphabet. One character
#: outside it (the "ã" of "São Paulo", a curly quote in a headline) and the
#: whole message is sent as 16-bit units, 67 to a segment: 306 characters
#: were then five segments, not the two the limit was set for.
MAX_TEXT_UCS2 = 134

#: The GSM 03.38 default alphabet, and the characters in it that take two places.
_GSM = frozenset(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)
_GSM_TWO_PLACES = frozenset("^{}\\[~]|€")


def _places(char: str, gsm: bool) -> int:
    """How much of a segment ``char`` takes, in the alphabet the message is sent in."""
    if gsm:
        return 2 if char in _GSM_TWO_PLACES else 1
    return len(char.encode("utf-16-le")) // 2


def two_segments(text: str) -> str:
    """``text``, cut to two segments of the encoding it will be sent in.

    Cut at a word, and ended with three full stops (not the ellipsis
    character, which is outside the GSM alphabet).
    """
    outside = next((at for at, char in enumerate(text) if not _is_gsm(char)), None)
    if outside is None:
        return _cut(text, gsm=True)
    # One character outside the alphabet and the whole message goes in 16-bit
    # units, 134 of them. What stands before that character may be more than
    # that: a curly quote at the 250th character must not cost the 249 before it.
    as_it_is = _cut(text, gsm=False)
    before = _cut(text[:outside], gsm=True, more_follows=True)
    return before if len(before) > len(as_it_is) else as_it_is


def _is_gsm(text: str) -> bool:
    return all(char in _GSM or char in _GSM_TWO_PLACES for char in text)


def _cut(text: str, *, gsm: bool, more_follows: bool = False) -> str:
    """``text`` cut to two segments, counted in the GSM alphabet or in 16-bit units.

    Args:
        text: The message.
        gsm: Which of the two it is counted in.
        more_follows: The caller has already left some of the message out, so
            what is returned ends with the three full stops whatever its length.
    """
    room = MAX_TEXT if gsm else MAX_TEXT_UCS2
    if not more_follows and sum(_places(char, gsm) for char in text) <= room:
        return text
    used, kept = 0, len(text)
    for at, char in enumerate(text):
        used += _places(char, gsm)
        if used > room - 3:
            kept = at
            break
    else:
        return text.rstrip(" ,;:.") + "..."
    return text[:kept].rsplit(" ", 1)[0].rstrip(",;:.") + "..."


class SmsRefusedError(RuntimeError):
    """The provider did not accept a message.

    Attributes:
        code: The provider's error code, when it gave one.
    """

    def __init__(self, message: str, code: int | None = None) -> None:
        """Record the provider's reason and code."""
        super().__init__(message)
        self.code = code


class SmsOptedOutError(SmsRefusedError):
    """The number's owner has told the provider to stop (replied STOP)."""


class SmsUnreachableError(RuntimeError):
    """The provider could not be reached: nothing was sent and nothing was refused."""


@runtime_checkable
class SmsSender(Protocol):
    """Contract for sending one text message."""

    def send(self, *, to: str, body: str) -> None:
        """Send ``body`` to the E.164 number ``to``; raise if it is not accepted."""
        ...


class TwilioSms:
    """Sends through Twilio's Programmable Messaging API."""

    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        from_number: str = "",
        messaging_service_sid: str = "",
    ) -> None:
        """Hold the operator's account and the sender it texts from.

        Raises:
            ValueError: If the account SID or the sender is not what Twilio issues.
        """
        if not re.fullmatch(r"AC[0-9a-fA-F]{32}", account_sid):
            raise ValueError("a Twilio account SID is 'AC' followed by 32 hexadecimal digits")
        if not auth_token:
            raise ValueError("the Twilio auth token is empty")
        if messaging_service_sid:
            if not re.fullmatch(r"MG[0-9a-fA-F]{32}", messaging_service_sid):
                raise ValueError("a messaging service SID is 'MG' followed by 32 hex digits")
        elif not E164.fullmatch(from_number):
            raise ValueError("the sending number must be written like +14155550123")
        self._sid = account_sid
        self._authorization = "Basic " + base64.b64encode(
            f"{account_sid}:{auth_token}".encode()
        ).decode("ascii")
        self._sender = (
            {"MessagingServiceSid": messaging_service_sid}
            if messaging_service_sid
            else {"From": from_number}
        )

    def send(self, *, to: str, body: str) -> None:
        """Send one message.

        Raises:
            SmsOptedOutError: If the number's owner has replied STOP.
            SmsRefusedError: If Twilio did not accept the message, with its reason.
            SmsUnreachableError: If Twilio could not be reached (no connection,
                a timeout). These used to leave as the HTTP library's own
                exceptions, which no caller expected.
            ValueError: If ``to`` is not an E.164 number.
        """
        import requests

        from omni_mercury_engine.datasets.exceptions import OfflineModeError
        from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError

        if not E164.fullmatch(to):
            raise ValueError("a phone number must be written like +14155550123")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self._sid}/Messages.json"
        try:
            SafeHTTPClient.post_form(
                url,
                form_data={"To": to, "Body": body, **self._sender},
                headers={"Authorization": self._authorization},
                timeout=20.0,
            )
        except requests.HTTPError as exc:
            code, reason = _twilio_error(exc)
            kind = SmsOptedOutError if code == _OPTED_OUT else SmsRefusedError
            raise kind(reason, code) from None
        except (requests.RequestException, OSError, UnsafeURLError, OfflineModeError) as exc:
            # A name that does not resolve and offline mode are refused by the
            # egress gate before any request: nothing left either. The
            # exception's own text can quote the request; its kind is enough.
            raise SmsUnreachableError(
                f"Twilio could not be reached ({type(exc).__name__})"
            ) from None


def _twilio_error(exc: Exception) -> tuple[int | None, str]:
    """The code and message in Twilio's error object, or the HTTP status when there is none."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        payload: dict[str, Any] = response.json() if response is not None else {}
    except Exception:
        payload = {}
    code = payload.get("code")
    message = str(payload.get("message") or f"HTTP {status}")
    return (int(code) if isinstance(code, int) else None), (
        f"Twilio error {code}: {message}" if code else f"Twilio: {message}"
    )


def build_sms_sender() -> SmsSender | None:
    """The deployment's text-message sender, or ``None`` when it has none configured.

    Raises:
        ValueError: If the settings are present and are not what the provider issues.
    """
    sid = os.environ.get(ACCOUNT_SID_ENV, "").strip()
    token = os.environ.get(AUTH_TOKEN_ENV, "").strip()
    if not sid or not token:
        return None
    return TwilioSms(
        account_sid=sid,
        auth_token=token,
        from_number=os.environ.get(FROM_ENV, "").strip(),
        messaging_service_sid=os.environ.get(SERVICE_SID_ENV, "").strip(),
    )


def text_message(
    finding: dict[str, Any],
    subscriber: dict[str, Any] | None = None,
    away: float | None = None,
    note: str = "",
) -> str:
    """A finding as one text: what, where from the reader, the first thing to do.

    ``note`` is one more sentence about the reader's place (for an
    earthquake, what USGS says of the shaking there).
    """
    parts = [f"Mercury watch: {' '.join(str(finding['headline']).split())}."]
    place = str((subscriber or {}).get("place") or "")
    if place and away is not None:
        parts.append(
            f'In effect at "{place}".' if away == 0.0 else f'About {away:,.0f} km from "{place}".'
        )
    if note:
        parts.append(note)
    action = " ".join(str(finding.get("action") or "").split())
    if action:
        first = re.split(r"(?<=[.!?])\s", action, maxsplit=1)[0]
        parts.append(first if first.endswith((".", "!", "?")) else first + ".")
    parts.append("In an emergency call your emergency number.")
    return two_segments(" ".join(parts))
