# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What counts as an email address, in one place.

The account system and the watch each had a rule, and they disagreed. An
account could be registered and verified as ``o'brien@example.com``; the
watch's stricter rule then refused that same address, so its owner got an
internal error on subscribing, and a change *to* such an address was left
half done. Whatever the account system has verified by delivering mail to it
is an address the watch must be able to tell.
"""

from __future__ import annotations

import re

__all__ = ["MAX_EMAIL_LENGTH", "is_email_address"]

#: RFC 5321's limit; also bounds the check below before the pattern runs.
MAX_EMAIL_LENGTH = 320

_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+$")


def is_email_address(email: str) -> bool:
    """Whether ``email`` has the shape of an address, in linear time.

    Args:
        email: The candidate address, already stripped by the caller.

    Returns:
        ``True`` if it has exactly one ``@``, non-empty local and domain
        parts, no whitespace, and a dot inside the domain. Delivery is the
        real test; this refuses what could not be one.
    """
    if len(email) > MAX_EMAIL_LENGTH or not _SHAPE.match(email):
        return False
    domain = email.rsplit("@", 1)[1]
    return "." in domain and not domain.startswith(".") and not domain.endswith(".")
