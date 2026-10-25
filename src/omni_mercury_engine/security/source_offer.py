# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The corresponding-source offer every deployment serves (AGPL-3.0 section 13).

Section 13 of the GNU Affero General Public License requires anyone who runs a
**modified** Mercury for users over a network to offer those users the complete
corresponding source of the version they run, at no charge, from a network
server. This module builds that offer so compliance is the default: a stock
deployment points at the upstream repository and is correct by existing; an
operator running a modified version sets :data:`SOURCE_URL_ENV` to where their
source is published (and, ideally, :data:`SOURCE_REVISION_ENV` to the exact
commit), and the offer follows.

Configured values are validated, never trusted: a source URL must be an
``https://`` URL with a host and no embedded credentials (a published offer
must not leak a token); a revision must be a hexadecimal Git object id. An
invalid value is logged and replaced by the upstream default rather than
served -- an offer that points nowhere is worse than one that points upstream.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "LICENSE_ID",
    "LICENSE_URL",
    "SOURCE_REVISION_ENV",
    "SOURCE_URL_ENV",
    "UPSTREAM_SOURCE_URL",
    "configured_revision",
    "configured_source_url",
    "source_offer",
]

#: The license this distribution is released under (SPDX).
LICENSE_ID = "AGPL-3.0-or-later"
#: The license text.
LICENSE_URL = "https://www.gnu.org/licenses/agpl-3.0.html"
#: Where the upstream project publishes its corresponding source.
UPSTREAM_SOURCE_URL = "https://github.com/Steel-SecAdv-LLC/Mercury-Agent"

#: Operator override: where the source of *this* deployment is published. A
#: deployment that runs a modified Mercury must set it (AGPL section 13).
SOURCE_URL_ENV = "MERCURY_SOURCE_URL"
#: Operator override: the exact Git commit this deployment was built from.
SOURCE_REVISION_ENV = "MERCURY_SOURCE_REVISION"

_MAX_URL_CHARS = 2048
#: A Git object id: SHA-1 (abbreviated to at least 7, up to 40) or SHA-256 (64).
_REVISION = re.compile(r"[0-9a-f]{7,40}|[0-9a-f]{64}")


def _valid_source_url(value: str) -> bool:
    """Whether ``value`` is a servable ``https://`` source URL."""
    if not value or len(value) > _MAX_URL_CHARS or not value.isprintable() or " " in value:
        return False
    parts = urlsplit(value)
    return (
        parts.scheme == "https"
        and bool(parts.hostname)
        and parts.username is None
        and parts.password is None
    )


def configured_source_url(environ: Mapping[str, str] | None = None) -> str:
    """Return where this deployment's corresponding source is published.

    Args:
        environ: Environment to read (defaults to :data:`os.environ`).

    Returns:
        The valid value of :data:`SOURCE_URL_ENV`, or
        :data:`UPSTREAM_SOURCE_URL` when it is unset or invalid (an invalid
        value is logged, never served).
    """
    env = os.environ if environ is None else environ
    raw = env.get(SOURCE_URL_ENV, "").strip()
    if not raw:
        return UPSTREAM_SOURCE_URL
    if _valid_source_url(raw):
        return raw
    logger.warning(
        "%s is not an https:// URL with a host and no credentials; "
        "serving the upstream source location instead",
        SOURCE_URL_ENV,
    )
    return UPSTREAM_SOURCE_URL


def configured_revision(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the Git commit this deployment was built from, when declared.

    Args:
        environ: Environment to read (defaults to :data:`os.environ`).

    Returns:
        The lower-cased value of :data:`SOURCE_REVISION_ENV` when it is a
        hexadecimal Git object id; ``None`` when unset or invalid (an invalid
        value is logged, never served).
    """
    env = os.environ if environ is None else environ
    raw = env.get(SOURCE_REVISION_ENV, "").strip().lower()
    if not raw:
        return None
    if _REVISION.fullmatch(raw):
        return raw
    logger.warning("%s is not a hexadecimal Git object id; omitting it", SOURCE_REVISION_ENV)
    return None


def source_offer(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Build the corresponding-source offer.

    Args:
        environ: Environment to read (defaults to :data:`os.environ`).

    Returns:
        The license, the running version, where its source is published, the
        exact revision when declared, and the upstream project for provenance.
    """
    from omni_mercury_engine._version import __version__

    source_url = configured_source_url(environ)
    return {
        "license": LICENSE_ID,
        "license_url": LICENSE_URL,
        "version": __version__,
        "source_url": source_url,
        "revision": configured_revision(environ),
        "upstream": UPSTREAM_SOURCE_URL,
        "notice": (
            "This service runs Mercury Agent, licensed under the GNU Affero "
            "General Public License v3.0 or later. If it runs a modified "
            "version, section 13 of that license entitles you to its complete "
            "corresponding source, at no charge, from source_url."
        ),
    }
