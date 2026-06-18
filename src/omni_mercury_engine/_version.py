# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Single source of truth for the Mercury Agent distribution version.

The version is resolved from the installed package metadata at import time, so
every surface that reports it — the package ``__version__``, the API server's
``API_VERSION``, OpenTelemetry/OTLP ``service.version`` resource attributes, and
any operator-facing banner — stays in lockstep with the built and installed
distribution and cannot silently drift when a release is cut.

``_FALLBACK_VERSION`` is used only when the distribution is not installed (for
example, importing straight from a fresh source checkout). It mirrors
``[project].version`` in ``pyproject.toml`` and is kept in sync by the release
process; CI is the backstop that proves the installed metadata and this literal
agree.
"""

from __future__ import annotations

from importlib.metadata import (
    PackageNotFoundError,
    version as _pkg_version,
)

#: Distribution name as declared in ``pyproject.toml`` (``[project].name``).
_DISTRIBUTION_NAME = "mercury-agent"

#: Source-tree fallback; mirrors ``[project].version`` for uninstalled checkouts.
_FALLBACK_VERSION = "1.8.0"

try:
    __version__ = _pkg_version(_DISTRIBUTION_NAME)
except (PackageNotFoundError, ImportError):
    __version__ = _FALLBACK_VERSION


def get_version() -> str:
    """Return the resolved Mercury Agent distribution version string."""
    return __version__
