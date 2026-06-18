# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Single source of truth for the Mercury Agent distribution version.

The version is resolved at import time so the canonical version surfaces — the
package ``__version__``, the API server's ``API_VERSION``, the ``mercury
--version`` CLI, OpenTelemetry/OTLP ``service.version`` attributes, and the
operator-facing health/version banners — stay in lockstep with the
distribution and cannot silently drift when a release is cut. (Deliberately
coarse, stable client identifiers — e.g. the outbound ``User-Agent`` tokens —
are not release-versioned and are intentionally out of scope.)

Resolution order (first that succeeds wins):

1. **Installed distribution metadata** (``importlib.metadata`` — the normal
   installed/production path; the value baked in at build time from
   ``pyproject.toml``).
2. **The source ``pyproject.toml`` ``[project].version``**, read directly, for an
   *uninstalled* source checkout. ``pyproject.toml`` exists only in the source
   tree — exactly the case metadata cannot cover — so a fresh checkout still
   reports the true, current version with no install step.
3. **``_FALLBACK_VERSION``** — a last-resort literal, reached only when *neither*
   runtime source is available (e.g. an installed wheel whose metadata was
   stripped). It is the only hand-maintained value here and is kept equal to
   ``[project].version``.

Because (2) reads the real ``pyproject.toml``, the hand-maintained literal can no
longer drift in the ordinary source-checkout case; it is a genuine last resort,
not the routine fallback.
"""

from __future__ import annotations

from importlib.metadata import (
    PackageNotFoundError,
    version as _pkg_version,
)
from pathlib import Path

#: Distribution name as declared in ``pyproject.toml`` (``[project].name``).
_DISTRIBUTION_NAME = "mercury-agent"

#: Last-resort literal — reached only when neither installed metadata nor the
#: source ``pyproject.toml`` is readable. Kept equal to ``[project].version``.
_FALLBACK_VERSION = "2.0.0"


def _version_from_pyproject() -> str | None:
    """Return ``[project].version`` from the repo ``pyproject.toml``, or ``None``.

    Returns ``None`` when the file or key is missing or unreadable — for example
    an installed wheel, which does not ship ``pyproject.toml`` (and where the
    metadata lookup has already provided the version anyway).
    """
    try:
        import tomllib

        # This file is ``<root>/src/omni_mercury_engine/_version.py``; the repo
        # root (where ``pyproject.toml`` lives) is ``parents[2]`` —
        # parents[0]=omni_mercury_engine, [1]=src, [2]=root.
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except (OSError, KeyError, ValueError, TypeError, IndexError, ImportError):
        return None


def _resolve_version() -> str:
    """Resolve the distribution version (metadata → pyproject → literal)."""
    try:
        return _pkg_version(_DISTRIBUTION_NAME)
    except (PackageNotFoundError, ImportError):
        return _version_from_pyproject() or _FALLBACK_VERSION


__version__ = _resolve_version()


def get_version() -> str:
    """Return the resolved Mercury Agent distribution version string."""
    return __version__
