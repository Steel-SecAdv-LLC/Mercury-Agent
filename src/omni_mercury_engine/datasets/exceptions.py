# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Custom exceptions for dataset loading. Zero silent failures — every loader either returns real data or raises."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Module-level flag: synthetic data is NEVER allowed by default.
# Set MERCURY_ALLOW_SYNTHETIC=1 in environment to permit synthetic fallback.
#
# This is a dynamic flag — it reads the env var on every truthiness check,
# so tests can toggle os.environ["MERCURY_ALLOW_SYNTHETIC"] at runtime.


class _DynamicSyntheticFlag:
    """Bool-like flag that reads MERCURY_ALLOW_SYNTHETIC from env dynamically."""

    __slots__ = ()

    def __bool__(self) -> bool:
        """Implement the Python data model method."""
        return os.environ.get("MERCURY_ALLOW_SYNTHETIC", "0") == "1"

    def __repr__(self) -> str:
        """Return the developer representation."""
        return f"ALLOW_SYNTHETIC={bool(self)}"


ALLOW_SYNTHETIC = _DynamicSyntheticFlag()

# Air-gapped / offline operation: when MERCURY_OFFLINE is truthy, every
# dataset-layer network fetch is refused at the single HTTP chokepoint
# (``base.http_get_with_retry``) before any socket is opened. Cached data
# keeps working; anything uncached fails closed with a remediation hint.
# Read dynamically (never at import time) so tests and operators can
# toggle it without a process restart — the same contract as MERCURY_ENV.
MERCURY_OFFLINE_VAR = "MERCURY_OFFLINE"


def offline_mode_active() -> bool:
    """Whether air-gapped mode is requested via ``MERCURY_OFFLINE``."""
    return os.environ.get(MERCURY_OFFLINE_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class OfflineModeError(RuntimeError):
    """Raised when a network fetch is attempted while MERCURY_OFFLINE is set.

    Fail-closed by design: offline mode never silently degrades to stale or
    synthetic data — it serves the local cache or refuses loudly.
    """

    def __init__(self, url: str) -> None:
        """Initialize with the refused URL and a remediation hint."""
        self.url = url
        super().__init__(
            f"MERCURY_OFFLINE is set; refusing network fetch of {url}. "
            "Offline mode serves only primed local caches and loopback "
            "services (e.g. a local Ollama model on 127.0.0.1). Prime "
            "dataset caches while online (e.g. `python "
            "scripts/prefetch_datasets.py --adbench cardio thyroid ...`) "
            "or unset MERCURY_OFFLINE to allow egress."
        )


class DataSourceUnavailableError(RuntimeError):
    """Raised when a real data source cannot be reached and synthetic fallback is disabled.

    This exception replaces all silent synthetic fallbacks. Every loader must
    either return real (or cached) data with verified metadata, or raise this
    exception with a descriptive message including the loader name, URL attempted,
    and the HTTP status or underlying exception.

    Attributes:
        loader_name: Name of the loader that failed.
        source_url: URL that was attempted.
        reason: Human-readable reason for the failure.
    """

    def __init__(
        self,
        loader_name: str,
        source_url: str = "",
        reason: str = "",
        *,
        status_code: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.loader_name = loader_name
        self.source_url = source_url
        self.reason = reason
        self.status_code = status_code

        parts = [f"{loader_name}: data source unavailable"]
        if source_url:
            parts.append(f"URL: {source_url}")
        if status_code is not None:
            parts.append(f"HTTP {status_code}")
        if reason:
            parts.append(reason)

        message = ". ".join(parts)
        super().__init__(message)


def check_synthetic_allowed(loader_name: str, reason: str = "") -> bool:
    """Check whether synthetic fallback is permitted.

    Args:
        loader_name: Name of the requesting loader.
        reason: Why real data is unavailable.

    Returns:
        True if ALLOW_SYNTHETIC is set, False otherwise.

    Side effect:
        Logs a WARNING if synthetic is allowed.
        Raises DataSourceUnavailableError if not allowed.
    """
    # Re-check env var at call time so tests can toggle it dynamically.
    allowed = os.environ.get("MERCURY_ALLOW_SYNTHETIC", "0") == "1"
    if allowed:
        logger.warning(
            "%s: falling back to SYNTHETIC data (MERCURY_ALLOW_SYNTHETIC=1). Reason: %s",
            loader_name,
            reason or "real data unavailable",
        )
        return True
    raise DataSourceUnavailableError(loader_name=loader_name, reason=reason)


__all__ = [
    "ALLOW_SYNTHETIC",
    "MERCURY_OFFLINE_VAR",
    "DataSourceUnavailableError",
    "OfflineModeError",
    "check_synthetic_allowed",
    "offline_mode_active",
]
