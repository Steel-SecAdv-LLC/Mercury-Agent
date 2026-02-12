"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

Custom exceptions for dataset loading.
Zero silent failures — every loader either returns real data or raises.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Module-level flag: synthetic data is NEVER allowed by default.
# Set MERCURY_ALLOW_SYNTHETIC=1 in environment to permit synthetic fallback.
ALLOW_SYNTHETIC: bool = os.environ.get("MERCURY_ALLOW_SYNTHETIC", "0") == "1"


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
    if ALLOW_SYNTHETIC:
        logger.warning(
            "%s: falling back to SYNTHETIC data (MERCURY_ALLOW_SYNTHETIC=1). Reason: %s",
            loader_name,
            reason or "real data unavailable",
        )
        return True
    raise DataSourceUnavailableError(loader_name=loader_name, reason=reason)


__all__ = [
    "ALLOW_SYNTHETIC",
    "DataSourceUnavailableError",
    "check_synthetic_allowed",
]
