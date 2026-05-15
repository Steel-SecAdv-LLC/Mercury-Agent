"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Base class for all domain-specific data loaders.

Every domain loader MUST implement:
- fetch_realtime() -> pd.DataFrame  (live data pull)
- fetch_historical(event_id: str) -> pd.DataFrame  (specific event)
- list_events() -> list[dict]  (available ground-truth events)
- get_ground_truth(event_id: str) -> np.ndarray  (binary labels)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.security.safe_http import SafeHTTPClient

logger = logging.getLogger(__name__)

# Default cache directory for downloaded data
_DEFAULT_CACHE_DIR = Path(
    os.environ.get("MERCURY_CACHE_DIR", str(Path.home() / ".mercury" / "cache"))
)


class BaseDomainLoader(ABC):
    """
    Base class for all domain data loaders.

    Provides common infrastructure for:
    - HTTP fetching with retry logic and exponential backoff
    - Local file caching with TTL
    - Data hashing for provenance tracking
    - Standardized event catalog interface

    Subclasses must implement all four abstract methods to connect
    to their domain-specific APIs and data sources.
    """

    #: Human-readable domain name (e.g., "earthquake", "tsunami")
    DOMAIN: str = ""

    #: Data source URL for documentation
    SOURCE_URL: str = ""

    #: Whether an API key is required
    REQUIRES_API_KEY: bool = False

    #: Environment variable name for API key (if required)
    API_KEY_ENV_VAR: str = ""

    #: Cache TTL in seconds (default 1 hour)
    CACHE_TTL: int = 3600

    def __init__(
        self,
        cache_dir: Path | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: int = 60,
    ) -> None:
        """
        Initialize the domain loader.

        Args:
            cache_dir: Directory for caching downloaded data.
            api_key: API key (if required). Falls back to env var.
            max_retries: Maximum number of retry attempts for HTTP requests.
            retry_backoff: Exponential backoff base in seconds.
            timeout: HTTP request timeout in seconds.
        """
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR / self.DOMAIN
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout

        # Resolve API key
        if api_key:
            self._api_key = api_key
        elif self.REQUIRES_API_KEY and self.API_KEY_ENV_VAR:
            self._api_key = os.environ.get(self.API_KEY_ENV_VAR, "")
        else:
            self._api_key = ""

        if self.REQUIRES_API_KEY and not self._api_key:
            logger.warning(
                "%s loader: authentication not configured — "
                "set the required API key environment variable. Some operations may fail.",
                self.DOMAIN,
            )

    # =========================================================================
    # Abstract interface — every domain loader must implement these
    # =========================================================================

    @abstractmethod
    def fetch_realtime(self) -> pd.DataFrame:
        """
        Fetch most recent data from live source.

        Returns:
            DataFrame with domain-specific features as columns.

        Raises:
            DataSourceUnavailableError: If the API is unreachable.
        """
        ...

    @abstractmethod
    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """
        Fetch data for a specific historical event.

        Args:
            event_id: Identifier for the event (domain-specific format).

        Returns:
            DataFrame with domain-specific features as columns.

        Raises:
            DataSourceUnavailableError: If the data is unavailable.
            ValueError: If event_id is not recognized.
        """
        ...

    @abstractmethod
    def list_events(self) -> list[dict[str, Any]]:
        """
        Return catalog of events with ground truth.

        Returns:
            List of dicts, each with at least:
                - "event_id": str
                - "name": str
                - "date": str (ISO format)
                - "description": str
        """
        ...

    @abstractmethod
    def get_ground_truth(self, event_id: str) -> np.ndarray:
        """
        Return binary anomaly labels for an event.

        Args:
            event_id: Identifier for the event.

        Returns:
            1-D numpy array of binary labels (0=normal, 1=anomaly).

        Raises:
            ValueError: If event_id is not recognized.
        """
        ...

    # =========================================================================
    # Feature engineering (override in subclass for domain-specific features)
    # =========================================================================

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Transform raw data into feature matrix for Mercury detector.

        Default implementation returns all numeric columns as a numpy array.
        Override in subclass for domain-specific feature engineering.

        Args:
            raw_data: Raw data from fetch_historical or fetch_realtime.

        Returns:
            2-D numpy array of shape (n_samples, n_features).
        """
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        arr = raw_data[numeric_cols].values.astype(np.float64)
        # Replace inf with nan, then fill nan with column median
        arr = np.where(np.isinf(arr), np.nan, arr)
        for col_idx in range(arr.shape[1]):
            col = arr[:, col_idx]
            mask = np.isnan(col)
            if mask.any():
                median_val = np.nanmedian(col)
                col[mask] = median_val if np.isfinite(median_val) else 0.0
        return arr

    # =========================================================================
    # HTTP fetch with retry
    # =========================================================================

    def _fetch_url(
        self,
        url: str,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        """
        Fetch URL content via :class:`SafeHTTPClient` with retry logic.

        Egress contract enforced by ``SafeHTTPClient`` for this helper:

        * **HTTPS-only.** ``http://`` is refused -- the loader does
          not opt into ``allow_http``. Every dataset URL we ship is
          a public HTTPS endpoint.
        * **TRUSTED_DOMAINS allowlist enforced.** The resolved host
          must appear in
          :attr:`omni_mercury_engine.security.input_validation.TrustedEndpoints.TRUSTED_DOMAINS`.
          A subclass adding a new dataset host MUST add it to that
          set (or the request will fail closed with
          ``UnsafeURLError``). Operator-supplied egress must not use
          this helper; it must call :class:`SafeHTTPClient` directly
          with ``user_configured=True`` and the narrowest possible
          private-network policy.

        Note that ``user_configured`` is *not* set by this helper:
        loader URLs are class-constant, vetted, and DNS-resolvable to
        public addresses, so the IP-resolution gate would impose a
        per-request DNS lookup with no SSRF benefit beyond what the
        allowlist already provides. Operator-supplied egress (Ollama,
        SearXNG) does not flow through ``_fetch_url`` -- those
        callers go to :class:`SafeHTTPClient` directly with
        ``user_configured=True`` and (where appropriate)
        ``allow_private=True``.

        Args:
            url: URL to fetch.
            params: Query parameters.
            headers: HTTP headers.

        Returns:
            Response body as bytes.

        Raises:
            UnsafeURLError: URL failed the ``SafeHTTPClient`` gates
                (HTTPS-only or TRUSTED_DOMAINS allowlist). Raised on
                the first attempt with **no** retries -- a bad URL
                will be just as bad next time, and retrying would
                only mask the real cause from the operator.
            ValueError: Other configuration-shaped failures (malformed
                URL, bad params). Also re-raised immediately.
            ConnectionError: All transient retries exhausted on
                network / HTTP errors. Chains via ``__cause__`` to
                the last underlying exception so the operator-facing
                traceback names the real failure (timeout, refused
                connection, 5xx response) rather than burying it.
        """
        default_headers = {"User-Agent": "Mercury-Agent/1.0 (Steel Security Advisors)"}
        if headers:
            default_headers.update(headers)

        last_exc: Exception | None = None
        last_error_kind = "unknown"
        for attempt in range(self.max_retries + 1):
            try:
                # No user_configured=True here: we want the
                # TRUSTED_DOMAINS gate (which user_configured would
                # bypass) to enforce the allowlist for class-constant
                # dataset URLs. Operator-configured egress does not
                # use _fetch_url; it goes to SafeHTTPClient directly.
                return SafeHTTPClient.get_bytes(
                    url,
                    params=params,
                    headers=default_headers,
                    timeout=self.timeout,
                )
            except ValueError:
                # UnsafeURLError is a ValueError subclass; both signal
                # a configuration fault (bad scheme, off-allowlist host,
                # malformed URL). Retrying cannot fix configuration --
                # re-raise immediately so the real cause is visible.
                # Note: requests.HTTPError is an IOError, not a
                # ValueError, so HTTP 4xx/5xx still flow into the
                # transient-retry path below.
                raise
            except Exception as exc:
                last_exc = exc
                last_error_kind = type(exc).__name__
                if attempt < self.max_retries:
                    wait = self.retry_backoff * (2**attempt)
                    logger.warning(
                        "%s fetch attempt %d/%d failed (%s). Retrying in %.1fs.",
                        self.DOMAIN,
                        attempt + 1,
                        self.max_retries + 1,
                        last_error_kind,
                        wait,
                    )
                    time.sleep(wait)

        # Chain to the last underlying exception so the operator-facing
        # traceback names the real failure (the original socket / HTTP
        # error) rather than just the wrapper. ``raise X from None``
        # would suppress the cause; the explicit ``from last_exc`` is
        # the operator-actionable choice.
        raise ConnectionError(
            f"{self.DOMAIN}: Failed to fetch data after "
            f"{self.max_retries + 1} attempts ({last_error_kind})"
        ) from last_exc

    def _fetch_json(
        self,
        url: str,
        params: dict[str, str] | None = None,
    ) -> Any:
        """
        Fetch and parse JSON from URL.

        Args:
            url: URL to fetch.
            params: Query parameters.

        Returns:
            Parsed JSON data.
        """
        data = self._fetch_url(url, params=params)
        return json.loads(data)

    def _fetch_csv(
        self,
        url: str,
        params: dict[str, str] | None = None,
        **pandas_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch and parse CSV from URL.

        Args:
            url: URL to fetch.
            params: Query parameters.
            **pandas_kwargs: Additional kwargs for pd.read_csv.

        Returns:
            DataFrame from the CSV data.
        """
        import io

        data = self._fetch_url(url, params=params)
        return pd.read_csv(io.BytesIO(data), **pandas_kwargs)

    # =========================================================================
    # Caching
    # =========================================================================

    def _get_cache_path(self, key: str) -> Path:
        """
        Get cache file path for a given key.

        Args:
            key: Cache key (will be hashed for filename).

        Returns:
            Path to cached file.
        """
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hashed}.json"

    def _read_cache(self, key: str) -> Any | None:
        """
        Read data from cache if valid (not expired).

        Args:
            key: Cache key.

        Returns:
            Cached data or None if expired/missing.
        """
        path = self._get_cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                cached = json.load(f)
            if time.time() - cached.get("timestamp", 0) > self.CACHE_TTL:
                return None
            return cached.get("data")
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _write_cache(self, key: str, data: Any) -> None:
        """
        Write data to cache.

        Args:
            key: Cache key.
            data: Data to cache (must be JSON-serializable).
        """
        path = self._get_cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump({"timestamp": time.time(), "data": data}, f)
        except (TypeError, OSError) as exc:
            logger.debug("Cache write failed for %s: %s", key, exc)

    # =========================================================================
    # Data provenance
    # =========================================================================

    @staticmethod
    def compute_data_hash(data: np.ndarray) -> str:
        """
        Compute SHA-256 hash of data array for provenance tracking.

        Args:
            data: Numpy array to hash.

        Returns:
            Hex digest of SHA-256 hash.
        """
        return hashlib.sha256(data.tobytes()).hexdigest()

    def get_provenance(self, event_id: str, data: np.ndarray) -> dict[str, Any]:
        """
        Generate provenance metadata for benchmark results.

        Args:
            event_id: Event identifier.
            data: Feature matrix used for detection.

        Returns:
            Dict with provenance fields including timestamp,
            data hash, git commit, and Mercury version.
        """
        import shutil

        from omni_mercury_engine.security.safe_exec import (
            UnsafeSubprocessError,
            safe_exec,
        )

        git_path = shutil.which("git")
        if git_path is None:
            git_commit = "unknown"
        else:
            try:
                completed = safe_exec(
                    [git_path, "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                )
                git_commit = completed.stdout.strip() if completed.returncode == 0 else "unknown"
            except (UnsafeSubprocessError, FileNotFoundError, OSError):
                git_commit = "unknown"

        return {
            "domain": self.DOMAIN,
            "event_id": event_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "data_hash": self.compute_data_hash(data),
            "data_shape": list(data.shape),
            "git_commit": git_commit,
            "mercury_version": _get_mercury_version(),
            "source_url": self.SOURCE_URL,
        }


def _get_mercury_version() -> str:
    """Get Mercury-Agent version string."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("omni-mercury-engine")
    except (PackageNotFoundError, ImportError):
        return "dev"
