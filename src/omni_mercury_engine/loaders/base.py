# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for all domain-specific data loaders.

Every domain loader MUST implement:
- fetch_realtime() -> pd.DataFrame  (live data pull)
- fetch_historical(event_id: str) -> pd.DataFrame  (specific event)
- list_events() -> list[dict]  (available ground-truth events)
- get_ground_truth(event_id: str) -> np.ndarray[Any, Any]  (binary labels)
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
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from omni_mercury_engine.datasets.exceptions import OfflineModeError
from omni_mercury_engine.security.safe_http import SafeHTTPClient

logger = logging.getLogger(__name__)

# Default cache directory for downloaded data
_DEFAULT_CACHE_DIR = Path(
    os.environ.get("MERCURY_CACHE_DIR", str(Path.home() / ".mercury" / "cache"))
)


class FetchHTTPError(ConnectionError):
    """Retry-exhausted fetch failure that preserves the HTTP status.

    ``_fetch_url`` used to flatten every transport failure into a bare
    ``ConnectionError`` whose message carried only the exception class name,
    so a throttling response (HTTP 429) was indistinguishable from a DNS
    outage without walking ``__cause__``. Subclassing ``ConnectionError``
    keeps every existing ``except ConnectionError`` / ``except OSError``
    consumer working unchanged while ``status_code`` lets callers branch on
    the actual upstream verdict (mirrors
    :class:`omni_mercury_engine.datasets.exceptions.DataSourceUnavailableError`).

    The underlying transport exception is never chained (``__cause__`` is
    ``None`` and the implicit context is suppressed): requests/urllib3
    error messages embed the fully-composed request URL, and for keyed
    loaders that URL carries the credential, so ``status_code`` — not the
    chain — is the supported way to inspect the upstream verdict.

    Attributes:
        status_code: HTTP status of the last failed attempt, or ``None``
            when the failure never reached the HTTP layer (DNS, timeout,
            refused connection).
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Initialize with a message and the last observed HTTP status.

        Args:
            message: Operator-facing failure description.
            status_code: HTTP status of the last failed attempt, if any.
        """
        super().__init__(message)
        self.status_code = status_code


class BaseDomainLoader(ABC):
    """Base class for all domain data loaders.

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

    #: Additional environment-variable names accepted for the API key, tried in
    #: order after :attr:`API_KEY_ENV_VAR` is found unset. This exists so a loader
    #: stays wired when the deployment/secret name differs from the canonical one
    #: (e.g. the NASA FIRMS key is documented as ``NASA_FIRMS_MAP_KEY`` but the
    #: repository secret is ``FIRMS_MAP_KEY``). The canonical name remains
    #: :attr:`API_KEY_ENV_VAR`; fallbacks are a compatibility safety net, not a
    #: rename. Empty by default, so loaders that don't set it are unchanged.
    API_KEY_ENV_FALLBACKS: tuple[str, ...] = ()

    #: Cache TTL in seconds (default 1 hour)
    CACHE_TTL: int = 3600

    # Label provenance, declared at source. One of the values in
    # ``datasets.metadata.VALID_LABEL_SOURCES``. Loaders that manufacture
    # anomaly labels by thresholding a scored feature (``magnitude >= cut``,
    # ``kp >= 7``, ``FRP >= p90``, a z-score fence) or by synthetic
    # reconstruction MUST override this to ``"statistical"`` so the
    # governed-fusion headline excludes them as circular and the autonomous
    # fitness signal reads only transparently-labelled events. Defaults to
    # ``"ground_truth"``; override to be transparent. The frozen audit lives in
    # ``omni_mercury_engine.loaders.label_provenance.LABEL_PROVENANCE_REGISTRY``
    # and the CI gate in ``tests/loaders/test_label_provenance_gate.py``.
    LABEL_SOURCE: str = "ground_truth"

    def __init__(
        self,
        cache_dir: Path | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: int = 60,
    ) -> None:
        """Initialize the domain loader.

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

        # Resolve API key: explicit arg, then the canonical env var, then any
        # documented fallback names (deployment/secret-name compatibility).
        if api_key:
            self._api_key = api_key
        elif self.REQUIRES_API_KEY and self.API_KEY_ENV_VAR:
            self._api_key = os.environ.get(self.API_KEY_ENV_VAR, "")
            for fallback in self.API_KEY_ENV_FALLBACKS:
                if self._api_key:
                    break
                self._api_key = os.environ.get(fallback, "")
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
        """Fetch most recent data from live source.

        Returns:
            DataFrame with domain-specific features as columns.

        Raises:
            DataSourceUnavailableError: If the API is unreachable.
        """
        ...

    @abstractmethod
    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch data for a specific historical event.

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
        """Return catalog of events with ground truth.

        Returns:
            List of dicts, each with at least:
                - "event_id": str
                - "name": str
                - "date": str (ISO format)
                - "description": str
        """
        ...

    @abstractmethod
    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Return binary anomaly labels for an event.

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

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform raw data into feature matrix for Mercury detector.

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
        """Fetch URL content via :class:`SafeHTTPClient` with retry logic.

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
            FetchHTTPError: All transient retries exhausted on
                network / HTTP errors (a ``ConnectionError`` subclass,
                so existing handlers keep working). Carries the last
                observed HTTP status in ``status_code`` — ``None`` for
                pre-HTTP failures. The underlying exception is
                deliberately NOT chained (``__cause__ is None``,
                implicit context suppressed): requests/urllib3 error
                messages embed the fully-composed request URL, and for
                keyed loaders that URL carries the credential (the
                FIRMS MAP key as a path segment; EIA / OpenWeatherMap
                / Alpha Vantage / NASA keys as query parameters), so
                chaining would leak the secret into every rendered
                traceback and log. The safe diagnostics live in the
                message instead: target host, attempt count, exception
                class name, and HTTP status.
                HTTP 429 fails fast without further attempts: a
                windowed quota cannot recover inside a 2-8 s backoff,
                and retrying only multiplies the burn against the
                upstream's limit.
        """
        default_headers = {"User-Agent": "Mercury-Agent/1.0 (Steel Security Advisors)"}
        if headers:
            default_headers.update(headers)

        last_error_kind = "unknown"
        last_status: int | None = None
        attempts_made = 0
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
            except OfflineModeError:
                # Offline mode is a deterministic, pre-socket refusal
                # (``SafeHTTPClient.validate_url`` raises it before any
                # network call), not a transient fault. It is a
                # ``RuntimeError``, so without this it would fall into
                # the generic handler below and be retried through the
                # full 2+4+8 s backoff and then masked as a
                # ``FetchHTTPError`` -- turning an instant, typed
                # fail-closed into a ~14 s wait that loses the offline
                # signal. Re-raise immediately and untouched, mirroring
                # the ``ValueError`` branch.
                raise
            except Exception as exc:
                last_error_kind = type(exc).__name__
                attempts_made = attempt + 1
                # Duck-typed so ``requests`` stays a deferred import:
                # requests.HTTPError carries .response.status_code.
                last_status = getattr(getattr(exc, "response", None), "status_code", None)
                if last_status == 429:
                    # A rate limit is a quota verdict, not a transient
                    # fault: the window resets on the upstream's clock,
                    # not on our backoff, so further attempts only burn
                    # more of the same quota (FIRMS: ~4 min of blind
                    # retries per tripped call before this guard).
                    break
                if attempt < self.max_retries:
                    wait = self.retry_backoff * (2**attempt)
                    logger.warning(
                        "%s fetch attempt %d/%d failed (%s%s). Retrying in %.1fs.",
                        self.DOMAIN,
                        attempt + 1,
                        self.max_retries + 1,
                        last_error_kind,
                        f", HTTP {last_status}" if last_status is not None else "",
                        wait,
                    )
                    time.sleep(wait)

        # SECURITY: never chain ``last_exc``. requests.HTTPError messages
        # embed the fully-composed request URL ("404 Client Error: Not
        # Found for url: https://host/path?query") and urllib3 connection
        # errors embed the path+query ("Max retries exceeded with url:
        # /path?query"). For keyed loaders that URL carries the
        # credential — the FIRMS MAP key is a path segment; EIA /
        # OpenWeatherMap / Alpha Vantage / NASA keys ride in query
        # params — so an explicit cause (or the implicit context) leaks
        # the secret into every traceback and log that renders the
        # chain. ``from None`` severs both (PEP 409: ``__cause__ =
        # None``, ``__suppress_context__ = True``); the message keeps
        # every diagnostic that is safe to keep: host, attempt count,
        # exception class name, HTTP status. Pinned by
        # ``tests/loaders/test_base_loader.py::TestFetchCredentialRedaction``.
        host = urlparse(url).hostname or "unknown-host"
        status_detail = f", HTTP {last_status}" if last_status is not None else ""
        raise FetchHTTPError(
            f"{self.DOMAIN}: Failed to fetch data from {host} after "
            f"{attempts_made} attempt{'s' if attempts_made != 1 else ''} "
            f"({last_error_kind}{status_detail})",
            status_code=last_status,
        ) from None

    def _fetch_json(
        self,
        url: str,
        params: dict[str, str] | None = None,
    ) -> Any:
        """Fetch and parse JSON from URL.

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
        """Fetch and parse CSV from URL.

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

    @staticmethod
    def _iter_feed_rows(raw: Any, columns: tuple[str, ...]) -> list[dict[str, Any]]:
        """Normalise a JSON table feed into per-row field dictionaries.

        Upstream feeds serve two shapes for the same logical table and have
        migrated between them without warning (SWPC did exactly this):

        * **array-of-arrays** -- the first row is a header naming the columns,
          every later row is positional.
        * **array-of-objects** -- each row is already a mapping keyed by
          column name, with no header row.

        Reading a mapping row positionally raises ``KeyError: 1``, which is
        precisely how the ``noaa-planetary-k-index`` migration surfaced (and
        went unseen for three weekly runs). This helper accepts either shape
        so a future flip in either direction is a no-op rather than an
        outage. It lives on the base loader so every JSON-table consumer
        (SWPC, DONKI, JPL fireball) shares one absorber instead of each
        loader hardcoding one shape.

        Args:
            raw: Decoded JSON body of the feed.
            columns: Field names in positional order, used to key the
                array-of-arrays form and to select from the object form.

        Returns:
            One dictionary per data row, keyed by *columns*. Missing fields
            are ``None``. Returns ``[]`` for an empty or unrecognised body.
        """
        if not isinstance(raw, list) or not raw:
            return []

        if isinstance(raw[0], dict):
            # Object rows: select the requested fields, tolerating a feed that
            # carries extra columns (RTSW ships ~30 alongside the 4 we use).
            return [
                {name: row.get(name) for name in columns} for row in raw if isinstance(row, dict)
            ]

        # Positional rows: the first row is the header and is not data. When
        # the header actually names every requested column, map by name -- a
        # feed that carries extra columns or reorders them then still parses
        # correctly instead of silently mis-mapping positions.
        header = raw[0]
        if (
            isinstance(header, (list, tuple))
            and all(isinstance(name, str) for name in header)
            and set(columns) <= set(header)
        ):
            positions = {name: header.index(name) for name in columns}
            return [
                {name: (row[pos] if pos < len(row) else None) for name, pos in positions.items()}
                for row in raw[1:]
                if isinstance(row, (list, tuple))
            ]

        # Header does not name the requested columns: fall back to the
        # caller's positional order.
        return [
            dict(zip(columns, row))
            for row in raw[1:]
            if isinstance(row, (list, tuple)) and len(row) >= len(columns)
        ]

    # =========================================================================
    # Caching
    # =========================================================================

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key.

        Args:
            key: Cache key (will be hashed for filename).

        Returns:
            Path to cached file.
        """
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hashed}.json"

    def _read_cache(self, key: str) -> Any | None:
        """Read data from cache if valid (not expired).

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
        """Write data to cache.

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
    def compute_data_hash(data: np.ndarray[Any, Any]) -> str:
        """Compute SHA-256 hash of data array for provenance tracking.

        Args:
            data: Numpy array to hash.

        Returns:
            Hex digest of SHA-256 hash.
        """
        return hashlib.sha256(data.tobytes()).hexdigest()

    def get_provenance(self, event_id: str, data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Generate provenance metadata for benchmark results.

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
