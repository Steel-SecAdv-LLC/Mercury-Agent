# Copyright (C) 2025 Steel Security Advisors LLC
"""Comprehensive tests for loaders/base.py module."""

from __future__ import annotations

import ipaddress
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.loaders.base import BaseDomainLoader, _get_mercury_version
from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError

# =============================================================================
# Concrete test implementation of the abstract BaseDomainLoader
# =============================================================================


class StubLoader(BaseDomainLoader):
    """Concrete implementation of BaseDomainLoader for testing."""

    DOMAIN = "test"
    SOURCE_URL = "https://example.com/api"

    def fetch_realtime(self) -> pd.DataFrame:
        return pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        return pd.DataFrame({"value": [1.0, 2.0, 3.0], "label": [0, 0, 1]})

    def list_events(self) -> list[dict[str, Any]]:
        return [{"event_id": "e1", "name": "Test Event", "date": "2024-01-01"}]

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        return np.array([0, 0, 1])


# =============================================================================
# SSRF Validation Tests
# =============================================================================


class TestSSRFValidation:
    """Tests for the SSRF gate that backs ``BaseDomainLoader._fetch_url``.

    ``_fetch_url`` no longer carries its own validator: every outbound
    request is funnelled through :class:`SafeHTTPClient` with HTTPS-only
    + ``TrustedEndpoints.TRUSTED_DOMAINS`` allowlist enforcement. These
    tests pin that contract directly so a regression in the central
    gate (or a refactor that bypasses it via ``user_configured=True``)
    fails the loader suite.

    The IP-resolution gate (private / loopback / IMDS) lives in
    :class:`SafeHTTPClient` and is exhaustively tested in
    ``tests/security/test_safe_http.py``. It does not fire for loader
    ``https://`` URLs because the allowlist already constrains the
    host set; duplicating those assertions here would create dead
    coverage that drifts as the gate evolves.
    """

    @staticmethod
    def _validate(url: str) -> None:
        """Mirror exactly the kwargs ``_fetch_url`` passes to ``get_bytes``."""
        SafeHTTPClient.validate_url(url)

    def test_trusted_https_url_passes(self) -> None:
        """A class-constant dataset URL on the allowlist passes."""
        # earthquake.usgs.gov is in TRUSTED_DOMAINS; this is the
        # canonical loader URL pattern.
        self._validate("https://earthquake.usgs.gov/fdsnws/event/1/query")

    def test_untrusted_host_blocked(self) -> None:
        """An HTTPS URL outside TRUSTED_DOMAINS is refused."""
        with pytest.raises(UnsafeURLError, match="not in trusted"):
            self._validate("https://attacker.example.com/exfil")

    def test_untrusted_host_has_no_loader_escape_hatch(self) -> None:
        """Loader egress has no per-call bypass for TRUSTED_DOMAINS."""
        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("8.8.8.8")],
            ),
            pytest.raises(UnsafeURLError, match="not in trusted"),
        ):
            self._validate("https://attacker.example.com/exfil")

    def test_user_configured_host_with_private_ip_blocked_without_allow_private(self) -> None:
        """Operator-configured hosts still hit the SSRF / IMDS gate.

        Dynamic endpoints belong on the explicit ``user_configured``
        path, not on a loader-specific allowlist bypass.  That path
        accepts operator-chosen public hosts while refusing private
        pivots unless ``allow_private=True`` is also set.
        """
        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("10.0.0.5")],
            ),
            pytest.raises(UnsafeURLError, match="private/link-local/IMDS"),
        ):
            SafeHTTPClient.validate_url(
                "https://attacker.example.com/exfil",
                user_configured=True,
            )

    def test_http_scheme_blocked_for_trusted_host(self) -> None:
        """Plain HTTP is rejected even for an allowlisted host."""
        with pytest.raises(UnsafeURLError, match="scheme 'http'"):
            self._validate("http://earthquake.usgs.gov/path")

    def test_ftp_scheme_blocked(self) -> None:
        """``ftp://`` is never permitted."""
        with pytest.raises(UnsafeURLError, match="scheme 'ftp'"):
            self._validate("ftp://evil.com/file")

    def test_file_scheme_blocked(self) -> None:
        """``file://`` is never permitted."""
        with pytest.raises(UnsafeURLError, match="scheme 'file'"):
            self._validate("file:///etc/passwd")

    def test_data_scheme_blocked(self) -> None:
        """``data:`` is never permitted."""
        with pytest.raises(UnsafeURLError, match="scheme 'data'"):
            self._validate("data:text/html,<h1>evil</h1>")

    def test_javascript_scheme_blocked(self) -> None:
        """``javascript:`` is never permitted."""
        with pytest.raises(UnsafeURLError, match="scheme 'javascript'"):
            self._validate("javascript:alert(1)")

    def test_missing_hostname(self) -> None:
        """A URL with no host raises before any allowlist or DNS work."""
        # The scheme gate runs first; ``https://`` with empty netloc
        # falls through scheme check then trips the missing-host
        # branch.
        with pytest.raises(UnsafeURLError, match="no host component"):
            self._validate("https://")


class TestFetchUrlExceptionRouting:
    """Configuration faults raised by ``SafeHTTPClient`` must NOT retry.

    ``_fetch_url`` historically caught every exception in a single
    broad ``except Exception:`` block and retried up to ``max_retries``
    times, then re-raised the original failure wrapped in
    ``ConnectionError``. That masked the real cause when the failure
    was a configuration fault (off-allowlist host, bad scheme,
    malformed URL): a refused SSRF pivot surfaced to the operator
    as a generic connection failure several seconds later.

    The loader now splits the catch surface: ``UnsafeURLError`` /
    ``ValueError`` re-raise immediately on the first attempt; only
    transient network / HTTP errors flow into the retry loop.
    """

    def test_unsafe_url_raised_immediately_no_retry(self, tmp_path: Any) -> None:
        """Off-allowlist URL surfaces ``UnsafeURLError`` on attempt 1."""
        loader = StubLoader(cache_dir=tmp_path / "cache")
        # Track how many times the underlying gate is invoked. A retry
        # would call ``get_bytes`` more than once; we want exactly one.
        call_count = {"n": 0}
        real = SafeHTTPClient.get_bytes

        def tracked(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            return real(*args, **kwargs)

        with (
            patch(
                "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes",
                side_effect=tracked,
            ),
            pytest.raises(UnsafeURLError, match="not in trusted"),
        ):
            loader._fetch_url("https://attacker.example.com/exfil")
        assert call_count["n"] == 1, "UnsafeURLError must NOT trigger retries"

    def test_scheme_error_raised_immediately_no_retry(self, tmp_path: Any) -> None:
        """Bad scheme surfaces immediately, no retry storm."""
        loader = StubLoader(cache_dir=tmp_path / "cache")
        call_count = {"n": 0}
        real = SafeHTTPClient.get_bytes

        def tracked(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            return real(*args, **kwargs)

        with (
            patch(
                "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes",
                side_effect=tracked,
            ),
            pytest.raises(UnsafeURLError, match="scheme"),
        ):
            loader._fetch_url("ftp://earthquake.usgs.gov/data")
        assert call_count["n"] == 1

    def test_transient_network_error_still_retries(self, tmp_path: Any) -> None:
        """``OSError`` from the gate is treated as transient and retried."""
        loader = StubLoader(cache_dir=tmp_path / "cache")
        loader.max_retries = 2  # 3 total attempts
        loader.retry_backoff = 0.0  # no sleep
        call_count = {"n": 0}

        def transient(*args: Any, **kwargs: Any) -> None:
            call_count["n"] += 1
            raise OSError("simulated transient socket failure")

        with (
            patch(
                "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes",
                side_effect=transient,
            ),
            pytest.raises(ConnectionError, match="Failed to fetch data"),
        ):
            loader._fetch_url("https://earthquake.usgs.gov/fdsnws/event/1/query")
        # 1 initial + 2 retries == 3 attempts. Confirms transient
        # errors still flow into the retry loop and were not
        # accidentally re-routed by the new ValueError branch.
        assert call_count["n"] == 3

    def test_retry_exhaustion_chains_underlying_exception(self, tmp_path: Any) -> None:
        """``ConnectionError`` after retry-exhaustion chains via ``__cause__``.

        Wrapping the failure in ``ConnectionError`` is the operator-
        facing API contract, but losing the underlying exception in
        the traceback makes diagnosis harder than it has to be.
        PR #210 wires ``raise ConnectionError(...) from last_exc`` so
        the original socket / HTTP failure is one frame away.
        """
        loader = StubLoader(cache_dir=tmp_path / "cache")
        loader.max_retries = 1
        loader.retry_backoff = 0.0

        original = OSError("simulated transient socket failure")

        with (
            patch(
                "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes",
                side_effect=original,
            ),
            pytest.raises(ConnectionError) as exc_info,
        ):
            loader._fetch_url("https://earthquake.usgs.gov/fdsnws/event/1/query")

        assert exc_info.value.__cause__ is original, (
            "ConnectionError did not chain to the underlying exception; "
            "operators lose the real cause in the traceback."
        )


class TestAllowUntrustedRemovedFromLoader:
    """The loader API surface MUST not accept the removed
    ``allow_untrusted`` keyword.

    PR #210 deletes the per-call escape hatch from ``_fetch_url``.
    Operators that previously used it should switch to calling
    :class:`SafeHTTPClient` directly with ``user_configured=True``;
    documenting that migration is the job of
    ``TestMigrationFromAllowUntrusted`` in
    ``tests/security/test_safe_http.py``. This test pins the
    loader-side removal so a stale call-site does not creep back
    in.
    """

    def test_fetch_url_rejects_allow_untrusted_kwarg(self, tmp_path: Any) -> None:
        loader = StubLoader(cache_dir=tmp_path / "cache")
        with pytest.raises(TypeError, match="allow_untrusted"):
            loader._fetch_url(  # type: ignore[call-arg]
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                allow_untrusted=True,
            )

    def test_fetch_url_signature_has_no_allow_untrusted(self) -> None:
        """Belt-and-braces: the parameter is not present in the signature."""
        import inspect

        sig = inspect.signature(BaseDomainLoader._fetch_url)
        assert "allow_untrusted" not in sig.parameters


# =============================================================================
# Cache Tests
# =============================================================================


class TestCaching:
    """Tests for file-based caching operations."""

    @pytest.fixture
    def loader(self, tmp_path: Any) -> Any:
        """Create StubLoader with temp cache directory."""
        return StubLoader(cache_dir=tmp_path / "test_cache")

    def test_write_and_read_cache(self, loader: Any) -> None:
        """Test cache write followed by read."""
        loader._write_cache("test_key", {"data": [1, 2, 3]})
        cached = loader._read_cache("test_key")
        assert cached == {"data": [1, 2, 3]}

    def test_read_cache_miss(self, loader: Any) -> None:
        """Test reading a non-existent cache key returns None."""
        assert loader._read_cache("nonexistent_key") is None

    def test_cache_expiration(self, loader: Any) -> None:
        """Test that expired cache entries return None."""
        loader._write_cache("expiring_key", {"data": "old"})

        # Manually expire by setting TTL to 0
        original_ttl = loader.CACHE_TTL
        loader.CACHE_TTL = 0

        import time

        time.sleep(0.01)

        result = loader._read_cache("expiring_key")
        assert result is None

        loader.CACHE_TTL = original_ttl

    def test_cache_path_hashing(self, loader: Any) -> None:
        """Test that cache paths are derived from hashed keys."""
        path = loader._get_cache_path("my_key")
        assert path.suffix == ".json"
        assert path.parent == loader.cache_dir

    def test_write_cache_handles_unserializable(self, loader: Any) -> None:
        """Test cache write handles non-serializable data gracefully."""
        # numpy array is not JSON-serializable
        loader._write_cache("bad_key", np.array([1, 2, 3]))
        # Should not raise, just log debug


# =============================================================================
# Data Provenance Tests
# =============================================================================


class TestDataProvenance:
    """Tests for data provenance tracking."""

    @pytest.fixture
    def loader(self, tmp_path: Any) -> Any:
        return StubLoader(cache_dir=tmp_path / "prov_cache")

    def test_compute_data_hash(self) -> None:
        """Test deterministic SHA-256 hashing of numpy arrays."""
        data = np.array([1.0, 2.0, 3.0])
        h1 = BaseDomainLoader.compute_data_hash(data)
        h2 = BaseDomainLoader.compute_data_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_data_hash_different_data(self) -> None:
        """Test that different data produces different hashes."""
        h1 = BaseDomainLoader.compute_data_hash(np.array([1.0]))
        h2 = BaseDomainLoader.compute_data_hash(np.array([2.0]))
        assert h1 != h2

    def test_get_provenance(self, loader: Any) -> None:
        """Test provenance metadata generation."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        prov = loader.get_provenance("event_1", data)

        assert prov["domain"] == "test"
        assert prov["event_id"] == "event_1"
        assert "timestamp" in prov
        assert "data_hash" in prov
        assert prov["data_shape"] == [2, 2]
        assert "mercury_version" in prov
        assert prov["source_url"] == "https://example.com/api"


# =============================================================================
# Feature Engineering Tests
# =============================================================================


class TestFeatureEngineering:
    """Tests for default feature engineering."""

    @pytest.fixture
    def loader(self, tmp_path: Any) -> Any:
        return StubLoader(cache_dir=tmp_path / "feat_cache")

    def test_engineer_features_numeric_only(self, loader: Any) -> None:
        """Test that only numeric columns are extracted."""
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0],
                "num2": [4.0, 5.0, 6.0],
                "text": ["a", "b", "c"],
            }
        )
        features = loader.engineer_features(df)
        assert features.shape == (3, 2)

    def test_engineer_features_handles_inf(self, loader: Any) -> None:
        """Test that inf values are replaced."""
        df = pd.DataFrame(
            {
                "val": [1.0, float("inf"), 3.0],
            }
        )
        features = loader.engineer_features(df)
        assert np.all(np.isfinite(features))

    def test_engineer_features_handles_nan(self, loader: Any) -> None:
        """Test that NaN values are filled with median."""
        df = pd.DataFrame(
            {
                "val": [1.0, float("nan"), 3.0, 5.0],
            }
        )
        features = loader.engineer_features(df)
        assert np.all(np.isfinite(features))
        # NaN should be replaced with median of [1.0, 3.0, 5.0] = 3.0
        assert features[1, 0] == 3.0


# =============================================================================
# Version Helper Tests
# =============================================================================


class TestVersionHelper:
    """Tests for _get_mercury_version helper."""

    def test_returns_string(self) -> None:
        """Test that version returns a string."""
        version = _get_mercury_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_returns_dev_on_error(self) -> None:
        """Test that import error returns 'dev'."""
        from importlib.metadata import PackageNotFoundError

        with patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("omni-mercury-engine"),
        ):
            version = _get_mercury_version()
            assert version == "dev"


# =============================================================================
# Loader Initialization Tests
# =============================================================================


class TestLoaderInitialization:
    """Tests for BaseDomainLoader initialization."""

    def test_default_cache_dir(self) -> None:
        """Test default cache directory is created."""
        loader = StubLoader()
        assert loader.cache_dir.exists()

    def test_custom_cache_dir(self, tmp_path: Any) -> None:
        """Test custom cache directory."""
        custom = tmp_path / "custom_cache"
        loader = StubLoader(cache_dir=custom)
        assert loader.cache_dir == custom
        assert custom.exists()

    def test_api_key_from_param(self, tmp_path: Any) -> None:
        """Test API key from parameter."""
        loader = StubLoader(cache_dir=tmp_path / "c", api_key="my-key")
        assert loader._api_key == "my-key"

    def test_retry_config(self, tmp_path: Any) -> None:
        """Test retry configuration."""
        loader = StubLoader(
            cache_dir=tmp_path / "c",
            max_retries=5,
            retry_backoff=3.0,
            timeout=120,
        )
        assert loader.max_retries == 5
        assert loader.retry_backoff == 3.0
        assert loader.timeout == 120
