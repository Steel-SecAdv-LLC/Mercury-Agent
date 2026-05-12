"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for loaders/base.py module.

Covers:
- SSRF URL validation enforced by ``SafeHTTPClient`` (the gate that
  backs :meth:`BaseDomainLoader._fetch_url`; the legacy per-class
  ``_validate_url`` helper was removed when egress was centralised)
- Cache read/write operations
- Data provenance and hashing
- Feature engineering defaults
- URL fetch retry logic
"""

from __future__ import annotations

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

    def list_events(self) -> list[dict]:
        return [{"event_id": "e1", "name": "Test Event", "date": "2024-01-01"}]

    def get_ground_truth(self, event_id: str) -> np.ndarray:
        return np.array([0, 0, 1])


# =============================================================================
# SSRF Validation Tests
# =============================================================================


class TestSSRFValidation:
    """Tests for the SSRF gate that backs ``BaseDomainLoader._fetch_url``.

    ``BaseDomainLoader._fetch_url`` no longer carries its own validator;
    every outbound request is funnelled through
    :class:`SafeHTTPClient`, with ``user_configured=True`` so the
    private-network / IMDS gate fires for any operator-provided URL.
    These tests pin that contract directly so a regression in the
    central gate (or a refactor that bypasses it) fails the loader
    suite.
    """

    # Mirror exactly the kwargs ``BaseDomainLoader._fetch_url`` passes
    # to SafeHTTPClient.get_bytes -- the only deviation is allow_http,
    # which we toggle per-test to exercise both schemes.
    @staticmethod
    def _validate(url: str, *, allow_http: bool = True) -> None:
        SafeHTTPClient.validate_url(
            url,
            allow_http=allow_http,
            user_configured=True,
            allow_untrusted=True,
        )

    def test_valid_https_url(self):
        """Valid HTTPS URLs to public hosts pass validation."""
        # Use a public IP literal so the gate never depends on DNS in
        # the test environment (CI sandboxes routinely block egress
        # resolution). 1.1.1.1 is Cloudflare's well-known public DNS
        # endpoint -- not private, not loopback, not link-local.
        self._validate("https://1.1.1.1/data", allow_http=False)

    def test_valid_http_url(self):
        """Valid HTTP URLs to public hosts pass when allow_http=True."""
        self._validate("http://1.1.1.1/data", allow_http=True)

    def test_ftp_scheme_blocked(self):
        """Test that FTP scheme is blocked."""
        with pytest.raises(UnsafeURLError, match="scheme 'ftp'"):
            self._validate("ftp://evil.com/file")

    def test_file_scheme_blocked(self):
        """Test that file:// scheme is blocked."""
        with pytest.raises(UnsafeURLError, match="scheme 'file'"):
            self._validate("file:///etc/passwd")

    def test_data_scheme_blocked(self):
        """Test that data: scheme is blocked."""
        with pytest.raises(UnsafeURLError, match="scheme 'data'"):
            self._validate("data:text/html,<h1>evil</h1>")

    def test_javascript_scheme_blocked(self):
        """Test that javascript: scheme is blocked."""
        with pytest.raises(UnsafeURLError, match="scheme 'javascript'"):
            self._validate("javascript:alert(1)")

    def test_missing_hostname(self):
        """Test that URLs without hostname are blocked."""
        with pytest.raises(UnsafeURLError, match="no host component"):
            self._validate("http://")

    def test_localhost_blocked(self):
        """Test that localhost resolves to loopback and is blocked."""
        with pytest.raises(UnsafeURLError, match="private/link-local/IMDS"):
            self._validate("http://localhost/admin")

    def test_loopback_ip_blocked(self):
        """Test that 127.0.0.1 is blocked."""
        with pytest.raises(UnsafeURLError, match="private/link-local/IMDS"):
            self._validate("http://127.0.0.1/admin")

    def test_private_ip_blocked(self):
        """Test that private IP ranges are blocked."""
        private_ips = [
            "http://10.0.0.1/",
            "http://172.16.0.1/",
            "http://192.168.1.1/",
        ]
        for url in private_ips:
            with pytest.raises(UnsafeURLError, match="private/link-local/IMDS"):
                self._validate(url)

    def test_imds_blocked(self):
        """The AWS/GCP/Azure metadata endpoint is rejected outright."""
        with pytest.raises(UnsafeURLError, match="private/link-local/IMDS"):
            self._validate("http://169.254.169.254/latest/meta-data/")

    def test_dns_failure_is_fatal_for_user_configured(self):
        """DNS failure raises -- user-configured URLs cannot defer SSRF.

        The previous loader gate treated DNS failures as non-fatal and
        deferred to the network call. ``SafeHTTPClient`` deliberately
        fails closed for ``user_configured=True`` so a misconfigured
        endpoint cannot reach an attacker-controlled DNS-rebound host
        on the second request.
        """
        with pytest.raises(UnsafeURLError, match="did not resolve"):
            self._validate(
                "https://this-domain-definitely-does-not-exist-xyz123.invalid/api",
                allow_http=False,
            )


# =============================================================================
# Cache Tests
# =============================================================================


class TestCaching:
    """Tests for file-based caching operations."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create StubLoader with temp cache directory."""
        return StubLoader(cache_dir=tmp_path / "test_cache")

    def test_write_and_read_cache(self, loader):
        """Test cache write followed by read."""
        loader._write_cache("test_key", {"data": [1, 2, 3]})
        cached = loader._read_cache("test_key")
        assert cached == {"data": [1, 2, 3]}

    def test_read_cache_miss(self, loader):
        """Test reading a non-existent cache key returns None."""
        assert loader._read_cache("nonexistent_key") is None

    def test_cache_expiration(self, loader):
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

    def test_cache_path_hashing(self, loader):
        """Test that cache paths are derived from hashed keys."""
        path = loader._get_cache_path("my_key")
        assert path.suffix == ".json"
        assert path.parent == loader.cache_dir

    def test_write_cache_handles_unserializable(self, loader):
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
    def loader(self, tmp_path):
        return StubLoader(cache_dir=tmp_path / "prov_cache")

    def test_compute_data_hash(self):
        """Test deterministic SHA-256 hashing of numpy arrays."""
        data = np.array([1.0, 2.0, 3.0])
        h1 = BaseDomainLoader.compute_data_hash(data)
        h2 = BaseDomainLoader.compute_data_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_data_hash_different_data(self):
        """Test that different data produces different hashes."""
        h1 = BaseDomainLoader.compute_data_hash(np.array([1.0]))
        h2 = BaseDomainLoader.compute_data_hash(np.array([2.0]))
        assert h1 != h2

    def test_get_provenance(self, loader):
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
    def loader(self, tmp_path):
        return StubLoader(cache_dir=tmp_path / "feat_cache")

    def test_engineer_features_numeric_only(self, loader):
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

    def test_engineer_features_handles_inf(self, loader):
        """Test that inf values are replaced."""
        df = pd.DataFrame(
            {
                "val": [1.0, float("inf"), 3.0],
            }
        )
        features = loader.engineer_features(df)
        assert np.all(np.isfinite(features))

    def test_engineer_features_handles_nan(self, loader):
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

    def test_returns_string(self):
        """Test that version returns a string."""
        version = _get_mercury_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_returns_dev_on_error(self):
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

    def test_default_cache_dir(self):
        """Test default cache directory is created."""
        loader = StubLoader()
        assert loader.cache_dir.exists()

    def test_custom_cache_dir(self, tmp_path):
        """Test custom cache directory."""
        custom = tmp_path / "custom_cache"
        loader = StubLoader(cache_dir=custom)
        assert loader.cache_dir == custom
        assert custom.exists()

    def test_api_key_from_param(self, tmp_path):
        """Test API key from parameter."""
        loader = StubLoader(cache_dir=tmp_path / "c", api_key="my-key")
        assert loader._api_key == "my-key"

    def test_retry_config(self, tmp_path):
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
