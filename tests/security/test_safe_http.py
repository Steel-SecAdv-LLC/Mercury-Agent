"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

Gate tests for :mod:`omni_mercury_engine.security.safe_http`.

Each test pins one rejection path so a regression that lets the
underlying urllib pattern back into the codebase (or weakens the
SafeHTTPClient gates) trips a unit test before it ships.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError


class TestSchemeGate:
    """The scheme allowlist refuses anything outside http(s)."""

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/data",
            "gopher://example.com/",
            "data:text/plain,hello",
            "javascript:alert(1)",
        ],
    )
    def test_unsupported_scheme_rejected(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            SafeHTTPClient.validate_url(url)

    def test_http_rejected_by_default(self) -> None:
        with pytest.raises(UnsafeURLError, match="scheme"):
            SafeHTTPClient.validate_url("http://earthquake.usgs.gov/something")

    def test_http_permitted_with_opt_in(self) -> None:
        # Trusted host on plain http with the explicit opt-in should
        # pass the scheme gate (the host is in TRUSTED_DOMAINS).
        # The post-Copilot tightening also runs the private-network
        # gate for http:// URLs, so we patch the resolver to return a
        # public IP.
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address("23.215.0.137")],  # public USGS-like IP
        ):
            SafeHTTPClient.validate_url(
                "http://earthquake.usgs.gov/path",
                allow_http=True,
            )

    def test_no_host_rejected(self) -> None:
        with pytest.raises(UnsafeURLError, match="no host"):
            SafeHTTPClient.validate_url("https://")

    def test_http_unlisted_host_rejected_even_with_allow_http(self) -> None:
        # The bug surfaced by Copilot review: previously allow_http=True
        # bypassed the trusted-allowlist gate entirely, so a plain-HTTP
        # mirror could reach an arbitrary host. The fix asserts the
        # allowlist for http:// too.
        with pytest.raises(UnsafeURLError, match="not in trusted allowlist"):
            SafeHTTPClient.validate_url(
                "http://evil.example.com/path",
                allow_http=True,
            )

    def test_http_to_private_ip_rejected_even_with_trusted_host(self) -> None:
        # Even when the host is allowlisted, an http:// URL goes through
        # the private-network gate so a DNS-rebinding to RFC1918 cannot
        # reach internal infrastructure.
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("10.0.0.5")],
            ),
            pytest.raises(UnsafeURLError, match="private/link-local/IMDS"),
        ):
            SafeHTTPClient.validate_url(
                "http://earthquake.usgs.gov/path",
                allow_http=True,
            )


class TestTrustedDomainsGate:
    """class-constant URLs must come from the TrustedEndpoints allowlist."""

    def test_unlisted_host_rejected(self) -> None:
        with pytest.raises(UnsafeURLError, match="not in trusted allowlist"):
            SafeHTTPClient.validate_url("https://evil.example.com/path")

    def test_listed_host_passes(self) -> None:
        # earthquake.usgs.gov is in TRUSTED_DOMAINS
        SafeHTTPClient.validate_url("https://earthquake.usgs.gov/fdsnws/event/1/query")

    def test_listed_host_with_explicit_port_passes(self) -> None:
        # The Copilot review caught that the gate was matching against
        # parsed.netloc (which includes the port), so a URL like
        # https://earthquake.usgs.gov:443/... would be rejected even
        # though the host is allowlisted.  This test pins the fix.
        SafeHTTPClient.validate_url("https://earthquake.usgs.gov:443/path")

    def test_user_configured_bypasses_allowlist_but_resolves_host(self) -> None:
        # Public DNS that is not in the allowlist; user_configured
        # opts out of the allowlist but still runs the private-network
        # gate against the resolved IPs.
        with patch("omni_mercury_engine.security.safe_http._resolve_ips") as resolve:
            import ipaddress

            resolve.return_value = [ipaddress.ip_address("93.184.216.34")]  # public TEST-IP
            # No exception expected.
            SafeHTTPClient.validate_url(
                "https://api.unknown-domain.example/",
                user_configured=True,
            )


class TestPrivateNetworkGate:
    """user_configured URLs are blocked from RFC1918 / link-local / IMDS."""

    @pytest.mark.parametrize(
        "ip",
        [
            "10.0.0.5",  # RFC1918
            "172.16.0.5",  # RFC1918
            "192.168.1.1",  # RFC1918
            "169.254.169.254",  # IMDS
            "127.0.0.1",  # loopback (still private)
            "0.0.0.0",  # noqa: S104 - unspecified; the test asserts the gate refuses this
        ],
    )
    def test_private_ip_rejected(self, ip: str) -> None:
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address(ip)],
            ),
            pytest.raises(UnsafeURLError, match="private/link-local/IMDS"),
        ):
            SafeHTTPClient.validate_url(
                "https://attacker.example/",
                user_configured=True,
            )

    def test_public_ip_allowed(self) -> None:
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address("93.184.216.34")],
        ):
            SafeHTTPClient.validate_url(
                "https://example.com/",
                user_configured=True,
            )

    def test_unresolvable_host_rejected(self) -> None:
        with (
            patch(
                "omni_mercury_engine.security.safe_http.socket.getaddrinfo",
                side_effect=OSError("no DNS"),
            ),
            pytest.raises(UnsafeURLError, match="did not resolve"),
        ):
            SafeHTTPClient.validate_url(
                "https://unresolvable.invalid/",
                user_configured=True,
            )


class TestAllowPrivateGate:
    """allow_private=True permits RFC1918 but still blocks IMDS / loopback / multicast."""

    @pytest.mark.parametrize(
        "ip",
        [
            "10.0.0.5",  # RFC1918
            "172.16.0.5",  # RFC1918
            "192.168.1.1",  # RFC1918
        ],
    )
    def test_rfc1918_allowed_with_opt_in(self, ip: str) -> None:
        # The self-hosted SearXNG / on-VPC inference case: caller has
        # acknowledged the target is on their private network.
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address(ip)],
        ):
            SafeHTTPClient.validate_url(
                "http://searxng.internal/search",
                allow_http=True,
                user_configured=True,
                allow_private=True,
            )

    @pytest.mark.parametrize(
        ("ip", "fragment"),
        [
            ("169.254.169.254", "always-blocked"),  # IMDS
            ("127.0.0.1", "always-blocked"),  # loopback
            ("224.0.0.1", "always-blocked"),  # multicast
            ("240.0.0.1", "always-blocked"),  # reserved
        ],
    )
    def test_imds_and_friends_blocked_even_with_allow_private(self, ip: str, fragment: str) -> None:
        # The metadata service is the actual SSRF prize. allow_private
        # MUST NOT unlock it.
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address(ip)],
            ),
            pytest.raises(UnsafeURLError, match=fragment),
        ):
            SafeHTTPClient.validate_url(
                "http://searxng.internal/search",
                allow_http=True,
                user_configured=True,
                allow_private=True,
            )


class TestLoopbackOnlyGate:
    """loopback_only refuses any non-127/8 IP, even RFC1918."""

    @pytest.mark.parametrize(
        "ip",
        [
            "192.168.1.1",  # RFC1918 but not loopback
            "10.0.0.5",  # RFC1918 but not loopback
            "8.8.8.8",  # public
            "169.254.169.254",  # IMDS
        ],
    )
    def test_non_loopback_rejected(self, ip: str) -> None:
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address(ip)],
            ),
            pytest.raises(UnsafeURLError, match="non-loopback"),
        ):
            SafeHTTPClient.validate_url(
                "http://ollama.local:11434/api/tags",
                allow_http=True,
                user_configured=True,
                loopback_only=True,
            )

    def test_loopback_v4_allowed(self) -> None:
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address("127.0.0.1")],
        ):
            SafeHTTPClient.validate_url(
                "http://localhost:11434/api/tags",
                allow_http=True,
                user_configured=True,
                loopback_only=True,
            )

    def test_loopback_v6_allowed(self) -> None:
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address("::1")],
        ):
            SafeHTTPClient.validate_url(
                "http://[::1]:11434/api/tags",
                allow_http=True,
                user_configured=True,
                loopback_only=True,
            )


class TestNoNoSecForUrlopen:
    """The codebase must not contain any urlopen or B310 nosec under src/."""

    def test_no_urlopen_outside_safe_http(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2] / "src" / "omni_mercury_engine"
        for path in root.rglob("*.py"):
            if path.name == "safe_http.py":
                continue
            content = path.read_text(encoding="utf-8")
            assert "urllib.request.urlopen" not in content, f"urlopen still present in {path}"

    def test_no_b310_nosec_anywhere(self) -> None:
        import pathlib
        import re

        # The pattern is split so this test file itself is not flagged
        # by the count-the-suppressions verification grep.
        suppression_re = re.compile(r"#\s*nosec\s+B" + "310")
        root = pathlib.Path(__file__).resolve().parents[2] / "src" / "omni_mercury_engine"
        for path in root.rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            assert not suppression_re.search(content), f"B310 nosec found in {path}"


# The string ``allow`` and ``untrusted`` are split so the audit grep
# ``grep -rn allow_untrusted src/`` finds zero hits even though the
# regression-guard test must refer to the kwarg by name to assert it
# does not exist.
_REMOVED_BYPASS_KWARG = "allow" + "_" + "untrusted"


class TestNoAllowUntrustedEscapeHatch:
    """The ``allow_untrusted`` bypass kwarg must not be reintroduced.

    PR #202 introduced a per-call escape hatch on ``SafeHTTPClient``
    (and the loader / dataset wrappers) called ``allow_untrusted``
    whose sole effect was to skip the ``TRUSTED_DOMAINS`` host
    allowlist. The follow-up audit found zero production callers,
    and a parameter with no production caller is pre-installed
    attack surface masquerading as flexibility. The kwarg was
    deleted; this class is the regression guard.

    Each test asserts that passing the removed kwarg raises
    ``TypeError`` (the standard Python signature error). If a future
    refactor silently reintroduces the parameter, the kwarg will
    once again be accepted and these tests will fail loudly.
    """

    def test_kwarg_does_not_exist_on_validate_url(self) -> None:
        with pytest.raises(TypeError, match=_REMOVED_BYPASS_KWARG):
            SafeHTTPClient.validate_url(
                "https://earthquake.usgs.gov/path",
                **{_REMOVED_BYPASS_KWARG: True},
            )

    def test_kwarg_does_not_exist_on_get(self) -> None:
        with pytest.raises(TypeError, match=_REMOVED_BYPASS_KWARG):
            SafeHTTPClient.get(
                "https://earthquake.usgs.gov/path",
                **{_REMOVED_BYPASS_KWARG: True},
            )

    def test_kwarg_does_not_exist_on_get_bytes(self) -> None:
        with pytest.raises(TypeError, match=_REMOVED_BYPASS_KWARG):
            SafeHTTPClient.get_bytes(
                "https://earthquake.usgs.gov/path",
                **{_REMOVED_BYPASS_KWARG: True},
            )

    def test_kwarg_does_not_exist_on_get_json(self) -> None:
        with pytest.raises(TypeError, match=_REMOVED_BYPASS_KWARG):
            SafeHTTPClient.get_json(
                "https://earthquake.usgs.gov/path",
                **{_REMOVED_BYPASS_KWARG: True},
            )

    def test_kwarg_does_not_exist_on_get_text(self) -> None:
        with pytest.raises(TypeError, match=_REMOVED_BYPASS_KWARG):
            SafeHTTPClient.get_text(
                "https://earthquake.usgs.gov/path",
                **{_REMOVED_BYPASS_KWARG: True},
            )

    def test_kwarg_does_not_exist_on_post_json(self) -> None:
        with pytest.raises(TypeError, match=_REMOVED_BYPASS_KWARG):
            SafeHTTPClient.post_json(
                "https://earthquake.usgs.gov/path",
                json_body={},
                **{_REMOVED_BYPASS_KWARG: True},
            )

    def test_kwarg_does_not_exist_on_loader_fetch_url(self, tmp_path) -> None:
        """Loader-level pass-through must also reject the removed kwarg."""
        import numpy as np
        import pandas as pd

        from omni_mercury_engine.loaders.base import BaseDomainLoader

        class _Stub(BaseDomainLoader):
            DOMAIN = "test"

            def fetch_realtime(self) -> pd.DataFrame:
                return pd.DataFrame()

            def fetch_historical(self, event_id: str) -> pd.DataFrame:
                return pd.DataFrame()

            def list_events(self) -> list[dict]:
                return []

            def get_ground_truth(self, event_id: str) -> np.ndarray:
                return np.array([])

        loader = _Stub(cache_dir=tmp_path)
        with pytest.raises(TypeError, match=_REMOVED_BYPASS_KWARG):
            loader._fetch_url(
                "https://earthquake.usgs.gov/path",
                **{_REMOVED_BYPASS_KWARG: True},
            )

    def test_kwarg_does_not_exist_on_http_get_with_retry(self) -> None:
        """Dataset-level pass-through must also reject the removed kwarg."""
        from omni_mercury_engine.datasets.base import http_get_with_retry

        with pytest.raises(TypeError, match=_REMOVED_BYPASS_KWARG):
            http_get_with_retry(
                "https://earthquake.usgs.gov/path",
                **{_REMOVED_BYPASS_KWARG: True},
            )


class TestRedirectRejection:
    """3xx responses must surface as ``UnsafeURLError``, not silent corruption.

    ``allow_redirects=False`` makes ``requests`` return the redirect
    response verbatim, and ``raise_for_status`` does not flag 3xx as an
    error -- only 4xx/5xx. Without an explicit rejection, a 301/302/307
    to an off-allowlist host (or a redirect to a private-network address
    via a public-DNS rebind) would silently surface as a successful
    response body to the caller. The fix raises ``UnsafeURLError`` with
    the Location header verbatim so the pivot is loud and debuggable.
    """

    @pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
    def test_redirect_rejected(self, status_code: int) -> None:
        from unittest.mock import MagicMock

        fake_response = MagicMock()
        fake_response.status_code = status_code
        fake_response.headers = {"Location": "https://attacker.example.com/exfil"}

        with (
            patch(
                "omni_mercury_engine.security.safe_http.requests.request",
                return_value=fake_response,
            ),
            pytest.raises(
                UnsafeURLError,
                match="refusing redirect",
            ) as exc_info,
        ):
            SafeHTTPClient.get_bytes("https://earthquake.usgs.gov/fdsnws/event/1/query")
        # The Location header value must appear verbatim in the
        # error message so the operator can see exactly where the
        # 3xx was trying to pivot.
        assert "attacker.example.com" in str(exc_info.value)
        assert str(status_code) in str(exc_info.value)

    def test_redirect_with_missing_location_header_still_rejected(self) -> None:
        """A 302 without a Location header (rare but legal) still raises."""
        from unittest.mock import MagicMock

        fake_response = MagicMock()
        fake_response.status_code = 302
        fake_response.headers = {}  # no Location

        with (
            patch(
                "omni_mercury_engine.security.safe_http.requests.request",
                return_value=fake_response,
            ),
            pytest.raises(UnsafeURLError, match="<no Location header>"),
        ):
            SafeHTTPClient.get_bytes("https://earthquake.usgs.gov/fdsnws/event/1/query")

    def test_200_response_passes_redirect_check(self) -> None:
        """Sanity: a 200 does not trip the 3xx gate."""
        from unittest.mock import MagicMock

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.headers = {}
        fake_response.content = b"body"
        # raise_for_status() on a 200 is a no-op; MagicMock returns
        # another MagicMock for it, which is fine.
        fake_response.raise_for_status.return_value = None
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)

        with patch(
            "omni_mercury_engine.security.safe_http.requests.request",
            return_value=fake_response,
        ):
            body = SafeHTTPClient.get_bytes("https://earthquake.usgs.gov/fdsnws/event/1/query")
        assert body == b"body"
