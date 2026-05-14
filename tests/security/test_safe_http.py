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


class TestRedirectRejection:
    """3xx responses must surface as UnsafeURLError with the Location header.

    The transport disables redirect following so an attacker (or upstream
    drift) cannot bounce a trusted-allowlist URL to an arbitrary host;
    the requests library reports the 3xx without raising, and our code
    must catch it explicitly. Without this branch a redirect body
    (often an HTML stub) would silently replace the resource the caller
    expected.
    """

    @staticmethod
    def _build_3xx_response(status_code: int, location: str | None) -> object:
        """Return a stand-in for ``requests.Response`` carrying a 3xx + Location."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.status_code = status_code
        response.headers = {"Location": location} if location else {}
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    @pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
    def test_3xx_rejected_with_location_in_message(self, status_code: int) -> None:
        """Every 3xx code raises and the Location header appears in the message."""
        import ipaddress
        from unittest.mock import MagicMock

        fake_response = self._build_3xx_response(
            status_code, "https://raw.githubusercontent.com/elsewhere"
        )

        # The code under test does ``requests.Session().request(...)``.
        # Patch ``requests.Session`` at the place ``_request`` will look
        # it up after its deferred import.
        fake_session = MagicMock()
        fake_session.request = MagicMock(return_value=fake_response)
        fake_session.mount = MagicMock()

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("93.184.216.34")],
            ),
            patch(
                "omni_mercury_engine.security.safe_http.TrustedEndpoints.validate_url_host",
                return_value=True,
            ),
            patch(
                "omni_mercury_engine.security.safe_http._PinnedDNSHTTPAdapter.build",
                return_value=MagicMock(),
            ),
            patch("requests.Session", return_value=fake_session),
            pytest.raises(UnsafeURLError) as exc_info,
        ):
            SafeHTTPClient.get("https://example.com/raw/path")

        msg = str(exc_info.value)
        assert str(status_code) in msg
        assert "raw.githubusercontent.com" in msg
        assert "TRUSTED_DOMAINS" in msg or "redirect" in msg.lower()

    def test_3xx_without_location_header_still_rejected(self) -> None:
        """A 3xx with no Location must still raise (clearer than silent body return)."""
        import ipaddress
        from unittest.mock import MagicMock

        response = self._build_3xx_response(302, None)
        fake_session = MagicMock()
        fake_session.request = MagicMock(return_value=response)
        fake_session.mount = MagicMock()

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("93.184.216.34")],
            ),
            patch(
                "omni_mercury_engine.security.safe_http.TrustedEndpoints.validate_url_host",
                return_value=True,
            ),
            patch(
                "omni_mercury_engine.security.safe_http._PinnedDNSHTTPAdapter.build",
                return_value=MagicMock(),
            ),
            patch("requests.Session", return_value=fake_session),
            pytest.raises(UnsafeURLError, match="302"),
        ):
            SafeHTTPClient.get("https://example.com/path")


class TestCGNATBlocked:
    """RFC 6598 shared CGNAT space (100.64.0.0/10) must not pass the SSRF gate.

    ``ipaddress`` classifies CGNAT as neither private nor reserved, so
    the convenience attributes used elsewhere on the standard library
    type miss it. A user-configured URL resolving to a CGNAT address
    would otherwise pass the gate; the explicit network membership
    check in ``_is_private_or_imds`` closes that hole.
    """

    @pytest.mark.parametrize("ip", ["100.64.0.1", "100.127.255.254"])
    def test_cgnat_rejected_for_user_configured(self, ip: str) -> None:
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address(ip)],
            ),
            pytest.raises(UnsafeURLError, match="private/link-local/IMDS"),
        ):
            SafeHTTPClient.validate_url(
                "https://internal.example/api",
                user_configured=True,
            )

    @pytest.mark.parametrize("ip", ["100.64.0.1", "100.127.255.254"])
    def test_cgnat_rejected_even_with_allow_private(self, ip: str) -> None:
        """``allow_private=True`` opens RFC1918 but must NOT open CGNAT."""
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address(ip)],
            ),
            pytest.raises(UnsafeURLError, match="always-blocked"),
        ):
            SafeHTTPClient.validate_url(
                "https://internal.example/api",
                user_configured=True,
                allow_private=True,
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
