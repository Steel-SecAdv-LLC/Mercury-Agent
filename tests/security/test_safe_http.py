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
        SafeHTTPClient.validate_url(
            "http://earthquake.usgs.gov/path",
            allow_http=True,
        )

    def test_no_host_rejected(self) -> None:
        with pytest.raises(UnsafeURLError, match="no host"):
            SafeHTTPClient.validate_url("https://")


class TestTrustedDomainsGate:
    """class-constant URLs must come from the TrustedEndpoints allowlist."""

    def test_unlisted_host_rejected(self) -> None:
        with pytest.raises(UnsafeURLError, match="not in trusted allowlist"):
            SafeHTTPClient.validate_url("https://evil.example.com/path")

    def test_listed_host_passes(self) -> None:
        # earthquake.usgs.gov is in TRUSTED_DOMAINS
        SafeHTTPClient.validate_url("https://earthquake.usgs.gov/fdsnws/event/1/query")

    def test_user_configured_bypasses_allowlist_but_resolves_host(self) -> None:
        # Public DNS that is not in the allowlist; user_configured
        # opts out of the allowlist but still runs the private-network
        # gate against the resolved IPs.
        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips"
        ) as resolve:
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
            "0.0.0.0",  # unspecified
        ],
    )
    def test_private_ip_rejected(self, ip: str) -> None:
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address(ip)],
        ):
            with pytest.raises(UnsafeURLError, match="private/link-local/IMDS"):
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
        with patch(
            "omni_mercury_engine.security.safe_http.socket.getaddrinfo",
            side_effect=OSError("no DNS"),
        ):
            with pytest.raises(UnsafeURLError, match="did not resolve"):
                SafeHTTPClient.validate_url(
                    "https://unresolvable.invalid/",
                    user_configured=True,
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

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address(ip)],
        ):
            with pytest.raises(UnsafeURLError, match="non-loopback"):
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
            assert "urllib.request.urlopen" not in content, (
                f"urlopen still present in {path}"
            )

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
