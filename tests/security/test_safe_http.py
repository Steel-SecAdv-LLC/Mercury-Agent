# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate tests for :mod:`omni_mercury_engine.security.safe_http`.

Each test pins one rejection path so a regression that lets the
underlying urllib pattern back into the codebase (or weakens the
SafeHTTPClient gates) trips a unit test before it ships.
"""

from __future__ import annotations

from typing import Any
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


class TestRequestTimeDNSRecheck:
    """The request-time DNS resolve must catch a host that rebinds to a
    private/IMDS/CGNAT address after ``validate_url`` already passed.

    Without these tests the DNS-rebinding TOCTOU fix would silently
    weaken: a regression that drops the private-IP check from
    ``_request`` would re-open the gap. Each case mocks two distinct
    behaviours -- ``TrustedEndpoints.validate_url_host`` passes (so the
    pre-flight allowlist check accepts the URL), but ``_resolve_ips``
    returns a private/IMDS/CGNAT address at request time.
    """

    @pytest.mark.parametrize(
        "rebound_ip,fragment",
        [
            ("10.0.0.5", "private/link-local/IMDS/CGNAT"),
            ("169.254.169.254", "private/link-local/IMDS/CGNAT"),
            ("100.64.0.5", "private/link-local/IMDS/CGNAT"),
            ("127.0.0.1", "private/link-local/IMDS/CGNAT"),
            ("172.16.0.10", "private/link-local/IMDS/CGNAT"),
        ],
    )
    def test_default_rebind_to_non_public_rejected(self, rebound_ip: str, fragment: str) -> None:
        """Trusted-allowlist host that resolves to non-public address at request time."""
        import ipaddress
        from unittest.mock import MagicMock

        # The class-constant trusted-allowlist URL path: validate_url
        # passes the allowlist check and skips its own resolve.
        # _request's recheck then sees the rebound private address
        # and raises with the DNS-rebinding signature message.
        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address(rebound_ip)],
            ),
            patch(
                "omni_mercury_engine.security.safe_http.TrustedEndpoints.validate_url_host",
                return_value=True,
            ),
            patch(
                "omni_mercury_engine.security.safe_http._PinnedDNSHTTPAdapter.build",
                return_value=MagicMock(),
            ),
            pytest.raises(UnsafeURLError, match=fragment),
        ):
            SafeHTTPClient.get("https://example.com/api")

    def test_allow_private_still_blocks_imds_on_rebind(self) -> None:
        """``allow_private=True`` lets RFC1918 through, but IMDS still blocks."""
        import ipaddress
        from unittest.mock import MagicMock

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("169.254.169.254")],
            ),
            patch(
                "omni_mercury_engine.security.safe_http._PinnedDNSHTTPAdapter.build",
                return_value=MagicMock(),
            ),
            pytest.raises(UnsafeURLError, match="always-blocked"),
        ):
            SafeHTTPClient.get(
                "https://internal.example/api",
                user_configured=True,
                allow_private=True,
            )

    def test_allow_private_accepts_rfc1918_on_rebind(self) -> None:
        """``allow_private=True`` should let RFC1918 through on rebind too."""
        import ipaddress
        from unittest.mock import MagicMock

        fake_session = MagicMock()
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_session.request = MagicMock(return_value=fake_response)
        fake_session.mount = MagicMock()
        fake_session.close = MagicMock()

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("10.0.0.5")],
            ),
            patch(
                "omni_mercury_engine.security.safe_http._PinnedDNSHTTPAdapter.build",
                return_value=MagicMock(),
            ),
            patch("requests.Session", return_value=fake_session),
        ):
            response = SafeHTTPClient.get(
                "https://internal.example/api",
                user_configured=True,
                allow_private=True,
            )
        assert response is fake_response


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


class TestHostHeaderPreservesPort:
    """``Host`` header MUST carry the explicit port for non-default-port URLs.

    urllib3 derives ``Host`` from the connection-pool host, which the
    pinned adapter sets to the IP literal; we set the header explicitly
    so virtual-hosted upstreams see the real hostname.  For services on
    non-default ports (e.g. ``http://localhost:11434`` Ollama,
    ``http://searxng:8888``, etc.) dropping the port produced
    ``Host: localhost`` and routed traffic to whatever vhost happened
    to be default on the IP -- the wrong upstream.
    """

    @staticmethod
    def _200_response() -> object:
        from unittest.mock import MagicMock

        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status = MagicMock()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    def _run(self, url: str, *, loopback_only: bool = False) -> dict[str, str]:
        """Issue a gated request and return the Host header that went out.

        Loopback URLs use ``127.0.0.1`` so the loopback gate passes;
        non-loopback URLs use a public IP literal (``93.184.216.34``,
        example.com) so the private/IMDS gate does not fire.
        """
        import ipaddress
        from unittest.mock import MagicMock

        resolved_ip = "127.0.0.1" if loopback_only else "93.184.216.34"
        fake_session = MagicMock()
        fake_session.request = MagicMock(return_value=self._200_response())
        fake_session.mount = MagicMock()

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address(resolved_ip)],
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
        ):
            SafeHTTPClient.get(
                url,
                allow_http=True,
                user_configured=loopback_only,
                loopback_only=loopback_only,
            )

        called_headers: dict[str, str] = fake_session.request.call_args.kwargs["headers"]
        return called_headers

    def test_non_default_port_preserved_in_host_header(self) -> None:
        """``http://localhost:11434`` -> ``Host: localhost:11434``."""
        headers = self._run("http://localhost:11434/api/tags", loopback_only=True)
        assert headers["Host"] == "localhost:11434"

    def test_default_http_port_omitted_from_host_header(self) -> None:
        """``http://example.com:80`` -> ``Host: example.com`` (default-port drop)."""
        headers = self._run("http://example.com:80/path")
        assert headers["Host"] == "example.com"

    def test_default_https_port_omitted_from_host_header(self) -> None:
        """``https://example.com:443`` -> ``Host: example.com`` (default-port drop)."""
        headers = self._run("https://example.com:443/path")
        assert headers["Host"] == "example.com"

    def test_no_port_in_url_yields_bare_hostname(self) -> None:
        """``https://example.com/path`` -> ``Host: example.com`` (no port present)."""
        headers = self._run("https://example.com/path")
        assert headers["Host"] == "example.com"

    def test_non_default_https_port_preserved(self) -> None:
        """``https://example.com:8443`` -> ``Host: example.com:8443``."""
        headers = self._run("https://example.com:8443/path")
        assert headers["Host"] == "example.com:8443"


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


# =============================================================================
# Coverage added with PR #210: removal of the ``allow_untrusted`` HTTP escape
# hatch. The tests below pin the new API surface (the obsolete kwarg is gone),
# document the supported replacement path, and exercise the wrappers,
# multi-IP failover, IPv6 always-blocked set, and the ``allow_redirects=False``
# transport contract that previous suites did not cover.
# =============================================================================


class TestAllowUntrustedKwargRemoval:
    """``allow_untrusted`` is removed from every SafeHTTPClient method.

    Before PR #210 there was a per-call ``allow_untrusted=True`` escape
    hatch that skipped the ``TRUSTED_DOMAINS`` allowlist. The parameter
    had no production caller and was an attack surface waiting to be
    misused; PR #210 deletes it.  These tests are the
    architectural-contract pins: the kwarg name must surface as
    ``TypeError`` so a future ressurection has to land a fresh public
    API change and cannot creep in via a stale kwarg path.
    """

    @pytest.mark.parametrize(
        "method_name",
        ["validate_url", "get", "get_bytes", "get_json", "get_text", "post_json"],
    )
    def test_kwarg_removed_from_public_api(self, method_name: str) -> None:
        method = getattr(SafeHTTPClient, method_name)
        kwargs: dict[str, object] = {"allow_untrusted": True}
        if method_name == "post_json":
            kwargs["json_body"] = {"k": "v"}
        with pytest.raises(TypeError, match="allow_untrusted"):
            method("https://earthquake.usgs.gov/path", **kwargs)

    def test_kwarg_removed_from_private_request(self) -> None:
        """Internal ``_request`` rejects the obsolete kwarg too."""
        # ``getattr`` keeps the lookup dynamic so the runtime kwarg
        # rejection -- the sole defence layer -- is what we actually
        # exercise; this mirrors the public-API test above and avoids
        # asking mypy to validate a deliberately-invalid signature.
        # The B009 noqa records that the dynamic lookup is intentional.
        request_method = getattr(SafeHTTPClient, "_request")  # noqa: B009
        kwargs: dict[str, object] = {"allow_untrusted": True}
        with pytest.raises(TypeError, match="allow_untrusted"):
            request_method("GET", "https://earthquake.usgs.gov/path", **kwargs)

    def test_signature_does_not_contain_allow_untrusted(self) -> None:
        """Belt-and-braces inspection: no method exposes the kwarg by name."""
        import inspect

        for method_name in (
            "validate_url",
            "_request",
            "get",
            "get_bytes",
            "get_json",
            "get_text",
            "post_json",
        ):
            method = getattr(SafeHTTPClient, method_name)
            sig = inspect.signature(method)
            assert (
                "allow_untrusted" not in sig.parameters
            ), f"{method_name} still exposes 'allow_untrusted' in its signature."


class TestMigrationFromAllowUntrusted:
    """Document the supported replacement for the removed ``allow_untrusted``.

    Operators that previously passed ``allow_untrusted=True`` to reach
    a dynamic public host now pass ``user_configured=True`` instead.
    For internal RFC1918 destinations they additionally pass
    ``allow_private=True``. These tests assert both happy paths and the
    invariants that they preserve (IMDS still blocked, scheme still
    HTTPS-only, allowlist still bypassed).
    """

    def test_replacement_path_public_https_host(self) -> None:
        """``user_configured=True`` accepts an off-allowlist public host."""
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address("93.184.216.34")],  # public TEST-IP
        ):
            SafeHTTPClient.validate_url(
                "https://api.operator-chosen.example/v1/signal",
                user_configured=True,
            )

    def test_replacement_path_still_blocks_imds(self) -> None:
        """``user_configured=True`` does NOT unlock the metadata service."""
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("169.254.169.254")],
            ),
            pytest.raises(UnsafeURLError, match="private/link-local/IMDS"),
        ):
            SafeHTTPClient.validate_url(
                "https://still-an-attacker.example/exfil",
                user_configured=True,
            )

    def test_replacement_path_still_https_only(self) -> None:
        """``user_configured=True`` does not relax the scheme gate."""
        with pytest.raises(UnsafeURLError, match="scheme 'http'"):
            SafeHTTPClient.validate_url(
                "http://api.operator-chosen.example/v1/signal",
                user_configured=True,
            )

    def test_internal_rfc1918_path_requires_allow_private(self) -> None:
        """RFC1918 destinations need an explicit ``allow_private=True`` opt-in."""
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address("10.0.0.10")],
            ),
            pytest.raises(UnsafeURLError, match="private/link-local/IMDS"),
        ):
            SafeHTTPClient.validate_url(
                "https://internal-api.vpc.local/v1/signal",
                user_configured=True,
            )

    def test_internal_rfc1918_path_with_allow_private_succeeds(self) -> None:
        """``user_configured=True`` + ``allow_private=True`` is the on-VPC path."""
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address("10.0.0.10")],
        ):
            SafeHTTPClient.validate_url(
                "https://internal-api.vpc.local/v1/signal",
                user_configured=True,
                allow_private=True,
            )


class TestIPv6AlwaysBlocked:
    """The IPv6 always-blocked set must fire for every refused range.

    The IPv4 always-blocked tests in :class:`TestAllowPrivateGate` only
    cover IPv4 ranges. The IPv6 mirror -- loopback ``::1``, link-local
    ``fe80::/10``, multicast ``ff00::/8``, and the unspecified address
    ``::`` -- shipped in :data:`_ALWAYS_BLOCKED_V6` but had no test
    coverage. A future refactor that drops one of those networks would
    have shipped unnoticed.
    """

    @pytest.mark.parametrize(
        ("ip", "fragment"),
        [
            ("::1", "always-blocked"),  # loopback
            ("fe80::1", "always-blocked"),  # link-local
            ("ff02::1", "always-blocked"),  # multicast (all-nodes)
            ("ff05::2", "always-blocked"),  # multicast (all-routers)
            ("::", "always-blocked"),  # unspecified
        ],
    )
    def test_ipv6_blocked_even_with_allow_private(self, ip: str, fragment: str) -> None:
        import ipaddress

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[ipaddress.ip_address(ip)],
            ),
            pytest.raises(UnsafeURLError, match=fragment),
        ):
            SafeHTTPClient.validate_url(
                "https://internal.example/api",
                user_configured=True,
                allow_private=True,
            )

    def test_ipv6_loopback_url_rejected_by_default(self) -> None:
        """A URL like ``https://[::1]/...`` is rejected without explicit opt-in."""
        with pytest.raises(UnsafeURLError, match="private/link-local/IMDS"):
            SafeHTTPClient.validate_url(
                "https://[::1]/api",
                user_configured=True,
            )

    def test_ipv6_link_local_url_rejected(self) -> None:
        """``https://[fe80::1]/`` is rejected -- link-local is non-routable."""
        with pytest.raises(UnsafeURLError, match="private/link-local/IMDS"):
            SafeHTTPClient.validate_url(
                "https://[fe80::1]/api",
                user_configured=True,
            )


class TestWrappersEnforceGate:
    """Every public wrapper (``get``, ``get_bytes``, ``get_json``,
    ``get_text``, ``post_json``) routes through :meth:`validate_url`
    before any network work.

    The wrappers are thin pass-throughs to ``_request``, but if a
    refactor ever moved the validation into a subclass override or
    duplicated the parameter list, a wrapper could silently lose
    the gate. These tests catch that by exercising each wrapper with
    a URL that fails the scheme gate -- the assertion is that no
    ``requests.Session`` is even constructed.
    """

    @pytest.mark.parametrize(
        ("wrapper", "extra_kwargs"),
        [
            ("get", {}),
            ("get_bytes", {}),
            ("get_json", {}),
            ("get_text", {}),
            ("post_json", {"json_body": {"k": "v"}}),
        ],
    )
    def test_bad_scheme_rejected_without_touching_network(
        self, wrapper: str, extra_kwargs: dict[str, Any]
    ) -> None:
        method = getattr(SafeHTTPClient, wrapper)
        with patch("requests.Session") as session_factory:
            with pytest.raises(UnsafeURLError, match="scheme 'ftp'"):
                method("ftp://example.com/path", **extra_kwargs)
            assert session_factory.call_count == 0, (
                f"{wrapper} reached requests.Session despite a scheme failure; "
                "the gate is not on the wrapper path."
            )

    @pytest.mark.parametrize(
        ("wrapper", "extra_kwargs"),
        [
            ("get", {}),
            ("get_bytes", {}),
            ("get_json", {}),
            ("get_text", {}),
            ("post_json", {"json_body": {"k": "v"}}),
        ],
    )
    def test_unlisted_host_rejected_without_touching_network(
        self, wrapper: str, extra_kwargs: dict[str, Any]
    ) -> None:
        method = getattr(SafeHTTPClient, wrapper)
        with patch("requests.Session") as session_factory:
            with pytest.raises(UnsafeURLError, match="not in trusted allowlist"):
                method("https://attacker.example.com/exfil", **extra_kwargs)
            assert session_factory.call_count == 0


class TestMultiIPFailover:
    """When ``_resolve_ips`` returns multiple addresses, ``_request``
    must try the next candidate after a transient connection failure
    on the first; if every IP fails, it surfaces the last exception
    rather than a confusing ``None``/``response=None`` path.

    Closes the gap left by the DNS-pinning fix: pinning to a single
    IP would otherwise eliminate the multi-IP resilience the stdlib
    socket layer used to provide. The contract is "pin per attempt,
    iterate IPs across attempts."

    Note on test IPs: we use real global-public IPs (``8.8.8.8``,
    ``1.1.1.1``) because the request-time DNS recheck inside
    ``_request`` runs the private/IMDS filter on every resolved
    address, and Python's ``ipaddress.is_private`` returns ``True``
    for the IETF TEST-NET ranges (192.0.2/24, 198.51.100/24,
    203.0.113/24). The mocks never actually open a socket, so using
    real allocated IPs here is harmless.
    """

    @staticmethod
    def _200_response() -> object:
        from unittest.mock import MagicMock

        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status = MagicMock()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    def test_second_ip_used_when_first_fails(self) -> None:
        """First IP raises ``ConnectionError``; second IP serves the response."""
        import ipaddress
        from unittest.mock import MagicMock

        import requests

        good_response = self._200_response()

        first_session = MagicMock()
        first_session.request = MagicMock(
            side_effect=requests.ConnectionError("pinned-IP unreachable")
        )
        first_session.mount = MagicMock()
        first_session.close = MagicMock()

        second_session = MagicMock()
        second_session.request = MagicMock(return_value=good_response)
        second_session.mount = MagicMock()

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[
                    ipaddress.ip_address("8.8.8.8"),  # public DNS (Google)
                    ipaddress.ip_address("1.1.1.1"),  # public DNS (Cloudflare)
                ],
            ),
            patch(
                "omni_mercury_engine.security.safe_http.TrustedEndpoints.validate_url_host",
                return_value=True,
            ),
            patch(
                "omni_mercury_engine.security.safe_http._PinnedDNSHTTPAdapter.build",
                return_value=MagicMock(),
            ),
            patch("requests.Session", side_effect=[first_session, second_session]),
        ):
            response = SafeHTTPClient.get("https://example.com/api")

        assert response is good_response
        first_session.close.assert_called_once()
        second_session.request.assert_called_once()

    def test_all_ips_fail_surfaces_last_exception(self) -> None:
        """Both candidate IPs unreachable -> the last ConnectionError propagates."""
        import ipaddress
        from unittest.mock import MagicMock

        import requests

        def make_failing_session(label: str) -> MagicMock:
            s = MagicMock()
            s.request = MagicMock(side_effect=requests.ConnectionError(f"{label} unreachable"))
            s.mount = MagicMock()
            s.close = MagicMock()
            return s

        sessions = [make_failing_session("first"), make_failing_session("second")]

        with (
            patch(
                "omni_mercury_engine.security.safe_http._resolve_ips",
                return_value=[
                    ipaddress.ip_address("8.8.8.8"),
                    ipaddress.ip_address("1.1.1.1"),
                ],
            ),
            patch(
                "omni_mercury_engine.security.safe_http.TrustedEndpoints.validate_url_host",
                return_value=True,
            ),
            patch(
                "omni_mercury_engine.security.safe_http._PinnedDNSHTTPAdapter.build",
                return_value=MagicMock(),
            ),
            patch("requests.Session", side_effect=sessions),
            pytest.raises(requests.ConnectionError, match="second unreachable"),
        ):
            SafeHTTPClient.get("https://example.com/api")

        for s in sessions:
            s.close.assert_called_once()


class TestRedirectsDisabled:
    """The transport MUST pass ``allow_redirects=False`` to
    ``requests.Session.request``.

    The 3xx-rejection branch in :class:`TestRedirectRejection` covers
    the explicit raise on a 3xx response, but only because we already
    have a 3xx in hand. The deeper invariant is that ``requests`` is
    never permitted to follow a redirect transparently. A regression
    that flips ``allow_redirects`` to ``True`` (or removes the kwarg
    so it defaults to True) would let a redirected response substitute
    for the resource without triggering the explicit 3xx branch.
    """

    @staticmethod
    def _200_response() -> object:
        from unittest.mock import MagicMock

        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status = MagicMock()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    def test_allow_redirects_false_is_passed_to_requests(self) -> None:
        """``session.request`` always receives ``allow_redirects=False``."""
        import ipaddress
        from unittest.mock import MagicMock

        fake_session = MagicMock()
        fake_session.request = MagicMock(return_value=self._200_response())
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
        ):
            SafeHTTPClient.get("https://example.com/path")

        called = fake_session.request.call_args
        assert called.kwargs.get("allow_redirects") is False, (
            "session.request was called without allow_redirects=False; "
            "the redirect-following prevention is not on the request path."
        )


class TestValidateUrlShortCircuit:
    """A class-constant ``https://`` URL with the host on
    ``TRUSTED_DOMAINS`` MUST NOT trigger a DNS lookup at validation
    time.

    The IP-resolution gate is conditional (``needs_ip_gate``) precisely
    so the loader bulk path doesn't pay per-request DNS overhead for
    URLs that are already on the allowlist. A regression that resolves
    unconditionally would (1) waste a DNS round-trip on every request
    and (2) make the loader suite fail in offline / no-DNS environments
    (the offline-compose deployment target).
    """

    def test_trusted_https_does_not_call_getaddrinfo(self) -> None:
        """No ``socket.getaddrinfo`` call for a trusted ``https://`` URL."""
        with patch(
            "omni_mercury_engine.security.safe_http.socket.getaddrinfo",
            side_effect=AssertionError(
                "getaddrinfo invoked for a trusted https:// URL; the "
                "needs_ip_gate short-circuit is broken."
            ),
        ):
            SafeHTTPClient.validate_url("https://earthquake.usgs.gov/fdsnws/event/1/query")

    def test_http_url_does_call_getaddrinfo(self) -> None:
        """``http://`` URLs trigger the resolve even when the host is trusted."""
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address("23.215.0.137")],
        ) as resolver:
            SafeHTTPClient.validate_url(
                "http://earthquake.usgs.gov/path",
                allow_http=True,
            )
            assert resolver.called, (
                "_resolve_ips was not called for an http:// URL; the gate "
                "lost its defence-in-depth IP check for plain HTTP."
            )

    def test_user_configured_does_call_getaddrinfo(self) -> None:
        """A user-configured URL always resolves so the private-IP gate fires."""
        import ipaddress

        with patch(
            "omni_mercury_engine.security.safe_http._resolve_ips",
            return_value=[ipaddress.ip_address("93.184.216.34")],
        ) as resolver:
            SafeHTTPClient.validate_url(
                "https://api.operator-chosen.example/",
                user_configured=True,
            )
            assert resolver.called


class TestResolveIPsLiteralShortCircuit:
    """``_resolve_ips`` MUST treat an IP literal as itself and not call DNS.

    Without this short-circuit the SSRF gate could be bypassed by a
    raw IP in the URL: a hostname resolver that returns no records
    for ``127.0.0.1`` would otherwise mask the loopback address from
    the private-IP filter. The literal-IP path keeps the gate transparent.
    """

    @pytest.mark.parametrize(
        "ip_literal",
        [
            "127.0.0.1",
            "169.254.169.254",
            "10.0.0.5",
            "::1",
            "fe80::1",
            "100.64.0.1",  # CGNAT literal
        ],
    )
    def test_literal_ip_classified_directly(self, ip_literal: str) -> None:
        """An IP literal as host is parsed as itself; no DNS happens."""
        from omni_mercury_engine.security.safe_http import _resolve_ips

        with patch(
            "omni_mercury_engine.security.safe_http.socket.getaddrinfo",
            side_effect=AssertionError(
                f"getaddrinfo invoked for IP literal '{ip_literal}'; the "
                "_resolve_ips short-circuit is broken."
            ),
        ):
            ips = _resolve_ips(ip_literal)
        assert len(ips) == 1
        assert str(ips[0]) == ip_literal


class TestCredentialRedactionInErrors:
    """No SafeHTTPClient error path may surface a composed credential.

    requests renders HTTPError as ``"<status> ... for url: <full URL>"``
    and urllib3 renders connection failures as ``"... Max retries
    exceeded with url: /path?query"`` (measured shapes) — for keyed
    callers (Alpha Vantage ``apikey``, OpenWeatherMap ``appid``, NASA
    ``api_key`` in query params) both embed the live credential.  Every
    re-raise out of ``_request`` must therefore carry redacted text with
    the chain severed, and the gate's own refusal messages must redact
    the URL they name.
    """

    _SECRET = "LIVEKEY12345"
    _KEYED_URL = f"https://api.example.com/query?apikey={_SECRET}&function=DAILY"

    @staticmethod
    def _fake_plumbing(fake_session: Any) -> list[Any]:
        """The patch set every _request test needs (mirrors TestRedirectRejection)."""
        import ipaddress
        from unittest.mock import MagicMock

        return [
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
        ]

    def test_scheme_refusal_message_redacted(self) -> None:
        """validate_url's scheme refusal names the URL — redacted."""
        with pytest.raises(UnsafeURLError) as exc_info:
            SafeHTTPClient.validate_url(f"ftp://h.example/p?api_key={self._SECRET}")
        assert self._SECRET not in str(exc_info.value)
        assert "ftp" in str(exc_info.value)

    def test_redirect_refusal_redacts_both_urls(self) -> None:
        """The 3xx refusal names request URL AND Location — an upstream
        answering with a signed redirect target would otherwise leak both."""
        from unittest.mock import MagicMock

        location = "https://cdn.example.com/file?signature=SIGSECRET99&expires=1"
        response = MagicMock()
        response.status_code = 302
        response.headers = {"Location": location}
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        fake_session = MagicMock()
        fake_session.request = MagicMock(return_value=response)
        fake_session.mount = MagicMock()

        import contextlib

        with contextlib.ExitStack() as stack:
            for p in self._fake_plumbing(fake_session):
                stack.enter_context(p)
            with pytest.raises(UnsafeURLError) as exc_info:
                SafeHTTPClient.get(self._KEYED_URL)

        message = str(exc_info.value)
        assert self._SECRET not in message
        assert "SIGSECRET99" not in message
        assert "cdn.example.com" in message  # host survives for diagnosis

    def test_http_error_redacted_type_and_response_preserved(self) -> None:
        """4xx/5xx re-raises the same requests.HTTPError type with the
        credentialed URL scrubbed, .response intact, chain severed."""
        from unittest.mock import MagicMock

        import requests

        real_error = requests.HTTPError(f"403 Client Error: Forbidden for url: {self._KEYED_URL}")
        response = MagicMock()
        response.status_code = 403
        response.headers = {}
        response.raise_for_status = MagicMock(side_effect=real_error)
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        fake_session = MagicMock()
        fake_session.request = MagicMock(return_value=response)
        fake_session.mount = MagicMock()

        import contextlib

        with contextlib.ExitStack() as stack:
            for p in self._fake_plumbing(fake_session):
                stack.enter_context(p)
            with pytest.raises(requests.HTTPError) as exc_info:
                SafeHTTPClient.get(self._KEYED_URL)

        exc = exc_info.value
        assert self._SECRET not in str(exc)
        assert "403" in str(exc)
        assert exc.response is response  # status branching keeps working
        assert exc.__cause__ is None, "original (leaking) HTTPError chained back in"
        assert exc.__suppress_context__ is True

    def test_transport_exhaustion_redacted_type_preserved(self) -> None:
        """Connection failure on every pinned IP re-raises the same
        exception TYPE with urllib3's path+query text scrubbed and the
        chain severed."""
        from unittest.mock import MagicMock

        import requests

        transport_error = requests.ConnectionError(
            "HTTPSConnectionPool(host='api.example.com', port=443): Max retries "
            f"exceeded with url: /query?apikey={self._SECRET}&function=DAILY "
            "(Caused by NewConnectionError('...'))"
        )
        fake_session = MagicMock()
        fake_session.request = MagicMock(side_effect=transport_error)
        fake_session.mount = MagicMock()
        fake_session.close = MagicMock()

        import contextlib

        with contextlib.ExitStack() as stack:
            for p in self._fake_plumbing(fake_session):
                stack.enter_context(p)
            with pytest.raises(requests.ConnectionError) as exc_info:
                SafeHTTPClient.get(self._KEYED_URL)

        exc = exc_info.value
        assert self._SECRET not in str(exc)
        assert "function=DAILY" in str(exc)  # diagnostics survive
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True
