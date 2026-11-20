# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for trusted-proxy client-IP resolution (rate-limit bypass fix).

The attack this module guards against: ``X-Forwarded-For`` is client-writable,
so reading its LEFT-most entry lets any caller rotate the value and mint a
fresh rate-limit bucket per request. Resolution must (a) ignore the header
entirely unless a proxy tier is declared, (b) read only the right-most trusted
hop, and (c) fail closed to the TCP peer on anything malformed. Includes a
Hypothesis sweep asserting the resolver never crashes and never emits a
non-IP identifier on arbitrary adversarial header content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.api.client_ip import resolve_client_ip, trusted_proxy_hops

if TYPE_CHECKING:
    import pytest


class TestUntrustedDefault:
    """With no trusted hops (default), the header must be ignored entirely."""

    def test_spoofed_header_ignored(self) -> None:
        """A forged XFF cannot override the TCP peer."""
        assert resolve_client_ip("203.0.113.9", "1.2.3.4", hops=0) == "203.0.113.9"

    def test_rotating_spoof_yields_same_bucket(self) -> None:
        """The historical bypass: rotating XFF values must NOT rotate identity."""
        seen = {resolve_client_ip("203.0.113.9", f"10.0.0.{i}", hops=0) for i in range(50)}
        assert seen == {"203.0.113.9"}

    def test_no_peer_no_header_is_unknown(self) -> None:
        """With nothing usable, callers share one throttled bucket."""
        assert resolve_client_ip(None, None, hops=0) == "unknown"

    def test_env_default_is_zero_hops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset (and junk) MERCURY_TRUSTED_PROXY_HOPS mean zero trust."""
        monkeypatch.delenv("MERCURY_TRUSTED_PROXY_HOPS", raising=False)
        assert trusted_proxy_hops() == 0
        monkeypatch.setenv("MERCURY_TRUSTED_PROXY_HOPS", "banana")
        assert trusted_proxy_hops() == 0
        monkeypatch.setenv("MERCURY_TRUSTED_PROXY_HOPS", "-3")
        assert trusted_proxy_hops() == 0
        monkeypatch.setenv("MERCURY_TRUSTED_PROXY_HOPS", "2")
        assert trusted_proxy_hops() == 2


class TestTrustedProxyTier:
    """With N declared hops, only the N-th-from-right entry is honoured."""

    def test_single_proxy_takes_rightmost(self) -> None:
        """One trusted hop → the right-most entry is the client."""
        assert resolve_client_ip("10.0.0.1", "6.6.6.6, 198.51.100.7", hops=1) == "198.51.100.7"

    def test_client_prefix_spoof_is_ignored(self) -> None:
        """Client-prepended garbage left of the trusted hop changes nothing."""
        real = resolve_client_ip("10.0.0.1", "198.51.100.7", hops=1)
        spoofed = resolve_client_ip("10.0.0.1", "1.1.1.1, 2.2.2.2, 3.3.3.3, 198.51.100.7", hops=1)
        assert real == spoofed == "198.51.100.7"

    def test_two_hops_takes_second_from_right(self) -> None:
        """Two trusted hops (LB + reverse proxy) → second-from-right entry."""
        header = "6.6.6.6, 198.51.100.7, 10.0.0.2"
        assert resolve_client_ip("10.0.0.1", header, hops=2) == "198.51.100.7"

    def test_fewer_entries_than_hops_falls_back_to_peer(self) -> None:
        """A direct hit on the origin with a forged short header is the peer."""
        assert resolve_client_ip("203.0.113.9", "1.2.3.4", hops=2) == "203.0.113.9"

    def test_malformed_trusted_entry_falls_back(self) -> None:
        """A non-IP value in the trusted slot fails closed to the peer."""
        assert resolve_client_ip("203.0.113.9", "6.6.6.6, not-an-ip", hops=1) == "203.0.113.9"

    def test_ipv6_and_port_forms_normalise(self) -> None:
        """Bracketed IPv6, v6 case, and host:port forms all canonicalise.

        IPv6 canonicalises to its /64: the delegation unit providers hand one
        subscriber. Per-address buckets gave a single /64 holder 2^64 rate
        identities and the power to evict every legitimate bucket from the
        limiter's bounded LRU.
        """
        assert resolve_client_ip("10.0.0.1", "[2001:DB8::1]:443", hops=1) == "2001:db8::/64"
        assert resolve_client_ip("10.0.0.1", "2001:DB8::2", hops=1) == "2001:db8::/64"
        assert resolve_client_ip("10.0.0.1", "198.51.100.7:8080", hops=1) == "198.51.100.7"

    def test_ipv6_same_slash64_shares_one_bucket(self) -> None:
        """Two addresses in one /64 are one rate identity; another /64 is not."""
        first = resolve_client_ip("10.0.0.1", "2001:db8:1:2:aaaa::1", hops=1)
        second = resolve_client_ip("10.0.0.1", "2001:db8:1:2:bbbb::2", hops=1)
        other = resolve_client_ip("10.0.0.1", "2001:db8:1:3::1", hops=1)
        assert first == second == "2001:db8:1:2::/64"
        assert other != first


class TestIPv6BucketSharingEndToEnd:
    """The identity string is not the claim; the BUCKET is.

    These drive the API's own ``RequestRateLimiter`` with real Starlette
    requests from IPv6 peers: exhausting the burst from one address must
    starve every other address in the same /64, and must not touch a
    neighbouring /64. A per-address keying (the defect this closes) fails the
    first assertion: the second address would get a fresh full burst.
    """

    @staticmethod
    def _request(peer: str) -> object:
        pytest.importorskip("starlette", reason="the request shim needs the [api] extra")
        from starlette.requests import Request

        return Request({"type": "http", "client": (peer, 443), "headers": []})

    def _limiter(self, burst: int) -> object:
        pytest.importorskip("fastapi", reason="the API limiter needs the [api] extra")
        from omni_mercury_engine.api.auth import RequestRateLimiter

        # A refill slow enough that nothing replenishes within the test.
        return RequestRateLimiter(requests_per_minute=1, burst_size=burst)

    def test_same_slash64_draws_from_one_bucket(self) -> None:
        limiter = self._limiter(burst=3)
        attacker = self._request("2001:db8:1:2:aaaa::1")
        for _ in range(3):
            allowed, _ = limiter.is_allowed(attacker)  # type: ignore[attr-defined]
            assert allowed
        # The burst is spent. A *different* address in the same /64 — the
        # rotation a single /64 holder would use — is refused, not refilled.
        sibling = self._request("2001:db8:1:2:ffff:eeee:dddd:cccc")
        allowed, info = limiter.is_allowed(sibling)  # type: ignore[attr-defined]
        assert not allowed
        assert info["remaining"] == 0

    def test_neighbouring_slash64_keeps_its_own_bucket(self) -> None:
        limiter = self._limiter(burst=2)
        for _ in range(2):
            limiter.is_allowed(self._request("2001:db8:1:2::1"))  # type: ignore[attr-defined]
        assert not limiter.is_allowed(self._request("2001:db8:1:2::9"))[0]  # type: ignore[attr-defined]
        # 2001:db8:1:3::/64 is another subscriber: untouched by the exhaustion.
        assert limiter.is_allowed(self._request("2001:db8:1:3::1"))[0]  # type: ignore[attr-defined]

    def test_ipv4_is_still_per_address(self) -> None:
        """IPv4 keeps exact-address identity; only IPv6 is prefix-grouped."""
        limiter = self._limiter(burst=1)
        assert limiter.is_allowed(self._request("203.0.113.9"))[0]  # type: ignore[attr-defined]
        assert not limiter.is_allowed(self._request("203.0.113.9"))[0]  # type: ignore[attr-defined]
        assert limiter.is_allowed(self._request("203.0.113.10"))[0]  # type: ignore[attr-defined]

    def test_missing_header_with_hops_uses_peer(self) -> None:
        """A proxied deployment still resolves direct (header-less) hits sanely."""
        assert resolve_client_ip("10.0.0.1", None, hops=1) == "10.0.0.1"


class TestAdversarialSweep:
    """Property-based fuzz over hostile header content."""

    @settings(max_examples=300, deadline=None)
    @given(
        header=st.text(max_size=200),
        hops=st.integers(min_value=0, max_value=5),
    )
    def test_never_crashes_and_output_is_bounded(self, header: str, hops: int) -> None:
        """Arbitrary header bytes never crash resolution or leak raw text.

        The result must always be the validated peer, a validated IP literal
        from the header, or the ``unknown`` sentinel — never attacker-shaped
        free text (which would poison rate-limit keys and audit logs).
        """
        import ipaddress

        result = resolve_client_ip("203.0.113.9", header, hops=hops)
        if result not in ("203.0.113.9", "unknown"):
            # Anything else must parse as a canonical IP literal.
            assert str(ipaddress.ip_address(result)) == result
