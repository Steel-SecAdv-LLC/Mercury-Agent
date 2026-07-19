# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the VPC-air-gap mode (MERCURY_OFFLINE + allow_private).

Pins the interaction the previous offline gate could not express: cut from the
PUBLIC internet while still reaching an on-prem RFC1918 service (Ollama /
SearXNG). The security-critical invariants are that this carve-out is opt-in,
never unlocks IMDS, and never permits public egress — the air-gap holds; only
private-network reachability is added.

All cases use IP literals so no DNS lookup is performed.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.datasets.exceptions import OfflineModeError
from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError


def _validate(url: str, *, allow_private: bool) -> str:
    """Run validate_url for an operator endpoint; return the outcome label."""
    try:
        SafeHTTPClient.validate_url(
            url, allow_http=True, user_configured=True, allow_private=allow_private
        )
        return "permit"
    except OfflineModeError:
        return "offline_refuse"
    except UnsafeURLError:
        return "ssrf_refuse"


def _set_mode(monkeypatch: pytest.MonkeyPatch, *, offline: bool, vpc: bool) -> None:
    for var, on in (("MERCURY_OFFLINE", offline), ("MERCURY_OFFLINE_ALLOW_PRIVATE", vpc)):
        if on:
            monkeypatch.setenv(var, "1")
        else:
            monkeypatch.delenv(var, raising=False)


def test_default_offline_refuses_rfc1918(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the VPC opt-in, offline mode stays loopback-only (RFC1918 refused)."""
    _set_mode(monkeypatch, offline=True, vpc=False)
    assert _validate("http://10.0.0.5:11434/api/tags", allow_private=True) == "offline_refuse"


def test_vpc_offline_permits_rfc1918(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the VPC opt-in + allow_private, an RFC1918 on-prem host is reachable."""
    _set_mode(monkeypatch, offline=True, vpc=True)
    assert _validate("http://10.0.0.5:11434/api/tags", allow_private=True) == "permit"
    assert _validate("http://192.168.1.9/x", allow_private=True) == "permit"
    assert _validate("http://172.16.4.4/x", allow_private=True) == "permit"


def test_vpc_offline_still_refuses_public(monkeypatch: pytest.MonkeyPatch) -> None:
    """The air-gap holds: a public target is refused even in VPC mode."""
    _set_mode(monkeypatch, offline=True, vpc=True)
    assert _validate("http://8.8.8.8/x", allow_private=True) == "offline_refuse"


def test_vpc_offline_never_unlocks_imds(monkeypatch: pytest.MonkeyPatch) -> None:
    """IMDS (169.254.169.254) is refused in VPC mode — the primary SSRF prize."""
    _set_mode(monkeypatch, offline=True, vpc=True)
    outcome = _validate("http://169.254.169.254/latest/meta-data/", allow_private=True)
    assert outcome in {"offline_refuse", "ssrf_refuse"}


def test_vpc_requires_caller_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The VPC env var alone does nothing; the caller must pass allow_private."""
    _set_mode(monkeypatch, offline=True, vpc=True)
    # allow_private=False -> loopback-only air-gap, RFC1918 refused.
    assert _validate("http://10.0.0.5/x", allow_private=False) == "offline_refuse"


def test_loopback_permitted_in_all_offline_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loopback stays reachable offline regardless of the VPC opt-in."""
    for vpc in (False, True):
        _set_mode(monkeypatch, offline=True, vpc=vpc)
        SafeHTTPClient.validate_url(
            "http://127.0.0.1:11434/api/tags",
            allow_http=True,
            user_configured=True,
            loopback_only=True,
        )


def test_online_behavior_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """With MERCURY_OFFLINE unset, the gate behaves exactly as before."""
    _set_mode(monkeypatch, offline=False, vpc=True)  # vpc var set but offline unset
    assert _validate("http://10.0.0.5/x", allow_private=True) == "permit"
    assert _validate("http://10.0.0.5/x", allow_private=False) == "ssrf_refuse"


def test_ollama_egress_kwargs_selects_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ollama adapter picks loopback_only vs allow_private by host + opt-in."""
    pytest.importorskip("requests")
    from omni_mercury_engine.models.foundation.ollama_adapter import (
        OllamaConfig,
        OllamaLLMAdapter,
    )

    def _kwargs(host: str, *, vpc: bool) -> dict[str, bool]:
        _set_mode(monkeypatch, offline=True, vpc=vpc)
        adapter = OllamaLLMAdapter.__new__(OllamaLLMAdapter)
        adapter.ollama_config = OllamaConfig(host=host, model="llama3.2:1b")
        return adapter._egress_kwargs()

    assert _kwargs("localhost", vpc=True) == {"loopback_only": True}
    assert _kwargs("127.0.0.1", vpc=True) == {"loopback_only": True}
    # A VPC host with the opt-in uses allow_private; without it, fails closed.
    assert _kwargs("10.0.1.5", vpc=True) == {"allow_private": True}
    assert _kwargs("10.0.1.5", vpc=False) == {"loopback_only": True}
