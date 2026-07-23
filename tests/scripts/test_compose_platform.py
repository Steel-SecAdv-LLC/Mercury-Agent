# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guards for the platform compose overlay (docker-compose.platform.yml).

CI has no Docker daemon, so these are structural checks over the YAML: the
overlay must stay a pure layer over the base file (same app service name),
every ``MERCURY_*`` variable it names must be documented in the
PLATFORM_HARDENING configuration reference, the durable state must live on
the named volume the docs promise, and the Caddy/proxy contract
(one trusted hop, /health checking, the shipped Caddyfile) must hold.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[2]
_BASE = _REPO / "docker-compose.yml"
_OVERLAY = _REPO / "docker-compose.platform.yml"
_CADDYFILE = _REPO / "deploy" / "Caddyfile"
_HARDENING_DOC = _REPO / "docs" / "PLATFORM_HARDENING.md"


def _load(path: Path) -> dict[str, Any]:
    """Parse one compose file."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _documented_mercury_vars() -> set[str]:
    """Every MERCURY_* variable named in the configuration reference doc.

    The reference table collapses families into shorthand rows —
    ``MERCURY_SCRYPT_N/R/P``, ``MERCURY_SMTP_HOST`` + ``(_PORT/_USERNAME/…)``,
    ``MERCURY_QUOTA_WINDOW_SECONDS`` / ``_MAX_REQUESTS`` — so this expands
    each shorthand: slash-alternates swap the last component, and ``_SUFFIX``
    fragments attach to every prefix of a full variable named on the same
    line. Over-generation is fine; the result is an allowlist.
    """
    documented: set[str] = set()
    for line in _HARDENING_DOC.read_text(encoding="utf-8").splitlines():
        full_vars = re.findall(r"MERCURY_[A-Z0-9_]+", line)
        documented.update(full_vars)
        # MERCURY_SCRYPT_N/R/P → MERCURY_SCRYPT_{N,R,P}
        for base, alternates in re.findall(r"(MERCURY_[A-Z0-9_]+)((?:/[A-Z0-9]+)+)", line):
            root = base.rsplit("_", 1)[0]
            for alternate in alternates.strip("/").split("/"):
                documented.add(f"{root}_{alternate}")
        # `_MAX_REQUESTS` next to MERCURY_QUOTA_WINDOW_SECONDS →
        # MERCURY_QUOTA_MAX_REQUESTS (tried against every prefix).
        fragments = re.findall(r"`?_([A-Z][A-Z0-9_]+)`?", line.replace("MERCURY_", ""))
        for var in full_vars:
            parts = var.split("_")
            for cut in range(1, len(parts)):
                prefix = "_".join(parts[:cut])
                documented.update(f"{prefix}_{fragment}" for fragment in fragments)
    return documented


def test_overlay_extends_the_base_app_service() -> None:
    """The overlay layers onto the base service (no forked service name)."""
    base = _load(_BASE)
    overlay = _load(_OVERLAY)
    assert "mercury-agent" in base["services"]
    assert "mercury-agent" in overlay["services"]
    assert "caddy" in overlay["services"]


def test_overlay_names_only_documented_env_vars() -> None:
    """Every MERCURY_* key the overlay sets is in the configuration reference."""
    overlay = _load(_OVERLAY)
    documented = _documented_mercury_vars()
    env = overlay["services"]["mercury-agent"]["environment"]
    mercury_keys = {key for key in env if key.startswith("MERCURY_")}
    assert mercury_keys, "the overlay should configure the platform env"
    undocumented = sorted(mercury_keys - documented)
    assert not undocumented, f"undocumented env vars in overlay: {undocumented}"


def test_durable_state_lives_on_the_named_volume() -> None:
    """Keystore + audit dir sit under the /var/lib/mercury named volume."""
    overlay = _load(_OVERLAY)
    service = overlay["services"]["mercury-agent"]
    env = service["environment"]
    assert env["MERCURY_KEYSTORE_PATH"] == "/var/lib/mercury/mercury.db"
    assert env["MERCURY_AUDIT_LOG_DIR"] == "/var/lib/mercury/audit"
    assert "mercury-platform-data:/var/lib/mercury" in service["volumes"]
    assert "mercury-platform-data" in overlay["volumes"]


def test_proxy_contract_is_one_trusted_hop() -> None:
    """The app trusts exactly the one X-Forwarded-For hop Caddy appends."""
    overlay = _load(_OVERLAY)
    env = overlay["services"]["mercury-agent"]["environment"]
    assert env["MERCURY_TRUSTED_PROXY_HOPS"] == "${MERCURY_TRUSTED_PROXY_HOPS:-1}"


def test_caddy_service_and_caddyfile_agree() -> None:
    """Caddy terminates 80/443, mounts the shipped Caddyfile, checks /health."""
    overlay = _load(_OVERLAY)
    caddy = overlay["services"]["caddy"]
    assert "80:80" in caddy["ports"] and "443:443" in caddy["ports"]
    assert any("deploy/Caddyfile" in volume for volume in caddy["volumes"])
    assert "mercury-agent" in caddy["depends_on"]

    caddyfile = _CADDYFILE.read_text(encoding="utf-8")
    assert "app.mercuryagent.global" in caddyfile
    assert "reverse_proxy mercury-agent:8000" in caddyfile
    assert "health_uri /health" in caddyfile


def test_runbook_documents_the_overlay_command() -> None:
    """The Deployment docs carry the exact compose command and secrets step."""
    text = _HARDENING_DOC.read_text(encoding="utf-8")
    assert "docker compose -f docker-compose.yml -f docker-compose.platform.yml up -d" in text
    assert "generate_secret_key.py --all" in text
