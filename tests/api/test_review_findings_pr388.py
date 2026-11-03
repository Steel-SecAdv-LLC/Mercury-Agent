# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression pins for the review findings on the console and crypto changes.

Each test reproduces one finding from the automated review of the pull request
that served the console and routed operational crypto through AMA, and fails
on the implementation that drew the finding:

* a truncated legacy ``pbkdf2_sha256`` row verified against a prefix;
* the self-model probe read one of the two switches the FINDΩYOU™ bridge needs;
* the console's boot progress bar kept animating under reduced motion;
* the core detection routes recorded an authenticated caller's run as
  ``anonymous``, where the caller's own history filter could not find it.

The memory-scoping finding is pinned in ``test_voice_recent_detections.py``
and the two loader exclusions in the label-provenance gate.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from omni_mercury_engine.api import frontend, passwords
from omni_mercury_engine.api.auth import User

if TYPE_CHECKING:
    import pytest

_PORT_CSS = frontend._STATIC_DIR / "console" / "port.css"


def test_truncated_legacy_pbkdf2_row_never_verifies() -> None:
    """A one-byte stored digest used to verify any password matching that byte."""
    import hashlib

    salt = b"\x11" * 16
    full = hashlib.pbkdf2_hmac("sha256", b"correct horse", salt, 1000)
    good = f"pbkdf2_sha256$1000${salt.hex()}${full.hex()}"
    assert passwords.verify_password("correct horse", good) is True
    truncated = f"pbkdf2_sha256$1000${salt.hex()}${full[:1].hex()}"
    # Deriving to the stored length made this a 1-in-256 check; try every byte.
    assert all(passwords.verify_password(f"guess-{i}", truncated) is False for i in range(300))
    assert passwords.verify_password("correct horse", truncated) is False


def test_self_model_reports_either_bridge_switch_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bridge needs ``findoyou`` and ``ama_optional``; a probe that read one lied."""
    from omni_mercury_engine import self_model

    spec = next(c for c in self_model.CAPABILITIES if c.name == "findoyou_evidence_bridge")
    assert spec.all_switches == ("findoyou", "ama_optional")
    monkeypatch.setenv("MERCURY_DECOUPLE_FINDOYOU", "1")
    monkeypatch.setenv("MERCURY_DECOUPLE_AMA_OPTIONAL", "0")
    probe = self_model._probe_capability(spec)
    assert probe["state"] == "unavailable", probe
    assert "ama_optional" in probe["reason"] and "decouple ama-optional on" in probe["reason"]


def test_console_progress_bar_yields_to_reduced_motion() -> None:
    """The prototype restored the infinite boot animation with !important."""
    css = _PORT_CSS.read_text(encoding="utf-8")
    assert not re.search(r"\.mcli-progress\{animation:[^}]*infinite !important", css)
    assert (
        "@media (prefers-reduced-motion: no-preference){html:not(.mn-reduce-motion) .mcli-progress"
        in css
    )
    assert "@media (prefers-reduced-motion: reduce){.mcli-progress{animation:none;}}" in css
    assert "html.mn-reduce-motion .mcli-progress{animation:none !important;}" in css


def test_core_detection_records_the_authenticated_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """An API-key or JWT caller's univariate run is attributed to their account."""
    from omni_mercury_engine.api import server
    from omni_mercury_engine.api.routes import export

    recorded: list[str] = []

    async def _capture(**kwargs):
        recorded.append(kwargs["user_id"])

    monkeypatch.setattr(export, "record_detection", _capture)
    owner = User(id="acct-42", username="owner", roles=["user"])
    server.app.dependency_overrides[server._get_optional_user] = lambda: owner
    try:
        with TestClient(server.app) as client:
            recorded.clear()  # the warm-up's synthetic calls are excluded by contract
            data = [0.1, 0.2, 0.1, 9.0, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.15, 0.1]
            assert client.post("/api/v1/detect/univariate", json={"data": data}).status_code == 200
            rows = [[0.1, 0.2], [0.1, 0.3], [9.0, 9.5], [0.2, 0.1], [0.1, 0.2], [0.2, 0.2]]
            assert (
                client.post("/api/v1/detect/multivariate", json={"data": rows}).status_code == 200
            )
    finally:
        server.app.dependency_overrides.pop(server._get_optional_user, None)
    assert recorded == ["acct-42", "acct-42"], recorded


def test_core_detection_without_credentials_stays_anonymous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omni_mercury_engine.api import server
    from omni_mercury_engine.api.routes import export

    recorded: list[str] = []

    async def _capture(**kwargs):
        recorded.append(kwargs["user_id"])

    monkeypatch.setattr(export, "record_detection", _capture)
    with TestClient(server.app) as client:
        recorded.clear()
        data = [0.1, 0.2, 0.1, 9.0, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.15, 0.1]
        assert client.post("/api/v1/detect/univariate", json={"data": data}).status_code == 200
    assert recorded == ["anonymous"], recorded
