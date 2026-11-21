# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Account emails link to the server that sent them.

``MERCURY_PUBLIC_BASE_URL`` defaulted to ``https://mercuryagent.global``, the
project's own site, and was absent from ``.env.example``. A server started
without it filed (or mailed) verification, password-reset and email-change
links to that host: found by registering on a local server and opening the
filed message, whose link was
``https://mercuryagent.global/verify-email?token=...``. The link did not reach
the server that issued the token, and the token went to a host the operator
does not run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the account service needs the [api] extra")

from omni_mercury_engine.api.auth_service import build_auth_service, public_base_url

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MERCURY_PUBLIC_BASE_URL",
        "MERCURY_SMTP_HOST",
        "OMNI_API_PORT",
        "MERCURY_KEYSTORE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_an_explicit_setting_is_used_without_its_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", " https://mercury.example.org/ ")
    assert public_base_url() == "https://mercury.example.org"


def test_the_development_mailer_links_to_this_server() -> None:
    assert public_base_url() == "http://127.0.0.1:8000"


def test_the_port_serve_listens_on_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNI_API_PORT", "8123")
    assert public_base_url() == "http://127.0.0.1:8123"


@pytest.mark.parametrize("port", ["", "  ", "eight", "80; rm"])
def test_a_port_that_is_not_a_number_falls_back(monkeypatch: pytest.MonkeyPatch, port: str) -> None:
    monkeypatch.setenv("OMNI_API_PORT", port)
    assert public_base_url() == "http://127.0.0.1:8000"


def test_real_mail_requires_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """With SMTP configured the mail leaves the machine; there is no safe guess."""
    monkeypatch.setenv("MERCURY_SMTP_HOST", "smtp.example.org")
    with pytest.raises(RuntimeError, match="MERCURY_PUBLIC_BASE_URL must be set"):
        public_base_url()
    with pytest.raises(RuntimeError, match="MERCURY_PUBLIC_BASE_URL must be set"):
        build_auth_service()
    monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", "https://mercury.example.org")
    assert public_base_url() == "https://mercury.example.org"


def test_the_project_site_is_never_the_default() -> None:
    assert "mercuryagent.global" not in public_base_url()
    service = build_auth_service()
    assert service._base_url == "http://127.0.0.1:8000"


def test_the_verification_link_in_a_filed_message_opens_on_this_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end through the development mailer: the link a user copies works locally."""
    monkeypatch.setenv("MERCURY_MAIL_OUTBOX", str(tmp_path))
    monkeypatch.setenv("OMNI_API_PORT", "8011")
    service = build_auth_service()
    service.register("link.check@example.org", "Correct-Horse-Battery-9!")
    (message,) = sorted(tmp_path.glob("*.txt"))
    text = message.read_text(encoding="utf-8")
    assert "http://127.0.0.1:8011/verify-email?token=" in text
    assert "mercuryagent.global/verify-email" not in text


def test_the_example_environment_documents_the_setting() -> None:
    text = (_REPO / ".env.example").read_text(encoding="utf-8")
    assert "MERCURY_PUBLIC_BASE_URL" in text
    assert "REQUIRED when MERCURY_SMTP_HOST is set" in text


def test_the_platform_deployment_sets_it_explicitly() -> None:
    """The hosted stack turns SMTP on, so it must carry the setting."""
    compose = (_REPO / "docker-compose.platform.yml").read_text(encoding="utf-8")
    assert "MERCURY_SMTP_HOST" in compose
    assert "MERCURY_PUBLIC_BASE_URL: ${MERCURY_PUBLIC_BASE_URL:-https://" in compose


@pytest.mark.parametrize("value", ["mercury.example.org", "ftp://mercury.example.org", "https://"])
def test_a_setting_that_is_not_a_web_url_is_refused(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Was: accepted as given, so emailed links read ``mercury.example.org/verify-email?...``."""
    monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", value)

    with pytest.raises(RuntimeError, match="needs a scheme and a host"):
        public_base_url()


def test_smtp_without_a_base_url_stops_the_server_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Was: the server booted, /health was healthy, and every account request answered 500."""
    from omni_mercury_engine.api import server

    server._require_account_email_base_url()  # development mailer: nothing to refuse

    monkeypatch.setenv("MERCURY_SMTP_HOST", "smtp.example.org")
    with pytest.raises(RuntimeError, match="MERCURY_PUBLIC_BASE_URL must be set"):
        server._require_account_email_base_url()

    monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", "https://mercury.example.org")
    server._require_account_email_base_url()


def test_an_enabled_frontend_that_failed_to_load_stops_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Was: the server booted with /health healthy and /login answering 404."""
    from omni_mercury_engine.api import server
    from omni_mercury_engine.api.router_manifest import RouterRecord, RouterState

    broken = [
        RouterRecord("frontend", "Account frontend", RouterState.UNAVAILABLE, "ImportError: x"),
        RouterRecord("accounts", "Account routes", RouterState.MOUNTED),
    ]
    monkeypatch.setattr(server, "router_manifest", lambda: broken)

    monkeypatch.setenv("MERCURY_FRONTEND_ENABLED", "false")
    server._require_frontend_backend()  # not asked for: the API alone is a valid server

    monkeypatch.setenv("MERCURY_FRONTEND_ENABLED", "true")
    with pytest.raises(RuntimeError, match="browser frontend did not load: ImportError: x"):
        server._require_frontend_backend()


def test_a_base_url_with_a_port_and_no_host_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """``https://:8080`` has a netloc and no host; it was accepted."""
    monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", "https://:8080")

    with pytest.raises(RuntimeError, match="needs a scheme and a host"):
        public_base_url()


@pytest.mark.parametrize(
    "value",
    ["https://host:8443/mercury", "HTTPS://Mercury.Example.org", "http://[::1]:8000"],
)
def test_a_path_a_port_and_an_ipv6_host_are_accepted(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("MERCURY_PUBLIC_BASE_URL", value)
    assert public_base_url() == value
