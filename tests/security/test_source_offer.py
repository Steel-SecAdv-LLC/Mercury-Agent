# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The AGPL section 13 corresponding-source offer: validated, always served."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine._version import __version__
from omni_mercury_engine.api.routes.source import router as source_router
from omni_mercury_engine.security.source_offer import (
    LICENSE_ID,
    LICENSE_URL,
    SOURCE_REVISION_ENV,
    SOURCE_URL_ENV,
    UPSTREAM_SOURCE_URL,
    configured_revision,
    configured_source_url,
    source_offer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FORK = "https://git.example.org/acme/mercury-fork"
SHA1 = "8d6d4e7a1b2c3d4e5f60718293a4b5c6d7e8f901"

#: Each live-application test gets its own client address (own rate bucket).
_ADDRESSES = (f"198.51.100.{n}" for n in itertools.count(200))


class TestSourceUrl:
    """Only a servable https URL is ever offered."""

    def test_unset_offers_upstream(self) -> None:
        """A stock deployment points at the upstream repository."""
        assert configured_source_url({}) == UPSTREAM_SOURCE_URL

    def test_blank_offers_upstream(self) -> None:
        """Whitespace is unset, not a URL."""
        assert configured_source_url({SOURCE_URL_ENV: "   "}) == UPSTREAM_SOURCE_URL

    def test_valid_fork_is_offered(self) -> None:
        """A modified deployment's published fork is offered as given."""
        assert configured_source_url({SOURCE_URL_ENV: FORK}) == FORK

    def test_surrounding_whitespace_is_stripped(self) -> None:
        """Copy-paste padding does not invalidate a good URL."""
        assert configured_source_url({SOURCE_URL_ENV: f"  {FORK}\n"}) == FORK

    @pytest.mark.parametrize(
        "bad",
        [
            "http://git.example.org/fork",  # not https
            "ftp://git.example.org/fork",
            "git.example.org/fork",  # no scheme
            "https://",  # no host
            "https:///path-only",
            "https://user:token@git.example.org/fork",  # credentials would leak
            "https://token@git.example.org/fork",
            "https://git.example.org/a fork",  # whitespace
            "https://git.example.org/\x00fork",  # non-printable
            "https://git.example.org/" + "a" * 2048,  # over length
            "javascript:alert(1)",
        ],
    )
    def test_invalid_falls_back_to_upstream(
        self, bad: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An invalid value is logged and never served."""
        with caplog.at_level(logging.WARNING, logger="omni_mercury_engine.security.source_offer"):
            assert configured_source_url({SOURCE_URL_ENV: bad}) == UPSTREAM_SOURCE_URL
        assert any(SOURCE_URL_ENV in r.message for r in caplog.records)

    def test_rejected_credentials_never_reach_the_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A token in a refused URL is not echoed into the logs either."""
        secret = "ghp_s3cr3tT0k3n"
        with caplog.at_level(logging.DEBUG, logger="omni_mercury_engine.security.source_offer"):
            configured_source_url({SOURCE_URL_ENV: f"https://{secret}@git.example.org/fork"})
        assert caplog.records
        assert all(secret not in r.getMessage() for r in caplog.records)

    def test_url_at_the_length_limit_is_served(self) -> None:
        """2048 characters is the limit, inclusive."""
        url = "https://git.example.org/" + "a" * (2048 - len("https://git.example.org/"))
        assert len(url) == 2048
        assert configured_source_url({SOURCE_URL_ENV: url}) == url

    def test_one_character_past_the_limit_is_refused(self) -> None:
        """2049 characters is refused: the limit is exact, not approximate."""
        url = "https://git.example.org/" + "a" * (2049 - len("https://git.example.org/"))
        assert len(url) == 2049
        assert configured_source_url({SOURCE_URL_ENV: url}) == UPSTREAM_SOURCE_URL

    def test_reads_process_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default mapping is the process environment."""
        monkeypatch.setenv(SOURCE_URL_ENV, FORK)
        assert configured_source_url() == FORK


class TestRevision:
    """Only a Git object id is ever served as the revision."""

    def test_unset_is_none(self) -> None:
        """No declared revision: the offer says so rather than guessing."""
        assert configured_revision({}) is None

    @pytest.mark.parametrize("good", [SHA1, SHA1[:7], "a" * 64])
    def test_object_ids_are_served(self, good: str) -> None:
        """Abbreviated and full SHA-1, and SHA-256, ids are accepted."""
        assert configured_revision({SOURCE_REVISION_ENV: good}) == good

    def test_upper_case_is_normalised(self) -> None:
        """Hex case is cosmetic; the served id is lower case."""
        assert configured_revision({SOURCE_REVISION_ENV: SHA1.upper()}) == SHA1

    @pytest.mark.parametrize(
        "bad",
        [SHA1[:6], "g" * 40, "a" * 41, "a" * 65, "v3.0.0", "main", "8d6d4e7 extra"],
    )
    def test_invalid_is_omitted(self, bad: str, caplog: pytest.LogCaptureFixture) -> None:
        """A non-id is logged and omitted, never served."""
        with caplog.at_level(logging.WARNING, logger="omni_mercury_engine.security.source_offer"):
            assert configured_revision({SOURCE_REVISION_ENV: bad}) is None
        assert any(SOURCE_REVISION_ENV in r.message for r in caplog.records)


class TestOffer:
    """The offer names the license, the running version and its source."""

    def test_stock_offer(self) -> None:
        """Every field, for a stock deployment."""
        offer = source_offer({})
        assert offer["license"] == LICENSE_ID == "AGPL-3.0-or-later"
        assert offer["license_url"] == LICENSE_URL
        assert offer["version"] == __version__
        assert offer["source_url"] == UPSTREAM_SOURCE_URL
        assert offer["revision"] is None
        assert offer["upstream"] == UPSTREAM_SOURCE_URL
        notice = str(offer["notice"])
        assert "Affero" in notice
        assert "section 13" in notice
        assert "source_url" in notice

    def test_modified_deployment_offer(self) -> None:
        """A fork's offer points at the fork and keeps the upstream for provenance."""
        offer = source_offer({SOURCE_URL_ENV: FORK, SOURCE_REVISION_ENV: SHA1})
        assert offer["source_url"] == FORK
        assert offer["revision"] == SHA1
        assert offer["upstream"] == UPSTREAM_SOURCE_URL

    def test_license_id_matches_the_distribution(self) -> None:
        """The served license is the one pyproject declares."""
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert f'license = {{text = "{LICENSE_ID}"}}' in pyproject

    def test_upstream_is_this_repository(self) -> None:
        """The upstream default is the repository the README names."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert UPSTREAM_SOURCE_URL in readme


class TestRoute:
    """GET /source serves the offer, publicly."""

    def test_route_serves_the_offer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The route is the module's offer, as JSON, for the process environment."""
        monkeypatch.setenv(SOURCE_URL_ENV, FORK)
        app = FastAPI()
        app.include_router(source_router)
        response = TestClient(app).get("/source")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == source_offer()
        assert response.json()["source_url"] == FORK

    def test_live_application_serves_it_without_credentials(self) -> None:
        """Registered on the real app and reachable with no credential at all."""
        from omni_mercury_engine.api.server import app

        client = TestClient(app, client=(next(_ADDRESSES), 50000))
        response = client.get("/source")
        assert response.status_code == 200
        assert response.json()["license"] == LICENSE_ID

    def test_documented_in_the_openapi_schema(self) -> None:
        """The offer is published, not hidden: it appears in the schema."""
        from omni_mercury_engine.api.server import app

        assert "/source" in app.openapi()["paths"]
