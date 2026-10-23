# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trust-surface routes: the public notice, security.txt, and the opt-in decoys.

Three layers:

* **Documents** -- ``/.well-known/security.txt`` and ``/agent-notice`` serve
  the notice module's content with the media types the standards require.
* **Decoys** -- each decoy records a tripwire, holds through the tarpit, and
  answers *indistinguishably* from the real thing: a differential test pins
  the auth challenge to a real protected route on the live application, and
  silent tripwires to a path that does not exist, for every method.
* **Contract** -- off by default, no decoy route exists; decoys never appear in
  the OpenAPI schema; no real route serves a decoy path; a spoofed
  ``X-Forwarded-For`` cannot frame an innocent address; any internal failure
  answers as a missing path, never a 500.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.exception_handlers import http_exception_handler
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from omni_mercury_engine.api.routes import decoys as decoy_routes
from omni_mercury_engine.api.routes.agent_notice import router as notice_router
from omni_mercury_engine.api.routes.decoys import register_decoys, secure_audit_sink
from omni_mercury_engine.security.agent_notice import as_dict
from omni_mercury_engine.security.deception import (
    DECEPTION_ENABLED_ENV,
    DEFAULT_DECOYS,
    DeceptionSensor,
    TarpitGate,
    TripwireEvent,
)
from omni_mercury_engine.security.secure_audit_logging import (
    AuditEventCategory,
    AuditEventSeverity,
)

#: Includes the methods a route-based tripwire answered with a telltale 405.
ALL_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "PROPFIND")
AUTH_DECOYS = [d for d in DEFAULT_DECOYS if d.response == "auth_challenge"]
SILENT_DECOYS = [d for d in DEFAULT_DECOYS if d.response == "not_found"]

#: Each live-application test gets its own client address, hence its own
#: rate-limit bucket: the live app's limiter is shared across the session.
_ADDRESSES = (f"198.51.100.{n}" for n in itertools.count(10))


class RecordingSleep:
    """Awaitable sleeper that records requested delays instead of sleeping."""

    def __init__(self) -> None:
        """Start with no recorded delays."""
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """Record ``seconds``."""
        self.delays.append(seconds)


class Harness:
    """A decoy-enabled app with an inspectable sensor, sink and tarpit."""

    def __init__(self, *, client_ip: str = "203.0.113.9") -> None:
        """Build the app; requests come from ``client_ip``."""
        self.events: list[TripwireEvent] = []
        self.sleep = RecordingSleep()
        self.sensor = DeceptionSensor(audit_sink=self.events.append)
        self.app = FastAPI()
        registered = register_decoys(
            self.app,
            enabled=True,
            sensor=self.sensor,
            gate=TarpitGate(4, sleep=self.sleep),
        )
        assert registered is self.sensor
        self.client = TestClient(self.app, client=(client_ip, 50000))


@pytest.fixture
def harness() -> Harness:
    """A fresh decoy harness."""
    return Harness()


@pytest.fixture
def live_client() -> TestClient:
    """A client of the real application, on a private rate-limit bucket."""
    from omni_mercury_engine.api.server import app

    return TestClient(app, client=(next(_ADDRESSES), 50000), raise_server_exceptions=False)


class TestDocuments:
    """The always-on public documents."""

    @pytest.fixture
    def client(self) -> TestClient:
        app = FastAPI()
        app.include_router(notice_router)
        return TestClient(app)

    def test_security_txt_media_type_and_fields(self, client: TestClient) -> None:
        """RFC 9116 section 3: text/plain with charset utf-8; Contact and Expires present."""
        response = client.get("/.well-known/security.txt")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert "\nContact: " in response.text
        assert response.text.count("\nExpires: ") == 1
        assert response.text.endswith("\n")

    def test_security_txt_honours_operator_contact(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A third-party deployment's contact replaces the upstream default."""
        monkeypatch.setenv("MERCURY_SECURITY_CONTACT", "mailto:ops@example.org")
        body = client.get("/.well-known/security.txt").text
        assert "Contact: mailto:ops@example.org\n" in body
        assert "SECURITY.md#reporting" not in body

    def test_malformed_operator_contact_never_breaks_the_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A contact the URI parser rejects is dropped; the file is still served."""
        app = FastAPI()
        app.include_router(notice_router)
        client = TestClient(app, raise_server_exceptions=False)
        monkeypatch.setenv("MERCURY_SECURITY_CONTACT", "https://[::1, mailto:ops@example.org")
        response = client.get("/.well-known/security.txt")
        assert response.status_code == 200
        assert "Contact: mailto:ops@example.org\n" in response.text
        assert "[::1" not in response.text

    def test_agent_notice_json(self, client: TestClient) -> None:
        """The notice is served as its module renders it."""
        response = client.get("/agent-notice")
        assert response.status_code == 200
        assert response.json() == as_dict()
        assert response.json()["binding"] is False

    def test_documents_accept_no_input(self, client: TestClient) -> None:
        """Read-only: writes are refused."""
        assert client.post("/agent-notice", json={"statement": "x"}).status_code == 405
        assert client.post("/.well-known/security.txt").status_code == 405

    def test_security_txt_excluded_from_schema(self) -> None:
        """security.txt is a well-known file, not an API operation."""
        app = FastAPI()
        app.include_router(notice_router)
        paths = app.openapi()["paths"]
        assert "/.well-known/security.txt" not in paths
        assert "/agent-notice" in paths


class TestDecoyResponses:
    """Decoys answer as the real thing and record as tripwires."""

    @pytest.mark.parametrize("decoy", AUTH_DECOYS, ids=lambda d: d.path)
    def test_auth_challenge(self, harness: Harness, decoy: Any) -> None:
        """Every auth decoy issues the protected-route challenge, for each of its methods."""
        for method in decoy.methods:
            response = harness.client.request(method, decoy.path)
            assert response.status_code == 401
            assert response.json() == {"detail": "Authentication required"}
            assert response.headers["www-authenticate"] == "Bearer, ApiKey"

    @pytest.mark.parametrize("decoy", SILENT_DECOYS, ids=lambda d: d.path)
    def test_silent_tripwire_matches_a_missing_path(self, harness: Harness, decoy: Any) -> None:
        """Every method and path variant answers exactly as a path that does not exist.

        Fails on the route-based tripwires this replaced: OPTIONS, TRACE and
        PROPFIND answered 405, and the trailing-slash form a 307 redirect.
        """
        for suffix in ("", "/", "?probe=1"):
            for method in ALL_METHODS:
                decoy_response = harness.client.request(
                    method, decoy.path + suffix, follow_redirects=False
                )
                missing = harness.client.request(
                    method, "/no-such-path-anywhere" + suffix, follow_redirects=False
                )
                assert decoy_response.status_code == missing.status_code == 404, (method, suffix)
                assert decoy_response.content == missing.content
                assert decoy_response.headers.get("content-type") == missing.headers.get(
                    "content-type"
                )

    @pytest.mark.parametrize("decoy", SILENT_DECOYS, ids=lambda d: d.path)
    def test_silent_tripwire_records_every_variant(self, harness: Harness, decoy: Any) -> None:
        """Observation covers every method and the trailing-slash form."""
        harness.client.request("PROPFIND", decoy.path)
        harness.client.get(decoy.path + "/")
        harness.client.get(decoy.path + "?probe=1")
        assert len(harness.sensor.outcomes("203.0.113.9")) == 1
        assert harness.events[0].method == "PROPFIND"
        assert harness.events[0].path == decoy.path

    def test_non_decoy_paths_are_never_observed(self, harness: Harness) -> None:
        """The middleware ignores every path that is not a silent tripwire."""
        for path in ("/no-such-path-anywhere", "/.envx", "/x/.env", "/.ENV"):
            harness.client.get(path)
        assert harness.events == []

    def test_first_touch_has_real_route_latency(self, harness: Harness) -> None:
        """No delay on first contact: a delayed first answer fingerprints the decoy."""
        harness.client.get("/api/v1/admin")
        harness.client.get("/.env")
        assert harness.sleep.delays == [0.5]
        assert len(harness.events) == 2

    def test_each_touch_is_recorded_and_tarpitted(self, harness: Harness) -> None:
        """Touches escalate the tarpit; the gate holds each scheduled delay."""
        for _ in range(3):
            harness.client.get("/api/v1/admin")
        assert harness.sleep.delays == [0.5, 1.0]
        assert len(harness.events) == 1
        assert harness.events[0].path == "/api/v1/admin"

    def test_breadth_is_audited_per_decoy(self, harness: Harness) -> None:
        """Each newly touched decoy is a new audit record."""
        harness.client.get("/api/v1/admin")
        harness.client.post("/api/v1/gate/override")
        harness.client.get("/.env")
        assert [e.path for e in harness.events] == [
            "/api/v1/admin",
            "/api/v1/gate/override",
            "/.env",
        ]
        assert harness.events[-1].distinct_decoys == 3

    @pytest.mark.parametrize(
        ("headers", "presented"),
        [
            ({}, False),
            ({"X-API-Key": "stolen-or-guessed"}, True),
            ({"Authorization": "Bearer abc"}, True),
        ],
    )
    def test_credential_presence_recorded(
        self, harness: Harness, headers: dict[str, str], presented: bool
    ) -> None:
        """Presenting credentials to a decoy is flagged; the credential is not stored."""
        harness.client.get("/api/v1/admin", headers=headers)
        event = harness.events[0]
        assert event.credential_presented is presented
        assert "stolen-or-guessed" not in repr(event)
        assert "abc" not in str(event.audit_details())


class TestUnforgeableKeying:
    """A forged forwarding header cannot frame an innocent address."""

    def test_spoofed_forwarded_for_ignored_by_default(self) -> None:
        """With no trusted proxy hops, the TCP peer is the source."""
        harness = Harness(client_ip="203.0.113.50")
        harness.client.get("/api/v1/admin", headers={"X-Forwarded-For": "192.0.2.99"})
        assert harness.events[0].source_key == "203.0.113.50"
        assert harness.sensor.outcomes("192.0.2.99") == ()

    def test_trusted_hop_honoured_only_when_declared(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With one declared hop, the right-most forwarded entry is the source."""
        monkeypatch.setenv("MERCURY_TRUSTED_PROXY_HOPS", "1")
        harness = Harness(client_ip="10.0.0.2")
        harness.client.get("/api/v1/admin", headers={"X-Forwarded-For": "192.0.2.99, 203.0.113.60"})
        assert harness.events[0].source_key == "203.0.113.60"


class TestFailClosed:
    """An internal failure answers as a missing path -- never a 500, never a leak."""

    @pytest.mark.parametrize(
        ("decoy", "logged"),
        [
            (AUTH_DECOYS[0], "answering as a missing path"),
            (SILENT_DECOYS[0], "failed to record; passing through"),
        ],
        ids=["auth_challenge", "silent"],
    )
    def test_sensor_failure_answers_not_found(
        self,
        harness: Harness,
        decoy: Any,
        logged: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A raising sensor yields the plain 404 of a nonexistent path, logged."""

        def broken(*args: object, **kwargs: object) -> TripwireEvent:
            raise RuntimeError("sensor down")

        monkeypatch.setattr(harness.sensor, "record", broken)
        missing = harness.client.get("/no-such-path-anywhere")
        with caplog.at_level(logging.ERROR, logger=decoy_routes.__name__):
            response = harness.client.get(decoy.path)
        assert response.status_code == 404
        assert response.content == missing.content
        assert logged in caplog.text


class TestRegistrationContract:
    """Off by default; when on, invisible to the schema."""

    def test_disabled_registers_nothing(self) -> None:
        """No decoy route exists when disabled."""
        app = FastAPI()
        assert register_decoys(app, enabled=False) is None
        assert not hasattr(app.state, "deception_sensor")
        client = TestClient(app)
        for decoy in DEFAULT_DECOYS:
            assert client.get(decoy.path).status_code == 404

    def test_environment_controls_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """enabled=None reads MERCURY_DECEPTION_ENABLED."""
        monkeypatch.delenv(DECEPTION_ENABLED_ENV, raising=False)
        assert register_decoys(FastAPI()) is None
        monkeypatch.setenv(DECEPTION_ENABLED_ENV, "true")
        monkeypatch.delenv("MERCURY_AUDIT_LOG_DIR", raising=False)
        app = FastAPI()
        sensor = register_decoys(app)
        assert isinstance(sensor, DeceptionSensor)
        assert app.state.deception_sensor is sensor
        assert sensor.decoys == DEFAULT_DECOYS

    def test_default_sensor_uses_the_shared_store_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """MERCURY_KEYSTORE_PATH gives the default sensor cluster-wide state."""
        monkeypatch.setenv(DECEPTION_ENABLED_ENV, "true")
        monkeypatch.delenv("MERCURY_AUDIT_LOG_DIR", raising=False)
        state_db = tmp_path / "state.db"
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(state_db))
        sensor = register_decoys(FastAPI())
        assert sensor is not None
        # Observable through behaviour, not internals: a second sensor built
        # the same way sees the first one's touch in its counts.
        sensor.record("198.51.100.7", DEFAULT_DECOYS[0].path, "GET")
        monkeypatch.setenv(DECEPTION_ENABLED_ENV, "true")
        twin = register_decoys(FastAPI())
        assert twin is not None
        event = twin.record("198.51.100.7", DEFAULT_DECOYS[0].path, "GET")
        assert event.hits_in_window == 2

    def test_decoys_absent_from_openapi(self, harness: Harness) -> None:
        """No decoy path appears in the published schema."""
        paths = set(harness.app.openapi().get("paths", {}))
        assert paths.isdisjoint(d.path for d in DEFAULT_DECOYS)


class TestLiveApplication:
    """Against the real application: placement, collisions, and the differential pin."""

    def test_documents_served(self, live_client: TestClient) -> None:
        """The real app serves both public documents."""
        assert live_client.get("/.well-known/security.txt").status_code == 200
        assert live_client.get("/agent-notice").json()["binding"] is False

    def test_no_real_route_serves_a_decoy_path(self, live_client: TestClient) -> None:
        """Measured, not inferred: with decoys off, every decoy path is a 404."""
        for decoy in DEFAULT_DECOYS:
            for method in ("GET", "POST"):
                status = live_client.request(method, decoy.path).status_code
                assert status == 404, f"{method} {decoy.path} answered {status}"

    def test_premise_live_app_uses_the_default_http_exception_handler(self) -> None:
        """The differential test below is valid only while this holds."""
        from omni_mercury_engine.api.server import app

        assert app.exception_handlers[StarletteHTTPException] is http_exception_handler

    def test_decoy_challenge_is_indistinguishable_from_a_real_protected_route(
        self, live_client: TestClient, harness: Harness
    ) -> None:
        """Differential pin: decoy 401 == real anonymous 401, field for field."""
        real = live_client.get("/api/v1/models")
        decoy = harness.client.get("/api/v1/admin")
        assert real.status_code == decoy.status_code == 401
        assert decoy.content == real.content
        assert decoy.headers["www-authenticate"] == real.headers["www-authenticate"]
        assert decoy.headers["content-type"] == real.headers["content-type"]

    def test_decoy_variants_behave_as_a_real_route(
        self, live_client: TestClient, harness: Harness
    ) -> None:
        """Wrong methods and the trailing-slash form answer as a real GET/POST route does."""
        for method in ("OPTIONS", "PROPFIND", "DELETE"):
            real = live_client.request(method, "/api/v1/models")
            decoy = harness.client.request(method, "/api/v1/admin")
            assert decoy.status_code == real.status_code == 405, method
            assert decoy.headers["allow"] == real.headers["allow"], method
        real_slash = live_client.get("/api/v1/models/", follow_redirects=False)
        decoy_slash = harness.client.get("/api/v1/admin/", follow_redirects=False)
        assert decoy_slash.status_code == real_slash.status_code == 307


class TestSecureAuditSink:
    """Tripwires land on the shared tamper-evident chain, or in structured logs."""

    @staticmethod
    def _event() -> TripwireEvent:
        return DeceptionSensor().record("203.0.113.9", "/api/v1/admin", "GET")

    def test_writes_a_security_incident(self) -> None:
        """Category, severity, actor, resource and the dedicated IP field."""
        calls: list[dict[str, Any]] = []

        class FakeLogger:
            def log(self, **kwargs: Any) -> str:
                calls.append(kwargs)
                return "AE-1"

        event = self._event()
        secure_audit_sink(FakeLogger())(event)
        assert calls == [
            {
                "category": AuditEventCategory.SECURITY_INCIDENT,
                "action": "deception.tripwire",
                "outcome": "failure",
                "severity": AuditEventSeverity.SECURITY,
                "actor": "203.0.113.9",
                "resource": "/api/v1/admin",
                "details": event.audit_details(),
                "client_ip": "203.0.113.9",
            }
        ]

    def test_structured_log_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without a secure logger, the tripwire is a structured warning."""
        with caplog.at_level(logging.WARNING, logger=decoy_routes.__name__):
            secure_audit_sink(None)(self._event())
        assert "deception tripwire source=203.0.113.9 decoy=/api/v1/admin" in caplog.text
        assert "credential_presented=False" in caplog.text


class TestSessionCredential:
    """A stolen browser session probing a decoy is the highest-value signal."""

    def test_session_cookie_name_matches_the_account_routes(self) -> None:
        """Pinned, not imported: the decoy surface never loads the account stack."""
        from omni_mercury_engine.api.routes.accounts import SESSION_COOKIE

        assert decoy_routes._SESSION_COOKIE == SESSION_COOKIE

    @pytest.mark.parametrize(
        ("cookies", "presented"),
        [
            ({"mercury_session": "hijacked"}, True),
            ({"mercury_csrf": "token-only"}, False),
            ({"unrelated": "x"}, False),
        ],
    )
    @pytest.mark.parametrize("path", ["/api/v1/admin", "/.env"], ids=["route", "silent"])
    def test_session_cookie_flagged(
        self, harness: Harness, path: str, cookies: dict[str, str], presented: bool
    ) -> None:
        """Fails on the replaced check, which looked only at headers.

        A session cookie flags the touch on both decoy kinds; a CSRF cookie
        alone is not a credential. The cookie value is never recorded.
        """
        harness.client.cookies.clear()
        harness.client.get(path, cookies=cookies)
        event = harness.events[0]
        assert event.credential_presented is presented
        assert "hijacked" not in repr(event)


class TestCollisionRefusal:
    """A decoy the application already serves is refused, never registered."""

    @staticmethod
    def _app_with_real_routes() -> FastAPI:
        app = FastAPI()

        @app.get("/api/v1/things/{thing_id}")
        def thing(thing_id: str) -> dict[str, str]:
            return {"thing": thing_id}

        @app.get("/real/")
        def real_slash() -> dict[str, str]:
            return {"ok": "yes"}

        return app

    def test_colliding_decoys_refused_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Path-parameter, exact and slash-twin collisions are all refused."""
        from omni_mercury_engine.security.deception import DecoyRoute

        app = self._app_with_real_routes()
        events: list[TripwireEvent] = []
        decoys = (
            DecoyRoute("/api/v1/things/export", "collides via {thing_id}"),
            DecoyRoute("/api/v1/things/backup", "collides via {thing_id}", "not_found"),
            DecoyRoute("/real", "collides via trailing-slash twin"),
            DecoyRoute("/safe/private", "no collision"),
        )
        sensor = DeceptionSensor(decoys=decoys, audit_sink=events.append)
        with caplog.at_level(logging.ERROR, logger=decoy_routes.__name__):
            register_decoys(app, enabled=True, sensor=sensor, gate=TarpitGate(4))
        assert [d.path for d in app.state.deception_decoys] == ["/safe/private"]
        assert caplog.text.count("served by a real route") == 3
        client = TestClient(app)
        assert client.get("/api/v1/things/export").json() == {"thing": "export"}
        assert client.get("/api/v1/things/backup").json() == {"thing": "backup"}
        assert events == []
        assert client.get("/safe/private").status_code == 401
        assert [e.path for e in events] == ["/safe/private"]

    def test_configured_private_decoys_are_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Private paths from the environment are live alongside the defaults."""
        monkeypatch.setenv(DECEPTION_ENABLED_ENV, "true")
        monkeypatch.setenv("MERCURY_DECOY_PATHS", "/api/v1/private/console")
        monkeypatch.setenv("MERCURY_SILENT_DECOY_PATHS", "/private.bak")
        monkeypatch.delenv("MERCURY_AUDIT_LOG_DIR", raising=False)
        app = FastAPI()
        sensor = register_decoys(app)
        assert sensor is not None
        client = TestClient(app, client=("203.0.113.70", 50000))
        assert client.get("/api/v1/private/console").status_code == 401
        assert client.get("/private.bak").status_code == 404
        assert len(sensor.outcomes("203.0.113.70")) == 2


class TestOperatorConfiguration:
    """Every trust-surface variable reaches the operator's deployment surfaces.

    Failed against the replaced tree: none of the four appeared in the
    configuration reference, the Helm values, the kustomize env file or the
    compose overlay, so a third-party deployment served the upstream
    security contact with nothing prompting the operator to change it.
    """

    _ROOT = Path(__file__).resolve().parents[2]
    _VARS = (
        "MERCURY_SECURITY_CONTACT",
        DECEPTION_ENABLED_ENV,
        "MERCURY_DECOY_PATHS",
        "MERCURY_SILENT_DECOY_PATHS",
    )

    def test_the_names_are_the_modules_constants(self) -> None:
        from omni_mercury_engine.security import agent_notice, deception

        assert self._VARS == (
            agent_notice.SECURITY_CONTACT_ENV,
            deception.DECEPTION_ENABLED_ENV,
            deception.DECOY_PATHS_ENV,
            deception.SILENT_DECOY_PATHS_ENV,
        )

    def test_configuration_reference_documents_each(self) -> None:
        text = (self._ROOT / "docs" / "PLATFORM_HARDENING.md").read_text(encoding="utf-8")
        reference = text[text.index("## Configuration reference") :]
        for name in self._VARS:
            assert f"| `{name}` |" in reference, name

    def test_helm_values_carry_each_with_decoys_off(self) -> None:
        import yaml

        values = yaml.safe_load(
            (self._ROOT / "helm" / "mercury-agent" / "values.yaml").read_text(encoding="utf-8")
        )
        app_config = values["config"]["app"]
        assert {name: app_config[name] for name in self._VARS} == {
            "MERCURY_SECURITY_CONTACT": "",
            DECEPTION_ENABLED_ENV: "false",
            "MERCURY_DECOY_PATHS": "",
            "MERCURY_SILENT_DECOY_PATHS": "",
        }

    def test_kustomize_env_carries_each_with_decoys_off(self) -> None:
        lines = (
            (self._ROOT / "k8s" / "base" / "config" / "mercury-agent.env")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        env = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))
        assert {name: env[name] for name in self._VARS} == {
            "MERCURY_SECURITY_CONTACT": "",
            DECEPTION_ENABLED_ENV: "false",
            "MERCURY_DECOY_PATHS": "",
            "MERCURY_SILENT_DECOY_PATHS": "",
        }

    def test_compose_overlay_passes_each_through_with_decoys_off(self) -> None:
        import yaml

        compose = yaml.safe_load(
            (self._ROOT / "docker-compose.platform.yml").read_text(encoding="utf-8")
        )
        env = compose["services"]["mercury-agent"]["environment"]
        assert env[DECEPTION_ENABLED_ENV] == f"${{{DECEPTION_ENABLED_ENV}:-false}}"
        for name in (
            "MERCURY_SECURITY_CONTACT",
            "MERCURY_DECOY_PATHS",
            "MERCURY_SILENT_DECOY_PATHS",
        ):
            assert env[name] == f"${{{name}:-}}"

    def test_blank_shipped_values_mean_the_documented_defaults(self) -> None:
        """The blanks the surfaces ship resolve to upstream contact and no decoys."""
        from omni_mercury_engine.security.agent_notice import configured_contacts
        from omni_mercury_engine.security.deception import (
            DEFAULT_DECOYS,
            configured_decoys,
            deception_enabled,
        )

        shipped = {
            "MERCURY_SECURITY_CONTACT": "",
            DECEPTION_ENABLED_ENV: "false",
            "MERCURY_DECOY_PATHS": "",
            "MERCURY_SILENT_DECOY_PATHS": "",
        }
        assert configured_contacts(shipped) == configured_contacts({})
        assert deception_enabled(shipped) is False
        assert configured_decoys(shipped) == DEFAULT_DECOYS


class TestSilentTripwireSitsInsideTheStack:
    """A silent tripwire observes only what the application itself answers.

    Failed against the replaced registration, which added the middleware
    outermost: a request the rate limiter refused was still recorded, written
    to the store and held by the tarpit before its 429 -- an unthrottled write
    path, and a delayed 429 where a real missing path answers at once.
    """

    @staticmethod
    def _app(monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, DeceptionSensor, RecordingSleep]:
        from omni_mercury_engine.api.server import RateLimitMiddleware
        from omni_mercury_engine.security.deception import TarpitPolicy

        monkeypatch.setenv("OMNI_RATE_LIMIT_ENABLED", "true")
        monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=60,
            burst_size=3,
        )
        sleep = RecordingSleep()
        sensor = DeceptionSensor(policy=TarpitPolicy(free_hits=0))
        register_decoys(app, enabled=True, sensor=sensor, gate=TarpitGate(4, sleep=sleep))
        return app, sensor, sleep

    def test_refused_probe_is_neither_recorded_nor_held(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, sensor, sleep = self._app(monkeypatch)
        client = TestClient(app, client=("203.0.113.80", 50000))
        statuses = [client.get("/.env").status_code for _ in range(6)]
        assert statuses == [404, 404, 404, 429, 429, 429]
        assert len(sleep.delays) == 3  # only the three the application answered
        last = sensor.record("203.0.113.80", "/.env", "GET")
        assert last.hits_in_window == 4  # three observed touches, then this one

    def test_refusal_matches_a_real_missing_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app, _, sleep = self._app(monkeypatch)
        decoy = TestClient(app, client=("203.0.113.81", 50000))
        plain = TestClient(app, client=("203.0.113.82", 50000))
        for _ in range(3):
            decoy.get("/.env")
            plain.get("/no-such-path")
        held = len(sleep.delays)
        decoy_refusal = decoy.get("/.env")
        plain_refusal = plain.get("/no-such-path")
        assert len(sleep.delays) == held  # the refusal itself is never held
        assert decoy_refusal.status_code == plain_refusal.status_code == 429
        assert decoy_refusal.content == plain_refusal.content
        assert sorted(decoy_refusal.headers) == sorted(plain_refusal.headers)

    def test_registration_after_start_is_refused(self) -> None:
        app = FastAPI()
        TestClient(app).get("/")  # builds the middleware stack
        with pytest.raises(RuntimeError, match="after an application has started"):
            register_decoys(app, enabled=True, sensor=DeceptionSensor(), gate=TarpitGate(1))
