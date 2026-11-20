# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The served API's bounds have one source of truth, and every surface reads it.

Before this, ten serving bounds lived as literals or independent
``os.environ`` reads across five files, operators could not see the active
numbers without reading source, and the OpenAPI ``x-rate-limit`` extension
was the literal ``{100, 20}`` whatever ``OMNI_RATE_LIMIT_*`` was set to — a
tuned deployment advertised a limit it did not enforce.  These tests pin:

* validation is loud (a bad value raises naming its variable);
* an override reaches every enforcing site AND every advertising surface
  identically (the drift test — it fails on the hardcoded prior state);
* the flagship admission bound's 429s are counted and served.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.serving_limits import (
    ServingLimitConfigError,
    load_serving_limits,
    serving_limits,
)

_ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_KEYS = {
    "max_body_bytes",
    "max_upload_body_bytes",
    "flagship_max_queue",
    "flagship_max_rows",
    "flagship_max_domain_chars",
    "flagship_max_subject_id_chars",
    "alert_bus_max_subscribers",
    "ipv6_rate_limit_prefix",
    "rate_limit_requests_per_minute",
    "rate_limit_burst",
}


class TestValidation:
    def test_defaults_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "MERCURY_MAX_BODY_BYTES",
            "MERCURY_FLAGSHIP_MAX_QUEUE",
            "MERCURY_IPV6_RATE_LIMIT_PREFIX",
        ):
            monkeypatch.delenv(key, raising=False)
        limits = load_serving_limits()
        assert limits.max_body_bytes == 10 * 1024 * 1024
        assert limits.flagship_max_queue == 8
        assert limits.ipv6_rate_limit_prefix == 64

    @pytest.mark.parametrize(
        ("env", "value", "match"),
        [
            ("MERCURY_FLAGSHIP_MAX_QUEUE", "0", "out of range"),
            ("MERCURY_FLAGSHIP_MAX_ROWS", "-5", "out of range"),
            ("MERCURY_IPV6_RATE_LIMIT_PREFIX", "129", r"in \[1, 128\]"),
            ("MERCURY_IPV6_RATE_LIMIT_PREFIX", "0", r"in \[1, 128\]"),
            ("MERCURY_MAX_BODY_BYTES", "ten", "not an integer"),
            ("MERCURY_ALERT_BUS_MAX_SUBSCRIBERS", "1.5", "not an integer"),
        ],
    )
    def test_bad_values_fail_loudly_naming_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, env: str, value: str, match: str
    ) -> None:
        monkeypatch.setenv(env, value)
        with pytest.raises(ServingLimitConfigError, match=match) as excinfo:
            load_serving_limits()
        assert env in str(excinfo.value)

    def test_released_rate_limit_vars_keep_their_zero_semantics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMNI_RATE_LIMIT_* are released; their minimum stays 0, unchanged."""
        monkeypatch.setenv("OMNI_RATE_LIMIT_BURST", "0")
        assert load_serving_limits().rate_limit_burst == 0

    def test_every_limit_is_described(self) -> None:
        view = serving_limits().to_dict()
        assert set(view) == _EXPECTED_KEYS
        for name, limit in view.items():
            assert limit["env"], name
            assert limit["unit"], name
            assert limit["description"], name
            assert isinstance(limit["value"], int), name


class TestEverySurfaceReadsTheSameValue:
    def test_override_reaches_enforcement_and_advertisement_identically(self) -> None:
        """The drift test: fails on the prior state, where OpenAPI's
        x-rate-limit was the literal 100 whatever the environment said.

        Run in a fresh interpreter because the limits are read once at
        import (the pydantic field caps and the admission semaphore bind
        at import, exactly as before this module).
        """
        script = r"""
import json
from fastapi.testclient import TestClient
from omni_mercury_engine.api.server import app, RateLimitMiddleware
from omni_mercury_engine.api.client_ip import _normalize_ip
client = TestClient(app)
schema = client.get("/openapi.json").json()
limits = client.get("/api/v1/detect/flagship/limits").json()["limits"]
print(json.dumps({
    "openapi_rpm": schema["info"]["x-rate-limit"]["requests_per_minute"],
    "endpoint_rpm": limits["rate_limit_requests_per_minute"]["value"],
    "enforced_rpm": RateLimitMiddleware(app=lambda *a: None).requests_per_minute,
    "openapi_rows": schema["components"]["schemas"]["FlagshipDetectRequest"]
        ["properties"]["data"]["maxItems"],
    "endpoint_rows": limits["flagship_max_rows"]["value"],
    "ipv6": _normalize_ip("2001:db8:1:2:3::1"),
}))
"""
        env = {
            **os.environ,
            "OMNI_RATE_LIMIT_REQUESTS_PER_MINUTE": "37",
            "MERCURY_FLAGSHIP_MAX_ROWS": "500",
            "MERCURY_IPV6_RATE_LIMIT_PREFIX": "56",
        }
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
        assert observed["openapi_rpm"] == observed["endpoint_rpm"] == observed["enforced_rpm"] == 37
        assert observed["openapi_rows"] == observed["endpoint_rows"] == 500
        assert observed["ipv6"] == "2001:db8:1::/56"

    def test_openapi_carries_the_full_limit_set(self) -> None:
        pytest.importorskip("fastapi", reason="the API server needs the [api] extra")
        from fastapi.testclient import TestClient

        from omni_mercury_engine.api.server import app

        schema = TestClient(app).get("/openapi.json").json()
        assert set(schema["info"]["x-serving-limits"]) == _EXPECTED_KEYS


class TestAdmissionRejectionsAreCounted:
    def test_a_full_queue_returns_429_and_is_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("fastapi", reason="the API routes need the [api] extra")
        from fastapi import HTTPException

        from omni_mercury_engine.api.routes import detection

        # A queue with no free slot: every call is refused without ever
        # building the engine.
        monkeypatch.setattr(detection, "_flagship_admission", threading.BoundedSemaphore(1))
        detection._flagship_admission.acquire()
        stats = detection._AdmissionStats()
        monkeypatch.setattr(detection, "_admission_stats", stats)

        with pytest.raises(HTTPException) as excinfo:
            detection._run_flagship_detection(np.zeros((1, 4)), None, False)
        assert excinfo.value.status_code == 429
        assert excinfo.value.headers == {"Retry-After": "1"}
        snapshot = stats.snapshot()
        assert snapshot["rejected_total"] == 1
        assert snapshot["admitted_total"] == 0
        assert snapshot["rejection_rate"] == 1.0

    def test_admission_counters_track_in_flight_and_peak(self) -> None:
        pytest.importorskip("fastapi", reason="the API routes need the [api] extra")
        from omni_mercury_engine.api.routes.detection import _AdmissionStats

        stats = _AdmissionStats()
        stats.admit()
        stats.admit()
        stats.release()
        stats.reject()
        snapshot = stats.snapshot()
        assert snapshot["admitted_total"] == 2
        assert snapshot["in_flight"] == 1
        assert snapshot["peak_in_flight"] == 2
        assert snapshot["rejection_rate"] == pytest.approx(1 / 3)

    def test_decisions_summary_and_limits_endpoint_serve_admission(self) -> None:
        pytest.importorskip("fastapi", reason="the API server needs the [api] extra")
        from fastapi.testclient import TestClient

        from omni_mercury_engine.api.server import app

        client = TestClient(app)
        decisions = client.get("/api/v1/detect/flagship/decisions").json()
        limits = client.get("/api/v1/detect/flagship/limits").json()
        for payload in (decisions, limits):
            admission = payload["admission"]
            assert {"admitted_total", "rejected_total", "in_flight", "max_queue"} <= set(admission)
