# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the served bounds: arbitrary bodies meet the byte cap and the field caps.

The body-size middleware and the flagship field caps were pinned by example
(``tests/api/test_body_size_limit.py``, ``tests/test_serving_limits.py``).
Here Hypothesis drives them with arbitrary input, and every number asserted
is read from the single source of truth (``serving_limits``), never typed in:

* **Byte cap** -- any chunking of any body, under any ``Content-Length``
  header (absent, exact, lying, negative, not a number): the app behind the
  middleware never reads past the cap, a body over it is always 413, and a
  body within it arrives byte for byte.
* **Field caps** -- any ``domain`` / ``subject_id`` length around its cap is
  accepted exactly when it is within it; the row cap holds at the boundary.
* **Served routes** -- arbitrary JSON bodies to ``/detect/flagship`` and the
  streaming tier ``/detect/tier`` never produce a 5xx, and a request that
  reaches the flagship engine is within every cap.

The fuzzer found a real defect on its first run, pinned in
``TestNonFiniteInputIsA422``: a body that failed validation on a non-finite
number (``{"data": Infinity}``, 18 bytes) was answered **500**.  The JSON
decoder accepts ``NaN`` / ``Infinity``; the validation error echoes the
offending input; FastAPI's default handler passed it to a ``JSONResponse``,
which refuses non-finite floats and raised.  Every route that validates a
body did it, signed in or not.  It is 422 now, in FastAPI's own shape.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import (
    HealthCheck,
    given,
    settings,
    strategies as st,
)

pytest.importorskip("fastapi", reason="the API server needs the [api] extra")

from fastapi.testclient import TestClient
from pydantic import ValidationError

from omni_mercury_engine.api.routes import detection
from omni_mercury_engine.api.routes.detection import FlagshipDetectRequest
from omni_mercury_engine.api.server import BodySizeLimitMiddleware, app
from omni_mercury_engine.serving_limits import serving_limits

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import numpy as np

_LIMITS = serving_limits()
_JSON = {"content-type": "application/json"}
# Hypothesis re-enters the test body per example; the fixtures used below
# (rate limiter off, a stand-in engine) are safe to share across examples.
_SHARED = [HealthCheck.too_slow, HealthCheck.function_scoped_fixture]

_floats = st.floats(allow_nan=True, allow_infinity=True)
_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    _floats,
    st.text(max_size=40),
)


# --------------------------------------------------------------------------- #
# Byte cap
# --------------------------------------------------------------------------- #
def _run_middleware(
    chunks: list[bytes], header: bytes | None, path: str, default_limit: int, upload_limit: int
) -> tuple[int, bytes]:
    """Drive the middleware; returns (status, the bytes the app behind it read)."""
    read = bytearray()

    async def consuming_app(scope: Any, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            read.extend(message.get("body") or b"")
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = BodySizeLimitMiddleware(consuming_app)
    middleware.default_limit, middleware.upload_limit = default_limit, upload_limit
    queue = list(chunks)
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> dict[str, Any]:
        body = queue.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(queue)}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    headers = [(b"content-length", header)] if header is not None else []
    scope = {"type": "http", "path": path, "headers": headers}
    asyncio.run(middleware(scope, receive, send))
    return int(sent[0]["status"]), bytes(read)


class TestByteCapUnderArbitraryBodies:
    @settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        chunks=st.lists(st.binary(max_size=40), min_size=1, max_size=6),
        declared=st.one_of(
            st.none(),  # chunked: no Content-Length at all
            st.integers(min_value=-10, max_value=400),  # exact, lying or negative
            # Not a decimal integer: declares nothing, so the count decides.
            st.sampled_from([b"", b"abc", b"1e3", b"0x10", b"9" * 5000]),
        ),
        default_limit=st.integers(min_value=0, max_value=96),
        upload_extra=st.integers(min_value=1, max_value=96),
        upload_path=st.booleans(),
    )
    def test_the_app_never_reads_past_the_cap(
        self,
        chunks: list[bytes],
        declared: int | bytes | None,
        default_limit: int,
        upload_extra: int,
        upload_path: bool,
    ) -> None:
        upload_limit = default_limit + upload_extra
        path = "/api/v1/models/upload" if upload_path else "/api/v1/detect/flagship"
        limit = upload_limit if upload_path else default_limit
        header = str(declared).encode() if isinstance(declared, int) else declared
        body = b"".join(chunks)

        status, read = _run_middleware(chunks, header, path, default_limit, upload_limit)

        assert len(read) <= limit, "the app read past the byte cap"
        declared_over = isinstance(declared, int) and declared > limit
        if declared_over or len(body) > limit:
            assert status == 413
        else:
            assert status == 200
            assert read == body  # within the cap, the body arrives untouched
        if declared_over:
            assert read == b""  # refused on the header alone, before any read


# --------------------------------------------------------------------------- #
# Field caps
# --------------------------------------------------------------------------- #
class TestFlagshipFieldCaps:
    @settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        field=st.sampled_from(["domain", "subject_id"]),
        length=st.integers(min_value=0, max_value=3 * _LIMITS.flagship_max_subject_id_chars),
        char=st.sampled_from(["a", " ", "é", "漢", "\U0001f600", "\x00"]),
    )
    def test_a_string_is_accepted_exactly_when_within_its_cap(
        self, field: str, length: int, char: str
    ) -> None:
        cap = {
            "domain": _LIMITS.flagship_max_domain_chars,
            "subject_id": _LIMITS.flagship_max_subject_id_chars,
        }[field]
        payload = {"data": [[0.0, 1.0]], field: char * length}
        if length <= cap:
            assert len(getattr(FlagshipDetectRequest.model_validate(payload), field)) == length
        else:
            with pytest.raises(ValidationError):
                FlagshipDetectRequest.model_validate(payload)

    def test_the_row_cap_holds_at_the_boundary(self) -> None:
        cap = _LIMITS.flagship_max_rows
        assert len(FlagshipDetectRequest.model_validate({"data": [[0.0]] * cap}).data) == cap
        with pytest.raises(ValidationError):
            FlagshipDetectRequest.model_validate({"data": [[0.0]] * (cap + 1)})


# --------------------------------------------------------------------------- #
# Served routes
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(rate_limit_off: None) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def admitted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[int, ...], Any, Any]]:
    """Stand in for the engine; record what the route let through to it."""
    seen: list[tuple[tuple[int, ...], Any, Any]] = []

    def engine(
        matrix: np.ndarray[Any, Any],
        domain: str | None,
        explain: bool,
        gdpr_report: bool = False,
        subject_id: str | None = None,
        reference_scope: str | None = None,
        disclose_reference: bool = True,
    ) -> dict[str, Any]:
        # An unauthenticated caller shares a reference and is never handed it.
        assert (reference_scope, disclose_reference) == ("anonymous", False)
        seen.append((matrix.shape, domain, subject_id))
        return {"anomaly_prob": 0.1, "is_anomaly": False, "severity": 0.0}

    monkeypatch.setattr(detection, "_run_flagship_detection", engine)
    return seen


_row = st.lists(_floats, max_size=4)
_flagship_body = st.fixed_dictionaries(
    {},
    optional={
        "data": st.one_of(st.lists(_row, max_size=6), _leaves, st.lists(_leaves, max_size=4)),
        "domain": st.one_of(st.text(max_size=2 * _LIMITS.flagship_max_domain_chars), _leaves),
        "subject_id": st.one_of(
            st.text(max_size=2 * _LIMITS.flagship_max_subject_id_chars), _leaves
        ),
        "explain": _leaves,
        "gdpr_report": _leaves,
        "unexpected": _leaves,
    },
)
_series = st.lists(_floats, max_size=40)
_tier_body = st.fixed_dictionaries(
    {"data": st.one_of(_series, _leaves)},
    optional={
        "subset": st.one_of(st.none(), st.lists(st.text(max_size=12), max_size=3)),
        "method": st.one_of(st.none(), st.text(max_size=12)),
        "contamination": _floats,
        "conformal_alpha": st.one_of(st.none(), _floats),
        "reference": st.one_of(st.none(), _series),
        "labels": st.one_of(st.none(), st.lists(st.integers(-2, 3), max_size=40)),
    },
)


class TestServedRoutesUnderArbitraryBodies:
    @settings(max_examples=300, deadline=None, suppress_health_check=_SHARED)
    @given(body=_flagship_body)
    def test_flagship_never_5xx_and_admits_only_capped_requests(
        self,
        client: TestClient,
        admitted: list[tuple[tuple[int, ...], Any, Any]],
        body: dict[str, Any],
    ) -> None:
        before = len(admitted)
        response = client.post("/api/v1/detect/flagship", content=json.dumps(body), headers=_JSON)
        assert response.status_code < 500, response.text[:200]
        for shape, domain, subject_id in admitted[before:]:
            assert len(shape) == 2 and 1 <= shape[0] <= _LIMITS.flagship_max_rows
            assert domain is None or len(domain) <= _LIMITS.flagship_max_domain_chars
            assert subject_id is None or len(subject_id) <= _LIMITS.flagship_max_subject_id_chars

    @settings(max_examples=150, deadline=None, suppress_health_check=_SHARED)
    @given(body=_tier_body)
    def test_the_streaming_tier_never_5xx(self, client: TestClient, body: dict[str, Any]) -> None:
        response = client.post("/api/v1/detect/tier", content=json.dumps(body), headers=_JSON)
        assert response.status_code < 500, response.text[:200]


class TestNonFiniteInputIsA422:
    """Fails on the prior code: each of these was a 500."""

    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/api/v1/detect/tier", '{"data": Infinity}'),
            ("/api/v1/detect/tier", '{"data": NaN}'),
            ("/api/v1/detect/flagship", '{"data": [[1.0]], "domain": NaN}'),
            ("/api/v1/detect/fusion", '{"data": -Infinity}'),
            ("/api/v1/detect/three-r", '{"data": [1, 2, 3], "recursion_depth": Infinity}'),
            ("/api/v1/hazard/visualize", '{"hazard": NaN}'),
        ],
    )
    def test_a_non_finite_number_that_fails_validation_is_refused_as_422(
        self, client: TestClient, path: str, body: str
    ) -> None:
        response = client.post(path, content=body, headers=_JSON)
        assert response.status_code == 422, response.text[:200]
        detail = response.json()["detail"]  # strict JSON: the client parsed it
        # FastAPI's own shape, and the refused value named rather than dropped.
        assert {"type", "loc", "msg"} <= detail[0].keys()
        assert detail[0]["input"] in {"nan", "inf", "-inf"}

    def test_a_finite_validation_error_is_unchanged(self, client: TestClient) -> None:
        response = client.post("/api/v1/detect/tier", content='{"data": "x"}', headers=_JSON)
        assert response.status_code == 422
        assert response.json()["detail"][0]["input"] == "x"
