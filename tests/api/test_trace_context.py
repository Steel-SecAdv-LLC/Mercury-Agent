# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""W3C Trace Context propagation on the API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omni_mercury_engine.api.trace_context import (
    continue_or_start,
    parse_traceparent,
    parse_tracestate,
)

_VALID = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


@pytest.fixture
def client() -> TestClient:
    from omni_mercury_engine.api.server import app

    return TestClient(app)


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  # forbidden version
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",  # zero trace-id
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",  # zero parent-id
        "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",  # upper-case hex
        _VALID + "-extra",  # version 00 has exactly four fields
    ],
)
def test_invalid_traceparent_is_rejected(header: str | None) -> None:
    assert parse_traceparent(header) is None


def test_future_version_is_parsed_by_its_first_fields() -> None:
    parsed = parse_traceparent("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-future")
    assert parsed == ("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", "01")


def test_continue_keeps_trace_and_mints_span() -> None:
    ctx = continue_or_start(_VALID, "vendor=abc")
    assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert ctx.parent_id == "00f067aa0ba902b7"
    assert ctx.span_id != ctx.parent_id and len(ctx.span_id) == 16
    assert ctx.tracestate == "vendor=abc"
    fresh = continue_or_start("garbage", "vendor=abc")
    assert fresh.parent_id is None and fresh.tracestate is None


def test_response_continues_incoming_trace(client: TestClient) -> None:
    response = client.get("/health", headers={"traceparent": _VALID, "tracestate": "k=v"})
    version, trace_id, span_id, flags = response.headers["traceparent"].split("-")
    assert (version, trace_id, flags) == ("00", "4bf92f3577b34da6a3ce929d0e0e4736", "01")
    assert span_id != "00f067aa0ba902b7"
    assert response.headers["tracestate"] == "k=v"


def test_response_starts_a_trace_without_header(client: TestClient) -> None:
    response = client.get("/health")
    parsed = parse_traceparent(response.headers["traceparent"])
    assert parsed is not None
    assert "tracestate" not in response.headers


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("k=v", "k=v"),
        ("rojo=00f067aa0ba902b7, congo=t61rcWkgMzE", "rojo=00f067aa0ba902b7,congo=t61rcWkgMzE"),
        ("tenant1@vendor=x ,\t,b=y", "tenant1@vendor=x,b=y"),  # OWS and empty members
        ("a-_*/0=has spaces inside", "a-_*/0=has spaces inside"),
    ],
)
def test_valid_tracestate_is_normalised(header: str, expected: str) -> None:
    assert parse_tracestate(header) == expected


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        " , ",
        "Upper=v",  # keys are lower-case
        "k=v,k=w",  # duplicate key
        "k=",  # empty value
        "k=del\x7f",  # DEL is outside the value alphabet
        "k=a=b",  # '=' not allowed in a value
        "k=café",  # non-ASCII
        "k=v\r\nInjected: 1",  # header injection attempt
        "noequals",
        "@vendor=x",  # empty tenant
        ",".join(f"k{i}=v" for i in range(33)),  # > 32 members
        "k=" + "v" * 600,  # > 512 characters
    ],
)
def test_invalid_tracestate_is_dropped(header: str | None) -> None:
    assert parse_tracestate(header) is None


def test_thirty_two_members_are_kept() -> None:
    header = ",".join(f"k{i}=v" for i in range(32))
    assert parse_tracestate(header) == header


def test_invalid_tracestate_is_not_echoed(client: TestClient) -> None:
    response = client.get(
        "/health", headers={"traceparent": _VALID, "tracestate": "k=v," + "x" * 600}
    )
    assert response.headers["traceparent"].split("-")[1] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert "tracestate" not in response.headers
    assert continue_or_start(_VALID, "BAD=1").tracestate is None


def test_multiple_tracestate_headers_are_combined(client: TestClient) -> None:
    response = client.get(
        "/health",
        headers=[("traceparent", _VALID), ("tracestate", "a=1"), ("tracestate", "b=2")],
    )
    assert response.headers["tracestate"] == "a=1,b=2"
