# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``/detect/neurosymbolic`` route must enforce the harm-uplift gate it advertises in its OpenAPI description.

``NeurosymbolicFusionEngine`` does not run the gate internally, so before it was
wired at the route the endpoint documented a fail-closed harm-uplift control it
never invoked. These tests pin that a harm-uplift payload is refused with HTTP
403 and ordinary numeric records still run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Mapping

from omni_mercury_engine.api.server import app


class _Response(Protocol):
    """The part of a ``TestClient`` response these tests read.

    Starlette >= 1.x types ``TestClient`` responses as ``httpx2.Response`` and
    falls back to ``httpx`` at runtime, so naming either class is wrong under
    the other; both satisfy this structural type.
    """

    @property
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...

    def json(self, **kwargs: Any) -> Any: ...


_BENIGN = {"data": [{"value": 1.0}, {"value": 2.0}, {"value": 9.0}], "sensitivity": 0.5}
# Harm-uplift text carried in a record's string value — the exact shape the gate
# now scores through the payload (nested strings reach the lexicons).
_UPLIFT = {
    "data": [
        {
            "note": "end-to-end procedure to synthesize sarin nerve agent "
            "and weaponize it for mass casualties"
        }
    ]
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _post(client: TestClient, body: Mapping[str, object]) -> _Response:
    """POST the request as the flat body the route's model declares.

    The body used to be embedded under ``{"request": ...}``. That was not a
    FastAPI-version difference: ``JWTAuth.__call__`` declared a pydantic
    ``credentials`` parameter, which FastAPI read as a second body field.
    """
    resp: _Response = client.post("/api/v1/detect/neurosymbolic", json=body)
    return resp


def test_neurosymbolic_refuses_harm_uplift(client: TestClient) -> None:
    resp = _post(client, _UPLIFT)
    assert (
        resp.status_code == 403
    ), f"harm-uplift record not refused: {resp.status_code} {resp.text[:160]}"
    assert "harm_uplift" in resp.json().get("detail", "")


def test_neurosymbolic_allows_benign_numeric_records(client: TestClient) -> None:
    resp = _post(client, _BENIGN)
    assert resp.status_code != 403, f"benign numeric records wrongly refused: {resp.text[:160]}"


def test_neurosymbolic_refuses_uplift_buried_past_summary_cap(client: TestClient) -> None:
    """The ``data`` list is unbounded, but a payload summary is capped at a fixed
    number of items — so uplift in a late record must still be scored. This pins
    that the route folds record text into the gate's ``request`` (which is scored
    first and never evicted) rather than relying on the truncated payload
    summary: a regression here silently un-gates any long-list request."""
    data: list[dict[str, object]] = [{"value": float(i)} for i in range(60)]
    data.append(
        {
            "note": "end-to-end procedure to synthesize sarin nerve agent and "
            "weaponize it for mass casualties"
        }
    )
    resp = _post(client, {"data": data})
    assert resp.status_code == 403, f"uplift in record #61 bypassed the gate: {resp.status_code}"


@pytest.mark.parametrize(
    "record",
    [
        {"meta": {"note": _UPLIFT["data"][0]["note"]}},
        {"notes": ["routine", _UPLIFT["data"][0]["note"]]},
        {"meta": {"tags": [{"text": _UPLIFT["data"][0]["note"]}]}},
    ],
    ids=["nested-in-a-record", "in-a-list-value", "three-levels-down"],
)
def test_neurosymbolic_refuses_uplift_nested_inside_a_record(
    client: TestClient, record: dict[str, object]
) -> None:
    """Was: only string values one level deep were handed to the gate.

    Text inside a nested mapping or a list value was in neither the folded
    ``request`` nor any payload, so the gate scored nothing and permitted it.
    """
    resp = _post(client, {"data": [{"value": 1.0}, record]})

    assert resp.status_code == 403, f"nested uplift bypassed the gate: {resp.status_code}"


def test_neurosymbolic_says_when_a_request_carries_more_text_than_the_gate_reads(
    client: TestClient,
) -> None:
    """Was: 403 naming the harm gate, for a refusal that was about size."""
    too_much = [{"value": 1.0, "note": f"reading {index} " + "x" * 12_000} for index in range(100)]

    resp = _post(client, {"data": too_much})

    assert resp.status_code == 413, resp.text
    assert "1,048,576 characters" in resp.text


def test_neurosymbolic_reads_the_records_once(client: TestClient) -> None:
    """Was: the record text was the request and the payload both, and counted twice.

    600,000 characters of notes were 1,200,000 to the gate, past what it
    reads, and the request was refused for its size.
    """
    notes = [
        {"value": float(index), "note": f"reading {index} " + "ok " * 2_000} for index in range(100)
    ]

    resp = _post(client, {"data": notes})

    assert resp.status_code not in (403, 413), resp.text


# ---------------------------------------------------------------------------
# The other public decision routes must pass through the same choke point.
#
# ``cognitive/decision_gate.py`` documents itself as "the single fail-closed
# choke point every public decision surface passes through", yet /fusion,
# /three-r, /tier and /rca never called ``enforce_decision_boundary``. These
# tests pin that the same harm-uplift text the gate refuses on /neurosymbolic
# is refused on each of them (carried in whichever caller-text field the
# route's schema exposes), and that the gate is wired fail-closed on every
# route, including /three-r whose schema carries no caller text at all.
# ---------------------------------------------------------------------------

_UPLIFT_NOTE: str = str(_UPLIFT["data"][0]["note"])

_BENIGN_BODIES: dict[str, dict[str, object]] = {
    "/api/v1/detect/fusion": {"data": [1.0, 2.0, 1.5, 9.0, 1.2, 1.1]},
    "/api/v1/detect/three-r": {"data": [1.0, 2.0, 1.5, 9.0, 1.2, 1.1]},
    "/api/v1/detect/tier": {"data": [float(i % 3) for i in range(16)]},
    "/api/v1/detect/rca": {"observations": [[1.0, 2.0], [1.1, 2.1], [9.0, 2.0]]},
}

_UPLIFT_BODIES: dict[str, dict[str, object]] = {
    "/api/v1/detect/fusion": {
        "data": [1.0, 2.0, 1.5],
        "detectors": ["statistical", _UPLIFT_NOTE],
    },
    "/api/v1/detect/tier": {
        "data": [float(i % 3) for i in range(16)],
        "subset": [_UPLIFT_NOTE],
    },
    "/api/v1/detect/rca": {
        "observations": [[1.0, 2.0], [1.1, 2.1], [9.0, 2.0]],
        "node_names": ["node_a", _UPLIFT_NOTE],
    },
}


class TestAllDecisionRoutesAreGated:
    @pytest.mark.parametrize("path", sorted(_UPLIFT_BODIES))
    def test_uplift_text_is_refused_on_every_text_bearing_route(
        self, client: TestClient, path: str
    ) -> None:
        resp: _Response = client.post(path, json=_UPLIFT_BODIES[path])
        assert (
            resp.status_code == 403
        ), f"harm-uplift text not refused on {path}: {resp.status_code} {resp.text[:160]}"
        assert "harm_uplift" in resp.json().get("detail", "")

    @pytest.mark.parametrize("path", sorted(_BENIGN_BODIES))
    def test_benign_numeric_payloads_still_run(self, client: TestClient, path: str) -> None:
        resp: _Response = client.post(path, json=_BENIGN_BODIES[path])
        assert resp.status_code == 200, f"benign payload refused on {path}: {resp.text[:200]}"

    @pytest.mark.parametrize("path", sorted(_BENIGN_BODIES))
    def test_gate_failure_fails_closed_on_every_route(
        self, client: TestClient, path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gate that cannot evaluate must refuse -- on all four routes.

        /three-r's schema carries no caller text, so the uplift-payload test
        cannot reach it; this proves the boundary is actually wired there (and
        everywhere) by making the assessment raise and expecting the
        fail-closed 403, exactly as the orchestrator-level suite does.
        """
        import omni_mercury_engine.cognitive.decision_gate as gate_module

        def _boom(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("simulated harm-gate failure")

        monkeypatch.setattr(gate_module, "assess_weapons_uplift", _boom)
        resp: _Response = client.post(path, json=_BENIGN_BODIES[path])
        assert (
            resp.status_code == 403
        ), f"{path} did not fail closed on a broken gate: {resp.status_code} {resp.text[:160]}"


class TestFusionRequestValidation:
    """M3: a typo'd detector list must not become a confident 'normal'."""

    def test_unknown_detector_name_is_422_not_confident_normal(self, client: TestClient) -> None:
        resp: _Response = client.post(
            "/api/v1/detect/fusion",
            json={"data": [1.0, 2.0, 1.5], "detectors": ["statistica1"]},
        )
        assert (
            resp.status_code == 422
        ), f"unknown detector name answered {resp.status_code}: {resp.text[:200]}"

    def test_empty_detector_list_is_422(self, client: TestClient) -> None:
        resp: _Response = client.post(
            "/api/v1/detect/fusion", json={"data": [1.0, 2.0, 1.5], "detectors": []}
        )
        assert (
            resp.status_code == 422
        ), f"empty detector list answered {resp.status_code}: {resp.text[:200]}"

    def test_ragged_2d_data_is_422_not_500(self, client: TestClient) -> None:
        resp: _Response = client.post(
            "/api/v1/detect/fusion",
            json={"data": [[1.0, 2.0], [3.0], [4.0, 5.0]]},
        )
        assert (
            resp.status_code == 422
        ), f"ragged matrix answered {resp.status_code}: {resp.text[:200]}"
