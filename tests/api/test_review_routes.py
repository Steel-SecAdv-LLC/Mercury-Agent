# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``/api/v1/review``: upload a document, an image or a recording, get a decision.

The served path, end to end on the real application: each committed sample
uploaded as ``multipart/form-data`` and as base64 JSON comes back as a
flagship-shaped result with a valid ``DecisionRecord``; every bound the
flagship route enforces is enforced here through the same objects (the
per-surface body cap, the rate limiter, the admission bound, the engine
lock); and a refusal by an ethical gate is a 403 carrying a blocked record,
with nothing measured or scored.

No harmful text appears here: a refusal comes from a stand-in gate.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi", reason="fastapi (optional '[api]' extra) is required")
pytest.importorskip("torch")
pytest.importorskip("PIL")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api.routes import (
    detection,
    review as review_routes,
)
from omni_mercury_engine.api.server import BodySizeLimitMiddleware, RateLimitMiddleware, app
from omni_mercury_engine.cognitive import decision_gate
from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
from omni_mercury_engine.decision import DecisionRecord
from omni_mercury_engine.serving_limits import load_serving_limits, serving_limits
from tests.review import builders

if TYPE_CHECKING:
    from collections.abc import MutableMapping

SAMPLES = Path(__file__).resolve().parents[2] / "examples" / "data" / "review"
SAMPLE = {
    "document": SAMPLES / "sample_document.txt",
    "image": SAMPLES / "sample_image.png",
    "audio": SAMPLES / "sample_recording.wav",
}
KINDS = tuple(SAMPLE)
UPLOAD, INLINE = "/api/v1/review/upload", "/api/v1/review/inline"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """One client over the real application for the module."""
    return TestClient(app)


@pytest.fixture(scope="module")
def uploaded(client: TestClient) -> dict[str, dict[str, Any]]:
    """Each committed sample, uploaded once as multipart."""
    results = {}
    for kind, path in SAMPLE.items():
        response = client.post(UPLOAD, files={"file": (path.name, path.read_bytes())})
        assert response.status_code == 200, response.text[:400]
        results[kind] = response.json()
    return results


def _inline(data: bytes, name: str = "", **extra: Any) -> dict[str, Any]:
    return {"filename": name, "content_base64": base64.b64encode(data).decode(), **extra}


class TestAnUploadBecomesADecision:
    @pytest.mark.parametrize("kind", KINDS)
    def test_each_kind_returns_a_valid_decision_record(
        self, uploaded: dict[str, dict[str, Any]], kind: str
    ) -> None:
        body = uploaded[kind]
        record = DecisionRecord.from_dict(body["decision"])
        assert record.calibrated is True and record.coverage == pytest.approx(0.9)
        assert body["conformal"]["regime"] == "batch_max"
        assert body["drift_detection"]["reference_source"] == "calibration_batches:batch_family"
        assert body["review"]["kind"] == kind
        assert body["review"]["filename"] == SAMPLE[kind].name
        assert list(body["review"]["answers"]) == ["why_this_segment", "how_sure", "the_record"]
        assert body["decision"]["signals"]["model_provenance"]["checkpoint"] == f"review_{kind}.pt"
        assert body["decision"]["signals"]["ethical_gate_passed"] is True

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_json_route_gives_the_same_decision_as_the_multipart_one(
        self, client: TestClient, uploaded: dict[str, dict[str, Any]], kind: str
    ) -> None:
        path = SAMPLE[kind]
        response = client.post(INLINE, json=_inline(path.read_bytes(), path.name))
        assert response.status_code == 200, response.text[:400]
        body = response.json()
        assert body["decision"] == uploaded[kind]["decision"]
        assert body["review"] == uploaded[kind]["review"]

    def test_the_upload_is_recognised_by_its_bytes_not_its_name(
        self, client: TestClient, uploaded: dict[str, dict[str, Any]]
    ) -> None:
        response = client.post(
            UPLOAD, files={"file": ("holiday.txt", SAMPLE["image"].read_bytes())}
        )
        assert response.status_code == 200
        assert response.json()["review"]["kind"] == "image"
        assert response.json()["decision"] == uploaded["image"]["decision"]

    def test_the_filename_is_reduced_to_its_base_name(self, client: TestClient) -> None:
        data = SAMPLE["document"].read_bytes()
        response = client.post(INLINE, json=_inline(data, "../../etc/notes.txt"))
        assert response.status_code == 200
        assert response.json()["review"]["filename"] == "notes.txt"

    def test_every_review_decision_is_on_the_served_trail(
        self, client: TestClient, uploaded: dict[str, dict[str, Any]]
    ) -> None:
        described = client.get("/api/v1/review").json()
        assert described["schema"] == "mercury.review/v1"
        assert described["decisions"]["recorded"] is True
        assert described["decisions"]["total"] >= len(uploaded)
        assert described["decisions"]["calibrated_rate"] == pytest.approx(1.0)

    def test_the_description_serves_the_enforced_bounds(self, client: TestClient) -> None:
        described = client.get("/api/v1/review").json()
        served = serving_limits()
        assert set(described["formats"]) == set(KINDS)
        assert {name: limit["value"] for name, limit in described["limits"].items()} == {
            "review_max_body_bytes": served.review_max_body_bytes,
            "review_max_text_chars": served.review_max_text_chars,
            "review_max_pixels": served.review_max_pixels,
            "review_max_audio_seconds": served.review_max_audio_seconds,
            "review_max_concurrent": served.review_max_concurrent,
        }
        assert described["admission"]["max_concurrent"] == served.review_max_concurrent == 2

    def test_a_review_is_recorded_in_the_detection_history(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded: list[dict[str, Any]] = []

        async def keep(**kwargs: Any) -> None:
            recorded.append(kwargs)

        monkeypatch.setattr(review_routes, "record_detection", keep)
        path = SAMPLE["audio"]
        body = client.post(UPLOAD, files={"file": (path.name, path.read_bytes())}).json()
        assert len(recorded) == 1
        assert recorded[0]["method"] == "review_audio"
        # The upload itself is not kept: its digest identifies it.
        assert recorded[0]["data"] == [body["review"]["sha256"]]
        assert recorded[0]["results"]["summary"]["segments"] == body["review"]["segments"]


class TestRefusalsSayWhy:
    def test_compressed_audio_is_415_with_the_conversion(self, client: TestClient) -> None:
        response = client.post(UPLOAD, files={"file": ("voice.mp3", b"ID3\x04\x00" + b"\x00" * 64)})
        assert response.status_code == 415
        assert "ffmpeg" in response.json()["detail"]

    def test_an_artifact_too_short_to_be_its_own_reference_is_422(self, client: TestClient) -> None:
        response = client.post(INLINE, json=_inline(builders.prose(60).encode(), "short.txt"))
        assert response.status_code == 422
        assert "at least 12" in response.json()["detail"]

    def test_a_kind_that_does_not_match_is_422(self, client: TestClient) -> None:
        data = SAMPLE["document"].read_bytes()
        response = client.post(UPLOAD, files={"file": ("a.txt", data)}, data={"kind": "image"})
        assert response.status_code == 422
        assert "document" in response.json()["detail"]

    def test_an_unknown_kind_is_a_validation_error(self, client: TestClient) -> None:
        response = client.post(
            UPLOAD, files={"file": ("a.txt", b"some words")}, data={"kind": "video"}
        )
        assert response.status_code == 422

    def test_invalid_base64_and_an_empty_upload_are_422(self, client: TestClient) -> None:
        bad = client.post(INLINE, json={"filename": "a.txt", "content_base64": "not base64 !!"})
        assert bad.status_code == 422 and "base64" in bad.json()["detail"]
        assert client.post(UPLOAD, files={"file": ("a.txt", b"")}).status_code == 422

    @pytest.mark.parametrize(
        ("bound", "data", "name"),
        [
            ("review_max_text_chars", builders.prose(900).encode(), "long.txt"),
            ("review_max_pixels", builders.png(builders.texture(128, 128)), "big.png"),
            ("review_max_audio_seconds", builders.wav(builders.tone(8.0)), "long.wav"),
        ],
    )
    def test_past_a_decode_bound_is_413_naming_the_bound(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        bound: str,
        data: bytes,
        name: str,
    ) -> None:
        """The route enforces the served ``review_max_*`` limits, not the library defaults."""
        monkeypatch.setenv(f"MERCURY_{bound.upper()}", "5")
        monkeypatch.setattr(review_routes, "_LIMITS", load_serving_limits())
        response = client.post(UPLOAD, files={"file": (name, data)})
        assert response.status_code == 413
        assert "5" in response.json()["detail"]


class TestTheBodyCapIsTheReviewSurfacesOwn:
    def test_the_review_surface_has_its_own_served_cap(self) -> None:
        async def never(scope: Any, receive: Any, send: Any) -> None:  # pragma: no cover
            raise AssertionError("not called")

        middleware = BodySizeLimitMiddleware(never)
        limits = serving_limits()
        assert middleware._limit_for(UPLOAD) == limits.review_max_body_bytes
        assert middleware._limit_for(INLINE) == limits.review_max_body_bytes
        assert middleware._limit_for("/api/v1/detect/flagship") == limits.max_body_bytes
        assert limits.max_body_bytes < limits.review_max_body_bytes == 32 * 1024 * 1024

    def test_a_declared_length_over_the_cap_is_refused_before_the_body(
        self, client: TestClient
    ) -> None:
        over = serving_limits().review_max_body_bytes + 1
        response = client.post(
            INLINE,
            content=b"{}",
            headers={"content-type": "application/json", "content-length": str(over)},
        )
        assert response.status_code == 413

    @pytest.mark.parametrize("path", [UPLOAD, INLINE])
    def test_the_cap_is_enforced_as_the_body_streams(self, path: str) -> None:
        """A chunked upload declares no length; it is counted as it arrives and cut off."""

        async def reading(scope: Any, receive: Any, send: Any) -> None:
            more = True
            while more:
                message = await receive()
                more = message.get("more_body", False)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = BodySizeLimitMiddleware(reading)
        middleware.review_limit = 1000
        sent: list[MutableMapping[str, Any]] = []
        queue = [b"x" * 600, b"x" * 600]

        async def receive() -> dict[str, Any]:
            body = queue.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(queue)}

        async def send(message: MutableMapping[str, Any]) -> None:
            sent.append(message)

        asyncio.run(middleware({"type": "http", "path": path, "headers": []}, receive, send))
        assert sent[0]["status"] == 413


class TestTheRateLimiterMetersUploads:
    def test_uploads_spend_the_same_token_bucket_as_every_api_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMNI_RATE_LIMIT_ENABLED", "true")
        limited = FastAPI()
        limited.include_router(review_routes.router)
        limited.add_middleware(
            RateLimitMiddleware,  # type: ignore[arg-type, unused-ignore]
            requests_per_minute=6,
            burst_size=3,
        )
        with TestClient(limited) as limited_client:
            statuses = [
                limited_client.post(UPLOAD, files={"file": ("a.mp3", b"ID3\x04" + b"\x00" * 32)})
                for _ in range(5)
            ]
        assert [response.status_code for response in statuses[:3]] == [415, 415, 415]
        assert [response.status_code for response in statuses[3:]] == [429, 429]
        assert int(statuses[3].headers["Retry-After"]) >= 1

    def test_the_review_surface_is_not_among_the_limiters_exemptions(self) -> None:
        import inspect

        source = inspect.getsource(RateLimitMiddleware.dispatch)
        assert "/api/v1/review" not in source and "review" not in source


class TestTheAdmissionBoundAndTheEngineLockAreTheFlagships:
    def test_a_review_holds_one_slot_of_the_flagship_admission_bound(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        full = threading.BoundedSemaphore(1)
        assert full.acquire(blocking=False)
        monkeypatch.setattr(detection, "_flagship_admission", full)
        before = detection._admission_stats.snapshot()["rejected_total"]
        path = SAMPLE["document"]
        response = client.post(UPLOAD, files={"file": (path.name, path.read_bytes())})
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "1"
        assert detection._admission_stats.snapshot()["rejected_total"] == before + 1

    def test_reviews_in_flight_have_a_bound_of_their_own(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Measuring an upload is where a review's memory goes: at capacity, 429, nothing run."""
        full = threading.BoundedSemaphore(1)
        assert full.acquire(blocking=False)
        monkeypatch.setattr(review_routes, "_review_slots", full)
        prepared: list[int] = []
        from omni_mercury_engine import review

        monkeypatch.setattr(review, "prepare", lambda *_a, **_k: prepared.append(1))
        admitted = detection._admission_stats.snapshot()["admitted_total"]
        before = client.get("/api/v1/review").json()["admission"]["rejected_total"]
        path = SAMPLE["image"]
        response = client.post(UPLOAD, files={"file": (path.name, path.read_bytes())})
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "2"
        assert "at capacity" in response.json()["detail"]
        assert prepared == []
        # Refused before it asked for a flagship slot, and counted where an operator looks.
        assert detection._admission_stats.snapshot()["admitted_total"] == admitted
        assert client.get("/api/v1/review").json()["admission"]["rejected_total"] == before + 1

    def test_a_review_slot_is_released_whatever_the_outcome(self, client: TestClient) -> None:
        for _ in range(serving_limits().review_max_concurrent + 1):
            refused = client.post(UPLOAD, files={"file": ("a.mp3", b"ID3\x04" + b"\x00" * 32)})
            assert refused.status_code == 415
        path = SAMPLE["audio"]
        assert (
            client.post(UPLOAD, files={"file": (path.name, path.read_bytes())}).status_code == 200
        )

    def test_the_slot_is_released_whatever_the_outcome(self, client: TestClient) -> None:
        in_flight = detection._admission_stats.snapshot()["in_flight"]
        path = SAMPLE["image"]
        assert (
            client.post(UPLOAD, files={"file": (path.name, path.read_bytes())}).status_code == 200
        )
        assert (
            client.post(UPLOAD, files={"file": ("a.mp3", b"ID3\x04" + b"\x00" * 32)}).status_code
            == 415
        )
        assert detection._admission_stats.snapshot()["in_flight"] == in_flight

    def test_scoring_runs_under_the_flagship_engine_lock(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every fusion engine shares one process-global: a review must not score beside a detection."""
        held: list[bool] = []
        from omni_mercury_engine import review

        scored = review.review_prepared

        def watching(prepared: Any, engine: Any) -> dict[str, Any]:
            held.append(detection._flagship_lock.locked())
            result: dict[str, Any] = scored(prepared, engine)
            return result

        monkeypatch.setattr(review, "review_prepared", watching)
        path = SAMPLE["audio"]
        assert (
            client.post(UPLOAD, files={"file": (path.name, path.read_bytes())}).status_code == 200
        )
        assert held == [True]
        assert detection._flagship_lock.locked() is False

    def test_decoding_does_not_hold_the_engine_lock(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A long decode must not stall the flagship route."""
        held: list[bool] = []
        from omni_mercury_engine import review

        prepare = review.prepare

        def watching(*args: Any, **kwargs: Any) -> Any:
            held.append(detection._flagship_lock.locked())
            return prepare(*args, **kwargs)

        monkeypatch.setattr(review, "prepare", watching)
        path = SAMPLE["document"]
        assert (
            client.post(UPLOAD, files={"file": (path.name, path.read_bytes())}).status_code == 200
        )
        assert held == [False]


class TestAnEthicalRefusalIsA403WithABlockedRecord:
    @staticmethod
    def _refuse(details: dict[str, Any]) -> Any:
        def refuses(*_args: Any, **_kwargs: Any) -> Any:
            raise EthicalConstraintViolationError(
                action="review", score=0.0, threshold=1.0, check="harm_uplift", details=details
            )

        return refuses

    @pytest.mark.parametrize("path", [UPLOAD, INLINE])
    def test_the_harm_gate_refusing_stops_the_review_before_any_engine_runs(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        engines: list[str] = []
        monkeypatch.setattr(
            decision_gate, "enforce_decision_boundary", self._refuse({"stand_in": True})
        )
        monkeypatch.setattr(review_routes, "_served_review_engine", engines.append)
        data = SAMPLE["document"].read_bytes()
        response = (
            client.post(path, files={"file": ("a.txt", data)})
            if path == UPLOAD
            else client.post(path, json=_inline(data, "a.txt"))
        )
        assert response.status_code == 403
        detail = response.json()["detail"]
        assert detail["ethics"]["check"] == "harm_uplift"
        blocked = DecisionRecord.from_dict(detail["decision"])
        assert blocked.abstained is True and blocked.response.fail_closed is True
        assert engines == []

    def test_more_text_than_the_gate_reads_is_413_not_a_guess(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        too_much = {"reason": decision_gate.UNASSESSABLE, "error": "more text than the gate reads"}
        monkeypatch.setattr(decision_gate, "enforce_decision_boundary", self._refuse(too_much))
        data = SAMPLE["document"].read_bytes()
        response = client.post(UPLOAD, files={"file": ("a.txt", data)})
        assert response.status_code == 413
        assert "harm-uplift" in response.json()["detail"]

    def test_an_engine_gate_refusing_is_a_403_too(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine import review

        def refuses(*_args: Any, **_kwargs: Any) -> Any:
            raise EthicalConstraintViolationError(
                action="review", score=0.0, threshold=1.0, check="sigma_immutable", details={}
            )

        monkeypatch.setattr(review, "review_prepared", refuses)
        path = SAMPLE["image"]
        response = client.post(UPLOAD, files={"file": (path.name, path.read_bytes())})
        assert response.status_code == 403
        assert response.json()["detail"]["ethics"]["check"] == "sigma_immutable"

    def test_an_internal_failure_is_a_bounded_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine import review

        def breaks(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("/srv/secret/path/model.pt could not be read")

        monkeypatch.setattr(review, "review_prepared", breaks)
        path = SAMPLE["image"]
        response = client.post(UPLOAD, files={"file": (path.name, path.read_bytes())})
        assert response.status_code == 500
        assert "secret" not in response.text


class TestTheRoutesAreDocumented:
    def test_openapi_describes_both_ways_in_and_the_refusals(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert "get" in paths["/api/v1/review"]
        for path in (UPLOAD, INLINE):
            described = paths[path]["post"]["description"]
            for phrase in ("batch-level", "harm-uplift gate", "413", "415", "422", "403", "429"):
                assert phrase in described, (path, phrase)
        assert "multipart/form-data" in paths[UPLOAD]["post"]["requestBody"]["content"]

    def test_the_serving_limits_endpoint_lists_the_review_bounds(self, client: TestClient) -> None:
        limits = client.get("/api/v1/detect/flagship/limits").json()["limits"]
        assert limits["review_max_body_bytes"]["env"] == "MERCURY_REVIEW_MAX_BODY_BYTES"
        assert limits["review_max_text_chars"]["unit"] == "characters"
        assert "artifact reviews" in limits["flagship_max_queue"]["description"]
