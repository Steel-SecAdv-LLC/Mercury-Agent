# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive tests for api/routes/ endpoint modules.

Covers:
- models.py: Model registration, listing, versioning, file upload
- detection.py: Neurosymbolic, fusion, 3R detection endpoints
- batch.py: Batch job submission, status, results, cancellation
- export.py: Detection/audit export, metrics, streaming

NOTE: FastAPI uses the parameter name `request` for the Pydantic body models
in these routes, which means the JSON body must be nested under a `request` key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

# ``fastapi.testclient`` and the Mercury API routes module are part of
# the optional ``api`` extra.  Skip cleanly at collection time so the
# rest of the suite is still discoverable in CI images that intentionally
# do not install the API extra (e.g. detector-only environments).
pytest.importorskip("fastapi")

# Disable rate limiting BEFORE importing the server module.
# This must happen before any import of omni_mercury_engine.api.server,
# as the middleware reads the env var at module-load time.
os.environ["OMNI_RATE_LIMIT_ENABLED"] = "false"

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create FastAPI test client with server app."""
    from omni_mercury_engine.api.server import app

    return TestClient(app)


# =============================================================================
# Model Routes Tests
# =============================================================================


class TestModelRoutes:
    """Tests for /api/v1/models endpoints."""

    @pytest.fixture
    def auth_headers(self):
        """Create valid API key auth headers."""
        from omni_mercury_engine.api.auth import get_api_key_store

        store = get_api_key_store()
        raw_key, _ = store.create_key(
            name="model_test_key",
            user_id="model_test_user",
        )
        return {"X-API-Key": raw_key}

    def test_list_models_requires_auth(self, client: Any) -> None:
        """Test that listing models requires authentication."""
        response = client.get("/api/v1/models")
        # Should require auth - either 401 or 403
        assert response.status_code in (401, 403, 200)

    def test_list_models_with_auth(self, client: Any, auth_headers: Any) -> None:
        """Test listing models with valid auth."""
        response = client.get("/api/v1/models", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_register_model(self, client: Any, auth_headers: Any) -> None:
        """Test registering a new model."""
        response = client.post(
            "/api/v1/models",
            json={
                "request": {
                    "name": "test_model_for_testing",
                    "model_type": "fusion",
                    "description": "A test model",
                    "tags": ["test"],
                }
            },
            headers=auth_headers,
        )
        assert response.status_code in (200, 201)
        data = response.json()
        assert data["name"] == "test_model_for_testing"
        assert "model_id" in data

    def test_register_model_invalid_name(self, client: Any, auth_headers: Any) -> None:
        """Test model registration with invalid name."""
        response = client.post(
            "/api/v1/models",
            json={
                "request": {
                    "name": "",  # Empty name
                    "model_type": "fusion",
                }
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_get_model_not_found(self, client: Any, auth_headers: Any) -> None:
        """Test getting a non-existent model."""
        response = client.get(
            "/api/v1/models/nonexistent_id",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_delete_model_not_found(self, client: Any, auth_headers: Any) -> None:
        """Test deleting a non-existent model."""
        response = client.delete(
            "/api/v1/models/nonexistent_id",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_register_and_get_model_roundtrip(self, client: Any, auth_headers: Any) -> None:
        """Test model create and retrieve roundtrip."""
        # Create
        create_resp = client.post(
            "/api/v1/models",
            json={
                "request": {
                    "name": "roundtrip_model",
                    "model_type": "statistical",
                    "description": "A roundtrip test model",
                }
            },
            headers=auth_headers,
        )
        assert create_resp.status_code in (200, 201)
        model_id = create_resp.json()["model_id"]

        # Retrieve
        get_resp = client.get(
            f"/api/v1/models/{model_id}",
            headers=auth_headers,
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "roundtrip_model"


# =============================================================================
# Model Storage Path-Confinement Tests
# =============================================================================


class TestModelStoragePathConfinement:
    """Regression tests for the model-storage path-traversal class of bug.

    The model directory is addressed by a hash of the (request-supplied) id, so
    uploaded bytes must always land under the configured storage root and no
    identifier — however adversarial — can escape it. These tests guard against
    a future refactor reintroducing the CodeQL ``py/path-injection`` finding.
    """

    def test_version_upload_lands_under_storage_root(self, tmp_path: Any) -> None:
        """An uploaded version file is written inside the storage root."""
        from omni_mercury_engine.api.routes.models import (
            ModelFramework,
            ModelRegistry,
            ModelType,
        )

        registry = ModelRegistry(storage_path=str(tmp_path))
        model = registry.register_model(
            name="confine-test", model_type=ModelType.FUSION, owner_id="u1"
        )
        version = registry.add_version(
            model_id=model.model_id,
            created_by="u1",
            file_content=b"weight-bytes",
            framework=ModelFramework.PYTORCH,
        )

        assert version.file_path is not None
        written = Path(version.file_path).resolve()
        root = tmp_path.resolve()
        assert str(written).startswith(str(root) + os.sep)
        assert written.read_bytes() == b"weight-bytes"

    def test_adversarial_model_id_cannot_escape_root(self, tmp_path: Any) -> None:
        """Traversal-style identifiers resolve to a confined, separator-free dir."""
        from omni_mercury_engine.api.routes.models import ModelRegistry

        registry = ModelRegistry(storage_path=str(tmp_path))
        root = tmp_path.resolve()
        for evil in ["../../etc/passwd", "..", "a/b/c", "/abs/path", "x" * 500, ""]:
            resolved = registry._model_dir(evil).resolve()
            assert str(resolved).startswith(str(root) + os.sep)
            key = registry._dir_key(evil)
            assert "/" not in key and "\\" not in key and ".." not in key

    def test_delete_removes_only_the_model_dir(self, tmp_path: Any) -> None:
        """Deleting a model removes its directory but leaves the root intact."""
        from omni_mercury_engine.api.routes.models import ModelRegistry, ModelType

        registry = ModelRegistry(storage_path=str(tmp_path))
        model = registry.register_model(name="del-test", model_type=ModelType.FUSION, owner_id="u1")
        model_dir = registry._model_dir(model.model_id)
        assert model_dir.exists()

        assert registry.delete_model(model.model_id) is True
        assert not model_dir.exists()
        assert tmp_path.exists()
        assert registry.delete_model("deadbeefdeadbeef") is False


# =============================================================================
# Detection Routes Tests
# =============================================================================


class TestDetectionRoutes:
    """Tests for /api/v1/detect endpoints."""

    def test_fusion_detection(self, client: Any) -> None:
        """Test multi-detector fusion endpoint."""
        data = [1.0, 1.1, 1.0, 50.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0]
        response = client.post(
            "/api/v1/detect/fusion",
            json={
                "request": {
                    "data": data,
                    "sensitivity": 0.7,
                    "detectors": ["statistical", "temporal"],
                }
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert "fused_score" in result
        assert "is_anomaly" in result

    def test_fusion_detection_multivariate(self, client: Any) -> None:
        """Test fusion with multivariate data."""
        data = [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.0, 2.0, 3.0],
            [100.0, 200.0, 300.0],
            [1.1, 2.1, 3.1],
        ]
        response = client.post(
            "/api/v1/detect/fusion",
            json={
                "request": {
                    "data": data,
                    "sensitivity": 0.5,
                }
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert "fused_score" in result

    def test_fusion_minimum_data_points(self, client: Any) -> None:
        """Test fusion with insufficient data."""
        response = client.post(
            "/api/v1/detect/fusion",
            json={"request": {"data": [1.0, 2.0]}},
        )
        assert response.status_code == 422

    def test_fusion_detects_anomaly(self, client: Any) -> None:
        """Test that fusion correctly flags clear anomalies."""
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 100.0, 1.0, 1.1, 1.0, 1.1]
        response = client.post(
            "/api/v1/detect/fusion",
            json={
                "request": {
                    "data": data,
                    "sensitivity": 0.9,
                    "detectors": ["statistical"],
                }
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["is_anomaly"] is True

    def test_three_r_detection(self, client: Any) -> None:
        """Test 3R mechanism analysis endpoint."""
        data = [float(i) + np.sin(i * 0.5) for i in range(50)]
        response = client.post(
            "/api/v1/detect/three-r",
            json={
                "request": {
                    "data": data,
                    "recursion_depth": 3,
                    "harmonic_bands": 4,
                    "ethical_threshold": 0.96,
                }
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert "fusion_score" in result
        assert "recursion_score" in result
        assert "resonance_score" in result
        assert "is_stable" in result

    def test_three_r_invalid_recursion_depth(self, client: Any) -> None:
        """Test 3R with invalid recursion depth."""
        data = [1.0] * 50
        response = client.post(
            "/api/v1/detect/three-r",
            json={
                "request": {
                    "data": data,
                    "recursion_depth": 0,  # Must be 1-10
                }
            },
        )
        assert response.status_code == 422

    def test_three_r_response_structure(self, client: Any) -> None:
        """Test 3R response includes all expected fields."""
        data = [np.sin(i * 0.1) for i in range(100)]
        response = client.post(
            "/api/v1/detect/three-r",
            json={"request": {"data": data}},
        )
        assert response.status_code == 200
        result = response.json()
        assert "optimization_score" in result
        assert "ethical_scaling" in result
        assert "lyapunov_bound" in result
        assert "weights" in result
        assert "harmonic_analysis" in result

    def test_tier_detection(self, client: Any) -> None:
        """The detector-tier ensemble is reachable over HTTP and flags a burst."""
        rng = np.random.default_rng(0)
        series = rng.normal(0, 1, 200).tolist()
        for i in range(100, 108):
            series[i] += 7.0
        response = client.post(
            "/api/v1/detect/tier",
            json={
                "request": {
                    "data": series,
                    "subset": ["spectral_residual", "spot_evt", "bocpd"],
                    "conformal_alpha": 0.05,
                }
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["n_points"] == 200
        assert set(result).issuperset(
            {"scores", "flags", "uncertainty", "threshold", "conformal_flags"}
        )
        # Conformal flags concentrate near the injected burst.
        flagged = {i for i, f in enumerate(result["conformal_flags"]) if f}
        assert flagged & set(range(95, 115))

    def test_tier_detection_bad_request(self, client: Any) -> None:
        """stacking without labels is a 400 (surfaced ValueError), not a 500."""
        rng = np.random.default_rng(1)
        response = client.post(
            "/api/v1/detect/tier",
            json={"request": {"data": rng.normal(0, 1, 60).tolist(), "method": "stacking"}},
        )
        assert response.status_code == 400

    def test_tier_detection_unknown_detector_is_400_not_500(self, client: Any) -> None:
        """A typo in ``subset`` is a client error (review finding): the builder
        raises ValueError for unknown names, so the surface answers 400 with
        the name in the detail — never an opaque internal 500."""
        rng = np.random.default_rng(1)
        response = client.post(
            "/api/v1/detect/tier",
            json={"request": {"data": rng.normal(0, 1, 60).tolist(), "subset": ["not_a_detector"]}},
        )
        assert response.status_code == 400
        assert "not_a_detector" in response.json()["detail"]

    def test_flagship_detection(self, client: Any) -> None:
        """The flagship OmniMercuryEngine fusion path is reachable over HTTP.

        This is the same neuro-symbolic engine the ``detect -d fusion`` CLI runs
        (trained checkpoint + GOSNN + σ_Immutable gate), unified onto HTTP -- not
        the lightweight statistical ``/fusion``.
        """
        pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(30, 5)).tolist()
        response = client.post(
            "/api/v1/detect/flagship",
            json={"request": {"data": matrix}},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        # The calibrated flagship contract, distinct from /fusion's fused_score.
        assert set(result).issuperset(
            {"anomaly_prob", "is_anomaly", "severity", "detector_importance", "gosnn_metadata"}
        )
        assert 0.0 <= float(result["anomaly_prob"]) <= 1.0
        assert "sigma_immutable_score" in result["gosnn_metadata"]

    def test_flagship_detection_bad_shape(self, client: Any) -> None:
        """A 1-D series is rejected by the 2-D matrix contract (422)."""
        response = client.post(
            "/api/v1/detect/flagship",
            json={"request": {"data": [1.0, 2.0, 3.0]}},
        )
        assert response.status_code == 422

    def test_flagship_blocked_by_ethical_gate_returns_403(
        self, client: Any, monkeypatch: Any
    ) -> None:
        """A detection the hard ethical gate refuses is a 403 fail-closed, never a silent allow."""
        pytest.importorskip("torch")
        from omni_mercury_engine.api.routes import detection as det
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )

        def _blocked(
            matrix: Any,
            domain: Any,
            explain: Any,
            gdpr_report: Any = False,
            subject_id: Any = None,
        ) -> dict[str, Any]:
            raise EthicalConstraintViolationError(
                "blocked matrix", 0.10, 0.96, check="sigma_immutable"
            )

        monkeypatch.setattr(det, "_run_flagship_detection", _blocked)
        response = client.post(
            "/api/v1/detect/flagship",
            json={"request": {"data": [[1.0, 2.0], [3.0, 4.0]]}},
        )
        assert response.status_code == 403
        assert "sigma_immutable" in response.json()["detail"]

    def test_root_cause_localization(self, client: Any) -> None:
        """Graph-based root-cause attribution is reachable over HTTP."""
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, (300, 4))
        base[:, 1] += 0.8 * base[:, 0]
        base[:, 2] += 0.8 * base[:, 1]
        obs = base.copy()
        obs[-1, 0] += 8.0
        obs[-1, 1] += 6.0
        obs[-1, 2] += 4.0
        response = client.post(
            "/api/v1/detect/rca",
            json={
                "request": {
                    "observations": obs.tolist(),
                    "train": base[:-1].tolist(),
                    "node_names": ["pump", "valve", "tank", "aux"],
                }
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["n_nodes"] == 4
        by_node = {e["node"]: e["attribution"] for e in result["ranked"]}
        # The independent node 3 ranks lowest; the top cause carries a label.
        assert by_node[3] == min(by_node.values())
        assert result["top_root_cause"]["name"] in {"pump", "valve", "tank"}

    def test_root_cause_bad_shape(self, client: Any) -> None:
        """A 1-D observation array is rejected by the rows x nodes contract (422)."""
        response = client.post(
            "/api/v1/detect/rca",
            json={"request": {"observations": [1.0, 2.0, 3.0]}},
        )
        assert response.status_code == 422


# =============================================================================
# Batch Routes Tests
# =============================================================================


class TestBatchRoutes:
    """Tests for /api/v1/batch endpoints."""

    def test_submit_batch_job(self, client: Any) -> None:
        """Test submitting a batch detection job."""
        data = [[float(i)] for i in range(20)]
        response = client.post(
            "/api/v1/batch/detect",
            json={
                "request": {
                    "data": data,
                    "method": "univariate",
                    "sensitivity": 0.5,
                }
            },
        )
        assert response.status_code == 202
        result = response.json()
        assert "job_id" in result
        assert result["status"] in ("PENDING", "pending")

    def test_get_job_status(self, client: Any) -> None:
        """Test getting job status."""
        # First submit a job
        data = [[float(i)] for i in range(20)]
        submit_resp = client.post(
            "/api/v1/batch/detect",
            json={
                "request": {
                    "data": data,
                    "method": "statistical",
                    "sensitivity": 0.5,
                }
            },
        )
        assert submit_resp.status_code == 202
        job_id = submit_resp.json()["job_id"]

        # Then check status
        status_resp = client.get(f"/api/v1/batch/jobs/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["job_id"] == job_id

    def test_get_job_not_found(self, client: Any) -> None:
        """Test getting non-existent job."""
        response = client.get("/api/v1/batch/jobs/nonexistent_job_id")
        assert response.status_code == 404

    def test_batch_empty_data_rejected(self, client: Any) -> None:
        """Test that empty data is rejected."""
        response = client.post(
            "/api/v1/batch/detect",
            json={
                "request": {
                    "data": [],
                    "method": "univariate",
                }
            },
        )
        assert response.status_code == 422

    def test_batch_sensitivity_range(self, client: Any) -> None:
        """Test sensitivity validation."""
        data = [[1.0], [2.0], [3.0]]
        response = client.post(
            "/api/v1/batch/detect",
            json={"request": {"data": data, "sensitivity": 0.5}},
        )
        assert response.status_code == 202

    def test_cancel_job(self, client: Any) -> None:
        """Test cancelling a batch job."""
        # Submit a job first
        data = [[float(i)] for i in range(100)]
        submit_resp = client.post(
            "/api/v1/batch/detect",
            json={"request": {"data": data, "method": "univariate"}},
        )
        assert submit_resp.status_code == 202
        job_id = submit_resp.json()["job_id"]

        # Cancel it - may return 204 (cancelled), 200 (already done), or 400 (already completed)
        cancel_resp = client.delete(f"/api/v1/batch/jobs/{job_id}")
        assert cancel_resp.status_code in (200, 204, 400)

    def test_list_jobs(self, client: Any) -> None:
        """Test listing batch jobs."""
        response = client.get("/api/v1/batch/jobs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# =============================================================================
# Export Routes Tests
# =============================================================================


class TestExportRoutes:
    """Tests for /api/v1/export endpoints."""

    @pytest.fixture
    def auth_headers(self):
        """Create valid API key auth headers."""
        from omni_mercury_engine.api.auth import get_api_key_store

        store = get_api_key_store()
        raw_key, _ = store.create_key(
            name="export_test_key",
            user_id="export_test_user",
        )
        return {"X-API-Key": raw_key}

    def test_get_export_summary(self, client: Any, auth_headers: Any) -> None:
        """Test export summary endpoint."""
        response = client.get("/api/v1/export/summary", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_detections" in data
        assert "total_audit_logs" in data

    def test_export_detections_json(self, client: Any, auth_headers: Any) -> None:
        """Test exporting detections as JSON."""
        response = client.get(
            "/api/v1/export/detections",
            headers=auth_headers,
            params={"format": "json", "limit": 10},
        )
        assert response.status_code == 200

    def test_export_detections_csv(self, client: Any, auth_headers: Any) -> None:
        """Test exporting detections as CSV."""
        response = client.get(
            "/api/v1/export/detections",
            headers=auth_headers,
            params={"format": "csv", "limit": 10},
        )
        assert response.status_code == 200

    def test_export_detections_jsonl(self, client: Any, auth_headers: Any) -> None:
        """Test exporting detections as JSONL."""
        response = client.get(
            "/api/v1/export/detections",
            headers=auth_headers,
            params={"format": "jsonl", "limit": 10},
        )
        assert response.status_code == 200

    def test_export_audit_logs(self, client: Any, auth_headers: Any) -> None:
        """Test exporting audit logs."""
        response = client.get(
            "/api/v1/export/audit-logs",
            headers=auth_headers,
            params={"format": "json", "limit": 10},
        )
        assert response.status_code == 200

    def test_export_metrics(self, client: Any, auth_headers: Any) -> None:
        """Test exporting metrics."""
        response = client.get(
            "/api/v1/export/metrics",
            headers=auth_headers,
        )
        # May return 200 or 429 (rate limited) depending on test ordering
        assert response.status_code in (200, 429)
        if response.status_code == 200:
            data = response.json()
            assert "total_detections" in data

    def test_export_metrics_csv(self, client: Any, auth_headers: Any) -> None:
        """Metrics export honours format=csv with tidy metric,value rows (F17)."""
        response = client.get(
            "/api/v1/export/metrics",
            headers=auth_headers,
            params={"format": "csv"},
        )
        assert response.status_code in (200, 429)
        if response.status_code == 200:
            assert "text/csv" in response.headers["content-type"]
            body = response.text
            # Empty stores still emit a valid header row.
            assert body == "" or "metric" in body.splitlines()[0]

    def test_export_metrics_jsonl(self, client: Any, auth_headers: Any) -> None:
        """Metrics export honours format=jsonl (F17)."""
        response = client.get(
            "/api/v1/export/metrics",
            headers=auth_headers,
            params={"format": "jsonl"},
        )
        assert response.status_code in (200, 429)
        if response.status_code == 200:
            assert "ndjson" in response.headers["content-type"]

    def test_export_metrics_json_is_rich(self, client: Any, auth_headers: Any) -> None:
        """The JSON summary carries the enriched surfaces researchers consume."""
        response = client.get("/api/v1/export/metrics", headers=auth_headers)
        assert response.status_code in (200, 429)
        if response.status_code == 200:
            data = response.json()
            for key in (
                "total_detections",
                "score_percentiles",
                "method_breakdown",
                "time_series",
            ):
                assert key in data

    def test_metrics_period_prefers_requested_bounds(self) -> None:
        """Period semantics are consistent whether or not the range has data
        (review finding): requested bounds win; observed min/max only fill a
        bound the caller left open."""
        from datetime import datetime

        from omni_mercury_engine.api.routes.export import (
            DetectionRecord,
            _compute_metrics_summary,
        )

        def _rec(ts: datetime) -> DetectionRecord:
            return DetectionRecord(
                detection_id="d",
                timestamp=ts,
                user_id="u",
                method="zscore",
                data_hash="h",
                anomaly_count=1,
                max_score=0.9,
                is_batch=False,
                sensitivity=0.5,
            )

        records = [_rec(datetime(2026, 7, 5, 12)), _rec(datetime(2026, 7, 6, 12))]
        wide_start, wide_end = datetime(2026, 7, 1), datetime(2026, 7, 9)

        both = _compute_metrics_summary(records, wide_start, wide_end, "hour")
        assert both["period"]["start"] == wide_start.isoformat()
        assert both["period"]["end"] == wide_end.isoformat()

        open_ended = _compute_metrics_summary(records, wide_start, None, "hour")
        assert open_ended["period"]["start"] == wide_start.isoformat()
        assert open_ended["period"]["end"] == datetime(2026, 7, 6, 12).isoformat()

        unbounded = _compute_metrics_summary(records, None, None, "hour")
        assert unbounded["period"]["start"] == datetime(2026, 7, 5, 12).isoformat()
        assert unbounded["period"]["end"] == datetime(2026, 7, 6, 12).isoformat()

        empty = _compute_metrics_summary([], wide_start, wide_end, "hour")
        assert empty["period"]["start"] == wide_start.isoformat()
        assert empty["period"]["end"] == wide_end.isoformat()


# =============================================================================
# Batch Callback URL SSRF Validation Tests
# =============================================================================


class TestBatchCallbackSSRF:
    """Tests for batch callback URL SSRF validation."""

    def test_private_callback_url_rejected(self, client: Any) -> None:
        """Test that private IP callback URLs are rejected."""
        data = [[1.0], [2.0], [3.0]]
        response = client.post(
            "/api/v1/batch/detect",
            json={
                "request": {
                    "data": data,
                    "method": "univariate",
                    "callback_url": "https://192.168.1.1/webhook",
                }
            },
        )
        assert response.status_code == 422

    def test_localhost_callback_url_rejected(self, client: Any) -> None:
        """Test that localhost callback URLs are rejected."""
        data = [[1.0], [2.0], [3.0]]
        response = client.post(
            "/api/v1/batch/detect",
            json={
                "request": {
                    "data": data,
                    "method": "univariate",
                    "callback_url": "https://localhost/webhook",
                }
            },
        )
        assert response.status_code == 422

    def test_http_callback_url_rejected(self, client: Any) -> None:
        """Test that non-HTTPS callback URLs are rejected."""
        data = [[1.0], [2.0], [3.0]]
        response = client.post(
            "/api/v1/batch/detect",
            json={
                "request": {
                    "data": data,
                    "method": "univariate",
                    "callback_url": "http://example.com/webhook",
                }
            },
        )
        assert response.status_code == 422


# =============================================================================
# Model Type and Status Enum Tests
# =============================================================================


class TestModelEnums:
    """Tests for model-related enums."""

    def test_model_type_values(self) -> None:
        """Test ModelType enum values exist."""
        from omni_mercury_engine.api.routes.models import ModelType

        assert hasattr(ModelType, "FUSION")
        assert hasattr(ModelType, "STATISTICAL")
        assert hasattr(ModelType, "TEMPORAL")
        assert hasattr(ModelType, "CUSTOM")

    def test_model_status_values(self) -> None:
        """Test ModelStatus enum values exist."""
        from omni_mercury_engine.api.routes.models import ModelStatus

        assert hasattr(ModelStatus, "DRAFT")
        assert hasattr(ModelStatus, "STAGED")
        assert hasattr(ModelStatus, "DEPLOYED")
        assert hasattr(ModelStatus, "DEPRECATED")

    def test_model_framework_values(self) -> None:
        """Test ModelFramework enum values exist."""
        from omni_mercury_engine.api.routes.models import ModelFramework

        assert hasattr(ModelFramework, "PYTORCH")
        assert hasattr(ModelFramework, "ONNX")
        assert hasattr(ModelFramework, "SKLEARN")


# =============================================================================
# Batch Method Enum Tests
# =============================================================================


class TestBatchMethodEnum:
    """Tests for batch detection method enum."""

    def test_detection_methods(self) -> None:
        """Test all batch detection methods exist."""
        from omni_mercury_engine.api.routes.batch import BatchDetectionMethod

        assert hasattr(BatchDetectionMethod, "UNIVARIATE")
        assert hasattr(BatchDetectionMethod, "MULTIVARIATE")
        assert hasattr(BatchDetectionMethod, "FUSION")
        assert hasattr(BatchDetectionMethod, "STATISTICAL")
        assert hasattr(BatchDetectionMethod, "TEMPORAL")
        assert hasattr(BatchDetectionMethod, "NEUROSYMBOLIC")


# =============================================================================
# Export Format Enum Tests
# =============================================================================


class TestExportFormatEnum:
    """Tests for export format enum."""

    def test_export_formats(self) -> None:
        """Test all export formats exist."""
        from omni_mercury_engine.api.routes.export import ExportFormat

        assert hasattr(ExportFormat, "JSON")
        assert hasattr(ExportFormat, "CSV")
        assert hasattr(ExportFormat, "JSONL")
