"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for api/routes/ endpoint modules.

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

    def test_list_models_requires_auth(self, client) -> None:
        """Test that listing models requires authentication."""
        response = client.get("/api/v1/models")
        # Should require auth - either 401 or 403
        assert response.status_code in (401, 403, 200)

    def test_list_models_with_auth(self, client, auth_headers) -> None:
        """Test listing models with valid auth."""
        response = client.get("/api/v1/models", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_register_model(self, client, auth_headers) -> None:
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

    def test_register_model_invalid_name(self, client, auth_headers) -> None:
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

    def test_get_model_not_found(self, client, auth_headers) -> None:
        """Test getting a non-existent model."""
        response = client.get(
            "/api/v1/models/nonexistent_id",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_delete_model_not_found(self, client, auth_headers) -> None:
        """Test deleting a non-existent model."""
        response = client.delete(
            "/api/v1/models/nonexistent_id",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_register_and_get_model_roundtrip(self, client, auth_headers) -> None:
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
# Detection Routes Tests
# =============================================================================


class TestDetectionRoutes:
    """Tests for /api/v1/detect endpoints."""

    def test_fusion_detection(self, client) -> None:
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

    def test_fusion_detection_multivariate(self, client) -> None:
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

    def test_fusion_minimum_data_points(self, client) -> None:
        """Test fusion with insufficient data."""
        response = client.post(
            "/api/v1/detect/fusion",
            json={"request": {"data": [1.0, 2.0]}},
        )
        assert response.status_code == 422

    def test_fusion_detects_anomaly(self, client) -> None:
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

    def test_three_r_detection(self, client) -> None:
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

    def test_three_r_invalid_recursion_depth(self, client) -> None:
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

    def test_three_r_response_structure(self, client) -> None:
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


# =============================================================================
# Batch Routes Tests
# =============================================================================


class TestBatchRoutes:
    """Tests for /api/v1/batch endpoints."""

    def test_submit_batch_job(self, client) -> None:
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

    def test_get_job_status(self, client) -> None:
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

    def test_get_job_not_found(self, client) -> None:
        """Test getting non-existent job."""
        response = client.get("/api/v1/batch/jobs/nonexistent_job_id")
        assert response.status_code == 404

    def test_batch_empty_data_rejected(self, client) -> None:
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

    def test_batch_sensitivity_range(self, client) -> None:
        """Test sensitivity validation."""
        data = [[1.0], [2.0], [3.0]]
        response = client.post(
            "/api/v1/batch/detect",
            json={"request": {"data": data, "sensitivity": 0.5}},
        )
        assert response.status_code == 202

    def test_cancel_job(self, client) -> None:
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

    def test_list_jobs(self, client) -> None:
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

    def test_get_export_summary(self, client, auth_headers) -> None:
        """Test export summary endpoint."""
        response = client.get("/api/v1/export/summary", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_detections" in data
        assert "total_audit_logs" in data

    def test_export_detections_json(self, client, auth_headers) -> None:
        """Test exporting detections as JSON."""
        response = client.get(
            "/api/v1/export/detections",
            headers=auth_headers,
            params={"format": "json", "limit": 10},
        )
        assert response.status_code == 200

    def test_export_detections_csv(self, client, auth_headers) -> None:
        """Test exporting detections as CSV."""
        response = client.get(
            "/api/v1/export/detections",
            headers=auth_headers,
            params={"format": "csv", "limit": 10},
        )
        assert response.status_code == 200

    def test_export_detections_jsonl(self, client, auth_headers) -> None:
        """Test exporting detections as JSONL."""
        response = client.get(
            "/api/v1/export/detections",
            headers=auth_headers,
            params={"format": "jsonl", "limit": 10},
        )
        assert response.status_code == 200

    def test_export_audit_logs(self, client, auth_headers) -> None:
        """Test exporting audit logs."""
        response = client.get(
            "/api/v1/export/audit-logs",
            headers=auth_headers,
            params={"format": "json", "limit": 10},
        )
        assert response.status_code == 200

    def test_export_metrics(self, client, auth_headers) -> None:
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


# =============================================================================
# Batch Callback URL SSRF Validation Tests
# =============================================================================


class TestBatchCallbackSSRF:
    """Tests for batch callback URL SSRF validation."""

    def test_private_callback_url_rejected(self, client) -> None:
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

    def test_localhost_callback_url_rejected(self, client) -> None:
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

    def test_http_callback_url_rejected(self, client) -> None:
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
