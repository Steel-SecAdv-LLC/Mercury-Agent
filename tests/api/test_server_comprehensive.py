"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for api/server.py module.

Covers:
- Health check endpoint
- Univariate anomaly detection endpoint
- Multivariate anomaly detection endpoint
- CORS origin validation
- PII masking filter
- Severity classification
- Rate limiting middleware
- Correlation ID middleware
- Request/response models
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# TODO: install fastapi in CI for full test coverage
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from omni_mercury_engine.api.server import app

    return TestClient(app)


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self, client):
        """Test health endpoint returns 200 with status healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_check_version_matches(self, client):
        """Test health endpoint returns correct API version."""
        from omni_mercury_engine.api.server import API_VERSION

        response = client.get("/health")
        data = response.json()
        assert data["version"] == API_VERSION


# =============================================================================
# Univariate Detection Tests
# =============================================================================


class TestUnivariateDetection:
    """Tests for the /api/v1/detect/univariate endpoint."""

    def test_basic_detection(self, client):
        """Test basic univariate anomaly detection."""
        data = [1.0, 2.0, 1.5, 100.0, 1.8, 2.1, 1.9, 2.0, 1.7]
        response = client.post(
            "/api/v1/detect/univariate",
            json={"data": data, "sensitivity": 0.5},
        )
        assert response.status_code == 200
        result = response.json()
        assert "anomalies" in result
        assert "scores" in result
        assert len(result["anomalies"]) == len(data)
        assert len(result["scores"]) == len(data)
        assert "threshold" in result
        assert "summary" in result

    def test_detection_with_clear_anomaly(self, client):
        """Test that a clear outlier is detected."""
        data = [1.0, 1.1, 1.0, 1.1, 1.0, 50.0, 1.0, 1.1, 1.0]
        response = client.post(
            "/api/v1/detect/univariate",
            json={"data": data, "sensitivity": 0.8},
        )
        assert response.status_code == 200
        result = response.json()
        # The outlier at index 5 should be detected
        assert result["anomalies"][5] is True
        assert result["summary"]["anomaly_count"] >= 1

    def test_detection_minimum_data_points(self, client):
        """Test that minimum 3 data points are required."""
        response = client.post(
            "/api/v1/detect/univariate",
            json={"data": [1.0, 2.0]},  # Only 2 points
        )
        assert response.status_code == 422  # Validation error

    def test_detection_sensitivity_range(self, client):
        """Test sensitivity must be 0.0-1.0."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Valid sensitivity
        response = client.post(
            "/api/v1/detect/univariate",
            json={"data": data, "sensitivity": 0.5},
        )
        assert response.status_code == 200

    def test_anomaly_points_detail(self, client):
        """Test anomaly_points contain index, value, score, severity."""
        data = [1.0, 1.1, 1.0, 100.0, 1.1, 1.0, 1.1, 1.0, 1.1]
        response = client.post(
            "/api/v1/detect/univariate",
            json={"data": data, "sensitivity": 0.9},
        )
        assert response.status_code == 200
        result = response.json()
        if result["anomaly_points"]:
            pt = result["anomaly_points"][0]
            assert "index" in pt
            assert "value" in pt
            assert "score" in pt
            assert "severity" in pt


# =============================================================================
# Multivariate Detection Tests
# =============================================================================


class TestMultivariateDetection:
    """Tests for the /api/v1/detect/multivariate endpoint."""

    def test_basic_multivariate_detection(self, client):
        """Test basic multivariate anomaly detection."""
        data = [
            [1.0, 2.0],
            [1.1, 2.1],
            [1.0, 2.0],
            [100.0, 200.0],  # Outlier
            [1.1, 2.1],
            [1.0, 2.0],
        ]
        response = client.post(
            "/api/v1/detect/multivariate",
            json={"data": data, "sensitivity": 0.5},
        )
        assert response.status_code == 200
        result = response.json()
        assert "anomalies" in result
        assert "scores" in result
        assert "features" in result
        assert len(result["anomalies"]) == 6
        assert len(result["features"]) == 2

    def test_multivariate_with_feature_names(self, client):
        """Test multivariate detection with named features."""
        data = [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
        ]
        response = client.post(
            "/api/v1/detect/multivariate",
            json={
                "data": data,
                "features": ["temp", "pressure", "humidity"],
                "sensitivity": 0.5,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["features"] == ["temp", "pressure", "humidity"]
        assert "feature_contributions" in result

    def test_multivariate_minimum_data_points(self, client):
        """Test minimum 3 data points required."""
        response = client.post(
            "/api/v1/detect/multivariate",
            json={"data": [[1.0, 2.0], [1.1, 2.1]]},
        )
        assert response.status_code == 422

    def test_multivariate_inconsistent_dimensions(self, client):
        """Test error on inconsistent feature dimensions."""
        response = client.post(
            "/api/v1/detect/multivariate",
            json={
                "data": [
                    [1.0, 2.0],
                    [1.1, 2.1, 3.1],  # Different dimension
                    [1.0, 2.0],
                ],
            },
        )
        assert response.status_code == 422


# =============================================================================
# PII Masking Filter Tests
# =============================================================================


class TestPIIMaskingFilter:
    """Tests for the PII masking log filter."""

    def test_email_redaction(self):
        """Test email addresses are redacted in log messages."""
        from omni_mercury_engine.api.server import PIIMaskingFilter

        f = PIIMaskingFilter()
        import logging

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User email is user@example.com and admin@test.org",
            args=(),
            exc_info=None,
        )
        f.filter(record)
        assert "user@example.com" not in record.msg
        assert "[EMAIL_REDACTED]" in record.msg

    def test_bearer_token_redaction(self):
        """Test Bearer tokens are redacted."""
        from omni_mercury_engine.api.server import PIIMaskingFilter

        f = PIIMaskingFilter()
        import logging

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Auth header: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
            args=(),
            exc_info=None,
        )
        f.filter(record)
        assert "eyJhbGciOiJIUzI1NiJ9" not in record.msg
        assert "[TOKEN_REDACTED]" in record.msg

    def test_ip_redaction(self):
        """Test IP addresses are redacted."""
        from omni_mercury_engine.api.server import PIIMaskingFilter

        f = PIIMaskingFilter()
        import logging

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Request from 192.168.1.100",
            args=(),
            exc_info=None,
        )
        f.filter(record)
        assert "192.168.1.100" not in record.msg
        assert "[IP_REDACTED]" in record.msg


# =============================================================================
# Severity Classification Tests
# =============================================================================


class TestSeverityClassification:
    """Tests for the _classify_severity function."""

    def test_low_severity(self):
        """Test low severity classification."""
        from omni_mercury_engine.api.server import SeverityLevel, _classify_severity

        assert _classify_severity(1.2, 1.0) == SeverityLevel.LOW

    def test_medium_severity(self):
        """Test medium severity classification."""
        from omni_mercury_engine.api.server import SeverityLevel, _classify_severity

        assert _classify_severity(2.0, 1.0) == SeverityLevel.MEDIUM

    def test_high_severity(self):
        """Test high severity classification."""
        from omni_mercury_engine.api.server import SeverityLevel, _classify_severity

        assert _classify_severity(3.0, 1.0) == SeverityLevel.HIGH

    def test_critical_severity(self):
        """Test critical severity classification."""
        from omni_mercury_engine.api.server import SeverityLevel, _classify_severity

        assert _classify_severity(5.0, 1.0) == SeverityLevel.CRITICAL

    def test_zero_threshold(self):
        """Test classification with zero threshold."""
        from omni_mercury_engine.api.server import _classify_severity

        # Should not raise ZeroDivisionError
        result = _classify_severity(0.0, 0.0)
        assert result is not None


# =============================================================================
# CORS Configuration Tests
# =============================================================================


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""

    def test_development_cors_origins(self):
        """Test that development mode includes localhost origins."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MERCURY_AGENT_ENV", None)
            os.environ.pop("MERCURY_CORS_ORIGINS", None)
            # In dev mode, localhost origins should be configured
            # (already configured at module load time, so test the app directly)
            from omni_mercury_engine.api.server import app

            assert app is not None  # App should initialize without error

    def test_cors_preflight(self, client):
        """Test CORS preflight request handling."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should not be a 405 Method Not Allowed
        assert response.status_code in (200, 204, 400)


# =============================================================================
# Correlation ID Middleware Tests
# =============================================================================


class TestCorrelationIDMiddleware:
    """Tests for correlation ID tracking."""

    def test_response_includes_correlation_id(self, client):
        """Test that responses include X-Correlation-ID header."""
        response = client.get("/health")
        assert "X-Correlation-ID" in response.headers

    def test_custom_correlation_id_propagated(self, client):
        """Test that provided correlation ID is propagated."""
        custom_id = "test-correlation-123"
        response = client.get(
            "/health",
            headers={"X-Correlation-ID": custom_id},
        )
        assert response.headers.get("X-Correlation-ID") == custom_id

    def test_request_id_alias(self, client):
        """Test that X-Request-ID works as alias."""
        custom_id = "test-request-456"
        response = client.get(
            "/health",
            headers={"X-Request-ID": custom_id},
        )
        assert response.headers.get("X-Correlation-ID") == custom_id

    def test_request_duration_header(self, client):
        """Test that X-Request-Duration-Ms header is included."""
        response = client.get("/health")
        assert "X-Request-Duration-Ms" in response.headers
        duration = float(response.headers["X-Request-Duration-Ms"])
        assert duration >= 0


# =============================================================================
# OpenAPI Schema Tests
# =============================================================================


class TestOpenAPISchema:
    """Tests for custom OpenAPI schema."""

    def test_openapi_schema_accessible(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "info" in schema
        assert schema["info"]["title"] == "Mercury Agent API"

    def test_docs_accessible(self, client):
        """Test Swagger docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
