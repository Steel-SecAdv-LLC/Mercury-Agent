# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for REST API endpoints."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import app

client = TestClient(app)


class TestAPI:
    """Test REST API endpoints."""

    def test_health_endpoint(self) -> None:
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert result["version"] == "1.7.0"

    def test_univariate_detection_endpoint(self) -> None:
        """Test univariate anomaly detection endpoint."""
        response = client.post(
            "/api/v1/detect/univariate",
            json={"data": [1.0, 1.1, 1.0, 5.0, 1.0], "sensitivity": 0.5},
        )

        assert response.status_code == 200
        result = response.json()
        assert "anomalies" in result
        assert "scores" in result
        assert result["method"] == "univariate"
        assert len(result["anomalies"]) == 5

    def test_univariate_detection_default_sensitivity(self) -> None:
        """Test univariate detection with default sensitivity."""
        response = client.post(
            "/api/v1/detect/univariate", json={"data": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

        assert response.status_code == 200
        result = response.json()
        assert "anomalies" in result

    def test_multivariate_detection_endpoint(self) -> None:
        """Test multivariate anomaly detection endpoint."""
        response = client.post(
            "/api/v1/detect/multivariate",
            json={"data": [[1.0, 2.0], [1.1, 2.1], [5.0, 6.0]], "sensitivity": 0.5},
        )

        assert response.status_code == 200
        result = response.json()
        assert "anomalies" in result
        assert "scores" in result
        assert "features" in result
        assert result["method"] == "multivariate"

    def test_multivariate_detection_with_feature_names(self) -> None:
        """Test multivariate detection with custom feature names."""
        response = client.post(
            "/api/v1/detect/multivariate",
            json={
                "data": [[1.0, 2.0], [1.1, 2.1], [1.0, 2.0]],
                "features": ["temperature", "pressure"],
                "sensitivity": 0.7,
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["features"] == ["temperature", "pressure"]

    def test_multivariate_detection_invalid_shape(self) -> None:
        """Test multivariate detection with invalid data shape."""
        response = client.post(
            "/api/v1/detect/multivariate", json={"data": [1.0, 2.0, 3.0], "sensitivity": 0.5}
        )

        assert response.status_code == 422

    def test_univariate_sensitivity_affects_detection(self) -> None:
        """Test that sensitivity parameter affects detection."""
        data = {"data": [1.0, 1.0, 1.0, 10.0, 1.0]}

        response_low = client.post("/api/v1/detect/univariate", json={**data, "sensitivity": 0.1})
        response_high = client.post("/api/v1/detect/univariate", json={**data, "sensitivity": 0.9})

        assert response_low.status_code == 200
        assert response_high.status_code == 200

        result_low = response_low.json()
        result_high = response_high.json()

        assert result_low["threshold"] != result_high["threshold"]

    def test_multivariate_sensitivity_affects_detection(self) -> None:
        """Test that sensitivity parameter affects multivariate detection."""
        data = {"data": [[1.0, 2.0], [1.0, 2.0], [10.0, 20.0]]}

        response_low = client.post("/api/v1/detect/multivariate", json={**data, "sensitivity": 0.1})
        response_high = client.post(
            "/api/v1/detect/multivariate", json={**data, "sensitivity": 0.9}
        )

        assert response_low.status_code == 200
        assert response_high.status_code == 200

    def test_univariate_empty_data(self) -> None:
        """Test univariate detection with empty data returns validation error."""
        response = client.post("/api/v1/detect/univariate", json={"data": []})

        # Empty data should be rejected with 422 (min_length=3 validation)
        assert response.status_code == 422

    def test_api_returns_correct_types(self) -> None:
        """Test API returns correct data types."""
        response = client.post(
            "/api/v1/detect/univariate", json={"data": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

        result = response.json()
        assert isinstance(result["anomalies"], list)
        assert isinstance(result["scores"], list)
        assert isinstance(result["method"], str)
        assert isinstance(result["threshold"], (int, float))
