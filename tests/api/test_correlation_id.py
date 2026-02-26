"""
Mercury Agent - Correlation ID Middleware Tests

Tests for the request correlation ID tracking middleware.

Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GPL-3.0-or-later
"""

from __future__ import annotations

import uuid

import pytest


class TestCorrelationIDMiddleware:
    """Tests for CorrelationIDMiddleware."""

    def test_correlation_id_context_var_exists(self):
        """Test that correlation_id context variable is defined."""
        from omni_mercury_engine.api.server import correlation_id_ctx

        assert correlation_id_ctx is not None
        # Default should be empty string
        assert correlation_id_ctx.get() == ""

    def test_correlation_id_context_var_set_get(self):
        """Test setting and getting correlation ID."""
        from omni_mercury_engine.api.server import correlation_id_ctx

        test_id = str(uuid.uuid4())
        token = correlation_id_ctx.set(test_id)

        try:
            assert correlation_id_ctx.get() == test_id
        finally:
            correlation_id_ctx.reset(token)

    def test_middleware_class_exists(self):
        """Test that CorrelationIDMiddleware class exists."""
        from omni_mercury_engine.api.server import CorrelationIDMiddleware

        assert CorrelationIDMiddleware is not None
        assert hasattr(CorrelationIDMiddleware, "dispatch")
        assert hasattr(CorrelationIDMiddleware, "HEADER_NAME")
        assert hasattr(CorrelationIDMiddleware, "HEADER_ALIAS")

    def test_middleware_header_constants(self):
        """Test header name constants."""
        from omni_mercury_engine.api.server import CorrelationIDMiddleware

        assert CorrelationIDMiddleware.HEADER_NAME == "X-Correlation-ID"
        assert CorrelationIDMiddleware.HEADER_ALIAS == "X-Request-ID"


class TestCorrelationIDIntegration:
    """Integration tests for correlation ID with FastAPI."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient

            from omni_mercury_engine.api.server import app

            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI test client not available")

    def test_health_endpoint_returns_correlation_id(self, client):
        """Test that health endpoint returns correlation ID header."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        # Should be a valid UUID
        correlation_id = response.headers["X-Correlation-ID"]
        uuid.UUID(correlation_id)  # Raises if invalid

    def test_request_duration_header(self, client):
        """Test that request duration header is present."""
        response = client.get("/health")

        assert "X-Request-Duration-Ms" in response.headers
        duration = float(response.headers["X-Request-Duration-Ms"])
        assert duration >= 0

    def test_custom_correlation_id_preserved(self, client):
        """Test that custom correlation ID is preserved."""
        custom_id = str(uuid.uuid4())
        response = client.get("/health", headers={"X-Correlation-ID": custom_id})

        assert response.headers["X-Correlation-ID"] == custom_id

    def test_request_id_alias(self, client):
        """Test that X-Request-ID alias works."""
        custom_id = str(uuid.uuid4())
        response = client.get("/health", headers={"X-Request-ID": custom_id})

        assert response.headers["X-Correlation-ID"] == custom_id

    def test_correlation_id_priority(self, client):
        """Test that X-Correlation-ID takes priority over X-Request-ID."""
        correlation_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        response = client.get(
            "/health",
            headers={
                "X-Correlation-ID": correlation_id,
                "X-Request-ID": request_id,
            },
        )

        # X-Correlation-ID should take priority
        assert response.headers["X-Correlation-ID"] == correlation_id


class TestCorrelationIDFormat:
    """Tests for correlation ID format validation."""

    def test_generated_id_is_valid_uuid(self):
        """Test that generated IDs are valid UUIDs."""
        try:
            from fastapi.testclient import TestClient

            from omni_mercury_engine.api.server import app

            client = TestClient(app)
            response = client.get("/health")

            correlation_id = response.headers.get("X-Correlation-ID")
            assert correlation_id is not None

            # Should be a valid UUID4
            parsed = uuid.UUID(correlation_id)
            assert parsed.version == 4
        except ImportError:
            pytest.skip("FastAPI test client not available")

    def test_custom_id_formats_accepted(self):
        """Test that various ID formats are accepted."""
        try:
            from fastapi.testclient import TestClient

            from omni_mercury_engine.api.server import app

            client = TestClient(app)

            # Test various formats
            test_ids = [
                str(uuid.uuid4()),  # UUID
                "request-12345",  # Custom format
                "abc123",  # Short ID
                "trace-id-with-dashes",  # Dashed
            ]

            for test_id in test_ids:
                response = client.get("/health", headers={"X-Correlation-ID": test_id})
                assert response.headers["X-Correlation-ID"] == test_id
        except ImportError:
            pytest.skip("FastAPI test client not available")
