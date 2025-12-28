"""
Tests for omni_mercury_engine.utils.logging module.

Tests structured logging, correlation IDs, and performance logging.
"""

from __future__ import annotations

import json
import logging
import time

import pytest

from omni_mercury_engine.utils.logging import (
    ColoredFormatter,
    PerformanceLogger,
    StructuredFormatter,
    configure_logging,
    correlation_context,
    get_correlation_id,
    get_logger,
    log_function_call,
    set_correlation_id,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter class."""

    def test_basic_formatting(self):
        """Test basic JSON formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data
        assert "location" in data
        assert data["location"]["file"] == "test.py"
        assert data["location"]["line"] == 10

    def test_include_hostname(self):
        """Test including hostname in output."""
        formatter = StructuredFormatter(include_hostname=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "hostname" in data
        assert isinstance(data["hostname"], str)

    def test_exclude_timestamp(self):
        """Test excluding timestamp from output."""
        formatter = StructuredFormatter(include_timestamp=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" not in data

    def test_pii_redaction(self):
        """Test that PII fields are redacted."""
        formatter = StructuredFormatter(redact_pii=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.password = "secret123"
        record.api_key = "key123"
        record.user_data = "normal_data"

        result = formatter.format(record)
        data = json.loads(result)

        assert data["password"] == "[REDACTED]"
        assert data["api_key"] == "[REDACTED]"
        assert data["user_data"] == "normal_data"

    def test_pii_redaction_disabled(self):
        """Test that PII fields are not redacted when disabled."""
        formatter = StructuredFormatter(redact_pii=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.password = "secret123"

        result = formatter.format(record)
        data = json.loads(result)

        assert data["password"] == "secret123"

    def test_extra_fields(self):
        """Test adding extra fields to all log entries."""
        formatter = StructuredFormatter(
            extra_fields={"service": "test-service", "version": "1.0.0"}
        )
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["service"] == "test-service"
        assert data["version"] == "1.0.0"

    def test_exception_info(self):
        """Test that exception info is included."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "Test error" in data["exception"]

    def test_correlation_id_included(self):
        """Test that correlation ID is included when set."""
        formatter = StructuredFormatter()

        with correlation_context("test-corr-id"):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            data = json.loads(result)

            assert data["correlation_id"] == "test-corr-id"


class TestColoredFormatter:
    """Tests for ColoredFormatter class."""

    def test_basic_formatting(self):
        """Test basic colored formatting."""
        formatter = ColoredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should contain ANSI color codes
        assert "\033[" in result
        assert "Test message" in result

    def test_color_codes_by_level(self):
        """Test that different levels get different colors."""
        formatter = ColoredFormatter()

        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

        results = []
        for level in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )
            results.append(formatter.format(record))

        # Each should have different color prefix
        # Extract color codes and verify they differ
        assert len(set(results)) == len(results)


class TestPerformanceLogger:
    """Tests for PerformanceLogger class."""

    def test_initialization(self):
        """Test performance logger initialization."""
        perf = PerformanceLogger("test_component")
        assert perf.component == "test_component"
        assert perf._metrics == {}

    def test_measure_context_manager(self):
        """Test measuring operation duration."""
        perf = PerformanceLogger("test")

        with perf.measure("test_operation"):
            time.sleep(0.01)

        assert "test_operation" in perf._metrics
        assert len(perf._metrics["test_operation"]) == 1
        assert perf._metrics["test_operation"][0] >= 0.01

    def test_multiple_measurements(self):
        """Test multiple measurements of same operation."""
        perf = PerformanceLogger("test")

        for _ in range(5):
            with perf.measure("test_op"):
                time.sleep(0.001)

        assert len(perf._metrics["test_op"]) == 5

    def test_log_metrics(self):
        """Test aggregated metrics logging."""
        perf = PerformanceLogger("test")

        for _ in range(10):
            with perf.measure("test_op"):
                time.sleep(0.001)

        stats = perf.log_metrics()

        assert "test_op" in stats
        assert stats["test_op"]["count"] == 10
        assert "total_ms" in stats["test_op"]
        assert "mean_ms" in stats["test_op"]
        assert "min_ms" in stats["test_op"]
        assert "max_ms" in stats["test_op"]
        assert "std_ms" in stats["test_op"]  # Multiple samples

    def test_reset(self):
        """Test resetting metrics."""
        perf = PerformanceLogger("test")

        with perf.measure("test_op"):
            pass

        perf.reset()
        assert perf._metrics == {}


class TestCorrelationContext:
    """Tests for correlation ID management."""

    def test_get_correlation_id_none(self):
        """Test getting correlation ID when not set."""
        assert get_correlation_id() is None

    def test_set_correlation_id(self):
        """Test setting correlation ID."""
        set_correlation_id("test-id")
        assert get_correlation_id() == "test-id"
        # Clean up
        from omni_mercury_engine.utils.logging import _correlation_context

        delattr(_correlation_context, "correlation_id")

    def test_correlation_context_explicit_id(self):
        """Test context manager with explicit ID."""
        with correlation_context("my-corr-id") as corr_id:
            assert corr_id == "my-corr-id"
            assert get_correlation_id() == "my-corr-id"

        assert get_correlation_id() is None

    def test_correlation_context_auto_id(self):
        """Test context manager with auto-generated ID."""
        with correlation_context() as corr_id:
            assert corr_id is not None
            assert len(corr_id) == 36  # UUID format
            assert get_correlation_id() == corr_id

        assert get_correlation_id() is None

    def test_nested_correlation_contexts(self):
        """Test nested correlation contexts."""
        with correlation_context("outer"):
            assert get_correlation_id() == "outer"
            with correlation_context("inner"):
                assert get_correlation_id() == "inner"
            assert get_correlation_id() == "outer"

        assert get_correlation_id() is None


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        logger = logging.getLogger("omni_mercury_engine")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    def test_configure_with_defaults(self):
        """Test configuration with default settings."""
        configure_logging()

        logger = logging.getLogger("omni_mercury_engine")
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_configure_debug_level(self):
        """Test configuration with DEBUG level."""
        configure_logging(level="DEBUG")

        logger = logging.getLogger("omni_mercury_engine")
        assert logger.level == logging.DEBUG

    def test_configure_json_format(self):
        """Test configuration with JSON format."""
        configure_logging(json_format=True)

        logger = logging.getLogger("omni_mercury_engine")
        assert isinstance(logger.handlers[0].formatter, StructuredFormatter)

    def test_configure_colored_format(self):
        """Test configuration with colored format."""
        configure_logging(json_format=False)

        logger = logging.getLogger("omni_mercury_engine")
        assert isinstance(logger.handlers[0].formatter, ColoredFormatter)

    def test_configure_integer_level(self):
        """Test configuration with integer level."""
        configure_logging(level=logging.WARNING)

        logger = logging.getLogger("omni_mercury_engine")
        assert logger.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test getting a logger by name."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_instance(self):
        """Test that same name returns same logger."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2


class TestLogFunctionCall:
    """Tests for log_function_call decorator."""

    def setup_method(self):
        """Set up test logging."""
        configure_logging(level="DEBUG")

    def test_decorator_logs_entry_exit(self, caplog):
        """Test that decorator logs entry and exit."""

        @log_function_call()
        def sample_function(x, y):
            return x + y

        with caplog.at_level(logging.DEBUG):
            result = sample_function(1, 2)

        assert result == 3
        # Check that logs were created (entries contain function name)
        log_text = caplog.text.lower()
        assert "sample_function" in log_text or "entering" in log_text

    def test_decorator_logs_exception(self, caplog):
        """Test that decorator logs exceptions."""

        @log_function_call()
        def failing_function():
            raise ValueError("Test error")

        with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
            failing_function()

    def test_decorator_with_custom_logger(self, caplog):
        """Test decorator with custom logger."""
        custom_logger = logging.getLogger("custom")

        @log_function_call(logger=custom_logger)
        def sample_function():
            return True

        with caplog.at_level(logging.DEBUG, logger="custom"):
            result = sample_function()

        assert result is True


class TestPIIPatterns:
    """Tests for PII pattern detection."""

    @pytest.mark.parametrize(
        "key,expected_redacted",
        [
            ("password", True),
            ("Password", True),
            ("user_password", True),
            ("secret", True),
            ("api_secret", True),
            ("token", True),
            ("auth_token", True),
            ("api_key", True),
            ("apikey", True),
            ("auth", True),
            ("credential", True),
            ("ssn", True),
            ("social_security", True),
            ("credit_card", True),
            ("cvv", True),
            ("username", False),
            ("email", False),
            ("data", False),
            ("name", False),
        ],
    )
    def test_pii_pattern_matching(self, key, expected_redacted):
        """Test that PII patterns are correctly identified."""
        formatter = StructuredFormatter(redact_pii=True)
        result = formatter._redact_value(key, "test_value")

        if expected_redacted:
            assert result == "[REDACTED]"
        else:
            assert result == "test_value"
