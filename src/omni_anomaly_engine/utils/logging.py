"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Structured logging module for OMNI AVA.

This module provides a comprehensive logging system with:
- Structured JSON logging for production environments
- Colored console output for development
- Correlation ID tracking for distributed tracing
- Performance metrics logging
- PII redaction capabilities

Example:
    Basic usage::

        from omni_anomaly_engine.utils.logging import get_logger, configure_logging

        # Configure logging for the application
        configure_logging(level="INFO", json_format=True)

        # Get a logger for your module
        logger = get_logger(__name__)

        # Log with structured context
        logger.info("Processing request", request_id="abc123", user_id="user456")

    With correlation IDs::

        from omni_anomaly_engine.utils.logging import correlation_context

        with correlation_context() as corr_id:
            logger.info("Starting operation", correlation_id=corr_id)
            # All logs within this context will have the same correlation_id
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Generator

# Thread-local storage for correlation IDs
_correlation_context = threading.local()

# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# PII patterns to redact (can be extended)
PII_PATTERNS = [
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "auth",
    "credential",
    "ssn",
    "social_security",
    "credit_card",
    "cvv",
]


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    This formatter outputs log records as JSON objects, making them
    easy to parse and analyze in log aggregation systems like ELK,
    Splunk, or CloudWatch.

    Attributes:
        include_timestamp: Whether to include ISO timestamp.
        include_hostname: Whether to include hostname.
        redact_pii: Whether to redact potential PII fields.

    Example:
        >>> formatter = StructuredFormatter(include_hostname=True)
        >>> handler.setFormatter(formatter)
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_hostname: bool = False,
        redact_pii: bool = True,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the structured formatter.

        Args:
            include_timestamp: Include ISO 8601 timestamp.
            include_hostname: Include hostname in output.
            redact_pii: Redact potential PII fields.
            extra_fields: Extra fields to include in all log entries.
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_hostname = include_hostname
        self.redact_pii = redact_pii
        self.extra_fields = extra_fields or {}

        if include_hostname:
            import socket

            self._hostname = socket.gethostname()

    def _redact_value(self, key: str, value: Any) -> Any:
        """Redact potential PII values.

        Args:
            key: The field key.
            value: The field value.

        Returns:
            Original value or "[REDACTED]" if PII detected.
        """
        if not self.redact_pii:
            return value

        key_lower = key.lower()
        for pattern in PII_PATTERNS:
            if pattern in key_lower:
                return "[REDACTED]"
        return value

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_entry: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add timestamp
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.now(UTC).isoformat()

        # Add hostname
        if self.include_hostname:
            log_entry["hostname"] = self._hostname

        # Add correlation ID if present
        corr_id = getattr(_correlation_context, "correlation_id", None)
        if corr_id:
            log_entry["correlation_id"] = corr_id

        # Add location info
        log_entry["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
            ]:
                log_entry[key] = self._redact_value(key, value)

        # Add configured extra fields
        log_entry.update(self.extra_fields)

        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development.

    This formatter adds ANSI color codes to log output for better
    readability during development.

    Attributes:
        COLORS: Mapping of log levels to ANSI color codes.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        """Initialize the colored formatter.

        Args:
            fmt: Log format string.
            datefmt: Date format string.
        """
        super().__init__(fmt or DEFAULT_FORMAT, datefmt or DEFAULT_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors.

        Args:
            record: The log record to format.

        Returns:
            Colored string representation.
        """
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class PerformanceLogger:
    """Logger for performance metrics.

    This class provides utilities for logging performance metrics
    such as execution time, throughput, and latency.

    Example:
        >>> perf_logger = PerformanceLogger("detection")
        >>> with perf_logger.measure("batch_processing"):
        ...     process_batch(data)
        >>> perf_logger.log_metrics()
    """

    def __init__(self, component: str, logger: logging.Logger | None = None) -> None:
        """Initialize performance logger.

        Args:
            component: Component name for metric namespacing.
            logger: Logger instance to use. Creates one if not provided.
        """
        self.component = component
        self.logger = logger or logging.getLogger(f"omni_anomaly_engine.perf.{component}")
        self._metrics: dict[str, list[Any]] = {}

    @contextmanager
    def measure(self, operation: str) -> Generator[Any, None, None]:
        """Context manager to measure operation duration.

        Args:
            operation: Name of the operation being measured.

        Yields:
            None

        Example:
            >>> with perf_logger.measure("feature_extraction"):
            ...     features = extract_features(data)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(duration)

            self.logger.debug(
                f"Operation '{operation}' completed",
                extra={
                    "operation": operation,
                    "duration_ms": duration * 1000,
                    "component": self.component,
                },
            )

    def log_metrics(self) -> dict[str, dict[str, float]]:
        """Log aggregated metrics.

        Returns:
            Dictionary of metric statistics.
        """
        stats = {}
        for operation, durations in self._metrics.items():
            if durations:
                import statistics

                stats[operation] = {
                    "count": len(durations),
                    "total_ms": sum(durations) * 1000,
                    "mean_ms": statistics.mean(durations) * 1000,
                    "min_ms": min(durations) * 1000,
                    "max_ms": max(durations) * 1000,
                }
                if len(durations) > 1:
                    stats[operation]["std_ms"] = statistics.stdev(durations) * 1000

        self.logger.info(
            f"Performance metrics for {self.component}",
            extra={"metrics": stats, "component": self.component},
        )
        return stats

    def reset(self) -> None:
        """Reset collected metrics."""
        self._metrics.clear()


def get_correlation_id() -> str | None:
    """Get the current correlation ID.

    Returns:
        Current correlation ID or None if not set.
    """
    return getattr(_correlation_context, "correlation_id", None)


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set.
    """
    _correlation_context.correlation_id = correlation_id


@contextmanager
def correlation_context(correlation_id: str | None = None) -> Generator[Any, None, None]:
    """Context manager for correlation ID tracking.

    This context manager sets a correlation ID for all log messages
    within the context, enabling distributed tracing.

    Args:
        correlation_id: Specific ID to use, or generates UUID if None.

    Yields:
        The correlation ID being used.

    Example:
        >>> with correlation_context() as corr_id:
        ...     logger.info("Starting request", request_id=request_id)
        ...     result = process_request()
        ...     logger.info("Request completed", result=result)
    """
    old_id = getattr(_correlation_context, "correlation_id", None)
    new_id = correlation_id or str(uuid.uuid4())
    _correlation_context.correlation_id = new_id
    try:
        yield new_id
    finally:
        if old_id:
            _correlation_context.correlation_id = old_id
        else:
            delattr(_correlation_context, "correlation_id")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    This is a convenience function that returns a properly configured
    logger for the OMNI AVA application.

    Args:
        name: Logger name, typically __name__ of the module.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


def configure_logging(
    level: str | int = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
    include_hostname: bool = False,
    redact_pii: bool = True,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Configure logging for the application.

    This function sets up the logging system with the specified
    configuration. It should be called once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Use JSON structured logging format.
        log_file: Optional file path for file logging.
        include_hostname: Include hostname in JSON logs.
        redact_pii: Redact potential PII from logs.
        extra_fields: Extra fields to include in all log entries.

    Example:
        >>> configure_logging(
        ...     level="DEBUG",
        ...     json_format=True,
        ...     log_file="/var/log/omni-ava/app.log"
        ... )
    """
    # Get the root logger for omni_anomaly_engine
    root_logger = logging.getLogger("omni_anomaly_engine")

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter = StructuredFormatter(
            include_hostname=include_hostname,
            redact_pii=redact_pii,
            extra_fields=extra_fields,
        )
    else:
        formatter = ColoredFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            StructuredFormatter(
                include_hostname=include_hostname,
                redact_pii=redact_pii,
                extra_fields=extra_fields,
            )
        )
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False


def log_function_call(logger: logging.Logger | None = None):
    """Decorator to log function entry and exit.

    Args:
        logger: Logger to use. If None, creates one based on function module.

    Returns:
        Decorator function.

    Example:
        >>> @log_function_call()
        ... def process_data(data):
        ...     return transformed_data
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            logger.debug(f"Entering {func_name}", extra={"function": func_name})
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                logger.debug(
                    f"Exiting {func_name}",
                    extra={"function": func_name, "duration_ms": duration * 1000},
                )
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.exception(
                    f"Exception in {func_name}: {e}",
                    extra={"function": func_name, "duration_ms": duration * 1000},
                )
                raise

        return wrapper

    return decorator
