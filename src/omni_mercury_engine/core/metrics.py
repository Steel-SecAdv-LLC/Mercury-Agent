# Copyright (c) 2024-2025 Steel-SecAdv-LLC
# SPDX-License-Identifier: MIT
"""
Prometheus metrics for Mercury-Agent anomaly detection system.

This module provides centralized metric definitions that match the
prometheus-rules.yaml expectations. Metrics gracefully degrade to
no-ops when prometheus_client is not available.

Metrics emitted:
- omni_detection_requests_total: Counter for detection requests by detector type
- omni_detection_duration_seconds: Histogram for detection duration by detector type
- omni_detection_success_total: Counter for successful detections by detector type
- omni_model_inference_errors_total: Counter for model inference errors
- omni_fusion_inference_duration_seconds: Histogram for fusion layer inference time
- omni_detector_extraction_duration_seconds: Histogram for feature extraction time
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to no-op implementations
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrics enabled")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not available - metrics will be no-ops")


class NoOpMetric:
    """No-op metric implementation when prometheus_client is not available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def labels(self, *args: Any, **kwargs: Any) -> NoOpMetric:
        return self

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, amount: float) -> None:
        pass

    def time(self) -> NoOpContextManager:
        return NoOpContextManager()


class NoOpContextManager:
    """No-op context manager for timing."""

    def __enter__(self) -> NoOpContextManager:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _create_counter(name: str, description: str, labelnames: list[str]) -> Any:
    """Create a Counter metric or no-op if prometheus_client unavailable."""
    if PROMETHEUS_AVAILABLE:
        return Counter(name, description, labelnames)
    return NoOpMetric()


def _create_histogram(
    name: str, description: str, labelnames: list[str], buckets: tuple[float, ...] | None = None
) -> Any:
    """Create a Histogram metric or no-op if prometheus_client unavailable."""
    if PROMETHEUS_AVAILABLE:
        if buckets:
            return Histogram(name, description, labelnames, buckets=buckets)
        return Histogram(name, description, labelnames)
    return NoOpMetric()


def _create_gauge(name: str, description: str, labelnames: list[str]) -> Any:
    """Create a Gauge metric or no-op if prometheus_client unavailable."""
    if PROMETHEUS_AVAILABLE:
        return Gauge(name, description, labelnames)
    return NoOpMetric()


# Detection request metrics (matches prometheus-rules.yaml expectations)
DETECTION_REQUESTS = _create_counter(
    "omni_detection_requests_total",
    "Total number of detection requests",
    ["detector_type"],
)

DETECTION_DURATION = _create_histogram(
    "omni_detection_duration_seconds",
    "Detection processing duration in seconds",
    ["detector_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

DETECTION_SUCCESS = _create_counter(
    "omni_detection_success_total",
    "Total number of successful detections",
    ["detector_type"],
)

# Model inference metrics
MODEL_INFERENCE_ERRORS = _create_counter(
    "omni_model_inference_errors_total",
    "Total number of model inference errors",
    ["model_name"],
)

# Fusion layer metrics
FUSION_INFERENCE_DURATION = _create_histogram(
    "omni_fusion_inference_duration_seconds",
    "Fusion layer inference duration in seconds",
    ["fusion_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Feature extraction metrics
DETECTOR_EXTRACTION_DURATION = _create_histogram(
    "omni_detector_extraction_duration_seconds",
    "Feature extraction duration per detector in seconds",
    ["detector_name"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

DETECTOR_EXTRACTION_SUCCESS = _create_counter(
    "omni_detector_extraction_success_total",
    "Total number of successful feature extractions",
    ["detector_name"],
)

DETECTOR_EXTRACTION_ERRORS = _create_counter(
    "omni_detector_extraction_errors_total",
    "Total number of feature extraction errors",
    ["detector_name"],
)

# Feature cache metrics
FEATURE_CACHE_HITS = _create_counter(
    "omni_feature_cache_hits_total",
    "Total number of feature cache hits",
    ["detector_name"],
)

FEATURE_CACHE_MISSES = _create_counter(
    "omni_feature_cache_misses_total",
    "Total number of feature cache misses",
    ["detector_name"],
)

# Model quality metrics (live performance)
MODEL_ROC_AUC = _create_gauge(
    "omni_model_roc_auc",
    "Current ROC-AUC score for model",
    ["model_version", "dataset"],
)

MODEL_F1_SCORE = _create_gauge(
    "omni_model_f1_score",
    "Current F1 score for model",
    ["model_version", "dataset"],
)


@contextmanager
def time_detection(detector_type: str) -> Generator[None, None, None]:
    """Context manager to time detection operations.

    Args:
        detector_type: Type of detector being timed

    Example:
        with time_detection("statistical"):
            result = detector.detect(data)
    """
    DETECTION_REQUESTS.labels(detector_type=detector_type).inc()
    start_time = time.perf_counter()
    try:
        yield
        DETECTION_SUCCESS.labels(detector_type=detector_type).inc()
    finally:
        duration = time.perf_counter() - start_time
        DETECTION_DURATION.labels(detector_type=detector_type).observe(duration)


@contextmanager
def time_feature_extraction(detector_name: str) -> Generator[None, None, None]:
    """Context manager to time feature extraction operations.

    Args:
        detector_name: Name of detector being timed

    Example:
        with time_feature_extraction("isolation_forest"):
            features = detector.extract_features(data)
    """
    start_time = time.perf_counter()
    try:
        yield
        DETECTOR_EXTRACTION_SUCCESS.labels(detector_name=detector_name).inc()
    except Exception:
        DETECTOR_EXTRACTION_ERRORS.labels(detector_name=detector_name).inc()
        raise
    finally:
        duration = time.perf_counter() - start_time
        DETECTOR_EXTRACTION_DURATION.labels(detector_name=detector_name).observe(duration)


@contextmanager
def time_fusion_inference(fusion_type: str = "hybrid") -> Generator[None, None, None]:
    """Context manager to time fusion layer inference.

    Args:
        fusion_type: Type of fusion being timed (e.g., "hybrid", "attention", "voting")

    Example:
        with time_fusion_inference("hybrid"):
            result = fusion_model(features)
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        FUSION_INFERENCE_DURATION.labels(fusion_type=fusion_type).observe(duration)


def record_inference_error(model_name: str) -> None:
    """Record a model inference error.

    Args:
        model_name: Name of the model that had an error
    """
    MODEL_INFERENCE_ERRORS.labels(model_name=model_name).inc()


def record_cache_hit(detector_name: str) -> None:
    """Record a feature cache hit.

    Args:
        detector_name: Name of the detector
    """
    FEATURE_CACHE_HITS.labels(detector_name=detector_name).inc()


def record_cache_miss(detector_name: str) -> None:
    """Record a feature cache miss.

    Args:
        detector_name: Name of the detector
    """
    FEATURE_CACHE_MISSES.labels(detector_name=detector_name).inc()


def update_model_metrics(
    model_version: str, dataset: str, roc_auc: float | None = None, f1_score: float | None = None
) -> None:
    """Update model quality metrics.

    Args:
        model_version: Version identifier for the model
        dataset: Dataset name
        roc_auc: ROC-AUC score (optional)
        f1_score: F1 score (optional)
    """
    if roc_auc is not None:
        MODEL_ROC_AUC.labels(model_version=model_version, dataset=dataset).set(roc_auc)
    if f1_score is not None:
        MODEL_F1_SCORE.labels(model_version=model_version, dataset=dataset).set(f1_score)


def is_prometheus_available() -> bool:
    """Check if Prometheus metrics are available.

    Returns:
        True if prometheus_client is installed and metrics are active
    """
    return PROMETHEUS_AVAILABLE
