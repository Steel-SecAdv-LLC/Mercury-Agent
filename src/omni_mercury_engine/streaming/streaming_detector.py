# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Streaming anomaly detector with async data ingestion.

Provides a StreamingDetector wrapper that accepts data points one at a time, maintains a rolling
window, and produces anomaly scores using MercuryAnomalyDetector. Supports async ingestion via
asyncio + aiohttp for real-time API data feeds.

Latency target: < 1 second from data receipt to anomaly score.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

import numpy as np

logger = logging.getLogger(__name__)


def _get_detector_class() -> type[MercuryAnomalyDetector]:
    """Lazy import MercuryAnomalyDetector to avoid torch at import time."""
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    return MercuryAnomalyDetector


class StreamingDetector:
    """Rolling-window anomaly detector for streaming data.

    Maintains a fixed-size window of recent observations and runs
    MercuryAnomalyDetector on the window after each new data point
    (or batch of points). The detector is re-fit periodically on the
    window to adapt to concept drift.

    Example::

        detector = StreamingDetector(window_size=1000, refit_interval=100)

        # Process individual data points
        for point in data_stream:
            result = detector.ingest(point)
            if result and result["is_anomaly"][-1]:
                print(f"Anomaly detected! Score: {result['scores'][-1]:.3f}")

        # Or async ingestion
        async for point in async_data_source:
            result = await detector.async_ingest(point)
    """

    def __init__(
        self,
        window_size: int = 1000,
        refit_interval: int = 100,
        min_samples: int = 50,
        detector_config: dict[str, Any] | None = None,
        on_anomaly: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize the streaming detector.

        Args:
            window_size: Number of recent observations to maintain.
            refit_interval: Re-fit the detector every N new points.
            min_samples: Minimum samples needed before detection starts.
            detector_config: Configuration dict for MercuryAnomalyDetector.
            on_anomaly: Optional callback when anomaly detected.
        """
        self.window_size = window_size
        self.refit_interval = refit_interval
        self.min_samples = min_samples
        self.on_anomaly = on_anomaly

        _cls = _get_detector_class()
        self._detector = _cls(config=detector_config or {})
        self._window: deque[np.ndarray[Any, Any]] = deque(maxlen=window_size)
        self._points_since_refit: int = 0
        self._total_points: int = 0
        self._is_fitted: bool = False
        self._last_refit_time: float = 0.0
        self._lock = asyncio.Lock()

        # Performance tracking
        self._latencies: deque[float] = deque(maxlen=100)

    @property
    def is_ready(self) -> bool:
        """Whether the detector has enough data to produce scores."""
        return len(self._window) >= self.min_samples

    @property
    def window_data(self) -> np.ndarray[Any, Any]:
        """Current window as a numpy array."""
        if not self._window:
            return np.empty((0, 0))
        return np.array(list(self._window))

    @property
    def mean_latency_ms(self) -> float:
        """Mean detection latency in milliseconds."""
        if not self._latencies:
            return 0.0
        return float(np.mean(list(self._latencies)) * 1000)

    def _refit(self) -> None:
        """Re-fit the detector on the current window."""
        if len(self._window) < self.min_samples:
            return
        data = self.window_data
        try:
            self._detector.fit(data)
            self._is_fitted = True
            self._points_since_refit = 0
            self._last_refit_time = time.monotonic()
            logger.debug("StreamingDetector refit on %d samples.", len(self._window))
        except Exception as exc:
            logger.warning("StreamingDetector refit failed: %s", exc)

    def ingest(self, point: np.ndarray[Any, Any] | list[float] | float) -> dict[str, Any] | None:
        """Ingest a single data point and optionally return detection results.

        Args:
            point: A single observation. Can be a scalar, 1-D array
                (one feature per element), or already a numpy array.

        Returns:
            Detection result dict if detection was run, None if not
            enough data yet. The result dict contains the same keys
            as MercuryAnomalyDetector.detect() output.
        """
        start = time.monotonic()

        # Normalize input
        arr = np.atleast_1d(np.asarray(point, dtype=np.float64))
        self._window.append(arr)
        self._total_points += 1
        self._points_since_refit += 1

        # Check if we need to refit
        if self._points_since_refit >= self.refit_interval or not self._is_fitted:
            self._refit()

        # Run detection if fitted
        if not self._is_fitted:
            return None

        try:
            # Detect on the most recent point(s)
            recent = arr.reshape(1, -1)
            result = self._detector.detect(recent)

            elapsed = time.monotonic() - start
            self._latencies.append(elapsed)

            # Enrich result with streaming metadata
            result["streaming_metadata"] = {
                "total_points_ingested": self._total_points,
                "window_size": len(self._window),
                "points_since_refit": self._points_since_refit,
                "latency_ms": elapsed * 1000,
                "mean_latency_ms": self.mean_latency_ms,
            }

            # Fire callback if anomaly detected
            if self.on_anomaly and result.get("is_anomaly") is not None:
                is_anom = np.asarray(result["is_anomaly"])
                if np.any(is_anom):
                    self.on_anomaly(result)

            return result

        except Exception as exc:
            logger.warning("StreamingDetector detection failed: %s", exc)
            return None

    def ingest_batch(
        self, batch: np.ndarray[Any, Any] | list[list[float]]
    ) -> dict[str, Any] | None:
        """Ingest a batch of data points.

        Args:
            batch: Array of shape (n_samples, n_features).

        Returns:
            Detection result for the entire batch, or None.
        """
        start = time.monotonic()
        arr = np.atleast_2d(np.asarray(batch, dtype=np.float64))

        for row in arr:
            self._window.append(row)
        self._total_points += len(arr)
        self._points_since_refit += len(arr)

        if self._points_since_refit >= self.refit_interval or not self._is_fitted:
            self._refit()

        if not self._is_fitted:
            return None

        try:
            result = self._detector.detect(arr)
            elapsed = time.monotonic() - start
            self._latencies.append(elapsed)

            result["streaming_metadata"] = {
                "total_points_ingested": self._total_points,
                "window_size": len(self._window),
                "batch_size": len(arr),
                "latency_ms": elapsed * 1000,
            }

            if self.on_anomaly and result.get("is_anomaly") is not None:
                is_anom = np.asarray(result["is_anomaly"])
                if np.any(is_anom):
                    self.on_anomaly(result)

            return result

        except Exception as exc:
            logger.warning("StreamingDetector batch detection failed: %s", exc)
            return None

    async def async_ingest(
        self, point: np.ndarray[Any, Any] | list[float] | float
    ) -> dict[str, Any] | None:
        """Async version of ingest for use with asyncio event loops.

        Args:
            point: A single observation.

        Returns:
            Detection result dict or None.
        """
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(None, self.ingest, point)

    async def async_ingest_batch(
        self, batch: np.ndarray[Any, Any] | list[list[float]]
    ) -> dict[str, Any] | None:
        """Async version of ingest_batch.

        Args:
            batch: Array of shape (n_samples, n_features).

        Returns:
            Detection result dict or None.
        """
        async with self._lock:
            return await asyncio.get_event_loop().run_in_executor(None, self.ingest_batch, batch)

    def reset(self) -> None:
        """Reset the detector state, clearing all buffered data."""
        self._window.clear()
        self._points_since_refit = 0
        self._total_points = 0
        self._is_fitted = False
        self._latencies.clear()
        _cls = _get_detector_class()
        self._detector = _cls(
            config=self._detector.config if hasattr(self._detector, "config") else {}
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current streaming statistics.

        Returns:
            Dict with performance and state metrics.
        """
        return {
            "total_points_ingested": self._total_points,
            "window_size": len(self._window),
            "window_capacity": self.window_size,
            "is_fitted": self._is_fitted,
            "points_since_refit": self._points_since_refit,
            "mean_latency_ms": self.mean_latency_ms,
            "is_ready": self.is_ready,
        }
